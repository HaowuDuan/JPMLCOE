"""Unscented Kalman Filter for nonlinear systems."""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple
from ...core.filter_base import Filter
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel
from ...utils.linalg import safe_cholesky, symmetrize


class UnscentedKalmanFilter(Filter):
    """
    Unscented Kalman Filter (UKF) for nonlinear state-space models.

    The UKF uses sigma points to capture the mean and covariance through
    nonlinear transformations, avoiding the need for Jacobian computation.
    This works well when:
    - Nonlinearities are moderate
    - Distributions are roughly Gaussian
    - State dimension is moderate (scales as O(n²) with n states)

    The filter can fail when:
    - Distributions are highly non-Gaussian
    - State dimension is very high (computational cost)
    - Sigma points become unstable or invalid

    Uses the standard unscented transform with parameters:
    - α: controls spread of sigma points (default 1e-3)
    - β: incorporates prior knowledge (default 2.0)
    - κ: secondary scaling parameter (default 0.0)
    """

    def __init__(
        self,
        model: StateSpaceModel,
        mean_0: np.ndarray = None,
        Sigma_0: np.ndarray = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        sample_initial_mean: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Unscented Kalman Filter.

        Args:
            model: State-space model instance
            mean_0: Initial mean (state_dim,). If None, defaults to zeros.
            Sigma_0: Initial covariance matrix (state_dim, state_dim).
                     If None, uses stationary variance if available, else identity.
            alpha: Spread parameter for sigma points (small positive)
            beta: Incorporates prior knowledge (2 is optimal for Gaussian)
            kappa: Secondary scaling parameter (0 for default)
            sample_initial_mean: If True (default), sample initial mean from model's initial distribution
                                (requires model to have sample_initial_state method).
                                If False, use mean_0 or zeros.
            random_seed: Random seed for sampling initial mean (only used if sample_initial_mean=True)
        """
        self.model = model
        self.state_dim = model.state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute lambda parameter
        self.lambda_ = alpha**2 * (self.state_dim + kappa) - self.state_dim

        # Weights for mean and covariance
        W_m_0 = self.lambda_ / (self.state_dim + self.lambda_)
        W_c_0 = W_m_0 + (1 - alpha**2 + beta)
        W_i = 1.0 / (2 * (self.state_dim + self.lambda_))

        weights_mean_np = np.concatenate([[W_m_0], np.full(2 * self.state_dim, W_i)])
        weights_cov_np = np.concatenate([[W_c_0], np.full(2 * self.state_dim, W_i)])

        self.weights_mean = tf.constant(weights_mean_np, dtype=tf.float32)
        self.weights_cov = tf.constant(weights_cov_np, dtype=tf.float32)

        # Store initial state
        if sample_initial_mean:
            # Sample initial mean from model's distribution
            if not hasattr(self.model, 'sample_initial_state'):
                raise ValueError("Model must have sample_initial_state method to use sample_initial_mean=True")
            seed = tf.constant([random_seed if random_seed is not None else 0, 0], dtype=tf.int32)
            mean_0_tf = self.model.sample_initial_state(seed)
            self.mean_0 = tf.cast(mean_0_tf, tf.float32)
        elif mean_0 is None:
            # Try to use model's initial_state_mean property
            if hasattr(self.model, 'initial_state_mean'):
                mean_0_val = self.model.initial_state_mean
                self.mean_0 = tf.cast(mean_0_val, tf.float32) if isinstance(mean_0_val, tf.Tensor) else tf.constant(mean_0_val, dtype=tf.float32)
            else:
                self.mean_0 = tf.zeros(self.state_dim, dtype=tf.float32)
        else:
            self.mean_0 = tf.constant(mean_0, dtype=tf.float32)

        if Sigma_0 is None:
            # Try to use model's initial_state_cov property
            if hasattr(self.model, 'initial_state_cov'):
                Sigma_0_val = self.model.initial_state_cov
                self.Sigma_0 = tf.cast(Sigma_0_val, tf.float32) if isinstance(Sigma_0_val, tf.Tensor) else tf.constant(Sigma_0_val, dtype=tf.float32)
            # Otherwise use stationary covariance if available
            elif hasattr(self.model, 'stationary_var'):
                self.Sigma_0 = tf.eye(self.state_dim, dtype=tf.float32) * self.model.stationary_var
            else:
                self.Sigma_0 = tf.eye(self.state_dim, dtype=tf.float32)
        else:
            self.Sigma_0 = tf.constant(Sigma_0, dtype=tf.float32)

        # Filter state
        self.mean = None
        self.cov = None

        # Storage
        self.log_likelihoods = []

        self.reset()

    @tf.function(reduce_retracing=True)
    def _compute_sigma_points(self, mean: tf.Tensor, cov: tf.Tensor) -> tf.Tensor:
        """
        Compute sigma points using the unscented transform.

        Args:
            mean: State mean, shape (state_dim,)
            cov: State covariance, shape (state_dim, state_dim)

        Returns:
            Tensor of shape (2*state_dim + 1, state_dim) containing sigma points
        """
        n = self.state_dim
        lambda_ = self.lambda_

        # Compute matrix square root: sqrt((n + λ)·P)
        sqrt_cov = safe_cholesky((n + lambda_) * cov)

        # Initialize sigma points using TensorArray
        sigma_points = tf.TensorArray(dtype=tf.float32, size=2 * n + 1)

        # First sigma point is the mean
        sigma_points = sigma_points.write(0, mean)

        # Generate positive and negative sigma points
        for i in tf.range(n):
            sigma_points = sigma_points.write(i + 1, mean + sqrt_cov[:, i])
            sigma_points = sigma_points.write(i + n + 1, mean - sqrt_cov[:, i])

        return sigma_points.stack()

    def reset(self):
        """Reset filter to initial state and clear history."""
        self.mean = tf.Variable(self.mean_0, dtype=tf.float32)
        self.cov = tf.Variable(self.Sigma_0, dtype=tf.float32)
        self.log_likelihoods = []

    @tf.function(reduce_retracing=True)
    def _predict_step(self, mean: tf.Tensor, cov: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prediction step using unscented transform.

        Args:
            mean: Current mean estimate
            cov: Current covariance estimate

        Returns:
            Tuple of (predicted_mean, predicted_cov)
        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(mean, cov)

        # Propagate through state transition
        sigma_points_pred = tf.map_fn(
            lambda sp: self.model.state_transition_mean(sp),
            sigma_points,
            dtype=tf.float32
        )

        # Predict mean: weighted sum of propagated sigma points
        mean_pred = tf.reduce_sum(
            tf.expand_dims(self.weights_mean, -1) * sigma_points_pred,
            axis=0
        )

        # Predict covariance: weighted sum of outer products
        diff = sigma_points_pred - mean_pred
        cov_pred = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(self.weights_cov, -1), -1) *
            tf.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )

        # Add process noise
        Q = self.model.state_transition_cov(mean_pred)
        cov_pred = cov_pred + Q
        cov_pred = symmetrize(cov_pred)

        return mean_pred, cov_pred

    @tf.function(reduce_retracing=True)
    def _update_step(
        self,
        mean: tf.Tensor,
        cov: tf.Tensor,
        observation: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Update step using unscented transform.

        Args:
            mean: Predicted mean
            cov: Predicted covariance
            observation: Observation vector

        Returns:
            Tuple of (updated_mean, updated_cov, log_likelihood)
        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(mean, cov)

        # Propagate through observation model
        y_sigma = tf.map_fn(
            lambda sp: self.model.observation_mean(sp),
            sigma_points,
            dtype=tf.float32
        )

        # Predicted observation mean
        y_pred = tf.reduce_sum(
            tf.expand_dims(self.weights_mean, -1) * y_sigma,
            axis=0
        )

        # Innovation covariance
        diff_y = y_sigma - y_pred
        S = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(self.weights_cov, -1), -1) *
            tf.einsum('ij,ik->ijk', diff_y, diff_y),
            axis=0
        )

        # Add observation noise
        R = self.model.observation_cov(mean)
        S = S + R

        # Cross-covariance between state and observation
        diff_x = sigma_points - mean
        P_xy = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(self.weights_cov, -1), -1) *
            tf.einsum('ij,ik->ijk', diff_x, diff_y),
            axis=0
        )

        # Kalman gain
        K = P_xy @ tf.linalg.inv(S)

        # Innovation
        innovation = observation - y_pred

        # Update mean and covariance
        mean_updated = mean + tf.linalg.matvec(K, innovation)
        cov_updated = cov - K @ S @ tf.transpose(K)
        cov_updated = symmetrize(cov_updated)

        # Log-likelihood
        sign, logdet = tf.linalg.slogdet(2.0 * np.pi * S)
        innovation_col = tf.reshape(innovation, [-1, 1])
        mahalanobis = tf.reduce_sum(innovation * tf.squeeze(tf.linalg.solve(S, innovation_col), axis=-1))
        log_lik = -0.5 * (logdet + mahalanobis)

        return mean_updated, cov_updated, log_lik

    def predict(self):
        """Prediction step using unscented transform."""
        mean_pred, cov_pred = self._predict_step(self.mean.value(), self.cov.value())

        self.mean.assign(mean_pred)
        self.cov.assign(cov_pred)

    def update(self, observation: np.ndarray):
        """
        Update step using unscented transform.

        Args:
            observation: Observation vector
        """
        obs_tf = tf.constant(observation, dtype=tf.float32)

        mean_updated, cov_updated, log_lik = self._update_step(
            self.mean.value(),
            self.cov.value(),
            obs_tf
        )

        self.mean.assign(mean_updated)
        self.cov.assign(cov_updated)

        # Store log-likelihood as TF scalar (converted in filter())
        self.log_likelihoods.append(log_lik)

    def filter(self, observations: np.ndarray) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Array of shape (T, obs_dim) containing observations

        Returns:
            FilterResult with filtered means, covariances, and diagnostics
        """
        T = observations.shape[0]

        # Reset to initial state and clear history
        self.reset()

        means = []
        covs = []

        for t in range(T):
            self.predict()
            self.update(observations[t])
            # Store TF tensors — convert once after loop
            means.append(self.mean.value())
            covs.append(self.cov.value())

        # Convert accumulated TF tensors to numpy once
        means_np = tf.stack(means).numpy()
        covs_np = tf.stack(covs).numpy()
        log_liks_tf = tf.stack(self.log_likelihoods)
        total_log_likelihood = float(tf.reduce_sum(log_liks_tf).numpy())

        return FilterResult(
            means=means_np,
            covs=covs_np,
            log_likelihood=total_log_likelihood,
            metadata={
                'filter_type': 'UnscentedKalmanFilter',
                'alpha': self.alpha,
                'beta': self.beta,
                'kappa': self.kappa,
                'log_likelihoods': log_liks_tf.numpy()
            }
        )
