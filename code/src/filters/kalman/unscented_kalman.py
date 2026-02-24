"""Unscented Kalman Filter for nonlinear systems."""

import tensorflow as tf
import numpy as np
import time
from typing import Optional, Tuple, Callable
from ...core.filter_base import Filter
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel
from ...utils.linalg import safe_cholesky, safe_inv, symmetrize


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
        self.dtype = getattr(model, 'dtype', tf.float64)
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

        self.weights_mean = tf.constant(weights_mean_np, dtype=self.dtype)
        self.weights_cov = tf.constant(weights_cov_np, dtype=self.dtype)

        # Store initial state
        if sample_initial_mean:
            # Sample initial mean from model's distribution
            if not hasattr(self.model, 'sample_initial_state'):
                raise ValueError("Model must have sample_initial_state method to use sample_initial_mean=True")
            seed = tf.constant([random_seed if random_seed is not None else 0, 0], dtype=tf.int32)
            mean_0_tf = self.model.sample_initial_state(seed)
            self.mean_0 = tf.cast(mean_0_tf, self.dtype)
        elif mean_0 is None:
            self.mean_0 = tf.cast(self.model.mu_0, self.dtype)
        else:
            self.mean_0 = tf.constant(mean_0, dtype=self.dtype)

        if Sigma_0 is None:
            self.Sigma_0 = tf.cast(self.model.Sigma_0, self.dtype)
        else:
            self.Sigma_0 = tf.constant(Sigma_0, dtype=self.dtype)

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
        sigma_points = tf.TensorArray(dtype=self.dtype, size=2 * n + 1)

        # First sigma point is the mean
        sigma_points = sigma_points.write(0, mean)

        # Generate positive and negative sigma points
        for i in tf.range(n):
            sigma_points = sigma_points.write(i + 1, mean + sqrt_cov[:, i])
            sigma_points = sigma_points.write(i + n + 1, mean - sqrt_cov[:, i])

        return sigma_points.stack()

    def reset(self):
        """Reset filter to initial state and clear history."""
        self.mean = tf.Variable(self.mean_0, dtype=self.dtype)
        self.cov = tf.Variable(self.Sigma_0, dtype=self.dtype)
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
            dtype=self.dtype
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
            dtype=self.dtype
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
        K = P_xy @ safe_inv(S)

        # Innovation
        innovation = observation - y_pred

        # Update mean and covariance (Joseph-form equivalent for UKF)
        # P = P_pred - K @ P_xy^T - P_xy @ K^T + K @ S @ K^T
        # Mathematically identical to P - KSK^T but numerically more stable.
        mean_updated = mean + tf.linalg.matvec(K, innovation)
        cov_updated = cov - K @ tf.transpose(P_xy) - P_xy @ tf.transpose(K) + K @ S @ tf.transpose(K)
        cov_updated = symmetrize(cov_updated)

        # Log-likelihood
        sign, logdet = tf.linalg.slogdet(2.0 * np.pi * S)
        innovation_col = tf.reshape(innovation, [-1, 1])
        mahalanobis = tf.reduce_sum(innovation * tf.squeeze(tf.linalg.solve(S, innovation_col), axis=-1))
        log_lik = -0.5 * (logdet + mahalanobis)

        return mean_updated, cov_updated, log_lik

    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Total log marginal likelihood as a differentiable TF scalar.

        Reimplements the UKF predict/update loop using local tensor state
        (no side effects on self.mean/self.cov). Reads Q and R from the model
        dynamically at each step so parameter gradients are preserved.

        NOT decorated with @tf.function — runs eagerly or is traced by TFP
        internally for HMC gradient computation.

        Args:
            observations: (T, obs_dim), dtype matching filter
            seed: Unused (UKF is deterministic). Present for protocol compat.

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        T = observations.shape[0]

        mean = tf.identity(self.mean_0)
        cov = tf.identity(self.Sigma_0)
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in range(T):
            # === PREDICT ===
            sigma_pts = self._compute_sigma_points(mean, cov)
            sigma_pred = tf.map_fn(
                lambda sp: self.model.state_transition_mean(sp),
                sigma_pts, dtype=self.dtype
            )
            mean_pred = tf.reduce_sum(
                tf.expand_dims(self.weights_mean, -1) * sigma_pred, axis=0
            )
            diff = sigma_pred - mean_pred
            Q = self.model.state_transition_cov(mean_pred)
            cov_pred = symmetrize(
                tf.reduce_sum(
                    tf.expand_dims(tf.expand_dims(self.weights_cov, -1), -1) *
                    tf.einsum('ij,ik->ijk', diff, diff), axis=0
                ) + Q
            )

            # === UPDATE ===
            sigma_pts2 = self._compute_sigma_points(mean_pred, cov_pred)
            y_sigma = tf.map_fn(
                lambda sp: self.model.observation_mean(sp),
                sigma_pts2, dtype=self.dtype
            )
            y_pred = tf.reduce_sum(
                tf.expand_dims(self.weights_mean, -1) * y_sigma, axis=0
            )
            diff_y = y_sigma - y_pred
            R = self.model.observation_cov(mean_pred)
            S = symmetrize(
                tf.reduce_sum(
                    tf.expand_dims(tf.expand_dims(self.weights_cov, -1), -1) *
                    tf.einsum('ij,ik->ijk', diff_y, diff_y), axis=0
                ) + R
            )
            diff_x = sigma_pts2 - mean_pred
            P_xy = tf.reduce_sum(
                tf.expand_dims(tf.expand_dims(self.weights_cov, -1), -1) *
                tf.einsum('ij,ik->ijk', diff_x, diff_y), axis=0
            )
            K = P_xy @ safe_inv(S)
            innovation = observations[t] - y_pred
            mean = mean_pred + tf.linalg.matvec(K, innovation)
            # Joseph-form equivalent: P_pred - K@P_xy^T - P_xy@K^T + K@S@K^T
            cov = symmetrize(cov_pred - K @ tf.transpose(P_xy) - P_xy @ tf.transpose(K) + K @ S @ tf.transpose(K))

            # Log-likelihood: log p(y_t | y_{1:t-1})
            _, logdet = tf.linalg.slogdet(2.0 * np.pi * S)
            inn_col = tf.reshape(innovation, [-1, 1])
            mahalanobis = tf.reduce_sum(
                innovation * tf.squeeze(tf.linalg.solve(S, inn_col), axis=-1)
            )
            total_log_lik = total_log_lik + (-0.5 * (logdet + mahalanobis))

        return total_log_lik

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
        obs_tf = tf.constant(observation, dtype=self.dtype)

        mean_updated, cov_updated, log_lik = self._update_step(
            self.mean.value(),
            self.cov.value(),
            obs_tf
        )

        self.mean.assign(mean_updated)
        self.cov.assign(cov_updated)

        # Store log-likelihood as TF scalar (converted in filter())
        self.log_likelihoods.append(log_lik)

    def filter(self, observations: np.ndarray,
               progress_callback: Optional[Callable[[int, int, float], None]] = None) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Array of shape (T, obs_dim) containing observations
            progress_callback: Optional callback(t, T, step_time_sec) called after each step

        Returns:
            FilterResult with filtered means, covariances, and diagnostics
        """
        T = observations.shape[0]

        # Reset to initial state and clear history
        self.reset()

        means = []
        covs = []

        for t in range(T):
            t0 = time.perf_counter()
            self.predict()
            self.update(observations[t])
            # Store TF tensors — convert once after loop
            means.append(self.mean.value())
            covs.append(self.cov.value())
            if progress_callback is not None:
                progress_callback(t, T, time.perf_counter() - t0)

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
