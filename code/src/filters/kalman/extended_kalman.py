"""Extended Kalman Filter for nonlinear systems."""

import tensorflow as tf
import numpy as np
import time
from typing import Optional, Tuple, Callable
from ...core.filter_base import Filter
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel
from ...utils.linalg import symmetrize, safe_solve, safe_cholesky


class ExtendedKalmanFilter(Filter):
    """
    Extended Kalman Filter (EKF) for nonlinear state-space models.

    The EKF linearizes the nonlinear functions around the current state estimate
    using first-order Taylor expansion (Jacobian matrices). This works well when
    the nonlinearity is mild, but can fail when:
    - Nonlinearities are strong
    - Jacobians are difficult to compute or inaccurate
    - State estimates are far from the true state

    The filter maintains:
    - Mean: x̂_n|n (filtered state estimate)
    - Covariance: P_n|n (filtered state covariance)
    """

    def __init__(
        self,
        model: StateSpaceModel,
        mean_0: np.ndarray = None,
        Sigma_0: np.ndarray = None,
        sample_initial_mean: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Extended Kalman Filter.

        Args:
            model: State-space model instance
            mean_0: Initial mean (state_dim,). If None, defaults to zeros.
            Sigma_0: Initial covariance matrix (state_dim, state_dim).
                     If None, uses stationary variance if available, else identity.
            sample_initial_mean: If True (default), sample initial mean from model's initial distribution
                                (requires model to have sample_initial_state method).
                                If False, use mean_0 or zeros.
            random_seed: Random seed for sampling initial mean (only used if sample_initial_mean=True)
        """
        self.model = model
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.state_dim = model.state_dim

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

        # Filter state (will be tf.Variable for mutable state)
        self.mean = None
        self.cov = None

        # Storage for log-likelihoods
        self.log_likelihoods = []

        self.reset()

    def reset(self):
        """Reset filter to initial state and clear history."""
        self.mean = tf.Variable(self.mean_0, dtype=self.dtype)
        self.cov = tf.Variable(self.Sigma_0, dtype=self.dtype)
        self.log_likelihoods = []

    @tf.function(reduce_retracing=True)
    def _predict_step(self, mean: tf.Tensor, cov: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prediction step: propagate state estimate forward.

        Uses linearization: x' ≈ f(x̂) + F·(x - x̂), where F is the Jacobian.

        Args:
            mean: Current mean estimate
            cov: Current covariance estimate

        Returns:
            Tuple of (predicted_mean, predicted_cov)
        """
        # Predict mean: x̂_{n|n-1} = f(x̂_{n-1|n-1})
        mean_pred = self.model.state_transition_mean(mean)

        # Predict covariance: P_{n|n-1} = F·P_{n-1|n-1}·F^T + Q
        F = self.model.state_jacobian(mean)
        Q = self.model.state_transition_cov(mean)
        cov_pred = F @ cov @ tf.transpose(F) + Q
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
        Update step: incorporate observation.

        Uses linearization: y ≈ h(x̂) + H·(x - x̂), where H is the Jacobian.

        Args:
            mean: Predicted mean
            cov: Predicted covariance
            observation: Observation vector

        Returns:
            Tuple of (updated_mean, updated_cov, log_likelihood)
        """
        # Predicted observation mean
        y_pred = self.model.observation_mean(mean)

        # Observation covariance
        R = self.model.observation_cov(mean)

        # Observation Jacobian
        H = self.model.observation_jacobian(mean)

        # Innovation: ν = y - y_pred
        innovation = observation - y_pred

        # Innovation covariance: S = H·P·H^T + R
        S = H @ cov @ tf.transpose(H) + R

        # Check if H is close to zero
        H_norm = tf.reduce_max(tf.abs(H))

        def update_state():
            # Kalman gain: K = P·H^T·S^{-1}
            # Solve S @ K^T = H @ P for K^T, then transpose
            # Using safe_cholesky for numerical stability
            L = safe_cholesky(S)
            K_T = tf.linalg.cholesky_solve(L, H @ cov)
            K = tf.transpose(K_T)

            # Update mean: x̂_{n|n} = x̂_{n|n-1} + K·ν
            mean_updated = mean + tf.linalg.matvec(K, innovation)

            # Update covariance: Joseph form for numerical stability
            # P_{n|n} = (I - K·H)·P_{n|n-1}·(I - K·H)^T + K·R·K^T
            I = tf.eye(self.state_dim, dtype=self.dtype)
            I_KH = I - K @ H
            cov_updated = I_KH @ cov @ tf.transpose(I_KH) + K @ R @ tf.transpose(K)
            cov_updated = symmetrize(cov_updated)

            return mean_updated, cov_updated

        def no_update():
            # No update possible - state estimate remains unchanged
            return mean, cov

        # Conditionally update based on H norm
        mean_updated, cov_updated = tf.cond(
            H_norm > 1e-10,
            update_state,
            no_update
        )

        # Log-likelihood: log p(y_n | y_{1:n-1})
        sign, logdet = tf.linalg.slogdet(2.0 * np.pi * S)
        # Compute Mahalanobis distance using safe_solve
        S_inv_innovation = safe_solve(S, innovation, method='cholesky')
        mahalanobis = tf.reduce_sum(innovation * S_inv_innovation)
        log_lik = -0.5 * (logdet + mahalanobis)

        return mean_updated, cov_updated, log_lik

    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Total log marginal likelihood as a differentiable TF scalar.

        Reimplements the EKF predict/update loop using local tensor state
        (no side effects on self.mean/self.cov). Reads Q and R from the model
        dynamically at each step, so when DifferentiableModel sets noise params
        as tensors via setattr, the gradient chain is preserved.

        NOT decorated with @tf.function — either runs eagerly or is traced
        by TFP internally for HMC gradient computation.

        Args:
            observations: (T, obs_dim), dtype matching filter
            seed: Unused (EKF is deterministic). Present for protocol compat.

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        T = observations.shape[0]

        mean = tf.identity(self.mean_0)
        cov = tf.identity(self.Sigma_0)
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in range(T):
            # === PREDICT ===
            mean_pred = self.model.state_transition_mean(mean)
            F = self.model.state_jacobian(mean)
            Q = self.model.state_transition_cov(mean)
            cov_pred = F @ cov @ tf.transpose(F) + Q
            cov_pred = symmetrize(cov_pred)

            # === UPDATE ===
            y_pred = self.model.observation_mean(mean_pred)
            R = self.model.observation_cov(mean_pred)
            H = self.model.observation_jacobian(mean_pred)
            innovation = observations[t] - y_pred
            S = H @ cov_pred @ tf.transpose(H) + R

            # Kalman gain via Cholesky
            L_S = safe_cholesky(S)
            K_T = tf.linalg.cholesky_solve(L_S, H @ cov_pred)
            K = tf.transpose(K_T)

            # Update mean and covariance (Joseph form)
            mean = mean_pred + tf.linalg.matvec(K, innovation)
            I = tf.eye(self.state_dim, dtype=self.dtype)
            I_KH = I - K @ H
            cov = I_KH @ cov_pred @ tf.transpose(I_KH) + K @ R @ tf.transpose(K)
            cov = symmetrize(cov)

            # Log-likelihood: log p(y_t | y_{1:t-1})
            _, logdet = tf.linalg.slogdet(2.0 * np.pi * S)
            S_inv_inn = safe_solve(S, innovation, method='cholesky')
            mahalanobis = tf.reduce_sum(innovation * S_inv_inn)
            total_log_lik = total_log_lik + (-0.5 * (logdet + mahalanobis))

        return total_log_lik

    def predict(self):
        """Prediction step: propagate state estimate forward."""
        mean_pred, cov_pred = self._predict_step(self.mean.value(), self.cov.value())

        self.mean.assign(mean_pred)
        self.cov.assign(cov_pred)

    def update(self, observation: np.ndarray):
        """
        Update step: incorporate observation.

        Args:
            observation: Observation, shape (obs_dim,)
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
                'filter_type': 'ExtendedKalmanFilter',
                'log_likelihoods': log_liks_tf.numpy()
            }
        )
