"""Stochastic Volatility model (TensorFlow)."""

import numpy as np
import tensorflow as tf
from scipy.special import digamma as _digamma

from ..core.model_base import StateSpaceModel

# Log-chi-squared(1) distribution moments:
#   If r ~ N(0,1), then log(r^2) ~ log(chi^2_1)
#   E[log(r^2)] = psi(1/2) + log(2)  where psi = digamma
#   Var[log(r^2)] = pi^2 / 2
_LOG_CHI2_MEAN = float(_digamma(0.5) + np.log(2.0))   # ≈ -1.2704
_LOG_CHI2_VAR = float(np.pi ** 2 / 2.0)                # ≈ 4.9348


class StochasticVolatilityModel(StateSpaceModel):
    """
    1D Stochastic Volatility Model.

    State evolution (linear):
        x_t = alpha * x_{t-1} + sigma * w_t,  w_t ~ N(0, 1)

    Observation (nonlinear, non-Gaussian):
        y_t = beta * exp(x_t / 2) * v_t,  v_t ~ N(0, 1)

    Log-space mode (log_space=True):
        Transforms observations as z_t = log(y_t^2), giving:
            z_t = log(beta^2) + x_t + log(v_t^2)
        where log(v_t^2) ~ log(chi^2_1) ≈ N(-1.2704, pi^2/2).
        This makes the observation LINEAR in x_t with additive noise,
        enabling EKF/UKF to work (H_x = 1 instead of 0).

        Data generation still uses the original model (sample_observation
        returns raw y_t). The log transform is applied when filtering via
        a separate transform_observations() method.

    Parameters:
        alpha: persistence parameter (0 < alpha < 1)
        sigma: volatility of volatility
        beta: scale parameter
        log_space: if True, filter-facing methods use log-squared transform
        y_floor: floor value for |y| before taking log (avoids log(0))
    """

    def __init__(self, alpha: float = 0.91, sigma: float = 1.0,
                 beta: float = 0.5, log_space: bool = False,
                 y_floor: float = 1e-8, dtype=None):
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        if dtype is None:
            dtype = tf.float32
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.log_space = log_space
        self._y_floor = y_floor

        # TF constants
        self._alpha_tf = tf.constant(alpha, dtype=dtype)
        self._sigma_tf = tf.constant(sigma, dtype=dtype)
        self._beta_tf = tf.constant(beta, dtype=dtype)
        self._stationary_var = self._sigma_tf ** 2 / (1.0 - self._alpha_tf ** 2)
        self._pi2 = tf.constant(2.0 * np.pi, dtype=dtype)

        # Log-space constants
        self._log_chi2_mean = tf.constant(_LOG_CHI2_MEAN, dtype=dtype)
        self._log_chi2_var = tf.constant(_LOG_CHI2_VAR, dtype=dtype)
        self._log_beta2 = tf.math.log(self._beta_tf ** 2)

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def mu_0(self) -> tf.Tensor:
        return tf.zeros([1], dtype=self.dtype)

    @property
    def Sigma_0(self) -> tf.Tensor:
        return tf.reshape(self._stationary_var, [1, 1])

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        if self.log_space:
            return tf.reshape(self._log_chi2_var, [1, 1])
        return tf.reshape(self._beta_tf ** 2, [1, 1])

    @property
    def process_noise_cov(self) -> tf.Tensor:
        return tf.reshape(self._sigma_tf ** 2, [1, 1])

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        std = tf.sqrt(self._stationary_var)
        z = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return z * std

    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        w = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(self._alpha_tf * x[0] + self._sigma_tf * w[0], [1])

    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        v = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(self._beta_tf * tf.exp(x[0] / 2.0) * v[0], [1])

    # ------------------------------------------------------------------
    # Deterministic methods
    # ------------------------------------------------------------------

    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reshape(self._alpha_tf * x[0], [1])

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        return self.process_noise_cov

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reshape(self._alpha_tf, [1, 1])

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        """E[y | x]. In log-space: log(beta^2) + x + E[log(chi^2_1)]. Original: 0."""
        if self.log_space:
            return tf.reshape(self._log_beta2 + x[0] + self._log_chi2_mean, [1])
        return tf.zeros_like(x)

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Var[y | x]. In log-space: pi^2/2 (constant). Original: beta^2 * exp(x)."""
        if self.log_space:
            return tf.reshape(self._log_chi2_var, [1, 1])
        return tf.reshape(self._beta_tf ** 2 * tf.exp(x[0]), [1, 1])

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """dh/dx. In log-space: 1 (linear). Original: 0."""
        if self.log_space:
            return tf.ones([1, 1], dtype=self.dtype)
        return tf.zeros([1, 1], dtype=self.dtype)

    def observation_hessian(self, x: tf.Tensor) -> tf.Tensor:
        return tf.zeros([1, 1, 1], dtype=self.dtype)

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        """h(x) for filter use. In log-space: log(beta^2) + x + E[eps]. Original: 0."""
        if self.log_space:
            return tf.reshape(self._log_beta2 + x[0] + self._log_chi2_mean, [1])
        return tf.zeros_like(x)

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Log p(y | x). y is raw observation (not log-transformed)."""
        if self.log_space:
            # In log-space: z = log(y^2), model is z = log(beta^2) + x + eps
            # where eps ~ log(chi^2_1) approx N(mean, var)
            y_clipped = tf.maximum(tf.abs(y[0]), self._y_floor)
            z = tf.math.log(y_clipped ** 2)
            predicted = self._log_beta2 + x[0] + self._log_chi2_mean
            return -0.5 * (tf.math.log(self._pi2 * self._log_chi2_var)
                           + (z - predicted) ** 2 / self._log_chi2_var)
        var = self._beta_tf ** 2 * tf.exp(x[0])
        return -0.5 * (tf.math.log(self._pi2 * var) + y[0] ** 2 / var)

    # ------------------------------------------------------------------
    # Observation transform (for log-space mode)
    # ------------------------------------------------------------------

    def transform_observations(self, observations: np.ndarray) -> np.ndarray:
        """Transform raw observations for filtering.

        In log-space mode: z_t = log(y_t^2) with floor to avoid log(0).
        In original mode: returns observations unchanged.

        Args:
            observations: (T, obs_dim) raw observations from generate_data

        Returns:
            Transformed observations, same shape.
        """
        if not self.log_space:
            return observations
        y_clipped = np.maximum(np.abs(observations), self._y_floor)
        return np.log(y_clipped ** 2)

    # ------------------------------------------------------------------
    # Batch methods
    # ------------------------------------------------------------------

    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        std = tf.sqrt(self._stationary_var)
        return tf.random.stateless_normal([n, 1], seed=seed, dtype=self.dtype) * std

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor, t=None) -> tf.Tensor:
        w = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
        return self._alpha_tf * particles + self._sigma_tf * w

    def state_transition_mean_batch(self, particles: tf.Tensor, t=None) -> tf.Tensor:
        return self._alpha_tf * particles

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return self.process_noise_cov

    def log_observation_prob_batch(self, observation: tf.Tensor,
                                   particles: tf.Tensor) -> tf.Tensor:
        if self.log_space:
            # observation is already log-transformed z = log(y^2)
            predicted = self._log_beta2 + particles[:, 0] + self._log_chi2_mean
            diff_sq = (observation[0] - predicted) ** 2
            return -0.5 * (tf.math.log(self._pi2 * self._log_chi2_var)
                           + diff_sq / self._log_chi2_var)
        obs_vars = self._beta_tf ** 2 * tf.exp(particles[:, 0])
        diff_squared = observation[0] ** 2
        mahalanobis = diff_squared / obs_vars
        logdet = tf.math.log(self._pi2 * obs_vars)
        return -0.5 * (logdet + mahalanobis)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        if self.log_space:
            return tf.ones([N, 1, 1], dtype=self.dtype)
        return tf.zeros([N, 1, 1], dtype=self.dtype)

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        if self.log_space:
            return self._log_beta2 + particles + self._log_chi2_mean
        return tf.zeros_like(particles)

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        return tf.fill([N, 1, 1], self._alpha_tf)
