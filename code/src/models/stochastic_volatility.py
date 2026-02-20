"""Stochastic Volatility model (TensorFlow)."""

import numpy as np
import tensorflow as tf

from ..core.model_base import StateSpaceModel


class StochasticVolatilityModel(StateSpaceModel):
    """
    1D Stochastic Volatility Model.

    State evolution (linear):
        x_t = alpha * x_{t-1} + sigma * w_t,  w_t ~ N(0, 1)

    Observation (nonlinear, non-Gaussian):
        y_t = beta * exp(x_t / 2) * v_t,  v_t ~ N(0, 1)

    Parameters:
        alpha: persistence parameter (0 < alpha < 1)
        sigma: volatility of volatility
        beta: scale parameter
    """

    def __init__(self, alpha: float = 0.91, sigma: float = 1.0,
                 beta: float = 0.5, dtype=None):
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

        # TF constants
        self._alpha_tf = tf.constant(alpha, dtype=dtype)
        self._sigma_tf = tf.constant(sigma, dtype=dtype)
        self._beta_tf = tf.constant(beta, dtype=dtype)
        self._stationary_var = self._sigma_tf ** 2 / (1.0 - self._alpha_tf ** 2)
        self._pi2 = tf.constant(2.0 * np.pi, dtype=dtype)

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
        """E[y | x] = 0 for stochastic volatility."""
        return tf.zeros_like(x)

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Var[y | x] = beta^2 * exp(x)."""
        return tf.reshape(self._beta_tf ** 2 * tf.exp(x[0]), [1, 1])

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """dE[y|x]/dx = 0 since E[y|x] = 0."""
        return tf.zeros([1, 1], dtype=self.dtype)

    def observation_hessian(self, x: tf.Tensor) -> tf.Tensor:
        return tf.zeros([1, 1, 1], dtype=self.dtype)

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(x)

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        var = self._beta_tf ** 2 * tf.exp(x[0])
        return -0.5 * (tf.math.log(self._pi2 * var) + y[0] ** 2 / var)

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
        obs_vars = self._beta_tf ** 2 * tf.exp(particles[:, 0])
        diff_squared = observation[0] ** 2
        mahalanobis = diff_squared / obs_vars
        logdet = tf.math.log(self._pi2 * obs_vars)
        return -0.5 * (logdet + mahalanobis)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        return tf.zeros([N, 1, 1], dtype=self.dtype)

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(particles)

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        return tf.fill([N, 1, 1], self._alpha_tf)
