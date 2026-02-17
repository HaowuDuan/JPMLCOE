"""Kitagawa Model (Andrieu et al., 2010) — TensorFlow.

Univariate nonlinear/non-Gaussian state-space model:
    X_n = X_{n-1}/2 + 25*X_{n-1}/(1 + X_{n-1}^2) + 8*cos(1.2*n) + V_n
    Y_n = X_n^2/20 + W_n

where X_1 ~ N(0, 5), V_n ~ N(0, sigma_V^2), W_n ~ N(0, sigma_W^2).
"""

import numpy as np
import tensorflow as tf

from ..core.model_base import StateSpaceModel


def _as_sigma(sigma, dtype):
    """Cast sigma to TF tensor if it isn't already."""
    if isinstance(sigma, tf.Tensor):
        return sigma
    return tf.constant(float(sigma), dtype=dtype)


class KitagawaModel(StateSpaceModel):
    """
    Kitagawa Model (Model from Andrieu et al., 2010).

    State evolution (nonlinear):
        x_n = x_{n-1}/2 + 25*x_{n-1}/(1 + x_{n-1}^2) + 8*cos(1.2*n) + v_n

    Observation (nonlinear):
        y_n = x_n^2/20 + w_n

    Note: The state transition depends on the time step n through the
    deterministic term 8*cos(1.2*n). The model tracks the current time
    step internally.
    """

    def __init__(
        self,
        sigma_V: float = 10.0,
        sigma_W: float = 1.0,
        initial_var: float = 5.0,
        dtype=None,
    ):
        if sigma_V <= 0:
            raise ValueError(f"sigma_V must be positive, got {sigma_V}")
        if sigma_W <= 0:
            raise ValueError(f"sigma_W must be positive, got {sigma_W}")
        if initial_var <= 0:
            raise ValueError(f"initial_var must be positive, got {initial_var}")

        if dtype is None:
            dtype = tf.float32
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        self.sigma_V = sigma_V
        self.sigma_W = sigma_W
        self.initial_var = initial_var

        # Current time step (1-indexed: first transition produces x_1)
        self.t = 0

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
        iv = self.initial_var
        if isinstance(iv, tf.Tensor):
            return tf.reshape(iv, [1, 1])
        return tf.constant([[float(iv)]], dtype=self.dtype)

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        return tf.reshape(sigma_W ** 2, [1, 1])

    @property
    def process_noise_cov(self) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        std = tf.sqrt(_as_sigma(self.initial_var, self.dtype))
        z = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        self.t = 0
        return z * std

    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        self.t += 1
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        t_val = tf.cast(self.t, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
        w = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(mean, [1]) + sigma_V * w

    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        mean = x0 ** 2 / 20.0
        w = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(mean, [1]) + sigma_W * w

    # ------------------------------------------------------------------
    # Deterministic methods
    # ------------------------------------------------------------------

    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        t_val = tf.cast(self.t, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
        return tf.reshape(mean, tf.shape(x))

    def state_transition_mean_with_t(self, x: tf.Tensor, t: int) -> tf.Tensor:
        """E[x_n | x_{n-1}] = f(x_{n-1}, n) with explicit time step."""
        t_val = tf.cast(t, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
        return tf.reshape(mean, tf.shape(x))

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        x0 = x[0] if len(x.shape) == 1 else x
        jac = 0.5 + 25.0 * (1.0 - x0 ** 2) / (1.0 + x0 ** 2) ** 2
        return tf.reshape(jac, [1, 1])

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        x0 = x[0] if len(x.shape) == 1 else x
        return tf.reshape(x0 ** 2 / 20.0, tf.shape(x))

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        return tf.reshape(sigma_W ** 2, [1, 1])

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        x0 = x[0] if len(x.shape) == 1 else x
        return tf.reshape(x0 / 10.0, [1, 1])

    def observation_hessian(self, x: tf.Tensor) -> tf.Tensor:
        return tf.constant([[[0.1]]], dtype=self.dtype)

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        return self.observation_mean(x)

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        mean = x0 ** 2 / 20.0
        var = sigma_W ** 2
        pi = tf.constant(np.pi, dtype=self.dtype)
        return -0.5 * (tf.math.log(2.0 * pi * var) + (y[0] - mean) ** 2 / var)

    # ------------------------------------------------------------------
    # Batch methods
    # ------------------------------------------------------------------

    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        iv = _as_sigma(self.initial_var, self.dtype)
        std = tf.sqrt(iv)
        return tf.random.stateless_normal([n, 1], seed=seed, dtype=self.dtype) * std

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        t_val = tf.cast(self.t, self.dtype)
        x = particles[:, 0:1]
        mean = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
        w = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
        return mean + sigma_V * w

    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        t_val = tf.cast(self.t, self.dtype)
        x = particles[:, 0:1]
        return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    def log_observation_prob_batch(self, observation: tf.Tensor,
                                   particles: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        x = particles[:, 0]
        means = x ** 2 / 20.0
        var = sigma_W ** 2
        diff = observation[0] - means
        pi = tf.constant(np.pi, dtype=self.dtype)
        return -0.5 * (tf.math.log(2.0 * pi * var) + diff ** 2 / var)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        x = particles[:, 0]
        return tf.reshape(x / 10.0, [-1, 1, 1])

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        x = particles[:, 0]
        return tf.reshape(x ** 2 / 20.0, [-1, 1])

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        x = particles[:, 0]
        jac = 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2
        return tf.reshape(jac, [-1, 1, 1])
