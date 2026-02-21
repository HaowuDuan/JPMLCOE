"""Cubic Sensor Model — mildly nonlinear 1D state-space model.

State evolution (linear AR(1)):
    x_t = a * x_{t-1} + v_t,   v_t ~ N(0, sigma_V^2)

Observation (mildly nonlinear):
    y_t = x_t + c * x_t^3 + w_t,   w_t ~ N(0, sigma_W^2)

The cubic coefficient c controls the degree of nonlinearity.
With c=0 this reduces to a linear-Gaussian model.
The observation Jacobian 1 + 3c*x^2 > 0 everywhere (injective).
"""

import numpy as np
import tensorflow as tf

from ..core.model_base import StateSpaceModel


def _as_sigma(sigma, dtype):
    """Convert sigma to TF tensor with given dtype."""
    return tf.cast(tf.convert_to_tensor(sigma), dtype)


class CubicSensorModel(StateSpaceModel):
    """
    Cubic Sensor Model.

    State evolution (linear):
        x_t = a * x_{t-1} + v_t,   v_t ~ N(0, sigma_V^2)

    Observation (mildly nonlinear):
        y_t = x_t + c * x_t^3 + w_t,   w_t ~ N(0, sigma_W^2)
    """

    def __init__(
        self,
        a: float = 0.9,
        c: float = 0.05,
        sigma_V: float = 1.0,
        sigma_W: float = 1.0,
        initial_var: float = 5.0,
        dtype=None,
    ):
        if not (0 < abs(a) < 1):
            raise ValueError(f"a must satisfy |a| < 1 for stability, got {a}")
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

        self.a = a
        self.c = c
        self.sigma_V = sigma_V
        self.sigma_W = sigma_W
        self.initial_var = initial_var

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
        return z * std

    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        a = tf.constant(self.a, dtype=self.dtype) if not isinstance(self.a, tf.Tensor) else self.a
        x0 = x[0] if len(x.shape) == 1 else x
        mean = a * x0
        w = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(mean, [1]) + sigma_V * w

    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        c = tf.constant(self.c, dtype=self.dtype)
        mean = x0 + c * x0 ** 3
        w = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(mean, [1]) + sigma_W * w

    # ------------------------------------------------------------------
    # Deterministic methods
    # ------------------------------------------------------------------

    def state_transition_mean(self, x: tf.Tensor, t=None) -> tf.Tensor:
        a = tf.constant(self.a, dtype=self.dtype) if not isinstance(self.a, tf.Tensor) else self.a
        x0 = x[0] if len(x.shape) == 1 else x
        return tf.reshape(a * x0, tf.shape(x))

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        a = tf.constant(self.a, dtype=self.dtype) if not isinstance(self.a, tf.Tensor) else self.a
        return tf.reshape(a, [1, 1])

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        x0 = x[0] if len(x.shape) == 1 else x
        c = tf.constant(self.c, dtype=self.dtype)
        return tf.reshape(x0 + c * x0 ** 3, tf.shape(x))

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        return tf.reshape(sigma_W ** 2, [1, 1])

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        x0 = x[0] if len(x.shape) == 1 else x
        c = tf.constant(self.c, dtype=self.dtype)
        jac = 1.0 + 3.0 * c * x0 ** 2
        return tf.reshape(jac, [1, 1])

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        return self.observation_mean(x)

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        c = tf.constant(self.c, dtype=self.dtype)
        mean = x0 + c * x0 ** 3
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

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor, t=None) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        a = tf.constant(self.a, dtype=self.dtype) if not isinstance(self.a, tf.Tensor) else self.a
        x = particles[:, 0:1]
        mean = a * x
        w = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
        return mean + sigma_V * w

    def state_transition_mean_batch(self, particles: tf.Tensor, t=None) -> tf.Tensor:
        a = tf.constant(self.a, dtype=self.dtype) if not isinstance(self.a, tf.Tensor) else self.a
        x = particles[:, 0:1]
        return a * x

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    def log_observation_prob_batch(self, observation: tf.Tensor,
                                   particles: tf.Tensor) -> tf.Tensor:
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        c = tf.constant(self.c, dtype=self.dtype)
        x = particles[:, 0]
        means = x + c * x ** 3
        var = sigma_W ** 2
        diff = observation[0] - means
        pi = tf.constant(np.pi, dtype=self.dtype)
        return -0.5 * (tf.math.log(2.0 * pi * var) + diff ** 2 / var)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        c = tf.constant(self.c, dtype=self.dtype)
        x = particles[:, 0]
        jac = 1.0 + 3.0 * c * x ** 2
        return tf.reshape(jac, [-1, 1, 1])

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        c = tf.constant(self.c, dtype=self.dtype)
        x = particles[:, 0]
        return tf.reshape(x + c * x ** 3, [-1, 1])

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        a = tf.constant(self.a, dtype=self.dtype) if not isinstance(self.a, tf.Tensor) else self.a
        N = tf.shape(particles)[0]
        return tf.fill([N, 1, 1], a)
