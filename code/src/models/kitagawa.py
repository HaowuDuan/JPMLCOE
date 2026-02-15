"""Kitagawa Model (Andrieu et al., 2010).

Univariate nonlinear/non-Gaussian state-space model:
    X_n = X_{n-1}/2 + 25*X_{n-1}/(1 + X_{n-1}^2) + 8*cos(1.2*n) + V_n
    Y_n = X_n^2/20 + W_n

where X_1 ~ N(0, 5), V_n ~ N(0, sigma_V^2), W_n ~ N(0, sigma_W^2).
"""

import numpy as np
import tensorflow as tf
from typing import Optional

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

    Parameters:
        sigma_V: Process noise standard deviation
        sigma_W: Observation noise standard deviation
        initial_var: Variance of initial state distribution N(0, initial_var)

    Note: The state transition depends on the time step n through the
    deterministic term 8*cos(1.2*n). The model tracks the current time
    step internally, advancing it on each call to sample_state_transition
    or state_transition_mean. For batch filtering, use the _with_t variants
    or set self.t directly.
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
    def mu_0(self):
        return tf.zeros([1], dtype=self.dtype)

    @property
    def Sigma_0(self):
        iv = self.initial_var
        if isinstance(iv, tf.Tensor):
            return tf.reshape(iv, [1, 1])
        return tf.constant([[float(iv)]], dtype=self.dtype)

    @property
    def initial_state_mean(self):
        return self.mu_0

    @property
    def initial_state_cov(self):
        return self.Sigma_0

    # ----------------------------------------------------------------
    # Deterministic parts of the model (numpy helpers)
    # ----------------------------------------------------------------

    def _f(self, x: float, n: int) -> float:
        """Deterministic part of state transition: f(x, n)."""
        return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * n)

    def _df_dx(self, x: float) -> float:
        """df/dx = 1/2 + 25*(1 - x^2)/(1 + x^2)^2"""
        return 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2

    def _h(self, x: float) -> float:
        """h(x) = x^2 / 20."""
        return x ** 2 / 20.0

    def _dh_dx(self, x: float) -> float:
        """dh/dx = x / 10."""
        return x / 10.0

    # ----------------------------------------------------------------
    # NumPy sampling methods (for data generation)
    # ----------------------------------------------------------------

    def sample_initial_state(self, rng):
        """Sample from initial distribution: X_0 ~ N(0, initial_var)."""
        if hasattr(rng, 'standard_normal'):
            # NumPy rng
            self.t = 0
            return np.array([rng.normal(0, np.sqrt(float(self.initial_var)))], dtype=self.np_dtype)
        # TF seed
        seed = rng
        std = tf.sqrt(_as_sigma(self.initial_var, self.dtype))
        z = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        self.t = 0
        return z * std

    def sample_state_transition(self, x, rng):
        """Sample next state: x_n = f(x_{n-1}, n) + V_n."""
        if hasattr(rng, 'standard_normal'):
            self.t += 1
            mean = self._f(float(x[0]), self.t)
            return np.array([mean + float(self.sigma_V) * rng.normal()], dtype=self.np_dtype)
        # TF seed
        seed = rng
        self.t += 1
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        t_val = tf.cast(self.t, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
        w = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(mean, [1]) + sigma_V * w

    def sample_observation(self, x, rng):
        """Sample observation: y_n = x_n^2/20 + W_n."""
        if hasattr(rng, 'standard_normal'):
            mean = self._h(float(x[0]))
            return np.array([mean + float(self.sigma_W) * rng.normal()], dtype=self.np_dtype)
        seed = rng
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        x0 = x[0] if len(x.shape) == 1 else x
        mean = x0 ** 2 / 20.0
        w = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        return tf.reshape(mean, [1]) + sigma_W * w

    # ----------------------------------------------------------------
    # TF-compatible deterministic methods (for EKF and HMC)
    # ----------------------------------------------------------------

    def state_transition_mean(self, x):
        """E[x_n | x_{n-1}] = f(x_{n-1}, n). Uses current self.t."""
        if isinstance(x, tf.Tensor):
            t_val = tf.cast(self.t, self.dtype)
            x0 = x[0] if len(x.shape) == 1 else x
            mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
            return tf.reshape(mean, tf.shape(x))
        return np.array([self._f(x[0], self.t)], dtype=self.np_dtype)

    def state_transition_mean_with_t(self, x, t: int):
        """E[x_n | x_{n-1}] = f(x_{n-1}, n) with explicit time step."""
        if isinstance(x, tf.Tensor):
            t_val = tf.cast(t, self.dtype)
            x0 = x[0] if len(x.shape) == 1 else x
            mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
            return tf.reshape(mean, tf.shape(x))
        return np.array([self._f(x[0], t)], dtype=self.np_dtype)

    def state_transition_cov(self, x):
        """Cov[x_n | x_{n-1}] = sigma_V^2."""
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    def state_jacobian(self, x):
        """Jacobian of f w.r.t. x: df/dx (does not depend on t)."""
        if isinstance(x, tf.Tensor):
            x0 = x[0] if len(x.shape) == 1 else x
            jac = 0.5 + 25.0 * (1.0 - x0 ** 2) / (1.0 + x0 ** 2) ** 2
            return tf.reshape(jac, [1, 1])
        return np.array([[self._df_dx(x[0])]], dtype=self.np_dtype)

    def observation_mean(self, x):
        """E[y_n | x_n] = x_n^2 / 20."""
        if isinstance(x, tf.Tensor):
            x0 = x[0] if len(x.shape) == 1 else x
            return tf.reshape(x0 ** 2 / 20.0, tf.shape(x))
        return np.array([self._h(x[0])], dtype=self.np_dtype)

    def observation_cov(self, x):
        """Cov[y_n | x_n] = sigma_W^2."""
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        return tf.reshape(sigma_W ** 2, [1, 1])

    def observation_jacobian(self, x):
        """Jacobian of h w.r.t. x: dh/dx = x/10."""
        if isinstance(x, tf.Tensor):
            x0 = x[0] if len(x.shape) == 1 else x
            return tf.reshape(x0 / 10.0, [1, 1])
        return np.array([[self._dh_dx(x[0])]], dtype=self.np_dtype)

    def observation_hessian(self, x):
        """Hessian of h w.r.t. x: d^2h/dx^2 = 1/10."""
        if isinstance(x, tf.Tensor):
            return tf.constant([[[0.1]]], dtype=x.dtype)
        return np.array([[[1.0 / 10.0]]], dtype=self.np_dtype)

    def observation_function(self, x):
        """Observation function h(x) for flow filters."""
        return self.observation_mean(x)

    def observe(self, x):
        """Observation operator h(x) for kernel flow filter."""
        return self.observation_mean(x)

    # ----------------------------------------------------------------
    # Log-probability
    # ----------------------------------------------------------------

    def log_observation_prob(self, y, x):
        """log p(y_n | x_n) = log N(y_n; x_n^2/20, sigma_W^2)."""
        if isinstance(x, tf.Tensor):
            sigma_W = _as_sigma(self.sigma_W, self.dtype)
            x0 = x[0] if len(x.shape) == 1 else x
            mean = x0 ** 2 / 20.0
            var = sigma_W ** 2
            pi = tf.constant(np.pi, dtype=self.dtype)
            return -0.5 * (tf.math.log(2.0 * pi * var) + (y[0] - mean) ** 2 / var)
        mean = self._h(x[0])
        var = float(self.sigma_W) ** 2
        return -0.5 * (np.log(2 * np.pi * var) + (y[0] - mean) ** 2 / var)

    # ----------------------------------------------------------------
    # Properties for flow filters (dynamic — work with tf.Tensor sigma)
    # ----------------------------------------------------------------

    @property
    def observation_noise_cov(self):
        """Observation noise covariance R = sigma_W^2."""
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        return tf.reshape(sigma_W ** 2, [1, 1])

    @property
    def process_noise_cov(self):
        """Process noise covariance Q = sigma_V^2."""
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    # ----------------------------------------------------------------
    # Batch methods (vectorized for particle filters — TF-compatible)
    # ----------------------------------------------------------------

    def state_transition_batch(self, particles, seed):
        """Stochastic batch state transition: x' = f(x, t) + sigma_V * w.

        Args:
            particles: (N, 1) particle states
            seed: TF random seed

        Returns:
            (N, 1) next states
        """
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        t_val = tf.cast(self.t, self.dtype)
        x = particles[:, 0:1]
        mean = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
        w = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
        return mean + sigma_V * w

    def state_transition_mean_batch(self, particles):
        """Vectorized state transition mean for N particles.

        Args:
            particles: (N, 1) array of states (TF tensor or numpy)

        Returns:
            (N, 1) transition means
        """
        if isinstance(particles, tf.Tensor):
            t_val = tf.cast(self.t, self.dtype)
            x = particles[:, 0:1]
            return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
        x = particles[:, 0]
        means = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * self.t)
        return means[:, np.newaxis]

    def state_transition_cov_batch(self, particles):
        """Q is constant — return single matrix."""
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        return tf.reshape(sigma_V ** 2, [1, 1])

    def log_observation_prob_batch(self, observation, particles):
        """Vectorized log p(y | x) for all particles.

        Args:
            observation: (1,) observation
            particles: (N, 1) particle states

        Returns:
            (N,) log probabilities
        """
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            means = x ** 2 / 20.0
            var = sigma_W ** 2
            diff = observation[0] - means
            pi = tf.constant(np.pi, dtype=self.dtype)
            return -0.5 * (tf.math.log(2.0 * pi * var) + diff ** 2 / var)
        x = particles[:, 0]
        means = x ** 2 / 20.0
        var = float(self.sigma_W) ** 2
        diff = observation[0] - means
        return -0.5 * (np.log(2 * np.pi * var) + diff ** 2 / var)

    def observation_jacobian_batch(self, particles):
        """Vectorized observation Jacobian: dh/dx = x/10.

        Args:
            particles: (N, 1)

        Returns:
            (N, 1, 1) Jacobians
        """
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            return tf.reshape(x / 10.0, [-1, 1, 1])
        x = particles[:, 0]
        return (x / 10.0)[:, np.newaxis, np.newaxis]

    def observation_function_batch(self, particles):
        """Vectorized h(x) = x^2/20.

        Args:
            particles: (N, 1)

        Returns:
            (N, 1)
        """
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            return tf.reshape(x ** 2 / 20.0, [-1, 1])
        x = particles[:, 0]
        return (x ** 2 / 20.0)[:, np.newaxis]

    def state_jacobian_batch(self, particles):
        """Vectorized state Jacobian df/dx.

        Args:
            particles: (N, 1)

        Returns:
            (N, 1, 1) Jacobians
        """
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            jac = 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2
            return tf.reshape(jac, [-1, 1, 1])
        x = particles[:, 0]
        jac = 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2
        return jac[:, np.newaxis, np.newaxis]

    def sample_initial_state_batch(self, n, seed):
        """Sample n initial states.

        Args:
            n: Number of samples
            seed: TF random seed

        Returns:
            Initial states (n, 1)
        """
        iv = _as_sigma(self.initial_var, self.dtype)
        std = tf.sqrt(iv)
        return tf.random.stateless_normal([n, 1], seed=seed, dtype=self.dtype) * std

    # ----------------------------------------------------------------
    # Legacy TF methods (kept for backward compatibility)
    # ----------------------------------------------------------------

    @tf.function
    def sample_state_transition_tf(self, x_tf: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """TensorFlow state transition sampling (vectorized)."""
        input_shape = tf.shape(x_tf)
        w = tf.random.stateless_normal(input_shape, seed=seed, dtype=self.dtype)
        sigma_V = _as_sigma(self.sigma_V, self.dtype)
        t_val = tf.cast(self.t, self.dtype)
        if len(x_tf.shape) == 1:
            x = x_tf[0]
        else:
            x = x_tf[:, 0:1]
        mean = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
        if len(x_tf.shape) == 1:
            return tf.expand_dims(mean, 0) + sigma_V * w
        else:
            return mean + sigma_V * w

    @tf.function
    def log_observation_prob_tf(self, y_tf: tf.Tensor, x_tf: tf.Tensor) -> tf.Tensor:
        """TensorFlow observation log-probability (vectorized)."""
        sigma_W = _as_sigma(self.sigma_W, self.dtype)
        var = sigma_W ** 2
        if len(x_tf.shape) == 1:
            mean = x_tf[0] ** 2 / 20.0
            return -0.5 * (tf.math.log(2.0 * np.pi * var) + (y_tf[0] - mean) ** 2 / var)
        else:
            means = x_tf[:, 0] ** 2 / 20.0
            diff = y_tf[0] - means
            return -0.5 * (tf.math.log(2.0 * np.pi * var) + diff ** 2 / var)

    @tf.function
    def sample_initial_state_batch_tf(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Sample n initial states using TensorFlow."""
        iv = _as_sigma(self.initial_var, self.dtype)
        std = tf.sqrt(iv)
        return tf.random.stateless_normal([n, 1], seed=seed, dtype=self.dtype) * std
