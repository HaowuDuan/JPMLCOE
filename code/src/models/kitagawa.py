"""Kitagawa Model (Andrieu et al., 2010).

Univariate nonlinear/non-Gaussian state-space model:
    X_n = X_{n-1}/2 + 25*X_{n-1}/(1 + X_{n-1}^2) + 8*cos(1.2*n) + V_n
    Y_n = X_n^2/20 + W_n

where X_1 ~ N(0, 5), V_n ~ N(0, sigma_V^2), W_n ~ N(0, sigma_W^2).
"""

import numpy as np
from typing import Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


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

        import tensorflow as tf
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

    # ----------------------------------------------------------------
    # Deterministic parts of the model
    # ----------------------------------------------------------------

    def _f(self, x: float, n: int) -> float:
        """Deterministic part of state transition: f(x, n)."""
        return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * n)

    def _df_dx(self, x: float) -> float:
        """Derivative of f w.r.t. x (time-independent part):
        df/dx = 1/2 + 25*(1 - x^2)/(1 + x^2)^2
        """
        return 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2

    def _h(self, x: float) -> float:
        """Observation function: h(x) = x^2 / 20."""
        return x ** 2 / 20.0

    def _dh_dx(self, x: float) -> float:
        """Derivative of observation function: dh/dx = x / 10."""
        return x / 10.0

    def _d2h_dx2(self, x: float) -> float:
        """Second derivative of observation function: d^2h/dx^2 = 1/10."""
        return 1.0 / 10.0

    # ----------------------------------------------------------------
    # NumPy sampling methods
    # ----------------------------------------------------------------

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample from initial distribution: X_0 ~ N(0, initial_var)."""
        self.t = 0
        return np.array([rng.normal(0, np.sqrt(self.initial_var))], dtype=self.np_dtype)

    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample next state: x_n = f(x_{n-1}, n) + V_n."""
        self.t += 1
        mean = self._f(x[0], self.t)
        return np.array([mean + self.sigma_V * rng.normal()], dtype=self.np_dtype)

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample observation: y_n = x_n^2/20 + W_n."""
        mean = self._h(x[0])
        return np.array([mean + self.sigma_W * rng.normal()], dtype=self.np_dtype)

    # ----------------------------------------------------------------
    # Deterministic methods (for EKF / UKF)
    # ----------------------------------------------------------------

    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """E[x_n | x_{n-1}] = f(x_{n-1}, n). Uses current self.t."""
        return np.array([self._f(x[0], self.t)], dtype=self.np_dtype)

    def state_transition_mean_with_t(self, x: np.ndarray, t: int) -> np.ndarray:
        """E[x_n | x_{n-1}] = f(x_{n-1}, n) with explicit time step."""
        return np.array([self._f(x[0], t)], dtype=self.np_dtype)

    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Cov[x_n | x_{n-1}] = sigma_V^2."""
        return np.array([[self.sigma_V ** 2]], dtype=self.np_dtype)

    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of f w.r.t. x: df/dx (does not depend on t)."""
        return np.array([[self._df_dx(x[0])]], dtype=self.np_dtype)

    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """E[y_n | x_n] = x_n^2 / 20."""
        return np.array([self._h(x[0])], dtype=self.np_dtype)

    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Cov[y_n | x_n] = sigma_W^2."""
        return np.array([[self.sigma_W ** 2]], dtype=self.np_dtype)

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h w.r.t. x: dh/dx = x/10."""
        return np.array([[self._dh_dx(x[0])]], dtype=self.np_dtype)

    def observation_hessian(self, x: np.ndarray) -> np.ndarray:
        """Hessian of h w.r.t. x: d^2h/dx^2 = 1/10."""
        return np.array([[[self._d2h_dx2(x[0])]]], dtype=self.np_dtype)

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters."""
        return self.observation_mean(x)

    def observe(self, x: np.ndarray) -> np.ndarray:
        """Observation operator h(x) for kernel flow filter."""
        return self.observation_mean(x)

    # ----------------------------------------------------------------
    # Log-probability
    # ----------------------------------------------------------------

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """log p(y_n | x_n) = log N(y_n; x_n^2/20, sigma_W^2)."""
        mean = self._h(x[0])
        var = self.sigma_W ** 2
        return -0.5 * (np.log(2 * np.pi * var) + (y[0] - mean) ** 2 / var)

    # ----------------------------------------------------------------
    # Properties for flow filters
    # ----------------------------------------------------------------

    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R."""
        return np.array([[self.sigma_W ** 2]], dtype=self.np_dtype)

    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q."""
        return np.array([[self.sigma_V ** 2]], dtype=self.np_dtype)

    # ----------------------------------------------------------------
    # Batch methods (vectorized for particle filters)
    # ----------------------------------------------------------------

    def state_transition_mean_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized state transition mean for N particles.

        Args:
            particles: (N, 1) array of states

        Returns:
            (N, 1) array of transition means
        """
        x = particles[:, 0]
        means = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * self.t)
        return means[:, np.newaxis]

    def state_transition_cov_batch(self, particles: np.ndarray) -> np.ndarray:
        """Q is constant - return single matrix."""
        return np.array([[self.sigma_V ** 2]], dtype=self.np_dtype)

    def log_observation_prob_batch(self, observation: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Vectorized log p(y | x) for all particles.

        Args:
            observation: (1,) observation
            particles: (N, 1) particle states

        Returns:
            (N,) log probabilities
        """
        x = particles[:, 0]
        means = x ** 2 / 20.0
        var = self.sigma_W ** 2
        diff = observation[0] - means
        return -0.5 * (np.log(2 * np.pi * var) + diff ** 2 / var)

    def observation_jacobian_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized observation Jacobian.

        Args:
            particles: (N, 1)

        Returns:
            (N, 1, 1) Jacobians
        """
        x = particles[:, 0]
        return (x / 10.0)[:, np.newaxis, np.newaxis]

    def observation_function_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized h(x) = x^2/20.

        Args:
            particles: (N, 1)

        Returns:
            (N, 1)
        """
        x = particles[:, 0]
        return (x ** 2 / 20.0)[:, np.newaxis]

    def state_jacobian_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized state Jacobian df/dx.

        Args:
            particles: (N, 1)

        Returns:
            (N, 1, 1) Jacobians
        """
        x = particles[:, 0]
        jac = 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2
        return jac[:, np.newaxis, np.newaxis]

    # ----------------------------------------------------------------
    # TensorFlow methods
    # ----------------------------------------------------------------

    if TF_AVAILABLE:
        @tf.function
        def sample_state_transition_tf(self, x_tf: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
            """TensorFlow state transition sampling (vectorized).

            Args:
                x_tf: (N, 1) or (1,)
                seed: Random seed

            Returns:
                Next states, same shape as input
            """
            input_shape = tf.shape(x_tf)
            w = tf.random.stateless_normal(input_shape, seed=seed)
            t_val = tf.cast(self.t, self.dtype)
            if len(x_tf.shape) == 1:
                x = x_tf[0]
            else:
                x = x_tf[:, 0:1]
            mean = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
            if len(x_tf.shape) == 1:
                return tf.expand_dims(mean, 0) + tf.constant(self.sigma_V, dtype=self.dtype) * w
            else:
                return mean + tf.constant(self.sigma_V, dtype=self.dtype) * w

        @tf.function
        def log_observation_prob_tf(self, y_tf: tf.Tensor, x_tf: tf.Tensor) -> tf.Tensor:
            """TensorFlow observation log-probability (vectorized).

            Args:
                y_tf: (1,) observation
                x_tf: (N, 1) or (1,)

            Returns:
                Log probabilities, shape (N,) or scalar
            """
            var = tf.constant(self.sigma_W ** 2, dtype=self.dtype)
            if len(x_tf.shape) == 1:
                mean = x_tf[0] ** 2 / 20.0
                return -0.5 * (tf.math.log(2.0 * np.pi * var) + (y_tf[0] - mean) ** 2 / var)
            else:
                means = x_tf[:, 0] ** 2 / 20.0
                diff = y_tf[0] - means
                return -0.5 * (tf.math.log(2.0 * np.pi * var) + diff ** 2 / var)

        @tf.function
        def sample_initial_state_batch_tf(self, n: int, seed: tf.Tensor) -> tf.Tensor:
            """Sample n initial states using TensorFlow.

            Args:
                n: Number of samples
                seed: Random seed

            Returns:
                Initial states (n, 1)
            """
            std = tf.constant(np.sqrt(self.initial_var), dtype=self.dtype)
            return tf.random.stateless_normal([n, 1], seed=seed) * std
