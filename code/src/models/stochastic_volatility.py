"""Stochastic Volatility model."""

import numpy as np
from typing import Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


class StochasticVolatilityModel(StateSpaceModel):
    """
    1D Stochastic Volatility Model.

    State evolution (linear):
        x_t = α·x_{t-1} + σ·w_t,  w_t ~ N(0, 1)

    Observation (nonlinear, non-Gaussian):
        y_t = β·exp(x_t/2)·v_t,  v_t ~ N(0, 1)

    Parameters:
        α: persistence parameter (0 < α < 1)
        σ: volatility of volatility
        β: scale parameter

    Key features:
    - Linear state evolution
    - Nonlinear observation (exponential)
    - Non-Gaussian observation likelihood
    - Stationary variance: σ²/(1 - α²)
    """

    def __init__(self, alpha: float = 0.91, sigma: float = 1.0, beta: float = 0.5, dtype=None):
        """
        Initialize Stochastic Volatility Model.

        Args:
            alpha: Persistence parameter (0 < alpha < 1)
            sigma: Volatility of volatility
            beta: Scale parameter
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        import tensorflow as tf
        if dtype is None:
            dtype = tf.float32
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta

        # Stationary variance
        self.stationary_var = (sigma ** 2) / (1 - alpha ** 2)

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return 1

    # NumPy methods

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample from stationary distribution: N(0, σ²/(1-α²))."""
        return np.array([rng.normal(0, np.sqrt(self.stationary_var))])

    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = α·x + σ·w."""
        w = rng.normal(0, 1)
        return np.array([self.alpha * x[0] + self.sigma * w])

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample observation: y = β·exp(x/2)·v."""
        v = rng.normal(0, 1)
        return np.array([self.beta * np.exp(x[0] / 2) * v])

    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = α·x."""
        return np.array([self.alpha * x[0]])

    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Var[x' | x] = σ²."""
        return np.array([[self.sigma ** 2]])

    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂f/∂x = α."""
        return np.array([[self.alpha]])

    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[y | x] = 0."""
        return np.array([0.0])

    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Var[y | x] = β²·exp(x)."""
        return np.array([[self.beta ** 2 * np.exp(x[0])]])

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation mean.

        Note: Since E[y | x] = 0, the Jacobian is 0.
        For EKF, this means the filter cannot update based on the observation mean.
        Alternative approaches use the observation variance or squared observations.
        """
        return np.array([[0.0]])

    def observation_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Hessian of observation function: ∂²h/∂x².
        
        Since E[y|x] = 0 (constant), the Hessian is also 0.
        
        Returns:
            Tensor of shape (obs_dim=1, state_dim=1, state_dim=1), all zeros.
        """
        return np.zeros((1, 1, 1))

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).

        p(y | x) = N(y | 0, β²·exp(x))
        """
        var = self.beta ** 2 * np.exp(x[0])
        return -0.5 * (np.log(2 * np.pi * var) + (y[0] ** 2) / var)

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters: returns observation mean."""
        return self.observation_mean(x)

    def observe(self, x: np.ndarray) -> np.ndarray:
        """Observation operator h(x) for kernel flow filter: returns observation mean."""
        return self.observation_mean(x)

    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters.

        For stochastic volatility, use observation covariance at stationary mean (x=0).
        """
        return np.array([[self.beta ** 2]])

    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return np.array([[self.sigma ** 2]])

    # Batch methods for optimized particle filtering

    def state_transition_mean_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized state transition mean: α·x."""
        return self.alpha * particles

    def state_transition_cov_batch(self, particles: np.ndarray) -> np.ndarray:
        """Q is constant - return single matrix."""
        return np.array([[self.sigma ** 2]])

    def log_observation_prob_batch(self, observation: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Vectorized observation log-prob for all particles."""
        # For stochastic volatility: y ~ N(0, β²·exp(x))
        # This is state-dependent R, so we need per-particle computation
        # However, we can still vectorize the operations

        # Observation covariances for each particle: β²·exp(x)
        obs_vars = (self.beta ** 2) * np.exp(particles[:, 0])  # (N,)

        # y - 0 = y (observation mean is 0)
        diff_squared = observation[0] ** 2  # scalar

        # Mahalanobis: (y - 0)² / σ²(x) for each particle
        mahalanobis = diff_squared / obs_vars  # (N,)

        # Log determinant: log(2π·σ²(x))
        logdet = np.log(2 * np.pi * obs_vars)  # (N,)

        return -0.5 * (logdet + mahalanobis)

    # TensorFlow methods

    if TF_AVAILABLE:
        @tf.function
        def sample_state_transition_tf(self, x_tf: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of state transition sampling (vectorized).

            Args:
                x_tf: Current states, shape (N, 1) or (1,)
                seed: Random seed

            Returns:
                Next states, same shape as input
            """
            # Determine shape
            input_shape = tf.shape(x_tf)

            # Generate noise with correct shape
            w = tf.random.stateless_normal(input_shape, seed=seed, dtype=self.dtype)

            # State transition: x' = α·x + σ·w
            return tf.constant(self.alpha, dtype=self.dtype) * x_tf + tf.constant(self.sigma, dtype=self.dtype) * w

        @tf.function
        def log_observation_prob_tf(self, y_tf: tf.Tensor, x_tf: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of observation log-probability (vectorized).

            Args:
                y_tf: Observation, shape (1,)
                x_tf: States, shape (N, 1) or (1,)

            Returns:
                Log probabilities, shape (N,) or scalar
            """
            # Handle both single and batch inputs
            if len(x_tf.shape) == 1:
                # Single state (1,)
                var = self.beta ** 2 * tf.exp(x_tf[0])
                log_prob = -0.5 * (tf.math.log(2.0 * np.pi * var) + (y_tf[0] ** 2) / var)
            else:
                # Batch of states (N, 1)
                var = self.beta ** 2 * tf.exp(x_tf[:, 0])
                log_prob = -0.5 * (tf.math.log(2.0 * np.pi * var) + (y_tf[0] ** 2) / var)

            return log_prob

        @tf.function
        def sample_initial_state_batch_tf(self, n: int, seed: tf.Tensor) -> tf.Tensor:
            """
            Sample n initial states using TensorFlow.

            Args:
                n: Number of samples
                seed: Random seed

            Returns:
                Initial states (n, 1)
            """
            samples = tf.random.stateless_normal([n, 1], seed=seed, dtype=self.dtype) * np.sqrt(self.stationary_var)
            return samples
