"""Range-Bearing tracking model."""

import numpy as np
from typing import Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


class RangeBearingModel(StateSpaceModel):
    """
    Range-Bearing Observation Model.

    State: 2D position [x, y]
    Observation: [range, bearing]

    State evolution (linear with process noise):
        x_t = F·x_{t-1} + w_t,  w_t ~ N(0, Q)

    Observation (nonlinear):
        range = sqrt((x - x_sensor)² + (y - y_sensor)²) + v_range
        bearing = atan2(y - y_sensor, x - x_sensor) + v_bearing

    where:
        - (x_sensor, y_sensor): Sensor position
        - v_range ~ N(0, σ_range²): Range measurement noise
        - v_bearing ~ N(0, σ_bearing²): Bearing measurement noise

    Parameters:
        F: State transition matrix (2, 2) - typically identity for constant position
        Q: Process noise covariance (2, 2)
        sensor_pos: Sensor position [x_sensor, y_sensor]
        sigma_range: Standard deviation of range measurement noise
        sigma_bearing: Standard deviation of bearing measurement noise (in radians)
        mu_0: Initial state mean [x_0, y_0]
        Sigma_0: Initial state covariance (2, 2)
    """

    def __init__(
        self,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        sensor_pos: np.ndarray = np.array([0.0, 0.0]),
        sigma_range: float = 0.1,
        sigma_bearing: float = 0.01,
        mu_0: Optional[np.ndarray] = None,
        Sigma_0: Optional[np.ndarray] = None
    ):
        """
        Initialize Range-Bearing Model.

        Args:
            F: State transition matrix (2, 2). If None, uses identity.
            Q: Process noise covariance (2, 2). If None, uses 0.01 * I.
            sensor_pos: Sensor position [x_sensor, y_sensor]
            sigma_range: Standard deviation of range measurement noise
            sigma_bearing: Standard deviation of bearing measurement noise (radians)
            mu_0: Initial state mean [x_0, y_0]. If None, uses [1.0, 1.0].
            Sigma_0: Initial state covariance (2, 2). If None, uses I.
        """
        if sigma_range <= 0:
            raise ValueError(f"sigma_range must be positive, got {sigma_range}")
        if sigma_bearing <= 0:
            raise ValueError(f"sigma_bearing must be positive, got {sigma_bearing}")

        # Default state transition (identity for constant position)
        if F is None:
            F = np.eye(2)
        if F.shape != (2, 2):
            raise ValueError(f"F must be (2, 2), got {F.shape}")
        self.F = F

        # Default process noise
        if Q is None:
            Q = 0.01 * np.eye(2)
        if Q.shape != (2, 2):
            raise ValueError(f"Q must be (2, 2), got {Q.shape}")
        self.Q = Q

        # Sensor position
        sensor_pos = np.asarray(sensor_pos)
        if sensor_pos.shape != (2,):
            raise ValueError(f"sensor_pos must be (2,), got {sensor_pos.shape}")
        self.sensor_pos = sensor_pos

        # Observation noise parameters
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing
        self.R = np.diag([sigma_range ** 2, sigma_bearing ** 2])

        # Initial state distribution
        self.mu_0 = mu_0 if mu_0 is not None else np.array([1.0, 1.0])
        self.Sigma_0 = Sigma_0 if Sigma_0 is not None else np.eye(2)

        if self.mu_0.shape != (2,):
            raise ValueError(f"mu_0 must be (2,), got {self.mu_0.shape}")
        if self.Sigma_0.shape != (2, 2):
            raise ValueError(f"Sigma_0 must be (2, 2), got {self.Sigma_0.shape}")

        # TensorFlow constants
        if TF_AVAILABLE:
            self.F_tf = tf.constant(self.F, dtype=tf.float32)
            self.Q_tf = tf.constant(self.Q, dtype=tf.float32)
            self.R_tf = tf.constant(self.R, dtype=tf.float32)
            self.sensor_pos_tf = tf.constant(self.sensor_pos, dtype=tf.float32)
            self.mu_0_tf = tf.constant(self.mu_0, dtype=tf.float32)
            self.Sigma_0_tf = tf.constant(self.Sigma_0, dtype=tf.float32)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 2

    # NumPy methods

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0)."""
        return rng.multivariate_normal(self.mu_0, self.Sigma_0)

    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = F·x + w, w ~ N(0, Q)."""
        w = rng.multivariate_normal(np.zeros(2), self.Q)
        return self.F @ x + w

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample observation: [range, bearing] with additive noise.

        range = sqrt((x - x_sensor)² + (y - y_sensor)²) + v_range
        bearing = atan2(y - y_sensor, x - x_sensor) + v_bearing
        """
        # Relative position
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]

        # True range and bearing
        range_true = np.sqrt(dx ** 2 + dy ** 2)
        bearing_true = np.arctan2(dy, dx)

        # Add noise
        v_range = rng.normal(0, self.sigma_range)
        v_bearing = rng.normal(0, self.sigma_bearing)

        return np.array([range_true + v_range, bearing_true + v_bearing])

    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = F·x."""
        return self.F @ x

    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Cov[x' | x] = Q."""
        return self.Q

    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂(F·x)/∂x = F."""
        return self.F

    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Mean of observation: E[y | x] = [range, bearing].

        range = sqrt((x - x_sensor)² + (y - y_sensor)²)
        bearing = atan2(y - y_sensor, x - x_sensor)
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        range_val = np.sqrt(dx ** 2 + dy ** 2)
        bearing_val = np.arctan2(dy, dx)
        return np.array([range_val, bearing_val])

    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Cov[y | x] = R (constant)."""
        return self.R

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation: ∂h/∂x.

        h(x) = [sqrt((x-xs)² + (y-ys)²), atan2(y-ys, x-xs)]

        ∂range/∂x = (x - xs) / range
        ∂range/∂y = (y - ys) / range
        ∂bearing/∂x = -(y - ys) / range²
        ∂bearing/∂y = (x - xs) / range²
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        range_val = np.sqrt(dx ** 2 + dy ** 2)

        # Avoid division by zero
        if range_val < 1e-10:
            # At sensor position, use unit vector direction
            return np.array([[1.0, 0.0], [0.0, 1.0]])

        # Jacobian matrix
        H = np.array([
            [dx / range_val, dy / range_val],
            [-dy / (range_val ** 2), dx / (range_val ** 2)]
        ])
        return H

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).

        p(y | x) = N(y | h(x), R)
        where h(x) = [range, bearing]
        """
        from ..utils.distributions import log_gaussian_prob
        mean = self.observation_mean(x)
        return log_gaussian_prob(y, mean, self.R)

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters: returns [range, bearing]."""
        return self.observation_mean(x)

    def observe(self, x: np.ndarray) -> np.ndarray:
        """Observation function for kernel flow filters: returns [range, bearing]."""
        return self.observation_mean(x)

    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return self.Q

    # Batch methods for optimized particle filtering

    def state_transition_mean_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized state transition mean: F @ particles.T → transpose."""
        return (self.F @ particles.T).T

    def state_transition_cov_batch(self, particles: np.ndarray) -> np.ndarray:
        """Q is constant - return single matrix."""
        return self.Q

    def log_observation_prob_batch(self, observation: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Vectorized range-bearing log-prob for all particles."""
        # Relative positions
        dx = particles[:, 0] - self.sensor_pos[0]  # (N,)
        dy = particles[:, 1] - self.sensor_pos[1]  # (N,)

        # Range and bearing
        range_vals = np.sqrt(dx**2 + dy**2)  # (N,)
        bearing_vals = np.arctan2(dy, dx)  # (N,)

        # Means: (N, 2)
        means = np.column_stack([range_vals, bearing_vals])

        # diff: (N, 2)
        diff = observation - means

        # Cholesky of R (2x2 diagonal)
        L_R = np.linalg.cholesky(self.R)

        # Solve: y = L_R^{-1} @ diff.T → (2, N)
        y = np.linalg.solve(L_R, diff.T)

        # Mahalanobis: (N,)
        mahalanobis = np.sum(y**2, axis=0)

        logdet = 2 * np.sum(np.log(np.diag(L_R)))

        return -0.5 * (2 * np.log(2 * np.pi) + logdet + mahalanobis)

    # TensorFlow methods

    if TF_AVAILABLE:
        @tf.function
        def sample_state_transition_tf(self, x_tf: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of state transition sampling (vectorized).

            Args:
                x_tf: Current state (2,) or (N, 2) - single or batch
                seed: Random seed

            Returns:
                Next state (2,) or (N, 2) - same shape as input
            """
            # Determine input shape for noise generation
            input_shape = tf.shape(x_tf)
            
            # Sample noise: w ~ N(0, Q)
            L = tf.linalg.cholesky(self.Q_tf)
            z = tf.random.stateless_normal(input_shape, seed=seed)
            
            # Apply Cholesky factor to get correlated noise
            if len(x_tf.shape) == 1:
                # Single state (2,)
                w = tf.linalg.matvec(L, z)
                return tf.linalg.matvec(self.F_tf, x_tf) + w
            else:
                # Batch of states (N, 2)
                w = tf.linalg.matvec(L, z)
                return tf.linalg.matvec(self.F_tf, x_tf) + w

        @tf.function
        def log_observation_prob_tf(self, y_tf: tf.Tensor, x_tf: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of observation log-probability.

            Args:
                y_tf: Observation (2,) - [range, bearing]
                x_tf: State (2,) or (N, 2) - [x, y] (single or batch)

            Returns:
                Log probability (scalar or (N,))
            """
            if len(x_tf.shape) == 1:
                # Single state (2,)
                dx = x_tf[0] - self.sensor_pos_tf[0]
                dy = x_tf[1] - self.sensor_pos_tf[1]
                range_val = tf.sqrt(dx ** 2 + dy ** 2)
                bearing_val = tf.atan2(dy, dx)
                mean = tf.stack([range_val, bearing_val])

                diff = y_tf - mean

                sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R_tf)
                mahalanobis = tf.reduce_sum(diff * tf.linalg.solve(self.R_tf, diff))

                return -0.5 * (logdet + mahalanobis)
            else:
                # Batch of states (N, 2)
                dx = x_tf[:, 0] - self.sensor_pos_tf[0]
                dy = x_tf[:, 1] - self.sensor_pos_tf[1]
                range_val = tf.sqrt(dx ** 2 + dy ** 2)
                bearing_val = tf.atan2(dy, dx)
                mean = tf.stack([range_val, bearing_val], axis=1)  # (N, 2)
                
                diff = y_tf - mean  # (N, 2)
                
                sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R_tf)
                mahalanobis = tf.reduce_sum(
                    diff * tf.linalg.matvec(tf.linalg.inv(self.R_tf), diff),
                    axis=1
                )
                
                return -0.5 * (logdet + mahalanobis)  # (N,)

        @tf.function
        def sample_initial_state_batch_tf(self, n: int, seed: tf.Tensor) -> tf.Tensor:
            """
            Sample n initial states using TensorFlow.

            Args:
                n: Number of samples
                seed: Random seed

            Returns:
                Initial states (n, 2)
            """
            # Sample from N(mu_0, Sigma_0)
            L = tf.linalg.cholesky(self.Sigma_0_tf)
            z = tf.random.stateless_normal([n, 2], seed=seed)

            return self.mu_0_tf + tf.linalg.matvec(L, z, transpose_a=True)
