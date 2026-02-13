"""Range-Bearing tracking model."""

import tensorflow as tf
from typing import Optional
import numpy as np

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
        self.F = tf.constant(F, dtype=tf.float32)

        # Default process noise
        if Q is None:
            Q = 0.01 * np.eye(2)
        if Q.shape != (2, 2):
            raise ValueError(f"Q must be (2, 2), got {Q.shape}")
        self.Q = tf.constant(Q, dtype=tf.float32)

        # Sensor position
        sensor_pos = np.asarray(sensor_pos)
        if sensor_pos.shape != (2,):
            raise ValueError(f"sensor_pos must be (2,), got {sensor_pos.shape}")
        self.sensor_pos = tf.constant(sensor_pos, dtype=tf.float32)

        # Observation noise parameters
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing
        R = np.diag([sigma_range ** 2, sigma_bearing ** 2])
        self.R = tf.constant(R, dtype=tf.float32)

        # Initial state distribution
        mu_0_np = mu_0 if mu_0 is not None else np.array([1.0, 1.0])
        Sigma_0_np = Sigma_0 if Sigma_0 is not None else np.eye(2)

        if mu_0_np.shape != (2,):
            raise ValueError(f"mu_0 must be (2,), got {mu_0_np.shape}")
        if Sigma_0_np.shape != (2, 2):
            raise ValueError(f"Sigma_0 must be (2, 2), got {Sigma_0_np.shape}")

        self.mu_0 = tf.constant(mu_0_np, dtype=tf.float32)
        self.Sigma_0 = tf.constant(Sigma_0_np, dtype=tf.float32)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 2

    @tf.function
    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0)."""
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([2], seed=seed, dtype=tf.float32)
        return self.mu_0 + tf.linalg.matvec(L, z)

    @tf.function
    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample from state transition: x' = F·x + w, w ~ N(0, Q)."""
        L = tf.linalg.cholesky(self.Q)
        z = tf.random.stateless_normal([2], seed=seed, dtype=tf.float32)
        w = tf.linalg.matvec(L, z)
        return tf.linalg.matvec(self.F, x) + w

    @tf.function
    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample observation: [range, bearing] with additive noise.

        range = sqrt((x - x_sensor)² + (y - y_sensor)²) + v_range
        bearing = atan2(y - y_sensor, x - x_sensor) + v_bearing
        """
        # Relative position
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]

        # True range and bearing
        range_true = tf.sqrt(dx ** 2 + dy ** 2)
        bearing_true = tf.atan2(dy, dx)

        # Add noise
        noise = tf.random.stateless_normal([2], seed=seed, dtype=tf.float32)
        noise = noise * tf.constant([self.sigma_range, self.sigma_bearing], dtype=tf.float32)

        return tf.stack([range_true, bearing_true]) + noise

    @tf.function
    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of state transition: E[x' | x] = F·x."""
        return tf.linalg.matvec(self.F, x)

    @tf.function
    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of state transition: Cov[x' | x] = Q."""
        return self.Q

    @tf.function
    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of state transition: ∂(F·x)/∂x = F."""
        return self.F

    @tf.function
    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        """
        Mean of observation: E[y | x] = [range, bearing].

        range = sqrt((x - x_sensor)² + (y - y_sensor)²)
        bearing = atan2(y - y_sensor, x - x_sensor)
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        range_val = tf.sqrt(dx ** 2 + dy ** 2)
        bearing_val = tf.atan2(dy, dx)
        return tf.stack([range_val, bearing_val])

    @tf.function
    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of observation: Cov[y | x] = R (constant)."""
        return self.R

    @tf.function
    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
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
        range_val = tf.sqrt(dx ** 2 + dy ** 2)

        # Avoid division by zero
        range_val = tf.maximum(range_val, 1e-10)

        # Jacobian matrix
        H = tf.stack([
            [dx / range_val, dy / range_val],
            [-dy / (range_val ** 2), dx / (range_val ** 2)]
        ])
        return H

    @tf.function
    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Log probability of observation: log p(y | x).

        p(y | x) = N(y | h(x), R)
        where h(x) = [range, bearing]
        """
        mean = self.observation_mean(x)
        diff = y - mean

        # Compute log determinant
        sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R)

        # Compute Mahalanobis distance
        diff_col = tf.reshape(diff, [-1, 1])
        mahalanobis = tf.reduce_sum(diff * tf.squeeze(tf.linalg.solve(self.R, diff_col), axis=-1))

        return -0.5 * (logdet + mahalanobis)

    @tf.function
    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        """Observation function h(x) for flow filters: returns [range, bearing]."""
        return self.observation_mean(x)

    @tf.function
    def observe(self, x: tf.Tensor) -> tf.Tensor:
        """Observation function for kernel flow filters: returns [range, bearing]."""
        return self.observation_mean(x)

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        """Observation noise covariance R for flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> tf.Tensor:
        """Process noise covariance Q for flow filters."""
        return self.Q

    # Batch methods for optimized particle filtering

    @tf.function
    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized state transition mean: particles @ F.T."""
        return particles @ tf.transpose(self.F)

    @tf.function
    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Q is constant - return single matrix."""
        return self.Q

    @tf.function
    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized range-bearing log-prob for all particles."""
        # Relative positions
        dx = particles[:, 0] - self.sensor_pos[0]  # (N,)
        dy = particles[:, 1] - self.sensor_pos[1]  # (N,)

        # Range and bearing
        range_vals = tf.sqrt(dx**2 + dy**2)  # (N,)
        bearing_vals = tf.atan2(dy, dx)  # (N,)

        # Means: (N, 2)
        means = tf.stack([range_vals, bearing_vals], axis=1)

        # diff: (N, 2)
        diff = observation - means

        # Cholesky of R (2x2 diagonal)
        L_R = tf.linalg.cholesky(self.R)

        # Solve: y = L_R^{-1} @ diff.T → (2, N)
        y = tf.linalg.triangular_solve(L_R, tf.transpose(diff), lower=True)

        # Mahalanobis: (N,)
        mahalanobis = tf.reduce_sum(y**2, axis=0)

        logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_R)))

        return -0.5 * (2.0 * tf.math.log(2.0 * np.pi) + logdet + mahalanobis)

    @tf.function
    def sample_state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """
        Vectorized state transition sampling.

        Args:
            particles: Current states (N, 2)
            seed: Random seed

        Returns:
            Next states (N, 2)
        """
        # Sample noise: w ~ N(0, Q)
        N = tf.shape(particles)[0]
        L = tf.linalg.cholesky(self.Q)
        z = tf.random.stateless_normal([N, 2], seed=seed, dtype=tf.float32)

        # Apply Cholesky factor: w = L @ z.T → (2, N), then transpose
        w = tf.linalg.matmul(z, L, transpose_b=True)  # (N, 2)

        # Apply state transition: F @ particles.T → transpose
        return particles @ tf.transpose(self.F) + w

    @tf.function
    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample n initial states.

        Args:
            n: Number of samples
            seed: Random seed

        Returns:
            Initial states (n, 2)
        """
        # Sample from N(mu_0, Sigma_0)
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([n, 2], seed=seed, dtype=tf.float32)

        # Broadcast mu_0 and apply transformation
        return self.mu_0 + tf.linalg.matmul(z, L, transpose_b=True)
