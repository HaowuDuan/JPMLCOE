"""Two-Sensor Bearing-Only Static Estimation Model.

This model implements the numerical example from Section 4 of the particle flow paper,
designed to create a stiff problem due to highly anisotropic prior covariance.

Physical System:
    - Stationary target at position [x, y] in 2D plane
    - Two passive bearing-only sensors at fixed positions

State:
    x = [x, y]^T  (2D position)

State Evolution (Static):
    x_{t+1} = x_t  (no dynamics, stationary target)

Observations:
    z = [bearing_1, bearing_2]^T
    bearing_i = arctan2(y - y_{s,i}, x - x_{s,i}) + v_i

where:
    - (x_{s,i}, y_{s,i}): Position of sensor i
    - v_i ~ N(0, sigma_bearing^2): Bearing measurement noise

Default Parameters (from paper):
    - Prior mean: m_0 = [3.0, 5.0]^T
    - Prior covariance: P_0 = [[1000.0, 0], [0, 2.0]]  (highly anisotropic!)
    - Sensor positions: [(3.5, 0), (-3.5, 0)]
    - Measurement noise: R = [[0.04, 0], [0, 0.04]]  (sigma = 0.2 rad)
"""

import numpy as np
from typing import Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


class TwoSensorBearingOnlyModel(StateSpaceModel):
    """
    Two-Sensor Bearing-Only Static Estimation Model.

    Designed to test particle flow methods on stiff problems with
    ill-conditioned prior covariances (high anisotropy).

    State: 2D position [x, y]
    Observation: 2D bearing measurements [bearing_1, bearing_2]
    Dynamics: Static (identity transition)

    Parameters:
        mu_0: Initial state mean (2,), default [3.0, 5.0]
        Sigma_0: Initial state covariance (2, 2), default [[1000, 0], [0, 2.0]]
        sensor_positions: Sensor positions (2, 2), default [[3.5, 0], [-3.5, 0]]
        sigma_bearing: Bearing measurement noise std (radians), default 0.2
    """

    def __init__(
        self,
        mu_0: Optional[np.ndarray] = None,
        Sigma_0: Optional[np.ndarray] = None,
        sensor_positions: Optional[np.ndarray] = None,
        sigma_bearing: float = 0.2
    ):
        """
        Initialize Two-Sensor Bearing-Only Model.

        Args:
            mu_0: Initial state mean [x_0, y_0]. Default: [3.0, 5.0]
            Sigma_0: Initial state covariance (2, 2). Default: [[1000, 0], [0, 2.0]]
            sensor_positions: Sensor positions (2, 2), [[x1, y1], [x2, y2]].
                            Default: [[3.5, 0.0], [-3.5, 0.0]]
            sigma_bearing: Bearing measurement noise std (radians). Default: 0.2
        """
        if sigma_bearing <= 0:
            raise ValueError(f"sigma_bearing must be positive, got {sigma_bearing}")

        # Default initial state (from paper)
        self.mu_0 = mu_0 if mu_0 is not None else np.array([3.0, 5.0])

        # Default initial covariance (highly anisotropic - this creates stiffness!)
        self.Sigma_0 = Sigma_0 if Sigma_0 is not None else np.array([
            [1000.0, 0.0],
            [0.0, 2.0]
        ])

        # Default sensor positions (on x-axis, symmetric)
        self.sensor_positions = sensor_positions if sensor_positions is not None else np.array([
            [3.5, 0.0],   # Sensor 1
            [-3.5, 0.0]   # Sensor 2
        ])

        # Validate shapes
        if self.mu_0.shape != (2,):
            raise ValueError(f"mu_0 must be (2,), got {self.mu_0.shape}")
        if self.Sigma_0.shape != (2, 2):
            raise ValueError(f"Sigma_0 must be (2, 2), got {self.Sigma_0.shape}")
        if self.sensor_positions.shape != (2, 2):
            raise ValueError(f"sensor_positions must be (2, 2), got {self.sensor_positions.shape}")

        # Observation noise parameters
        self.sigma_bearing = sigma_bearing
        self.R = np.diag([sigma_bearing ** 2, sigma_bearing ** 2])

        # State transition is identity (static target)
        self.F = np.eye(2)
        self.Q = np.zeros((2, 2))  # No process noise (static)

        # TensorFlow constants
        if TF_AVAILABLE:
            self.mu_0_tf = tf.constant(self.mu_0, dtype=tf.float32)
            self.Sigma_0_tf = tf.constant(self.Sigma_0, dtype=tf.float32)
            self.sensor_positions_tf = tf.constant(self.sensor_positions, dtype=tf.float32)
            self.R_tf = tf.constant(self.R, dtype=tf.float32)
            self.F_tf = tf.constant(self.F, dtype=tf.float32)
            self.Q_tf = tf.constant(self.Q, dtype=tf.float32)

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
        """
        Sample from state transition (static target).

        For static estimation: x' = x (no dynamics, no noise)
        """
        return x.copy()

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample observation: [bearing_1, bearing_2] with additive noise.

        bearing_i = arctan2(y - y_{s,i}, x - x_{s,i}) + v_i
        where v_i ~ N(0, sigma_bearing^2)
        """
        bearings = np.zeros(2)

        for i in range(2):
            # Relative position to sensor i
            dx = x[0] - self.sensor_positions[i, 0]
            dy = x[1] - self.sensor_positions[i, 1]

            # True bearing
            bearing_true = np.arctan2(dy, dx)

            # Add noise
            v_i = rng.normal(0, self.sigma_bearing)
            bearings[i] = bearing_true + v_i

        return bearings

    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = x (static)."""
        return x.copy()

    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Cov[x' | x] = 0 (no noise, static)."""
        return self.Q

    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂f/∂x = I (identity)."""
        return self.F

    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Mean of observation: E[y | x] = [bearing_1, bearing_2].

        bearing_i = arctan2(y - y_{s,i}, x - x_{s,i})
        """
        bearings = np.zeros(2)

        for i in range(2):
            dx = x[0] - self.sensor_positions[i, 0]
            dy = x[1] - self.sensor_positions[i, 1]
            bearings[i] = np.arctan2(dy, dx)

        return bearings

    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Cov[y | x] = R (constant)."""
        return self.R

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation: ∂h/∂x where h(x) = [bearing_1, bearing_2].

        For bearing_i = arctan2(y - y_{s,i}, x - x_{s,i}):
            ∂bearing_i/∂x = -(y - y_{s,i}) / r_i^2
            ∂bearing_i/∂y = (x - x_{s,i}) / r_i^2

        where r_i^2 = (x - x_{s,i})^2 + (y - y_{s,i})^2

        Returns:
            H: Jacobian matrix (2, 2)
                [[∂bearing_1/∂x, ∂bearing_1/∂y],
                 [∂bearing_2/∂x, ∂bearing_2/∂y]]
        """
        H = np.zeros((2, 2))

        for i in range(2):
            dx = x[0] - self.sensor_positions[i, 0]
            dy = x[1] - self.sensor_positions[i, 1]
            r_squared = dx ** 2 + dy ** 2

            # Avoid division by zero
            if r_squared < 1e-10:
                # At sensor position, use unit direction
                H[i, 0] = 0.0
                H[i, 1] = 1.0
            else:
                H[i, 0] = -dy / r_squared  # ∂bearing_i/∂x
                H[i, 1] = dx / r_squared   # ∂bearing_i/∂y

        return H

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).

        p(y | x) = N(y | h(x), R)
        where h(x) = [bearing_1, bearing_2]
        """
        from ..utils.distributions import log_gaussian_prob
        mean = self.observation_mean(x)

        # Handle bearing wrapping for angular differences
        diff = y - mean
        # Wrap to [-π, π]
        diff = np.arctan2(np.sin(diff), np.cos(diff))

        # Manual computation with wrapped differences
        log_det = np.log(np.linalg.det(2 * np.pi * self.R))
        R_inv = np.linalg.inv(self.R)
        mahalanobis = diff.T @ R_inv @ diff

        return -0.5 * (log_det + mahalanobis)

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters: returns [bearing_1, bearing_2]."""
        return self.observation_mean(x)

    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters (zero for static)."""
        return self.Q

    # Batch methods for optimized particle filtering

    def state_transition_mean_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized state transition mean: F @ particles.T → transpose (identity for static)."""
        return (self.F @ particles.T).T

    def state_transition_cov_batch(self, particles: np.ndarray) -> np.ndarray:
        """Q is constant - return single matrix."""
        return self.Q

    def log_observation_prob_batch(self, observation: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Vectorized two-sensor bearing log-prob for all particles."""
        # Relative positions to sensor 1
        dx1 = particles[:, 0] - self.sensor1_pos[0]  # (N,)
        dy1 = particles[:, 1] - self.sensor1_pos[1]  # (N,)

        # Relative positions to sensor 2
        dx2 = particles[:, 0] - self.sensor2_pos[0]  # (N,)
        dy2 = particles[:, 1] - self.sensor2_pos[1]  # (N,)

        # Bearings from both sensors
        bearing1 = np.arctan2(dy1, dx1)  # (N,)
        bearing2 = np.arctan2(dy2, dx2)  # (N,)

        # Means: (N, 2)
        means = np.column_stack([bearing1, bearing2])

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
            TensorFlow version of state transition sampling (static).

            Args:
                x_tf: Current state (2,) or (N, 2)
                seed: Random seed (unused for static model)

            Returns:
                Next state (same as current, static target)
            """
            return tf.identity(x_tf)

        @tf.function
        def log_observation_prob_tf(self, y_tf: tf.Tensor, x_tf: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of observation log-probability.

            Args:
                y_tf: Observation (2,) - [bearing_1, bearing_2]
                x_tf: State (2,) or (N, 2) - [x, y]

            Returns:
                Log probability (scalar or (N,))
            """
            if len(x_tf.shape) == 1:
                # Single state
                bearings = []
                for i in range(2):
                    dx = x_tf[0] - self.sensor_positions_tf[i, 0]
                    dy = x_tf[1] - self.sensor_positions_tf[i, 1]
                    bearing = tf.atan2(dy, dx)
                    bearings.append(bearing)
                h_x = tf.stack(bearings)

                # Wrapped difference
                diff = y_tf - h_x
                diff = tf.atan2(tf.sin(diff), tf.cos(diff))

                # Log probability
                sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R_tf)
                mahalanobis = tf.reduce_sum(diff * tf.linalg.solve(self.R_tf, diff))

                return -0.5 * (logdet + mahalanobis)
            else:
                # Batch of states (N, 2)
                N = tf.shape(x_tf)[0]
                bearings1 = tf.atan2(
                    x_tf[:, 1] - self.sensor_positions_tf[0, 1],
                    x_tf[:, 0] - self.sensor_positions_tf[0, 0]
                )
                bearings2 = tf.atan2(
                    x_tf[:, 1] - self.sensor_positions_tf[1, 1],
                    x_tf[:, 0] - self.sensor_positions_tf[1, 0]
                )
                h_x = tf.stack([bearings1, bearings2], axis=1)  # (N, 2)

                # Wrapped difference
                diff = y_tf - h_x
                diff = tf.atan2(tf.sin(diff), tf.cos(diff))

                # Log probability
                sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R_tf)
                mahalanobis = tf.reduce_sum(
                    diff * tf.linalg.matvec(tf.linalg.inv(self.R_tf), diff),
                    axis=1
                )

                return -0.5 * (logdet + mahalanobis)

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
