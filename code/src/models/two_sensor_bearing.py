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

import tensorflow as tf
import numpy as np
from typing import Optional

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
        sigma_bearing: float = 0.2,
        dtype=tf.float32
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
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        if sigma_bearing <= 0:
            raise ValueError(f"sigma_bearing must be positive, got {sigma_bearing}")

        # Default initial state (from paper)
        mu_0_np = mu_0 if mu_0 is not None else np.array([3.0, 5.0])

        # Default initial covariance (highly anisotropic - this creates stiffness!)
        Sigma_0_np = Sigma_0 if Sigma_0 is not None else np.array([
            [1000.0, 0.0],
            [0.0, 2.0]
        ])

        # Default sensor positions (on x-axis, symmetric)
        sensor_positions_np = sensor_positions if sensor_positions is not None else np.array([
            [3.5, 0.0],   # Sensor 1
            [-3.5, 0.0]   # Sensor 2
        ])

        # Validate shapes
        if mu_0_np.shape != (2,):
            raise ValueError(f"mu_0 must be (2,), got {mu_0_np.shape}")
        if Sigma_0_np.shape != (2, 2):
            raise ValueError(f"Sigma_0 must be (2, 2), got {Sigma_0_np.shape}")
        if sensor_positions_np.shape != (2, 2):
            raise ValueError(f"sensor_positions must be (2, 2), got {sensor_positions_np.shape}")

        # Convert to TensorFlow
        self._mu_0 = tf.constant(mu_0_np, dtype=self.dtype)
        self._Sigma_0 = tf.constant(Sigma_0_np, dtype=self.dtype)
        self.sensor_positions = tf.constant(sensor_positions_np, dtype=self.dtype)

        # Observation noise parameters
        self.sigma_bearing = sigma_bearing
        R_np = np.diag([sigma_bearing ** 2, sigma_bearing ** 2])
        self.R = tf.constant(R_np, dtype=self.dtype)

        # State transition is identity (static target)
        F_np = np.eye(2)
        self.F = tf.constant(F_np, dtype=self.dtype)

        Q_np = np.zeros((2, 2))  # No process noise (static)
        self.Q = tf.constant(Q_np, dtype=self.dtype)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 2

    @property
    def mu_0(self) -> tf.Tensor:
        return self._mu_0

    @property
    def Sigma_0(self) -> tf.Tensor:
        return self._Sigma_0

    @tf.function
    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0)."""
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([2], seed=seed, dtype=self.dtype)
        return self.mu_0 + tf.linalg.matvec(L, z)

    @tf.function
    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample from state transition (static target): x' = x."""
        return tf.identity(x)

    @tf.function
    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample observation: [bearing_1, bearing_2] with additive noise.

        bearing_i = arctan2(y - y_{s,i}, x - x_{s,i}) + v_i
        where v_i ~ N(0, sigma_bearing^2)
        """
        bearings = tf.TensorArray(dtype=self.dtype, size=2)

        for i in tf.range(2):
            # Relative position to sensor i
            dx = x[0] - self.sensor_positions[i, 0]
            dy = x[1] - self.sensor_positions[i, 1]

            # True bearing
            bearing_true = tf.atan2(dy, dx)

            bearings = bearings.write(i, bearing_true)

        bearings_stack = bearings.stack()

        # Add noise
        noise = tf.random.stateless_normal([2], seed=seed, dtype=self.dtype)
        noise = noise * self.sigma_bearing

        return bearings_stack + noise

    @tf.function
    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of state transition: E[x' | x] = x (static)."""
        return tf.identity(x)

    @tf.function
    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of state transition: Cov[x' | x] = 0 (no noise, static)."""
        return self.Q

    @tf.function
    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of state transition: ∂f/∂x = I (identity)."""
        return self.F

    @tf.function
    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        """
        Mean of observation: E[y | x] = [bearing_1, bearing_2].

        bearing_i = arctan2(y - y_{s,i}, x - x_{s,i})
        """
        bearings = tf.TensorArray(dtype=self.dtype, size=2)

        for i in tf.range(2):
            dx = x[0] - self.sensor_positions[i, 0]
            dy = x[1] - self.sensor_positions[i, 1]
            bearing = tf.atan2(dy, dx)
            bearings = bearings.write(i, bearing)

        return bearings.stack()

    @tf.function
    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of observation: Cov[y | x] = R (constant)."""
        return self.R

    @tf.function
    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
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
        H = tf.TensorArray(dtype=self.dtype, size=2)

        for i in tf.range(2):
            dx = x[0] - self.sensor_positions[i, 0]
            dy = x[1] - self.sensor_positions[i, 1]
            r_squared = dx ** 2 + dy ** 2

            # Avoid division by zero
            r_squared = tf.maximum(r_squared, 1e-10)

            row = tf.stack([
                -dy / r_squared,  # ∂bearing_i/∂x
                dx / r_squared    # ∂bearing_i/∂y
            ])
            H = H.write(i, row)

        return H.stack()

    @tf.function
    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Log probability of observation: log p(y | x).

        p(y | x) = N(y | h(x), R)
        where h(x) = [bearing_1, bearing_2]
        """
        mean = self.observation_mean(x)

        # Handle bearing wrapping for angular differences
        diff = y - mean
        # Wrap to [-π, π]
        diff = tf.atan2(tf.sin(diff), tf.cos(diff))

        # Log probability computation
        sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R)
        diff_col = tf.reshape(diff, [-1, 1])
        mahalanobis = tf.reduce_sum(diff * tf.squeeze(tf.linalg.solve(self.R, diff_col), axis=-1))

        return -0.5 * (logdet + mahalanobis)

    @tf.function
    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        """Observation function h(x) for flow filters: returns [bearing_1, bearing_2]."""
        return self.observation_mean(x)

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        """Observation noise covariance R for flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> tf.Tensor:
        """Process noise covariance Q for flow filters (zero for static)."""
        return self.Q

    # Batch methods for optimized particle filtering

    @tf.function
    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized Jacobian for two-sensor bearing: (N, 2, 2)."""
        # Sensor 1
        dx1 = particles[:, 0] - self.sensor_positions[0, 0]
        dy1 = particles[:, 1] - self.sensor_positions[0, 1]
        r1_sq = tf.maximum(dx1**2 + dy1**2, 1e-10)

        # Sensor 2
        dx2 = particles[:, 0] - self.sensor_positions[1, 0]
        dy2 = particles[:, 1] - self.sensor_positions[1, 1]
        r2_sq = tf.maximum(dx2**2 + dy2**2, 1e-10)

        # H[i] = [[-dy1/r1^2, dx1/r1^2], [-dy2/r2^2, dx2/r2^2]]
        row0 = tf.stack([-dy1 / r1_sq, dx1 / r1_sq], axis=1)  # (N, 2)
        row1 = tf.stack([-dy2 / r2_sq, dx2 / r2_sq], axis=1)  # (N, 2)
        return tf.stack([row0, row1], axis=1)  # (N, 2, 2)

    @tf.function
    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized h(x) = [bearing_1, bearing_2]: (N, 2)."""
        dx1 = particles[:, 0] - self.sensor_positions[0, 0]
        dy1 = particles[:, 1] - self.sensor_positions[0, 1]
        dx2 = particles[:, 0] - self.sensor_positions[1, 0]
        dy2 = particles[:, 1] - self.sensor_positions[1, 1]
        bearing1 = tf.atan2(dy1, dx1)
        bearing2 = tf.atan2(dy2, dx2)
        return tf.stack([bearing1, bearing2], axis=1)  # (N, 2)

    @tf.function
    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """F is constant (identity) — broadcast to (N, 2, 2)."""
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self.F, 0), [N, 1, 1])

    @tf.function
    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized state transition mean: identity for static target."""
        return tf.identity(particles)

    @tf.function
    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Q is constant - return single matrix."""
        return self.Q

    @tf.function
    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized two-sensor bearing log-prob for all particles."""
        # Relative positions to sensor 1
        dx1 = particles[:, 0] - self.sensor_positions[0, 0]  # (N,)
        dy1 = particles[:, 1] - self.sensor_positions[0, 1]  # (N,)

        # Relative positions to sensor 2
        dx2 = particles[:, 0] - self.sensor_positions[1, 0]  # (N,)
        dy2 = particles[:, 1] - self.sensor_positions[1, 1]  # (N,)

        # Bearings from both sensors
        bearing1 = tf.atan2(dy1, dx1)  # (N,)
        bearing2 = tf.atan2(dy2, dx2)  # (N,)

        # Means: (N, 2)
        means = tf.stack([bearing1, bearing2], axis=1)

        # diff: (N, 2) — wrap to [-pi, pi] for angular differences
        diff = observation - means
        diff = tf.atan2(tf.sin(diff), tf.cos(diff))

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
        Vectorized state transition sampling (static).

        Args:
            particles: Current states (N, 2)
            seed: Random seed (unused for static model)

        Returns:
            Next states (N, 2) - same as input for static target
        """
        return tf.identity(particles)

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
        z = tf.random.stateless_normal([n, 2], seed=seed, dtype=self.dtype)

        return self.mu_0 + tf.linalg.matmul(z, L, transpose_b=True)
