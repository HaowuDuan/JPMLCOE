"""Single-Target Acoustic Tracking Model with 25 Sensors for Paper Reproduction.

This model implements a paper-inspired single-target acoustic tracking scenario
based on Li & Coates (2017), but simplified to 1 target instead of 4.

Key features:
- 1 target with 4-dimensional state
- 25 sensors in 5×5 grid (same as full model)
- Distinction between V_true (data generation) and V_filter (algorithm use)
- Paper's exact parameters but single-target
"""

import numpy as np
from typing import Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


class AcousticTrackingLiteModel(StateSpaceModel):
    """
    Single-Target Amplitude-based Acoustic Tracking Model (Paper-Inspired).

    State: [x, y, vx, vy]
        - (x, y): target position in 2D plane
        - (vx, vy): target velocity in 2D plane

    State evolution (constant velocity model):
        x_{t+1} = F @ x_t + w_t, w_t ~ N(0, V)

    Observation (amplitude decay at each sensor):
        z_s = Ψ / (r_s^2 + d_0) + noise
        where r_s = ||target - sensor_s||

    Key difference from base acoustic_tracking.py:
        - 25 sensors in 5×5 grid (vs 4 sensors at corners)
        - V_true vs V_filter distinction
        - Uses paper's Target 1 initial state

    Parameters from Li & Coates paper Section V:
        - State dim: 4 (single target)
        - Number of sensors: 25 (5×5 grid)
        - Measurement noise: sigma_w^2 = 0.01
        - V_true (data generation): (1/20) * [[1/3,0,0.5,0], [0,1/3,0,0.5], [0.5,0,1,0], [0,0.5,0,1]]
        - V_filter (algorithms): [[3,0,0.1,0], [0,3,0,0.1], [0.1,0,0.03,0], [0,0.1,0,0.03]]
    """

    def __init__(
        self,
        sensor_grid_size: int = 5,
        source_intensity: float = 10.0,
        regularization: float = 0.1,
        measurement_noise_std: float = 0.1,
        use_true_process_noise: bool = False,
        dt: float = 1.0
    ):
        """
        Initialize Single-Target Acoustic Tracking Model with 25 Sensors.

        Args:
            sensor_grid_size: Size of sensor grid (5 → 5×5 = 25 sensors)
            source_intensity: Source intensity Ψ (paper uses 10)
            regularization: Regularization parameter d_0 (paper uses 0.1)
            measurement_noise_std: Measurement noise std (sigma_w = 0.1)
            use_true_process_noise: If True, use V_true; if False, use V_filter
            dt: Time step
        """
        self.dt = dt
        self.source_intensity = source_intensity
        self.regularization = regularization
        self.measurement_noise_std = measurement_noise_std

        # Build 5×5 sensor grid spanning [0, 40] × [0, 40]
        self.sensor_grid_size = sensor_grid_size
        self.sensor_positions = self._build_sensor_grid()
        self.n_sensors = len(self.sensor_positions)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # V_true: Paper's true process noise for data generation
        self.V_true = (1.0 / 20.0) * np.array([
            [1.0/3.0,  0.0,      0.5,  0.0],
            [0.0,      1.0/3.0,  0.0,  0.5],
            [0.5,      0.0,      1.0,  0.0],
            [0.0,      0.5,      0.0,  1.0]
        ])

        # V_filter: Paper's filter process noise (larger uncertainty)
        self.V_filter = np.array([
            [3.0,  0.0,   0.1,  0.0],
            [0.0,  3.0,   0.0,  0.1],
            [0.1,  0.0,   0.03, 0.0],
            [0.0,  0.1,   0.0,  0.03]
        ])

        # Q is either V_true or V_filter depending on use case
        self.Q = self.V_true if use_true_process_noise else self.V_filter

        # Observation noise covariance (25×25)
        self.R = np.eye(self.n_sensors) * (measurement_noise_std ** 2)

        # Paper's Target 1 initial state (page 8, Section V-A1)
        self.paper_initial_state = np.array([12.0, 6.0, 0.001, 0.001])

    def _build_sensor_grid(self) -> np.ndarray:
        """Build 5×5 sensor grid at intersections of 10m spacing."""
        sensors = []
        spacing = 40.0 / (self.sensor_grid_size - 1)
        for i in range(self.sensor_grid_size):
            for j in range(self.sensor_grid_size):
                x = i * spacing
                y = j * spacing
                sensors.append([x, y])
        return np.array(sensors)

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def obs_dim(self) -> int:
        return self.n_sensors  # 25

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
        """Vectorized amplitude decay log-prob for all particles."""
        # Positions: (N, 2) - extract x, y from [x, y, vx, vy]
        pos = particles[:, :2]  # (N, 2)

        # Distances from all particles to all sensors
        # pos: (N, 2), sensor_positions: (M, 2)
        # Broadcasting: (N, 1, 2) - (1, M, 2) → (N, M, 2)
        diff = pos[:, np.newaxis, :] - self.sensor_positions[np.newaxis, :, :]
        r_squared = np.sum(diff**2, axis=2)  # (N, M)

        # Amplitudes: (N, M)
        amplitudes = self.source_intensity / (r_squared + self.regularization)

        # diff from observation: (N, M)
        obs_diff = observation - amplitudes

        # Assuming R is diagonal: sigma^2 * I
        variance = self.measurement_noise_std ** 2

        # Mahalanobis: sum((diff/sigma)^2) → (N,)
        mahalanobis = np.sum(obs_diff**2, axis=1) / variance

        # Log determinant for diagonal R
        logdet = self.n_sensors * np.log(2 * np.pi * variance)

        return -0.5 * (logdet + mahalanobis)

    @property
    def initial_state_mean(self) -> np.ndarray:
        """Initial state mean for TFP and Kalman filters (Target 1 from paper)."""
        return self.paper_initial_state.copy()

    @property
    def initial_state_cov(self) -> np.ndarray:
        """
        Initial state covariance for TFP and Kalman filters.
        
        Paper specification: σ = 10 for positions, σ = 1 for velocities.
        """
        return np.diag([100.0, 100.0, 1.0, 1.0])  # σ² = [10², 10², 1², 1²]

    # NumPy sampling methods

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample initial state.
        
        For paper reproduction, returns paper's Target 1 initial state.
        """
        return self.paper_initial_state.copy()

    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = F @ x + w, w ~ N(0, Q)."""
        mean = self.F @ x
        noise = rng.multivariate_normal(np.zeros(4), self.Q)
        return mean + noise

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample observation: amplitude at each sensor with Gaussian noise."""
        amplitudes = np.zeros(self.n_sensors)

        for s in range(self.n_sensors):
            # Distance from source to sensor s
            dx = x[0] - self.sensor_positions[s, 0]
            dy = x[1] - self.sensor_positions[s, 1]
            r_squared = dx**2 + dy**2

            # Amplitude decay model: z = Ψ / (r^2 + d_0)
            amplitudes[s] = self.source_intensity / (r_squared + self.regularization)

        # Add Gaussian noise
        noise = rng.normal(0, self.measurement_noise_std, self.n_sensors)
        return amplitudes + noise

    # Deterministic methods (for Kalman/EKF/UKF)

    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = F @ x."""
        return self.F @ x

    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition (constant for linear model)."""
        return self.Q

    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition (constant F matrix for linear model)."""
        return self.F

    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[y | x] = h(x)."""
        return self.observe(x)

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters."""
        return self.observation_mean(x)

    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation (constant R matrix)."""
        return self.R

    def observe(self, x: np.ndarray) -> np.ndarray:
        """
        Deterministic observation function (no noise).
        
        Returns the mean observation for state x.
        """
        amplitudes = np.zeros(self.n_sensors)

        for s in range(self.n_sensors):
            dx = x[0] - self.sensor_positions[s, 0]
            dy = x[1] - self.sensor_positions[s, 1]
            r_squared = dx**2 + dy**2
            amplitudes[s] = self.source_intensity / (r_squared + self.regularization)

        return amplitudes

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of observation function: H = ∂h/∂x.

        For each sensor s:
        ∂z_s/∂x = -Ψ * 2 * (x - s_x) / (r^2 + d_0)^2
        ∂z_s/∂y = -Ψ * 2 * (y - s_y) / (r^2 + d_0)^2
        ∂z_s/∂vx = 0
        ∂z_s/∂vy = 0

        Returns:
            H: Jacobian matrix of shape (n_sensors, 4) = (25, 4)
        """
        H = np.zeros((self.n_sensors, 4))

        for s in range(self.n_sensors):
            dx = x[0] - self.sensor_positions[s, 0]
            dy = x[1] - self.sensor_positions[s, 1]
            r_squared = dx**2 + dy**2
            denominator = (r_squared + self.regularization) ** 2

            # Partial derivatives
            H[s, 0] = -self.source_intensity * 2 * dx / denominator  # ∂z_s/∂x
            H[s, 1] = -self.source_intensity * 2 * dy / denominator  # ∂z_s/∂y
            # H[s, 2] = 0  # ∂z_s/∂vx (already initialized to 0)
            # H[s, 3] = 0  # ∂z_s/∂vy (already initialized to 0)

        return H

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Compute log p(y | x) = log N(y; h(x), R).

        Args:
            y: Observation (n_sensors,)
            x: State (4,)

        Returns:
            Log probability
        """
        h_x = self.observe(x)
        residual = y - h_x
        
        # Log determinant of 2πR
        sign, logdet = np.linalg.slogdet(2.0 * np.pi * self.R)
        
        # Mahalanobis distance
        R_inv = np.linalg.inv(self.R)
        mahalanobis = residual.T @ R_inv @ residual
        
        return -0.5 * (logdet + mahalanobis)

    # TensorFlow methods (for TensorFlow particle filter)

    if TF_AVAILABLE:
        @tf.function
        def sample_state_transition_tf(self, x_tf: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of state transition sampling (vectorized).

            Args:
                x_tf: Current states, shape (N, 4) or (4,)
                seed: Random seed

            Returns:
                Next states, same shape as input
            """
            input_shape = tf.shape(x_tf)
            
            F_tf = tf.constant(self.F, dtype=tf.float32)
            Q_tf = tf.constant(self.Q, dtype=tf.float32)

            # Mean: F @ x
            if len(x_tf.shape) == 1:
                mean = F_tf @ x_tf
            else:
                mean = tf.linalg.matvec(F_tf, x_tf)

            # Noise: sample from N(0, Q)
            L = tf.linalg.cholesky(Q_tf)
            z = tf.random.stateless_normal(input_shape, seed=seed)
            if len(x_tf.shape) == 1:
                noise = L @ z
            else:
                noise = tf.linalg.matvec(L, z)

            return mean + noise

        @tf.function
        def log_observation_prob_tf(self, y_tf: tf.Tensor, x_tf: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of observation log-probability (vectorized).

            Args:
                y_tf: Observation, shape (n_sensors,) = (25,)
                x_tf: States, shape (N, 4) or (4,)

            Returns:
                Log probabilities, shape (N,) or scalar
            """
            sensor_positions = tf.constant(self.sensor_positions, dtype=tf.float32)
            R_tf = tf.constant(self.R, dtype=tf.float32)
            R_inv = tf.linalg.inv(R_tf)
            sign, logdet = tf.linalg.slogdet(2.0 * np.pi * R_tf)

            psi = tf.constant(self.source_intensity, dtype=tf.float32)
            d0 = tf.constant(self.regularization, dtype=tf.float32)

            # Handle both single and batch inputs
            if len(x_tf.shape) == 1:
                # Single state (4,)
                amplitudes = []
                for s in range(self.n_sensors):
                    sensor_x = sensor_positions[s, 0]
                    sensor_y = sensor_positions[s, 1]
                    dx = x_tf[0] - sensor_x
                    dy = x_tf[1] - sensor_y
                    r_squared = dx**2 + dy**2
                    amp = psi / (r_squared + d0)
                    amplitudes.append(amp)

                h_x = tf.stack(amplitudes)
                residual = y_tf - h_x

                # Mahalanobis distance
                mahalanobis = tf.reduce_sum(residual * (R_inv @ residual))
                log_prob = -0.5 * (logdet + mahalanobis)
            else:
                # Batch of states (N, 4)
                amplitudes = []
                for s in range(self.n_sensors):
                    sensor_x = sensor_positions[s, 0]
                    sensor_y = sensor_positions[s, 1]
                    dx = x_tf[:, 0] - sensor_x
                    dy = x_tf[:, 1] - sensor_y
                    r_squared = dx**2 + dy**2
                    amp = psi / (r_squared + d0)
                    amplitudes.append(amp)

                h_x = tf.stack(amplitudes, axis=1)  # (N, n_sensors)
                residual = y_tf - h_x  # (N, n_sensors)

                # Mahalanobis distance for each particle
                mahalanobis = tf.reduce_sum(
                    residual * tf.linalg.matvec(R_inv, residual),
                    axis=1
                )
                log_prob = -0.5 * (logdet + mahalanobis)

            return log_prob

        @tf.function
        def sample_initial_state_tf(self, seed: tf.Tensor, n_samples: int = 1) -> tf.Tensor:
            """
            TensorFlow version of initial state sampling.

            Args:
                seed: Random seed
                n_samples: Number of samples

            Returns:
                Initial states, shape (n_samples, 4) or (4,) if n_samples=1
            """
            initial_state = tf.constant(self.paper_initial_state, dtype=tf.float32)
            
            if n_samples == 1:
                return initial_state
            else:
                return tf.tile(tf.expand_dims(initial_state, 0), [n_samples, 1])

