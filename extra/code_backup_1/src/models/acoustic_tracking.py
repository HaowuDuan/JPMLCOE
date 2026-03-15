"""Acoustic Tracking Model matching Li & Coates paper.

This model matches the acoustic tracking scenario in Section V of:
"Particle Filtering with Invertible Particle Flow" by Li & Coates.

Uses amplitude decay measurements from multiple sensors.
"""

import numpy as np
from typing import Optional, List
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


class AcousticTrackingModel(StateSpaceModel):
    """
    Amplitude-based Acoustic Source Tracking Model (Li & Coates Paper).

    State: [x, y, vx, vy]
        - (x, y): source position in 2D plane
        - (vx, vy): source velocity in 2D plane

    State evolution (nearly constant velocity):
        x_{t+1} = F @ x_t + w_t, w_t ~ N(0, Q)

    Observation (amplitude decay at each sensor):
        z_j = Ψ / (r_j^2 + d_0) + noise

    where:
        - Ψ: source intensity (10 in paper)
        - r_j = ||x - s_j||: distance to sensor j
        - s_j: position of sensor j
        - d_0: regularization parameter (0.1 in paper)
        - noise ~ N(0, sigma_w^2)

    Parameters from Li & Coates paper:
        - State dim: 4
        - Number of sensors: 4 (corner positions)
        - Measurement noise: sigma_w^2 = 0.01
        - Process noise: Q_filter (larger than ground truth)
    """

    def __init__(
        self,
        sensor_positions: Optional[list] = None,
        source_intensity: float = 10.0,
        regularization: float = 0.1,
        measurement_noise_std: float = 0.1,
        process_noise_cov: Optional[list] = None,
        dt: float = 1.0
    ):
        """
        Initialize Amplitude-based Acoustic Tracking Model.

        Args:
            sensor_positions: List of sensor positions [[s1_x, s1_y], ...],
                             default: 4 sensors at corners of 40x40m area
            source_intensity: Source intensity Ψ (paper uses 10)
            regularization: Regularization parameter d_0 (paper uses 0.1)
            measurement_noise_std: Measurement noise std (sigma_w)
            process_noise_cov: Process noise covariance Q (4x4),
                              default: Paper's filter covariance
            dt: Time step
        """
        self.dt = dt
        self.source_intensity = source_intensity
        self.regularization = regularization
        self.measurement_noise_std = measurement_noise_std

        # Default: 4 sensors at corners of 40m x 40m area
        if sensor_positions is None:
            self.sensor_positions = np.array([
                [0.0, 0.0],      # Bottom-left
                [40.0, 0.0],     # Bottom-right
                [0.0, 40.0],     # Top-left
                [40.0, 40.0]     # Top-right
            ])
        else:
            self.sensor_positions = np.array(sensor_positions)

        self.n_sensors = len(self.sensor_positions)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Process noise covariance - paper's filter covariance
        if process_noise_cov is None:
            # From Li & Coates paper Section V, Page 8
            # Q = (1/20) * [[1/3, 0, 0.5, 0], [0, 1/3, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]]
            self.Q = (1.0 / 20.0) * np.array([
                [1.0/3.0,  0.0,      0.5,  0.0],
                [0.0,      1.0/3.0,  0.0,  0.5],
                [0.5,      0.0,      1.0,  0.0],
                [0.0,      0.5,      0.0,  1.0]
            ])
        else:
            self.Q = np.array(process_noise_cov)

        # Observation noise covariance
        self.R = np.eye(self.n_sensors) * (measurement_noise_std ** 2)

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def obs_dim(self) -> int:
        return self.n_sensors

    @property
    def initial_state_mean(self) -> np.ndarray:
        """
        Initial state mean for Kalman filters.
        
        Returns center of sampling region [10, 30] x [10, 30] with zero velocity.
        This matches the expected value of the sampling distribution.
        """
        return np.array([20.0, 20.0, 0.0, 0.0])

    @property
    def initial_state_cov(self) -> np.ndarray:
        """
        Initial state covariance for Kalman filters.
        
        Position: uniform([10, 30]) has variance = (30-10)^2 / 12 ≈ 33.33
        Velocity: N(0, 1) has variance = 1.0
        """
        return np.diag([33.33, 33.33, 1.0, 1.0])

    # NumPy sampling methods

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from initial state distribution.

        Default: Uniform position in [10, 30] x [10, 30], velocity ~ N(0, 1)
        """
        x = rng.uniform(10, 30)
        y = rng.uniform(10, 30)
        vx = rng.normal(0, 1)
        vy = rng.normal(0, 1)
        return np.array([x, y, vx, vy])

    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = F @ x + w, w ~ N(0, Q)."""
        mean = self.F @ x
        noise = rng.multivariate_normal(np.zeros(4), self.Q)
        return mean + noise

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample observation: amplitude at each sensor with Gaussian noise."""
        amplitudes = np.zeros(self.n_sensors)

        for j in range(self.n_sensors):
            # Distance from source to sensor j
            dx = x[0] - self.sensor_positions[j, 0]
            dy = x[1] - self.sensor_positions[j, 1]
            r_squared = dx**2 + dy**2

            # Amplitude decay model: z_j = Ψ / (r^2 + d_0)
            true_amplitude = self.source_intensity / (r_squared + self.regularization)

            # Add Gaussian noise
            amplitudes[j] = true_amplitude + rng.normal(0, self.measurement_noise_std)

        return amplitudes

    # Deterministic methods (for Kalman/EKF/UKF)

    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = F @ x."""
        return self.F @ x

    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Cov[x' | x] = Q."""
        return self.Q

    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂f/∂x = F."""
        return self.F

    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[y | x] = h(x)."""
        amplitudes = np.zeros(self.n_sensors)

        for j in range(self.n_sensors):
            dx = x[0] - self.sensor_positions[j, 0]
            dy = x[1] - self.sensor_positions[j, 1]
            r_squared = dx**2 + dy**2
            amplitudes[j] = self.source_intensity / (r_squared + self.regularization)

        return amplitudes

    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Cov[y | x] = R."""
        return self.R

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function: ∂h/∂x.

        h_j(x) = Ψ / (r_j^2 + d_0)

        where r_j^2 = (x - s_j_x)^2 + (y - s_j_y)^2

        ∂h_j/∂x = -2 * Ψ * (x - s_j_x) / (r_j^2 + d_0)^2
        ∂h_j/∂y = -2 * Ψ * (y - s_j_y) / (r_j^2 + d_0)^2
        ∂h_j/∂vx = 0
        ∂h_j/∂vy = 0

        Returns:
            H: (n_sensors, 4) Jacobian matrix
        """
        H = np.zeros((self.n_sensors, 4))
        psi = self.source_intensity
        d0 = self.regularization

        for j in range(self.n_sensors):
            dx = x[0] - self.sensor_positions[j, 0]
            dy = x[1] - self.sensor_positions[j, 1]
            r_squared = dx**2 + dy**2

            # Gradient components: ∂(Ψ/(r^2+d0))/∂x = -2*Ψ*(x-sx)/(r^2+d0)^2
            denominator_squared = (r_squared + d0) ** 2
            H[j, 0] = -2.0 * psi * dx / denominator_squared  # ∂h_j/∂x
            H[j, 1] = -2.0 * psi * dy / denominator_squared  # ∂h_j/∂y
            H[j, 2] = 0.0  # ∂h_j/∂vx
            H[j, 3] = 0.0  # ∂h_j/∂vy

        return H

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).

        p(y | x) = N(y | h(x), R)
        """
        h_x = self.observation_mean(x)
        residual = y - h_x

        # Compute log probability using Cholesky for stability
        try:
            L = np.linalg.cholesky(self.R)
            z = np.linalg.solve(L, residual)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            mahalanobis = np.sum(z**2)
        except np.linalg.LinAlgError:
            # Fallback
            R_inv = np.linalg.pinv(self.R)
            log_det = np.log(np.linalg.det(2 * np.pi * self.R) + 1e-10)
            mahalanobis = residual.T @ R_inv @ residual
            return -0.5 * (log_det + mahalanobis)

        return -0.5 * (log_det + self.obs_dim * np.log(2 * np.pi) + mahalanobis)

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters."""
        return self.observation_mean(x)

    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return self.Q

    # TensorFlow methods for vectorized particle filter

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
            # Determine shape
            input_shape = tf.shape(x_tf)

            # Convert F and Q to tensors
            F_tf = tf.constant(self.F, dtype=tf.float32)
            Q_tf = tf.constant(self.Q, dtype=tf.float32)

            # Mean: F @ x
            if len(x_tf.shape) == 1:
                mean = F_tf @ x_tf
            else:
                mean = tf.linalg.matvec(F_tf, x_tf)

            # Noise: sample from N(0, Q)
            # Use Cholesky decomposition: Q = L @ L^T
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
                y_tf: Observation, shape (n_sensors,)
                x_tf: States, shape (N, 4) or (4,)

            Returns:
                Log probabilities, shape (N,) or scalar
            """
            sensor_positions = tf.constant(self.sensor_positions, dtype=tf.float32)
            R_tf = tf.constant(self.R, dtype=tf.float32)
            R_inv = tf.linalg.inv(R_tf)
            log_det_2pi_R = tf.math.log(tf.linalg.det(2.0 * np.pi * R_tf))

            psi = tf.constant(self.source_intensity, dtype=tf.float32)
            d0 = tf.constant(self.regularization, dtype=tf.float32)

            # Handle both single and batch inputs
            if len(x_tf.shape) == 1:
                # Single state (4,)
                amplitudes = []
                for j in range(self.n_sensors):
                    sensor_x = sensor_positions[j, 0]
                    sensor_y = sensor_positions[j, 1]
                    dx = x_tf[0] - sensor_x
                    dy = x_tf[1] - sensor_y
                    r_squared = dx**2 + dy**2
                    amp = psi / (r_squared + d0)
                    amplitudes.append(amp)

                h_x = tf.stack(amplitudes)
                residual = y_tf - h_x

                # Mahalanobis distance
                mahalanobis = tf.reduce_sum(residual * (R_inv @ residual))
                log_prob = -0.5 * (log_det_2pi_R + mahalanobis)
            else:
                # Batch of states (N, 4)
                amplitudes = []
                for j in range(self.n_sensors):
                    sensor_x = sensor_positions[j, 0]
                    sensor_y = sensor_positions[j, 1]
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
                log_prob = -0.5 * (log_det_2pi_R + mahalanobis)

            return log_prob

        @tf.function
        def sample_initial_state_batch_tf(self, n: int, seed: tf.Tensor) -> tf.Tensor:
            """
            Sample n initial states using TensorFlow.

            Args:
                n: Number of samples
                seed: Random seed

            Returns:
                Initial states (n, 4)
            """
            seed1, seed2 = tf.random.experimental.stateless_split(seed)

            # Position: uniform [10, 30] x [10, 30]
            xy = tf.random.stateless_uniform([n, 2], seed=seed1, minval=10.0, maxval=30.0)

            # Velocity: N(0, 1)
            v = tf.random.stateless_normal([n, 2], seed=seed2)

            return tf.concat([xy, v], axis=1)
