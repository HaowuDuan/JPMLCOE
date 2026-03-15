"""Multi-Target Acoustic Tracking Model for Paper Reproduction.

This model implements the full multi-target acoustic tracking scenario from Section V of:
"Particle Filtering with Invertible Particle Flow" by Li & Coates (2017).

Key features:
- 4 independent targets with 16-dimensional joint state
- 25 sensors in 5×5 grid
- Additive observations (each sensor measures sum of all target amplitudes)
- Distinction between V_true (data generation) and V_filter (algorithm use)
"""

import numpy as np
from typing import Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


class AcousticTrackingFullModel(StateSpaceModel):
    """
    Multi-Target Amplitude-based Acoustic Tracking Model (Li & Coates Paper).

    State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3, x4, y4, vx4, vy4]
        - (xi, yi): position of target i in 2D plane
        - (vxi, vyi): velocity of target i in 2D plane

    State evolution (constant velocity model for each target):
        x_{t+1} = F @ x_t + w_t, w_t ~ N(0, V)
        where F is block diagonal with 4 identical blocks

    Observation (additive amplitude decay at each sensor):
        z_s = Σ(i=1..4) Ψ / (r_s,i^2 + d_0) + noise
        where r_s,i = ||target_i - sensor_s||

    Parameters from Li & Coates paper Section V:
        - State dim: 16 (4 targets × 4 states each)
        - Number of sensors: 25 (5×5 grid)
        - Measurement noise: sigma_w^2 = 0.01
        - V_true (data generation): (1/20) * [[1/3,0,0.5,0], [0,1/3,0,0.5], [0.5,0,1,0], [0,0.5,0,1]]
        - V_filter (algorithms): [[3,0,0.1,0], [0,3,0,0.1], [0.1,0,0.03,0], [0,0.1,0,0.03]]
    """

    def __init__(
        self,
        n_targets: int = 4,
        sensor_grid_size: int = 5,
        source_intensity: float = 10.0,
        regularization: float = 0.1,
        measurement_noise_std: float = 0.1,
        use_true_process_noise: bool = False,
        dt: float = 1.0
    ):
        """
        Initialize Multi-Target Acoustic Tracking Model.

        Args:
            n_targets: Number of targets (default: 4 per paper)
            sensor_grid_size: Size of sensor grid (5 → 5×5 = 25 sensors)
            source_intensity: Source intensity Ψ (paper uses 10)
            regularization: Regularization parameter d_0 (paper uses 0.1)
            measurement_noise_std: Measurement noise std (sigma_w = 0.1)
            use_true_process_noise: If True, use V_true; if False, use V_filter
            dt: Time step
        """
        self.n_targets = n_targets
        self.dt = dt
        self.source_intensity = source_intensity
        self.regularization = regularization
        self.measurement_noise_std = measurement_noise_std

        # Build 5×5 sensor grid spanning [0, 40] × [0, 40]
        self.sensor_grid_size = sensor_grid_size
        self.sensor_positions = self._build_sensor_grid()
        self.n_sensors = len(self.sensor_positions)

        # Build single-target transition matrix
        F_single = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Build block diagonal F matrix (16×16 for 4 targets)
        F_blocks = [F_single for _ in range(n_targets)]
        self.F = self._block_diag(F_blocks)

        # V_true: Paper's true process noise for data generation
        V_true_single = (1.0 / 20.0) * np.array([
            [1.0/3.0,  0.0,      0.5,  0.0],
            [0.0,      1.0/3.0,  0.0,  0.5],
            [0.5,      0.0,      1.0,  0.0],
            [0.0,      0.5,      0.0,  1.0]
        ])

        # V_filter: Paper's filter process noise (larger uncertainty)
        V_filter_single = np.array([
            [3.0,  0.0,   0.1,  0.0],
            [0.0,  3.0,   0.0,  0.1],
            [0.1,  0.0,   0.03, 0.0],
            [0.0,  0.1,   0.0,  0.03]
        ])

        # Build block diagonal process noise matrices
        V_true_blocks = [V_true_single for _ in range(n_targets)]
        V_filter_blocks = [V_filter_single for _ in range(n_targets)]
        
        self.V_true = self._block_diag(V_true_blocks)
        self.V_filter = self._block_diag(V_filter_blocks)

        # Q is either V_true or V_filter depending on use case
        self.Q = self.V_true if use_true_process_noise else self.V_filter

        # Observation noise covariance (25×25)
        self.R = np.eye(self.n_sensors) * (measurement_noise_std ** 2)

        # Paper's 4 initial target states (page 8, Section V-A1)
        self.paper_initial_states = np.array([
            12.0, 6.0, 0.001, 0.001,      # Target 1
            32.0, 32.0, -0.001, -0.005,   # Target 2
            20.0, 13.0, -0.1, 0.01,       # Target 3
            15.0, 35.0, 0.002, 0.002      # Target 4
        ])

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

    def _block_diag(self, matrices):
        """Create block diagonal matrix from list of matrices."""
        from scipy.linalg import block_diag
        return block_diag(*matrices)

    @property
    def state_dim(self) -> int:
        return 4 * self.n_targets  # 16 for 4 targets

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
        """Vectorized amplitude decay log-prob for all particles (multi-target)."""
        # Particles: (N, 16) - 4 targets with [x, y, vx, vy] each
        n_particles = particles.shape[0]

        # Initialize total amplitudes: (N, M) where M = n_sensors
        total_amplitudes = np.zeros((n_particles, self.n_sensors))

        # Sum amplitudes from all 4 targets
        for target_idx in range(self.n_targets):
            # Extract position for this target
            pos_start = target_idx * 4
            pos = particles[:, pos_start:pos_start+2]  # (N, 2)

            # Distances from all particles to all sensors
            # Broadcasting: (N, 1, 2) - (1, M, 2) → (N, M, 2)
            diff = pos[:, np.newaxis, :] - self.sensor_positions[np.newaxis, :, :]
            r_squared = np.sum(diff**2, axis=2)  # (N, M)

            # Add this target's amplitudes
            total_amplitudes += self.source_intensity / (r_squared + self.regularization)

        # diff from observation: (N, M)
        obs_diff = observation - total_amplitudes

        # Assuming R is diagonal: sigma^2 * I
        variance = self.measurement_noise_std ** 2

        # Mahalanobis: sum((diff/sigma)^2) → (N,)
        mahalanobis = np.sum(obs_diff**2, axis=1) / variance

        # Log determinant for diagonal R
        logdet = self.n_sensors * np.log(2 * np.pi * variance)

        return -0.5 * (logdet + mahalanobis)

    @property
    def initial_state_mean(self) -> np.ndarray:
        """Initial state mean for TFP and Kalman filters (16-d for 4 targets)."""
        return self.paper_initial_states.copy()

    @property
    def initial_state_cov(self) -> np.ndarray:
        """
        Initial state covariance for TFP and Kalman filters.
        
        Paper specification: σ = 10 for positions, σ = 1 for velocities.
        Applied to all 4 targets.
        """
        single_target_cov = np.array([100.0, 100.0, 1.0, 1.0])  # σ² = [10², 10², 1², 1²]
        return np.diag(np.tile(single_target_cov, 4))  # Repeat for 4 targets

    # NumPy sampling methods

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample initial state.
        
        For paper reproduction, returns the paper's fixed initial states.
        """
        return self.paper_initial_states.copy()

    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = F @ x + w, w ~ N(0, Q)."""
        mean = self.F @ x
        noise = rng.multivariate_normal(np.zeros(self.state_dim), self.Q)
        return mean + noise

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample observation: additive amplitude at each sensor.
        
        Each sensor measures the sum of amplitudes from all 4 targets.
        """
        amplitudes = np.zeros(self.n_sensors)

        for s in range(self.n_sensors):
            sensor_pos = self.sensor_positions[s]
            
            # Sum contributions from all targets
            for c in range(self.n_targets):
                # Extract target c's position
                target_x = x[c * 4]
                target_y = x[c * 4 + 1]
                target_pos = np.array([target_x, target_y])
                
                # Distance from target to sensor
                r_squared = np.sum((target_pos - sensor_pos) ** 2)
                
                # Amplitude decay model: z = Ψ / (r^2 + d_0)
                amplitudes[s] += self.source_intensity / (r_squared + self.regularization)

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
            sensor_pos = self.sensor_positions[s]
            
            # Sum contributions from all targets
            for c in range(self.n_targets):
                target_x = x[c * 4]
                target_y = x[c * 4 + 1]
                target_pos = np.array([target_x, target_y])
                
                r_squared = np.sum((target_pos - sensor_pos) ** 2)
                amplitudes[s] += self.source_intensity / (r_squared + self.regularization)

        return amplitudes

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of observation function: H = ∂h/∂x.

        For additive observations with 4 targets:
        ∂z_s/∂x_i = -Ψ * 2 * (x_i - s_x) / (r_i^2 + d_0)^2
        ∂z_s/∂y_i = -Ψ * 2 * (y_i - s_y) / (r_i^2 + d_0)^2
        ∂z_s/∂vx_i = 0
        ∂z_s/∂vy_i = 0

        Returns:
            H: Jacobian matrix of shape (n_sensors, state_dim) = (25, 16)
        """
        H = np.zeros((self.n_sensors, self.state_dim))

        for s in range(self.n_sensors):
            sensor_pos = self.sensor_positions[s]
            
            # Compute partial derivatives for each target
            for c in range(self.n_targets):
                target_x = x[c * 4]
                target_y = x[c * 4 + 1]
                target_pos = np.array([target_x, target_y])
                
                # Distance squared
                r_squared = np.sum((target_pos - sensor_pos) ** 2)
                denominator = (r_squared + self.regularization) ** 2
                
                # Partial derivatives
                # ∂z_s/∂x_i
                H[s, c * 4] = -self.source_intensity * 2 * (target_x - sensor_pos[0]) / denominator
                # ∂z_s/∂y_i
                H[s, c * 4 + 1] = -self.source_intensity * 2 * (target_y - sensor_pos[1]) / denominator
                # ∂z_s/∂vx_i = 0 (already initialized to 0)
                # ∂z_s/∂vy_i = 0 (already initialized to 0)

        return H

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Compute log p(y | x) = log N(y; h(x), R).

        Args:
            y: Observation (n_sensors,)
            x: State (state_dim,)

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
                x_tf: Current states, shape (N, 16) or (16,)
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
                x_tf: States, shape (N, 16) or (16,)

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
                # Single state (16,)
                amplitudes = []
                for s in range(self.n_sensors):
                    sensor_x = sensor_positions[s, 0]
                    sensor_y = sensor_positions[s, 1]
                    
                    # Sum over all targets
                    amp_sum = 0.0
                    for c in range(self.n_targets):
                        target_x = x_tf[c * 4]
                        target_y = x_tf[c * 4 + 1]
                        dx = target_x - sensor_x
                        dy = target_y - sensor_y
                        r_squared = dx**2 + dy**2
                        amp_sum += psi / (r_squared + d0)
                    
                    amplitudes.append(amp_sum)

                h_x = tf.stack(amplitudes)
                residual = y_tf - h_x

                # Mahalanobis distance
                mahalanobis = tf.reduce_sum(residual * (R_inv @ residual))
                log_prob = -0.5 * (logdet + mahalanobis)
            else:
                # Batch of states (N, 16)
                N = tf.shape(x_tf)[0]
                amplitudes = []
                
                for s in range(self.n_sensors):
                    sensor_x = sensor_positions[s, 0]
                    sensor_y = sensor_positions[s, 1]
                    
                    # Sum over all targets for each particle
                    amp_sum = tf.zeros(N, dtype=tf.float32)
                    for c in range(self.n_targets):
                        target_x = x_tf[:, c * 4]
                        target_y = x_tf[:, c * 4 + 1]
                        dx = target_x - sensor_x
                        dy = target_y - sensor_y
                        r_squared = dx**2 + dy**2
                        amp_sum += psi / (r_squared + d0)
                    
                    amplitudes.append(amp_sum)

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
                Initial states, shape (n_samples, 16) or (16,) if n_samples=1
            """
            initial_state = tf.constant(self.paper_initial_states, dtype=tf.float32)
            
            if n_samples == 1:
                return initial_state
            else:
                return tf.tile(tf.expand_dims(initial_state, 0), [n_samples, 1])

