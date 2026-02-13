"""Multi-Target Acoustic Tracking Model


- 4 independent targets with 16-dimensional joint state
- 25 sensors in 5×5 grid
- Distinction between V_true (data generation) and V_filter (algorithm use)
"""

import tensorflow as tf
import numpy as np
from typing import Optional

from ..core.model_base import StateSpaceModel


class AcousticTrackingFullModel(StateSpaceModel):
    """
    Multi-Target Amplitude-based Acoustic Tracking Model (Li & Coates Paper) - Pure TensorFlow.

    State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3, x4, y4, vx4, vy4]
        - (xi, yi): position of target i in 2D plane
        - (vxi, vyi): velocity of target i in 2D plane

    State evolution (constant velocity model for each target):
        x_{t+1} = F @ x_t + w_t, w_t ~ N(0, V)
        where F is block diagonal with 4 identical blocks

    Observation (additive amplitude decay at each sensor):
        z_s = Σ(i=1..4) Ψ / (r_s,i + d_0) + noise
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
        dt: float = 1.0,
        dtype=tf.float64
    ):
        """
        Initialize Multi-Target Acoustic Tracking Model (Pure TensorFlow).

        Args:
            n_targets: Number of targets (default: 4 per paper)
            sensor_grid_size: Size of sensor grid (5 → 5×5 = 25 sensors)
            source_intensity: Source intensity Ψ (paper uses 10)
            regularization: Regularization parameter d_0 (paper uses 0.1)
            measurement_noise_std: Measurement noise std (sigma_w = 0.1)
            use_true_process_noise: If True, use V_true; if False, use V_filter
            dt: Time step
            dtype: TensorFlow dtype for numerical precision (default: tf.float64)
        """
        self.n_targets = n_targets
        self.dt = dt
        self.source_intensity_val = source_intensity
        self.regularization_val = regularization
        self.measurement_noise_std = measurement_noise_std
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        # Build 5×5 sensor grid spanning [0, 40] × [0, 40]
        self.sensor_grid_size = sensor_grid_size
        sensor_positions_np = self._build_sensor_grid()
        self.sensor_positions = tf.constant(sensor_positions_np, dtype=self.dtype)
        self.n_sensors = len(sensor_positions_np)

        # Build single-target transition matrix
        F_single = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=self.np_dtype)

        # Build block diagonal F matrix (16×16 for 4 targets)
        F_blocks = [F_single for _ in range(n_targets)]
        F_np = self._block_diag(F_blocks)
        self.F = tf.constant(F_np, dtype=self.dtype)

        # V_true: Paper's true process noise for data generation
        V_true_single = (1.0 / 20.0) * np.array([
            [1.0/3.0,  0.0,      0.5,  0.0],
            [0.0,      1.0/3.0,  0.0,  0.5],
            [0.5,      0.0,      1.0,  0.0],
            [0.0,      0.5,      0.0,  1.0]
        ], dtype=self.np_dtype)

        # V_filter: Paper's filter process noise (larger uncertainty)
        V_filter_single = np.array([
            [3.0,  0.0,   0.1,  0.0],
            [0.0,  3.0,   0.0,  0.1],
            [0.1,  0.0,   0.03, 0.0],
            [0.0,  0.1,   0.0,  0.03]
        ], dtype=self.np_dtype)

        # Build block diagonal process noise matrices
        V_true_blocks = [V_true_single for _ in range(n_targets)]
        V_filter_blocks = [V_filter_single for _ in range(n_targets)]

        V_true_np = self._block_diag(V_true_blocks)
        V_filter_np = self._block_diag(V_filter_blocks)

        # Q is either V_true or V_filter depending on use case
        Q_np = V_true_np if use_true_process_noise else V_filter_np
        self.Q = tf.constant(Q_np, dtype=self.dtype)

        # Observation noise covariance (25×25)
        R_np = np.eye(self.n_sensors, dtype=self.np_dtype) * (measurement_noise_std ** 2)
        self.R = tf.constant(R_np, dtype=self.dtype)

        # Paper's 4 initial target states (page 8, Section V-A1)
        all_initial_states = [
            [12.0, 6.0, 0.001, 0.001],      # Target 1
            [32.0, 32.0, -0.001, -0.005],   # Target 2
            [20.0, 13.0, -0.1, 0.01],       # Target 3
            [15.0, 35.0, 0.002, 0.002],     # Target 4
        ]
        paper_initial_states = np.concatenate(all_initial_states[:n_targets]).astype(self.np_dtype)
        self.mu_0 = tf.constant(paper_initial_states, dtype=self.dtype)

        # Initial state covariance: σ = 10 for positions, σ = 1 for velocities
        single_target_cov = np.array([100.0, 100.0, 1.0, 1.0], dtype=self.np_dtype)  # σ² = [10², 10², 1², 1²]
        Sigma_0_np = np.diag(np.tile(single_target_cov, n_targets))
        self.Sigma_0 = tf.constant(Sigma_0_np, dtype=self.dtype)

        # TensorFlow constants for observation function
        self.psi = tf.constant(source_intensity, dtype=self.dtype)
        self.d0 = tf.constant(regularization, dtype=self.dtype)

    def _build_sensor_grid(self) -> np.ndarray:
        """Build 5×5 sensor grid at intersections of 10m spacing."""
        sensors = []
        spacing = 40.0 / (self.sensor_grid_size - 1)
        for i in range(self.sensor_grid_size):
            for j in range(self.sensor_grid_size):
                x = i * spacing
                y = j * spacing
                sensors.append([x, y])
        return np.array(sensors, dtype=self.np_dtype)

    def _block_diag(self, matrices):
        """Create block diagonal matrix from list of matrices."""
        from scipy.linalg import block_diag
        return block_diag(*matrices).astype(self.np_dtype)

    @property
    def state_dim(self) -> int:
        return 4 * self.n_targets  # 16 for 4 targets

    @property
    def obs_dim(self) -> int:
        return self.n_sensors  # 25

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        """Observation noise covariance R for flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> tf.Tensor:
        """Process noise covariance Q for flow filters."""
        return self.Q

    # TensorFlow methods

    @tf.function
    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0).

        Uses Gaussian sampling with mean = paper's initial states and
        covariance with σ=10 for positions, σ=1 for velocities.
        """
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([self.state_dim], seed=seed, dtype=self.dtype)
        return self.mu_0 + tf.linalg.matvec(L, z)

    @tf.function
    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample from state transition: x' = F @ x + w, w ~ N(0, Q)."""
        mean = tf.linalg.matvec(self.F, x)

        # Sample noise from N(0, Q)
        L = tf.linalg.cholesky(self.Q)
        z = tf.random.stateless_normal([self.state_dim], seed=seed, dtype=self.dtype)
        noise = tf.linalg.matvec(L, z)

        return mean + noise

    @tf.function
    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample observation: additive amplitude at each sensor.

        Each sensor measures the sum of amplitudes from all 4 targets.
        """
        amplitudes = self._compute_amplitudes(x)

        # Add Gaussian noise
        noise = tf.random.stateless_normal([self.n_sensors], seed=seed, dtype=self.dtype) * self.measurement_noise_std
        return amplitudes + noise

    @tf.function
    def _compute_amplitudes(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute deterministic amplitudes for state x.

        Args:
            x: State tensor, shape (16,) or (N, 16)

        Returns:
            Amplitudes, shape (n_sensors,) or (N, n_sensors)
        """
        if len(x.shape) == 1:
            # Single state (16,)
            amplitudes = []
            for s in range(self.n_sensors):
                sensor_pos = self.sensor_positions[s]

                # Sum contributions from all targets
                amp_sum = 0.0
                for c in range(self.n_targets):
                    target_x = x[c * 4]
                    target_y = x[c * 4 + 1]

                    dx = target_x - sensor_pos[0]
                    dy = target_y - sensor_pos[1]
                    r = tf.sqrt(dx**2 + dy**2 + 1e-10)

                    amp_sum += self.psi / (r + self.d0)

                amplitudes.append(amp_sum)

            return tf.stack(amplitudes)
        else:
            # Batch of states (N, 16)
            N = tf.shape(x)[0]
            amplitudes = []

            for s in range(self.n_sensors):
                sensor_pos = self.sensor_positions[s]

                # Sum contributions from all targets for each particle
                amp_sum = tf.zeros(N, dtype=self.dtype)
                for c in range(self.n_targets):
                    target_x = x[:, c * 4]
                    target_y = x[:, c * 4 + 1]

                    dx = target_x - sensor_pos[0]
                    dy = target_y - sensor_pos[1]
                    r = tf.sqrt(dx**2 + dy**2 + 1e-10)

                    amp_sum += self.psi / (r + self.d0)

                amplitudes.append(amp_sum)

            return tf.stack(amplitudes, axis=1)  # (N, n_sensors)

    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of state transition: E[x' | x] = F @ x."""
        return tf.linalg.matvec(self.F, x)

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of state transition (constant for linear model)."""
        return self.Q

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of state transition (constant F matrix for linear model)."""
        return self.F

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of observation: E[y | x] = h(x)."""
        return self._compute_amplitudes(x)

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        """Observation function h(x) for flow filters."""
        return self._compute_amplitudes(x)

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of observation (constant R matrix)."""
        return self.R

    @tf.function
    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute Jacobian of observation function: H = ∂h/∂x.

        For h = Ψ / (r + d_0) where r = ||target - sensor||:
        ∂z_s/∂x_i = -Ψ * (x_i - s_x) / (r * (r + d_0)^2)
        ∂z_s/∂y_i = -Ψ * (y_i - s_y) / (r * (r + d_0)^2)
        ∂z_s/∂vx_i = 0
        ∂z_s/∂vy_i = 0

        Returns:
            H: Jacobian matrix of shape (n_sensors, state_dim) = (25, 16)
        """
        H_rows = []

        for s in range(self.n_sensors):
            sensor_pos = self.sensor_positions[s]
            row = []

            # Compute partial derivatives for each target
            for c in range(self.n_targets):
                target_x = x[c * 4]
                target_y = x[c * 4 + 1]

                dx = target_x - sensor_pos[0]
                dy = target_y - sensor_pos[1]
                r = tf.sqrt(dx**2 + dy**2 + 1e-10)
                denominator = r * (r + self.d0) ** 2

                # Partial derivatives: dh/dx = -Psi * dx / (r * (r + d0)^2)
                dh_dx = -self.psi * dx / denominator
                dh_dy = -self.psi * dy / denominator

                row.extend([dh_dx, dh_dy, tf.constant(0.0, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype)])

            H_rows.append(tf.stack(row))

        return tf.stack(H_rows)

    @tf.function
    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(y | x) = log N(y; h(x), R).

        Args:
            y: Observation (n_sensors,)
            x: State (state_dim,) or (N, state_dim)

        Returns:
            Log probability (scalar or (N,))
        """
        h_x = self._compute_amplitudes(x)
        residual = y - h_x

        # Log determinant of 2πR
        sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R)

        if len(x.shape) == 1:
            # Single state
            # Mahalanobis distance
            R_inv = tf.linalg.inv(self.R)
            mahalanobis = tf.reduce_sum(residual * (R_inv @ residual))
            return -0.5 * (logdet + mahalanobis)
        else:
            # Batch of states
            R_inv = tf.linalg.inv(self.R)
            mahalanobis = tf.reduce_sum(
                residual * tf.linalg.matvec(R_inv, residual),
                axis=1
            )
            return -0.5 * (logdet + mahalanobis)

    # Batch methods for optimized particle filtering

    @tf.function
    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample n initial states using TensorFlow.

        Samples from N(mu_0, Sigma_0) where mu_0 is the paper's fixed states
        and Sigma_0 has σ=10 for positions, σ=1 for velocities.

        Args:
            n: Number of samples
            seed: Random seed

        Returns:
            Initial states, shape (n, 16)
        """
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([n, self.state_dim], seed=seed, dtype=self.dtype)
        return self.mu_0 + tf.linalg.matmul(z, L, transpose_b=True)

    @tf.function
    def sample_state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """
        Vectorized state transition sampling.

        Args:
            particles: Current states (N, 16)
            seed: Random seed

        Returns:
            Next states (N, 16)
        """
        N = tf.shape(particles)[0]

        # Mean: particles @ F^T
        mean = particles @ tf.transpose(self.F)

        # Noise: sample from N(0, Q)
        L = tf.linalg.cholesky(self.Q)
        z = tf.random.stateless_normal([N, self.state_dim], seed=seed, dtype=self.dtype)
        noise = tf.linalg.matmul(z, L, transpose_b=True)

        return mean + noise

    # Alias for compatibility with different filter implementations
    @tf.function
    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Alias for sample_state_transition_batch for filter compatibility."""
        return self.sample_state_transition_batch(particles, seed)

    @tf.function
    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized Jacobian for acoustic tracking: (N, n_sensors, state_dim).

        ∂z_s/∂x_i = -Ψ * (x_i - s_x) / (r * (r + d_0)^2)
        ∂z_s/∂y_i = -Ψ * (y_i - s_y) / (r * (r + d_0)^2)
        ∂z_s/∂vx_i = 0, ∂z_s/∂vy_i = 0
        """
        N = tf.shape(particles)[0]
        # Initialize H as zeros: (N, n_sensors, state_dim)
        H = tf.zeros([N, self.n_sensors, self.state_dim], dtype=self.dtype)

        for c in range(self.n_targets):
            # Extract target positions: (N,)
            target_x = particles[:, c * 4]
            target_y = particles[:, c * 4 + 1]

            for s in range(self.n_sensors):
                sensor_pos = self.sensor_positions[s]
                dx = target_x - sensor_pos[0]  # (N,)
                dy = target_y - sensor_pos[1]  # (N,)
                r = tf.sqrt(dx**2 + dy**2 + 1e-10)  # (N,)
                denom = r * (r + self.d0) ** 2  # (N,)

                dh_dx = -self.psi * dx / denom  # (N,)
                dh_dy = -self.psi * dy / denom  # (N,)

                # Build indices for scatter update
                idx_x = c * 4      # column for target x position
                idx_y = c * 4 + 1  # column for target y position

                # Create update tensors and add to H
                update_x = tf.zeros([N, self.n_sensors, self.state_dim], dtype=self.dtype)
                update_y = tf.zeros([N, self.n_sensors, self.state_dim], dtype=self.dtype)

                # Use index assignment via one_hot masking
                sensor_mask = tf.one_hot(s, self.n_sensors, dtype=self.dtype)  # (n_sensors,)
                col_mask_x = tf.one_hot(idx_x, self.state_dim, dtype=self.dtype)  # (state_dim,)
                col_mask_y = tf.one_hot(idx_y, self.state_dim, dtype=self.dtype)  # (state_dim,)

                # (N,) * (n_sensors,) * (state_dim,) -> (N, n_sensors, state_dim)
                H = H + dh_dx[:, tf.newaxis, tf.newaxis] * sensor_mask[tf.newaxis, :, tf.newaxis] * col_mask_x[tf.newaxis, tf.newaxis, :]
                H = H + dh_dy[:, tf.newaxis, tf.newaxis] * sensor_mask[tf.newaxis, :, tf.newaxis] * col_mask_y[tf.newaxis, tf.newaxis, :]

        return H

    @tf.function
    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized h(x) for batch of states: (N, n_sensors)."""
        return self._compute_amplitudes(particles)

    @tf.function
    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """F is constant — broadcast to (N, state_dim, state_dim)."""
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self.F, 0), [N, 1, 1])

    @tf.function
    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized state transition mean: particles @ F.T."""
        return particles @ tf.transpose(self.F)

    @tf.function
    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized amplitude decay log-prob for all particles."""
        return self.log_observation_prob(observation, particles)
