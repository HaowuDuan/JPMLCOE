"""Acoustic Tracking Model matching Li & Coates paper (TensorFlow).

This model matches the acoustic tracking scenario in Section V of:
"Particle Filtering with Invertible Particle Flow" by Li & Coates.

Uses amplitude decay measurements from multiple sensors.
"""

import numpy as np
import tensorflow as tf
from typing import Optional

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
        z_j = Psi / (r_j^2 + d_0) + noise

    where:
        - Psi: source intensity (10 in paper)
        - r_j = ||x - s_j||: distance to sensor j
        - d_0: regularization parameter (0.1 in paper)
    """

    def __init__(
        self,
        sensor_positions: Optional[list] = None,
        source_intensity: float = 10.0,
        regularization: float = 0.1,
        measurement_noise_std: float = 0.1,
        process_noise_cov: Optional[list] = None,
        dt: float = 1.0,
        dtype=None,
    ):
        if dtype is None:
            dtype = tf.float32
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32
        self.dt = dt
        self.source_intensity = source_intensity
        self.regularization = regularization
        self.measurement_noise_std = measurement_noise_std

        # Default: 4 sensors at corners of 40m x 40m area
        if sensor_positions is None:
            sp_np = np.array([
                [0.0, 0.0], [40.0, 0.0], [0.0, 40.0], [40.0, 40.0]
            ])
        else:
            sp_np = np.array(sensor_positions)
        self.n_sensors = len(sp_np)

        # TF constants
        self._sensor_positions_tf = tf.constant(sp_np, dtype=dtype)
        self._psi_tf = tf.constant(source_intensity, dtype=dtype)
        self._d0_tf = tf.constant(regularization, dtype=dtype)
        self._meas_noise_std_tf = tf.constant(measurement_noise_std, dtype=dtype)

        # State transition matrix (constant velocity model)
        F_np = np.array([
            [1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]
        ])
        self._F_tf = tf.constant(F_np, dtype=dtype)

        # Process noise covariance
        if process_noise_cov is None:
            Q_np = (1.0 / 20.0) * np.array([
                [1.0/3.0, 0.0, 0.5, 0.0],
                [0.0, 1.0/3.0, 0.0, 0.5],
                [0.5, 0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0],
            ])
        else:
            Q_np = np.array(process_noise_cov)
        self._Q_tf = tf.constant(Q_np, dtype=dtype)
        self._L_Q = tf.linalg.cholesky(self._Q_tf)

        # Observation noise covariance
        self._R_tf = tf.eye(self.n_sensors, dtype=dtype) * tf.constant(
            measurement_noise_std ** 2, dtype=dtype
        )

        # Initial distribution
        self._mu_0 = tf.constant([20.0, 20.0, 0.0, 0.0], dtype=dtype)
        self._Sigma_0 = tf.constant(
            np.diag([33.33, 33.33, 1.0, 1.0]), dtype=dtype
        )

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def obs_dim(self) -> int:
        return self.n_sensors

    @property
    def mu_0(self) -> tf.Tensor:
        return self._mu_0

    @property
    def Sigma_0(self) -> tf.Tensor:
        return self._Sigma_0

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        return self._R_tf

    @property
    def process_noise_cov(self) -> tf.Tensor:
        return self._Q_tf

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _amplitudes(self, x: tf.Tensor) -> tf.Tensor:
        """Compute amplitude decay h(x) for single state (4,)."""
        pos = x[:2]  # (2,)
        diff = pos - self._sensor_positions_tf  # (n_sensors, 2)
        r_sq = tf.reduce_sum(diff ** 2, axis=1)  # (n_sensors,)
        return self._psi_tf / (r_sq + self._d0_tf)

    def _amplitudes_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Compute amplitude decay for batch (N, 4) -> (N, n_sensors)."""
        pos = particles[:, :2]  # (N, 2)
        diff = pos[:, tf.newaxis, :] - self._sensor_positions_tf[tf.newaxis, :, :]
        r_sq = tf.reduce_sum(diff ** 2, axis=2)  # (N, n_sensors)
        return self._psi_tf / (r_sq + self._d0_tf)

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        s1, s2 = tf.random.experimental.stateless_split(seed)
        xy = tf.random.stateless_uniform([2], seed=s1, minval=10.0, maxval=30.0, dtype=self.dtype)
        v = tf.random.stateless_normal([2], seed=s2, dtype=self.dtype)
        return tf.concat([xy, v], axis=0)

    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        mean = tf.linalg.matvec(self._F_tf, x)
        z = tf.random.stateless_normal([4], seed=seed, dtype=self.dtype)
        return mean + tf.linalg.matvec(self._L_Q, z)

    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        h_x = self._amplitudes(x)
        noise = tf.random.stateless_normal(
            [self.n_sensors], seed=seed, dtype=self.dtype
        ) * self._meas_noise_std_tf
        return h_x + noise

    # ------------------------------------------------------------------
    # Deterministic methods
    # ------------------------------------------------------------------

    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(self._F_tf, x)

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        return self._Q_tf

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        return self._F_tf

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        return self._amplitudes(x)

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        return self._R_tf

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of amplitude decay h(x) w.r.t. state [x,y,vx,vy]."""
        pos = x[:2]
        diff = pos - self._sensor_positions_tf  # (n_sensors, 2)
        r_sq = tf.reduce_sum(diff ** 2, axis=1)  # (n_sensors,)
        denom_sq = (r_sq + self._d0_tf) ** 2
        # dh_j/dx = -2*psi*(x-sx)/(r^2+d0)^2, dh_j/dy = -2*psi*(y-sy)/(...)
        dh_dxy = -2.0 * self._psi_tf * diff / denom_sq[:, tf.newaxis]  # (n_sensors, 2)
        # dh/dvx = dh/dvy = 0
        zeros = tf.zeros([self.n_sensors, 2], dtype=self.dtype)
        return tf.concat([dh_dxy, zeros], axis=1)  # (n_sensors, 4)

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        return self._amplitudes(x)

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        h_x = self._amplitudes(x)
        residual = y - h_x
        variance = self._meas_noise_std_tf ** 2
        pi2 = tf.constant(2.0 * np.pi, dtype=self.dtype)
        logdet = tf.cast(self.n_sensors, self.dtype) * tf.math.log(pi2 * variance)
        mahalanobis = tf.reduce_sum(residual ** 2) / variance
        return -0.5 * (logdet + mahalanobis)

    # ------------------------------------------------------------------
    # Batch methods
    # ------------------------------------------------------------------

    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        s1, s2 = tf.random.experimental.stateless_split(seed)
        xy = tf.random.stateless_uniform([n, 2], seed=s1, minval=10.0, maxval=30.0, dtype=self.dtype)
        v = tf.random.stateless_normal([n, 2], seed=s2, dtype=self.dtype)
        return tf.concat([xy, v], axis=1)

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        mean = particles @ tf.transpose(self._F_tf)
        z = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
        return mean + z @ tf.transpose(self._L_Q)

    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return particles @ tf.transpose(self._F_tf)

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return self._Q_tf

    def log_observation_prob_batch(self, observation: tf.Tensor,
                                   particles: tf.Tensor) -> tf.Tensor:
        amplitudes = self._amplitudes_batch(particles)
        diff = observation - amplitudes
        variance = self._meas_noise_std_tf ** 2
        pi2 = tf.constant(2.0 * np.pi, dtype=self.dtype)
        mahalanobis = tf.reduce_sum(diff ** 2, axis=1) / variance
        logdet = tf.cast(self.n_sensors, self.dtype) * tf.math.log(pi2 * variance)
        return -0.5 * (logdet + mahalanobis)

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return self._amplitudes_batch(particles)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        pos = particles[:, :2]  # (N, 2)
        diff = pos[:, tf.newaxis, :] - self._sensor_positions_tf[tf.newaxis, :, :]  # (N, M, 2)
        r_sq = tf.reduce_sum(diff ** 2, axis=2)  # (N, M)
        denom_sq = (r_sq + self._d0_tf) ** 2
        dh_dxy = -2.0 * self._psi_tf * diff / denom_sq[:, :, tf.newaxis]  # (N, M, 2)
        N = tf.shape(particles)[0]
        zeros = tf.zeros([N, self.n_sensors, 2], dtype=self.dtype)
        return tf.concat([dh_dxy, zeros], axis=2)  # (N, M, 4)

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self._F_tf, 0), [N, 1, 1])
