"""Lorenz 96 model for high-dimensional filtering experiments (TensorFlow)."""

import numpy as np
import tensorflow as tf

from ..core.model_base import StateSpaceModel
from ..utils.ode_solvers import rk4_step


class Lorenz96Model(StateSpaceModel):
    """
    Lorenz 96 model for testing high-dimensional filtering algorithms.

    State evolution (Eq 24 from Hu & van Leeuwen 2021):
        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

    with cyclic boundary conditions: x_0 = x_n, x_{-1} = x_{n-1}, x_{n+1} = x_1.

    Time discretization uses 4th-order Runge-Kutta with step dt.

    Observations:
        y = H @ x + epsilon, epsilon ~ N(0, R)

    where H observes every `obs_spacing`-th variable (e.g., every 4th for 25% coverage).

    Reference:
        Hu, C.-C., & van Leeuwen, P. J. (2021). A particle flow filter for
        high-dimensional system applications. QJRMS 147(738), 2352-2374.
    """

    def __init__(
        self,
        state_dim: int = 1000,
        forcing: float = 8.0,
        dt: float = 0.05,
        obs_interval: int = 20,
        obs_spacing: int = 4,
        obs_noise_std: float = 1.0,
        process_noise_std: float = 0.0,
        initial_spread: float = 1.0,
        spinup_steps: int = 1000,
        dtype=None,
    ):
        if dtype is None:
            dtype = tf.float64
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        self._state_dim = state_dim
        self.forcing = forcing
        self.dt = dt
        self.obs_interval = obs_interval
        self.obs_spacing = obs_spacing
        self.obs_noise_std = obs_noise_std
        self.process_noise_std = process_noise_std
        self.initial_spread = initial_spread
        self.spinup_steps = spinup_steps

        # TF constants for hot-path ops
        self._forcing_tf = tf.constant(forcing, dtype=dtype)
        self._dt_tf = tf.constant(dt, dtype=dtype)
        self._obs_noise_std_tf = tf.constant(obs_noise_std, dtype=dtype)
        self._process_noise_std_tf = tf.constant(process_noise_std, dtype=dtype)
        self._initial_spread_tf = tf.constant(initial_spread, dtype=dtype)

        # Observation indices: every obs_spacing-th (0-indexed from obs_spacing-1)
        self.obs_indices = np.arange(obs_spacing - 1, state_dim, obs_spacing)
        self._obs_dim = len(self.obs_indices)

        # Observation matrix H (obs_dim x state_dim) — stored as TF constant
        H_np = np.zeros((self._obs_dim, state_dim))
        for i, idx in enumerate(self.obs_indices):
            H_np[i, idx] = 1.0
        self._H_tf = tf.constant(H_np, dtype=dtype)

        # Noise covariances
        self._R_tf = tf.eye(self._obs_dim, dtype=dtype) * tf.constant(
            obs_noise_std ** 2, dtype=dtype
        )
        self._Q_tf = tf.eye(state_dim, dtype=dtype) * tf.constant(
            process_noise_std ** 2, dtype=dtype
        )

        # Initial distribution
        self._mu_0 = tf.constant(np.ones(state_dim) * forcing, dtype=dtype)
        self._Sigma_0 = tf.eye(state_dim, dtype=dtype) * tf.constant(
            initial_spread ** 2, dtype=dtype
        )

        # Spinup: run numpy RK4 once to reach attractor, cache result as TF
        self._spinup_state_tf = self._do_spinup()

    def _do_spinup(self) -> tf.Tensor:
        """Run spinup in numpy (one-time), cache as TF constant."""
        rng = np.random.default_rng(0)
        x = np.ones(self._state_dim) * self.forcing + rng.normal(
            0, 0.01, self._state_dim
        )
        for _ in range(self.spinup_steps):
            x = self._rk4_step_np(x)
        return tf.constant(x, dtype=self.dtype)

    def _rk4_step_np(self, x: np.ndarray) -> np.ndarray:
        """Single numpy RK4 step (used only for spinup)."""
        def tend(v):
            return (np.roll(v, -1) - np.roll(v, 2)) * np.roll(v, 1) - v + self.forcing
        k1 = tend(x)
        k2 = tend(x + 0.5 * self.dt * k1)
        k3 = tend(x + 0.5 * self.dt * k2)
        k4 = tend(x + self.dt * k3)
        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

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
    # Core dynamics (TF)
    # ------------------------------------------------------------------

    def _lorenz96_tendency(self, x: tf.Tensor) -> tf.Tensor:
        """dx/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F.
        Works for (state_dim,) or (N, state_dim) via axis=-1."""
        return (
            (tf.roll(x, -1, axis=-1) - tf.roll(x, 2, axis=-1))
            * tf.roll(x, 1, axis=-1)
            - x
            + self._forcing_tf
        )

    def _integrate_rk4(self, x: tf.Tensor, n_steps: int) -> tf.Tensor:
        """Run n_steps of RK4 integration (no noise)."""
        for _ in range(n_steps):
            x = rk4_step(x, self._lorenz96_tendency, self._dt_tf)
        return x

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        noise = (
            tf.random.stateless_normal([self._state_dim], seed=seed, dtype=self.dtype)
            * self._initial_spread_tf
        )
        return self._spinup_state_tf + noise

    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        x_next = self._integrate_rk4(x, self.obs_interval)
        if self.process_noise_std > 0:
            noise = (
                tf.random.stateless_normal(
                    tf.shape(x), seed=seed, dtype=self.dtype
                )
                * self._process_noise_std_tf
            )
            x_next = x_next + noise
        return x_next

    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        mean = tf.linalg.matvec(self._H_tf, x)
        noise = (
            tf.random.stateless_normal(
                [self._obs_dim], seed=seed, dtype=self.dtype
            )
            * self._obs_noise_std_tf
        )
        return mean + noise

    # ------------------------------------------------------------------
    # Deterministic methods (for Kalman/EKF/UKF)
    # ------------------------------------------------------------------

    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        return self._integrate_rk4(x, self.obs_interval)

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        return self._Q_tf

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Numerical Jacobian via finite differences (TF)."""
        eps = tf.constant(1e-7, dtype=self.dtype)
        f0 = rk4_step(x, self._lorenz96_tendency, self._dt_tf)
        n = self._state_dim
        cols = []
        for j in range(n):
            e_j = tf.one_hot(j, n, dtype=self.dtype)
            f_pert = rk4_step(x + eps * e_j, self._lorenz96_tendency, self._dt_tf)
            cols.append((f_pert - f0) / eps)
        return tf.stack(cols, axis=1)

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(self._H_tf, x)

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        return self._R_tf

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        return self._H_tf

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(self._H_tf, x)

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        mean = tf.linalg.matvec(self._H_tf, x)
        diff = y - mean
        R_diag = tf.linalg.diag_part(self._R_tf)
        pi2 = tf.constant(2.0 * np.pi, dtype=self.dtype)
        log_det = tf.reduce_sum(tf.math.log(pi2 * R_diag))
        mahalanobis = tf.reduce_sum(diff ** 2 / R_diag)
        return -0.5 * (log_det + mahalanobis)

    # ------------------------------------------------------------------
    # Batch methods
    # ------------------------------------------------------------------

    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        noise = (
            tf.random.stateless_normal(
                [n, self._state_dim], seed=seed, dtype=self.dtype
            )
            * self._initial_spread_tf
        )
        return self._spinup_state_tf + noise

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Vectorized: tendency with axis=-1 handles (N, state_dim)."""
        result = self._integrate_rk4(particles, self.obs_interval)
        if self.process_noise_std > 0:
            noise = (
                tf.random.stateless_normal(
                    tf.shape(particles), seed=seed, dtype=self.dtype
                )
                * self._process_noise_std_tf
            )
            result = result + noise
        return result

    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return self._integrate_rk4(particles, self.obs_interval)

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return self._Q_tf

    def log_observation_prob_batch(
        self, observation: tf.Tensor, particles: tf.Tensor
    ) -> tf.Tensor:
        observed_states = tf.gather(particles, self.obs_indices, axis=1)
        diff = observation - observed_states
        variance = tf.constant(self.obs_noise_std ** 2, dtype=self.dtype)
        mahalanobis = tf.reduce_sum(diff ** 2, axis=1) / variance
        logdet = tf.constant(
            self._obs_dim * np.log(2 * np.pi * self.obs_noise_std ** 2),
            dtype=self.dtype,
        )
        return -0.5 * (logdet + mahalanobis)

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return particles @ tf.transpose(self._H_tf)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self._H_tf, 0), [N, 1, 1])
