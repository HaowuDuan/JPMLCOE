"""Linear-Gaussian state-space model - TensorFlow version."""

import tensorflow as tf
import numpy as np
from typing import Optional, Union, List

from ..core.model_base import StateSpaceModel


class LinearGaussianModel(StateSpaceModel):
    """
    Linear-Gaussian state-space model (TensorFlow-only).

    Model:
        X_n = F·X_{n-1} + B·V_n,  V_n ~ N(0, I)
        Y_n = H·X_n + D·W_n,  W_n ~ N(0, I)

    where:
        - F: State transition matrix (nx, nx)
        - B: Process noise matrix (nx, nv)
        - H: Observation matrix (ny, nx)
        - D: Observation noise matrix (ny, nw)
        - Q = B·B^T: Process noise covariance
        - R = D·D^T: Observation noise covariance
        - mu_0: Initial state mean (nx,)
        - Sigma_0: Initial state covariance (nx, nx)

    This is the standard linear-Gaussian model for which the Kalman Filter
    is optimal. Can also be used with EKF/UKF/PF (though KF is optimal).
    """

    def __init__(
        self,
        F: Union[np.ndarray, List, tf.Tensor],
        B: Union[np.ndarray, List, tf.Tensor],
        H: Union[np.ndarray, List, tf.Tensor],
        D: Union[np.ndarray, List, tf.Tensor],
        mu_0: Optional[Union[np.ndarray, List, tf.Tensor]] = None,
        Sigma_0: Optional[Union[np.ndarray, List, tf.Tensor]] = None,
        process_noise_std: Optional[float] = None,
        obs_noise_std: Optional[float] = None,
        transition_coeff: Optional[float] = None,
        observation_coeff: Optional[float] = None,
        dtype=tf.float32
    ):
        """
        Initialize Linear-Gaussian Model (TensorFlow-only).

        Args:
            F: State transition matrix (nx, nx)
            B: Process noise matrix (nx, nv). When process_noise_std is set,
               B is treated as the noise direction; Q = process_noise_std^2 * B@B^T.
            H: Observation matrix (ny, nx)
            D: Observation noise matrix (ny, nw). When obs_noise_std is set,
               D is treated as the noise direction; R = obs_noise_std^2 * D@D^T.
            mu_0: Initial state mean (nx,). If None, uses zero vector.
            Sigma_0: Initial state covariance (nx, nx). If None, uses identity.
            process_noise_std: Optional scalar noise scale for Q. When set,
                Q = process_noise_std^2 * B@B^T. Used for HMC parameter inference.
            obs_noise_std: Optional scalar noise scale for R. When set,
                R = obs_noise_std^2 * D@D^T. Used for HMC parameter inference.
            transition_coeff: Optional scalar for F (1D only). When set,
                F = [[transition_coeff]]. Used for HMC parameter inference.
            observation_coeff: Optional scalar for H (1D only). When set,
                H = [[observation_coeff]]. Used for HMC parameter inference.
        """
        # Store dtype
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        # Optional scalar parameters for dynamic F/H (for HMC inference)
        self.transition_coeff = transition_coeff
        self.observation_coeff = observation_coeff

        # Convert to TensorFlow tensors
        self._F_const = tf.constant(F, dtype=self.dtype)
        self.B = tf.constant(B, dtype=self.dtype)
        self._H_const = tf.constant(H, dtype=self.dtype)
        self.D = tf.constant(D, dtype=self.dtype)

        # Store dimensions (use constants to avoid property lookup during init)
        self.nx = int(self._F_const.shape[0])  # State dimension
        self.nv = int(self.B.shape[1])  # Process noise dimension
        self.ny = int(self._H_const.shape[0])  # Observation dimension
        self.nw = int(self.D.shape[1])  # Observation noise dimension

        # Validate dimensions
        if self._F_const.shape != (self.nx, self.nx):
            raise ValueError(f"F must be ({self.nx}, {self.nx}), got {self._F_const.shape}")
        if self.B.shape != (self.nx, self.nv):
            raise ValueError(f"B must be ({self.nx}, {self.nv}), got {self.B.shape}")
        if self._H_const.shape != (self.ny, self.nx):
            raise ValueError(f"H must be ({self.ny}, {self.nx}), got {self._H_const.shape}")
        if self.D.shape != (self.ny, self.nw):
            raise ValueError(f"D must be ({self.ny}, {self.nw}), got {self.D.shape}")

        # Base noise covariances (direction only, scale is separate)
        self._Q_base = self.B @ tf.transpose(self.B)
        self._R_base = self.D @ tf.transpose(self.D)

        # Optional scalar noise parameters (for HMC inference)
        # When set: Q = process_noise_std^2 * _Q_base, R = obs_noise_std^2 * _R_base
        # When None: Q = _Q_base, R = _R_base (backward compatible)
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std

        # Initial state distribution
        if mu_0 is None:
            self._mu_0 = tf.zeros(self.nx, dtype=self.dtype)
        else:
            self._mu_0 = tf.constant(mu_0, dtype=self.dtype)

        if Sigma_0 is None:
            self._Sigma_0 = tf.eye(self.nx, dtype=self.dtype)
        else:
            self._Sigma_0 = tf.constant(Sigma_0, dtype=self.dtype)

        if self._mu_0.shape != (self.nx,):
            raise ValueError(f"mu_0 must be ({self.nx},), got {self._mu_0.shape}")
        if self._Sigma_0.shape != (self.nx, self.nx):
            raise ValueError(f"Sigma_0 must be ({self.nx}, {self.nx}), got {self._Sigma_0.shape}")

    @property
    def state_dim(self) -> int:
        return self.nx

    @property
    def obs_dim(self) -> int:
        return self.ny

    @property
    def mu_0(self) -> tf.Tensor:
        return self._mu_0

    @property
    def Sigma_0(self) -> tf.Tensor:
        return self._Sigma_0

    @property
    def F(self):
        """State transition matrix — dynamic if transition_coeff is set."""
        if self.transition_coeff is not None:
            tc = self.transition_coeff
            if not isinstance(tc, tf.Tensor):
                tc = tf.constant(float(tc), dtype=self.dtype)
            return tf.reshape(tc, [1, 1])
        return self._F_const

    @F.setter
    def F(self, value):
        """Allow direct assignment for backward compat."""
        self._F_const = value

    @property
    def H(self):
        """Observation matrix — dynamic if observation_coeff is set."""
        if self.observation_coeff is not None:
            oc = self.observation_coeff
            if not isinstance(oc, tf.Tensor):
                oc = tf.constant(float(oc), dtype=self.dtype)
            return tf.reshape(oc, [1, 1])
        return self._H_const

    @H.setter
    def H(self, value):
        """Allow direct assignment for backward compat."""
        self._H_const = value

    @property
    def Q(self):
        """Process noise covariance — dynamic if process_noise_std is set."""
        if self.process_noise_std is not None:
            pns = self.process_noise_std
            if not isinstance(pns, tf.Tensor):
                pns = tf.constant(float(pns), dtype=self.dtype)
            return pns ** 2 * self._Q_base
        return self._Q_base

    @Q.setter
    def Q(self, value):
        """Allow direct assignment for backward compat (e.g. generate_data)."""
        self._Q_base = value

    @property
    def R(self):
        """Observation noise covariance — dynamic if obs_noise_std is set."""
        if self.obs_noise_std is not None:
            ons = self.obs_noise_std
            if not isinstance(ons, tf.Tensor):
                ons = tf.constant(float(ons), dtype=self.dtype)
            return ons ** 2 * self._R_base
        return self._R_base

    @R.setter
    def R(self, value):
        """Allow direct assignment for backward compat."""
        self._R_base = value

    # Sampling methods

    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0)."""
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([self.nx], seed=seed, dtype=self.dtype)
        return self.mu_0 + tf.linalg.matvec(L, z)

    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample from state transition: X' = F·X + B·v, v ~ N(0, I)."""
        v = tf.random.stateless_normal([self.nv], seed=seed, dtype=self.dtype)
        noise = tf.linalg.matvec(self.B, v)
        if self.process_noise_std is not None:
            pns = self.process_noise_std
            if not isinstance(pns, tf.Tensor):
                pns = tf.constant(float(pns), dtype=self.dtype)
            noise = pns * noise
        return tf.linalg.matvec(self.F, x) + noise

    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample observation: Y = H·X + D·W, W ~ N(0, I)."""
        w = tf.random.stateless_normal([self.nw], seed=seed, dtype=self.dtype)
        noise = tf.linalg.matvec(self.D, w)
        if self.obs_noise_std is not None:
            ons = self.obs_noise_std
            if not isinstance(ons, tf.Tensor):
                ons = tf.constant(float(ons), dtype=self.dtype)
            noise = ons * noise
        return tf.linalg.matvec(self.H, x) + noise

    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of state transition: E[X' | X] = F·X."""
        return tf.linalg.matvec(self.F, x)

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of state transition: Cov[X' | X] = Q (dynamic when process_noise_std set)."""
        return self.Q

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of state transition: ∂(F·x)/∂x = F."""
        return self.F

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of observation: E[Y | X] = H·X."""
        return tf.linalg.matvec(self.H, x)

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of observation: Cov[Y | X] = R (dynamic when obs_noise_std set)."""
        return self.R

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of observation: ∂(H·x)/∂x = H."""
        return self.H

    def observation_hessian(self, x: tf.Tensor) -> tf.Tensor:
        """
        Hessian of observation function: ∂²hᵢ/∂x².

        For linear observation h(x) = H·x, the Hessian is zero.

        Returns:
            Tensor of shape (obs_dim, state_dim, state_dim), all zeros.
        """
        return tf.zeros((self.obs_dim, self.state_dim, self.state_dim), dtype=self.dtype)

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Log probability of observation: log p(y | x). p(y | x) = N(y | H·x, R)."""
        from ..utils.distributions import log_gaussian_prob
        mean = tf.linalg.matvec(self.H, x)
        return log_gaussian_prob(y, mean, self.R)  # Uses dynamic R property

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        """Observation function h(x) for flow filters: returns H·x."""
        return tf.linalg.matvec(self.H, x)

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        """Observation noise covariance R for flow filters (dynamic)."""
        return self.R

    @property
    def process_noise_cov(self) -> tf.Tensor:
        """Process noise covariance Q for flow filters (dynamic)."""
        return self.Q

    # Batch methods for optimized particle filtering

    def state_transition_mean_batch(self, particles: tf.Tensor, t=None) -> tf.Tensor:
        """Vectorized state transition mean: particles @ F^T (more efficient than transposing twice)."""
        return particles @ tf.transpose(self.F)

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Q is constant - return single matrix."""
        return self.Q

    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized Gaussian log-prob for all particles."""
        # Mean: particles @ H^T
        means = particles @ tf.transpose(self.H)
        diff = observation - means

        # Cholesky factorization of R
        L_R = tf.linalg.cholesky(self.R)

        # Solve: y = L_R^{-1} @ diff.T → (ny, N)
        y = tf.linalg.triangular_solve(L_R, tf.transpose(diff), lower=True)

        # Mahalanobis distance: sum(y^2) per particle → (N,)
        mahalanobis = tf.reduce_sum(y**2, axis=0)

        # Log determinant
        logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_R)))

        log_2pi = tf.math.log(tf.constant(2.0 * 3.14159265358979323846, dtype=observation.dtype))
        return -0.5 * (tf.cast(self.obs_dim, observation.dtype) * log_2pi + logdet + mahalanobis)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """H — broadcast to (N, ny, nx)."""
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self.H, 0), [N, 1, 1])

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """h(x) = H @ x, vectorized: particles @ H^T -> (N, ny)."""
        return particles @ tf.transpose(self.H)

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """F — broadcast to (N, nx, nx)."""
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self.F, 0), [N, 1, 1])

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor, t=None) -> tf.Tensor:
        """Vectorized state transition: X' = F·X + B·v, v ~ N(0, I)."""
        N = tf.shape(particles)[0]
        v = tf.random.stateless_normal([N, self.nv], seed=seed, dtype=self.dtype)
        noise = tf.linalg.matmul(v, self.B, transpose_b=True)  # (N, nx)
        if self.process_noise_std is not None:
            pns = self.process_noise_std
            if not isinstance(pns, tf.Tensor):
                pns = tf.constant(float(pns), dtype=self.dtype)
            noise = pns * noise
        return particles @ tf.transpose(self.F) + noise

    @tf.function
    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample n initial states using TensorFlow.

        Args:
            n: Number of samples
            seed: Random seed

        Returns:
            Initial states (n, state_dim)
        """
        # Sample from N(mu_0, Sigma_0)
        # Use Cholesky: X = mu_0 + L·Z where L·L^T = Sigma_0, Z ~ N(0, I)
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([n, self.nx], seed=seed, dtype=self.dtype)

        # Correct batch multiplication: z @ L^T
        return self.mu_0 + tf.linalg.matmul(z, L, transpose_b=True)
