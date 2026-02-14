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
        """
        # Store dtype
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        # Convert to TensorFlow tensors
        self.F = tf.constant(F, dtype=self.dtype)
        self.B = tf.constant(B, dtype=self.dtype)
        self.H = tf.constant(H, dtype=self.dtype)
        self.D = tf.constant(D, dtype=self.dtype)

        # Store dimensions
        self.nx = int(self.F.shape[0])  # State dimension
        self.nv = int(self.B.shape[1])  # Process noise dimension
        self.ny = int(self.H.shape[0])  # Observation dimension
        self.nw = int(self.D.shape[1])  # Observation noise dimension

        # Validate dimensions
        if self.F.shape != (self.nx, self.nx):
            raise ValueError(f"F must be ({self.nx}, {self.nx}), got {self.F.shape}")
        if self.B.shape != (self.nx, self.nv):
            raise ValueError(f"B must be ({self.nx}, {self.nv}), got {self.B.shape}")
        if self.H.shape != (self.ny, self.nx):
            raise ValueError(f"H must be ({self.ny}, {self.nx}), got {self.H.shape}")
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
            self.mu_0 = tf.zeros(self.nx, dtype=self.dtype)
        else:
            self.mu_0 = tf.constant(mu_0, dtype=self.dtype)

        if Sigma_0 is None:
            self.Sigma_0 = tf.eye(self.nx, dtype=self.dtype)
        else:
            self.Sigma_0 = tf.constant(Sigma_0, dtype=self.dtype)

        if self.mu_0.shape != (self.nx,):
            raise ValueError(f"mu_0 must be ({self.nx},), got {self.mu_0.shape}")
        if self.Sigma_0.shape != (self.nx, self.nx):
            raise ValueError(f"Sigma_0 must be ({self.nx}, {self.nx}), got {self.Sigma_0.shape}")

    @property
    def state_dim(self) -> int:
        return self.nx

    @property
    def obs_dim(self) -> int:
        return self.ny

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

    @property
    def initial_state_cov(self) -> tf.Tensor:
        """Initial state covariance for EKF compatibility."""
        return self.Sigma_0

    @property
    def initial_state_mean(self) -> tf.Tensor:
        """Initial state mean for EKF compatibility."""
        return self.mu_0

    # TensorFlow methods

    def sample_initial_state(self, seed: Union[tf.Tensor, np.random.Generator]):
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0).

        Accepts either a numpy Generator (from generate_data) or TF seed (tuple of 2 ints)."""
        if hasattr(seed, 'standard_normal'):
            # NumPy rng from generate_data
            rng = seed
            z = rng.standard_normal(self.nx).astype(self.np_dtype)
            L = np.linalg.cholesky(np.array(self.Sigma_0))
            return np.array(self.mu_0) + L @ z
        return self._sample_initial_state_tf(seed)

    @tf.function
    def _sample_initial_state_tf(self, seed: tf.Tensor) -> tf.Tensor:
        """TF implementation for filters that pass seed."""
        L = tf.linalg.cholesky(self.Sigma_0)
        z = tf.random.stateless_normal([self.nx], seed=seed, dtype=self.dtype)
        return self.mu_0 + tf.linalg.matvec(L, z)

    def sample_state_transition(self, x, seed: Union[tf.Tensor, np.random.Generator]):
        """Sample from state transition: X' = F·X + B·v, v ~ N(0, I)."""
        if hasattr(seed, 'standard_normal'):
            rng = seed
            x_np = np.asarray(x)
            v = rng.standard_normal(self.nv).astype(self.np_dtype)
            F_np, B_np = np.array(self.F), np.array(self.B)
            return F_np @ x_np + B_np @ v
        return self._sample_state_transition_tf(x, seed)

    @tf.function
    def _sample_state_transition_tf(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """TF implementation for filters that pass seed."""
        v = tf.random.stateless_normal([self.nv], seed=seed, dtype=self.dtype)
        return tf.linalg.matvec(self.F, x) + tf.linalg.matvec(self.B, v)

    def sample_observation(self, x, seed: Union[tf.Tensor, np.random.Generator]):
        """Sample observation: Y = H·X + D·W, W ~ N(0, I)."""
        if hasattr(seed, 'standard_normal'):
            rng = seed
            x_np = np.asarray(x)
            w = rng.standard_normal(self.nw).astype(self.np_dtype)
            H_np, D_np = np.array(self.H), np.array(self.D)
            return H_np @ x_np + D_np @ w
        return self._sample_observation_tf(x, seed)

    @tf.function
    def _sample_observation_tf(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """TF implementation for filters that pass seed."""
        w = tf.random.stateless_normal([self.nw], seed=seed, dtype=self.dtype)
        return tf.linalg.matvec(self.H, x) + tf.linalg.matvec(self.D, w)

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

    @tf.function
    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized state transition mean: particles @ F^T (more efficient than transposing twice)."""
        return particles @ tf.transpose(self.F)

    @tf.function
    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Q is constant - return single matrix."""
        return self.Q

    @tf.function
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

        return -0.5 * (tf.cast(self.obs_dim, observation.dtype) * tf.math.log(2.0 * 3.14159265359) + logdet + mahalanobis)

    @tf.function
    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """H is constant — broadcast to (N, ny, nx)."""
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self.H, 0), [N, 1, 1])

    @tf.function
    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """h(x) = H @ x, vectorized: particles @ H^T -> (N, ny)."""
        return particles @ tf.transpose(self.H)

    @tf.function
    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """F is constant — broadcast to (N, nx, nx)."""
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self.F, 0), [N, 1, 1])

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
