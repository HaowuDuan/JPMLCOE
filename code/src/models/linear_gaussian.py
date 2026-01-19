"""Linear-Gaussian state-space model."""

import numpy as np
from typing import Optional, Union, List
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..core.model_base import StateSpaceModel


class LinearGaussianModel(StateSpaceModel):
    """
    Linear-Gaussian state-space model.

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
        F: Union[np.ndarray, List],
        B: Union[np.ndarray, List],
        H: Union[np.ndarray, List],
        D: Union[np.ndarray, List],
        mu_0: Optional[Union[np.ndarray, List]] = None,
        Sigma_0: Optional[Union[np.ndarray, List]] = None
    ):
        """
        Initialize Linear-Gaussian Model.

        Args:
            F: State transition matrix (nx, nx)
            B: Process noise matrix (nx, nv)
            H: Observation matrix (ny, nx)
            D: Observation noise matrix (ny, nw)
            mu_0: Initial state mean (nx,). If None, uses zero vector.
            Sigma_0: Initial state covariance (nx, nx). If None, uses identity.
        """
        # Convert to numpy arrays (handles both arrays and Hydra ListConfig)
        F = np.array(F, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        H = np.array(H, dtype=np.float64)
        D = np.array(D, dtype=np.float64)

        # Validate dimensions
        self.nx = F.shape[0]
        self.nv = B.shape[1]
        self.ny = H.shape[0]
        self.nw = D.shape[1]

        if F.shape != (self.nx, self.nx):
            raise ValueError(f"F must be ({self.nx}, {self.nx}), got {F.shape}")
        if B.shape != (self.nx, self.nv):
            raise ValueError(f"B must be ({self.nx}, {self.nv}), got {B.shape}")
        if H.shape != (self.ny, self.nx):
            raise ValueError(f"H must be ({self.ny}, {self.nx}), got {H.shape}")
        if D.shape != (self.ny, self.nw):
            raise ValueError(f"D must be ({self.ny}, {self.nw}), got {D.shape}")

        self.F = F
        self.B = B
        self.H = H
        self.D = D

        # Compute noise covariances
        self.Q = B @ B.T  # Process noise covariance
        self.R = D @ D.T  # Observation noise covariance

        # Initial state distribution
        self.mu_0 = np.array(mu_0, dtype=np.float64) if mu_0 is not None else np.zeros(self.nx)
        self.Sigma_0 = np.array(Sigma_0, dtype=np.float64) if Sigma_0 is not None else np.eye(self.nx)

        if self.mu_0.shape != (self.nx,):
            raise ValueError(f"mu_0 must be ({self.nx},), got {self.mu_0.shape}")
        if self.Sigma_0.shape != (self.nx, self.nx):
            raise ValueError(f"Sigma_0 must be ({self.nx}, {self.nx}), got {self.Sigma_0.shape}")

        # TensorFlow constants (if available)
        if TF_AVAILABLE:
            self.F_tf = tf.constant(self.F, dtype=tf.float32)
            self.B_tf = tf.constant(self.B, dtype=tf.float32)
            self.H_tf = tf.constant(self.H, dtype=tf.float32)
            self.D_tf = tf.constant(self.D, dtype=tf.float32)
            self.Q_tf = tf.constant(self.Q, dtype=tf.float32)
            self.R_tf = tf.constant(self.R, dtype=tf.float32)
            self.mu_0_tf = tf.constant(self.mu_0, dtype=tf.float32)
            self.Sigma_0_tf = tf.constant(self.Sigma_0, dtype=tf.float32)

    @property
    def state_dim(self) -> int:
        return self.nx

    @property
    def obs_dim(self) -> int:
        return self.ny

    # NumPy methods (for data generation and Kalman filters)

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0)."""
        return rng.multivariate_normal(self.mu_0, self.Sigma_0)

    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample from state transition: X' = F·X + B·V, V ~ N(0, I)."""
        v = rng.multivariate_normal(np.zeros(self.nv), np.eye(self.nv))
        return self.F @ x + self.B @ v

    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample observation: Y = H·X + D·W, W ~ N(0, I)."""
        w = rng.multivariate_normal(np.zeros(self.nw), np.eye(self.nw))
        return self.H @ x + self.D @ w

    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[X' | X] = F·X."""
        return self.F @ x

    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Cov[X' | X] = Q."""
        return self.Q

    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂(F·x)/∂x = F."""
        return self.F

    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[Y | X] = H·X."""
        return self.H @ x

    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Cov[Y | X] = R."""
        return self.R

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of observation: ∂(H·x)/∂x = H."""
        return self.H

    def observation_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Hessian of observation function: ∂²hᵢ/∂x².
        
        For linear observation h(x) = H·x, the Hessian is zero.
        
        Returns:
            Tensor of shape (obs_dim, state_dim, state_dim), all zeros.
        """
        return np.zeros((self.obs_dim, self.state_dim, self.state_dim))

    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).

        p(y | x) = N(y | H·x, R)
        """
        from ..utils.distributions import log_gaussian_prob
        mean = self.H @ x
        return log_gaussian_prob(y, mean, self.R)

    # For flow filters
    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters: returns H·x."""
        return self.H @ x

    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return self.Q

    # TensorFlow methods (for particle filters)

    if TF_AVAILABLE:
        @tf.function
        def sample_state_transition_tf(self, x_tf: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of state transition sampling.

            Args:
                x_tf: Current state (state_dim,)
                seed: Random seed for stateless sampling

            Returns:
                Next state (state_dim,)
            """
            # Sample noise: v ~ N(0, I)
            v = tf.random.stateless_normal([self.nv], seed=seed)

            # X' = F·X + B·v
            return tf.linalg.matvec(self.F_tf, x_tf) + tf.linalg.matvec(self.B_tf, v)

        @tf.function
        def log_observation_prob_tf(self, y_tf: tf.Tensor, x_tf: tf.Tensor) -> tf.Tensor:
            """
            TensorFlow version of observation log-probability.

            Args:
                y_tf: Observation (obs_dim,)
                x_tf: State (state_dim,)

            Returns:
                Log probability (scalar)
            """
            # Mean: H·x
            mean = tf.linalg.matvec(self.H_tf, x_tf)

            # Difference
            diff = y_tf - mean

            # log p(y|x) = -0.5 * [log|2πR| + (y-μ)^T R^{-1} (y-μ)]
            sign, logdet = tf.linalg.slogdet(2.0 * np.pi * self.R_tf)
            mahalanobis = tf.reduce_sum(diff * tf.linalg.solve(self.R_tf, diff))

            return -0.5 * (logdet + mahalanobis)

        @tf.function
        def sample_initial_state_batch_tf(self, n: int, seed: tf.Tensor) -> tf.Tensor:
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
            L = tf.linalg.cholesky(self.Sigma_0_tf)
            z = tf.random.stateless_normal([n, self.nx], seed=seed)

            return self.mu_0_tf + tf.linalg.matvec(L, z, transpose_a=True)
