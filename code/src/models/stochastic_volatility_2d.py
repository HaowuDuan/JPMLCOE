"""2D Stochastic Volatility model (TensorFlow).

Two-component model with level and log-volatility:

    State evolution:
        x_{t+1} = A x_t + e_t,  e_t ~ N(0, Sigma)

    where x = [x_1, x_2]^T, A is 2x2, Sigma is 2x2 positive definite.

    Observation:
        y_t = b * x_{1,t} + exp(x_{2,t} / 2) * v_t,  v_t ~ N(0, 1)

    x_1 is the "level" component (enters observation mean),
    x_2 is the "log-volatility" component (enters observation variance).

For EKF/UKF:
    - observation_mean(x) = b * x_1  (nonzero!)
    - observation_jacobian(x) = [b, 0]  (H_x != 0, so EKF works)
    - observation_cov(x) = exp(x_2)  (state-dependent noise)

Stationarity requires all eigenvalues of A have |lambda| < 1.
For diagonal A = diag(a1, a2), this means |a1| < 1 and |a2| < 1.

The stationary initial distribution is x_0 ~ N(0, P_0) where P_0
solves the discrete Lyapunov equation: P_0 = A P_0 A^T + Sigma.
"""

import numpy as np
import tensorflow as tf
from scipy import linalg as scipy_linalg

from ..core.model_base import StateSpaceModel


class StochasticVolatility2DModel(StateSpaceModel):
    """
    2D Stochastic Volatility Model.

    Parameters:
        a1, a2: diagonal entries of A (persistence parameters, |a_i| < 1)
        sigma1, sigma2: diagonal entries of sqrt(Sigma) (noise std devs)
        b: observation coefficient for x_1
        A: full 2x2 transition matrix (overrides a1, a2 if provided)
        Sigma: full 2x2 noise covariance (overrides sigma1, sigma2 if provided)
    """

    def __init__(
        self,
        a1: float = 0.95,
        a2: float = 0.91,
        sigma1: float = 0.5,
        sigma2: float = 1.0,
        b: float = 1.0,
        A=None,
        Sigma=None,
        dtype=None,
    ):
        if dtype is None:
            dtype = tf.float32
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        # Build A matrix
        if A is not None:
            A_np = np.array(A, dtype=self.np_dtype)
        else:
            A_np = np.diag([a1, a2]).astype(self.np_dtype)

        # Validate stationarity: all eigenvalues of A must have |lambda| < 1
        eigvals = np.linalg.eigvals(A_np)
        max_eig = np.max(np.abs(eigvals))
        if max_eig >= 1.0:
            raise ValueError(
                f"A is not stable: max |eigenvalue| = {max_eig:.4f} >= 1. "
                f"All eigenvalues must have magnitude < 1 for stationarity."
            )

        # Build Sigma matrix
        if Sigma is not None:
            Sigma_np = np.array(Sigma, dtype=self.np_dtype)
        else:
            Sigma_np = np.diag([sigma1 ** 2, sigma2 ** 2]).astype(self.np_dtype)

        # Validate Sigma is positive definite
        eigvals_sigma = np.linalg.eigvalsh(Sigma_np)
        if np.any(eigvals_sigma <= 0):
            raise ValueError(f"Sigma must be positive definite, eigenvalues: {eigvals_sigma}")

        # Store parameters
        self.b = b
        self.a1 = float(A_np[0, 0])
        self.a2 = float(A_np[1, 1])
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # TF constants
        self._A = tf.constant(A_np, dtype=dtype)
        self._Sigma = tf.constant(Sigma_np, dtype=dtype)
        self._b = tf.constant(b, dtype=dtype)
        self._pi2 = tf.constant(2.0 * np.pi, dtype=dtype)

        # Cholesky of Sigma for sampling: e = L @ z, z ~ N(0, I)
        self._L_Sigma = tf.linalg.cholesky(self._Sigma)

        # Stationary covariance: solve discrete Lyapunov equation
        # P_0 = A P_0 A^T + Sigma
        P0_np = scipy_linalg.solve_discrete_lyapunov(A_np, Sigma_np)
        P0_np = 0.5 * (P0_np + P0_np.T)  # symmetrize
        self._P0 = tf.constant(P0_np, dtype=dtype)
        self._L_P0 = tf.linalg.cholesky(self._P0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def mu_0(self) -> tf.Tensor:
        return tf.zeros([2], dtype=self.dtype)

    @property
    def Sigma_0(self) -> tf.Tensor:
        return self._P0

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        # This is a placeholder; actual obs noise is state-dependent exp(x_2).
        # For additive-noise EKF, observation_cov(x) is used instead.
        return tf.ones([1, 1], dtype=self.dtype)

    @property
    def process_noise_cov(self) -> tf.Tensor:
        return self._Sigma

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        z = tf.random.stateless_normal([2], seed=seed, dtype=self.dtype)
        return tf.linalg.matvec(self._L_P0, z)

    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        z = tf.random.stateless_normal([2], seed=seed, dtype=self.dtype)
        return tf.linalg.matvec(self._A, x) + tf.linalg.matvec(self._L_Sigma, z)

    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        v = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        # y = b * x_1 + exp(x_2 / 2) * v
        return tf.reshape(self._b * x[0] + tf.exp(x[1] / 2.0) * v[0], [1])

    # ------------------------------------------------------------------
    # Deterministic methods (for EKF/UKF)
    # ------------------------------------------------------------------

    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        """E[x_{t+1} | x_t] = A x_t"""
        return tf.linalg.matvec(self._A, x)

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Cov[x_{t+1} | x_t] = Sigma (additive noise)"""
        return self._Sigma

    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """df/dx = A (linear transition)"""
        return self._A

    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        """E[y | x] = b * x_1"""
        return tf.reshape(self._b * x[0], [1])

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Var[y | x] = exp(x_2) (state-dependent)"""
        return tf.reshape(tf.exp(x[1]), [1, 1])

    def observation_cov_corrected(self, mean: tf.Tensor, cov: tf.Tensor) -> tf.Tensor:
        """E[exp(x_2)] = exp(m_2 + P_22/2) by lognormal MGF."""
        m2 = mean[1]
        P22 = cov[1, 1]
        return tf.reshape(tf.exp(m2 + P22 / 2.0), [1, 1])

    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """dE[y|x]/dx = [b, 0]"""
        return tf.reshape(tf.stack([self._b, tf.constant(0.0, dtype=self.dtype)]), [1, 2])

    def observation_hessian(self, x: tf.Tensor) -> tf.Tensor:
        """d^2 E[y|x]/dx^2 = 0 (observation mean is linear in x)"""
        return tf.zeros([1, 2, 2], dtype=self.dtype)

    def observation_function(self, x: tf.Tensor) -> tf.Tensor:
        """h(x) = b * x_1 (deterministic part of observation)"""
        return tf.reshape(self._b * x[0], [1])

    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """log p(y | x) = log N(y; b*x_1, exp(x_2))"""
        mean = self._b * x[0]
        var = tf.exp(x[1])
        return -0.5 * (tf.math.log(self._pi2 * var) + (y[0] - mean) ** 2 / var)

    # ------------------------------------------------------------------
    # Batch methods
    # ------------------------------------------------------------------

    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        z = tf.random.stateless_normal([n, 2], seed=seed, dtype=self.dtype)
        # (n, 2) @ (2, 2)^T = (n, 2)
        return z @ tf.transpose(self._L_P0)

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor, t=None) -> tf.Tensor:
        z = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
        # particles @ A^T + z @ L^T
        return particles @ tf.transpose(self._A) + z @ tf.transpose(self._L_Sigma)

    def state_transition_mean_batch(self, particles: tf.Tensor, t=None) -> tf.Tensor:
        return particles @ tf.transpose(self._A)

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        return self._Sigma

    def log_observation_prob_batch(self, observation: tf.Tensor,
                                   particles: tf.Tensor) -> tf.Tensor:
        """log p(y | x) for batch of particles."""
        means = self._b * particles[:, 0]               # (N,)
        vars_ = tf.exp(particles[:, 1])                  # (N,)
        diff_sq = (observation[0] - means) ** 2          # (N,)
        return -0.5 * (tf.math.log(self._pi2 * vars_) + diff_sq / vars_)

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        # H = [b, 0] for each particle, shape (N, 1, 2)
        b_col = tf.fill([N, 1], self._b)
        z_col = tf.zeros([N, 1], dtype=self.dtype)
        return tf.expand_dims(tf.concat([b_col, z_col], axis=1), axis=1)

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """h(x) = b * x_1 for batch."""
        return tf.reshape(self._b * particles[:, 0], [-1, 1])

    @property
    def has_non_additive_obs_noise(self) -> bool:
        return True

    def observation_function_with_noise(self, x: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
        """h(x, r) = b * x_1 + exp(x_2 / 2) * r_1"""
        return tf.reshape(self._b * x[0] + tf.exp(x[1] / 2.0) * r[0], [1])

    def observation_function_with_noise_batch(self, particles: tf.Tensor,
                                               noise: tf.Tensor) -> tf.Tensor:
        """Batch: h(x, r) = b * x_1 + exp(x_2 / 2) * r"""
        means = self._b * particles[:, 0]          # (N,)
        scales = tf.exp(particles[:, 1] / 2.0)     # (N,)
        y = means + scales * noise[:, 0]            # (N,)
        return tf.reshape(y, [-1, 1])               # (N, 1)

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self._A, 0), [N, 1, 1])
