"""Particle Flow Filter (Hu & van Leeuwen 2021) — TensorFlow."""

import numpy as np
import tensorflow as tf
import time
from typing import Optional, List, Callable
import warnings

from ...core.types import FilterResult


class KernelMappingPF:
    """
    Particle Flow Filter (PFF) from Hu & van Leeuwen (2021).

    Implements the matrix-valued kernel flow filter suitable for high-dimensional
    systems with sparse observations. Particles maintain equal weights throughout
    the flow, avoiding weight degeneracy.

    Reference:
        Hu, C.-C., & van Leeuwen, P. J. (2021). A particle flow filter for
        high-dimensional system applications. Quarterly Journal of the Royal
        Meteorological Society, 147(738), 2352-2374.
    """

    def __init__(
        self,
        model,
        n_particles: int = 20,
        kernel_type: str = 'matrix',
        max_iter: int = 500,
        initial_dt: float = 0.05,
        alpha: Optional[float] = None,
        localization_radius: float = 4.0,
        use_preconditioner: bool = True,
        adaptive_dt: bool = True,
        convergence_tol: float = 1e-6
    ):
        """
        Initialize Particle Flow Filter.

        Args:
            model: StateSpaceModel with TF-native methods.
            n_particles: Number of particles (N_p in paper)
            kernel_type: 'scalar' or 'matrix'
            max_iter: Maximum pseudo-time iterations (default 500 from paper)
            initial_dt: Initial pseudo-time step Δs
            alpha: Kernel bandwidth scale factor (default: 1/N_p)
            localization_radius: Decorrelation length scale r_in (Eq 29)
            use_preconditioner: Use B as preconditioner D (Section 2.4)
            adaptive_dt: Enable adaptive pseudo-time stepping (Section 2.4)
            convergence_tol: Early stopping threshold for flow magnitude
        """
        self.model = model
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.np_dtype = np.float64 if self.dtype == tf.float64 else np.float32
        self.n_particles = n_particles
        self.state_dim = model.state_dim
        self.kernel_type = kernel_type
        self.max_iter = max_iter
        self.initial_dt = initial_dt
        self.alpha = alpha
        self.localization_radius = localization_radius
        self.use_preconditioner = use_preconditioner
        self.adaptive_dt = adaptive_dt
        self.convergence_tol = convergence_tol

        # Adaptive time stepping parameters (Section 2.4)
        self.dt_increase_factor = 1.4
        self.dt_decrease_factor = 1.4
        self.consecutive_decrease_threshold = 20

        # Precompute localization matrix (Eq 28-29) — one-time numpy, stored as TF
        self.loc_matrix = self._compute_localization_matrix()

        # Particle storage (N_p × n_x) as TF tensor
        self.particles = None

        # Diagnostics
        self.last_n_iterations = 0
        self.last_final_dt = initial_dt

    def _compute_localization_matrix(self) -> tf.Tensor:
        """
        Compute distance-dependent localization matrix C (Eq 29).

        C_ij = exp(-(d(i,j) / r_in)^2)

        Assumes 1D cyclic domain (Lorenz 96 style). Override for other geometries.
        """
        indices = np.arange(self.state_dim)
        dist = np.abs(indices[:, None] - indices[None, :])
        dist = np.minimum(dist, self.state_dim - dist)
        C = np.exp(-(dist / self.localization_radius)**2)
        return tf.constant(C, dtype=self.dtype)

    def _compute_prior_stats(self):
        """
        Compute localized prior mean and covariance (Algorithm 1).

        Returns:
            x_mean: Prior mean (state_dim,) as tf.Tensor
            B: Localized prior covariance (state_dim, state_dim) as tf.Tensor
        """
        x_mean = tf.reduce_mean(self.particles, axis=0)
        X = self.particles - x_mean

        # Sample covariance
        B = tf.matmul(X, X, transpose_a=True) / tf.cast(
            self.n_particles - 1, self.dtype
        )

        # Apply Schur product localization (Eq 28)
        B = B * self.loc_matrix

        # Regularization for numerical stability
        B = B + tf.eye(self.state_dim, dtype=self.dtype) * tf.constant(1e-8, dtype=self.dtype)

        return x_mean, B

    def update(self, y: tf.Tensor) -> dict:
        """
        Update particles from prior to posterior via kernel flow.

        Args:
            y: Observation vector (obs_dim,) as tf.Tensor

        Returns:
            diagnostics: Dictionary with iteration count, final dt, etc.
        """
        if self.particles is None:
            raise ValueError("Particles not initialized.")

        if self.kernel_type == 'scalar':
            return self._update_scalar(y)
        elif self.kernel_type == 'matrix':
            return self._update_matrix(y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _update_matrix(self, y: tf.Tensor) -> dict:
        """
        Vectorized implementation of Algorithm 1 with Matrix-Valued Kernel.

        Implements Equations 20-23 for the diagonal matrix-valued kernel,
        which prevents collapse in observed dimensions (Section 2.3).
        """
        # === Initialization (Prior at pseudo-time s=0) ===
        prior_mean, B = self._compute_prior_stats()

        # Preconditioner D (usually B, Section 2.4)
        D_precond = B if self.use_preconditioner else tf.eye(
            self.state_dim, dtype=self.dtype
        )

        # Kernel bandwidth diagonal (Eq 21): σ^(d)^2
        B_diag = tf.linalg.diag_part(B)
        B_diag = tf.maximum(B_diag, tf.constant(1e-8, dtype=B_diag.dtype))

        # Bandwidth scale α (Paper recommends 1/N_p for matrix kernel)
        alpha = self.alpha if self.alpha is not None else (1.0 / self.n_particles)

        # Precompute inverses
        B_inv = tf.linalg.pinv(B)
        R_inv = tf.linalg.inv(self.model.observation_noise_cov)

        # Adaptive time stepping initialization
        dt = self.initial_dt
        consecutive_decreases = 0
        prev_flow_magnitude = float('inf')

        sqrt_state_dim = tf.sqrt(tf.cast(self.state_dim, self.dtype))

        # === Pseudo-time Flow Iteration ===
        for iteration in range(self.max_iter):

            # --- Step 1: Compute ∇log p(x_s|y) for all particles (Eq 11) ---
            # Vectorized observation function and Jacobian
            h_particles = self.model.observation_function_batch(self.particles)  # (N, obs_dim)
            H_batch = self.model.observation_jacobian_batch(self.particles)  # (N, obs_dim, state_dim)
            innovations = y - h_particles  # (N, obs_dim)

            # Likelihood gradient (Eq 13): H^T @ R^{-1} @ (y - h(x))
            # grad_lik[n, s] = sum_o sum_j H[n,o,s] * R_inv[o,j] * innov[n,j]
            grad_lik = tf.einsum('nos,oj,nj->ns', H_batch, R_inv, innovations)

            # Prior gradient (Eq 15): -B^{-1}(x - x_0)
            diff_from_mean = self.particles - prior_mean  # (N, state_dim)
            grad_prior = -tf.linalg.matvec(B_inv, diff_from_mean)  # (N, state_dim)

            grad_log_post = grad_lik + grad_prior  # (N, state_dim)

            # --- Step 2: Compute Matrix-Valued Kernel (Eq 20-21) ---
            # diff[i,j,d] = x_i^(d) - x_j^(d)
            diff = (self.particles[:, tf.newaxis, :]
                    - self.particles[tf.newaxis, :, :])  # (N, N, state_dim)

            # Kernel scale: α * σ^(d)^2, broadcast to (1, 1, state_dim)
            scale = tf.reshape(alpha * B_diag, [1, 1, -1])

            # K^(d)(x_i, x_j) = exp(-0.5 * (x_i^(d) - x_j^(d))^2 / (α*σ^(d)^2))
            K = tf.exp(-0.5 * (diff ** 2) / scale)  # (N, N, state_dim)

            # --- Step 3: Compute Kernel Divergence (Eq 23) ---
            # ∇_{x_j} K^(d)(x_j, x_i) = +(x_i - x_j)/(α*σ^2) * K
            grad_K = (diff / scale) * K  # (N, N, state_dim)

            # --- Step 4: Weighted Gradient Term (Eq 6) ---
            # Sum over SOURCE particles j: K(x_i, x_j) * grad(x_j)
            term1 = tf.einsum('ijd,jd->id', K, grad_log_post)  # (N, state_dim)

            # --- Step 5: Integrate and Average (Eq 6) ---
            grad_K_sum = tf.reduce_sum(grad_K, axis=1)  # (N, state_dim)
            I_f = (term1 + grad_K_sum) / tf.cast(self.n_particles, self.dtype)

            # --- Step 6: Apply Preconditioner (Eq 7) ---
            # f_s = D * I_f → for each particle: D @ I_f[i]
            flow = tf.linalg.matvec(D_precond, I_f)  # (N, state_dim)

            # --- Step 7: Update Particles ---
            particles_new = self.particles + dt * flow

            # --- Step 8: Adaptive Time Stepping (Section 2.4) ---
            flow_magnitude = float(
                tf.reduce_mean(tf.norm(flow, axis=1)) / sqrt_state_dim
            )

            if self.adaptive_dt:
                if flow_magnitude < prev_flow_magnitude:
                    consecutive_decreases += 1
                    if consecutive_decreases >= self.consecutive_decrease_threshold:
                        dt *= self.dt_increase_factor
                        consecutive_decreases = 0
                else:
                    dt /= self.dt_decrease_factor
                    consecutive_decreases = 0

                prev_flow_magnitude = flow_magnitude

            # Update particles
            self.particles = particles_new

            # --- Early Stopping ---
            if flow_magnitude < self.convergence_tol:
                self.last_n_iterations = iteration + 1
                self.last_final_dt = dt
                break
        else:
            # Max iterations reached
            self.last_n_iterations = self.max_iter
            self.last_final_dt = dt
            warnings.warn(f"PFF did not converge in {self.max_iter} iterations. "
                         f"Final flow magnitude: {flow_magnitude:.2e}")

        return {
            'n_iterations': self.last_n_iterations,
            'final_dt': self.last_final_dt,
            'final_flow_magnitude': flow_magnitude
        }

    def _update_scalar(self, y: tf.Tensor) -> dict:
        """
        Vectorized implementation for Scalar Kernel (Eq 16-19).

        Uses isotropic Gaussian kernel with Mahalanobis distance.
        Less suitable for high-dim sparse observations (Section 2.3).
        """
        prior_mean, B = self._compute_prior_stats()
        D_precond = B if self.use_preconditioner else tf.eye(
            self.state_dim, dtype=self.dtype
        )

        alpha = self.alpha if self.alpha is not None else (1.0 / self.n_particles)

        # A = (αB)^{-1} for Mahalanobis distance (Eq 18)
        A = tf.linalg.pinv(alpha * B) / tf.cast(self.state_dim, self.dtype)
        B_inv = tf.linalg.pinv(B)
        R_inv = tf.linalg.inv(self.model.observation_noise_cov)

        dt = self.initial_dt
        consecutive_decreases = 0
        prev_flow_magnitude = float('inf')

        sqrt_state_dim = tf.sqrt(tf.cast(self.state_dim, self.dtype))

        for iteration in range(self.max_iter):
            # Compute gradients — vectorized
            h_particles = self.model.observation_function_batch(self.particles)
            H_batch = self.model.observation_jacobian_batch(self.particles)
            innovations = y - h_particles

            grad_lik = tf.einsum('nos,oj,nj->ns', H_batch, R_inv, innovations)
            diff_from_mean = self.particles - prior_mean
            grad_prior = -tf.linalg.matvec(B_inv, diff_from_mean)
            grad_log_post = grad_lik + grad_prior

            # Compute scalar kernel
            diff = (self.particles[:, tf.newaxis, :]
                    - self.particles[tf.newaxis, :, :])  # (N, N, state_dim)

            # Mahalanobis distance: (x_i - x_j)^T A (x_i - x_j)
            # Adiff[i,j,:] = diff[i,j,:] @ A^T = diff[i,j,:] @ A (symmetric)
            Adiff = tf.einsum('ijs,sd->ijd', diff, A)  # (N, N, state_dim)
            mahal = tf.reduce_sum(Adiff * diff, axis=2)  # (N, N)

            # K(x_i, x_j) = exp(-0.5 * mahal)
            K = tf.exp(-0.5 * mahal)  # (N, N)

            # Divergence (Eq 19): ∇_{x_j} K = A(x_i - x_j) K
            div_K = Adiff * K[:, :, tf.newaxis]  # (N, N, state_dim)

            # Gradient term: K(x_i, x_j) * grad(x_j) summed over j
            term1 = tf.matmul(K, grad_log_post)  # (N,N) @ (N,state_dim) -> (N,state_dim)

            # Integrate and apply preconditioner
            div_K_sum = tf.reduce_sum(div_K, axis=1)  # (N, state_dim)
            I_f = (term1 + div_K_sum) / tf.cast(self.n_particles, self.dtype)
            flow = tf.linalg.matvec(D_precond, I_f)  # (N, state_dim)

            # Update particles
            particles_new = self.particles + dt * flow

            # Adaptive time stepping
            flow_magnitude = float(
                tf.reduce_mean(tf.norm(flow, axis=1)) / sqrt_state_dim
            )

            if self.adaptive_dt:
                if flow_magnitude < prev_flow_magnitude:
                    consecutive_decreases += 1
                    if consecutive_decreases >= self.consecutive_decrease_threshold:
                        dt *= self.dt_increase_factor
                        consecutive_decreases = 0
                else:
                    dt /= self.dt_decrease_factor
                    consecutive_decreases = 0

                prev_flow_magnitude = flow_magnitude

            self.particles = particles_new

            # Early stopping
            if flow_magnitude < self.convergence_tol:
                self.last_n_iterations = iteration + 1
                self.last_final_dt = dt
                break
        else:
            self.last_n_iterations = self.max_iter
            self.last_final_dt = dt
            warnings.warn(f"PFF did not converge in {self.max_iter} iterations. "
                         f"Final flow magnitude: {flow_magnitude:.2e}")

        return {
            'n_iterations': self.last_n_iterations,
            'final_dt': self.last_final_dt,
            'final_flow_magnitude': flow_magnitude
        }

    def get_mean(self) -> np.ndarray:
        """Get ensemble mean as numpy array."""
        return tf.reduce_mean(self.particles, axis=0).numpy()

    def get_covariance(self) -> np.ndarray:
        """Get ensemble covariance as numpy array."""
        X = self.particles - tf.reduce_mean(self.particles, axis=0)
        cov = tf.matmul(X, X, transpose_a=True) / tf.cast(
            self.n_particles - 1, self.dtype
        )
        return cov.numpy()

    def initialize(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize particles from model's initial distribution.

        Args:
            rng: Random number generator (used only for seed generation)
        """
        if rng is None:
            rng = np.random.default_rng()

        seed = tf.constant(rng.integers(0, 2**31, size=2), dtype=tf.int32)
        self.particles = self.model.sample_initial_state_batch(
            self.n_particles, seed
        )

        # Reset diagnostics
        self.means_history: List[np.ndarray] = []
        self.covs_history: List[np.ndarray] = []
        self.iteration_counts: List[int] = []

    def predict(self, rng: Optional[np.random.Generator] = None):
        """
        Propagate particles through dynamics (forecast step).

        Args:
            rng: Random number generator for stochastic dynamics
        """
        if rng is None:
            rng = np.random.default_rng()

        seed = tf.constant(rng.integers(0, 2**31, size=2), dtype=tf.int32)
        self.particles = self.model.state_transition_batch(self.particles, seed)

    def filter(self, observations: np.ndarray,
               initial_particles: Optional[np.ndarray] = None,
               n_integration_steps: int = 1,
               progress_callback: Optional[Callable[[int, int, float], None]] = None) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Follows the predict-update cycle for each observation:
        1. Predict: Propagate particles through dynamics
        2. Update: Apply kernel flow to move prior -> posterior

        Args:
            observations: Array of shape (T, obs_dim) containing observations
            initial_particles: Optional initial particle ensemble (n_particles, state_dim)
            n_integration_steps: Number of dynamics steps between observations
            progress_callback: Optional callback(t, T, step_time_sec)

        Returns:
            FilterResult with filtered means, covariances, and metadata
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        T = len(observations)
        rng = np.random.default_rng()

        # Initialize particles
        if initial_particles is not None:
            self.particles = tf.constant(initial_particles, dtype=self.dtype)
        else:
            self.initialize(rng)

        # Storage
        means: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        iteration_counts: List[int] = []
        prior_particles: List[np.ndarray] = []
        posterior_particles: List[np.ndarray] = []

        # Filter loop: predict first, then update
        for t in range(T):
            t0 = time.perf_counter()
            # Predict: propagate through dynamics
            for _ in range(n_integration_steps):
                self.predict(rng)

            # Store prior particles (before update)
            prior_particles.append(self.particles.numpy())

            # Update: kernel flow from prior to posterior
            y_tf = tf.constant(observations[t], dtype=self.dtype)
            diagnostics = self.update(y_tf)
            iteration_counts.append(diagnostics['n_iterations'])

            # Store posterior particles (after update)
            posterior_particles.append(self.particles.numpy())

            # Record posterior estimates
            means.append(self.get_mean())
            covs.append(self.get_covariance())
            if progress_callback is not None:
                progress_callback(t, T, time.perf_counter() - t0)

        return FilterResult(
            means=np.array(means),
            covs=np.array(covs),
            log_likelihood=None,
            metadata={
                'filter_type': 'KernelMappingPF',
                'kernel_type': self.kernel_type,
                'n_particles': self.n_particles,
                'mean_iterations': float(np.mean(iteration_counts)),
                'total_iterations': int(np.sum(iteration_counts)),
                'n_integration_steps': n_integration_steps,
                'prior_particles': np.array(prior_particles),
                'posterior_particles': np.array(posterior_particles)
            }
        )
