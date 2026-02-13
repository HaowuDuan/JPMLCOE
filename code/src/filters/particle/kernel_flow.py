"""Particle Flow Filter (Hu & van Leeuwen 2021)."""

import numpy as np
import tensorflow as tf
from scipy import linalg
from typing import Optional, Tuple, List
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
            model: StateSpaceModel with attributes:
                - state_dim: Dimension of state space
                - observation_noise_cov: Observation error covariance R
                - observe(x): Observation function h(x)
                - observation_jacobian(x): Jacobian H(x) = dh/dx
            n_particles: Number of particles (N_p in paper)
            kernel_type: 'scalar' or 'matrix'
                - 'matrix': Recommended for high-dim, sparse observations (Eq 20-23)
                - 'scalar': For low-dim or dense observations (Eq 16-19)
            max_iter: Maximum pseudo-time iterations (default 500 from paper)
            initial_dt: Initial pseudo-time step Δs
                - Linear obs: 0.05 (paper default)
                - Nonlinear obs: 0.001 (paper recommendation)
            alpha: Kernel bandwidth scale factor
                - None (default): Auto-set to 1/N_p (paper recommendation)
            localization_radius: Decorrelation length scale r_in (Eq 29)
                - Default: 4 (paper's Lorenz 96 setting)
            use_preconditioner: Use B as preconditioner D (Section 2.4)
            adaptive_dt: Enable adaptive pseudo-time stepping (Section 2.4)
            convergence_tol: Early stopping threshold for flow magnitude
        """
        self.model = model
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

        # Precompute localization matrix (Eq 28-29)
        self.loc_matrix = self._compute_localization_matrix()

        # Particle storage (N_p × n_x)
        self.particles = None

        # Diagnostics
        self.last_n_iterations = 0
        self.last_final_dt = initial_dt

    def _compute_localization_matrix(self) -> np.ndarray:
        """
        Compute distance-dependent localization matrix C (Eq 29).

        C_ij = exp(-(d(i,j) / r_in)^2)

        Assumes 1D cyclic domain (Lorenz 96 style). Override for other geometries.
        """
        indices = np.arange(self.state_dim)
        dist = np.abs(indices[:, None] - indices[None, :])

        # Periodic boundary
        dist = np.minimum(dist, self.state_dim - dist)

        # Gaussian localization (Eq 29)
        C = np.exp(-(dist / self.localization_radius)**2)
        return C

    def _compute_prior_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute localized prior mean and covariance (Algorithm 1).

        Returns:
            x_mean: Prior mean (n_x,)
            B: Localized prior covariance (n_x, n_x)
        """
        x_mean = np.mean(self.particles, axis=0)
        X = self.particles - x_mean

        # Sample covariance
        B = (X.T @ X) / (self.n_particles - 1)

        # Apply Schur product localization (Eq 28)
        B = B * self.loc_matrix

        # Regularization for numerical stability
        B += np.eye(self.state_dim) * 1e-8

        return x_mean, B

    def update(self, y: np.ndarray) -> dict:
        """
        Update particles from prior to posterior via kernel flow.

        Args:
            y: Observation vector (obs_dim,)

        Returns:
            diagnostics: Dictionary with iteration count, final dt, etc.
        """
        if self.particles is None:
            raise ValueError("Particles not initialized. Set self.particles before calling update().")

        if self.kernel_type == 'scalar':
            return self._update_scalar(y)
        elif self.kernel_type == 'matrix':
            return self._update_matrix(y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _update_matrix(self, y: np.ndarray) -> dict:
        """
        Vectorized implementation of Algorithm 1 with Matrix-Valued Kernel.

        Implements Equations 20-23 for the diagonal matrix-valued kernel,
        which prevents collapse in observed dimensions (Section 2.3).
        """
        # === Initialization (Prior at pseudo-time s=0) ===
        prior_mean, B = self._compute_prior_stats()

        # Preconditioner D (usually B, Section 2.4)
        D_precond = B if self.use_preconditioner else np.eye(self.state_dim)

        # Kernel bandwidth diagonal (Eq 21): σ^(d)^2
        B_diag = np.diag(B)
        B_diag = np.maximum(B_diag, 1e-8)  # Avoid division by zero

        # Bandwidth scale α (Paper recommends 1/N_p for matrix kernel)
        alpha = self.alpha if self.alpha is not None else (1.0 / self.n_particles)

        # Precompute inverses
        B_inv = linalg.pinv(B)
        R_inv = linalg.inv(self.model.observation_noise_cov)

        # Adaptive time stepping initialization
        dt = self.initial_dt
        consecutive_decreases = 0
        prev_flow_magnitude = np.inf

        # === Pseudo-time Flow Iteration ===
        for iteration in range(self.max_iter):

            # --- Step 1: Compute ∇log p(x_s|y) for all particles (Eq 11) ---
            h_particles = np.array([self.model.observe(p) for p in self.particles])
            grad_log_post = np.zeros((self.n_particles, self.state_dim))

            for i in range(self.n_particles):
                # Likelihood gradient (Eq 13): H^T R^{-1} (y - h(x))
                H_i = self.model.observation_jacobian(self.particles[i])
                innovation = y - h_particles[i]
                grad_lik = H_i.T @ R_inv @ innovation

                # Prior gradient (Eq 15): -B^{-1}(x - x_0)
                grad_prior = -B_inv @ (self.particles[i] - prior_mean)

                grad_log_post[i] = grad_lik + grad_prior

            # --- Step 2: Compute Matrix-Valued Kernel (Eq 20-21) ---
            # Shape convention: [target_i, source_j, dimension_d]
            # diff[i,j,d] = x_i^(d) - x_j^(d)
            diff = self.particles[:, np.newaxis, :] - self.particles[np.newaxis, :, :]

            # Kernel scale: α * σ^(d)^2, broadcast to (1, 1, n_x)
            scale = (alpha * B_diag).reshape(1, 1, -1)

            # K^(d)(x_i, x_j) = exp(-0.5 * (x_i^(d) - x_j^(d))^2 / (α*σ^(d)^2))
            K = np.exp(-0.5 * (diff**2) / scale)

            # --- Step 3: Compute Kernel Divergence (Eq 23) ---
            # ∇_{x_j} K^(d)(x_j, x_i) = +(x_i - x_j)/(α*σ^2) * K
            # Repulsive force (positive sign)
            grad_K = (diff / scale) * K

            # --- Step 4: Weighted Gradient Term (Eq 6) ---
            # Sum over SOURCE particles j: K(x_i, x_j) * grad(x_j)
            # For matrix kernel: K[i,j,d] * grad[j,d] summed over j
            # Use einsum for clarity: sum over source index j
            term1 = np.einsum('ijd,jd->id', K, grad_log_post)  # (N, n_x)

            # --- Step 5: Integrate and Average (Eq 6) ---
            # I_f = (1/N_p) * Σ_j [K * ∇log p + ∇·K]
            grad_K_sum = np.sum(grad_K, axis=1)  # (N, n_x)
            I_f = (term1 + grad_K_sum) / self.n_particles

            # --- Step 6: Apply Preconditioner (Eq 7) ---
            # f_s = D * I_f
            flow = (D_precond @ I_f.T).T  # (N, n_x)

            # --- Step 7: Update Particles ---
            # x_{s+Δs} = x_s + Δs * f_s
            particles_new = self.particles + dt * flow

            # --- Step 8: Adaptive Time Stepping (Section 2.4) ---
            # Normalize by sqrt(state_dim) for dimension-independent convergence
            flow_magnitude = np.mean(linalg.norm(flow, axis=1)) / np.sqrt(self.state_dim)

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

    def _update_scalar(self, y: np.ndarray) -> dict:
        """
        Vectorized implementation for Scalar Kernel (Eq 16-19).

        Uses isotropic Gaussian kernel with Mahalanobis distance.
        Less suitable for high-dim sparse observations (Section 2.3).
        """
        prior_mean, B = self._compute_prior_stats()
        D_precond = B if self.use_preconditioner else np.eye(self.state_dim)

        alpha = self.alpha if self.alpha is not None else (1.0 / self.n_particles)

        # A = (αB)^{-1} for Mahalanobis distance (Eq 18)
        A = linalg.pinv(alpha * B)/ self.state_dim 
        B_inv = linalg.pinv(B)
        R_inv = linalg.inv(self.model.observation_noise_cov)

        dt = self.initial_dt
        consecutive_decreases = 0
        prev_flow_magnitude = np.inf

        for iteration in range(self.max_iter):
            # Compute gradients
            h_particles = np.array([self.model.observe(p) for p in self.particles])
            grad_log_post = np.zeros((self.n_particles, self.state_dim))

            for i in range(self.n_particles):
                H_i = self.model.observation_jacobian(self.particles[i])
                innovation = y - h_particles[i]
                grad_log_post[i] = (H_i.T @ R_inv @ innovation) - \
                                   (B_inv @ (self.particles[i] - prior_mean))

            # Compute scalar kernel
            diff = self.particles[:, np.newaxis, :] - self.particles[np.newaxis, :, :]
            diff_flat = diff.reshape(-1, self.state_dim)

            # Mahalanobis distance: (x_i - x_j)^T A (x_i - x_j)
            mahal = np.sum((diff_flat @ A) * diff_flat, axis=1)
            mahal = mahal.reshape(self.n_particles, self.n_particles)

            # K(x_i, x_j) = exp(-0.5 * mahal)
            K = np.exp(-0.5 * mahal)  # (N, N)

            # Divergence (Eq 19): ∇_{x_j} K = A(x_i - x_j) K
            # Repulsive force (positive sign)
            Adiff = (A @ diff_flat.T).T.reshape(self.n_particles, self.n_particles, self.state_dim)
            div_K = Adiff * K[:, :, np.newaxis]  # (N, N, n_x)

            # Gradient term: Sum over SOURCE particles j
            # K(x_i, x_j) * grad(x_j) summed over j
            # K is (N, N), grad_log_post is (N, n_x)
            # Result: (N, n_x) for each target particle i
            term1 = K @ grad_log_post  # Matrix multiply: (N,N) @ (N,n_x) -> (N,n_x)

            # Integrate and apply preconditioner
            div_K_sum = np.sum(div_K, axis=1)  # (N, n_x)
            I_f = (term1 + div_K_sum) / self.n_particles
            flow = (D_precond @ I_f.T).T

            # Update particles
            particles_new = self.particles + dt * flow

            # Adaptive time stepping
            # Normalize by sqrt(state_dim) for dimension-independent convergence
            flow_magnitude = np.mean(linalg.norm(flow, axis=1)) / np.sqrt(self.state_dim)

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
        """Get ensemble mean (may not be representative for multimodal posteriors)."""
        return np.mean(self.particles, axis=0)

    def get_covariance(self) -> np.ndarray:
        """Get ensemble covariance."""
        X = self.particles - self.get_mean()
        return (X.T @ X) / (self.n_particles - 1)

    def initialize(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize particles from model's initial distribution.
        
        Args:
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        def _make_seed():
            return tf.constant(rng.integers(0, 2**31, size=2), dtype=tf.int32)
        self.particles = np.array([
            np.asarray(self.model.sample_initial_state(_make_seed()))
            for _ in range(self.n_particles)
        ])
        
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
            
        for i in range(self.n_particles):
            seed = tf.constant(rng.integers(0, 2**31, size=2), dtype=tf.int32)
            x = tf.constant(self.particles[i], dtype=tf.float32)
            self.particles[i] = np.asarray(self.model.sample_state_transition(x, seed))
    
    def filter(self, observations: np.ndarray,
               initial_particles: Optional[np.ndarray] = None,
               n_integration_steps: int = 1) -> FilterResult:
        """
        Run the filter on a sequence of observations.
        
        Follows the predict-update cycle for each observation:
        1. Predict: Propagate particles through dynamics
        2. Update: Apply kernel flow to move prior -> posterior
        
        Args:
            observations: Array of shape (T, obs_dim) containing observations
            initial_particles: Optional initial particle ensemble (n_particles, state_dim)
            n_integration_steps: Number of dynamics steps between observations
                                (default 1, set to 20 for paper's Δt_obs = 1.0)
        
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
            self.particles = initial_particles.copy()
        else:
            self.initialize(rng)
        
        # Storage
        means: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        iteration_counts: List[int] = []
        prior_particles: List[np.ndarray] = []
        posterior_particles: List[np.ndarray] = []
        
        # Filter loop: predict first, then update
        # Note: Despite predict-first loop, this filter uses observe_initial=True
        # (standard alignment where observations[0] observes the initial state)
        for t in range(T):
            # Predict: propagate through dynamics
            for _ in range(n_integration_steps):
                self.predict(rng)
            
            # Store prior particles (before update)
            prior_particles.append(self.particles.copy())
            
            # Update: kernel flow from prior to posterior
            diagnostics = self.update(observations[t])
            iteration_counts.append(diagnostics['n_iterations'])
            
            # Store posterior particles (after update)
            posterior_particles.append(self.particles.copy())
            
            # Record posterior estimates
            means.append(self.get_mean())
            covs.append(self.get_covariance())
        
        return FilterResult(
            means=np.array(means),
            covs=np.array(covs),
            log_likelihood=None,  # Kernel flow doesn't compute marginal likelihood
            metadata={
                'filter_type': 'KernelMappingPF',
                'kernel_type': self.kernel_type,
                'n_particles': self.n_particles,
                'mean_iterations': float(np.mean(iteration_counts)),
                'total_iterations': int(np.sum(iteration_counts)),
                'n_integration_steps': n_integration_steps,
                'prior_particles': np.array(prior_particles),      # (T, N, state_dim)
                'posterior_particles': np.array(posterior_particles)  # (T, N, state_dim)
            }
        )
