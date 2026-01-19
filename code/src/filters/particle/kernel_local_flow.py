"""Kernel-based Local Flow particle filter."""

import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from .edh_flow import ExactDaumHuangFlow


class KernelLocalFlow(ExactDaumHuangFlow):
    """
    Kernel-based Local Flow filter - computes gain locally around each particle.

    Uses kernel density estimation to compute local covariances,
    better for multimodal or complex posteriors.

    This is NOT the Algorithm 2 from Ding & Coates (2013), but rather
    a KDE-based local flow approach using kernel-weighted covariances.
    """

    def __init__(self, model, n_particles: int = 1000,
                 n_lambda_steps: int = 100, integration_method: str = 'euler',
                 kernel_bandwidth: Optional[float] = None,
                 kernel_type: str = 'gaussian',
                 n_threads: Optional[int] = None):
        """
        Args:
            kernel_bandwidth: Bandwidth for kernel density estimation.
                            If None, use Silverman's rule: σ * N^{-1/(d+4)}
            kernel_type: 'gaussian' or 'epanechnikov'
            n_threads: Number of threads for parallelization
        """
        super().__init__(model, n_particles, n_lambda_steps, integration_method, n_threads)
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_type = kernel_type

    def _gaussian_kernel(self, distance: float, bandwidth: float) -> float:
        """Gaussian kernel: exp(-0.5 * (d/h)^2)"""
        return np.exp(-0.5 * (distance / bandwidth)**2)

    def _epanechnikov_kernel(self, distance: float, bandwidth: float) -> float:
        """Epanechnikov kernel: 0.75 * (1 - (d/h)^2) if d < h, else 0"""
        u = distance / bandwidth
        return 0.75 * (1 - u**2) if u < 1 else 0.0

    def _compute_kernel_weights(self, particle_i: np.ndarray,
                               all_particles: np.ndarray,
                               bandwidth: float) -> np.ndarray:
        """
        Compute kernel weights for particle i relative to all particles.

        Args:
            particle_i: Single particle, shape (state_dim,)
            all_particles: All particles, shape (N, state_dim)
            bandwidth: Kernel bandwidth

        Returns:
            weights: Normalized weights, shape (N,), sum to 1
        """
        # Compute distances
        distances = np.linalg.norm(all_particles - particle_i, axis=1)  # (N,)

        # Apply kernel
        if self.kernel_type == 'gaussian':
            weights = np.array([self._gaussian_kernel(d, bandwidth) for d in distances])
        elif self.kernel_type == 'epanechnikov':
            weights = np.array([self._epanechnikov_kernel(d, bandwidth) for d in distances])
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Normalize
        weights = weights / np.sum(weights)
        return weights

    def _compute_local_gain(self, particle_idx: int, particles: np.ndarray,
                           h_particles: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Compute local gain K_i for particle i using kernel-weighted covariances.

        Args:
            particle_idx: Index of particle to compute gain for
            particles: All particles, shape (N, state_dim)
            h_particles: h(x) evaluated at all particles, shape (N, obs_dim)
            bandwidth: Kernel bandwidth

        Returns:
            K_i: Local gain for particle i, shape (state_dim, obs_dim)
        """
        # Compute kernel weights for this particle
        weights = self._compute_kernel_weights(particles[particle_idx], particles, bandwidth)

        # Weighted means
        x_mean = np.sum(weights[:, np.newaxis] * particles, axis=0)  # (state_dim,)
        h_mean = np.sum(weights[:, np.newaxis] * h_particles, axis=0)  # (obs_dim,)

        # Center
        x_centered = particles - x_mean
        h_centered = h_particles - h_mean

        # Weighted covariances
        # Cov[x, h(x)] = Σ w_j (x_j - x_mean)(h_j - h_mean)^T
        cov_x_h = (x_centered.T * weights) @ h_centered  # (state_dim, obs_dim)
        cov_h_h = (h_centered.T * weights) @ h_centered  # (obs_dim, obs_dim)

        # Add observation noise
        R = self.model.observation_noise_cov
        S = cov_h_h + R

        # Compute local gain
        try:
            K_i = cov_x_h @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K_i = cov_x_h @ np.linalg.pinv(S)

        return K_i

    def _flow_step_euler(self, particles: np.ndarray, y: np.ndarray,
                        lambda_val: float, d_lambda: float) -> np.ndarray:
        """
        Local flow step - each particle uses its own local gain.
        """
        # Determine bandwidth (Silverman's rule if not specified)
        if self.kernel_bandwidth is None:
            particle_std = np.std(particles, axis=0).mean()
            bandwidth = particle_std * (self.n_particles ** (-1.0 / (self.state_dim + 4)))
        else:
            bandwidth = self.kernel_bandwidth

        # Evaluate h(x) once for all particles
        h_particles = self._compute_observation_matrix(particles)

        # Update each particle using its local gain (parallelized if n_threads > 1)
        particles_new = particles.copy()

        if self.n_threads > 1:
            def update_particle(i):
                # Compute local gain for particle i
                K_i = self._compute_local_gain(i, particles, h_particles, bandwidth)

                # Innovation for particle i
                innovation = y - h_particles[i]  # (obs_dim,)

                # Flow update
                dx = K_i @ innovation  # (state_dim,)
                return particles[i] + dx * d_lambda

            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                particles_new = np.array(list(executor.map(update_particle, range(self.n_particles))))
        else:
            for i in range(self.n_particles):
                # Compute local gain for particle i
                K_i = self._compute_local_gain(i, particles, h_particles, bandwidth)

                # Innovation for particle i
                innovation = y - h_particles[i]  # (obs_dim,)

                # Flow update
                dx = K_i @ innovation  # (state_dim,)
                particles_new[i] = particles[i] + dx * d_lambda

        return particles_new

    def _flow_step_rk4(self, particles: np.ndarray, y: np.ndarray,
                       lambda_val: float, d_lambda: float) -> np.ndarray:
        """RK4 with local gains - significantly more expensive."""
        # Would require 4 evaluations of local gains per particle
        # Fall back to Euler for now
        return self._flow_step_euler(particles, y, lambda_val, d_lambda)
