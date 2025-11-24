import numpy as np
from typing import Optional
from filters import ExactDaumHuangFlow


class KernelMappingPF(ExactDaumHuangFlow):
    """
    Kernel Mapping Particle Filter (Pulido & van Leeuwen 2019).
    
    Moves particles from prior to posterior via kernel-embedded gradient flow,
    minimizing KL divergence. No resampling needed - particles maintain equal weights.
    """
    
    def __init__(self, model, n_particles: int = 1000,
                 kernel_type: str = 'scalar',
                 max_iter: int = 50,
                 epsilon: float = 0.01,
                 alpha: Optional[float] = None,
                 n_threads: Optional[int] = None):
        """
        Initialize Kernel Mapping Particle Filter.
        
        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            kernel_type: 'scalar' or 'matrix' kernel
            max_iter: Number of flow iterations
            epsilon: Step size for gradient descent
            alpha: Kernel bandwidth (None = auto: state_dim for scalar, 1/N_p for matrix)
            n_threads: Number of threads for parallelization
        """
        # Initialize parent but ignore n_lambda_steps (not used in kernel MPF)
        super().__init__(model, n_particles, n_lambda_steps=0, n_threads=n_threads)
        self.kernel_type = kernel_type
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.alpha = alpha
    
    def update(self, y: np.ndarray):
        """
        Update step: move particles from prior to posterior via kernel flow.
        
        NO RESAMPLING - particles maintain equal weight 1/N throughout.
        
        Args:
            y: Observation, shape (obs_dim,)
        """
        if self.kernel_type == 'scalar':
            self._update_scalar(y)
        elif self.kernel_type == 'matrix':
            self._update_matrix(y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _compute_grad_log_posterior(self, particle_idx: int, y: np.ndarray,
                                   h_particles: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log posterior at particle j.
        
        ∇ log π(x) = ∇ log p(y|x) + ∇ log p(x|y_{1:t-1})
        
        Args:
            particle_idx: Index of particle
            y: Observation
            h_particles: Observation function evaluated at all particles, shape (N, obs_dim)
            
        Returns:
            grad: Gradient vector, shape (state_dim,)
        """
        x_j = self.particles[particle_idx]
        
        # Likelihood gradient: H^T R^{-1} (y - h(x))
        innovation = y - h_particles[particle_idx]
        H_j = self.model.observation_jacobian(x_j)
        grad_likelihood = H_j.T @ np.linalg.solve(self.model.observation_noise_cov, innovation)
        
        # Prior gradient: -Q^{-1}(x - x_mean)
        # Approximate prior p(x|y_{1:t-1}) by ensemble mean
        x_mean = np.mean(self.particles, axis=0)
        Q = self.model.process_noise_cov
        # Handle potentially singular Q with pseudoinverse
        try:
            grad_prior = -np.linalg.solve(Q, x_j - x_mean)
        except np.linalg.LinAlgError:
            # If Q is singular, use pseudoinverse
            grad_prior = -np.linalg.pinv(Q) @ (x_j - x_mean)
        
        return grad_likelihood + grad_prior
    
    def _update_scalar(self, y: np.ndarray):
        """
        Update with scalar (isotropic) kernel.
        
        K(x,x') = exp(-0.5 * ||x-x'||^2_A)
        where A = alpha * Q
        """
        # Set bandwidth
        alpha = self.alpha if self.alpha is not None else self.state_dim
        
        # Kernel bandwidth matrix: A = alpha * Q
        A = alpha * self.model.process_noise_cov
        # Add small regularization to ensure invertibility
        A_reg = A + np.eye(self.state_dim) * 1e-8
        try:
            A_inv = np.linalg.inv(A_reg)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if still singular
            A_inv = np.linalg.pinv(A_reg)
        
        for iteration in range(self.max_iter):
            # Compute h(x) for all particles
            h_particles = self._compute_observation_matrix(self.particles)
            
            # Compute gradient of log posterior for each particle
            grad_log_post = np.array([
                self._compute_grad_log_posterior(j, y, h_particles)
                for j in range(self.n_particles)
            ])
            
            # Update each particle
            particles_new = np.zeros_like(self.particles)
            
            for j in range(self.n_particles):
                velocity = np.zeros(self.state_dim)
                
                for i in range(self.n_particles):
                    # Distance
                    diff = self.particles[i] - self.particles[j]
                    
                    # Scalar kernel: K(x_i, x_j) = exp(-0.5 * diff^T A^{-1} diff)
                    exponent = -0.5 * (diff @ A_inv @ diff)
                    K_ij = np.exp(exponent)
                    
                    # Kernel gradient: ∇_x_i K(x_i, x_j) = -A^{-1} (x_i - x_j) K(x_i, x_j)
                    grad_K = -A_inv @ diff * K_ij
                    
                    # Velocity contribution: K * ∇log π + ∇K
                    velocity += K_ij * grad_log_post[i] + grad_K
                
                particles_new[j] = self.particles[j] + (self.epsilon / self.n_particles) * velocity
            
            self.particles = particles_new
    
    def _update_matrix(self, y: np.ndarray):
        """
        Update with matrix-valued (diagonal) kernel.
        
        K(x,x') = diag[K^(1)(x,x'), ..., K^(n_x)(x,x')]
        where K^(d)(x,x') = exp(-(x^(d) - x'^(d))^2 / (2*alpha*sigma^(d)^2))
        """
        # Set bandwidth
        alpha = self.alpha if self.alpha is not None else (1.0 / self.n_particles)
        
        # Component-wise standard deviations from prior
        sigma = np.std(self.particles, axis=0)  # (state_dim,)
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-8)
        
        for iteration in range(self.max_iter):
            # Compute h(x) for all particles
            h_particles = self._compute_observation_matrix(self.particles)
            
            # Compute gradient of log posterior for each particle
            grad_log_post = np.array([
                self._compute_grad_log_posterior(j, y, h_particles)
                for j in range(self.n_particles)
            ])
            
            # Update each particle
            particles_new = np.zeros_like(self.particles)
            
            for j in range(self.n_particles):
                velocity = np.zeros(self.state_dim)
                
                for i in range(self.n_particles):
                    # Component-wise distances
                    diff = self.particles[i] - self.particles[j]  # (state_dim,)
                    
                    # Matrix-valued kernel: diagonal with K^(d) for each component
                    # K^(d)(x_i, x_j) = exp(-(x_i^(d) - x_j^(d))^2 / (2*alpha*sigma^(d)^2))
                    exponent = -0.5 * (diff / (alpha * sigma))**2
                    K_diag = np.exp(exponent)  # (state_dim,)
                    
                    # Component-wise kernel gradient:
                    # ∇_x_i^(d) K^(d) = -(x_i^(d) - x_j^(d))/(alpha*sigma^(d)^2) * K^(d)
                    grad_K = -(diff / (alpha * sigma**2)) * K_diag  # (state_dim,)
                    
                    # Velocity contribution (component-wise multiplication)
                    # K_diag * grad_log_post[i] is element-wise product
                    velocity += K_diag * grad_log_post[i] + grad_K
                
                particles_new[j] = self.particles[j] + (self.epsilon / self.n_particles) * velocity
            
            self.particles = particles_new