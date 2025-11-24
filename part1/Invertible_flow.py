"""
Invertible Particle Flow Filters.

This module implements:
1. LEDH - Local Exact Daum-Huang particle flow filter
2. EDH - Exact Daum-Huang particle flow filter

Both use particle flow with EKF/UKF integration for prediction/update.
"""

import numpy as np
from typing import Tuple, Optional, List
from models import StateSpaceModel
from filters import ExtendedKalmanFilter, UnscentedKalmanFilter


class LEDHParticleFlowFilter:
    """
    Local Exact Daum-Huang (LEDH) Particle Flow Filter.
    
    Each particle has its own linearization point, leading to particle-specific
    flow parameters A^i(λ) and b^i(λ). Tracks Jacobian determinant θ^i for each
    particle to correct weight updates.
    
    Algorithm:
    1. Each particle has its own EKF/UKF for prediction to get P^i
    2. Propagate particles through dynamics: η_0^i = g_k(x_{k-1}^i, v_k)
    3. Flow migration with per-particle A^i(λ) and b^i(λ)
    4. Weight update with Jacobian correction: w^i ∝ p(x_k^i|x_{k-1}^i) * p(z_k|x_k^i) * θ^i / p(η_0^i|x_{k-1}^i)
    """
    
    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        n_lambda_steps: int = 100,
        filter_type: str = 'ekf',
        resample_threshold: float = 0.5,
        **filter_kwargs
    ):
        """
        Initialize LEDH Particle Flow Filter.
        
        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            filter_type: 'ekf' or 'ukf' for EKF/UKF prediction
            resample_threshold: Resample when ESS < threshold * n_particles
            **filter_kwargs: Additional arguments for EKF/UKF (e.g., alpha, beta, kappa for UKF)
        """
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.filter_type = filter_type
        self.resample_threshold = resample_threshold
        self.filter_kwargs = filter_kwargs
        
        # Particles and weights
        self.particles = None
        self.weights = None
        
        # Per-particle filters for covariance estimation
        self.particle_filters = []
        
        # Storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        
        # Random state
        self.random_state = np.random.default_rng()
    
    def _create_filter(self, initial_mean: np.ndarray, initial_cov: np.ndarray):
        """Create an EKF or UKF instance."""
        if self.filter_type == 'ekf':
            filt = ExtendedKalmanFilter(self.model)
        elif self.filter_type == 'ukf':
            # Filter out n_threads as UKF doesn't accept it
            ukf_kwargs = {k: v for k, v in self.filter_kwargs.items() if k != 'n_threads'}
            filt = UnscentedKalmanFilter(self.model, **ukf_kwargs)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        filt.initialize(initial_mean, initial_cov)
        return filt
    
    def initialize(self, initial_mean: Optional[np.ndarray] = None,
                   initial_cov: Optional[np.ndarray] = None,
                   random_state: Optional[np.random.Generator] = None):
        """
        Initialize particles from prior distribution.
        
        Args:
            initial_mean: Initial state mean
            initial_cov: Initial state covariance
            random_state: Optional random generator
        """
        if random_state is not None:
            self.random_state = random_state
        if initial_mean is None:
            initial_mean = np.zeros(self.state_dim)
        if initial_cov is None:
            initial_cov = np.eye(self.state_dim)
        
        # Sample particles from initial distribution
        self.particles = self.random_state.multivariate_normal(
            initial_mean, initial_cov, size=self.n_particles
        )
        
        # Initialize uniform weights
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Create per-particle filters
        self.particle_filters = []
        for i in range(self.n_particles):
            filt = self._create_filter(self.particles[i], initial_cov)
            self.particle_filters.append(filt)
        
        # Reset storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
    
    def _compute_A_b(self, eta_bar: np.ndarray, P: np.ndarray, 
                     z: np.ndarray, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flow parameters A(λ) and b(λ) for linearized flow.
        
        Based on equations (13) and (14) in the algorithm:
        A(λ) = -1/2 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
        b(λ) = (I + 2λA) * [(I + λA) * P * H^T * R^{-1} * z + A * η_bar_0]
        
        Args:
            eta_bar: Linearization point (particle location)
            P: State covariance from EKF/UKF prediction
            z: Observation
            lambda_val: Current pseudo-time λ
            
        Returns:
            Tuple of (A, b) matrices
        """
        # Get observation Jacobian and covariance at linearization point
        H = self.model.observation_jacobian(eta_bar)
        R = self.model.observation_cov(eta_bar)
        
        # Compute A(λ) = -1/2 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
        HPH = H @ P @ H.T
        S = lambda_val * HPH + R
        S_inv = np.linalg.inv(S)
        A = -0.5 * P @ H.T @ S_inv @ H
        
        # Compute b(λ)
        I = np.eye(self.state_dim)
        term1 = I + lambda_val * A
        term2 = I + 2 * lambda_val * A
        R_inv = np.linalg.inv(R)
        b = term2 @ (term1 @ P @ H.T @ R_inv @ z + A @ eta_bar)
        
        return A, b
    
    def predict(self):
        """
        Prediction step: propagate particles through dynamics.
        
        For each particle:
        1. Apply EKF/UKF prediction to get P^i
        2. Propagate particle through dynamics: η_0^i = g_k(x_{k-1}^i, v_k)
        """
        # Store old particles for weight calculation
        self.particles_prev = self.particles.copy()
        
        # Propagate each particle and update its filter
        for i in range(self.n_particles):
            # Update filter mean to current particle location
            self.particle_filters[i].mean = self.particles[i].copy()
            
            # EKF/UKF prediction to get P^i
            self.particle_filters[i].predict()
            
            # Propagate particle through dynamics with process noise
            self.particles[i] = self.model.sample_state_transition(
                self.particles[i], self.random_state
            )
    
    def update(self, y: np.ndarray):
        """
        Update step: migrate particles via flow and update weights.
        
        Args:
            y: Observation, shape (obs_dim,)
        """
        # Store particles after prediction (η_0) for weight calculation
        eta_0 = self.particles.copy()
        
        # Initialize flow particles and Jacobian determinants
        eta_1 = eta_0.copy()
        theta = np.ones(self.n_particles)
        
        # Store mean particles for linearization (η_bar^i = g_k(x_{k-1}^i, 0))
        eta_bar = np.array([
            self.model.state_transition_mean(self.particles_prev[i])
            for i in range(self.n_particles)
        ])
        
        # Discretize pseudo-time λ ∈ [0,1]
        lambdas = np.linspace(0, 1, self.n_lambda_steps + 1)
        
        # Flow integration
        for j in range(self.n_lambda_steps):
            lambda_val = lambdas[j+1]
            d_lambda = lambdas[j+1] - lambdas[j]
            
            # For each particle: compute A_j^i(λ) and b_j^i(λ)
            for i in range(self.n_particles):
                # Get P^i from particle filter
                P_i = self.particle_filters[i].cov
                
                # Compute A^i and b^i (linearization at η_bar^i)
                A_i, b_i = self._compute_A_b(eta_bar[i], P_i, y, lambda_val)
                
                # Migrate η_bar^i
                eta_bar[i] = eta_bar[i] + d_lambda * (A_i @ eta_bar[i] + b_i)
                
                # Migrate particle η_1^i
                eta_1[i] = eta_1[i] + d_lambda * (A_i @ eta_1[i] + b_i)
                
                # Update Jacobian determinant: θ^i *= |det(I + Δλ * A^i)|
                M_i = np.eye(self.state_dim) + d_lambda * A_i
                theta[i] *= np.abs(np.linalg.det(M_i))
        
        # Update particles to final flow state
        self.particles = eta_1
        
        # Compute weights: w^i = [p(x_k^i|x_{k-1}^i) * p(z_k|x_k^i) * θ^i / p(η_0^i|x_{k-1}^i)] * w_{k-1}^i
        log_weights = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            # p(x_k^i | x_{k-1}^i): transition probability (from dynamics model)
            # For simplicity, we approximate this as exp(-||x_k^i - mean||^2 / (2*Q))
            mean_transition = self.model.state_transition_mean(self.particles_prev[i])
            Q = self.model.state_transition_cov(self.particles_prev[i])
            diff = self.particles[i] - mean_transition
            log_p_transition = -0.5 * (diff.T @ np.linalg.inv(Q) @ diff + 
                                       np.log(np.linalg.det(2 * np.pi * Q)))
            
            # p(z_k | x_k^i): observation likelihood
            log_p_obs = self.model.log_observation_prob(y, self.particles[i])
            
            # p(η_0^i | x_{k-1}^i): same as p(x_k^i | x_{k-1}^i) for η_0
            diff_eta0 = eta_0[i] - mean_transition
            log_p_eta0 = -0.5 * (diff_eta0.T @ np.linalg.inv(Q) @ diff_eta0 + 
                                 np.log(np.linalg.det(2 * np.pi * Q)))
            
            # Weight update
            # Add small epsilon to prevent log(0)
            weight_eps = max(self.weights[i], 1e-300)
            log_weights[i] = (log_p_transition + log_p_obs + np.log(theta[i]) - 
                             log_p_eta0 + np.log(weight_eps))
        
        # Normalize weights (log-sum-exp trick)
        max_log_weight = np.max(log_weights)
        log_weights_normalized = log_weights - max_log_weight
        self.weights = np.exp(log_weights_normalized)
        self.weights = self.weights / np.sum(self.weights)
        
        # Save weights before resampling
        self.weights_history.append(self.weights.copy())
        
        # Log-likelihood: log p(y_k | y_{1:k-1})
        log_lik = max_log_weight + np.log(np.mean(np.exp(log_weights_normalized)))
        self.log_likelihoods.append(log_lik)
        
        # Update particle filters with observation
        for i in range(self.n_particles):
            self.particle_filters[i].mean = self.particles[i].copy()
            self.particle_filters[i].update(y)
        
        # Effective sample size
        ess = self._effective_sample_size()
        self.ess_history.append(ess)
        
        # Resample if ESS is too low
        if ess < self.resample_threshold * self.n_particles:
            self._systematic_resample()
    
    def _effective_sample_size(self) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(self.weights ** 2)
    
    def _systematic_resample(self):
        """Systematic resampling to avoid particle degeneracy."""
        n = self.n_particles
        
        # Generate systematic samples
        u = self.random_state.uniform(0, 1/n)
        indices = np.zeros(n, dtype=int)
        cumsum = np.cumsum(self.weights)
        
        i = 0
        for j in range(n):
            u_j = u + j / n
            while u_j > cumsum[i]:
                i += 1
            indices[j] = i
        
        # Resample particles and filters
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n
        
        # Resample filters
        new_filters = []
        for idx in indices:
            filt = self._create_filter(
                self.particles[idx], 
                self.particle_filters[idx].cov
            )
            new_filters.append(filt)
        self.particle_filters = new_filters
    
    def _estimate_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate mean and covariance from weighted particles."""
        mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        diff = self.particles - mean
        cov = np.sum(
            self.weights[:, np.newaxis, np.newaxis] * 
            np.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )
        return mean, cov
    
    def filter(self, observations: np.ndarray,
               initial_mean: Optional[np.ndarray] = None,
               initial_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the LEDH filter on a sequence of observations.
        
        Args:
            observations: Array of shape (T, obs_dim)
            initial_mean: Initial state mean
            initial_cov: Initial state covariance
            
        Returns:
            Tuple of (means, covs)
        """
        self.initialize(initial_mean, initial_cov)
        T = len(observations)
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
        
        return np.array(self.means), np.array(self.covs)


class EDHParticleFlowFilter:
    """
    Exact Daum-Huang (EDH) Particle Flow Filter.
    
    Uses global linearization around ensemble mean, leading to shared flow
    parameters A(λ) and b(λ) for all particles. The Jacobian determinant
    cancels in the weight update.
    
    Algorithm:
    1. Single global EKF/UKF prediction to get P
    2. Propagate particles through dynamics: η_0^i = g_k(x_{k-1}^i, v_k)
    3. Flow migration with shared A(λ) and b(λ)
    4. Weight update: w^i ∝ p(x_k^i|x_{k-1}^i) * p(z_k|x_k^i) / p(η_0^i|x_{k-1}^i)
    """
    
    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        n_lambda_steps: int = 100,
        filter_type: str = 'ekf',
        resample_threshold: float = 0.5,
        **filter_kwargs
    ):
        """
        Initialize EDH Particle Flow Filter.
        
        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            filter_type: 'ekf' or 'ukf' for EKF/UKF prediction
            resample_threshold: Resample when ESS < threshold * n_particles
            **filter_kwargs: Additional arguments for EKF/UKF
        """
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.filter_type = filter_type
        self.resample_threshold = resample_threshold
        self.filter_kwargs = filter_kwargs
        
        # Particles and weights
        self.particles = None
        self.weights = None
        
        # Global filter for covariance estimation
        self.global_filter = None
        
        # Storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        
        # Random state
        self.random_state = np.random.default_rng()
    
    def _create_filter(self, initial_mean: np.ndarray, initial_cov: np.ndarray):
        """Create an EKF or UKF instance."""
        if self.filter_type == 'ekf':
            filt = ExtendedKalmanFilter(self.model)
        elif self.filter_type == 'ukf':
            # Filter out n_threads as UKF doesn't accept it
            ukf_kwargs = {k: v for k, v in self.filter_kwargs.items() if k != 'n_threads'}
            filt = UnscentedKalmanFilter(self.model, **ukf_kwargs)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        filt.initialize(initial_mean, initial_cov)
        return filt
    
    def initialize(self, initial_mean: Optional[np.ndarray] = None,
                   initial_cov: Optional[np.ndarray] = None,
                   random_state: Optional[np.random.Generator] = None):
        """
        Initialize particles from prior distribution.
        
        Args:
            initial_mean: Initial state mean
            initial_cov: Initial state covariance
            random_state: Optional random generator
        """
        if random_state is not None:
            self.random_state = random_state
        if initial_mean is None:
            initial_mean = np.zeros(self.state_dim)
        if initial_cov is None:
            initial_cov = np.eye(self.state_dim)
        
        # Sample particles from initial distribution
        self.particles = self.random_state.multivariate_normal(
            initial_mean, initial_cov, size=self.n_particles
        )
        
        # Initialize uniform weights
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Create global filter
        self.global_filter = self._create_filter(initial_mean, initial_cov)
        
        # Reset storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
    
    def _compute_A_b(self, eta_bar: np.ndarray, P: np.ndarray, 
                     z: np.ndarray, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flow parameters A(λ) and b(λ) for linearized flow.
        
        Based on equations (10) and (11) in the algorithm:
        A(λ) = -1/2 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
        b(λ) = (I + 2λA) * [(I + λA) * P * H^T * R^{-1} * z + A * η_bar_0]
        
        Args:
            eta_bar: Linearization point (ensemble mean)
            P: State covariance from EKF/UKF prediction
            z: Observation
            lambda_val: Current pseudo-time λ
            
        Returns:
            Tuple of (A, b) matrices
        """
        # Get observation Jacobian and covariance at linearization point
        H = self.model.observation_jacobian(eta_bar)
        R = self.model.observation_cov(eta_bar)
        
        # Compute A(λ)
        HPH = H @ P @ H.T
        S = lambda_val * HPH + R
        S_inv = np.linalg.inv(S)
        A = -0.5 * P @ H.T @ S_inv @ H
        
        # Compute b(λ)
        I = np.eye(self.state_dim)
        term1 = I + lambda_val * A
        term2 = I + 2 * lambda_val * A
        R_inv = np.linalg.inv(R)
        b = term2 @ (term1 @ P @ H.T @ R_inv @ z + A @ eta_bar)
        
        return A, b
    
    def predict(self):
        """
        Prediction step: propagate particles through dynamics.
        
        1. Apply global EKF/UKF prediction to get P
        2. Propagate each particle through dynamics: η_0^i = g_k(x_{k-1}^i, v_k)
        """
        # Store old particles for weight calculation
        self.particles_prev = self.particles.copy()
        
        # Compute ensemble mean for global prediction
        ensemble_mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        
        # Global EKF/UKF prediction
        self.global_filter.mean = ensemble_mean
        self.global_filter.predict()
        
        # Propagate each particle through dynamics with process noise
        for i in range(self.n_particles):
            self.particles[i] = self.model.sample_state_transition(
                self.particles[i], self.random_state
            )
    
    def update(self, y: np.ndarray):
        """
        Update step: migrate particles via flow and update weights.
        
        Args:
            y: Observation, shape (obs_dim,)
        """
        # Store particles after prediction (η_0) for weight calculation
        eta_0 = self.particles.copy()
        
        # Initialize flow particles
        eta_1 = eta_0.copy()
        
        # Compute ensemble mean for linearization
        ensemble_mean_prev = np.sum(self.weights[:, np.newaxis] * self.particles_prev, axis=0)
        eta_bar_0 = self.model.state_transition_mean(ensemble_mean_prev)
        eta_bar = eta_bar_0.copy()
        
        # Get global P from filter
        P = self.global_filter.cov
        
        # Discretize pseudo-time λ ∈ [0,1]
        lambdas = np.linspace(0, 1, self.n_lambda_steps + 1)
        
        # Flow integration
        for j in range(self.n_lambda_steps):
            lambda_val = lambdas[j+1]
            d_lambda = lambdas[j+1] - lambdas[j]
            
            # Compute shared A(λ) and b(λ) (linearization at ensemble mean)
            A, b = self._compute_A_b(eta_bar, P, y, lambda_val)
            
            # Migrate ensemble mean η_bar
            eta_bar = eta_bar + d_lambda * (A @ eta_bar + b)
            
            # Migrate all particles with same A and b
            for i in range(self.n_particles):
                eta_1[i] = eta_1[i] + d_lambda * (A @ eta_1[i] + b)
        
        # Update particles to final flow state
        self.particles = eta_1
        
        # Compute weights: w^i = [p(x_k^i|x_{k-1}^i) * p(z_k|x_k^i) / p(η_0^i|x_{k-1}^i)] * w_{k-1}^i
        # Note: No θ term because Jacobian cancels for EDH
        log_weights = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            # p(x_k^i | x_{k-1}^i): transition probability
            mean_transition = self.model.state_transition_mean(self.particles_prev[i])
            Q = self.model.state_transition_cov(self.particles_prev[i])
            diff = self.particles[i] - mean_transition
            log_p_transition = -0.5 * (diff.T @ np.linalg.inv(Q) @ diff + 
                                       np.log(np.linalg.det(2 * np.pi * Q)))
            
            # p(z_k | x_k^i): observation likelihood
            log_p_obs = self.model.log_observation_prob(y, self.particles[i])
            
            # p(η_0^i | x_{k-1}^i): same as transition probability for η_0
            diff_eta0 = eta_0[i] - mean_transition
            log_p_eta0 = -0.5 * (diff_eta0.T @ np.linalg.inv(Q) @ diff_eta0 + 
                                 np.log(np.linalg.det(2 * np.pi * Q)))
            
            # Weight update (no θ term)
            # Add small epsilon to prevent log(0)
            weight_eps = max(self.weights[i], 1e-300)
            log_weights[i] = log_p_transition + log_p_obs - log_p_eta0 + np.log(weight_eps)
        
        # Normalize weights (log-sum-exp trick)
        max_log_weight = np.max(log_weights)
        log_weights_normalized = log_weights - max_log_weight
        self.weights = np.exp(log_weights_normalized)
        self.weights = self.weights / np.sum(self.weights)
        
        # Save weights before resampling
        self.weights_history.append(self.weights.copy())
        
        # Log-likelihood: log p(y_k | y_{1:k-1})
        log_lik = max_log_weight + np.log(np.mean(np.exp(log_weights_normalized)))
        self.log_likelihoods.append(log_lik)
        
        # Update global filter with observation
        ensemble_mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        self.global_filter.mean = ensemble_mean
        self.global_filter.update(y)
        
        # Effective sample size
        ess = self._effective_sample_size()
        self.ess_history.append(ess)
        
        # Resample if ESS is too low
        if ess < self.resample_threshold * self.n_particles:
            self._systematic_resample()
    
    def _effective_sample_size(self) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(self.weights ** 2)
    
    def _systematic_resample(self):
        """Systematic resampling to avoid particle degeneracy."""
        n = self.n_particles
        
        # Generate systematic samples
        u = self.random_state.uniform(0, 1/n)
        indices = np.zeros(n, dtype=int)
        cumsum = np.cumsum(self.weights)
        
        i = 0
        for j in range(n):
            u_j = u + j / n
            while u_j > cumsum[i]:
                i += 1
            indices[j] = i
        
        # Resample particles
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n
    
    def _estimate_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate mean and covariance from weighted particles."""
        mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        diff = self.particles - mean
        cov = np.sum(
            self.weights[:, np.newaxis, np.newaxis] * 
            np.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )
        return mean, cov
    
    def filter(self, observations: np.ndarray,
               initial_mean: Optional[np.ndarray] = None,
               initial_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the EDH filter on a sequence of observations.
        
        Args:
            observations: Array of shape (T, obs_dim)
            initial_mean: Initial state mean
            initial_cov: Initial state covariance
            
        Returns:
            Tuple of (means, covs)
        """
        self.initialize(initial_mean, initial_cov)
        T = len(observations)
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
        
        return np.array(self.means), np.array(self.covs)
