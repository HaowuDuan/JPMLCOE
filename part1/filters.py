"""
Filter implementations for linear and nonlinear/non-Gaussian state-space models.

This module provides four filtering algorithms:
1. Kalman Filter (KF) - For linear-Gaussian models
2. Extended Kalman Filter (EKF) - Jacobian-based linearization
3. Unscented Kalman Filter (UKF) - Sigma point method 
4. ParticleFilter 
5. Exact Daum Huang Flow
6. Local Exact Daum Huang Flow 
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from models import StateSpaceModel
from concurrent.futures import ThreadPoolExecutor
import os

class KalmanFilter:
    """
    Kalman Filter for linear-Gaussian state-space models.
    
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
        - Sigma: Initial covariance matrix (nx, nx)
    
    Supports both standard and Joseph-form covariance updates for numerical stability.
    """
    
    def __init__(
        self,
        F: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        D: np.ndarray,
        Sigma: np.ndarray,
        use_joseph_form: bool = False
    ):
        """
        Initialize the Kalman Filter.
        
        Args:
            F: State transition matrix (nx, nx)
            B: Process noise matrix (nx, nv)
            H: Observation matrix (ny, nx)
            D: Observation noise matrix (ny, nw)
            Sigma: Initial covariance matrix (nx, nx)
            use_joseph_form: If True, use Joseph stabilized covariance update
        """
        self.nx = F.shape[0]
        self.nv = B.shape[1]
        self.ny = H.shape[0]
        self.nw = D.shape[1]
        
        self.F = F
        self.B = B
        self.H = H
        self.D = D
        self.Sigma = Sigma
        
        self.Q = B @ B.T
        self.R = D @ D.T
        
        self.use_joseph_form = use_joseph_form
        
        self.reset()
        
        self.history = {
            'mean_pred': [],
            'cov_pred': [],
            'mean_updated': [],
            'cov_updated': [],
            'kalman_gain': [],
            'innovations': [],
            'condition_numbers': []
        }
    
    def reset(self):
        self.mean = np.zeros(self.nx)
        self.cov = self.Sigma.copy()
    
    def predict(self):
        """
        Prediction step: propagate state estimate forward.
        
        Returns:
            Tuple of (predicted_mean, predicted_cov)
        """
        self.mean = self.F @ self.mean
        self.cov = self.F @ self.cov @ self.F.T + self.Q
        
        self.history['mean_pred'].append(self.mean.copy())
        self.history['cov_pred'].append(self.cov.copy())
        
        return self.mean, self.cov
    
    def update(self, observation: np.ndarray):
        """
        Update step: incorporate observation.
        
        Args:
            observation: Observation vector, shape (ny,)
            
        Returns:
            Tuple of (updated_mean, updated_cov, kalman_gain)
        """
        innovation = observation - self.H @ self.mean
        innovation_cov = self.H @ self.cov @ self.H.T + self.R
        kalman_gain = self.cov @ self.H.T @ np.linalg.inv(innovation_cov)
        
        self.mean = self.mean + kalman_gain @ innovation
        
        if self.use_joseph_form:
            I = np.eye(self.nx)
            I_minus_KH = I - kalman_gain @ self.H
            self.cov = (I_minus_KH @ self.cov @ I_minus_KH.T +
                       kalman_gain @ self.R @ kalman_gain.T)
        else:
            I = np.eye(self.nx)
            self.cov = (I - kalman_gain @ self.H) @ self.cov
        
        self.history['mean_updated'].append(self.mean.copy())
        self.history['cov_updated'].append(self.cov.copy())
        self.history['kalman_gain'].append(kalman_gain.copy())
        self.history['innovations'].append(innovation.copy())
        self.history['condition_numbers'].append(np.linalg.cond(self.cov))
        
        return self.mean, self.cov, kalman_gain
    
    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the filter on a sequence of observations.
        
        Args:
            observations: Array of shape (T, ny) containing observations
            
        Returns:
            Tuple of (means, covariances) where:
            - means: Array of shape (T, nx) containing filtered state estimates
            - covariances: Array of shape (T, nx, nx) containing filtered covariances
        """
        T = observations.shape[0]
        
        self.reset()
        for key in self.history:
            self.history[key] = []
        
        means = []
        covariances = []
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            
            means.append(self.mean.copy())
            covariances.append(self.cov.copy())
        
        return np.array(means), np.array(covariances)
    
    def get_history(self):
        """Return filter history."""
        return self.history.copy()
   

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear state-space models.
    
    The EKF linearizes the nonlinear functions around the current state estimate
    using first-order Taylor expansion (Jacobian matrices). This works well when
    the nonlinearity is mild, but can fail when:
    - Nonlinearities are strong
    - Jacobians are difficult to compute or inaccurate
    - State estimates are far from the true state
    
    The filter maintains:
    - Mean: x̂_n|n (filtered state estimate)
    - Covariance: P_n|n (filtered state covariance)
    """
    
    def __init__(self, model: StateSpaceModel):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            model: State-space model instance
        """
        self.model = model
        self.state_dim = model.state_dim
        
        # Filter state
        self.mean = None
        self.cov = None
        
        # Storage for estimates and covariances over time
        self.means = []
        self.covs = []
        self.log_likelihoods = []
    
    def initialize(self, mean: Optional[np.ndarray] = None,
                   cov: Optional[np.ndarray] = None):
        """
        Initialize the filter state.
        
        Args:
            mean: Initial state mean. If None, uses model's stationary mean.
            cov: Initial state covariance. If None, uses model's stationary covariance.
        """
        if mean is None:
            mean = np.zeros(self.state_dim)
        if cov is None:
            # Use stationary covariance if available
            if hasattr(self.model, 'stationary_var'):
                cov = np.eye(self.state_dim) * self.model.stationary_var
            else:
                cov = np.eye(self.state_dim)
        
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.means = []
        self.covs = []
        self.log_likelihoods = []
    
    def predict(self):
        """
        Prediction step: propagate state estimate forward.
        
        Uses linearization: x' ≈ f(x̂) + F·(x - x̂), where F is the Jacobian.
        """
        # Predict mean: x̂_{n|n-1} = f(x̂_{n-1|n-1})
        self.mean = self.model.state_transition_mean(self.mean)
        
        # Predict covariance: P_{n|n-1} = F·P_{n-1|n-1}·F^T + Q
        F = self.model.state_jacobian(self.mean)
        Q = self.model.state_transition_cov(self.mean)
        self.cov = F @ self.cov @ F.T + Q
    
    def update(self, y: np.ndarray):
        """
        Update step: incorporate observation.
        
        Uses linearization: y ≈ h(x̂) + H·(x - x̂), where H is the Jacobian.
        
        Args:
            y: Observation, shape (obs_dim,)
        """
        # Predicted observation mean
        y_pred = self.model.observation_mean(self.mean)
        
        # Observation covariance
        R = self.model.observation_cov(self.mean)
        
        # Observation Jacobian
        H = self.model.observation_jacobian(self.mean)
        
        # Innovation: ν = y - y_pred
        innovation = y - y_pred
        
        # Innovation covariance: S = H·P·H^T + R
        S = H @ self.cov @ H.T + R
        
        # Handle case where H is zero (e.g., observation mean doesn't depend on state)
        # In this case, the filter cannot update the state estimate based on observations
        # This is a limitation of EKF for certain models
        if np.allclose(H, 0):
            # No update possible - state estimate remains unchanged
            # Note: This is expected for models where observation mean is constant
            pass
        else:
            # Kalman gain: K = P·H^T·S^{-1}
            K = self.cov @ H.T @ np.linalg.inv(S)
            
            # Update mean: x̂_{n|n} = x̂_{n|n-1} + K·ν
            self.mean = self.mean + K @ innovation
            
            # Update covariance: P_{n|n} = (I - K·H)·P_{n|n-1}
            I = np.eye(self.state_dim)
            self.cov = (I - K @ H) @ self.cov
        
        # Log-likelihood: log p(y_n | y_{1:n-1})
        log_lik = -0.5 * (innovation.T @ np.linalg.inv(S) @ innovation +
                          np.log(np.linalg.det(2 * np.pi * S)))
        self.log_likelihoods.append(log_lik)
    
    def filter(self, observations: np.ndarray,
               initial_mean: Optional[np.ndarray] = None,
               initial_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the filter on a sequence of observations.
        
        Args:
            observations: Array of shape (T, obs_dim) containing observations
            initial_mean: Initial state mean (optional)
            initial_cov: Initial state covariance (optional)
            
        Returns:
            Tuple of (means, covs) where:
            - means: Array of shape (T, state_dim) containing filtered state estimates
            - covs: Array of shape (T, state_dim, state_dim) containing filtered covariances
        """
        self.initialize(initial_mean, initial_cov)
        T = len(observations)
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            self.means.append(self.mean.copy())
            self.covs.append(self.cov.copy())
        
        return np.array(self.means), np.array(self.covs)


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) for nonlinear state-space models.
    
    The UKF uses sigma points to capture the mean and covariance through
    nonlinear transformations, avoiding the need for Jacobian computation.
    This works well when:
    - Nonlinearities are moderate
    - Distributions are roughly Gaussian
    - State dimension is moderate (scales as O(n²) with n states)
    
    The filter can fail when:
    - Distributions are highly non-Gaussian
    - State dimension is very high (computational cost)
    - Sigma points become unstable or invalid
    
    Uses the standard unscented transform with parameters:
    - α: controls spread of sigma points (default 1e-3)
    - β: incorporates prior knowledge (default 2.0)
    - κ: secondary scaling parameter (default 0.0)
    """
    
    def __init__(self, model: StateSpaceModel, alpha: float = 1e-3,
                 beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize the Unscented Kalman Filter.
        
        Args:
            model: State-space model instance
            alpha: Spread parameter for sigma points (small positive)
            beta: Incorporates prior knowledge (2 is optimal for Gaussian)
            kappa: Secondary scaling parameter (0 for default)
        """
        self.model = model
        self.state_dim = model.state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Compute lambda parameter
        self.lambda_ = alpha**2 * (self.state_dim + kappa) - self.state_dim
        
        # Weights for mean and covariance
        W_m_0 = self.lambda_ / (self.state_dim + self.lambda_)
        W_c_0 = W_m_0 + (1 - alpha**2 + beta)
        W_i = 1.0 / (2 * (self.state_dim + self.lambda_))
        
        self.weights_mean = np.concatenate([[W_m_0], np.full(2 * self.state_dim, W_i)])
        self.weights_cov = np.concatenate([[W_c_0], np.full(2 * self.state_dim, W_i)])
        
        # Filter state
        self.mean = None
        self.cov = None
        
        # Storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
    
    def _compute_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Compute sigma points using the unscented transform.
        
        Args:
            mean: State mean, shape (state_dim,)
            cov: State covariance, shape (state_dim, state_dim)
            
        Returns:
            Array of shape (2*state_dim + 1, state_dim) containing sigma points
        """
        n = self.state_dim
        lambda_ = self.lambda_
        
        # Ensure covariance is positive definite (add small diagonal if needed)
        cov_regularized = cov + np.eye(n) * 1e-8
        
        # Compute matrix square root: sqrt((n + λ)·P)
        try:
            sqrt_cov = np.linalg.cholesky((n + lambda_) * cov_regularized)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh((n + lambda_) * cov_regularized)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            sqrt_cov = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Initialize sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        
        # Generate sigma points
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[:, i]
            sigma_points[i + n + 1] = mean - sqrt_cov[:, i]
        
        return sigma_points
    
    def initialize(self, mean: Optional[np.ndarray] = None,
                   cov: Optional[np.ndarray] = None):
        """Initialize the filter state."""
        if mean is None:
            mean = np.zeros(self.state_dim)
        if cov is None:
            if hasattr(self.model, 'stationary_var'):
                cov = np.eye(self.state_dim) * self.model.stationary_var
            else:
                cov = np.eye(self.state_dim)
        
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.means = []
        self.covs = []
        self.log_likelihoods = []
    
    def predict(self):
        """Prediction step using unscented transform."""
        # Generate sigma points
        sigma_points = self._compute_sigma_points(self.mean, self.cov)
        
        # Propagate through state transition (deterministic part only)
        # The noise will be added to the covariance separately
        sigma_points_pred = np.array([
            self.model.state_transition_mean(sp) for sp in sigma_points
        ])
        
        # Predict mean: weighted sum of propagated sigma points
        self.mean = np.sum(self.weights_mean[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Predict covariance: weighted sum of outer products
        diff = sigma_points_pred - self.mean
        self.cov = np.sum(
            self.weights_cov[:, np.newaxis, np.newaxis] * 
            np.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )
        
        # Add process noise
        Q = self.model.state_transition_cov(self.mean)
        self.cov = self.cov + Q
    
    def update(self, y: np.ndarray):
        """Update step using unscented transform."""
        # Generate sigma points
        sigma_points = self._compute_sigma_points(self.mean, self.cov)
        
        # Propagate through observation model
        y_sigma = np.array([
            self.model.observation_mean(sp) for sp in sigma_points
        ])
        
        # Predicted observation mean
        y_pred = np.sum(self.weights_mean[:, np.newaxis] * y_sigma, axis=0)
        
        # Innovation covariance
        diff_y = y_sigma - y_pred
        S = np.sum(
            self.weights_cov[:, np.newaxis, np.newaxis] *
            np.einsum('ij,ik->ijk', diff_y, diff_y),
            axis=0
        )
        
        # Add observation noise
        R = self.model.observation_cov(self.mean)
        S = S + R
        
        # Cross-covariance between state and observation
        diff_x = sigma_points - self.mean
        P_xy = np.sum(
            self.weights_cov[:, np.newaxis, np.newaxis] *
            np.einsum('ij,ik->ijk', diff_x, diff_y),
            axis=0
        )
        
        # Kalman gain
        K = P_xy @ np.linalg.inv(S)
        
        # Innovation
        innovation = y - y_pred
        
        # Update mean and covariance
        self.mean = self.mean + K @ innovation
        self.cov = self.cov - K @ S @ K.T
        
        # Log-likelihood
        log_lik = -0.5 * (innovation.T @ np.linalg.inv(S) @ innovation +
                          np.log(np.linalg.det(2 * np.pi * S)))
        self.log_likelihoods.append(log_lik)
    
    def filter(self, observations: np.ndarray,
               initial_mean: Optional[np.ndarray] = None,
               initial_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the filter on a sequence of observations.
        
        Args:
            observations: Array of shape (T, obs_dim) containing observations
            initial_mean: Initial state mean (optional)
            initial_cov: Initial state covariance (optional)
            
        Returns:
            Tuple of (means, covs) where:
            - means: Array of shape (T, state_dim) containing filtered state estimates
            - covs: Array of shape (T, state_dim, state_dim) containing filtered covariances
        """
        self.initialize(initial_mean, initial_cov)
        T = len(observations)
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            self.means.append(self.mean.copy())
            self.covs.append(self.cov.copy())
        
        return np.array(self.means), np.array(self.covs)


class ParticleFilter:
    """
    Particle Filter (PF) for nonlinear/non-Gaussian state-space models.
    
    Implements a bootstrap particle filter with systematic resampling.
    The PF uses Monte Carlo methods to represent the posterior distribution
    as a weighted set of particles.
    
    Strengths:
    - Handles arbitrary nonlinearities and non-Gaussian distributions
    - No linearization required
    - Asymptotically exact as N → ∞
    
    Limitations:
    - Computationally expensive (scales with number of particles)
    - Can suffer from particle degeneracy (most weights near zero)
    - Requires careful tuning of proposal distribution
    
    This implementation uses:
    - Bootstrap filter (proposal = state transition)
    - Systematic resampling when ESS < threshold
    - Effective Sample Size (ESS) tracking
    """
    
    def __init__(self, model: StateSpaceModel, n_particles: int = 1000,
                 resample_threshold: float = 0.5, n_threads: Optional[int] = None):
        """
        Initialize the Particle Filter.
        
        Args:
            model: State-space model instance
            n_particles: Number of particles (default 1000)
            resample_threshold: Resample when ESS/N < threshold (default 0.5)
            n_threads: Number of threads for parallelization (default: CPU count, None = no threading)
        """
        self.model = model
        self.state_dim = model.state_dim
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        
        # Threading setup
        if n_threads is None:
            self.n_threads = os.cpu_count() if os.cpu_count() else 1
        else:
            self.n_threads = max(1, int(n_threads))
        
        # Particles and weights
        self.particles = None
        self.weights = None
        
        # Storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.n_unique_particles = []
        self.resampled_at = []
        self.weights_history = []
        
        # Random state for reproducibility
        self.random_state = np.random.default_rng()
    
    def _effective_sample_size(self) -> float:
        """
        Compute the effective sample size (ESS).
        
        ESS = 1 / Σ(w_i²), where w_i are normalized weights.
        
        Returns:
            Effective sample size (between 1 and N)
        """
        if self.weights is None:
            return self.n_particles
        
        normalized_weights = self.weights / np.sum(self.weights)
        ess = 1.0 / np.sum(normalized_weights**2)
        return ess
    
    def _systematic_resample(self, timestep: int):
        """
        Systematic resampling of particles.
        
        Generates a single random number and uses it to select particles
        deterministically, ensuring better coverage than multinomial resampling.
        
        Args:
            timestep: Current timestep (for tracking when resampling occurs)
        """
        normalized_weights = self.weights / np.sum(self.weights)
        
        # Cumulative weights
        cumsum = np.cumsum(normalized_weights)
        
        # Generate systematic points
        u = self.random_state.uniform(0, 1 / self.n_particles)
        u_vals = u + np.arange(self.n_particles) / self.n_particles
        
        # Resample indices
        indices = np.searchsorted(cumsum, u_vals)
        indices = np.clip(indices, 0, self.n_particles - 1)
        
        # Resample particles
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Track unique particles after resampling
        unique_count = len(np.unique(self.particles, axis=0))
        self.n_unique_particles.append(unique_count)
        self.resampled_at.append(timestep)
    
    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """
        Initialize particles from the initial distribution.
        
        Args:
            random_state: Optional numpy random generator
        """
        if random_state is not None:
            self.random_state = random_state
        
        # Sample initial particles (parallelized if n_threads > 1)
        if self.n_threads > 1:
            # Pre-generate seeds to avoid race conditions
            seeds = self.random_state.integers(0, 2**32, size=self.n_particles)
            
            def sample_particle(args):
                i, seed = args
                # Create a new RNG for each thread to avoid race conditions
                thread_rng = np.random.default_rng(seed)
                return self.model.sample_initial_state(thread_rng)
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                self.particles = np.array(list(executor.map(sample_particle, enumerate(seeds))))
        else:
            self.particles = np.array([
                self.model.sample_initial_state(self.random_state)
                for _ in range(self.n_particles)
            ])
        
        # Uniform weights
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Reset storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.n_unique_particles = []
        self.resampled_at = []
        self.weights_history = []
    
    def predict(self):
        """Prediction step: propagate particles through state transition."""
        # Sample new particles from state transition (parallelized if n_threads > 1)
        if self.n_threads > 1:
            # Pre-generate seeds to avoid race conditions
            seeds = self.random_state.integers(0, 2**32, size=self.n_particles)
            
            def propagate_particle(args):
                i, seed = args
                # Create a new RNG for each thread to avoid race conditions
                thread_rng = np.random.default_rng(seed)
                return self.model.sample_state_transition(self.particles[i], thread_rng)
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                self.particles = np.array(list(executor.map(propagate_particle, zip(range(self.n_particles), seeds))))
        else:
            self.particles = np.array([
                self.model.sample_state_transition(self.particles[i], self.random_state)
                for i in range(self.n_particles)
            ])
    
    def update(self, y: np.ndarray, timestep: int):
        """
        Update step: weight particles by observation likelihood.
        
        Args:
            y: Observation, shape (obs_dim,)
            timestep: Current timestep (for resampling tracking)
        """
        # Compute observation log-likelihoods (parallelized if n_threads > 1)
        if self.n_threads > 1:
            def compute_log_weight(i):
                return self.model.log_observation_prob(y, self.particles[i])
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                log_weights = np.array(list(executor.map(compute_log_weight, range(self.n_particles))))
        else:
            log_weights = np.array([
                self.model.log_observation_prob(y, self.particles[i])
                for i in range(self.n_particles)
            ])
        
        # Normalize weights (log-sum-exp trick for numerical stability)
        max_log_weight = np.max(log_weights)
        log_weights_normalized = log_weights - max_log_weight
        self.weights = np.exp(log_weights_normalized)
        self.weights = self.weights / np.sum(self.weights)
        
        self.weights_history.append(self.weights.copy())
        
        # Log-likelihood estimate: log p(y_t | y_{1:t-1})
        # This is the average likelihood: mean of exp(log_weights)
        log_lik = max_log_weight + np.log(np.mean(np.exp(log_weights_normalized)))
        self.log_likelihoods.append(log_lik)
        
        # Effective sample size
        ess = self._effective_sample_size()
        self.ess_history.append(ess)
        
        # Resample if ESS is too low
        if ess < self.resample_threshold * self.n_particles:
            self._systematic_resample(timestep)
    
    def _estimate_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean and covariance from weighted particles.
        
        Returns:
            Tuple of (mean, cov) where:
            - mean: Weighted mean, shape (state_dim,)
            - cov: Weighted covariance, shape (state_dim, state_dim)
        """
        # Weighted mean
        mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        
        # Weighted covariance
        diff = self.particles - mean
        cov = np.sum(
            self.weights[:, np.newaxis, np.newaxis] *
            np.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )
        
        return mean, cov
    
    def filter(self, observations: np.ndarray,
               random_state: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the filter on a sequence of observations.
        
        Args:
            observations: Array of shape (T, obs_dim) containing observations
            random_state: Optional numpy random generator
            
        Returns:
            Tuple of (means, covs) where:
            - means: Array of shape (T, state_dim) containing filtered state estimates
            - covs: Array of shape (T, state_dim, state_dim) containing filtered covariances
        """
        self.initialize(random_state)
        T = len(observations)
        
        for t in range(T):
            self.predict()
            self.update(observations[t], timestep=t)
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
        
        return np.array(self.means), np.array(self.covs)
    
    def get_diagnostics(self) -> dict:
        """
        Return diagnostic information about filter performance.
        
        Returns:
            Dictionary containing:
            - ess: Array of effective sample sizes over time
            - n_unique: Array of unique particle counts after each resampling
            - resampled_at: List of timesteps where resampling occurred
            - weights_history: Array of weights over time, shape (T, N)
            - log_likelihood: Total log-likelihood
            - resampling_rate: Fraction of timesteps where resampling occurred
        """
        return {
            'ess': np.array(self.ess_history),
            'n_unique': np.array(self.n_unique_particles),
            'resampled_at': self.resampled_at,
            'weights_history': np.array(self.weights_history),
            'log_likelihood': np.sum(self.log_likelihoods),
            'resampling_rate': len(self.resampled_at) / len(self.ess_history) if self.ess_history else 0
        }


class ExactDaumHuangFlow:
    """
    Exact Daum-Huang particle flow filter.
    
    Moves particles from prior to posterior along deterministic flow,
    avoiding the resampling step entirely.
    
    Flow equation: dx/dλ = K(λ) · [y - h(x)]
    where K(λ) = Cov[x, h(x)] @ inv(Cov[h(x), h(x)] + R)
    """
    
    def __init__(self, model, n_particles: int = 1000, 
                 n_lambda_steps: int = 100, integration_method: str = 'euler',
                 n_threads: Optional[int] = None):
        """
        Initialize Exact Daum-Huang Flow Filter.
        
        Args:
            model: StateSpaceModel instance with state transition and observation models
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            integration_method: 'euler' or 'rk4' for ODE integration
            n_threads: Number of threads for parallelization (default: CPU count, None = no threading)
        """
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.integration_method = integration_method
        
        # Threading setup
        if n_threads is None:
            self.n_threads = os.cpu_count() if os.cpu_count() else 1
        else:
            self.n_threads = max(1, int(n_threads))
        
        # Particles (all have equal weight 1/N)
        self.particles = None
        
        # Storage
        self.means = []
        self.covs = []
        
        # Random state
        self.random_state = np.random.default_rng()
    
    def _compute_observation_matrix(self, particles: np.ndarray) -> np.ndarray:
        """
        Evaluate observation function h(x) for all particles.
        
        Args:
            particles: Shape (N, state_dim)
            
        Returns:
            h_particles: Shape (N, obs_dim)
        """
        if self.n_threads > 1:
            def compute_h(i):
                return self.model.observation_function(particles[i])
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                h_particles = np.array(list(executor.map(compute_h, range(self.n_particles))))
        else:
            h_particles = np.array([
                self.model.observation_function(particles[i])
                for i in range(self.n_particles)
            ])
        return h_particles
    
    def _compute_gain(self, particles: np.ndarray, lambda_val: float) -> np.ndarray:
        """
        Compute exact Daum-Huang gain K(λ) from particle ensemble.
        
        The gain is computed as:
        K = Cov[x, h(x)] @ inv(Cov[h(x), h(x)] + R)
        
        where covariances are empirical estimates from particles.
        
        Args:
            particles: Current particle locations, shape (N, state_dim)
            lambda_val: Current pseudo-time λ
            
        Returns:
            K: Gain matrix, shape (state_dim, obs_dim)
        """
        # Evaluate h(x) for all particles
        h_particles = self._compute_observation_matrix(particles)
        
        # Empirical means
        x_mean = np.mean(particles, axis=0)  # (state_dim,)
        h_mean = np.mean(h_particles, axis=0)  # (obs_dim,)
        
        # Center the particles
        x_centered = particles - x_mean  # (N, state_dim)
        h_centered = h_particles - h_mean  # (N, obs_dim)
        
        # Compute covariances empirically
        # Cov[x, h(x)] = (1/N) Σ (x_i - x_mean)(h_i - h_mean)^T
        cov_x_h = (x_centered.T @ h_centered) / self.n_particles  # (state_dim, obs_dim)
        
        # Cov[h(x), h(x)] = (1/N) Σ (h_i - h_mean)(h_i - h_mean)^T
        cov_h_h = (h_centered.T @ h_centered) / self.n_particles  # (obs_dim, obs_dim)
        
        # Add observation noise covariance R
        R = self.model.observation_noise_cov
        S = cov_h_h + R  # (obs_dim, obs_dim)
        
        # Compute gain: K = Cov[x,h] @ inv(S)
        # Use stable inversion
        try:
            K = cov_x_h @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse if singular
            K = cov_x_h @ np.linalg.pinv(S)
        
        return K  # (state_dim, obs_dim)
    
    def _flow_step_euler(self, particles: np.ndarray, y: np.ndarray, 
                        lambda_val: float, d_lambda: float) -> np.ndarray:
        """
        Take one Euler step along the Daum-Huang flow.
        
        Flow equation: dx/dλ = K(x,λ) · [y - h(x)]
        Euler discretization: x_new = x + K · [y - h(x)] · dλ
        
        Args:
            particles: Current particles, shape (N, state_dim)
            y: Observation, shape (obs_dim,)
            lambda_val: Current pseudo-time λ
            d_lambda: Step size in λ
            
        Returns:
            Updated particles, shape (N, state_dim)
        """
        # Compute gain at current λ
        K = self._compute_gain(particles, lambda_val)  # (state_dim, obs_dim)
        
        # Evaluate h(x) for all particles
        h_particles = self._compute_observation_matrix(particles)  # (N, obs_dim)
        
        # Innovation for each particle: y - h(x_i)
        innovations = y - h_particles  # (N, obs_dim)
        
        # Flow update: dx = K @ innovation * dλ
        # K is (state_dim, obs_dim), innovations[i] is (obs_dim,)
        dx = innovations @ K.T  # (N, state_dim)
        
        # Update particles
        particles_new = particles + dx * d_lambda
        
        return particles_new
    
    def _flow_step_rk4(self, particles: np.ndarray, y: np.ndarray,
                       lambda_val: float, d_lambda: float) -> np.ndarray:
        """
        Take one RK4 step along the Daum-Huang flow.
        
        More accurate than Euler, especially for larger step sizes.
        """
        def flow_derivative(x_particles, lam):
            """Compute dx/dλ at given particles and λ."""
            K = self._compute_gain(x_particles, lam)
            h_particles = self._compute_observation_matrix(x_particles)
            innovations = y - h_particles
            return innovations @ K.T
        
        # RK4 stages
        k1 = flow_derivative(particles, lambda_val)
        k2 = flow_derivative(particles + 0.5 * d_lambda * k1, lambda_val + 0.5 * d_lambda)
        k3 = flow_derivative(particles + 0.5 * d_lambda * k2, lambda_val + 0.5 * d_lambda)
        k4 = flow_derivative(particles + d_lambda * k3, lambda_val + d_lambda)
        
        # Combine
        particles_new = particles + (d_lambda / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return particles_new
    
    def predict(self):
        """Prediction step: propagate particles through state transition."""
        if self.n_threads > 1:
            # Pre-generate seeds to avoid race conditions
            seeds = self.random_state.integers(0, 2**32, size=self.n_particles)
            
            def propagate_particle(args):
                i, seed = args
                # Create a new RNG for each thread to avoid race conditions
                thread_rng = np.random.default_rng(seed)
                return self.model.sample_state_transition(self.particles[i], thread_rng)
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                self.particles = np.array(list(executor.map(propagate_particle, zip(range(self.n_particles), seeds))))
        else:
            self.particles = np.array([
                self.model.sample_state_transition(self.particles[i], self.random_state)
                for i in range(self.n_particles)
            ])
    
    def update(self, y: np.ndarray):
        """
        Update step: move particles from prior to posterior via flow.
        
        NO RESAMPLING - particles maintain their identity throughout.
        All particles have equal weight 1/N.
        
        Args:
            y: Observation, shape (obs_dim,)
        """
        # Discretize pseudo-time λ ∈ [0,1]
        lambdas = np.linspace(0, 1, self.n_lambda_steps + 1)
        d_lambda = lambdas[1] - lambdas[0]
        
        # Flow integration
        particles_flow = self.particles.copy()
        
        for i, lambda_val in enumerate(lambdas[:-1]):
            # Take flow step
            if self.integration_method == 'euler':
                particles_flow = self._flow_step_euler(particles_flow, y, lambda_val, d_lambda)
            elif self.integration_method == 'rk4':
                particles_flow = self._flow_step_rk4(particles_flow, y, lambda_val, d_lambda)
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")
        
        # Update particles (now at λ=1, representing posterior)
        self.particles = particles_flow
    
    def _estimate_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean and covariance from equally-weighted particles.
        
        Returns:
            mean: Shape (state_dim,)
            cov: Shape (state_dim, state_dim)
        """
        mean = np.mean(self.particles, axis=0)
        diff = self.particles - mean
        cov = (diff.T @ diff) / self.n_particles
        return mean, cov
    
    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles from initial distribution."""
        if random_state is not None:
            self.random_state = random_state
        
        if self.n_threads > 1:
            # Pre-generate seeds to avoid race conditions
            seeds = self.random_state.integers(0, 2**32, size=self.n_particles)
            
            def sample_particle(args):
                i, seed = args
                # Create a new RNG for each thread to avoid race conditions
                thread_rng = np.random.default_rng(seed)
                return self.model.sample_initial_state(thread_rng)
            
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                self.particles = np.array(list(executor.map(sample_particle, enumerate(seeds))))
        else:
            self.particles = np.array([
                self.model.sample_initial_state(self.random_state)
                for _ in range(self.n_particles)
            ])
        
        # Reset storage
        self.means = []
        self.covs = []
    
    def filter(self, observations: np.ndarray,
               random_state: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the filter on a sequence of observations.
        
        Args:
            observations: Shape (T, obs_dim)
            random_state: Optional random generator
            
        Returns:
            means: Shape (T, state_dim)
            covs: Shape (T, state_dim, state_dim)
        """
        self.initialize(random_state)
        T = len(observations)
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
        
        return np.array(self.means), np.array(self.covs)
    
    def get_diagnostics(self) -> dict:
        """Return diagnostic information about flow convergence."""
        return {
            'final_particles': self.particles
        }


class LocalExactDaumHuangFlow(ExactDaumHuangFlow):
    """
    Local Exact Daum-Huang Flow - computes gain locally around each particle.
    
    Uses kernel density estimation to compute local covariances,
    better for multimodal or complex posteriors.
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
            n_threads: Number of threads for parallelization (default: CPU count, None = no threading)
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