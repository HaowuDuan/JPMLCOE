"""Exact Daum-Huang (EDH) Invertible Particle Flow Filter

Implements Algorithm 2 from Li & Coates paper exactly as specified.
"""

import numpy as np
from typing import Tuple, Optional
from ...core.model_base import StateSpaceModel
from ...core.types import FilterResult
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ..kalman.unscented_kalman import UnscentedKalmanFilter


class EDHParticleFlowFilter:
    """
    Exact Daum-Huang (EDH) Particle Flow Filter (Invertible Variant).
    
    Implements Algorithm 2 from the paper with explicit Euler integration
    and the author's recommended exponential step sizes.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        n_lambda_steps: int = 29,  # Author's recommended value
        filter_type: str = 'ekf',
        resample_threshold: float = 0.5,
        debug_mode: bool = False,
        **filter_kwargs
    ):
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.filter_type = filter_type
        self.resample_threshold = resample_threshold
        self.filter_kwargs = filter_kwargs
        self.debug_mode = debug_mode

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
        self.resampled_at = []
        self.n_unique_particles = []

        # Debug storage
        if self.debug_mode:
            self.debug_info = {
                'timesteps': [],
                'flow_steps': [],
                'particles_before_flow': [],
                'particles_after_flow': [],
                'A_matrices': [],
                'b_vectors': [],
                'H_jacobians': [],
                'jacobian_dets': [],
                'eigenvalues': [],
                'condition_numbers': [],
                'particle_stats': []
            }
        else:
            self.debug_info = None

        # Random state
        self.random_state = np.random.default_rng()

    def _create_filter(self, initial_mean: np.ndarray, initial_cov: np.ndarray):
        if self.filter_type == 'ekf':
            filt = ExtendedKalmanFilter(self.model, mean_0=initial_mean, Sigma_0=initial_cov)
        elif self.filter_type == 'ukf':
            ukf_kwargs = {k: v for k, v in self.filter_kwargs.items() if k != 'n_threads'}
            filt = UnscentedKalmanFilter(self.model, mean_0=initial_mean, Sigma_0=initial_cov, **ukf_kwargs)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        filt.mean = initial_mean.copy()
        filt.cov = initial_cov.copy()
        return filt

    def initialize(self, initial_mean: Optional[np.ndarray] = None,
                   initial_cov: Optional[np.ndarray] = None,
                   random_state: Optional[np.random.Generator] = None):
        if random_state is not None:
            self.random_state = random_state
        if initial_mean is None:
            # Sample from model's initial distribution instead of using zeros
            initial_mean = self.model.sample_initial_state(self.random_state)
        if initial_cov is None:
            # Use stationary variance if available, otherwise identity
            if hasattr(self.model, 'stationary_var'):
                initial_cov = np.eye(self.state_dim) * self.model.stationary_var
            else:
                initial_cov = np.eye(self.state_dim)

        self.particles = self.random_state.multivariate_normal(
            initial_mean, initial_cov, size=self.n_particles
        )
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.global_filter = self._create_filter(initial_mean, initial_cov)

        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

    def _generate_step_sizes(self) -> np.ndarray:
        """
        Generate exponentially spaced step sizes (author's recommendation).
        
        ε_j = ε_1 * q^(j-1), where q = 1.2 and Σε_j = 1
        """
        q = 1.2
        N = self.n_lambda_steps
        # ε_1 = (1-q)/(1-q^N) ensures sum = 1
        epsilon_1 = (1 - q) / (1 - q**N)
        step_sizes = epsilon_1 * (q ** np.arange(N))
        return step_sizes

    def _compute_A_b(self, eta_bar: np.ndarray, eta_bar_0: np.ndarray,
                     P: np.ndarray, z: np.ndarray, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flow parameters A(λ) and b(λ) per Equations (10) and (11).
        
        Args:
            eta_bar: Current mean η̄ for linearization
            eta_bar_0: Initial mean η̄_0 at λ=0 (used in b formula)
            P: Predicted covariance P_{k|k-1}
            z: Observation
            lambda_val: Current pseudo-time λ
        """
        H = self.model.observation_jacobian(eta_bar)
        R = self.model.observation_noise_cov

        # A(λ) = -1/2 * P @ H^T @ (λ*H@P@H^T + R)^{-1} @ H
        HPH = H @ P @ H.T
        S = lambda_val * HPH + R
        
        try:
            S_inv_H = np.linalg.solve(S, H)
        except np.linalg.LinAlgError:
            S_inv_H = np.linalg.lstsq(S, H, rcond=None)[0]
        
        A = -0.5 * P @ H.T @ S_inv_H

        # e(λ) = h(η̄) - H @ η̄
        h_eta_bar = self.model.observation_function(eta_bar)
        e = h_eta_bar - H @ eta_bar

        # b(λ) = (I + 2λA) @ [(I + λA) @ P @ H^T @ R^{-1} @ (z - e) + A @ η̄_0]
        I = np.eye(self.state_dim)
        
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
        
        innovation = P @ H.T @ R_inv @ (z.ravel() - e.ravel())
        inner = (I + lambda_val * A) @ innovation + A @ eta_bar_0
        b = (I + 2 * lambda_val * A) @ inner

        return A, b

    def predict(self):
        """
        Prediction step (Algorithm 2, lines 4-8).
        """
        # Store previous particles for weight calculation
        self.particles_prev = self.particles.copy()

        # Line 4: EKF/UKF prediction to get P_{k|k-1}
        # Feed back WEIGHTED ensemble mean to global filter
        ensemble_mean_prev = np.sum(self.weights[:, np.newaxis] * self.particles_prev, axis=0)
        self.global_filter.mean = ensemble_mean_prev
        self.global_filter.predict()
        
        # Store the predicted mean BEFORE particles are propagated
        # This is η̄_0 = g_k(x̂_{k-1}, 0) - the DETERMINISTIC prediction
        self.eta_bar_0 = self.global_filter.mean.copy()

        # Lines 5-8: Propagate particles η_0^i = g_k(x_{k-1}^i, v_k) WITH noise
        eta_0 = np.zeros_like(self.particles)
        for i in range(self.n_particles):
            eta_0[i] = self.model.sample_state_transition(
                self.particles_prev[i], self.random_state
            )
        
        self.particles = eta_0.copy()
        self.eta_0 = eta_0.copy()  # Store for weight calculation

    def update(self, y: np.ndarray):
        """
        Update step (Algorithm 2, lines 9-29).
        
        Exactly follows the pseudocode:
        - Line 9: η̄ = η̄_0 (start from deterministic predicted mean)
        - Line 10: λ = 0
        - Lines 11-18: Flow loop with λ incremented BEFORE computing A,b
        - Lines 19-25: Weight update
        """
        P = self.global_filter.cov
        
        # Line 9: Set η̄ = η̄_0 (the deterministic predicted mean from predict())
        eta_bar = self.eta_bar_0.copy()
        eta_bar_0 = self.eta_bar_0.copy()  # Keep reference for b(λ) formula

        # Line 10: λ = 0
        lambda_val = 0.0
        
        # Generate step sizes (author's exponential schedule)
        step_sizes = self._generate_step_sizes()
        
        # Debug: Store particles before flow
        if self.debug_mode:
            particles_before = self.particles.copy()
            timestep_debug = {
                'timestep': len(self.means),
                'observation': y.copy(),
                'particles_before': particles_before,
                'flow_steps': []
            }

        # Lines 11-18: Flow loop
        for j in range(self.n_lambda_steps):
            epsilon_j = step_sizes[j]

            # Line 12: λ = λ + ε_j (INCREMENT FIRST per paper pseudocode)
            lambda_val = lambda_val + epsilon_j

            # Line 13: Compute A(λ), b(λ) at η̄ with NEW λ
            A, b = self._compute_A_b(eta_bar, eta_bar_0, P, y, lambda_val)
            
            # Debug: Capture flow step diagnostics
            if self.debug_mode:
                H = self.model.observation_jacobian(eta_bar)
                
                # Compute eigenvalues and condition number of A
                try:
                    eigvals = np.linalg.eigvals(A)
                    cond_A = np.linalg.cond(A)
                except:
                    eigvals = np.array([np.nan])
                    cond_A = np.nan
                
                # For invertible filters, track det(I + epsilon * A)
                I = np.eye(self.state_dim)
                try:
                    jacob_det = np.linalg.det(I + epsilon_j * A)
                except:
                    jacob_det = np.nan
                
                flow_step_debug = {
                    'step': j,
                    'lambda': lambda_val,
                    'epsilon': epsilon_j,
                    'A_matrix': A.copy(),
                    'b_vector': b.copy(),
                    'H_jacobian': H.copy(),
                    'eigenvalues': eigvals,
                    'condition_number': cond_A,
                    'jacobian_det': jacob_det,
                    'particle_mean': np.mean(self.particles, axis=0),
                    'particle_std': np.std(self.particles, axis=0)
                }
                timestep_debug['flow_steps'].append(flow_step_debug)
            
            # Line 14: Migrate η̄
            eta_bar = eta_bar + epsilon_j * (A @ eta_bar + b)

            # Lines 15-17: Migrate particles (vectorized)
            drift = self.particles @ A.T + b
            self.particles = self.particles + epsilon_j * drift

        # Lines 19-25: Weight update with improved numerical stability
        eta_1 = self.particles

        # A. Log-likelihood: log p(z_k | η_1^i)
        log_likelihood = np.array([
            self.model.log_observation_prob(y, eta_1[i])
            for i in range(self.n_particles)
        ])

        # B. Log transition ratio using Cholesky for stability
        Q = self.model.process_noise_cov

        try:
            # Cholesky decomposition for numerical stability
            L_Q = np.linalg.cholesky(Q)
            
            # Compute f(x_{k-1}^i) for each particle
            f_prev = np.array([
                self.model.state_transition_mean(p) for p in self.particles_prev
            ])
            
            # Solve L_Q * z = diff to get normalized residuals
            diff_0 = self.eta_0 - f_prev
            diff_1 = eta_1 - f_prev
            
            z_0 = np.linalg.solve(L_Q, diff_0.T).T
            z_1 = np.linalg.solve(L_Q, diff_1.T).T
            
            mahal_0 = np.sum(z_0 ** 2, axis=1)
            mahal_1 = np.sum(z_1 ** 2, axis=1)
            
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            Q_inv = np.linalg.pinv(Q)
            f_prev = np.array([
                self.model.state_transition_mean(p) for p in self.particles_prev
            ])
            diff_0 = self.eta_0 - f_prev
            diff_1 = eta_1 - f_prev
            mahal_0 = np.sum((diff_0 @ Q_inv) * diff_0, axis=1)
            mahal_1 = np.sum((diff_1 @ Q_inv) * diff_1, axis=1)

        # Log transition ratio (constants cancel in the ratio)
        log_transition_ratio = -0.5 * (mahal_1 - mahal_0)

        # C. Combined weight update with robust normalization
        log_weights = (np.log(np.maximum(self.weights, 1e-300)) + 
                       log_likelihood + 
                       log_transition_ratio)

        # Robust normalization
        log_weights_max = np.max(log_weights)
        log_weights_normalized = log_weights - log_weights_max

        # Clip extreme values to prevent overflow/underflow
        log_weights_normalized = np.clip(log_weights_normalized, -30, 30)

        self.weights = np.exp(log_weights_normalized)
        weight_sum = np.sum(self.weights)

        if weight_sum < 1e-300 or not np.isfinite(weight_sum):
            # Emergency: reset to uniform weights
            print(f"Warning: Weight collapse at step. Resetting to uniform.")
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= weight_sum

        self.weights_history.append(self.weights.copy())

        # ESS and resampling
        ess = self._effective_sample_size()
        self.ess_history.append(ess)

        if ess < self.resample_threshold * self.n_particles:
            self._systematic_resample()
            self.resampled_at.append(len(self.ess_history) - 1)
            self.n_unique_particles.append(len(np.unique(self.particles, axis=0)))

        # Log-likelihood for model evidence
        max_ll = np.max(log_likelihood)
        log_lik = max_ll + np.log(np.mean(np.exp(log_likelihood - max_ll)))
        self.log_likelihoods.append(log_lik)
        
        # Debug: Store after-flow diagnostics
        if self.debug_mode:
            timestep_debug['particles_after'] = self.particles.copy()
            timestep_debug['weights'] = self.weights.copy()
            timestep_debug['ess'] = ess
            timestep_debug['weight_stats'] = {
                'min': np.min(self.weights),
                'max': np.max(self.weights),
                'mean': np.mean(self.weights),
                'std': np.std(self.weights)
            }
            timestep_debug['particle_stats_after'] = {
                'mean': np.mean(self.particles, axis=0),
                'cov': np.cov(self.particles.T),
                'min': np.min(self.particles, axis=0),
                'max': np.max(self.particles, axis=0)
            }
            self.debug_info['timesteps'].append(timestep_debug)

        # Line 26: Update global filter
        ensemble_mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        self.global_filter.mean = ensemble_mean
        self.global_filter.update(y)

    def _effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def _systematic_resample(self):
        n = self.n_particles
        u = self.random_state.uniform(0, 1/n)
        indices = np.zeros(n, dtype=int)
        cumsum = np.cumsum(self.weights)
        i = 0
        for j in range(n):
            u_j = u + j / n
            while u_j > cumsum[i] and i < n - 1:
                i += 1
            indices[j] = i
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n

    def _estimate_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
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
               initial_cov: Optional[np.ndarray] = None) -> FilterResult:
        self.initialize(initial_mean, initial_cov)
        T = len(observations)

        for t in range(T):
            self.predict()
            self.update(observations[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)

        resampling_rate = len(self.resampled_at) / T if T > 0 else 0.0

        return FilterResult(
            means=np.array(self.means),
            covs=np.array(self.covs),
            log_likelihood=np.sum(self.log_likelihoods) if self.log_likelihoods else None,
            log_likelihoods=np.array(self.log_likelihoods) if self.log_likelihoods else None,
            ess=np.array(self.ess_history),
            weights_history=np.array(self.weights_history),
            resampled_at=self.resampled_at,
            n_unique=np.array(self.n_unique_particles) if self.n_unique_particles else None,
            metadata={
                'filter_type': 'EDHParticleFlowFilter',
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'resampling_rate': resampling_rate
            }
        )
