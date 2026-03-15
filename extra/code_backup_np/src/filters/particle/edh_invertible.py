"""Exact Daum-Huang (EDH) Invertible Particle Flow Filter

Implements Algorithm 2 from Li & Coates paper exactly as specified.
"""

import numpy as np
from typing import Tuple, Optional
from ...core.model_base import StateSpaceModel
from ...core.types import FilterResult
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ..kalman.unscented_kalman import UnscentedKalmanFilter
from ...utils.linalg import safe_solve
from ...utils.distributions import compute_flow_weights
from ...utils.ode_solvers import euler_step


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
        self.predicted_cov = None  # P_{k|k-1} from EKF/UKF
        self.eta_bar_0 = None      # η̄_0 (mean at λ=0)

        # Cache R_inv if R is constant
        self.R_inv_cache = None
        self.L_cache = None

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

    def _compute_flow_params(self, eta_bar_lambda: np.ndarray,
                            lambda_val: float,
                            observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute A(λ) and b(λ) from Equations (10) and (11).

        Args:
            eta_bar_lambda: Current mean particle η̄_λ for linearization
            lambda_val: Current pseudo-time λ ∈ [0,1]
            observation: Measurement z

        Returns:
            A: Matrix A(λ), shape (state_dim, state_dim)
            b: Vector b(λ), shape (state_dim,)
        """
        P = self.predicted_cov
        R = self.model.observation_noise_cov

        # Cache R_inv (computed once per update)
        if self.R_inv_cache is None:
            try:
                # Cholesky decomposition: R = L @ L.T
                L = np.linalg.cholesky(R)
                # Solve R @ R_inv = I using forward/backward substitution
                self.R_inv_cache = np.linalg.inv(L.T) @ np.linalg.inv(L)
                self.L_cache = L
            except np.linalg.LinAlgError:
                # R is not positive definite - this indicates a problem
                raise ValueError("Covariance matrix R must be positive definite")

        # Linearize at current mean: H(λ) = ∂h/∂x |_(η̄_λ)
        H_lambda = self.model.observation_jacobian(eta_bar_lambda)  # (obs_dim, state_dim)

        # Compute A(λ) from Equation (10)
        # A(λ) = -1/2 * P @ H(λ)^T @ (λ*H(λ)@P@H(λ)^T + R)^(-1) @ H(λ)

        HPH = H_lambda @ P @ H_lambda.T
        S = lambda_val * HPH + R

        # Solve S @ S_inv_H = H_lambda using safe_solve
        S_inv_H = safe_solve(S, H_lambda, method='cholesky')

        # Then compute A(λ)
        A_lambda = -0.5 * P @ H_lambda.T @ S_inv_H

        # Compute e(λ) for b(λ) - Equation (11)
        # e(λ) = h(η̄_λ, 0) - H(λ)@η̄_λ
        h_eta_bar = self.model.observation_function(eta_bar_lambda)
        e_lambda = h_eta_bar - H_lambda @ eta_bar_lambda  # (obs_dim,)

        # Compute b(λ) from Equation (11)
        # b(λ) = (I + 2λA(λ))[(I + λA(λ))P@H(λ)^T@R^(-1)@(z - e(λ)) + A(λ)@η̄_0]
        I = np.eye(self.state_dim)

        term1 = (I + lambda_val * A_lambda) @ P @ H_lambda.T @ self.R_inv_cache @ (observation - e_lambda)
        term2 = A_lambda @ self.eta_bar_0
        b_lambda = (I + 2 * lambda_val * A_lambda) @ (term1 + term2)  # (state_dim,)

        return A_lambda, b_lambda

    def _compute_drift(self, particles: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute drift for EDH flow: dη/dλ = Aη + b (vectorized).

        Works for both single vector (d,) and batch (N, d) via broadcasting.
        """
        return particles @ A.T + b

    def predict(self):
        """
        Prediction step (Algorithm 2, lines 4-8).
        """
        # Store previous particles for weight calculation
        self.particles_prev = self.particles.copy()

        # Line 4: EKF/UKF prediction to get P_{k|k-1}
        # Feed back WEIGHTED ensemble mean to global filter
        ensemble_mean_prev = np.sum(self.weights[:, np.newaxis] * self.particles_prev, axis=0)
        self.global_filter.mean = ensemble_mean_prev.copy()
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.copy()

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
        # Reset R_inv cache for this update
        self.R_inv_cache = None
        self.L_cache = None

        # Line 9: Set η̄ = η̄_0 (the deterministic predicted mean from predict())
        eta_bar = self.eta_bar_0.copy()

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
            A, b = self._compute_flow_params(eta_bar, lambda_val, y)
            
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
            
            # Line 14: Migrate η̄ using euler_step
            eta_bar = euler_step(eta_bar, self._compute_drift, epsilon_j, A, b)

            # Lines 15-17: Migrate particles (vectorized) using euler_step
            self.particles = euler_step(self.particles, self._compute_drift, epsilon_j, A, b)

        # Lines 19-25: Weight update using shared utility
        eta_1 = self.particles.copy()

        # Compute weights using shared utility (no Jacobians for EDH)
        self.weights = compute_flow_weights(
            eta_1=eta_1,
            eta_0=self.eta_0,
            particles_prev=self.particles_prev,
            observation=y,
            model=self.model,
            prev_weights=self.weights,
            jacobians=None,  # EDH doesn't use Jacobians
            clip_range=(-30, 30)
        )

        self.weights_history.append(self.weights.copy())

        # Log-likelihood for model evidence (compute separately for logging)
        log_likelihood = np.array([
            self.model.log_observation_prob(y, eta_1[i])
            for i in range(self.n_particles)
        ])

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
