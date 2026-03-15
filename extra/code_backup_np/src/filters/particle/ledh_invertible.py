"""Local Exact Daum-Huang (LEDH) Invertible Particle Flow Filter. Algorithm 1"""

import numpy as np
from typing import Tuple, Optional
from ...core.model_base import StateSpaceModel
from ...core.types import FilterResult
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ..kalman.unscented_kalman import UnscentedKalmanFilter
from ...utils.linalg import safe_solve
from ...utils.distributions import compute_flow_weights
from ...utils.ode_solvers import euler_step


class LEDHParticleFlowFilter:
    """
    Local Exact Daum-Huang (LEDH) Invertible Particle Flow Filter -  Algorithm 1.

    Fixes applied:
    - Corrected drift term in `_compute_A_b` to use current linearization point `eta_bar` instead of initial `eta_bar_0`.
    - Improved numerical stability using `np.linalg.solve`.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        n_lambda_steps: int = 29,
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

        self.particles = None
        self.weights = None
        self.particle_filters = []
        self.particle_covs = None
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []
        self.random_state = np.random.default_rng()

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

        self._generate_lambda_steps()

    def _generate_lambda_steps(self):
        q = 1.2
        epsilon_1 = (1 - q) / (1 - q**self.n_lambda_steps)
        self.lambda_steps = epsilon_1 * q**np.arange(self.n_lambda_steps)
        self.lambda_steps = self.lambda_steps / np.sum(self.lambda_steps)

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

        self.particle_filters = []
        self.particle_covs = np.zeros((self.n_particles, self.state_dim, self.state_dim))
        for i in range(self.n_particles):
            filt = self._create_filter(self.particles[i], initial_cov)
            self.particle_filters.append(filt)
            self.particle_covs[i] = initial_cov.copy()

        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

    def _compute_A_b(self, eta_bar: np.ndarray, eta_bar_0: np.ndarray, P: np.ndarray,
                     z: np.ndarray, lambda_val: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flow parameters A^i(λ) and b^i(λ) per Eqs (13)-(14).
        
        Args:
            eta_bar: Current linearization point η̄^i (migrates with flow)
            eta_bar_0: Initial deterministic prediction η̄_0^i (constant, used in b formula)
            P: Particle covariance P^i
            z: Observation
            lambda_val: Current pseudo-time λ
        """
        # Get observation Jacobian at current linearization point η̄^i (Line 15)
        H = self.model.observation_jacobian(eta_bar)
        R = self.model.observation_noise_cov

        # A = -0.5 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
        PHt = P @ H.T
        HPH = H @ PHt
        S = lambda_val * HPH + R
        
        # Solve S @ K_aux.T = PHt.T, i.e., K_aux = PHt @ S^{-1}
        K_aux = safe_solve(S, PHt.T, method='default').T
        A = -0.5 * K_aux @ H

        # e = h(η̄) - H η̄ (linearization error)
        h_eta = self.model.observation_function(eta_bar)
        e = h_eta - H @ eta_bar

        # b = (I + 2λA) * [ (I + λA) * P H^T R^{-1} (z - e) + A η̄_0 ]
        # NOTE: Per pseudocode Line 14, use η̄_0 (INITIAL), not current η̄
        I = np.eye(self.state_dim)
        term1 = I + lambda_val * A
        term2 = I + 2 * lambda_val * A
        
        # Solve R @ (PHt_Rinv).T = PHt.T, i.e., PHt_Rinv = PHt @ R^{-1}
        PHt_Rinv = safe_solve(R, PHt.T, method='default').T
        b_inner = (term1 @ PHt_Rinv @ (z - e)) + (A @ eta_bar_0)  # Use eta_bar_0!
        b = term2 @ b_inner

        return A, b

    def _compute_drift_single(self, x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute drift for a single state vector: dx/dλ = Ax + b."""
        return A @ x + b

    def predict(self):
        self.particles_prev = self.particles.copy()
        self.eta_0 = np.zeros_like(self.particles)
        self.eta_bar_0 = np.zeros_like(self.particles)

        for i in range(self.n_particles):
            self.particle_filters[i].mean = self.particles_prev[i].copy()
            self.particle_filters[i].predict()
            self.particle_covs[i] = self.particle_filters[i].cov.copy()
            
            self.eta_bar_0[i] = self.model.state_transition_mean(self.particles_prev[i])
            self.eta_0[i] = self.model.sample_state_transition(
                self.particles_prev[i], self.random_state
            )

    def update(self, y: np.ndarray):
        eta_1 = self.eta_0.copy()
        theta = np.ones(self.n_particles)
        eta_bar = self.eta_bar_0.copy()

        # Debug: Store particles before flow
        if self.debug_mode:
            particles_before = eta_1.copy()
            timestep_debug = {
                'timestep': len(self.means),
                'observation': y.copy(),
                'particles_before': particles_before,
                'flow_steps': []
            }

        lambda_val = 0.0

        for j in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[j]
            lambda_val += d_lambda  # Algorithm 1, Line 12: increment FIRST per paper

            # Debug: Store per-step diagnostics (sampling from particles)
            if self.debug_mode and j % 5 == 0:  # Sample every 5 steps to avoid too much data
                flow_step_debug = {
                    'step': j,
                    'lambda': lambda_val,
                    'epsilon': d_lambda,
                    'jacobian_dets': [],
                    'particle_mean': np.mean(eta_1, axis=0),
                    'particle_std': np.std(eta_1, axis=0)
                }

            for i in range(self.n_particles):
                P_i = self.particle_covs[i]

                # Per pseudocode Line 14-15: use eta_bar[i] for linearization, eta_bar_0[i] for b formula
                A_i, b_i = self._compute_A_b(eta_bar[i], self.eta_bar_0[i], P_i, y, lambda_val)

                # Use euler_step for both streams (shared A_i, b_i)
                eta_bar[i] = euler_step(eta_bar[i], self._compute_drift_single, d_lambda, A_i, b_i)
                eta_1[i] = euler_step(eta_1[i], self._compute_drift_single, d_lambda, A_i, b_i)

                # Track Jacobian determinant for weight computation
                M_i = np.eye(self.state_dim) + d_lambda * A_i
                det_M_i = np.linalg.det(M_i)
                theta[i] *= np.abs(det_M_i)

                # Debug: Sample some particles
                if self.debug_mode and j % 5 == 0 and i < 5:  # First 5 particles
                    flow_step_debug['jacobian_dets'].append(det_M_i)

            if self.debug_mode and j % 5 == 0:
                timestep_debug['flow_steps'].append(flow_step_debug)

        self.particles = eta_1

        # Compute weights using shared utility (with Jacobians for LEDH)
        self.weights = compute_flow_weights(
            eta_1=eta_1,
            eta_0=self.eta_0,
            particles_prev=self.particles_prev,
            observation=y,
            model=self.model,
            prev_weights=self.weights,
            jacobians=theta,  # LEDH uses Jacobian determinants
            clip_range=(-30, 30)
        )

        self.weights_history.append(self.weights.copy())

        # Compute observation probs separately for marginal likelihood calculation
        log_obs_probs = np.array([
            self.model.log_observation_prob(y, self.particles[i])
            for i in range(self.n_particles)
        ])
        
        # Marginal likelihood: use only observation probabilities (not full weights)
        # This matches EDH invertible and gives meaningful log-likelihood values
        max_ll = np.max(log_obs_probs)
        log_lik = max_ll + np.log(np.mean(np.exp(log_obs_probs - max_ll)))
        self.log_likelihoods.append(log_lik)

        for i in range(self.n_particles):
            self.particle_filters[i].mean = self.particles[i].copy()
            self.particle_filters[i].update(y)
            self.particle_covs[i] = self.particle_filters[i].cov.copy()

        ess = self._effective_sample_size()
        self.ess_history.append(ess)

        if ess < self.resample_threshold * self.n_particles:
            self._systematic_resample()
            self.resampled_at.append(len(self.ess_history) - 1)
            n_unique = len(np.unique(self.particles, axis=0))
            self.n_unique_particles.append(n_unique)
        
        # Debug: Store after-flow diagnostics
        if self.debug_mode:
            timestep_debug['particles_after'] = self.particles.copy()
            timestep_debug['weights'] = self.weights.copy()
            timestep_debug['theta'] = theta.copy()
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

    def _effective_sample_size(self) -> float:
        sum_sq_weights = np.sum(self.weights ** 2)
        if sum_sq_weights == 0:
            return 0.0
        return 1.0 / sum_sq_weights

    def _systematic_resample(self):
        n = self.n_particles
        u = self.random_state.uniform(0, 1/n)
        indices = np.zeros(n, dtype=int)
        cumsum = np.cumsum(self.weights)
        i = 0
        for j in range(n):
            u_j = u + j / n
            while i < n - 1 and u_j > cumsum[i]:
                i += 1
            indices[j] = i
        
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n
        
        new_filters = []
        new_covs = np.zeros_like(self.particle_covs)
        for idx_new, idx_old in enumerate(indices):
            filt = self._create_filter(
                self.particles[idx_new],
                self.particle_covs[idx_old]
            )
            new_filters.append(filt)
            new_covs[idx_new] = self.particle_covs[idx_old].copy()

        self.particle_filters = new_filters
        self.particle_covs = new_covs

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
                'filter_type': 'LEDHParticleFlowFilter',
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'resampling_rate': resampling_rate
            }
        )
