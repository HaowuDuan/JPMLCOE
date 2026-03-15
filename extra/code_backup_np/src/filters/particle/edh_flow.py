"""Exact Daum-Huang (EDH) particle flow filter."""

import numpy as np
from typing import Tuple, Optional
from .flow_base import FlowFilterBase
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ..kalman.unscented_kalman import UnscentedKalmanFilter
from ...utils.linalg import safe_solve
from ...utils.ode_solvers import euler_step, rk4_step


class ExactDaumHuangFlow(FlowFilterBase):
    """
    Exact Daum-Huang particle flow filter with feedback mechanism.

    Assumes Gaussian prior/predictive posterior with covariance P.
    Linearizes nonlinear measurement model at mean particle location.

    Flow equation: dη/dλ = A(λ)η + b(λ)
    where A(λ) and b(λ) are computed from Equations (10) and (11).

    All particles have equal weight 1/N throughout.
    
    Key modification: Implements feedback from particle ensemble to EKF/UKF
    as described in Algorithm 2 of the paper.
    """

    def __init__(self, model, n_particles: int = 1000,
                 n_lambda_steps: int = 100,
                 integration_method: str = 'euler',
                 filter_type: str = 'ekf',
                 n_threads: Optional[int] = None,
                 debug_mode: bool = False):
        """
        Initialize Exact Daum-Huang flow filter.

        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            integration_method: 'euler' or 'rk4' for ODE integration
            filter_type: 'ekf' or 'ukf' for covariance guidance
            n_threads: Number of threads (None = use CPU count, ignored for now)
            debug_mode: If True, collect detailed diagnostics
        """
        super().__init__(model, n_particles, n_lambda_steps, integration_method, n_threads)
        self.filter_type = filter_type
        self.debug_mode = debug_mode
        self.predicted_cov = None  # P_{k|k-1} from EKF/UKF
        self.eta_bar_0 = None      # η̄_0 (mean at λ=0)
        self.global_filter = None  # EKF/UKF for covariance guidance

        # Cache R_inv if R is constant
        self.R_inv_cache = None
        self.L_cache = None
        
        # TODO: debug mode is not integreated with the tests 
        if self.debug_mode:
            self.debug_info = {
                'timesteps': [],
                'flow_steps': [],
                'particles_before_flow': [],
                'particles_after_flow': [],
                'A_matrices': [],
                'b_vectors': [],
                'H_jacobians': [],
                'eigenvalues': [],
                'condition_numbers': [],
                'particle_stats': []
            }
        else:
            self.debug_info = None
        
        # Generate exponential lambda schedule (same as ledh_invertible)
        self._generate_lambda_steps()

    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles and global EKF/UKF for covariance guidance."""
        # Initialize particles from base class
        super().initialize(random_state)

        # Compute empirical mean and covariance from particles
        ensemble_mean = np.mean(self.particles, axis=0)
        
        if self.state_dim == 1:
            initial_cov = np.var(self.particles).reshape(1, 1)
        else:
            initial_cov = np.cov(self.particles.T)

        # Initialize global filter for covariance guidance
        if self.filter_type == 'ekf':
            self.global_filter = ExtendedKalmanFilter(self.model, mean_0=ensemble_mean, Sigma_0=initial_cov)
        elif self.filter_type == 'ukf':
            self.global_filter = UnscentedKalmanFilter(self.model, mean_0=ensemble_mean, Sigma_0=initial_cov)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        self.global_filter.mean = ensemble_mean.copy()
        self.global_filter.cov = initial_cov.copy()
        self.predicted_cov = initial_cov.copy()

    def _generate_lambda_steps(self):
        """
        Generate exponential decay schedule for lambda steps.
        Same as in ledh_invertible.py for consistency.
        
        Uses geometric sequence with ratio q=1.2, normalized to sum to 1.
        """
        q = 1.2
        epsilon_1 = (1 - q) / (1 - q**self.n_lambda_steps)
        self.lambda_steps = epsilon_1 * q**np.arange(self.n_lambda_steps)
        self.lambda_steps = self.lambda_steps / np.sum(self.lambda_steps)

    def _compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Use model's analytical Jacobian (more accurate than finite diff)."""
        return self.model.observation_jacobian(x)
    
    # TODO: double check the flow parameters   
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
        H_lambda = self._compute_jacobian(eta_bar_lambda)  # (obs_dim, state_dim)

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
        """Compute drift for EDH flow: dη/dλ = Aη + b (vectorized)."""
        return particles @ A.T + b

    def _compute_drift_time_dependent(self, particles: np.ndarray, lambda_val: float,
                                       observation: np.ndarray) -> np.ndarray:
        """
        Compute time-dependent drift for RK4 integration.

        Recomputes A(λ) and b(λ) at each λ value, using mean of particles for linearization.
        """
        eta_bar = np.mean(particles, axis=0)
        A, b = self._compute_flow_params(eta_bar, lambda_val, observation)
        return particles @ A.T + b

    def predict(self):
        """
        Prediction step - propagate particles and get covariance from EKF/UKF.

        Uses global EKF/UKF to estimate P_{k|k-1} for flow guidance.

        Implements feedback from particle ensemble to EKF/UKF as described
        in Algorithm 2 of the paper.
        """
        # FEEDBACK MECHANISM: Update global filter mean to ensemble mean
        # This is the key modification from Algorithm 2
        # Do NOT blend empirical covariances - this causes covariance explosion

        ensemble_mean = np.mean(self.particles, axis=0)
        self.global_filter.mean = ensemble_mean.copy()

        # Run global filter prediction to get P_{k|k-1}
        # The global filter maintains its own covariance via EKF/UKF equations
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.copy()

        # Store η̄_0: the DETERMINISTIC predicted mean (from particles mean)
        self.eta_bar_0 = self.global_filter.mean.copy()

        # Propagate particles through state transition (from base class)
        super().predict()

    def update(self, y: np.ndarray):
        """
        Update step: flow particles from λ=0 to λ=1.

        All particles maintain equal weight 1/N (no resampling).

        After particle flow completes, the ensemble mean is fed back to
        the EKF/UKF before its update step (Algorithm 2, line 18).

        Args:
            y: Observation, shape (obs_dim,)
        """
        # Reset R_inv cache for this update
        self.R_inv_cache = None
        self.L_cache = None

        # Use exponential lambda schedule
        particles_flow = self.particles.copy()
        lambda_val = 0.0
        
        # Debug: Store particles before flow
        if self.debug_mode:
            particles_before = particles_flow.copy()
            timestep_debug = {
                'timestep': len(self.means),
                'observation': y.copy(),
                'particles_before': particles_before,
                'flow_steps': []
            }

      
        # Line 12: λ = λ + ε_j (increment FIRST per paper)
        # Line 13: Calculate A_j(λ), b_j(λ) at NEW λ
        # Line 14: Migrate particles

        eta_bar=self.eta_bar_0
        
        for i in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[i]
            lambda_val += d_lambda  # Algorithm 2, Line 12: increment FIRST
            
            # Debug: Capture flow step diagnostics (sample steps)
            if self.debug_mode and i % 10 == 0:
                # FIX: Correct number of arguments (was passing 4, should be 3)
                A, b = self._compute_flow_params(eta_bar, lambda_val, y)
                H = self.model.observation_jacobian(eta_bar)
                
                try:
                    eigvals = np.linalg.eigvals(A)
                    cond_A = np.linalg.cond(A)
                except:
                    eigvals = np.array([np.nan])
                    cond_A = np.nan
                
                flow_step_debug = {
                    'step': i,
                    'lambda': lambda_val,
                    'epsilon': d_lambda,
                    'A_matrix': A.copy(),
                    'b_vector': b.copy(),
                    'H_jacobian': H.copy(),
                    'eigenvalues': eigvals,
                    'condition_number': cond_A,
                    'particle_mean': np.mean(particles_flow, axis=0),
                    'particle_std': np.std(particles_flow, axis=0)
                }
                timestep_debug['flow_steps'].append(flow_step_debug)
            
            if self.integration_method == 'euler':
                # Compute A, b at current λ and use shared euler_step
                A, b = self._compute_flow_params(eta_bar, lambda_val, y)
               
                particles_flow = euler_step(particles_flow, self._compute_drift, d_lambda, A, b)
            elif self.integration_method == 'rk4':
                # Use shared rk4_step with time-dependent drift (λ advances at each stage)
                
                particles_flow = rk4_step(
                    particles_flow, self._compute_drift_time_dependent, d_lambda,
                    y, t=lambda_val
                )
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")
            eta_bar = np.mean(particles_flow, axis=0)
        # Particles at λ=1 represent posterior
        self.particles = particles_flow
        
        # Debug: Store after-flow diagnostics
        if self.debug_mode:
            timestep_debug['particles_after'] = self.particles.copy()
            timestep_debug['particle_stats_after'] = {
                'mean': np.mean(self.particles, axis=0),
                'cov': np.cov(self.particles.T),
                'min': np.min(self.particles, axis=0),
                'max': np.max(self.particles, axis=0)
            }
            self.debug_info['timesteps'].append(timestep_debug)

        
        # Update global filter for next prediction cycle
        self.global_filter.update(y)

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about flow convergence."""
        return {
            'final_particles': self.particles,
            'predicted_cov': self.predicted_cov,
            'global_filter_mean': self.global_filter.mean,
            'global_filter_cov': self.global_filter.cov
        }
