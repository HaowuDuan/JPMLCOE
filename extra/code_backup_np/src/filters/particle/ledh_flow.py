"""Local Exact Daum-Huang (LEDH) particle flow filter"""

import numpy as np
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from .flow_base import FlowFilterBase
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ..kalman.unscented_kalman import UnscentedKalmanFilter
from ...utils.linalg import safe_solve
from ...utils.ode_solvers import euler_step


class LocalExactDaumHuangFlow(FlowFilterBase):
    """
    Local Exact Daum-Huang Flow with per-particle linearization.

    Key modifications from global EDH:
    1. Linearize measurement function at EACH particle location (not just at mean)
    2. Use GLOBAL predictive covariance P_{k|k-1} from a single EKF/UKF
    3. Compute A_i and b_i matrices individually for each particle
    
    This is modification (ii) from Section 3 of the paper.

    Flow equation: dx_i/dλ = A_i(λ) @ x_i + b_i(λ)
    where A_i uses local linearization H_i at particle i and global covariance P_{k|k-1}.
    """

    def __init__(self, model, n_particles: int = 1000,
                 n_lambda_steps: int = 100,
                 integration_method: str = 'euler',
                 filter_type: str = 'ekf',
                 use_feedback: bool = True,
                 regularization: float = 1e-8,
                 n_threads: Optional[int] = None,
                 debug_mode: bool = False):
        """
        Initialize Local Exact Daum-Huang flow filter.

        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            integration_method: 'euler' or 'rk4'
            filter_type: 'ekf' or 'ukf' for global covariance guidance
            use_feedback: If True, feed DH mean back to EKF/UKF
            regularization: Small value added to diagonal of S matrix for numerical stability
            n_threads: Number of threads for parallelization (None = auto)
            debug_mode: If True, collect detailed diagnostics
        """
        super().__init__(model, n_particles, n_lambda_steps, integration_method, n_threads)
        self.filter_type = filter_type
        self.use_feedback = use_feedback
        self.regularization = regularization
        self.debug_mode = debug_mode

        # Single GLOBAL EKF/UKF for covariance guidance
        self.global_filter = None
        self.predicted_cov = None  # P_{k|k-1} used for ALL particles

        # Cache for optimization
        self.R_inv_cache = None
        self.L_cache = None
        
        # Store mean at λ=0 for b computation
        self.eta_bar_0 = None
        
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

        # Compute empirical mean and covariance from all particles
        ensemble_mean = np.mean(self.particles, axis=0)

        if self.state_dim == 1:
            initial_cov = np.var(self.particles).reshape(1, 1)
        else:
            initial_cov = np.cov(self.particles.T)

        # Initialize SINGLE GLOBAL filter at ensemble mean
        if self.filter_type == 'ekf':
            self.global_filter = ExtendedKalmanFilter(
                self.model, mean_0=ensemble_mean, Sigma_0=initial_cov
            )
        elif self.filter_type == 'ukf':
            self.global_filter = UnscentedKalmanFilter(
                self.model, mean_0=ensemble_mean, Sigma_0=initial_cov
            )
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
        """Use model's analytical Jacobian."""
        return self.model.observation_jacobian(x)

    def _compute_local_flow_params(
        self, 
        particle: np.ndarray,
        lambda_val: float,
        observation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute A_i(λ) and b_i(λ) for particle i using LOCAL linearization.

        Key difference from global version:
        - H is computed at particle location (not mean)
        - P is still GLOBAL predictive covariance
        - eta_bar_0 is still GLOBAL mean at λ=0

        Args:
            particle: Current particle position, shape (state_dim,)
            lambda_val: Current pseudo-time λ ∈ [0,1]
            observation: Measurement z, shape (obs_dim,)

        Returns:
            A_i: Matrix A_i(λ), shape (state_dim, state_dim)
            b_i: Vector b_i(λ), shape (state_dim,)
        """
        P = self.predicted_cov  # GLOBAL covariance
        R = self.model.observation_noise_cov

        # Cache R_inv (computed once per update)
        if self.R_inv_cache is None:
            try:
                L = np.linalg.cholesky(R)
                self.R_inv_cache = np.linalg.inv(L.T) @ np.linalg.inv(L)
                self.L_cache = L
            except np.linalg.LinAlgError:
                raise ValueError("Covariance matrix R must be positive definite")

        # LOCAL linearization: H_i = ∂h/∂x |_(x_i)
        H_i = self._compute_jacobian(particle)  # (obs_dim, state_dim)

        # Compute A_i(λ) from Equation (10) with H_i
        # A_i(λ) = -1/2 * P @ H_i^T @ (λ*H_i@P@H_i^T + R)^(-1) @ H_i
        HPH = H_i @ P @ H_i.T
        S = lambda_val * HPH + R
        
        # Adaptive regularization based on matrix magnitude
        # Add a fraction of the diagonal to ensure positive definiteness
        min_diag = max(np.min(np.diag(S)), self.regularization)
        reg_strength = max(min_diag * 0.01, self.regularization)
        S = S + reg_strength * np.eye(S.shape[0])

        # Use safe_solve for robust linear system solving
        S_inv_H = safe_solve(S, H_i, method='cholesky')

        A_i = -0.5 * P @ H_i.T @ S_inv_H

        # Compute e_i(λ) for b_i(λ) using LOCAL particle
        # e_i(λ) = h(x_i) - H_i @ x_i
        h_particle = self.model.observation_function(particle)
        e_i = h_particle - H_i @ particle  # (obs_dim,)

        # Compute b_i(λ) from Equation (11) adapted for local linearization
        # b_i(λ) = (I + 2λA_i)[(I + λA_i)P@H_i^T@R^(-1)@(z - e_i) + A_i@η̄_0]
        # Note: We still use GLOBAL mean η̄_0, not individual particle_0
        I = np.eye(self.state_dim)

        term1 = (I + lambda_val * A_i) @ P @ H_i.T @ self.R_inv_cache @ (observation - e_i)
        term2 = A_i @ self.eta_bar_0  # Use GLOBAL mean at λ=0
        b_i = (I + 2 * lambda_val * A_i) @ (term1 + term2)  # (state_dim,)

        return A_i, b_i

    def _compute_drift_local(self, particles: np.ndarray, lambda_val: float,
                             observation: np.ndarray) -> np.ndarray:
        """
        Compute per-particle drift for LEDH flow: each particle gets its own A_i, b_i.

        Args:
            particles: Current particles, shape (N, state_dim)
            lambda_val: Current pseudo-time λ
            observation: Measurement z

        Returns:
            drift: Per-particle drift, shape (N, state_dim)
        """
        drift = np.zeros_like(particles)
        for i in range(len(particles)):
            A_i, b_i = self._compute_local_flow_params(particles[i], lambda_val, observation)
            particle_drift = A_i @ particles[i] + b_i

            # Limit drift magnitude to prevent explosive growth
            max_drift_norm = 100.0
            drift_norm = np.linalg.norm(particle_drift)
            if drift_norm > max_drift_norm:
                particle_drift = particle_drift * (max_drift_norm / drift_norm)

            drift[i] = particle_drift
        return drift

    def _update_single_particle(
        self, 
        particle_idx: int, 
        particles: np.ndarray,
        y: np.ndarray,
        lambda_val: float, 
        d_lambda: float
    ) -> np.ndarray:
        """
        Update a single particle using LOCAL linearization.

        Args:
            particle_idx: Index of particle to update
            particles: All particles at current λ, shape (N, state_dim)
            y: Observation, shape (obs_dim,)
            lambda_val: Current λ value
            d_lambda: Step size in λ

        Returns:
            Updated particle, shape (state_dim,)
        """
        particle = particles[particle_idx]

        # Compute A_i and b_i using LOCAL linearization at this particle
        A_i, b_i = self._compute_local_flow_params(particle, lambda_val, y)

        # Evaluate dx/dλ = A_i @ x_i + b_i
        drift = A_i @ particle + b_i
        
        # Limit drift magnitude to prevent explosive growth
        max_drift_norm = 100.0  # Maximum drift magnitude per lambda step
        drift_norm = np.linalg.norm(drift)
        if drift_norm > max_drift_norm:
            drift = drift * (max_drift_norm / drift_norm)

        # Euler step
        particle_new = particle + drift * d_lambda
        
        # Clip particles to prevent divergence (conservative bounds for range-bearing)
        # Keep particles within reasonable distance from sensor
        max_dist = 1000.0  # Maximum distance from origin
        particle_norm = np.linalg.norm(particle_new)
        if particle_norm > max_dist:
            particle_new = particle_new * (max_dist / particle_norm)

        return particle_new

    def _flow_step_euler(
        self,
        particles: np.ndarray,
        y: np.ndarray,
        lambda_val: float,
        d_lambda: float
    ) -> np.ndarray:
        """
        Local flow step using Euler integration.

        Each particle uses its own LOCAL linearization H_i and GLOBAL P_{k|k-1}.
        Uses shared euler_step utility with per-particle drift computation.

        Args:
            particles: Current particles, shape (N, state_dim)
            y: Observation, shape (obs_dim,)
            lambda_val: Current λ
            d_lambda: Step size

        Returns:
            Updated particles, shape (N, state_dim)
        """
        # Use shared euler_step with per-particle drift function
        particles_new = euler_step(particles, self._compute_drift_local, d_lambda, lambda_val, y)

        # Apply clipping to prevent divergence
        max_dist = 1000.0
        norms = np.linalg.norm(particles_new, axis=1, keepdims=True)
        mask = norms > max_dist
        if np.any(mask):
            particles_new = np.where(mask, particles_new * (max_dist / norms), particles_new)

        return particles_new

    def _flow_step_rk4(
        self, 
        particles: np.ndarray,
        y: np.ndarray, 
        lambda_val: float, 
        d_lambda: float
    ) -> np.ndarray:
        """
        RK4 integration with local linearizations.

        Note: This requires 4 linearizations per particle per step (expensive).
        """
        def compute_drift_for_particle(particle, lam):
            """Compute drift for a single particle at given λ."""
            A_i, b_i = self._compute_local_flow_params(particle, lam, y)
            return A_i @ particle + b_i

        particles_new = np.zeros_like(particles)

        for i in range(self.n_particles):
            p = particles[i]
            
            # RK4 stages for this particle
            k1 = compute_drift_for_particle(p, lambda_val)
            k2 = compute_drift_for_particle(p + 0.5 * d_lambda * k1, lambda_val + 0.5 * d_lambda)
            k3 = compute_drift_for_particle(p + 0.5 * d_lambda * k2, lambda_val + 0.5 * d_lambda)
            k4 = compute_drift_for_particle(p + d_lambda * k3, lambda_val + d_lambda)
            
            particles_new[i] = p + (d_lambda / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return particles_new

    def predict(self):
        """
        Prediction step with mean-only feedback.

        1. Update global filter mean to ensemble mean (if feedback enabled)
        2. Run global EKF/UKF prediction to get P_{k|k-1}
        3. Propagate each particle through dynamics with noise
        
        Note: We do NOT blend empirical covariances back into the global filter.
        The global filter provides covariance guidance only - blending causes
        covariance explosion due to positive feedback.
        """
        # FEEDBACK MECHANISM: Update global filter mean to ensemble mean ONLY
        # Do NOT touch the covariance - let EKF/UKF maintain its own estimate
        if self.use_feedback:
            ensemble_mean = np.mean(self.particles, axis=0)
            self.global_filter.mean = ensemble_mean.copy()

        # Run GLOBAL EKF/UKF prediction to get P_{k|k-1}
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.copy()

        # Store η̄_0: the DETERMINISTIC predicted mean (from global filter, not noisy particles)
        # This matches Algorithm 1 and ensures correct b(λ) computation
        self.eta_bar_0 = self.global_filter.mean.copy()

        # Propagate particles through state transition (from base class)
        super().predict()

    def update(self, y: np.ndarray):
        """
        Update step: flow particles from λ=0 to λ=1 using LOCAL linearizations.

        Each particle gets its own H_i but uses GLOBAL P_{k|k-1}.

        Args:
            y: Observation, shape (obs_dim,)
        """
        # Reset caches for this update
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

        # Integrate flow with LOCAL linearizations
        # Following Algorithm 1 from paper exactly:
        # Line 12: λ = λ + ε_j (increment FIRST per paper)
        # Lines 14-15: Calculate A_j^i(λ), b_j^i(λ) at NEW λ for each particle
        for i in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[i]
            lambda_val += d_lambda  # Algorithm 1, Line 12: increment FIRST

            # Debug: Capture flow step diagnostics (sample steps)
            if self.debug_mode and i % 10 == 0:
                # FIX: Correct number of arguments (was passing 4, should be 3)
                # Sample first particle's A and b
                A, b = self._compute_local_flow_params(particles_flow[0], lambda_val, y)
                H = self.model.observation_jacobian(particles_flow[0])
                
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
                particles_flow = self._flow_step_euler(
                    particles_flow, y, lambda_val, d_lambda
                )
            elif self.integration_method == 'rk4':
                particles_flow = self._flow_step_rk4(
                    particles_flow, y, lambda_val, d_lambda
                )
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")

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
        # This updates the global filter's posterior mean and covariance
        # based on the EKF/UKF update equations (not empirical estimates)
        self.global_filter.update(y)

    def get_diagnostics(self) -> dict:
        """Return diagnostic information."""
        return {
            'final_particles': self.particles,
            'predicted_cov': self.predicted_cov,
            'global_filter_mean': self.global_filter.mean,
            'global_filter_cov': self.global_filter.cov,
            'use_feedback': self.use_feedback,
            'eta_bar_0': self.eta_bar_0
        }





# """Local Exact Daum-Huang (LEDH) particle flow filter"""

# import numpy as np
# from typing import Optional, Tuple
# from concurrent.futures import ThreadPoolExecutor
# from .edh_flow import ExactDaumHuangFlow
# from ..kalman.extended_kalman import ExtendedKalmanFilter
# from ..kalman.unscented_kalman import UnscentedKalmanFilter


# class LocalExactDaumHuangFlow(ExactDaumHuangFlow):
#     """
#     Local Exact Daum-Huang Flow following Algorithm 1 from Li & Coates (2017).

#     Key modifications from global EDH (Algorithm 2):
#     1. Linearize measurement function at EACH particle location (not just at mean)
#     2. Use GLOBAL predictive covariance P_{k|k-1} from a single EKF/UKF
#     3. Compute A_i and b_i matrices individually for each particle

#     Flow equation: dx/dλ = A_i @ x + b_i
#     where A_i uses local linearization H_i and global covariance P_{k|k-1}.

#     This prevents the uncertainty collapse issue that occurs when using per-particle
#     EKF updates, which would double-count measurement information.
#     """

#     def __init__(self, model, n_particles: int = 1000,
#                  n_lambda_steps: int = 100,
#                  integration_method: str = 'euler',
#                  filter_type: str = 'ekf',
#                  use_ekf_feedback: bool = True,
#                  n_threads: Optional[int] = None):
#         """
#         Initialize Local Exact Daum-Huang flow filter.

#         Args:
#             model: StateSpaceModel instance
#             n_particles: Number of particles
#             n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
#             integration_method: 'euler' or 'rk4'
#             filter_type: 'ekf' or 'ukf' for global covariance guidance
#             use_ekf_feedback: If True, feed DH mean back to EKF (Algorithm 2, Line 19)
#             n_threads: Number of threads for parallelization
#         """
#         super().__init__(model, n_particles, n_lambda_steps, integration_method, n_threads)
#         self.filter_type = filter_type
#         self.use_ekf_feedback = use_ekf_feedback

#         # Single GLOBAL EKF/UKF for covariance guidance (Algorithm 1 from Li & Coates 2017)
#         self.global_filter = None
#         self.predictive_cov = None  # P_{k|k-1} used for flow

#         # Cache for optimization
#         self.R_inv_cache = None

#     def initialize(self, random_state: Optional[np.random.Generator] = None):
#         """Initialize particles and global EKF/UKF for covariance guidance (Algorithm 1)."""
#         # Initialize particles from base class
#         super().initialize(random_state)

#         # Compute empirical mean and covariance from all particles
#         ensemble_mean = np.mean(self.particles, axis=0)

#         # np.cov returns (state_dim, state_dim) for multi-D, but scalar for 1D
#         # Ensure it's always 2D: (state_dim, state_dim)
#         if self.state_dim == 1:
#             initial_cov = np.var(self.particles).reshape(1, 1)
#         else:
#             initial_cov = np.cov(self.particles.T)

#         # Initialize SINGLE GLOBAL filter at ensemble mean
#         if self.filter_type == 'ekf':
#             self.global_filter = ExtendedKalmanFilter(self.model, mean_0=ensemble_mean, Sigma_0=initial_cov)
#         elif self.filter_type == 'ukf':
#             self.global_filter = UnscentedKalmanFilter(self.model, mean_0=ensemble_mean, Sigma_0=initial_cov)
#         else:
#             raise ValueError(f"Unknown filter type: {self.filter_type}")

#         self.global_filter.mean = ensemble_mean.copy()
#         self.global_filter.cov = initial_cov.copy()

#         # Store predictive covariance (will be updated in predict step)
#         self.predictive_cov = initial_cov.copy()

#     def _compute_A_and_b(self, particle: np.ndarray, particle_0: np.ndarray, H: np.ndarray,
#                          P: np.ndarray, lambda_val: float,
#                          innovation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Compute A and b matrices for exact flow equation.

#         Based on equations (10-11) from the paper:
#             A = -0.5 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
#             b = (I + 2λA)[(I + λA)*P*H^T*R^{-1}*(z - e) + A*x_0]
#             where e = h(x) - H*x
#             dx/dλ = A*x + b

#         Args:
#             particle: Current particle state, shape (state_dim,)
#             particle_0: Initial particle state at λ=0, shape (state_dim,)
#             H: Measurement Jacobian at particle, shape (obs_dim, state_dim)
#             P: Covariance matrix from EKF/UKF, shape (state_dim, state_dim)
#             lambda_val: Current λ value
#             innovation: y - h(x), shape (obs_dim,)

#         Returns:
#             A: Matrix A, shape (state_dim, state_dim)
#             b: Vector b, shape (state_dim,)
#         """
#         R = self.model.observation_noise_cov
#         I = np.eye(self.state_dim)

#         # Cache R_inv (computed once per update)
#         if self.R_inv_cache is None:
#             try:
#                 self.R_inv_cache = np.linalg.inv(R)
#             except np.linalg.LinAlgError:
#                 self.R_inv_cache = np.linalg.pinv(R)

#         # Compute S = λ*H*P*H^T + R
#         HPH = H @ P @ H.T  # (obs_dim, obs_dim)
#         S = lambda_val * HPH + R

#         # Compute A = -0.5 * P * H^T * S^{-1} * H
#         try:
#             S_inv = np.linalg.inv(S)
#         except np.linalg.LinAlgError:
#             S_inv = np.linalg.pinv(S)

#         A = -0.5 * P @ H.T @ S_inv @ H  # (state_dim, state_dim)

#         # CRITICAL FIX: Compute (z - e) where e = h(x) - H*x
#         # innovation = y - h(x), so (z - e) = innovation + H*x
#         Hx = H @ particle
#         z_minus_e = innovation + Hx

#         # Compute b = (I + 2λA)[(I + λA)*P*H^T*R^{-1}*(z - e) + A*x_0]
#         # CRITICAL: Use particle_0 (initial position at λ=0) not current particle
#         R_inv = self.R_inv_cache

#         I_plus_lambda_A = I + lambda_val * A
#         I_plus_2lambda_A = I + 2 * lambda_val * A

#         term1 = I_plus_lambda_A @ P @ H.T @ R_inv @ z_minus_e
#         term2 = A @ particle_0  # Use initial particle position!
#         b = I_plus_2lambda_A @ (term1 + term2)  # (state_dim,)

#         return A, b

#     def _update_single_particle(self, particle_idx: int, particles: np.ndarray,
#                                particles_0: np.ndarray, y: np.ndarray,
#                                lambda_val: float, d_lambda: float) -> np.ndarray:
#         """
#         Update a single particle according to Algorithm 1 (LOCAL).

#         Args:
#             particle_idx: Index of particle to update
#             particles: All particles at current λ, shape (N, state_dim)
#             particles_0: Initial particles at λ=0, shape (N, state_dim)
#             y: Observation, shape (obs_dim,)
#             lambda_val: Current λ value
#             d_lambda: Step size in λ

#         Returns:
#             Updated particle, shape (state_dim,)
#         """
#         particle = particles[particle_idx]
#         particle_0 = particles_0[particle_idx]

#         # Step 1: Use GLOBAL predictive covariance P_{k|k-1} for ALL particles
#         P = self.predictive_cov

#         # Step 2: Linearize at this particle's location (LOCAL linearization)
#         H_i = self._compute_jacobian(particle)

#         # Step 3: Compute innovation y - h(x_i)
#         h_i = self.model.observation_function(particle)
#         innovation = y - h_i

#         # Step 4: Compute A_i and b_i using GLOBAL P and LOCAL H_i
#         A_i, b_i = self._compute_A_and_b(particle, particle_0, H_i, P, lambda_val, innovation)

#         # Step 5: Evaluate dx/dλ = A_i * x_i + b_i
#         dx_dlambda = A_i @ particle + b_i

#         # Step 6: Migrate particle
#         particle_new = particle + d_lambda * dx_dlambda

#         return particle_new

#     def _flow_step_euler(self, particles: np.ndarray, particles_0: np.ndarray,
#                         y: np.ndarray, lambda_val: float, d_lambda: float) -> np.ndarray:
#         """
#         Local flow step using Euler integration - Algorithm 1 implementation.

#         Each particle uses its own linearization H_i and GLOBAL predictive covariance P_{k|k-1}
#         to compute individual A_i, b_i.

#         Args:
#             particles: Current particles, shape (N, state_dim)
#             particles_0: Initial particles at λ=0, shape (N, state_dim)
#             y: Observation, shape (obs_dim,)
#             lambda_val: Current λ
#             d_lambda: Step size

#         Returns:
#             Updated particles, shape (N, state_dim)
#         """
#         particles_new = particles.copy()

#         # Update each particle with its own local linearization and GLOBAL P_{k|k-1}
#         if self.n_threads > 1:
#             def update_particle(i):
#                 return self._update_single_particle(i, particles, particles_0, y, lambda_val, d_lambda)

#             with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
#                 particles_new = np.array(list(executor.map(update_particle, range(self.n_particles))))
#         else:
#             for i in range(self.n_particles):
#                 particles_new[i] = self._update_single_particle(i, particles, particles_0, y, lambda_val, d_lambda)

#         return particles_new

#     def _flow_step_rk4(self, particles: np.ndarray, particles_0: np.ndarray,
#                        y: np.ndarray, lambda_val: float, d_lambda: float) -> np.ndarray:
#         """
#         RK4 integration with local gains.

#         Note: This is expensive as it requires 4 linearizations per particle per step.
#         """
#         # For simplicity, fall back to Euler
#         # Full RK4 would require computing k1, k2, k3, k4 for each particle
#         return self._flow_step_euler(particles, particles_0, y, lambda_val, d_lambda)

#     def predict(self):
#         """
#         Prediction step - Algorithm 1 (LOCAL) with GLOBAL covariance guidance.

#         1. Update global filter mean to ensemble mean (or use EKF feedback)
#         2. Run global EKF/UKF prediction to get P_{k|k-1}
#         3. Propagate each particle through dynamics with noise
#         """
#         # Store previous particles
#         particles_prev = self.particles.copy()

#         # Update global filter mean to ensemble mean
#         ensemble_mean = np.mean(particles_prev, axis=0)
#         if self.use_ekf_feedback:
#             # Use previous DH mean as EKF mean (feedback)
#             self.global_filter.mean = ensemble_mean.copy()

#         # Run GLOBAL EKF/UKF prediction to get P_{k|k-1}
#         self.global_filter.predict()
#         self.predictive_cov = self.global_filter.cov.copy()

#         # Propagate each particle through dynamics with noise
#         for i in range(self.n_particles):
#             # Sample from p(x_k | x_{k-1}^i)
#             self.particles[i] = self.model.sample_state_transition(
#                 particles_prev[i], self.random_state
#             )

#     def update(self, y: np.ndarray):
#         """
#         Update step - Algorithm 1 (LOCAL) with GLOBAL covariance guidance.

#         1. Flow particles using LOCAL linearizations and GLOBAL predictive P_{k|k-1}
#         2. Optionally update global EKF (for next cycle's covariance)

#         Args:
#             y: Observation, shape (obs_dim,)
#         """
#         # Reset R_inv cache for this update
#         self.R_inv_cache = None

#         # Store initial particles at λ=0 (needed for b computation)
#         particles_0 = self.particles.copy()

#         # Discretize pseudo-time λ ∈ [0, 1]
#         lambdas = np.linspace(0, 1, self.n_lambda_steps + 1)
#         d_lambda = lambdas[1] - lambdas[0]

#         particles_flow = particles_0.copy()

#         # Integrate flow with LOCAL linearizations and GLOBAL P_{k|k-1}
#         for i, lambda_val in enumerate(lambdas[:-1]):
#             if self.integration_method == 'euler':
#                 particles_flow = self._flow_step_euler(
#                     particles_flow, particles_0, y, lambda_val, d_lambda
#                 )
#             elif self.integration_method == 'rk4':
#                 particles_flow = self._flow_step_rk4(
#                     particles_flow, particles_0, y, lambda_val, d_lambda
#                 )
#             else:
#                 raise ValueError(f"Unknown integration method: {self.integration_method}")

#         # Particles at λ=1 represent posterior
#         self.particles = particles_flow

#         # Update global EKF to get P_{k|k} for next prediction cycle
#         # This does NOT affect the current particles - they've already been moved by flow
#         ensemble_mean = np.mean(self.particles, axis=0)
#         self.global_filter.mean = ensemble_mean.copy()
#         self.global_filter.update(y)
