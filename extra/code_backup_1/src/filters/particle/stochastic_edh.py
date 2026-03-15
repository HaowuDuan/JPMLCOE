"""
Stochastic Exact Daum-Huang (EDH) particle flow filter.

Implements the General Linear Log-Homotopy flow from:
Dai, L., & Daum, F. E. (2021). Stiffness Mitigation in Stochastic Particle Flow Filters.
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
from scipy.linalg import sqrtm
from .flow_base import FlowFilterBase
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ..kalman.unscented_kalman import UnscentedKalmanFilter
from ...utils.ode_solvers import euler_maruyama_step
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline



class StochasticEDHFlow(FlowFilterBase):
    """
    Stochastic Particle Flow Filter using General Homotopy.

    Flow Equation:
        dx = f(x, λ)dλ + √Q dw
    
    Drift f(x, λ) is derived from Theorem 2.1:
        f(x) = K₁(λ) ∇log p₀(x) + K₂(λ) ∇log h(x)

    Where:
        K₁ = 0.5*Q - 0.5*β̇*P*Hᵀ*R⁻¹*H*P
        K₂ = β̇*P

    This class supports:
    1. Phase 1: Linear Schedule (β(λ)=λ, β̇=1).
    2. Phase 2: Optimal Stiffness-Mitigating Schedule (β(λ) from BVP).
    """

    def __init__(
        self,
        model,
        n_particles: int = 1000,
        n_lambda_steps: int = 100,
        diffusion_scale: float = 0.0,
        schedule_func: Optional[Callable[[float], Tuple[float, float]]] = None,
        filter_type: str = 'ekf',
        use_ekf_covariance: bool = True,
        use_optimal_schedule: bool = False,
        n_threads: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            model: StateSpaceModel.
            n_particles: Number of particles.
            n_lambda_steps: Integration steps.
            diffusion_scale: Scalar q for Q = q*I.
            schedule_func: Function f(λ) -> (β, β_dot). 
                           If None, defaults to linear: (λ, 1.0).
            filter_type: 'ekf' or 'ukf' for global covariance guidance.
            use_ekf_covariance: If True, use EKF/UKF predicted covariance instead of empirical.
            use_optimal_schedule: If True, compute optimal stiffness-mitigating schedule.
        """
        super().__init__(model, n_particles, n_lambda_steps, integration_method='euler', n_threads=n_threads)
        self.diffusion_scale = diffusion_scale
        self.schedule_func = schedule_func
        self.filter_type = filter_type
        self.use_ekf_covariance = use_ekf_covariance
        self.use_optimal_schedule = use_optimal_schedule
        self.Q = None
        self.global_filter = None
        self.predicted_cov = None
        self.eta_bar_0 = None 

    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles and global EKF/UKF for covariance guidance."""
        super().initialize(random_state)
        
        # Compute empirical mean and covariance from particles
        ensemble_mean = np.mean(self.particles, axis=0)
        
        if self.state_dim == 1:
            initial_cov = np.var(self.particles).reshape(1, 1)
        else:
            initial_cov = np.cov(self.particles.T)
        
        # Initialize global filter for covariance guidance
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

    def predict(self):
        """
        Prediction step with EKF/UKF covariance guidance.
        
        Updates global filter from particle ensemble, runs global filter prediction,
        then propagates particles through state transition.
        """
        # Feedback: Update global filter mean from ensemble
        ensemble_mean = np.mean(self.particles, axis=0)
        self.global_filter.mean = ensemble_mean.copy()
        
        # Optionally update covariance to match ensemble (with blending)
        if self.state_dim == 1:
            ensemble_cov = np.var(self.particles).reshape(1, 1)
        else:
            ensemble_cov = np.cov(self.particles.T)
        
        alpha = 0.5  # Blend factor
        self.global_filter.cov = alpha * ensemble_cov + (1 - alpha) * self.global_filter.cov
        
        # Run global filter prediction to get P_{k|k-1}
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.copy()
        
        # Propagate particles through state transition (from base class)
        super().predict()

    def _get_schedule_values(self, lambda_val: float) -> Tuple[float, float]:
        """Returns β(λ) and β̇(λ)."""
        if self.schedule_func is None:
            # Linear schedule: beta = lambda, beta_dot = 1
            return lambda_val, 1.0
        return self.schedule_func(lambda_val)

    def _compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian H = ∂h/∂x using model's analytical Jacobian."""
        return self.model.observation_jacobian(x)

    def _compute_covariance(self, particles: np.ndarray) -> np.ndarray:
        """Compute empirical covariance P from given particles."""
        mean = np.mean(particles, axis=0)
        diff = particles - mean
        P = (diff.T @ diff) / (self.n_particles - 1)
        # Regularization for numerical stability
        return P + np.eye(self.state_dim) * 1e-9

    def _compute_drift_coefficients(
        self,
        eta_bar: np.ndarray,
        P: np.ndarray,
        observation: np.ndarray,
        lambda_val: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute drift A(λ) and b(λ) using FULL Hessians from Theorem 2.1.
        
        Paper formulas (for linear schedule β=λ):
        K₁ = 0.5Q + 0.5(∇²log p)⁻¹(∇²log h)(∇²log p)⁻¹
        K₂ = -(∇²log p)⁻¹
        
        Full ∇²log h includes observation Hessian term when available.
        """
        # 1. Schedule Parameters
        beta, beta_dot = self._get_schedule_values(lambda_val)
        
        # 2. Matrices & Inverses
        H = self._compute_jacobian(eta_bar)
        R = self.model.observation_noise_cov
        Q = np.eye(self.state_dim) * self.diffusion_scale
        
        try:
            P_inv = np.linalg.inv(P)
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            P_inv = np.linalg.pinv(P)
            R_inv = np.linalg.pinv(R)

        # 3. Compute FULL Hessian of log h
        # ∇²log h = -HᵀR⁻¹H + Σᵢ [R⁻¹(y - h(x))]ᵢ · (∂²hᵢ/∂x²)
        h_bar = self.model.observation_function(eta_bar)
        innovation = observation - h_bar
        weighted_innovation = R_inv @ innovation  # R⁻¹(y - h(x))
        
        # First term: -HᵀR⁻¹H (standard linearization)
        hess_log_h = -H.T @ R_inv @ H
        
        # Second term: observation Hessian contribution (if available)
        if hasattr(self.model, 'observation_hessian'):
            # obs_hessian shape: (obs_dim, state_dim, state_dim)
            obs_hessian = self.model.observation_hessian(eta_bar)
            for i in range(self.obs_dim):
                hess_log_h += weighted_innovation[i] * obs_hessian[i]
        
        # 4. Hessian of log p (Gaussian approximation - no analytical prior)
        hess_log_p = -P_inv
        
        # 5. Compute K1 and K2 using paper formulas
        # (∇²log p)⁻¹ = (-P⁻¹)⁻¹ = -P
        inv_hess_log_p = -P
        
        # K₁ = 0.5Q + 0.5(∇²log p)⁻¹(∇²log h)(∇²log p)⁻¹
        K1 = 0.5 * Q + 0.5 * (inv_hess_log_p @ hess_log_h @ inv_hess_log_p)
        
        # K₂ = -(∇²log p)⁻¹ = P
        K2 = -inv_hess_log_p  # = P
        
        # 6. Linearize Drift f(x) = K₁∇log p + K₂∇log h
        # A = K₁(∇²log p) + K₂(∇²log h)
        A = K1 @ hess_log_p + K2 @ hess_log_h

        # b = f(x̄) - Ax̄
        # ∇log p(x̄) = -P⁻¹(x̄ - μ₀) where μ₀ is predicted mean (Gaussian prior)
        grad_log_p = -P_inv @ (eta_bar - self.eta_bar_0)
        # ∇log h(x̄) = H^T R^{-1}(y - h(x̄))
        grad_log_h = H.T @ R_inv @ innovation

        drift_at_mean = K1 @ grad_log_p + K2 @ grad_log_h
        b = drift_at_mean - A @ eta_bar

        return A, b

    def _compute_drift(self, particles: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute drift for stochastic EDH flow: dη/dλ = Aη + b (vectorized)."""
        return particles @ A.T + b

    def update(self, y: np.ndarray):
        """
        Execute the particle flow from λ=0 to λ=1.

        Uses EKF/UKF predicted covariance for flow guidance when use_ekf_covariance=True.
        Computes optimal stiffness-mitigating schedule when use_optimal_schedule=True.
        """
        lambdas = np.linspace(0, 1, self.n_lambda_steps + 1)
        d_lambda = 1.0 / self.n_lambda_steps

        particles_flow = self.particles.copy()

        # Store predicted mean for Gaussian prior reference
        self.eta_bar_0 = np.mean(particles_flow, axis=0)

        # Initialize Diffusion for integration
        self.Q = np.eye(self.state_dim) * self.diffusion_scale
        
        # Compute optimal schedule if requested
        optimal_schedule_func = None
        if self.use_optimal_schedule:
            eta_bar_init = np.mean(particles_flow, axis=0)
            if self.use_ekf_covariance and self.predicted_cov is not None:
                P_init = self.predicted_cov
            else:
                P_init = self._compute_covariance(particles_flow)
            
            H_init = self._compute_jacobian(eta_bar_init)
            R = self.model.observation_noise_cov
            
            # Use StiffnessMitigationSolver to compute optimal schedule
            solver = StiffnessMitigationSolver(P_init, H_init, R)
            optimal_schedule_func = solver.solve(mu=1.0)  # Default stiffness weight

        for k in range(self.n_lambda_steps):
            lambda_curr = lambdas[k]
            
            # 1. Update Particle Statistics from FLOWING particles
            eta_bar = np.mean(particles_flow, axis=0)
            
            # Use EKF covariance (stable over time) or empirical covariance
            if self.use_ekf_covariance and self.predicted_cov is not None:
                P = self.predicted_cov
            else:
                P = self._compute_covariance(particles_flow)
            
            # 2. Compute Drift A(λ), b(λ) with appropriate schedule
            if optimal_schedule_func is not None:
                # Temporarily set schedule for this update
                old_schedule = self.schedule_func
                self.schedule_func = optimal_schedule_func
                A, b = self._compute_drift_coefficients(eta_bar, P, y, lambda_curr)
                self.schedule_func = old_schedule
            else:
                A, b = self._compute_drift_coefficients(eta_bar, P, y, lambda_curr)
            
            # 3. Euler-Maruyama Integration using shared utility
            # dx = (Ax + b)dλ + √Q dw
            particles_flow = euler_maruyama_step(
                particles_flow, self._compute_drift, d_lambda, A, b,
                diffusion_coeff=self.diffusion_scale
            )

        self.particles = particles_flow
        
        # Feedback: Update global filter mean from ensemble and run update
        ensemble_mean = np.mean(self.particles, axis=0)
        self.global_filter.mean = ensemble_mean.copy()
        self.global_filter.update(y)


"""
Optimal Schedule Solver for Stiffness Mitigation.

Implements Section 3 of:
Dai, L., & Daum, F. E. (2021). Stiffness Mitigation in Stochastic Particle Flow Filters.
arXiv:2107.04672
"""


class StiffnessMitigationSolver:
    """
    Solves for the optimal stiffness-mitigating schedule β(λ) for Particle Flow Filters.
    
    The problem minimizes the cost functional:
        J = ∫[0,1] [0.5 * u² + μ * κ(M)] dλ
    
    Leading to the Euler-Lagrange equation:
        d²β/dλ² = μ * ∂κ(M)/∂β
        
    Where M is defined (with α + β = 1 normalization):
        M(β) = P₀⁻¹ + β * (H^T R⁻¹ H)
    """

    def __init__(self, P_prior: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Initialize with system matrices.
        
        Args:
            P_prior: Prior covariance P₀.
            H: Observation Jacobian (linearized at prior mean).
            R: Observation noise covariance.
        """
        self.dim = P_prior.shape[0]
        
        # Precompute Information Matrices
        # J_prior = -∇²log p₀ = P₀⁻¹
        try:
            self.J_prior = np.linalg.inv(P_prior)
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular matrices
            self.J_prior = np.linalg.pinv(P_prior)
            R_inv = np.linalg.pinv(R)
            
        # J_meas = -∇²log h = H^T R⁻¹ H
        self.J_meas = H.T @ R_inv @ H
        
        # Cache for derivative computations to speed up ODE solver
        self._cache = {}

    def _compute_M(self, beta: float) -> np.ndarray:
        """
        Compute M(β) = J_prior + β * J_meas.
        """
        return self.J_prior + beta * self.J_meas

    def _condition_number_derivative_nuclear(self, beta: float) -> float:
        """
        Computes ∂κ/∂β for the Nuclear Norm (Trace Norm) condition number.
        
        Math Derivation:
            κ(M) = tr(M) * tr(M⁻¹)
            ∂κ/∂β = tr(M')tr(M⁻¹) + tr(M)tr((M⁻¹)')
            
        Using identity ∂(M⁻¹)/∂β = -M⁻¹ M' M⁻¹:
            ∂κ/∂β = tr(M')tr(M⁻¹) - tr(M)tr(M⁻¹ M' M⁻¹)
        
        NOTE: This derivation yields a MINUS sign between terms. Equation (28) in the 
        paper implies a PLUS sign, which appears to be a typo resulting from dropping 
        the negative sign in the inverse derivative. We use the correct calculus here.
        
        Args:
            beta: Homotopy parameter β ∈ [0,1]
            
        Returns:
            Derivative ∂κ/∂β
        """
        # Clip beta to ensure stability during intermediate solver steps
        beta = np.clip(beta, 0.0, 1.0)
        
        if beta in self._cache:
            return self._cache[beta]

        M = self._compute_M(beta)
        
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # ∂M/∂β = J_meas
        dM_dbeta = self.J_meas
        
        # Compute scalar traces
        tr_M = np.trace(M)
        tr_M_inv = np.trace(M_inv)
        tr_dM = np.trace(dM_dbeta)
        
        # tr(M⁻¹ * dM * M⁻¹) = tr(M⁻² * dM)
        term_quad = np.trace(M_inv @ dM_dbeta @ M_inv)
        
        # Correct calculus: tr(M')tr(M⁻¹) - tr(M)tr(M⁻² M')
        dkappa_dbeta = tr_dM * tr_M_inv - tr_M * term_quad
        
        self._cache[beta] = dkappa_dbeta
        return dkappa_dbeta

    def _ode_dynamics(self, lam, y, mu):
        """
        System dynamics for the BVP.
        State: y = [β, β']
        Returns: [β', β''] = [β', μ * ∂κ/∂β]
        """
        beta, beta_dot = y
        beta_double_dot = mu * self._condition_number_derivative_nuclear(beta)
        return [beta_dot, beta_double_dot]

    def solve(self, mu: float = 0.2, n_steps: int = 100) -> Callable[[float], Tuple[float, float]]:
        """
        Solve the Two-Point Boundary Value Problem (TPBVP) using the Shooting Method.
        
        Boundary conditions:
            β(0) = 0
            β(1) = 1
            
        Args:
            mu: Weight parameter μ ≥ 0.
            n_steps: Number of points for the output spline.
            
        Returns:
            schedule_func: A function that takes λ and returns (β, dβ/dλ)
        """
        # Remark 3.1: If μ ≈ 0, optimal solution is straight line β = λ
        if mu <= 1e-8:
            return lambda lam: (lam, 1.0)

        # Reset cache for new solve
        self._cache = {}

        def boundary_error(u0: float) -> float:
            """
            Shooting function: Returns error (β(1) - 1) for initial velocity u₀.
            """
            y0 = [0.0, u0]  # β(0)=0, β'(0)=u0
            
            sol = solve_ivp(
                fun=lambda t, y: self._ode_dynamics(t, y, mu),
                t_span=(0, 1),
                y0=y0,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            # Return error at final time
            return sol.y[0, -1] - 1.0

        # Find optimal initial slope u₀
        # We search in a reasonable bracket. If μ is large, u₀ might need to be higher.
        try:
            root_res = root_scalar(
                boundary_error,
                bracket=[0.1, 20.0],
                method='brentq',
                xtol=1e-5
            )
            optimal_u0 = root_res.root
        except ValueError:
            # Fallback if bracket is invalid (e.g. solution is outside [0.1, 20])
            root_res = root_scalar(
                boundary_error,
                x0=1.0,
                x1=1.5,
                method='secant',
                xtol=1e-5
            )
            optimal_u0 = root_res.root

        # Generate final trajectory with optimal u₀
        sol = solve_ivp(
            fun=lambda t, y: self._ode_dynamics(t, y, mu),
            t_span=(0, 1),
            y0=[0.0, optimal_u0],
            t_eval=np.linspace(0, 1, n_steps),
            method='RK45'
        )
        
        # Create spline for smooth interpolation
        beta_spline = CubicSpline(sol.t, sol.y[0])
        beta_dot_spline = beta_spline.derivative()
        
        def optimal_schedule(lam: float) -> Tuple[float, float]:
            """
            Returns (β, dβ/dλ) at given λ.
            """
            lam_cl = np.clip(lam, 0, 1)
            return float(beta_spline(lam_cl)), float(beta_dot_spline(lam_cl))
            
        return optimal_schedule
