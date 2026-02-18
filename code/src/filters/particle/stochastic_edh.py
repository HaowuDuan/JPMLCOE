"""
Stochastic Exact Daum-Huang (EDH) particle flow filter.

Implements the stochastic particle flow from:
  Dai & Daum (2021), "A New Parameterized Family of Stochastic Particle Flow Filters"
  (arXiv:2103.09676)

With optional stiffness-mitigating schedule from:
  Dai & Daum (2021), "Stiffness Mitigation in Stochastic Particle Flow Filters"
  (arXiv:2107.04672)
"""
import numpy as np
import tensorflow as tf
from .edh_flow import ExactDaumHuangFlow
from ...utils.flow_params import compute_flow_params_global
from ...utils.linalg import safe_inv
from ...utils.ode_solvers import euler_step


class StochasticEDHFlow(ExactDaumHuangFlow):
    """
    Stochastic EDH particle flow filter (TensorFlow).

    Extends the deterministic Exact Flow with diagonal diffusion:
        dx = [A(λ)x + b(λ)]dλ + √Q dW_λ

    where Q = diag(q_1, ..., q_d) is a diagonal diffusion matrix.
    Pass a scalar for isotropic Q = q·I, or a list [q_1,...,q_d] for
    per-dimension diffusion (e.g. [4.0, 0.4] for the paper's Section 4).

    The drift includes a score correction (Corollary 2.1, Eq. 8):
        f_stoch = f_det + (1/2)·Q·∇log p(x,λ)
    so A(λ) = A_det(λ) - (Q/2)·Σ(λ)⁻¹ and b(λ) = b_det(λ) + (Q/2)·Σ(λ)⁻¹·m(λ).
    The posterior is identical for all Q (Theorem 3.1, exact in continuous limit).

    Uses global (fixed) linearization at η̄_0 — no relinearization
    during the flow, consistent with the SDE derivation.

    Optional stiffness mitigation (schedule_mu > 0): solves for an
    optimal schedule β(λ) that minimizes the condition number of
    M(β) = P₀⁻¹ + β·H^TR⁻¹H via a BVP shooting method
    (arXiv:2107.04672, Section 3, Theorem 3.1).
    """

    def __init__(self, model, diffusion_scale=0.0,
                 schedule_mu: float = 0.0,
                 lambda_schedule: str = 'exponential',
                 **kwargs):
        """
        Args:
            model: State-space model.
            diffusion_scale: Diffusion intensity. Scalar q → isotropic Q = q·I.
                List/array [q_1, ..., q_d] → diagonal Q = diag(q_1, ..., q_d).
                0.0 or all-zeros = deterministic flow (equivalent to parent).
            schedule_mu: Weight μ for stiffness mitigation optimal control.
                0.0 = linear schedule β=λ (Remark 3.1).
                >0  = solve BVP for optimal β(λ) (Theorem 3.1).
            lambda_schedule: Step-size schedule for λ ∈ [0,1].
                'exponential': geometric growth (q=1.2), small steps near λ=0.
                'uniform':     equal steps dλ = 1/n (matches paper Section 4).
            **kwargs: Passed to ExactDaumHuangFlow.
        """
        self._lambda_schedule_override = lambda_schedule
        super().__init__(model, **kwargs)
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.np_dtype = np.float64 if self.dtype == tf.float64 else np.float32
        self.schedule_mu = schedule_mu
        self.seed_counter = 0

        # Normalize diffusion_scale to (state_dim,) numpy array
        q_raw = np.asarray(diffusion_scale, dtype=np.float64).ravel()
        if q_raw.size == 1:
            q_raw = np.full(self.state_dim, float(q_raw[0]))
        self.Q_diag = q_raw                           # shape (state_dim,)
        self._has_diffusion = bool(np.any(self.Q_diag > 0))

    def _generate_lambda_steps(self):
        """
        Override parent's exponential schedule with 'uniform' option.

        'uniform': equal steps dλ = 1/n, replicates paper Section 4
                   (exposes stiffness in the linear-schedule case).
        'exponential': inherited behaviour (small steps near λ=0).
        """
        if self._lambda_schedule_override == 'uniform':
            steps_np = np.ones(self.n_lambda_steps) / self.n_lambda_steps
            self.lambda_steps = tf.constant(steps_np, dtype=self.dtype)
        else:
            super()._generate_lambda_steps()

    # ------------------------------------------------------------------
    # Stiffness mitigation solver (arXiv:2107.04672, Section 3)
    # All BVP computation is in numpy — no gradients needed, avoids
    # TF eager kernel-launch overhead for thousands of 2×2 operations.
    # ------------------------------------------------------------------

    def _compute_optimal_schedule(self, P: tf.Tensor,
                                  R_inv: tf.Tensor
                                  ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Solve BVP β̈ = μ·∂κ/∂β, β(0)=0, β(1)=1 (Eqs. 26-27).

        Uses bisection shooting to find β̇(0) = u₀, then integrates
        to record β at each λ step. Entirely in numpy for speed.

        Returns:
            beta_values:  TF tensor, shape (n_lambda_steps,), β at each step
            dbeta_values: TF tensor, shape (n_lambda_steps,), dβ per step
        """
        # Convert to numpy once — no gradients needed for schedule solve
        H_np = self.model.observation_jacobian(self.eta_bar_0).numpy()
        J_prior = np.linalg.inv(P.numpy())   # P is PD, inv is safe
        J_meas  = H_np.T @ R_inv.numpy() @ H_np
        mu      = float(self.schedule_mu)
        lambda_steps_np = self.lambda_steps.numpy()  # (n_lambda_steps,)

        def _drift(state):
            """Numpy drift [β̇, μ·∂κ/∂β] for ODE state [β, β̇]."""
            beta, beta_dot = state[0], state[1]
            M     = J_prior + beta * J_meas
            M_inv = np.linalg.inv(M)
            dkappa = (np.trace(J_meas) * np.trace(M_inv)
                      - np.trace(M) * np.trace(M_inv @ J_meas @ M_inv))
            return np.array([beta_dot, mu * dkappa])

        def _shoot(u0, n_steps=500):
            """Euler integrate ODE from λ=0 to λ=1, return β(1)."""
            state = np.array([0.0, u0])
            dlam  = 1.0 / n_steps
            for _ in range(n_steps):
                state = state + dlam * _drift(state)
            return float(state[0])

        # --- Bisection to find u₀ ---
        u_lo, u_hi = 0.1, 20.0
        if _shoot(u_lo) > 1.0:
            u_lo = 0.01
        if _shoot(u_hi) < 1.0:
            u_hi = 50.0

        err_lo = _shoot(u_lo) - 1.0
        err_hi = _shoot(u_hi) - 1.0
        if err_lo * err_hi > 0:
            # Bracket invalid — fall back to linear schedule β = λ
            return tf.cumsum(self.lambda_steps), self.lambda_steps

        for _ in range(40):
            u_mid = (u_lo + u_hi) / 2.0
            if _shoot(u_mid) < 1.0:
                u_lo = u_mid
            else:
                u_hi = u_mid

        optimal_u0 = (u_lo + u_hi) / 2.0

        # --- Record β at each λ step point ---
        state = np.array([0.0, optimal_u0])
        beta_np = np.empty(self.n_lambda_steps, dtype=np.float64)
        for i in range(self.n_lambda_steps):
            state = state + lambda_steps_np[i] * _drift(state)
            beta_np[i] = state[0]

        dbeta_np = np.empty_like(beta_np)
        dbeta_np[0]  = beta_np[0]
        dbeta_np[1:] = np.diff(beta_np)

        return (tf.constant(beta_np,  dtype=self.dtype),
                tf.constant(dbeta_np, dtype=self.dtype))

    # ------------------------------------------------------------------
    # Main filter update
    # ------------------------------------------------------------------

    def update(self, y: tf.Tensor):
        """
        Stochastic flow from λ=0 to λ=1.

        Same as ExactDaumHuangFlow.update() except:
        1. H is fixed at η̄_0 (no re-linearization)
        2. Drift includes score correction: f = f_det + (q/2)∇log p
           (Theorem 2.1, Corollary 2.1, arXiv:2103.09676)
        3. Euler-Maruyama integration (adds √q·dW noise)
        4. Optional optimal schedule β(λ) for stiffness mitigation
        """
        # --- Setup (all TF tensors, no numpy conversions) ---
        observation = y  # Already TF tensor from flow_base.filter()
        P = self.predicted_cov  # Already TF tensor from predict()
        R = tf.cast(self.model.observation_noise_cov, self.dtype)
        eta_bar_0 = self.eta_bar_0  # Already TF tensor from predict()
        R_inv = safe_inv(R)
        particles_flow = self.particles.value()

        # Compute H once at η̄_0 (global linearization)
        H_fixed = self.model.observation_jacobian(eta_bar_0)

        # --- Precompute score correction quantities (constant over λ) ---
        # Score correction (Corollary 2.1, Eq. 8, generalised to diagonal Q):
        #   A = A_det - (Q/2)·Σ(λ)⁻¹   (row-scale precision by Q_diag/2)
        #   b = b_det + (Q/2)·Σ(λ)⁻¹·m(λ)  (element-wise)
        # where Σ(λ)⁻¹ = P⁻¹ + λ·Hᵀ R⁻¹ H  (precision)
        # and   Σ(λ)⁻¹·m(λ) = P⁻¹·η̄₀ + λ·Hᵀ R⁻¹·y  (info vector)
        if self._has_diffusion:
            Q_tf = tf.constant(self.Q_diag, dtype=self.dtype)  # (state_dim,)
            P_inv = safe_inv(P)
            H_T_R_inv_H = tf.transpose(H_fixed) @ R_inv @ H_fixed
            P_inv_eta = tf.linalg.matvec(P_inv, eta_bar_0)
            H_T_R_inv_y = tf.linalg.matvec(
                tf.transpose(H_fixed) @ R_inv, observation)

        # --- Compute optimal schedule if μ > 0 ---
        if self.schedule_mu > 0:
            beta_vals, dbeta_vals = self._compute_optimal_schedule(P, R_inv)

        # --- Flow loop (all TF tensor operations) ---
        lambda_val = tf.constant(0.0, dtype=self.dtype)
        for i in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[i]
            lambda_val = lambda_val + d_lambda

            # Schedule: β(λ) and effective step size dβ
            if self.schedule_mu > 0:
                homotopy_param = beta_vals[i]  # TF scalar from _compute_optimal_schedule
                step_size = dbeta_vals[i]
            else:
                # Linear schedule: β = λ, dβ = dλ
                homotopy_param = lambda_val
                step_size = d_lambda

            # Compute A_det(β), b_det(β) using global flow formulas
            # H is fixed at η̄_0, b uses z directly (no e correction)
            A, b = compute_flow_params_global(
                H_fixed, homotopy_param,
                observation, P, R, R_inv, eta_bar_0, self.state_dim
            )

            # Score correction: compensate drift for diffusion
            # (Theorem 2.1, Eq. 6; Corollary 2.1, Eq. 8, diagonal Q)
            if self._has_diffusion:
                correction_A = P_inv + homotopy_param * H_T_R_inv_H  # (d, d)
                correction_b = P_inv_eta + homotopy_param * H_T_R_inv_y  # (d,)
                # (Q/2)·Σ⁻¹: row-scale correction_A by Q_diag/2
                A = A - tf.expand_dims(Q_tf, 1) / 2 * correction_A
                b = b + (Q_tf / 2) * correction_b

            # Euler step with effective step size dβ
            particles_flow = euler_step(
                particles_flow, self._compute_drift,
                step_size, A, b
            )

            # SDE noise: √(Q_diag · dλ) · dW — uses dλ (not dβ), since
            # Brownian motion is parameterized by λ; Q_tf broadcasts over N
            if self._has_diffusion:
                seed = tf.constant([self.seed_counter, i], dtype=tf.int32)
                noise = tf.random.stateless_normal(
                    tf.shape(particles_flow), seed=seed,
                    dtype=particles_flow.dtype
                )
                particles_flow = particles_flow + noise * tf.sqrt(Q_tf * d_lambda)

        self.seed_counter += 1

        # --- Finalize ---
        self.particles.assign(particles_flow)
        self.global_filter.update(y.numpy() if isinstance(y, tf.Tensor) else y)
