"""
Stochastic Exact Daum-Huang (EDH) particle flow filter.

Implements the stochastic particle flow from:
  Dai & Daum (2021), "A New Parameterized Family of Stochastic Particle Flow Filters"
  (arXiv:2103.09676)

With optional stiffness-mitigating schedule from:
  Dai & Daum (2021), "Stiffness Mitigation in Stochastic Particle Flow Filters"
  (arXiv:2107.04672)
"""
import tensorflow as tf
from .edh_flow import ExactDaumHuangFlow
from ...utils.flow_params import compute_flow_params
from ...utils.ode_solvers import euler_step


class StochasticEDHFlow(ExactDaumHuangFlow):
    """
    Stochastic EDH particle flow filter (TensorFlow).

    Extends the deterministic Exact Flow with isotropic diffusion:
        dx = [A(λ)x + b(λ)]dλ + √(q·I) dW_λ

    The drift includes a score correction (Corollary 2.1, Eq. 8):
        f_stoch = f_det + (q/2)·∇log p(x,λ)
    so A(λ) = A_det(λ) - (q/2)·Σ(λ)⁻¹ and b(λ) = b_det(λ) + (q/2)·Σ(λ)⁻¹·m(λ).
    The posterior is identical for all Q (Theorem 3.1).

    Uses re-linearization at the flowing mean (same as edh_flow)
    for stability with nonlinear observation models.

    Optional stiffness mitigation (schedule_mu > 0): solves for an
    optimal schedule β(λ) that minimizes the condition number of
    M(β) = P₀⁻¹ + β·H^TR⁻¹H via a BVP shooting method
    (arXiv:2107.04672, Section 3, Theorem 3.1).
    """

    def __init__(self, model, diffusion_scale: float = 0.001,
                 schedule_mu: float = 0.0, **kwargs):
        """
        Args:
            model: State-space model.
            diffusion_scale: Scalar q for isotropic diffusion Q = q·I.
                0.0 = deterministic flow (equivalent to parent).
            schedule_mu: Weight μ for stiffness mitigation optimal control.
                0.0 = linear schedule β=λ (Remark 3.1).
                >0  = solve BVP for optimal β(λ) (Theorem 3.1).
            **kwargs: Passed to ExactDaumHuangFlow.
        """
        super().__init__(model, **kwargs)
        self.diffusion_scale = diffusion_scale
        self.schedule_mu = schedule_mu
        self.seed_counter = 0

    # ------------------------------------------------------------------
    # Stiffness mitigation solver (arXiv:2107.04672, Section 3)
    # ------------------------------------------------------------------

    def _schedule_drift(self, state: tf.Tensor,
                        J_prior: tf.Tensor,
                        J_meas: tf.Tensor) -> tf.Tensor:
        """
        Drift for the schedule ODE system [β, β̇].

        ODE (Eq. 26): β̈ = μ · ∂κ/∂β
        State: [β, β̇]  →  drift: [β̇, μ · ∂κ/∂β]

        ∂κ/∂β for nuclear norm κ(M) = tr(M)·tr(M⁻¹), M = J_prior + β·J_meas.
        Correct calculus via Lemma A.1 (Remark 3.2):
            ∂κ/∂β = tr(M')·tr(M⁻¹) - tr(M)·tr(M⁻¹ M' M⁻¹)
        Note: Eq. 28 in the paper has a sign typo (+ instead of -).
        """
        beta = state[0]
        beta_dot = state[1]

        M = J_prior + beta * J_meas
        M_inv = tf.linalg.inv(M)
        dkappa = (tf.linalg.trace(J_meas) * tf.linalg.trace(M_inv)
                  - tf.linalg.trace(M) * tf.linalg.trace(
                      M_inv @ J_meas @ M_inv))

        return tf.stack([beta_dot, self.schedule_mu * dkappa])

    def _shoot(self, u0: float, J_prior: tf.Tensor,
               J_meas: tf.Tensor, n_steps: int = 500) -> float:
        """Integrate schedule ODE from λ=0 to λ=1, return β(1)."""
        state = tf.constant([0.0, u0], dtype=tf.float32)
        dlam = tf.constant(1.0 / n_steps, dtype=tf.float32)
        for _ in range(n_steps):
            state = euler_step(state, self._schedule_drift,
                               dlam, J_prior, J_meas)
        return float(state[0])

    def _compute_optimal_schedule(self, P: tf.Tensor,
                                  R_inv: tf.Tensor
                                  ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Solve BVP β̈ = μ·∂κ/∂β, β(0)=0, β(1)=1 (Eqs. 26-27).

        Uses bisection shooting (Section 4) to find β̇(0) = u₀,
        then integrates to record β at each cumulative λ step point.

        With the optimal schedule, the flow uses β as the homotopy
        parameter and dβ as the Euler step size. This is equivalent
        to the reparameterization: f_opt(λ) = β̇(λ)·f_exact(β(λ)),
        so dx = f_exact(β)·dβ + √q·dW_λ.

        Returns:
            beta_values:  TF tensor, shape (n_lambda_steps,), β at each cumulative λ
            dbeta_values: TF tensor, shape (n_lambda_steps,), dβ per step
        """
        H_tf = self.model.observation_jacobian(self.eta_bar_0)
        J_prior = tf.linalg.inv(P)
        J_meas = tf.transpose(H_tf) @ R_inv @ H_tf

        # --- Bisection to find u₀ (Section 4) ---
        u_lo, u_hi = 0.1, 20.0

        # Widen bracket if needed
        if self._shoot(u_lo, J_prior, J_meas) > 1.0:
            u_lo = 0.01
        if self._shoot(u_hi, J_prior, J_meas) < 1.0:
            u_hi = 50.0

        # Verify bracket contains root
        err_lo = self._shoot(u_lo, J_prior, J_meas) - 1.0
        err_hi = self._shoot(u_hi, J_prior, J_meas) - 1.0
        if err_lo * err_hi > 0:
            # Bracket invalid — fall back to linear schedule β = λ
            return tf.cumsum(self.lambda_steps), self.lambda_steps

        for _ in range(40):
            u_mid = (u_lo + u_hi) / 2.0
            if self._shoot(u_mid, J_prior, J_meas) < 1.0:
                u_lo = u_mid
            else:
                u_hi = u_mid

        optimal_u0 = (u_lo + u_hi) / 2.0

        # --- Record β at each cumulative λ step point ---
        state = tf.constant([0.0, optimal_u0], dtype=tf.float32)
        beta_values = []

        for i in range(self.n_lambda_steps):
            state = euler_step(state, self._schedule_drift,
                               self.lambda_steps[i], J_prior, J_meas)
            beta_values.append(state[0])

        beta_tf = tf.stack(beta_values)
        dbeta_tf = beta_tf - tf.concat([[0.0], beta_tf[:-1]], axis=0)
        return beta_tf, dbeta_tf

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
        R = tf.constant(self.model.observation_noise_cov, dtype=tf.float32)
        eta_bar_0 = self.eta_bar_0  # Already TF tensor from predict()
        R_inv = tf.linalg.inv(R)
        particles_flow = self.particles.value()

        # --- Precompute score correction quantities (constant over λ) ---
        # Score correction (Corollary 2.1, Eq. 8):
        #   A = A_det - (q/2)·Σ(λ)⁻¹
        #   b = b_det + (q/2)·Σ(λ)⁻¹·m(λ)
        # where Σ(λ)⁻¹ = P⁻¹ + λ·Hᵀ R⁻¹ H  (precision)
        # and   Σ(λ)⁻¹·m(λ) = P⁻¹·η̄₀ + λ·Hᵀ R⁻¹·y  (info vector)
        q = self.diffusion_scale
        if q > 0:
            H_tf = self.model.observation_jacobian(eta_bar_0)
            P_inv = tf.linalg.inv(P)
            H_T_R_inv_H = tf.transpose(H_tf) @ R_inv @ H_tf
            P_inv_eta = tf.linalg.matvec(P_inv, eta_bar_0)
            H_T_R_inv_y = tf.linalg.matvec(
                tf.transpose(H_tf) @ R_inv, observation)

        # --- Compute optimal schedule if μ > 0 ---
        if self.schedule_mu > 0:
            beta_vals, dbeta_vals = self._compute_optimal_schedule(P, R_inv)

        # --- Flow loop (all TF tensor operations) ---
        lambda_val = tf.constant(0.0, dtype=tf.float32)
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

            # Compute A_det(β), b_det(β) using Exact Flow formulas
            # KEY: linearize at eta_bar_0 (fixed), NOT at flowing mean
            A, b = compute_flow_params(
                self.model, eta_bar_0, homotopy_param,
                observation, P, R, R_inv, eta_bar_0, self.state_dim
            )

            # Score correction: compensate drift for diffusion
            # (Theorem 2.1, Eq. 6; Corollary 2.1, Eq. 8)
            if q > 0:
                correction_A = P_inv + homotopy_param * H_T_R_inv_H
                correction_b = P_inv_eta + homotopy_param * H_T_R_inv_y
                A = A - (q / 2) * correction_A
                b = b + (q / 2) * correction_b

            # Euler step with effective step size dβ
            particles_flow = euler_step(
                particles_flow, self._compute_drift,
                step_size, A, b
            )

            # SDE noise: √(q · dλ) · dW — uses dλ (not dβ), since
            # Brownian motion is parameterized by λ
            if q > 0:
                seed = tf.constant([self.seed_counter, i], dtype=tf.int32)
                noise = tf.random.stateless_normal(
                    tf.shape(particles_flow), seed=seed,
                    dtype=particles_flow.dtype
                )
                particles_flow = particles_flow + noise * tf.sqrt(q * d_lambda)

        self.seed_counter += 1

        # --- Finalize ---
        self.particles.assign(particles_flow)
        self.global_filter.update(y.numpy() if isinstance(y, tf.Tensor) else y)
