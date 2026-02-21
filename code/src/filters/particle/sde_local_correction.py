"""
Stochastic EDH with local linearization correction.

Same as StochasticEDHFlow but includes the nonlinear correction
e₀ = h(η̄_0) - H·η̄_0, using (z - e₀) instead of z in the flow formulas.
This corrects the bias from global linearization on nonlinear observation models.
"""
import numpy as np
import tensorflow as tf
from .stochastic_edh import StochasticEDHFlow
from ...utils.flow_params import compute_flow_params_global
from ...utils.linalg import safe_inv, to_numpy
from ...utils.ode_solvers import euler_step


class SDELocalCorrection(StochasticEDHFlow):
    """
    Stochastic EDH with linearization correction e₀ = h(η̄_0) - H·η̄_0.

    Global linearization (H fixed at η̄_0, no relinearization) but uses
    z_corrected = z - e₀ in both the deterministic drift and score correction.
    This accounts for the nonlinear residual of h(x) around η̄_0.

    For linear observation models (h(x) = Hx), e₀ = 0 and this is
    identical to StochasticEDHFlow.
    """

    def update(self, y: tf.Tensor):
        """
        Stochastic flow from λ=0 to λ=1 with linearization correction.

        Same as StochasticEDHFlow.update() but uses z_corrected = z - e₀
        where e₀ = h(η̄_0) - H·η̄_0 is computed once.
        """
        # --- Setup ---
        observation = y
        P = self.predicted_cov
        R = tf.cast(self.model.observation_noise_cov, self.dtype)
        eta_bar_0 = self.eta_bar_0
        R_inv = safe_inv(R)
        particles_flow = self.particles.value()

        # Compute H once at η̄_0 (global linearization)
        H_fixed = self.model.observation_jacobian(eta_bar_0)

        # Compute linearization correction: e₀ = h(η̄_0) - H·η̄_0
        h_eta = self.model.observation_function(eta_bar_0)
        e_0 = h_eta - tf.linalg.matvec(H_fixed, eta_bar_0)

        # Corrected observation: z - e₀
        z_corrected = observation - e_0

        # --- Precompute score correction quantities (constant over λ) ---
        if self._has_diffusion:
            Q_tf = tf.constant(self.Q_diag, dtype=self.dtype)
            P_inv = safe_inv(P)
            H_T_R_inv_H = tf.transpose(H_fixed) @ R_inv @ H_fixed
            P_inv_eta = tf.linalg.matvec(P_inv, eta_bar_0)
            # Use z_corrected for score correction
            H_T_R_inv_y = tf.linalg.matvec(
                tf.transpose(H_fixed) @ R_inv, z_corrected)

        # --- Compute optimal schedule if μ > 0 ---
        if self.schedule_mu > 0:
            beta_vals, dbeta_vals = self._compute_optimal_schedule(P, R_inv)

        # --- Flow loop ---
        lambda_val = tf.constant(0.0, dtype=self.dtype)
        for i in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[i]
            lambda_val = lambda_val + d_lambda

            # Schedule
            if self.schedule_mu > 0:
                homotopy_param = beta_vals[i]
                step_size = dbeta_vals[i]
            else:
                homotopy_param = lambda_val
                step_size = d_lambda

            # Compute A_det(β), b_det(β) with corrected observation
            A, b = compute_flow_params_global(
                H_fixed, homotopy_param,
                z_corrected, P, R, R_inv, eta_bar_0, self.state_dim
            )

            # Score correction (diagonal Q)
            if self._has_diffusion:
                correction_A = P_inv + homotopy_param * H_T_R_inv_H
                correction_b = P_inv_eta + homotopy_param * H_T_R_inv_y
                A = A - tf.expand_dims(Q_tf, 1) / 2 * correction_A
                b = b + (Q_tf / 2) * correction_b

            # Euler step
            particles_flow = euler_step(
                particles_flow, self._compute_drift,
                step_size, A, b
            )

            # SDE noise: √(Q_diag · dλ) · dW
            if self._has_diffusion:
                seed = self._next_seed()
                noise = tf.random.stateless_normal(
                    tf.shape(particles_flow), seed=seed,
                    dtype=particles_flow.dtype
                )
                particles_flow = particles_flow + noise * tf.sqrt(Q_tf * d_lambda)

        # --- Finalize ---
        self.particles.assign(particles_flow)
        self.global_filter.update(to_numpy(y))
