"""
Tests for src/utils/flow_params.py — particle flow parameter computation.

These functions implement Equations (10) and (11) from Li & Coates (2017):
  A(λ) = -1/2 * P @ H^T @ (λ*H@P@H^T + R)^{-1} @ H
  b(λ) = (I + 2λA)[(I + λA)P@H^T@R^{-1}@(z - e) + A@η̄_0]

where e = h(x) - H@x is the linearization correction (zero for linear models).

# Run and save results:
#   .venv/bin/python -m pytest tests/test_utils_flow_params.py -v 2>&1 | tee tests/test_utils_flow_params_results.txt
"""

import pytest
import numpy as np
import tensorflow as tf

from src.utils.flow_params import (
    compute_flow_params,
    compute_flow_params_global,
    compute_flow_params_batch,
    compute_flow_params_batch_global,
)


# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture
def linear_model():
    """A simple 2D→1D linear-Gaussian model for testing flow params."""
    from src.models.linear_gaussian import LinearGaussianModel
    model = LinearGaussianModel(
        F=np.eye(2), B=0.1 * np.eye(2),
        H=np.array([[1.0, 0.0]]),  # observe only first component
        D=np.array([[np.sqrt(0.1)]]),
        dtype=tf.float64,
    )
    return model


@pytest.fixture
def flow_inputs(linear_model):
    """Standard inputs for flow param functions."""
    d = 2  # state_dim
    P = tf.constant(np.diag([4.0, 1.0]), dtype=tf.float64)
    R = tf.constant([[0.1]], dtype=tf.float64)
    R_inv = tf.constant([[10.0]], dtype=tf.float64)  # 1/0.1
    H = tf.constant([[1.0, 0.0]], dtype=tf.float64)
    z = tf.constant([3.0], dtype=tf.float64)
    eta_bar = tf.constant([0.0, 0.0], dtype=tf.float64)
    return {
        'model': linear_model,
        'H': H, 'P': P, 'R': R, 'R_inv': R_inv,
        'z': z, 'eta_bar': eta_bar, 'd': d,
    }


# ============================================================================
# compute_flow_params_global — T4.1 to T4.4
# ============================================================================

class TestComputeFlowParamsGlobal:

    def test_A_formula(self, flow_inputs):
        """T4.1 — A = -0.5 * P @ H^T @ (λ*HPH^T + R)^{-1} @ H.

        Hand-computation for 2D state, 1D observation:
          H = [1, 0], P = diag(4, 1), R = 0.1, λ = 0.5
          HPH^T = H @ P @ H^T = 4.0
          S = 0.5*4 + 0.1 = 2.1
          S^{-1} = 1/2.1
          H^T @ S^{-1} @ H = [[1/2.1, 0], [0, 0]]
          A = -0.5 * P @ [[1/2.1, 0], [0, 0]] = [[-4/(2*2.1), 0], [0, 0]]
        """
        p = flow_inputs
        lam = tf.constant(0.5, dtype=tf.float64)

        A_code, _ = compute_flow_params_global(
            p['H'], lam, p['z'], p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )

        # Manual computation
        H_np = p['H'].numpy()
        P_np = p['P'].numpy()
        R_np = p['R'].numpy()
        S_np = 0.5 * H_np @ P_np @ H_np.T + R_np
        A_expected = -0.5 * P_np @ H_np.T @ np.linalg.inv(S_np) @ H_np

        np.testing.assert_allclose(A_code.numpy(), A_expected, rtol=1e-5)

    def test_b_formula_at_lambda_zero(self, flow_inputs):
        """T4.2 — b at λ=0: b = P@H^T@R^{-1}@z + A@η̄_0.

        At λ=0: I + 0*A = I and I + 2*0*A = I, so
        b = I @ [I @ P @ H^T @ R_inv @ z + A @ eta_bar]
          = P @ H^T @ R_inv @ z + A @ eta_bar
        """
        p = flow_inputs
        lam = tf.constant(0.0, dtype=tf.float64)

        A_code, b_code = compute_flow_params_global(
            p['H'], lam, p['z'], p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )

        H_np = p['H'].numpy()
        P_np = p['P'].numpy()
        R_inv_np = p['R_inv'].numpy()
        z_np = p['z'].numpy()
        eta_bar_np = p['eta_bar'].numpy()

        b_expected = P_np @ H_np.T @ R_inv_np @ z_np + A_code.numpy() @ eta_bar_np
        np.testing.assert_allclose(b_code.numpy(), b_expected.flatten(), rtol=1e-5)

    def test_flow_converges_to_posterior_mean(self, flow_inputs):
        """T4.3 — Euler integration of dx/dt = A@x + b converges to posterior mean.

        For 1D observation of 2D state with H = [1, 0]:
          Only x_1 is observed, so the posterior of x_1 is updated while x_2
          stays at its prior mean.
          posterior_mean_1 = σ_p² * z / (σ_p² + σ_r²)
                           = 4.0 * 3.0 / (4.0 + 0.1) = 12.0 / 4.1 ≈ 2.927
        """
        p = flow_inputs
        n_steps = 500
        dt = 1.0 / n_steps

        x = tf.identity(p['eta_bar'])  # start at prior mean = [0, 0]

        lam_schedule = np.linspace(0.0, 1.0, n_steps + 1)
        for i in range(n_steps):
            lam = tf.constant(lam_schedule[i], dtype=tf.float64)
            A, b = compute_flow_params_global(
                p['H'], lam, p['z'], p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
            )
            dx = tf.linalg.matvec(A, x) + b
            x = x + dx * dt

        # Posterior mean for x_1: P_11 * z / (P_11 + R_11)
        posterior_mean_1 = 4.0 * 3.0 / (4.0 + 0.1)
        np.testing.assert_allclose(x.numpy()[0], posterior_mean_1, rtol=1e-2)
        # x_2 should stay near 0 (not observed)
        np.testing.assert_allclose(x.numpy()[1], 0.0, atol=0.1)

    def test_A_nonpositive_eigenvalues(self, flow_inputs):
        """T4.4 — A has non-positive eigenvalues (stable contracting flow).

        A = -0.5 * P @ H^T @ S^{-1} @ H is negative semi-definite because
        P is PD and H^T @ S^{-1} @ H is PSD.
        """
        p = flow_inputs
        lam = tf.constant(0.5, dtype=tf.float64)

        A_code, _ = compute_flow_params_global(
            p['H'], lam, p['z'], p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )

        eigvals = np.linalg.eigvals(A_code.numpy())
        assert np.all(eigvals.real <= 1e-10), (
            f"A should have non-positive eigenvalues, got: {eigvals.real}"
        )


# ============================================================================
# compute_flow_params (local, with e-correction) — T4.5
# ============================================================================

class TestComputeFlowParamsLocal:

    def test_matches_global_for_linear_model(self, flow_inputs):
        """T4.5 — Matches global when h(x) = H@x (linear model, e=0).

        For a linear observation model h(x) = H@x, the linearization error
        e = h(x) - H@x = 0.  So the local formula reduces to the global one
        when the linearization point is the ensemble mean η̄_0.
        """
        p = flow_inputs
        lam = tf.constant(0.5, dtype=tf.float64)

        A_local, b_local = compute_flow_params(
            p['model'], p['eta_bar'], lam, p['z'],
            p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )
        A_global, b_global = compute_flow_params_global(
            p['H'], lam, p['z'], p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )

        np.testing.assert_allclose(A_local.numpy(), A_global.numpy(), rtol=1e-5)
        np.testing.assert_allclose(b_local.numpy(), b_global.numpy(), rtol=1e-5)


# ============================================================================
# compute_flow_params_batch — T4.6
# ============================================================================

class TestComputeFlowParamsBatch:

    def test_matches_per_particle_loop(self, flow_inputs):
        """T4.6 — Batched result matches per-particle loop of compute_flow_params.

        Run compute_flow_params for each particle individually, then run
        compute_flow_params_batch for all particles at once.  Results must match.
        """
        p = flow_inputs
        N = 5
        lam = tf.constant(0.5, dtype=tf.float64)

        # Generate N particles near the prior mean
        rng = np.random.default_rng(42)
        particles = tf.constant(rng.normal(size=(N, p['d'])), dtype=tf.float64)

        # Per-particle loop
        A_list, b_list = [], []
        for i in range(N):
            A_i, b_i = compute_flow_params(
                p['model'], particles[i], lam, p['z'],
                p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
            )
            A_list.append(A_i.numpy())
            b_list.append(b_i.numpy())

        A_loop = np.stack(A_list)
        b_loop = np.stack(b_list)

        # Batched
        A_batch, b_batch = compute_flow_params_batch(
            p['model'], particles, lam, p['z'],
            p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )

        np.testing.assert_allclose(A_batch.numpy(), A_loop, rtol=1e-5)
        np.testing.assert_allclose(b_batch.numpy(), b_loop, rtol=1e-5)


# ============================================================================
# compute_flow_params_batch_global — T4.7 to T4.8
# ============================================================================

class TestComputeFlowParamsBatchGlobal:

    def test_matches_per_particle_loop(self, flow_inputs):
        """T4.7 — Batched global matches per-particle loop of compute_flow_params_global.

        Same strategy as T4.6 but with the global (no e-correction) variant.
        H_batch is precomputed once and passed in.
        """
        p = flow_inputs
        N = 5
        lam = tf.constant(0.5, dtype=tf.float64)

        # Generate N particles
        rng = np.random.default_rng(42)
        particles = tf.constant(rng.normal(size=(N, p['d'])), dtype=tf.float64)

        # Get batch Jacobians (for linear model, all identical)
        H_batch = p['model'].observation_jacobian_batch(particles)

        # Per-particle loop
        A_list, b_list = [], []
        for i in range(N):
            A_i, b_i = compute_flow_params_global(
                H_batch[i], lam, p['z'],
                p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
            )
            A_list.append(A_i.numpy())
            b_list.append(b_i.numpy())

        A_loop = np.stack(A_list)
        b_loop = np.stack(b_list)

        # Batched
        A_batch, b_batch = compute_flow_params_batch_global(
            H_batch, lam, p['z'],
            p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )

        np.testing.assert_allclose(A_batch.numpy(), A_loop, rtol=1e-5)
        np.testing.assert_allclose(b_batch.numpy(), b_loop, rtol=1e-5)

    def test_all_A_nonpositive_eigenvalues(self, flow_inputs):
        """T4.8 — All A matrices in the batch have non-positive eigenvalues."""
        p = flow_inputs
        N = 5
        lam = tf.constant(0.5, dtype=tf.float64)

        rng = np.random.default_rng(42)
        particles = tf.constant(rng.normal(size=(N, p['d'])), dtype=tf.float64)
        H_batch = p['model'].observation_jacobian_batch(particles)

        A_batch, _ = compute_flow_params_batch_global(
            H_batch, lam, p['z'],
            p['P'], p['R'], p['R_inv'], p['eta_bar'], p['d'],
        )

        for i in range(N):
            eigvals = np.linalg.eigvals(A_batch[i].numpy())
            assert np.all(eigvals.real <= 1e-10), (
                f"Particle {i}: A should have non-positive eigenvalues, got: {eigvals.real}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
