"""
Debug tests for Two-Sensor Bearing paper reproduction (arXiv:2107.04672, Section 4).

Problem: Both linear (schedule_mu=0.0) and optimal (schedule_mu=0.2) experiments give
identical MSE and tr(P), with tr(P) ~ 2.46 instead of paper's 1535.2 (linear) and
1028.8 (optimal).

Run:
  cd /Users/haowuduan/Documents/githubrepos/HMC-DPF-OT/code
  python -m pytest tests/test_two_sensor_paper.py -v -s
"""

import pytest
import numpy as np
import tensorflow as tf

from src.models.two_sensor_bearing import TwoSensorBearingOnlyModel
from src.filters.particle.stochastic_edh_paper import StochasticEDHFlowPaper
from src.utils.flow_params import compute_flow_params_global


# ============================================================================
# Test 1 — Model Parameters Match Paper
# ============================================================================

def test_model_parameters():
    """Verify model constants exactly match Section 4 of arXiv:2107.04672."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    np.testing.assert_allclose(model.mu_0.numpy(), [3.0, 5.0])
    np.testing.assert_allclose(np.diag(model.Sigma_0.numpy()), [1000.0, 2.0])
    np.testing.assert_allclose(np.diag(model.R.numpy()), [0.04, 0.04])
    np.testing.assert_allclose(model.sensor_positions.numpy(),
                               [[3.5, 0.0], [-3.5, 0.0]])


# ============================================================================
# Test 2 — Observation Jacobian H at Prior Mean
# ============================================================================

def test_observation_jacobian_at_prior():
    """Verify H = dh/dx at eta_bar_0 = [3.0, 5.0] matches analytical formula."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    eta_bar = tf.constant([3.0, 5.0], dtype=tf.float64)
    H = model.observation_jacobian(eta_bar).numpy()
    assert H.shape == (2, 2)

    # Row 0: sensor 1 at (3.5, 0)
    dx1, dy1 = 3.0 - 3.5, 5.0 - 0.0  # -0.5, 5.0
    r1sq = dx1**2 + dy1**2  # 25.25
    np.testing.assert_allclose(H[0, 0], -dy1 / r1sq, rtol=1e-5)  # ~ -0.198
    np.testing.assert_allclose(H[0, 1],  dx1 / r1sq, rtol=1e-5)  # ~ -0.0198

    # Row 1: sensor 2 at (-3.5, 0)
    dx2, dy2 = 3.0 - (-3.5), 5.0 - 0.0  # 6.5, 5.0
    r2sq = dx2**2 + dy2**2  # 67.25
    np.testing.assert_allclose(H[1, 0], -dy2 / r2sq, rtol=1e-5)
    np.testing.assert_allclose(H[1, 1],  dx2 / r2sq, rtol=1e-5)


# ============================================================================
# Test 3 — Flow Params A_det and b_det at Key Lambda Values
# ============================================================================

def test_flow_params_A_at_lambda_half():
    """Verify A(0.5) matches analytical formula and has negative eigenvalues."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    P = tf.constant(np.diag([1000.0, 2.0]), dtype=tf.float64)
    R = model.R
    R_inv = tf.linalg.inv(R)
    eta_bar = tf.constant([3.0, 5.0], dtype=tf.float64)
    H = model.observation_jacobian(eta_bar)
    z = tf.constant([0.4754, 1.1868], dtype=tf.float64)

    lam = tf.constant(0.5, dtype=tf.float64)
    A, b = compute_flow_params_global(H, lam, z, P, R, R_inv, eta_bar, 2)

    # A must have negative eigenvalues (stable flow)
    eigvals = np.linalg.eigvals(A.numpy())
    print(f"\nA eigenvalues at lambda=0.5: {eigvals}")
    assert np.all(np.real(eigvals) < 0), \
        f"A has non-negative eigenvalue: {eigvals}"

    # Check A formula: A = -0.5 * P @ H^T @ (lambda*HPH + R)^-1 @ H
    H_np = H.numpy()
    P_np = P.numpy()
    HPH = H_np @ P_np @ H_np.T
    S = 0.5 * HPH + np.diag([0.04, 0.04])
    A_expected = -0.5 * P_np @ H_np.T @ np.linalg.inv(S) @ H_np
    np.testing.assert_allclose(A.numpy(), A_expected, rtol=1e-5)


# ============================================================================
# Test 4 — Score Correction Eigenvalues at Key Lambda Values
# ============================================================================

def test_score_correction_eigenvalues():
    """Check A_stoch = A_det - (Q/2)*Sigma_inv eigenvalues at multiple lambda."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    P = tf.constant(np.diag([1000.0, 2.0]), dtype=tf.float64)
    R = model.R
    R_inv = tf.linalg.inv(R)
    eta_bar = tf.constant([3.0, 5.0], dtype=tf.float64)
    H = model.observation_jacobian(eta_bar)
    z = tf.constant([0.4754, 1.1868], dtype=tf.float64)
    Q_diag = np.array([4.0, 0.4])

    print("\nScore correction eigenvalue analysis:")
    for lam_val in [0.01, 0.1, 0.5, 1.0]:
        lam = tf.constant(lam_val, dtype=tf.float64)
        A_det, _ = compute_flow_params_global(H, lam, z, P, R, R_inv, eta_bar, 2)

        # Score correction: Sigma_inv = P_inv + lambda * H^T R^-1 H
        P_inv = tf.linalg.inv(P)
        Sigma_inv = P_inv + lam * tf.transpose(H) @ R_inv @ H

        # A_stoch = A_det - (Q/2) * Sigma_inv  (row-scaled)
        Q_half = tf.constant(Q_diag / 2, dtype=tf.float64)
        A_stoch = A_det - tf.expand_dims(Q_half, 1) * Sigma_inv

        eigvals = np.linalg.eigvals(A_stoch.numpy())
        print(f"  lambda={lam_val}: A_stoch eigenvalues = {eigvals}")
        # KEY: are eigenvalues positive (blow-up) or negative (collapse)?


# ============================================================================
# Test 5 — Lambda Schedule: Uniform
# ============================================================================

def test_uniform_lambda_schedule():
    """Verify uniform schedule gives exactly d_lambda=1/100 for n_lambda_steps=100."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=10, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], lambda_schedule='uniform')

    steps = filt.lambda_steps.numpy()
    np.testing.assert_allclose(steps, np.ones(100) / 100.0, rtol=1e-10)
    np.testing.assert_allclose(steps.sum(), 1.0, rtol=1e-10)


# ============================================================================
# Test 6 — BVP Solver: Optimal Schedule Differs from Linear
# ============================================================================

def test_bvp_produces_different_schedule():
    """With mu=0.2, BVP should produce beta*(lambda) != lambda."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt_opt = StochasticEDHFlowPaper(
        model, n_particles=10, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.2, lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    filt_opt.initialize(rng)
    filt_opt.predict()

    P = filt_opt.predicted_cov
    R_inv = tf.linalg.inv(model.R)
    beta_vals, dbeta_vals = filt_opt._compute_optimal_schedule(P, R_inv)

    beta_np = beta_vals.numpy()
    lambda_np = np.cumsum(np.ones(100) / 100.0)

    # Must be different from linear
    max_diff = np.max(np.abs(beta_np - lambda_np))
    print(f"\nMax |beta*(lambda) - lambda| = {max_diff:.4f}")
    assert max_diff > 0.01, "BVP returned linear schedule (may have bracket failure)"


# ============================================================================
# Test 7 — StochasticEDHFlowPaper Uses Exact Prior for EKF
# ============================================================================

def test_paper_filter_uses_exact_prior():
    """Verify initialize() sets EKF mean/cov to model.mu_0/Sigma_0, NOT particle empirical."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], lambda_schedule='uniform')

    rng = np.random.default_rng(42)
    filt.initialize(rng)

    # EKF must be at exact prior, not particle empirical
    np.testing.assert_allclose(filt.global_filter.mean.numpy(),
                               [3.0, 5.0], rtol=1e-6)
    np.testing.assert_allclose(np.diag(filt.global_filter.cov.numpy()),
                               [1000.0, 2.0], rtol=1e-6)
    # eta_bar_0 and predicted_cov must also be exact prior
    np.testing.assert_allclose(filt.eta_bar_0.numpy(), [3.0, 5.0], rtol=1e-6)
    np.testing.assert_allclose(np.diag(filt.predicted_cov.numpy()),
                               [1000.0, 2.0], rtol=1e-6)


# ============================================================================
# Test 8 — predict() Does NOT Feed Particle Mean to EKF
# ============================================================================

def test_predict_no_particle_feedback():
    """After predict(), eta_bar_0 must come from EKF, not particle empirical mean."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    filt.initialize(rng)

    particle_mean_before = filt.particles.value().numpy().mean(axis=0)
    filt.predict()
    eta_bar_after = filt.eta_bar_0.numpy()

    print(f"\nParticle empirical mean: {particle_mean_before}")
    print(f"eta_bar_0 after predict: {eta_bar_after}")

    # eta_bar_0 must be EKF predicted mean (~ [3,5] for identity transition)
    # NOT the particle empirical mean (which differs significantly)
    np.testing.assert_allclose(eta_bar_after, [3.0, 5.0], atol=0.1)

    # Particle empirical mean should differ significantly from [3,5]
    # (50 draws from broad prior); this confirms it's NOT being used
    diff = np.abs(particle_mean_before - np.array([3.0, 5.0]))
    assert diff[0] > 0.5 or diff[1] > 0.1, \
        "Particle mean accidentally equals prior mean — weaken test"


# ============================================================================
# Test 9 — Particle Trajectory: Is tr(P) Growing or Collapsing?
# ============================================================================

def test_particle_spread_during_flow():
    """Instrument the flow to print tr(P) at each step."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.0, lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    filt.initialize(rng)
    filt.predict()

    # Manually run the flow loop with instrumentation
    z = tf.constant([0.4754, 1.1868], dtype=tf.float64)
    P = filt.predicted_cov
    R = model.R
    R_inv = tf.linalg.inv(R)
    eta_bar_0 = filt.eta_bar_0
    H_fixed = model.observation_jacobian(eta_bar_0)
    Q_tf = tf.constant([4.0, 0.4], dtype=tf.float64)
    P_inv = tf.linalg.inv(P)
    H_T_R_inv_H = tf.transpose(H_fixed) @ R_inv @ H_fixed
    P_inv_eta = tf.linalg.matvec(P_inv, eta_bar_0)
    H_T_R_inv_y = tf.linalg.matvec(tf.transpose(H_fixed) @ R_inv, z)

    particles = filt.particles.value()
    lambda_val = tf.constant(0.0, dtype=tf.float64)

    print("\nParticle spread during flow:")
    print(f"  step   0: tr(P) = {np.trace(np.cov(particles.numpy().T)):.4f}")

    for i in range(100):
        d_lambda = filt.lambda_steps[i]
        lambda_val = lambda_val + d_lambda
        homotopy_param = lambda_val  # linear schedule

        A, b = compute_flow_params_global(
            H_fixed, homotopy_param, z, P, R, R_inv, eta_bar_0, 2)

        correction_A = P_inv + homotopy_param * H_T_R_inv_H
        correction_b = P_inv_eta + homotopy_param * H_T_R_inv_y
        A = A - tf.expand_dims(Q_tf, 1) / 2 * correction_A
        b = b + (Q_tf / 2) * correction_b

        drift = particles @ tf.transpose(A) + b
        particles = particles + d_lambda * drift

        seed = tf.constant([0, i], dtype=tf.int32)
        noise = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=tf.float64)
        particles = particles + noise * tf.sqrt(Q_tf * d_lambda)

        if (i + 1) % 10 == 0:
            cov = np.cov(particles.numpy().T)
            print(f"  step {i+1:3d}: tr(P) = {np.trace(cov):.4f}, "
                  f"mean = {particles.numpy().mean(axis=0)}")


# ============================================================================
# Test 9b — Stiffness: tr(P) vs n_lambda_steps
# ============================================================================

def test_stiffness_vs_num_steps():
    """With fewer lambda steps, Euler becomes unstable and tr(P) should blow up.

    At lambda=0.01, A_stoch eigenvalue ~ -45.9. Euler stability requires
    |1 + d_lambda * eig| < 1, i.e. d_lambda < 2/45.9 ~ 0.044.
    So n_lambda_steps < 23 should show instability.
    """
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    z_fixed = np.array([[0.4754, 1.1868]])

    print("\nStiffness vs n_lambda_steps (linear schedule):")
    print(f"  {'n_steps':>7s}  {'d_lambda':>8s}  {'tr(P)':>12s}  {'mean':>24s}")
    for n_steps in [10, 15, 20, 30, 50, 100]:
        filt = StochasticEDHFlowPaper(
            model, n_particles=50, n_lambda_steps=n_steps,
            diffusion_scale=[4.0, 0.4], schedule_mu=0.0, lambda_schedule='uniform')
        rng = np.random.default_rng(0)
        result = filt.filter(z_fixed, random_state=rng)
        trP = float(np.trace(result.covs[0]))
        mean = result.means[0]
        print(f"  {n_steps:7d}  {1.0/n_steps:8.4f}  {trP:12.2f}  {mean}")


# ============================================================================
# Test 10 — tr(P) Matches Paper Definition
# ============================================================================

def test_trP_is_particle_sample_cov():
    """Verify result.covs[0] uses /N (not /N-1) and matches _estimate_mean_cov."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.0, lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    z_fixed = np.array([[0.4754, 1.1868]])
    result = filt.filter(z_fixed, random_state=rng)

    # What _estimate_mean_cov computes: divides by N, not N-1
    particles_final = filt.particles.value().numpy()
    mean = particles_final.mean(axis=0)
    diff = particles_final - mean
    cov_biased = (diff.T @ diff) / 50  # divides by N

    cov_unbiased = np.cov(particles_final.T)  # divides by N-1

    print(f"\nresult.covs[0] (from filter):\n{result.covs[0]}")
    print(f"tr(P) reported: {np.trace(result.covs[0]):.4f}")
    print(f"tr(biased cov, /N): {np.trace(cov_biased):.4f}")
    print(f"tr(unbiased cov, /(N-1)): {np.trace(cov_unbiased):.4f}")

    # Must match the biased version (flow_base uses /N)
    np.testing.assert_allclose(result.covs[0], cov_biased, rtol=1e-5)


# ============================================================================
# Test 11 — Linear vs Optimal Give DIFFERENT Results
# ============================================================================

def test_linear_vs_optimal_differ():
    """With seed=0, linear (mu=0.0) and optimal (mu=0.2) must produce different tr(P)."""
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    z_fixed = np.array([[0.4754, 1.1868]])

    filt_lin = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.0, lambda_schedule='uniform')
    filt_opt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.2, lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    res_lin = filt_lin.filter(z_fixed, random_state=rng)

    rng = np.random.default_rng(0)
    res_opt = filt_opt.filter(z_fixed, random_state=rng)

    trP_lin = float(np.trace(res_lin.covs[0]))
    trP_opt = float(np.trace(res_opt.covs[0]))
    print(f"\nLinear tr(P) = {trP_lin:.2f}, Optimal tr(P) = {trP_opt:.2f}")

    # They must differ (otherwise BVP is falling back to linear).
    # With endpoint evaluation the filter converges well, so difference is modest.
    assert abs(trP_lin - trP_opt) > 0.01, \
        f"Linear and optimal give same tr(P) = {trP_lin:.2f}"


# ============================================================================
# Test 12 — Jacobian Stiffness Ratio Diagnostic
# ============================================================================

def test_jacobian_stiffness_ratio():
    """Compare R_stiff(F) vs κ_ν(M) at several β values.

    R_stiff uses eigenvalues of the flow Jacobian F = -½M⁻¹J_meas - diag(Q/2)·M.
    κ_ν uses the precision matrix M = P⁻¹ + β·H^TR⁻¹H.
    These are different — R_stiff should be much larger at β=0.
    """
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    P = np.diag([1000.0, 2.0])
    R_inv = np.linalg.inv(np.diag([0.04, 0.04]))
    eta_bar = np.array([3.0, 5.0])
    H = model.observation_jacobian(tf.constant(eta_bar, dtype=tf.float64)).numpy()
    J_prior = np.linalg.inv(P)
    J_meas = H.T @ R_inv @ H
    Q_diag = np.array([4.0, 0.4])

    print("\nJacobian stiffness ratio vs κ_ν(M):")
    print(f"  {'β':>6s}  {'R_stiff(F)':>12s}  {'R_stiff(F,Q=0)':>14s}  {'κ_ν(M)':>10s}")

    for beta in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
        M = J_prior + beta * J_meas
        M_inv = np.linalg.inv(M)

        # κ_ν(M) = tr(M)·tr(M⁻¹)
        kappa_nu = np.trace(M) * np.trace(M_inv)

        # F with diffusion
        F_stoch = -0.5 * M_inv @ J_meas - np.diag(Q_diag / 2.0) @ M
        eigvals_stoch = np.linalg.eigvals(F_stoch)
        re_stoch = np.abs(np.real(eigvals_stoch))
        R_stiff_stoch = np.max(re_stoch) / max(np.min(re_stoch), 1e-15)

        # F without diffusion (deterministic)
        F_det = -0.5 * M_inv @ J_meas
        eigvals_det = np.linalg.eigvals(F_det)
        re_det = np.abs(np.real(eigvals_det))
        R_stiff_det = np.max(re_det) / max(np.min(re_det), 1e-15)

        print(f"  {beta:6.2f}  {R_stiff_stoch:12.1f}  {R_stiff_det:14.1f}  {kappa_nu:10.1f}")

    # Validate: at β=0, R_stiff >> κ_ν
    M0 = J_prior
    F0_det = -0.5 * np.linalg.inv(M0) @ J_meas
    eig0 = np.abs(np.real(np.linalg.eigvals(F0_det)))
    R_stiff_0 = np.max(eig0) / max(np.min(eig0), 1e-15)
    kappa_0 = np.trace(M0) * np.trace(np.linalg.inv(M0))
    assert R_stiff_0 > kappa_0, \
        f"Expected R_stiff(0) > κ_ν(0), got {R_stiff_0:.1f} vs {kappa_0:.1f}"


# ============================================================================
# Test 13 — Jacobian Schedule Shape
# ============================================================================

def test_jacobian_schedule_shape():
    """Solve BVP with Jacobian stiffness and print schedule shape.

    Paper (arXiv:2107.04672) reports:
    - max|β*(λ) - λ| ≈ 0.14  (Figure 2b)
    - β̇*(0) ≈ 14  (Figure 2c)
    """
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=10, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.2,
        schedule_method='jacobian', lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    filt.initialize(rng)
    filt.predict()

    P = filt.predicted_cov
    R_inv = tf.linalg.inv(model.R)
    beta_vals, dbeta_vals = filt._compute_jacobian_schedule(P, R_inv)

    beta_np = beta_vals.numpy()
    dbeta_np = dbeta_vals.numpy()
    lambda_np = np.cumsum(np.ones(100) / 100.0)

    # Also compute κ_ν schedule for comparison
    beta_kn, dbeta_kn = filt._compute_optimal_schedule(P, R_inv)
    beta_kn_np = beta_kn.numpy()

    max_diff_jac = np.max(np.abs(beta_np - lambda_np))
    max_diff_kn = np.max(np.abs(beta_kn_np - lambda_np))
    beta_dot_0_jac = dbeta_np[0] / (1.0 / 100)  # dβ/dλ at first step
    beta_dot_0_kn = dbeta_kn.numpy()[0] / (1.0 / 100)

    print("\nSchedule comparison (Jacobian vs κ_ν vs paper):")
    print(f"  {'Method':<12s}  {'max|β*-λ|':>10s}  {'β̇*(0)':>8s}")
    print(f"  {'Jacobian':<12s}  {max_diff_jac:10.4f}  {beta_dot_0_jac:8.2f}")
    print(f"  {'κ_ν(M)':<12s}  {max_diff_kn:10.4f}  {beta_dot_0_kn:8.2f}")
    print(f"  {'Paper':<12s}  {'~0.14':>10s}  {'~14':>8s}")

    print(f"\nJacobian β*(λ) at key points:")
    for idx in [0, 4, 9, 24, 49, 74, 99]:
        lam = lambda_np[idx]
        print(f"  λ={lam:.2f}: β*_jac={beta_np[idx]:.4f}, "
              f"β*_kν={beta_kn_np[idx]:.4f}, linear={lam:.4f}")

    # Jacobian schedule should deviate more from linear than κ_ν schedule
    assert max_diff_jac > max_diff_kn, \
        f"Expected Jacobian schedule to deviate more, got {max_diff_jac:.4f} vs {max_diff_kn:.4f}"


# ============================================================================
# Test 14 — End-to-End: Linear vs κ_ν vs Jacobian
# ============================================================================

def test_jacobian_vs_kappa_nu_filter():
    """Compare filter performance across schedule methods at various step counts.

    With 100 steps, Euler is stable and all methods converge equally.
    With fewer steps (10-20), Euler becomes unstable near β=0 where the
    stiff eigenvalue of A ≈ -559 violates |1 + dλ·λ_max| < 1.
    The schedule should matter most in the unstable regime.

    Paper Table 1 (N=200, n_steps=100, 50 MC runs):
    - Linear:  tr(P) = 1535.2 ± 2506
    - Optimal: tr(P) = 1028.8 ± 1508
    """
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    z_fixed = np.array([[0.4754, 1.1868]])
    n_particles = 200
    n_mc = 10

    configs = [
        ('Linear',   0.0, 'kappa_nu'),
        ('κ_ν(M)',   0.2, 'kappa_nu'),
        ('Jacobian', 0.2, 'jacobian'),
    ]

    for n_steps in [10, 20, 50, 100]:
        print(f"\n--- K={n_steps} steps (dλ={1.0/n_steps:.3f}) ---")
        print(f"  {'Method':<12s}  {'mean tr(P)':>12s}  {'std tr(P)':>12s}  "
              f"{'median tr(P)':>12s}")

        for name, mu, method in configs:
            trP_list = []
            for mc in range(n_mc):
                filt = StochasticEDHFlowPaper(
                    model, n_particles=n_particles, n_lambda_steps=n_steps,
                    diffusion_scale=[4.0, 0.4], schedule_mu=mu,
                    schedule_method=method, lambda_schedule='uniform')
                rng = np.random.default_rng(mc)
                result = filt.filter(z_fixed, random_state=rng)
                trP_list.append(float(np.trace(result.covs[0])))

            trP_arr = np.array(trP_list)
            print(f"  {name:<12s}  {np.mean(trP_arr):12.1f}  "
                  f"{np.std(trP_arr):12.1f}  {np.median(trP_arr):12.1f}")

    print(f"\n  Paper reports (K=100): Linear=1535.2±2506, Optimal=1028.8±1508")
