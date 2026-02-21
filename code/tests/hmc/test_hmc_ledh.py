"""
Stage 2: LEDH vs BPF diagnostic tests.

Compares LEDH against the BPF baseline. Any difference must come from
the flow, Jacobian, or importance weight computation.

Ground truth: Kalman filter gives the exact log-likelihood for linear Gaussian.

Run:
  pytest code/tests/test_hmc_ledh.py -v -s 2>&1 | tee ledh_diagnosis.txt
"""

import numpy as np
import tensorflow as tf

from conftest_hmc_diag import (
    DTYPE, N_PARTICLES, N_LAMBDA_STEPS, SEED, T,
    TRUE_OBS_NOISE_STD, INIT_OBS_NOISE_STD, SIGMA_GRID,
    lg_model_and_data, observations_tf,          # fixtures
    _fresh_model, _make_bpf_kwargs, _make_ledh_kwargs,
    kf_log_likelihood, pf_log_likelihood,
    compute_gradient, finite_difference_gradient,
)
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC


# ============================================================================
# Tests 2.1–2.2: LEDH log-likelihood vs KF and BPF
# ============================================================================

class TestLEDHLogLikelihood:
    """Stage 2.1-2.2: LEDH log-likelihood vs KF and BPF."""

    def test_ledh_log_likelihood_vs_kf_and_bpf(self, lg_model_and_data, observations_tf):
        """Test 2.1: LEDH log-lik matches KF and BPF at true parameter."""
        _, observations, _ = lg_model_and_data

        kf_ll = kf_log_likelihood(TRUE_OBS_NOISE_STD, observations)

        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        bpf_kwargs['n_particles'] = 1000
        bpf_ll = pf_log_likelihood(BootstrapPFHMC, bpf_kwargs, TRUE_OBS_NOISE_STD, observations_tf)

        ledh_kwargs = _make_ledh_kwargs('systematic', stop_gradient=True)
        ledh_ll = pf_log_likelihood(
            LEDHParticleFlowFilterHMC, ledh_kwargs, TRUE_OBS_NOISE_STD, observations_tf)

        print(f"\n  KF   log-lik: {kf_ll:.4f}")
        print(f"  BPF  log-lik: {bpf_ll:.4f}")
        print(f"  LEDH log-lik: {ledh_ll:.4f}")
        print(f"  |LEDH - KF|:  {abs(ledh_ll - kf_ll):.4f}")
        print(f"  |LEDH - BPF|: {abs(ledh_ll - bpf_ll):.4f}")

        rel_err = abs(ledh_ll - kf_ll) / abs(kf_ll)
        print(f"  LEDH/KF relative error: {rel_err:.4f}")
        assert rel_err < 0.20, f"LEDH log-lik too far from KF: rel_err={rel_err:.4f}"

    def test_ledh_log_likelihood_surface(self, lg_model_and_data, observations_tf):
        """Test 2.2: Does LEDH log-lik surface peak at correct place?"""
        _, observations, _ = lg_model_and_data

        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        bpf_kwargs['n_particles'] = 1000
        ledh_kwargs = _make_ledh_kwargs('systematic', stop_gradient=True)

        print(f"\n  {'sigma':>6s}  {'kf_ll':>10s}  {'bpf_ll':>10s}  {'ledh_ll':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

        kf_lls, bpf_lls, ledh_lls = [], [], []
        for sigma in SIGMA_GRID:
            kf_ll = kf_log_likelihood(sigma, observations)
            bpf_ll = pf_log_likelihood(BootstrapPFHMC, bpf_kwargs, sigma, observations_tf)
            ledh_ll = pf_log_likelihood(
                LEDHParticleFlowFilterHMC, ledh_kwargs, sigma, observations_tf)
            kf_lls.append(kf_ll)
            bpf_lls.append(bpf_ll)
            ledh_lls.append(ledh_ll)
            print(f"  {sigma:6.2f}  {kf_ll:10.4f}  {bpf_ll:10.4f}  {ledh_ll:10.4f}")

        kf_best = SIGMA_GRID[np.argmax(kf_lls)]
        bpf_best = SIGMA_GRID[np.argmax(bpf_lls)]
        ledh_best = SIGMA_GRID[np.argmax(ledh_lls)]
        print(f"\n  KF   peaks at sigma={kf_best}")
        print(f"  BPF  peaks at sigma={bpf_best}")
        print(f"  LEDH peaks at sigma={ledh_best}")
        assert ledh_best == kf_best, \
            f"LEDH peaks at {ledh_best}, KF at {kf_best} — biased likelihood surface"


# ============================================================================
# Tests 2.3–2.5: LEDH gradient diagnostics
# ============================================================================

class TestLEDHGradient:
    """Stage 2.3-2.5: LEDH gradient diagnostics."""

    def test_ledh_gradient_direction(self, observations_tf):
        """Test 2.3: LEDH gradient points toward true value from both sides."""
        ledh_kwargs = _make_ledh_kwargs('systematic', stop_gradient=True)
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)

        # Above true
        _, ledh_grad_hi, _, _ = compute_gradient(
            LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf, 2.0)
        _, bpf_grad_hi, _, _ = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf, 2.0)

        # Below true
        _, ledh_grad_lo, _, _ = compute_gradient(
            LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf, 0.5)
        _, bpf_grad_lo, _, _ = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf, 0.5)

        print(f"\n  sigma=2.0: LEDH grad={ledh_grad_hi[0]:+.4f}, BPF grad={bpf_grad_hi[0]:+.4f}")
        print(f"  sigma=0.5: LEDH grad={ledh_grad_lo[0]:+.4f}, BPF grad={bpf_grad_lo[0]:+.4f}")

        assert np.sign(ledh_grad_hi[0]) == np.sign(bpf_grad_hi[0]), \
            f"LEDH/BPF gradient direction disagrees at sigma=2.0"
        assert np.sign(ledh_grad_lo[0]) == np.sign(bpf_grad_lo[0]), \
            f"LEDH/BPF gradient direction disagrees at sigma=0.5"

    def test_ledh_autodiff_vs_finite_difference(self, observations_tf):
        """Test 2.4: LEDH autodiff gradient matches finite difference."""
        configs = [
            ('systematic', True),
            ('soft', False),
            ('ot_entropy', False),
        ]

        print(f"\n  {'method':>12s}  {'autodiff':>10s}  {'FD':>10s}  {'rel_err':>10s}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

        for method, stop_grad in configs:
            ledh_kwargs = _make_ledh_kwargs(method, stop_grad)
            _, ad_grad, _, _ = compute_gradient(
                LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf)
            fd_grad = finite_difference_gradient(
                LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf)

            ad_val = ad_grad[0]
            rel_err = abs(ad_val - fd_grad) / max(abs(fd_grad), 1e-10)
            print(f"  {method:>12s}  {ad_val:+10.4f}  {fd_grad:+10.4f}  {rel_err:10.4f}")
            assert rel_err < 0.20, \
                f"LEDH {method}: autodiff/FD mismatch, rel_err={rel_err:.4f}"

    def test_ledh_gradient_by_resampling_method(self, observations_tf):
        """Test 2.5: Compare LEDH gradient across resampling methods."""
        configs = [
            ('systematic', True),
            ('soft', False),
            ('ot_entropy', False),
        ]

        # BPF reference
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        _, bpf_grad, _, bpf_ll = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf)

        print(f"\n  {'filter':>6s}  {'method':>12s}  {'|grad|':>10s}  {'grad':>10s}  {'log_lik':>10s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
        print(f"  {'BPF':>6s}  {'systematic':>12s}  {abs(bpf_grad[0]):10.4f}  {bpf_grad[0]:+10.4f}  {bpf_ll:10.4f}")

        ledh_grads = []
        for method, stop_grad in configs:
            ledh_kwargs = _make_ledh_kwargs(method, stop_grad)
            _, grad, _, ll = compute_gradient(
                LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf)
            ledh_grads.append((method, grad[0]))
            print(f"  {'LEDH':>6s}  {method:>12s}  {abs(grad[0]):10.4f}  {grad[0]:+10.4f}  {ll:10.4f}")

        # All LEDH methods should agree with BPF on direction
        bpf_sign = np.sign(bpf_grad[0])
        for method, g in ledh_grads:
            assert np.sign(g) == bpf_sign, \
                f"LEDH {method} gradient sign ({np.sign(g)}) disagrees with BPF ({bpf_sign})"


# ============================================================================
# Tests 2.6–2.7: LEDH flow and importance weight diagnostics
# ============================================================================

class TestLEDHFlowDiagnostics:
    """Stage 2.6-2.7: LEDH flow and importance weight diagnostics."""

    def test_ledh_jacobian_uniformity(self, lg_model_and_data, observations_tf):
        """Test 2.6: For linear Gaussian (constant H), Jacobians should be identical."""
        _, observations, _ = lg_model_and_data

        model = _fresh_model(TRUE_OBS_NOISE_STD)
        filt = LEDHParticleFlowFilterHMC(
            model, n_particles=N_PARTICLES, n_lambda_steps=N_LAMBDA_STEPS,
            resampling_method='systematic',
            stop_gradient_resampling=True,
            eager_mode=True,
        )
        filt.initialize(random_seed=SEED)

        # Run one predict + manual flow to extract per-particle Jacobians
        filt.predict(t=1)
        y = tf.constant(observations[0], dtype=DTYPE)

        # Run the flow manually to get log_theta per particle
        eta_1 = filt.eta_0.value()
        eta_bar = filt.eta_bar_0.value()
        log_theta = tf.zeros([N_PARTICLES], dtype=DTYPE)
        lambda_val = tf.constant(0.0, dtype=DTYPE)
        I_sd = tf.eye(filt.state_dim, dtype=DTYPE)

        from src.utils.flow_params import compute_flow_params_batch
        from src.utils.linalg import safe_log_abs_det, safe_inv

        R = model.observation_noise_cov
        R_inv = safe_inv(R)
        regularization = tf.constant(1e-8, dtype=DTYPE)

        for j in range(N_LAMBDA_STEPS):
            d_lambda = filt.lambda_steps[j]
            lambda_val = lambda_val + d_lambda

            A_batch, b_batch = compute_flow_params_batch(
                model, eta_bar, lambda_val, y,
                filt.particle_covs.value(),
                R, R_inv, filt.eta_bar_0.value(),
                filt.state_dim, regularization,
            )

            drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
            eta_bar = eta_bar + d_lambda * drift_bar

            drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
            eta_1 = eta_1 + d_lambda * drift_1

            M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
            log_det_M = safe_log_abs_det(M_batch)
            log_theta = log_theta + log_det_M

        log_theta_np = log_theta.numpy()
        mean_lt = np.mean(log_theta_np)
        std_lt = np.std(log_theta_np)
        cv = std_lt / max(abs(mean_lt), 1e-10)

        print(f"\n  Per-particle log_det_J (N={N_PARTICLES}):")
        print(f"    mean:  {mean_lt:.6f}")
        print(f"    std:   {std_lt:.6f}")
        print(f"    min:   {np.min(log_theta_np):.6f}")
        print(f"    max:   {np.max(log_theta_np):.6f}")
        print(f"    CV:    {cv:.6f}")
        assert cv < 0.01, f"Jacobians not uniform: CV={cv:.6f} (expect < 0.01 for linear model)"

    def test_ledh_importance_weights_uniformity(self, lg_model_and_data, observations_tf):
        """Test 2.7: At true parameter, importance weights should be nearly uniform."""
        _, observations, _ = lg_model_and_data

        model = _fresh_model(TRUE_OBS_NOISE_STD)
        filt = LEDHParticleFlowFilterHMC(
            model, n_particles=N_PARTICLES, n_lambda_steps=N_LAMBDA_STEPS,
            resampling_method='systematic',
            resampling_config={},
            weight_clip_range=50.0,
            stop_gradient_resampling=True,
            eager_mode=True,
        )
        filt.initialize(random_seed=SEED)

        obs_tf_local = tf.constant(observations, dtype=DTYPE)

        print(f"\n  {'t':>4s}  {'ESS':>8s}  {'ESS/N':>8s}  {'max_w/min_w':>12s}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*12}")

        ess_ratios = []
        for t in range(T):
            filt.predict(t=t + 1)
            filt.update(obs_tf_local[t])

            w = filt.weights.numpy()
            ess = 1.0 / np.sum(w ** 2)
            ess_ratio = ess / N_PARTICLES
            w_ratio = np.max(w) / max(np.min(w), 1e-30)
            ess_ratios.append(ess_ratio)

            if t < 5 or t >= T - 3 or t % 10 == 0:
                print(f"  {t+1:4d}  {ess:8.1f}  {ess_ratio:8.3f}  {w_ratio:12.1f}")

        avg_ess_ratio = np.mean(ess_ratios)
        print(f"\n  Average ESS/N: {avg_ess_ratio:.3f}")
        assert avg_ess_ratio > 0.5, \
            f"Average ESS/N={avg_ess_ratio:.3f} too low — high-variance importance weights"
