"""
Range-Bearing HMC Diagnostic Tests.

Diagnoses why HMC parameter inference fails for particle filters (BPF, LEDH)
on the Range-Bearing model while EKF/UKF work well.

Tests focus on:
  A. Gradient magnitude comparison across filters
  B. Gradient direction (does it point toward truth?)
  C. Autodiff vs finite difference (is backward pass correct?)
  D. 1D likelihood surface scans
  E. Per-parameter gradient analysis
  F. Curvature (Hessian diagonal) comparison
  G. LEDH post-fix verification

Run:
    cd code
    pytest tests/hmc/test_rb_hmc_diagnosis.py -v -s
"""

import pytest
import numpy as np
import tensorflow as tf

from conftest_rb_diag import (
    DTYPE, N_PARTICLES, SEED, T,
    TRUE_SIGMA_RANGE, TRUE_SIGMA_BEARING,
    INIT_SIGMA_RANGE, INIT_SIGMA_BEARING,
    SIGMA_SCAN_GRID,
    rb_model_and_data, rb_observations_tf,  # noqa: F401 — pytest fixtures
    _fresh_rb_model, _make_rb_param_specs,
    _make_rb_bpf_kwargs, _make_rb_ledh_kwargs,
    _make_fresh_rb_runner,
    ekf_log_likelihood, ukf_log_likelihood, pf_log_likelihood_rb,
    compute_gradient_rb, finite_difference_gradient_rb,
    compute_likelihood_gradient_rb,
)
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
from src.filters.kalman.unscented_kalman import UnscentedKalmanFilter
from src.DF.parameter_handler import ParameterHandler


# ============================================================================
# Filter configs: (name, filter_class, kwargs_factory)
# ============================================================================

def _ekf_kwargs():
    return {'sample_initial_mean': False}

def _ukf_kwargs():
    return {'alpha': 1.0, 'beta': 2.0, 'kappa': 0.0, 'sample_initial_mean': False}

FILTER_CONFIGS = [
    ('EKF',       ExtendedKalmanFilter,       _ekf_kwargs),
    ('UKF',       UnscentedKalmanFilter,       _ukf_kwargs),
    ('BPF_sys',   BootstrapPFHMC, lambda: _make_rb_bpf_kwargs('systematic', True)),
    ('BPF_ot',    BootstrapPFHMC, lambda: _make_rb_bpf_kwargs('ot_entropy', False)),
    ('BPF_soft',  BootstrapPFHMC, lambda: _make_rb_bpf_kwargs('soft', False)),
    ('LEDH_sys',  LEDHParticleFlowFilterHMC, lambda: _make_rb_ledh_kwargs('systematic', True)),
    ('LEDH_ot',   LEDHParticleFlowFilterHMC, lambda: _make_rb_ledh_kwargs('ot_entropy', False)),
    ('LEDH_soft', LEDHParticleFlowFilterHMC, lambda: _make_rb_ledh_kwargs('soft', False)),
]

# Subset for expensive tests
FAST_CONFIGS = [
    ('EKF',      ExtendedKalmanFilter,  _ekf_kwargs),
    ('UKF',      UnscentedKalmanFilter, _ukf_kwargs),
    ('BPF_ot',   BootstrapPFHMC, lambda: _make_rb_bpf_kwargs('ot_entropy', False)),
    ('BPF_sys',  BootstrapPFHMC, lambda: _make_rb_bpf_kwargs('systematic', True)),
]

# Param names in order (must match ParameterHandler ordering)
PARAM_NAMES = ['sigma_range', 'sigma_bearing']


# ============================================================================
# Test A: Gradient Magnitude Comparison
# ============================================================================

class TestGradientMagnitude:
    """Compare gradient magnitude across all filters at init (0.3, 0.3)."""

    def test_gradient_magnitude_all_filters(self, rb_observations_tf):
        print(f"\n  Gradient magnitude at init ({INIT_SIGMA_RANGE}, {INIT_SIGMA_BEARING}):")
        print(f"  {'filter':>10s}  {'|grad|':>10s}  {'grad_sr':>10s}  {'grad_sb':>10s}"
              f"  {'nlp':>10s}  {'ll':>10s}  {'ratio_vs_EKF':>14s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

        ekf_grad_norm = None
        results = []

        for name, filt_class, kwargs_fn in FILTER_CONFIGS:
            nlp, grad, constrained, ll = compute_gradient_rb(
                filt_class, kwargs_fn(), rb_observations_tf)
            grad_norm = np.linalg.norm(grad)

            if name == 'EKF':
                ekf_grad_norm = grad_norm

            ratio = f"{grad_norm / ekf_grad_norm:.1f}x" if ekf_grad_norm else "---"
            print(f"  {name:>10s}  {grad_norm:10.4f}  {grad[0]:+10.4f}  {grad[1]:+10.4f}"
                  f"  {nlp:10.4f}  {ll:10.4f}  {ratio:>14s}")
            results.append((name, grad, grad_norm))

        # Assert: no NaN/Inf gradients
        for name, grad, norm in results:
            assert np.all(np.isfinite(grad)), f"{name} has NaN/Inf gradient: {grad}"


# ============================================================================
# Test B: Gradient Direction
# ============================================================================

class TestGradientDirection:
    """At init (0.3, 0.3), gradient of NLP should be positive
    (decrease q → decrease sigma toward truth at 0.1, 0.1)."""

    def test_gradient_points_toward_truth(self, rb_observations_tf):
        print(f"\n  Gradient direction at init ({INIT_SIGMA_RANGE}, {INIT_SIGMA_BEARING}):")
        print(f"  {'filter':>10s}  {'grad_sr':>10s}  {'grad_sb':>10s}  {'dir_correct':>12s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

        for name, filt_class, kwargs_fn in FILTER_CONFIGS:
            nlp, grad, constrained, ll = compute_gradient_rb(
                filt_class, kwargs_fn(), rb_observations_tf)

            # For NLP gradient: positive means "NLP increases as q increases"
            # Since sigma(0.3) > truth(0.1), we want to decrease q,
            # so the descent direction (-grad) should be negative → grad should be positive
            sr_correct = grad[0] > 0
            sb_correct = grad[1] > 0
            both_correct = sr_correct and sb_correct
            label = "YES" if both_correct else "NO"
            if not sr_correct:
                label += " (sr wrong)"
            if not sb_correct:
                label += " (sb wrong)"

            print(f"  {name:>10s}  {grad[0]:+10.4f}  {grad[1]:+10.4f}  {label:>12s}")

        # Hard assert on EKF/UKF
        for name, filt_class, kwargs_fn in FILTER_CONFIGS[:2]:
            _, grad, _, _ = compute_gradient_rb(
                filt_class, kwargs_fn(), rb_observations_tf)
            assert grad[0] > 0, f"{name} sigma_range gradient should be positive at init"
            assert grad[1] > 0, f"{name} sigma_bearing gradient should be positive at init"


# ============================================================================
# Test C: Autodiff vs Finite Difference
# ============================================================================

class TestAutodiffVsFiniteDifference:
    """Check if autodiff matches finite differences for each filter."""

    @pytest.mark.parametrize("name,filt_class,kwargs_fn", FAST_CONFIGS,
                             ids=[c[0] for c in FAST_CONFIGS])
    def test_autodiff_vs_fd(self, rb_observations_tf, name, filt_class, kwargs_fn):
        kwargs = kwargs_fn()
        nlp, ad_grad, _, _ = compute_gradient_rb(
            filt_class, kwargs, rb_observations_tf)
        fd_grad = finite_difference_gradient_rb(
            filt_class, kwargs, rb_observations_tf)

        print(f"\n  {name} autodiff vs FD at init ({INIT_SIGMA_RANGE}, {INIT_SIGMA_BEARING}):")
        print(f"  {'component':>14s}  {'autodiff':>10s}  {'FD':>10s}  {'rel_err':>10s}  {'sign':>6s}")
        print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

        for i, pname in enumerate(PARAM_NAMES):
            rel_err = abs(ad_grad[i] - fd_grad[i]) / max(abs(fd_grad[i]), 1e-10)
            sign_match = "OK" if np.sign(ad_grad[i]) == np.sign(fd_grad[i]) else "FAIL"
            print(f"  {pname:>14s}  {ad_grad[i]:+10.4f}  {fd_grad[i]:+10.4f}"
                  f"  {rel_err:10.4f}  {sign_match:>6s}")

        # EKF/UKF: should match closely (smooth deterministic surface)
        if name in ('EKF', 'UKF'):
            for i in range(len(PARAM_NAMES)):
                rel_err = abs(ad_grad[i] - fd_grad[i]) / max(abs(fd_grad[i]), 1e-10)
                assert rel_err < 0.1, (
                    f"{name} {PARAM_NAMES[i]}: AD={ad_grad[i]:.4f} vs FD={fd_grad[i]:.4f}, "
                    f"rel_err={rel_err:.4f}"
                )

        # BPF_ot (differentiable resampling): should match reasonably
        if name == 'BPF_ot':
            for i in range(len(PARAM_NAMES)):
                rel_err = abs(ad_grad[i] - fd_grad[i]) / max(abs(fd_grad[i]), 1e-10)
                assert rel_err < 0.5, (
                    f"{name} {PARAM_NAMES[i]}: AD={ad_grad[i]:.4f} vs FD={fd_grad[i]:.4f}, "
                    f"rel_err={rel_err:.4f} — OT resampling should be differentiable"
                )

        # BPF_sys (non-differentiable): sign + magnitude within 100x
        if name == 'BPF_sys':
            for i in range(len(PARAM_NAMES)):
                if abs(fd_grad[i]) > 1.0:
                    assert np.sign(ad_grad[i]) == np.sign(fd_grad[i]), (
                        f"{name} {PARAM_NAMES[i]}: AD sign ({np.sign(ad_grad[i]):+.0f}) != "
                        f"FD sign ({np.sign(fd_grad[i]):+.0f})"
                    )
                    ratio = abs(fd_grad[i]) / max(abs(ad_grad[i]), 1e-10)
                    assert ratio < 100, (
                        f"{name} {PARAM_NAMES[i]}: FD/AD ratio={ratio:.1f}x — "
                        f"FD is unreliable through non-differentiable resampling"
                    )


# ============================================================================
# Test D: 1D Likelihood Surface Scans
# ============================================================================

class TestLikelihoodSurface1D:
    """Scan likelihood along sigma_range and sigma_bearing axes."""

    def test_scan_sigma_range(self, rb_observations_tf):
        """Vary sigma_range, fix sigma_bearing=true."""
        print(f"\n  Likelihood scan: sigma_range (sigma_bearing={TRUE_SIGMA_BEARING} fixed):")
        print(f"  {'sr':>6s}  {'ekf':>10s}  {'ukf':>10s}  {'bpf_ot':>10s}  {'ledh_ot':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        bpf_kwargs = _make_rb_bpf_kwargs('ot_entropy', False)
        ledh_kwargs = _make_rb_ledh_kwargs('ot_entropy', False)

        peaks = {'ekf': (None, -np.inf), 'ukf': (None, -np.inf),
                 'bpf_ot': (None, -np.inf), 'ledh_ot': (None, -np.inf)}

        for sr in SIGMA_SCAN_GRID:
            ekf_ll = ekf_log_likelihood(sr, TRUE_SIGMA_BEARING, rb_observations_tf)
            ukf_ll = ukf_log_likelihood(sr, TRUE_SIGMA_BEARING, rb_observations_tf)
            bpf_ll = pf_log_likelihood_rb(
                BootstrapPFHMC, bpf_kwargs, sr, TRUE_SIGMA_BEARING, rb_observations_tf)
            ledh_ll = pf_log_likelihood_rb(
                LEDHParticleFlowFilterHMC, ledh_kwargs, sr, TRUE_SIGMA_BEARING,
                rb_observations_tf)

            print(f"  {sr:6.2f}  {ekf_ll:10.4f}  {ukf_ll:10.4f}"
                  f"  {bpf_ll:10.4f}  {ledh_ll:10.4f}")

            for key, val in [('ekf', ekf_ll), ('ukf', ukf_ll),
                             ('bpf_ot', bpf_ll), ('ledh_ot', ledh_ll)]:
                if val > peaks[key][1]:
                    peaks[key] = (sr, val)

        print(f"\n  Peaks:")
        for key, (sr, ll) in peaks.items():
            print(f"    {key:>8s} peaks at sigma_range={sr}")

        # All should peak near the true value
        for key in ['ekf', 'ukf']:
            assert abs(peaks[key][0] - TRUE_SIGMA_RANGE) <= 0.1, (
                f"{key} peaks at {peaks[key][0]}, expected near {TRUE_SIGMA_RANGE}"
            )

    def test_scan_sigma_bearing(self, rb_observations_tf):
        """Vary sigma_bearing, fix sigma_range=true."""
        print(f"\n  Likelihood scan: sigma_bearing (sigma_range={TRUE_SIGMA_RANGE} fixed):")
        print(f"  {'sb':>6s}  {'ekf':>10s}  {'ukf':>10s}  {'bpf_ot':>10s}  {'ledh_ot':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        bpf_kwargs = _make_rb_bpf_kwargs('ot_entropy', False)
        ledh_kwargs = _make_rb_ledh_kwargs('ot_entropy', False)

        peaks = {'ekf': (None, -np.inf), 'ukf': (None, -np.inf),
                 'bpf_ot': (None, -np.inf), 'ledh_ot': (None, -np.inf)}

        for sb in SIGMA_SCAN_GRID:
            ekf_ll = ekf_log_likelihood(TRUE_SIGMA_RANGE, sb, rb_observations_tf)
            ukf_ll = ukf_log_likelihood(TRUE_SIGMA_RANGE, sb, rb_observations_tf)
            bpf_ll = pf_log_likelihood_rb(
                BootstrapPFHMC, bpf_kwargs, TRUE_SIGMA_RANGE, sb, rb_observations_tf)
            ledh_ll = pf_log_likelihood_rb(
                LEDHParticleFlowFilterHMC, ledh_kwargs, TRUE_SIGMA_RANGE, sb,
                rb_observations_tf)

            print(f"  {sb:6.2f}  {ekf_ll:10.4f}  {ukf_ll:10.4f}"
                  f"  {bpf_ll:10.4f}  {ledh_ll:10.4f}")

            for key, val in [('ekf', ekf_ll), ('ukf', ukf_ll),
                             ('bpf_ot', bpf_ll), ('ledh_ot', ledh_ll)]:
                if val > peaks[key][1]:
                    peaks[key] = (sb, val)

        print(f"\n  Peaks:")
        for key, (sb, ll) in peaks.items():
            print(f"    {key:>8s} peaks at sigma_bearing={sb}")

        for key in ['ekf', 'ukf']:
            assert abs(peaks[key][0] - TRUE_SIGMA_BEARING) <= 0.1, (
                f"{key} peaks at {peaks[key][0]}, expected near {TRUE_SIGMA_BEARING}"
            )


# ============================================================================
# Test E: Per-Parameter Gradient
# ============================================================================

class TestPerParameterGradient:
    """Analyze per-parameter gradient behavior."""

    def test_gradient_at_true_value(self, rb_observations_tf):
        """At true params (0.1, 0.1), gradient should be near zero."""
        print(f"\n  Gradient at true params ({TRUE_SIGMA_RANGE}, {TRUE_SIGMA_BEARING}):")
        print(f"  {'filter':>10s}  {'grad_sr':>10s}  {'grad_sb':>10s}  {'|grad|':>10s}  {'ll':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        for name, filt_class, kwargs_fn in FILTER_CONFIGS:
            nlp, grad, _, ll = compute_gradient_rb(
                filt_class, kwargs_fn(), rb_observations_tf,
                init_sr=TRUE_SIGMA_RANGE, init_sb=TRUE_SIGMA_BEARING)
            grad_norm = np.linalg.norm(grad)
            print(f"  {name:>10s}  {grad[0]:+10.4f}  {grad[1]:+10.4f}"
                  f"  {grad_norm:10.4f}  {ll:10.4f}")

    def test_gradient_ratio_range_vs_bearing(self, rb_observations_tf):
        """At init, is |grad_sr|/|grad_sb| consistent across filters?"""
        print(f"\n  Gradient ratio |grad_sr|/|grad_sb| at init ({INIT_SIGMA_RANGE}, {INIT_SIGMA_BEARING}):")
        print(f"  {'filter':>10s}  {'|grad_sr|':>10s}  {'|grad_sb|':>10s}  {'ratio':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        for name, filt_class, kwargs_fn in FILTER_CONFIGS:
            _, grad, _, _ = compute_gradient_rb(
                filt_class, kwargs_fn(), rb_observations_tf)
            abs_sr = abs(grad[0])
            abs_sb = abs(grad[1])
            ratio = abs_sr / max(abs_sb, 1e-10)
            print(f"  {name:>10s}  {abs_sr:10.4f}  {abs_sb:10.4f}  {ratio:10.4f}")


# ============================================================================
# Test F: Curvature Comparison
# ============================================================================

class TestCurvatureComparison:
    """Compare Hessian diagonal (curvature) across filters."""

    @pytest.mark.parametrize("name,filt_class,kwargs_fn", FAST_CONFIGS,
                             ids=[c[0] for c in FAST_CONFIGS])
    def test_hessian_diagonal(self, rb_observations_tf, name, filt_class, kwargs_fn):
        """FD Hessian diagonal at init (0.3, 0.3)."""
        kwargs = kwargs_fn()
        eps = 1e-3

        q0 = ParameterHandler(
            _make_rb_param_specs(), dtype=DTYPE
        ).unconstrained_init
        n_params = len(q0.numpy())

        # Evaluate NLP at center
        runner_c = _make_fresh_rb_runner(filt_class, kwargs, rb_observations_tf)
        nlp_center = float(runner_c._negative_log_posterior(q0).numpy())

        hess_diag = np.zeros(n_params)
        for i in range(n_params):
            e_i = np.zeros(n_params, dtype=np.float32)
            e_i[i] = eps
            e_i_tf = tf.constant(e_i)

            runner_p = _make_fresh_rb_runner(filt_class, kwargs, rb_observations_tf)
            nlp_plus = float(runner_p._negative_log_posterior(q0 + e_i_tf).numpy())

            runner_m = _make_fresh_rb_runner(filt_class, kwargs, rb_observations_tf)
            nlp_minus = float(runner_m._negative_log_posterior(q0 - e_i_tf).numpy())

            hess_diag[i] = (nlp_plus - 2 * nlp_center + nlp_minus) / (eps ** 2)

        print(f"\n  {name} Hessian diagonal at init ({INIT_SIGMA_RANGE}, {INIT_SIGMA_BEARING}):")
        print(f"    H_range   = {hess_diag[0]:12.4f}")
        print(f"    H_bearing = {hess_diag[1]:12.4f}")
        print(f"    condition = {max(abs(hess_diag)) / max(min(abs(hess_diag)), 1e-10):.2f}")

        # Just check finiteness
        assert np.all(np.isfinite(hess_diag)), f"{name} Hessian has NaN/Inf: {hess_diag}"


# ============================================================================
# Test G: LEDH Post-Fix Verification
# ============================================================================

class TestLEDHPostFix:
    """Verify LEDH works correctly after max_log_theta fix."""

    def test_ledh_likelihood_matches_bpf(self, rb_observations_tf):
        """LEDH LL should be close to BPF at true params."""
        bpf_kwargs = _make_rb_bpf_kwargs('ot_entropy', False)
        ledh_kwargs = _make_rb_ledh_kwargs('ot_entropy', False)

        bpf_ll = pf_log_likelihood_rb(
            BootstrapPFHMC, bpf_kwargs,
            TRUE_SIGMA_RANGE, TRUE_SIGMA_BEARING, rb_observations_tf)
        ledh_ll = pf_log_likelihood_rb(
            LEDHParticleFlowFilterHMC, ledh_kwargs,
            TRUE_SIGMA_RANGE, TRUE_SIGMA_BEARING, rb_observations_tf)
        ekf_ll = ekf_log_likelihood(
            TRUE_SIGMA_RANGE, TRUE_SIGMA_BEARING, rb_observations_tf)

        print(f"\n  Log-likelihood at true params ({TRUE_SIGMA_RANGE}, {TRUE_SIGMA_BEARING}):")
        print(f"    EKF:     {ekf_ll:10.4f}")
        print(f"    BPF_ot:  {bpf_ll:10.4f}  (diff from EKF: {bpf_ll - ekf_ll:+.4f})")
        print(f"    LEDH_ot: {ledh_ll:10.4f}  (diff from EKF: {ledh_ll - ekf_ll:+.4f})")

        # LEDH should be in the same ballpark as BPF/EKF
        assert abs(ledh_ll - ekf_ll) < 20.0, (
            f"LEDH LL ({ledh_ll:.4f}) too far from EKF ({ekf_ll:.4f})"
        )

    def test_ledh_gradient_sign_matches_ekf(self, rb_observations_tf):
        """LEDH gradient sign should match EKF at init."""
        ekf_kwargs = _ekf_kwargs()
        ledh_kwargs = _make_rb_ledh_kwargs('ot_entropy', False)

        _, ekf_grad, _, _ = compute_gradient_rb(
            ExtendedKalmanFilter, ekf_kwargs, rb_observations_tf)
        _, ledh_grad, _, _ = compute_gradient_rb(
            LEDHParticleFlowFilterHMC, ledh_kwargs, rb_observations_tf)

        print(f"\n  Gradient sign comparison at init ({INIT_SIGMA_RANGE}, {INIT_SIGMA_BEARING}):")
        for i, pname in enumerate(PARAM_NAMES):
            ekf_sign = np.sign(ekf_grad[i])
            ledh_sign = np.sign(ledh_grad[i])
            match = "OK" if ekf_sign == ledh_sign else "MISMATCH"
            print(f"    {pname:>14s}: EKF={ekf_grad[i]:+.4f} LEDH={ledh_grad[i]:+.4f}  {match}")

        for i, pname in enumerate(PARAM_NAMES):
            if abs(ekf_grad[i]) > 1.0:
                assert np.sign(ekf_grad[i]) == np.sign(ledh_grad[i]), (
                    f"LEDH {pname} gradient sign ({np.sign(ledh_grad[i]):+.0f}) != "
                    f"EKF sign ({np.sign(ekf_grad[i]):+.0f})"
                )
