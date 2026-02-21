"""
Stage 1: BPF baseline diagnostic tests.

Validates BPF log-likelihood and gradient across all 3 resampling methods.
BPF is simple (no flow, no Jacobian) — if it fails, the bug is in
resampling or the HMC runner, not the filter.

Ground truth: Kalman filter gives the exact log-likelihood for linear Gaussian.

Run:
  pytest code/tests/test_hmc_bpf.py -v -s 2>&1 | tee bpf_diagnosis.txt
"""

import numpy as np
import tensorflow as tf

from conftest_hmc_diag import (
    DTYPE, N_PARTICLES, SEED, T, TRUE_OBS_NOISE_STD, INIT_OBS_NOISE_STD,
    SIGMA_GRID,
    lg_model_and_data, observations_tf,          # fixtures
    _make_bpf_kwargs, kf_log_likelihood, pf_log_likelihood,
    compute_gradient, finite_difference_gradient,
    compute_prior_gradient, compute_likelihood_gradient,
)
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC


# ============================================================================
# Tests 1.1–1.2: BPF log-likelihood vs KF
# ============================================================================

class TestBPFLogLikelihood:
    """Stage 1.1-1.2: BPF log-likelihood vs KF ground truth."""

    def test_bpf_log_likelihood_vs_kf(self, lg_model_and_data, observations_tf):
        """Test 1.1: BPF log-lik matches KF at true parameter."""
        _, observations, _ = lg_model_and_data

        kf_ll = kf_log_likelihood(TRUE_OBS_NOISE_STD, observations)
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        bpf_kwargs['n_particles'] = 1000
        bpf_ll = pf_log_likelihood(BootstrapPFHMC, bpf_kwargs, TRUE_OBS_NOISE_STD, observations_tf)

        rel_err = abs(kf_ll - bpf_ll) / abs(kf_ll)
        print(f"\n  KF  log-lik: {kf_ll:.4f}")
        print(f"  BPF log-lik: {bpf_ll:.4f}")
        print(f"  Relative error: {rel_err:.4f}")
        assert rel_err < 0.10, f"BPF log-lik too far from KF: {rel_err:.4f}"

    def test_bpf_log_likelihood_surface(self, lg_model_and_data, observations_tf):
        """Test 1.2: BPF log-lik surface peaks at same place as KF."""
        _, observations, _ = lg_model_and_data

        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        bpf_kwargs['n_particles'] = 1000

        print(f"\n  {'sigma':>6s}  {'kf_ll':>10s}  {'bpf_ll':>10s}  {'diff':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

        kf_lls, bpf_lls = [], []
        for sigma in SIGMA_GRID:
            kf_ll = kf_log_likelihood(sigma, observations)
            bpf_ll = pf_log_likelihood(BootstrapPFHMC, bpf_kwargs, sigma, observations_tf)
            kf_lls.append(kf_ll)
            bpf_lls.append(bpf_ll)
            print(f"  {sigma:6.2f}  {kf_ll:10.4f}  {bpf_ll:10.4f}  {bpf_ll - kf_ll:+10.4f}")

        kf_best = SIGMA_GRID[np.argmax(kf_lls)]
        bpf_best = SIGMA_GRID[np.argmax(bpf_lls)]
        print(f"\n  KF  peaks at sigma={kf_best}")
        print(f"  BPF peaks at sigma={bpf_best}")
        assert kf_best == bpf_best, f"BPF peaks at {bpf_best}, KF at {kf_best}"


# ============================================================================
# Tests 1.3–1.6: BPF gradient diagnostics
# ============================================================================

class TestBPFGradient:
    """Stage 1.3-1.6: BPF gradient diagnostics."""

    def test_bpf_gradient_direction(self, observations_tf):
        """Test 1.3: BPF gradient points toward true value from both sides."""
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)

        # Above true: NLL grad should be positive (push sigma down)
        nlp_hi, grad_hi, sigma_hi, ll_hi = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf, init_obs_noise_std=2.0)

        # Below true: NLL grad should be negative (push sigma up)
        nlp_lo, grad_lo, sigma_lo, ll_lo = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf, init_obs_noise_std=0.5)

        print(f"\n  sigma=2.0: NLL grad={grad_hi[0]:+.4f}  (expect positive)")
        print(f"  sigma=0.5: NLL grad={grad_lo[0]:+.4f}  (expect negative)")

        assert grad_hi[0] > 0, f"Grad at sigma=2.0 should be positive, got {grad_hi[0]}"
        assert grad_lo[0] < 0, f"Grad at sigma=0.5 should be negative, got {grad_lo[0]}"

    def test_bpf_autodiff_vs_finite_difference(self, observations_tf):
        """Test 1.4: Autodiff gradient matches finite difference.

        Only asserts for ot_entropy (fully differentiable transport matrix).
        Soft resampling uses discrete index selection (searchsorted + gather)
        so FD captures discrete jumps that autodiff cannot see — mismatch
        is expected and printed for reference but not asserted.
        Systematic with stop_gradient=True is also print-only.
        """
        configs = [
            ('systematic', True),
            ('soft', False),
            ('ot_entropy', False),
        ]

        print(f"\n  {'method':>12s}  {'autodiff':>10s}  {'FD':>10s}  {'rel_err':>10s}  {'stop_grad':>10s}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        for method, stop_grad in configs:
            bpf_kwargs = _make_bpf_kwargs(method, stop_grad)
            _, ad_grad, _, _ = compute_gradient(
                BootstrapPFHMC, bpf_kwargs, observations_tf)
            fd_grad = finite_difference_gradient(
                BootstrapPFHMC, bpf_kwargs, observations_tf)

            ad_val = ad_grad[0]
            rel_err = abs(ad_val - fd_grad) / max(abs(fd_grad), 1e-10)
            print(f"  {method:>12s}  {ad_val:+10.4f}  {fd_grad:+10.4f}  {rel_err:10.4f}  {stop_grad!s:>10s}")

            # Only assert on ot_entropy — fully differentiable (no discrete indices)
            if method == 'ot_entropy':
                assert rel_err < 0.30, f"BPF {method}: autodiff/FD mismatch, rel_err={rel_err:.4f}"

    def test_bpf_gradient_by_resampling_method(self, observations_tf):
        """Test 1.5: Compare BPF gradient across resampling methods."""
        configs = [
            ('systematic', True),
            ('soft', False),
            ('ot_entropy', False),
        ]

        print(f"\n  {'method':>12s}  {'|grad|':>10s}  {'grad':>10s}  {'log_lik':>10s}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

        grads = []
        for method, stop_grad in configs:
            bpf_kwargs = _make_bpf_kwargs(method, stop_grad)
            nlp, grad, sigma, ll = compute_gradient(
                BootstrapPFHMC, bpf_kwargs, observations_tf)
            grads.append((method, grad[0]))
            print(f"  {method:>12s}  {abs(grad[0]):10.4f}  {grad[0]:+10.4f}  {ll:10.4f}")

        # All should point same direction (positive at sigma=2.0)
        signs = [np.sign(g) for _, g in grads]
        assert all(s == signs[0] for s in signs), \
            f"Gradient directions disagree: {[(m, g) for m, g in grads]}"

        # Check for gradient explosion (no method should be >100x another)
        magnitudes = [abs(g) for _, g in grads]
        ratio = max(magnitudes) / max(min(magnitudes), 1e-10)
        print(f"\n  Max/min magnitude ratio: {ratio:.1f}")
        assert ratio < 100, f"Gradient magnitude ratio {ratio:.1f} > 100 — explosion detected"

    def test_bpf_prior_gradient_autodiff_vs_fd(self):
        """Test 1.7: Prior-only gradient — Softplus + LogNormal, no filter involved.

        If this fails, the bug is in ParameterHandler (bijector or prior).
        If this passes but test 1.8 fails, the bug is in the filter backward pass.
        """
        neg_lp, ad_grad, fd_grad = compute_prior_gradient()
        rel_err = abs(ad_grad - fd_grad) / max(abs(fd_grad), 1e-10)

        print(f"\n  Prior gradient decomposition (sigma_init={INIT_OBS_NOISE_STD}):")
        print(f"    -log_prior = {neg_lp:.6f}")
        print(f"    autodiff   = {ad_grad:+.6f}")
        print(f"    FD         = {fd_grad:+.6f}")
        print(f"    rel_err    = {rel_err:.6f}")

        assert rel_err < 0.01, f"Prior gradient: autodiff/FD mismatch, rel_err={rel_err:.4f}"

    def test_bpf_likelihood_gradient_autodiff_vs_fd(self, observations_tf):
        """Test 1.8: Likelihood-only gradient — filter backward pass, no prior.

        Uses ot_entropy resampling (fully differentiable transport matrix)
        so autodiff and FD compute the same smooth function.
        Soft resampling has discrete index selection (searchsorted + gather)
        causing FD/autodiff mismatch — not suitable for this test.
        """
        bpf_kwargs = _make_bpf_kwargs('ot_entropy', stop_gradient=False)
        neg_ll, ad_grad, fd_grad = compute_likelihood_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf)
        rel_err = abs(ad_grad - fd_grad) / max(abs(fd_grad), 1e-10)

        print(f"\n  Likelihood gradient decomposition (sigma_init={INIT_OBS_NOISE_STD}):")
        print(f"    -log_lik = {neg_ll:.4f}")
        print(f"    autodiff = {ad_grad:+.6f}")
        print(f"    FD       = {fd_grad:+.6f}")
        print(f"    rel_err  = {rel_err:.6f}")

        assert rel_err < 0.30, f"Likelihood gradient: autodiff/FD mismatch, rel_err={rel_err:.4f}"

    def test_bpf_gradient_vs_timesteps(self, lg_model_and_data):
        """Test 1.6: Does BPF gradient grow linearly or exponentially with T?"""
        _, observations, _ = lg_model_and_data

        configs = [
            ('systematic', True),
            ('soft', False),
        ]

        print(f"\n  {'T':>4s}  {'method':>12s}  {'|grad|':>10s}  {'|grad|/T':>10s}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}")

        for method, stop_grad in configs:
            bpf_kwargs = _make_bpf_kwargs(method, stop_grad)
            grad_per_T = []

            for t in [5, 10, 20, 50]:
                obs_tf = tf.constant(observations[:t], dtype=DTYPE)
                _, grad, _, _ = compute_gradient(
                    BootstrapPFHMC, bpf_kwargs, obs_tf)
                mag = abs(grad[0])
                grad_per_T.append(mag / t)
                print(f"  {t:4d}  {method:>12s}  {mag:10.4f}  {mag/t:10.4f}")

            # |grad|/T should not grow by more than 10x
            ratio = max(grad_per_T) / max(min(grad_per_T), 1e-10)
            print(f"  {method}: max/min |grad|/T ratio = {ratio:.1f}")
            assert ratio < 10, f"BPF {method}: gradient grows super-linearly, ratio={ratio:.1f}"
