"""Test: Compare log_abs_det gradient implementations against finite differences.

Three implementations:
1. safe_log_abs_det (eager) — TF native gradient
2. _graph_safe_log_abs_det_impl (custom gradient, pinv backward)
3. _graph_safe_log_abs_det_fast_impl (custom gradient, fast inv + extra 1e-6 jitter)

The _fast variant uses inv(M + 1e-8*I + 1e-6*I) in backward but slogdet(M + 1e-8*I)
in forward. If this mismatch causes gradient bias, it would accumulate over the
29 flow steps in LEDH and could explain the 2.5x discrepancy.

Note: We call the inner @tf.custom_gradient functions directly, bypassing the
@tf.function(jit_compile=True) wrapper which requires XLA GPU support.
The math is identical.

Run: python -m pytest tests/hmc/test_log_abs_det_gradient.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.utils.linalg import (
    safe_log_abs_det,
    _graph_safe_log_abs_det_impl,
    _graph_safe_log_abs_det_fast_impl,
)

DTYPE = tf.float64
JITTER = 1e-8
H = 1e-5  # finite difference step


def _add_jitter(M):
    n = tf.shape(M)[-1]
    return M + JITTER * tf.eye(n, dtype=M.dtype)


def _fd_gradient(fn, M, h=H):
    """Element-wise finite difference gradient of scalar fn(M) w.r.t. M."""
    M_np = M.numpy()
    grad = np.zeros_like(M_np)
    for idx in np.ndindex(M_np.shape):
        M_plus = M_np.copy()
        M_minus = M_np.copy()
        M_plus[idx] += h
        M_minus[idx] -= h
        f_plus = fn(tf.constant(M_plus, dtype=DTYPE)).numpy()
        f_minus = fn(tf.constant(M_minus, dtype=DTYPE)).numpy()
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def _ad_gradient(fn, M):
    """Autodiff gradient of scalar fn(M) w.r.t. M."""
    with tf.GradientTape() as tape:
        tape.watch(M)
        val = fn(M)
    return tape.gradient(val, M).numpy(), val.numpy()


# Wrappers that add jitter (matching what the @tf.function wrappers do)
def _pinv_logdet(M):
    return _graph_safe_log_abs_det_impl(_add_jitter(M))

def _fast_logdet(M):
    return _graph_safe_log_abs_det_fast_impl(_add_jitter(M))


class TestSingleMatrix:
    """Test on a single 2x2 matrix (like LEDH flow step M = I + d_lambda * A)."""

    @pytest.fixture
    def M_2x2(self):
        """Typical M from LEDH flow: I + d_lambda * A, close to identity."""
        return tf.constant([[0.985, -0.015], [0.01, 0.97]], dtype=DTYPE)

    def test_safe_log_abs_det(self, M_2x2):
        ad, val = _ad_gradient(safe_log_abs_det, M_2x2)
        fd = _fd_gradient(safe_log_abs_det, M_2x2)
        rel_err = np.max(np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-10))
        print(f"\n  safe_log_abs_det:")
        print(f"    value: {val:.8f}")
        print(f"    ad grad:\n{ad}")
        print(f"    fd grad:\n{fd}")
        print(f"    max rel err: {rel_err:.6f}")
        assert rel_err < 0.01

    def test_pinv_log_abs_det(self, M_2x2):
        ad, val = _ad_gradient(_pinv_logdet, M_2x2)
        fd = _fd_gradient(_pinv_logdet, M_2x2)
        rel_err = np.max(np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-10))
        print(f"\n  graph_safe_log_abs_det (pinv):")
        print(f"    value: {val:.8f}")
        print(f"    ad grad:\n{ad}")
        print(f"    fd grad:\n{fd}")
        print(f"    max rel err: {rel_err:.6f}")
        assert rel_err < 0.01

    def test_fast_log_abs_det(self, M_2x2):
        ad, val = _ad_gradient(_fast_logdet, M_2x2)
        fd = _fd_gradient(_fast_logdet, M_2x2)
        rel_err = np.max(np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-10))
        print(f"\n  graph_safe_log_abs_det_fast (inv + 1e-6 jitter):")
        print(f"    value: {val:.8f}")
        print(f"    ad grad:\n{ad}")
        print(f"    fd grad:\n{fd}")
        print(f"    max rel err: {rel_err:.6f}")
        assert rel_err < 0.01


class TestAccumulatedJacobian:
    """Test accumulated log-det over multiple flow steps (like LEDH does).

    If per-step bias is small but accumulates over 15-29 steps,
    the total gradient could be significantly off.
    """

    @pytest.fixture
    def flow_matrices(self):
        """Generate N_STEPS matrices like I + d_lambda * A."""
        rng = np.random.default_rng(42)
        n_steps = 15
        d_lambda = 0.034  # ~1/29
        matrices = []
        for _ in range(n_steps):
            A = rng.normal(0, 0.5, (2, 2))
            M = np.eye(2) + d_lambda * A
            matrices.append(M)
        return [tf.constant(m, dtype=DTYPE) for m in matrices]

    def _accumulated_logdet(self, fn, matrices, scale):
        """Sum of fn(I + scale*(M-I)) over all matrices."""
        total = tf.constant(0.0, dtype=DTYPE)
        I = tf.eye(2, dtype=DTYPE)
        for M in matrices:
            M_scaled = I + scale * (M - I)
            total = total + fn(M_scaled)
        return total

    def _test_accumulated(self, fn, fn_name, matrices):
        scale = tf.constant(1.0, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(scale)
            val = self._accumulated_logdet(fn, matrices, scale)
        ad = tape.gradient(val, scale)
        ad_val = ad.numpy() if ad is not None else None

        # Numerical
        h = 1e-5
        val_p = self._accumulated_logdet(fn, matrices, tf.constant(1.0 + h, dtype=DTYPE))
        val_m = self._accumulated_logdet(fn, matrices, tf.constant(1.0 - h, dtype=DTYPE))
        fd = (val_p.numpy() - val_m.numpy()) / (2 * h)

        rel_err = abs(ad_val - fd) / max(abs(fd), 1e-10) if ad_val is not None else float('inf')
        print(f"\n  Accumulated {fn_name} (15 steps):")
        print(f"    value: {val.numpy():.8f}")
        print(f"    autodiff: {ad_val}")
        print(f"    finite diff: {fd:.8f}")
        print(f"    rel err: {rel_err:.6f}")
        return ad_val, fd, rel_err

    def test_accumulated_safe(self, flow_matrices):
        ad, fd, rel_err = self._test_accumulated(
            safe_log_abs_det, "safe_log_abs_det", flow_matrices)
        assert rel_err < 0.01

    def test_accumulated_pinv(self, flow_matrices):
        ad, fd, rel_err = self._test_accumulated(
            _pinv_logdet, "pinv_log_abs_det", flow_matrices)
        assert rel_err < 0.01

    def test_accumulated_fast(self, flow_matrices):
        ad, fd, rel_err = self._test_accumulated(
            _fast_logdet, "fast_log_abs_det", flow_matrices)
        assert rel_err < 0.01

    def test_compare_all_three(self, flow_matrices):
        """Direct comparison of all three gradient implementations."""
        results = {}
        for fn, name in [
            (safe_log_abs_det, "safe"),
            (_pinv_logdet, "pinv"),
            (_fast_logdet, "fast"),
        ]:
            ad, fd, rel_err = self._test_accumulated(fn, name, flow_matrices)
            results[name] = {'ad': ad, 'fd': fd, 'rel_err': rel_err}

        print(f"\n  === Comparison ===")
        print(f"  {'variant':>8s}  {'autodiff':>12s}  {'FD':>12s}  {'rel_err':>10s}")
        for name, r in results.items():
            ad_str = f"{r['ad']:12.6f}" if r['ad'] is not None else "        None"
            print(f"  {name:>8s}  {ad_str}  {r['fd']:12.6f}  {r['rel_err']:10.6f}")

        # Check if fast diverges from safe
        if results['safe']['ad'] is not None and results['fast']['ad'] is not None:
            ratio = results['fast']['ad'] / results['safe']['ad']
            print(f"\n  fast/safe autodiff ratio: {ratio:.6f} (want ~1.0)")
