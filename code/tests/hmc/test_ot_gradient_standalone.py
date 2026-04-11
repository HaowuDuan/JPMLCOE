"""
Standalone gradient validation for OT resampling.

Two tests, one per backward method:
  test_implicit  -- ot_entropy.py        (implicit Sinkhorn backward)
  test_approx    -- ot_entropy_approx.py (extrapolation backward)

Each test runs the same loop: for each of three modes
(weights_only, particles_only, full) compute the autodiff gradient
and the numerical (central FD) gradient on a single OT resampling
step, print the row, and assert relerr < 1e-6.

Modes correspond to which terms of the OT derivative chain are active:
  d x_res / dtheta = (dT/dx)(dx/dtheta) x   <- Term 1
                   + T (dx/dtheta)           <- Term 2
                   + (dT/dw)(dw/dtheta) x    <- Term 3

  weights_only    -> Term 3 only
  particles_only  -> Terms 1+2
  full            -> all three terms

Observable: mean(||x_res||^2). Particle perturbation: anisotropic scaling
[1+theta, 1] (survives the OT module's internal centering+std normalization).

Run:
  cd code && python -m pytest tests/hmc/test_ot_gradient_standalone.py -v -s
"""

import sys
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.resampling.ot_entropy import ot_entropy_resample as ot_implicit
from src.resampling.ot_entropy_approx import ot_entropy_resample as ot_approx

from tests.hmc._gradient_test_utils import save_result, reset_results


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DTYPE = tf.float64
N = 200
STATE_DIM = 2
EPSILON = 0.5
THETA_0 = 0.3
CONV_THRESHOLD = 1e-10
MAX_ITER = 5000
FD_H = 1e-4
TOL = 1e-6
RESAMPLE_SEED = tf.constant([0, 0], dtype=tf.int32)

MODES = ["weights_only", "particles_only", "full"]


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
BASE_PARTICLES = tf.random.stateless_normal((N, STATE_DIM), seed=[42, 0], dtype=DTYPE)
BASE_PARTICLES -= tf.reduce_mean(BASE_PARTICLES, axis=0, keepdims=True)
BASE_LOG_W = 0.5 * tf.random.stateless_normal((N,), seed=[43, 0], dtype=DTYPE)
BASE_W = tf.nn.softmax(BASE_LOG_W)
WEIGHT_DIRECTION = tf.linspace(tf.constant(-1.0, dtype=DTYPE),
                                tf.constant(1.0, dtype=DTYPE), N)


def inputs_for(theta, mode):
    if mode == "weights_only":
        return BASE_PARTICLES, tf.nn.softmax(BASE_LOG_W + theta * WEIGHT_DIRECTION)
    if mode == "particles_only":
        scale = tf.stack([1.0 + theta, tf.constant(1.0, dtype=DTYPE)])
        return BASE_PARTICLES * scale, BASE_W
    scale = tf.stack([1.0 + theta, tf.constant(1.0, dtype=DTYPE)])
    return BASE_PARTICLES * scale, tf.nn.softmax(BASE_LOG_W + theta * WEIGHT_DIRECTION)


def observable(particles):
    return tf.reduce_mean(tf.reduce_sum(particles ** 2, axis=-1))


def resample(particles, weights, method):
    fn = ot_implicit if method == "implicit" else ot_approx
    return fn(particles=particles, weights=weights, epsilon=EPSILON, seed=RESAMPLE_SEED,
              max_iter=MAX_ITER, convergence_threshold=CONV_THRESHOLD)


# OT custom gradient calls tf.gradients() internally -> needs graph mode.
@tf.function
def autodiff(theta, mode, method):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        p, w = inputs_for(theta, mode)
        out = observable(resample(p, w, method).particles)
    return tape.gradient(out, theta)


@tf.function
def forward(theta, mode, method):
    p, w = inputs_for(theta, mode)
    return observable(resample(p, w, method).particles)


def numerical(theta_val, mode, method):
    fp = float(forward(tf.constant(theta_val + FD_H, dtype=DTYPE), mode, method).numpy())
    fm = float(forward(tf.constant(theta_val - FD_H, dtype=DTYPE), mode, method).numpy())
    return (fp - fm) / (2.0 * FD_H)


def run_method(method):
    """Loop over the three modes, print a table, return rows."""
    theta = tf.constant(THETA_0, dtype=DTYPE)
    rows = []
    print(f"\n=== method: {method} ===")
    print(f"{'mode':<16}{'numerical':>15}{'autodiff':>15}{'rel_err':>14}")
    print("-" * 60)
    for mode in MODES:
        g_ad = float(autodiff(theta, mode, method).numpy())
        g_num = numerical(THETA_0, mode, method)
        e = abs(g_ad - g_num) / max(abs(g_num), 1e-12)
        rows.append((mode, g_num, g_ad, e))
        print(f"{mode:<16}{g_num:>15.6e}{g_ad:>15.6e}{e:>14.3e}")
    return rows


def _save(method, rows, passed):
    save_result(__file__, {
        "method": method,
        "n_particles": N,
        "state_dim": STATE_DIM,
        "epsilon": EPSILON,
        "theta_0": THETA_0,
        "fd_h": FD_H,
        "tolerance": TOL,
        "passed": passed,
        "modes": [
            {
                "mode": mode,
                "numerical": g_num,
                "autodiff": g_ad,
                "rel_err": e,
            }
            for mode, g_num, g_ad, e in rows
        ],
    })


# -----------------------------------------------------------------------------
# Two tests, one per method. Results persisted to tests/hmc/results/.
# -----------------------------------------------------------------------------
class TestOTGradient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_results(__file__)

    def _check(self, method):
        rows = run_method(method)
        passed = True
        try:
            for mode, g_num, _, e in rows:
                self.assertGreater(abs(g_num), 1e-6, f"{mode}: numerical grad ~ 0")
                self.assertLess(e, TOL, f"{mode}: {method} relerr {e:.3e} > {TOL}")
        except AssertionError:
            passed = False
            raise
        finally:
            _save(method, rows, passed)

    def test_implicit(self):
        self._check("implicit")

    def test_approx(self):
        self._check("approx")


if __name__ == "__main__":
    unittest.main(verbosity=2)
