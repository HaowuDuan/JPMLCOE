"""Unit tests for eigenvalue_barrier_penalty.

Verifies:
1. Penalty is zero when lambda_min(J) >> threshold (near-identity init).
2. Penalty is positive and matches the analytic hinge value when J has a
   small eigenvalue (forced via a custom mock map).
3. Gradient flows back through the barrier.

Run: pytest code/neural_operator/tests/test_eigenvalue_barrier.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from conditional_map import ConditionalConvexPotentialMap
from losses import eigenvalue_barrier_penalty
from _result_utils import save_result, reset_results


DTYPE = tf.float64


def setup_module():
    reset_results(__file__)


class MockTrunk:
    """Minimal stand-in exposing only `jacobian(x, c)` returning a fixed J."""

    def __init__(self, J):
        self._J = J

    def jacobian(self, x, c):
        return self._J


def test_zero_penalty_at_init():
    """Near-identity init: lambda_min ~ 1, threshold 0.1, penalty must be 0."""
    model = ConditionalConvexPotentialMap(
        in_dim=2, embed_dim=16, num_modules=4, context_dim=32,
        alpha_init_bias=-5.0,
    )
    x = tf.random.normal((10, 2), dtype=DTYPE)
    c = tf.random.normal((32,), dtype=DTYPE)
    _ = model(x, c)  # build
    loss, min_eig = eigenvalue_barrier_penalty(model, x, c, threshold=0.1)

    case = {
        "case_name": "zero_penalty_at_init",
        "loss": float(loss),
        "min_eig": float(min_eig),
        "expected_loss": 0.0,
        "passed": float(loss) == 0.0 and float(min_eig) > 0.5,
    }
    save_result(__file__, case)

    assert float(loss) == 0.0, f"expected 0, got {float(loss)}"
    assert float(min_eig) > 0.5, f"min_eig too small at init: {float(min_eig)}"
    print(f"  test_zero_penalty_at_init PASS: min_eig={float(min_eig):.3f}, loss={float(loss):.2e}")


def test_penalty_fires_below_threshold():
    """Force a J with eigenvalue 0.05; threshold 0.1 -> penalty = (0.1-0.05)^2 = 2.5e-3."""
    # Diagonal J with eigenvalues [0.05, 1.0]
    J = tf.constant(
        [[[0.05, 0.0], [0.0, 1.0]],
         [[0.05, 0.0], [0.0, 1.0]]],
        dtype=DTYPE,
    )  # (Q=2, d=2, d=2)
    mock = MockTrunk(J)
    x = tf.zeros((2, 2), dtype=DTYPE)
    c = tf.zeros((1,), dtype=DTYPE)
    loss, min_eig = eigenvalue_barrier_penalty(mock, x, c, threshold=0.1)
    expected = (0.1 - 0.05) ** 2  # hinge^2 = 0.0025

    case = {
        "case_name": "penalty_fires_below_threshold",
        "loss": float(loss),
        "min_eig": float(min_eig),
        "expected_loss": expected,
        "passed": abs(float(loss) - expected) < 1e-12 and abs(float(min_eig) - 0.05) < 1e-12,
    }
    save_result(__file__, case)

    assert abs(float(loss) - expected) < 1e-12, (
        f"expected {expected}, got {float(loss)}"
    )
    assert abs(float(min_eig) - 0.05) < 1e-12
    print(f"  test_penalty_fires_below_threshold PASS: loss={float(loss):.2e} (expected {expected:.2e})")


def test_penalty_gradient_flows():
    """Verify gradient flows back into model parameters via the barrier."""
    model = ConditionalConvexPotentialMap(
        in_dim=2, embed_dim=16, num_modules=4, context_dim=32,
        alpha_init_bias=2.0,  # alpha large -> J differs from I, easier to perturb
    )
    x = tf.random.normal((10, 2), dtype=DTYPE)
    c = tf.random.normal((32,), dtype=DTYPE)
    _ = model(x, c)  # build

    with tf.GradientTape() as tape:
        # Use an abnormally large threshold so the hinge actually activates
        loss, min_eig = eigenvalue_barrier_penalty(model, x, c, threshold=10.0)
    grads = tape.gradient(loss, model.trainable_variables)
    n_finite = sum(
        1 for g in grads if g is not None and tf.reduce_all(tf.math.is_finite(g)).numpy()
    )
    n_total = sum(1 for g in grads if g is not None)

    case = {
        "case_name": "penalty_gradient_flows",
        "loss": float(loss),
        "min_eig": float(min_eig),
        "n_finite_grads": n_finite,
        "n_total_grads": n_total,
        "passed": n_finite == n_total and n_total > 0,
    }
    save_result(__file__, case)

    assert n_finite == n_total and n_total > 0, (
        f"finite grads: {n_finite}/{n_total}"
    )
    print(
        f"  test_penalty_gradient_flows PASS: loss={float(loss):.4f}, "
        f"min_eig={float(min_eig):.3f}, finite grads={n_finite}/{n_total}"
    )
