"""Unit tests for eigenvalue_barrier_penalty.

Verifies:
1. Penalty is zero when lambda_min(J) >> threshold (near-identity init).
2. Penalty is positive and matches the analytic hinge value when J has a
   small eigenvalue (forced via a custom mock map).
3. Gradient flows back through the barrier.

Run: python code/neural_operator/tests/test_eigenvalue_barrier.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import tensorflow as tf

from conditional_map import ConditionalConvexPotentialMap
from losses import eigenvalue_barrier_penalty


DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')


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
    assert n_finite == n_total and n_total > 0, (
        f"finite grads: {n_finite}/{n_total}"
    )
    print(
        f"  test_penalty_gradient_flows PASS: loss={float(loss):.4f}, "
        f"min_eig={float(min_eig):.3f}, finite grads={n_finite}/{n_total}"
    )


if __name__ == '__main__':
    print("\n=== eigenvalue_barrier_penalty tests ===\n")
    tf.random.set_seed(42)

    test_zero_penalty_at_init()
    test_penalty_fires_below_threshold()
    test_penalty_gradient_flows()

    print("\n=== All eigenvalue barrier tests passed ===\n")
