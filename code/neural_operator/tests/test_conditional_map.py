"""Unit tests for ConditionalConvexPotentialMap.

Verify:
1. Forward shape correct
2. Analytic Jacobian matches tape.batch_jacobian (with c held fixed)
3. Jacobian is symmetric and PSD
4. At init (negative alpha bias), T(x; c) ~ x (residual near identity)
5. Different contexts give different transport maps
6. Differentiable through context (gradient flows from T back to c)

Run: pytest code/neural_operator/tests/test_conditional_map.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from conditional_map import ConditionalConvexPotentialMap


DTYPE = tf.float64


def test_forward_shape():
    model = ConditionalConvexPotentialMap(
        in_dim=2, embed_dim=32, num_modules=8, context_dim=64
    )
    x = tf.random.normal((5, 2), dtype=DTYPE)
    c = tf.random.normal((64,), dtype=DTYPE)
    y = model(x, c)
    assert y.shape == (5, 2), f"Expected (5, 2), got {y.shape}"
    print(f"  test_forward_shape PASS: y.shape = {y.shape}")


def test_analytic_vs_autodiff_jacobian():
    """Analytic Jacobian matches autodiff with c held fixed."""
    for d in [1, 2, 3]:
        model = ConditionalConvexPotentialMap(
            in_dim=d, embed_dim=16, num_modules=4, context_dim=32
        )
        x = tf.random.normal((4, d), dtype=DTYPE)
        c = tf.random.normal((32,), dtype=DTYPE)
        _ = model(x, c)  # build

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, c)
        J_auto = tape.batch_jacobian(y, x)
        J_anal = model.jacobian(x, c)

        rel_diff = float(
            tf.reduce_max(tf.abs(J_auto - J_anal))
            / tf.reduce_max(tf.abs(J_auto))
        )
        assert rel_diff < 1e-6, f"d={d}: rel_diff = {rel_diff:.2e}"
        print(f"  test_analytic_vs_autodiff_jacobian d={d} PASS: rel_diff = {rel_diff:.2e}")


def test_jacobian_symmetric():
    model = ConditionalConvexPotentialMap(
        in_dim=3, embed_dim=16, num_modules=8, context_dim=32
    )
    x = tf.random.normal((10, 3), dtype=DTYPE)
    c = tf.random.normal((32,), dtype=DTYPE)
    _ = model(x, c)
    J = model.jacobian(x, c)
    asym = float(tf.reduce_max(tf.abs(J - tf.linalg.matrix_transpose(J))))
    assert asym < 1e-10, f"Jacobian not symmetric: max asym = {asym}"
    print(f"  test_jacobian_symmetric PASS: max_asym = {asym:.2e}")


def test_jacobian_psd():
    model = ConditionalConvexPotentialMap(
        in_dim=3, embed_dim=16, num_modules=8, context_dim=32
    )
    x = tf.random.normal((10, 3), dtype=DTYPE)
    c = tf.random.normal((32,), dtype=DTYPE)
    _ = model(x, c)
    J = model.jacobian(x, c)
    eigvals = tf.linalg.eigvalsh(J)
    min_eig = float(tf.reduce_min(eigvals))
    # With residual + alpha_init_bias=-5, J should be near identity, min_eig ~ 1
    assert min_eig > 0.5, f"min_eigval = {min_eig:.4f}"
    print(f"  test_jacobian_psd PASS: min_eigval = {min_eig:.4f}")


def test_residual_near_identity_at_init():
    """With alpha_init_bias=-5, softplus(alpha) ~ 0 -> T(x; c) ~ x at init."""
    model = ConditionalConvexPotentialMap(
        in_dim=2, embed_dim=32, num_modules=8, context_dim=64,
        alpha_init_bias=-5.0,
    )
    x = tf.random.normal((10, 2), dtype=DTYPE) * 0.5
    c = tf.zeros((64,), dtype=DTYPE)
    y = model(x, c)
    diff = float(tf.reduce_mean(tf.abs(y - x)))
    print(f"  test_residual_near_identity_at_init: mean |T(x;c) - x| = {diff:.6f}")
    assert diff < 0.1, f"Expected near identity, got mean diff = {diff}"


def test_context_changes_map():
    """Different contexts should give different T(x)."""
    model = ConditionalConvexPotentialMap(
        in_dim=2, embed_dim=16, num_modules=4, context_dim=32,
        alpha_init_bias=0.0,  # let alpha be ~0.7 so module contributes
    )
    x = tf.random.normal((5, 2), dtype=DTYPE)
    c1 = tf.random.normal((32,), dtype=DTYPE)
    c2 = tf.random.normal((32,), dtype=DTYPE) * 5.0  # very different
    y1 = model(x, c1)
    y2 = model(x, c2)
    diff = float(tf.reduce_max(tf.abs(y1 - y2)))
    assert diff > 1e-3, f"Map did not respond to context change: diff = {diff}"
    print(f"  test_context_changes_map PASS: max diff = {diff:.4f}")


def test_differentiable_through_context():
    """Gradient should flow from T(x; c) back to c."""
    model = ConditionalConvexPotentialMap(
        in_dim=2, embed_dim=16, num_modules=4, context_dim=32,
        alpha_init_bias=0.0,
    )
    x = tf.random.normal((5, 2), dtype=DTYPE)
    c = tf.Variable(tf.random.normal((32,), dtype=DTYPE))

    with tf.GradientTape() as tape:
        y = model(x, c)
        loss = tf.reduce_sum(y ** 2)
    grad = tape.gradient(loss, c)
    assert grad is not None
    assert tf.reduce_all(tf.math.is_finite(grad)).numpy()
    print(f"  test_differentiable_through_context PASS: grad norm = {float(tf.norm(grad)):.4f}")


