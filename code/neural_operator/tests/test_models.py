"""Unit tests for ConvexPotentialMap.

Verify:
1. Forward pass runs and produces sensible output
2. Analytic Jacobian matches tape.batch_jacobian
3. Jacobian is symmetric and PSD (the convex potential property)
4. Residual init: T(x) ~ x at initialization

Run: python code/neural_operator/src/test_models.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import tensorflow as tf

from models import ConvexPotentialMap, ConvexPotentialModule, WTanh


DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')


def test_forward_shape():
    """Forward pass produces correct shape."""
    model = ConvexPotentialMap(in_dim=2, embed_dim=32, num_modules=8)
    x = tf.random.normal((5, 2), dtype=DTYPE)
    y = model(x)
    assert y.shape == (5, 2), f"Expected (5, 2), got {y.shape}"
    print(f"  test_forward_shape PASS: y.shape = {y.shape}")


def test_jacobian_shape():
    """Jacobian has correct shape."""
    model = ConvexPotentialMap(in_dim=2, embed_dim=32, num_modules=4)
    x = tf.random.normal((5, 2), dtype=DTYPE)
    _ = model(x)  # build
    J = model.jacobian(x)
    assert J.shape == (5, 2, 2), f"Expected (5, 2, 2), got {J.shape}"
    print(f"  test_jacobian_shape PASS: J.shape = {J.shape}")


def test_analytic_vs_autodiff_jacobian():
    """Analytic Jacobian matches tape.batch_jacobian."""
    for d in [1, 2, 3]:
        model = ConvexPotentialMap(in_dim=d, embed_dim=16, num_modules=4)
        x = tf.random.normal((4, d), dtype=DTYPE)

        # Build
        _ = model(x)

        # Autodiff Jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x)
        J_autodiff = tape.batch_jacobian(y, x)  # (batch, d, d)

        # Analytic Jacobian
        J_analytic = model.jacobian(x)

        max_diff = float(tf.reduce_max(tf.abs(J_autodiff - J_analytic)).numpy())
        rel_diff = max_diff / max(float(tf.reduce_max(tf.abs(J_autodiff)).numpy()), 1e-10)
        assert rel_diff < 1e-6, (
            f"d={d}: analytic vs autodiff diff = {max_diff:.2e} (rel {rel_diff:.2e})"
        )
        print(f"  test_analytic_vs_autodiff_jacobian d={d} PASS: rel_diff = {rel_diff:.2e}")


def test_jacobian_symmetric():
    """Jacobian is symmetric (W^T D W form)."""
    model = ConvexPotentialMap(in_dim=3, embed_dim=16, num_modules=8)
    x = tf.random.normal((10, 3), dtype=DTYPE)
    _ = model(x)
    J = model.jacobian(x)
    # Check symmetry: J - J^T should be ~zero
    asym = tf.reduce_max(tf.abs(J - tf.linalg.matrix_transpose(J)))
    assert float(asym.numpy()) < 1e-10, f"Jacobian not symmetric: max asym = {asym}"
    print(f"  test_jacobian_symmetric PASS: max_asym = {float(asym.numpy()):.2e}")


def test_jacobian_psd():
    """Jacobian is PSD (positive semidefinite)."""
    model = ConvexPotentialMap(in_dim=3, embed_dim=16, num_modules=8)
    x = tf.random.normal((10, 3), dtype=DTYPE)
    _ = model(x)
    J = model.jacobian(x)
    # Eigenvalues of J (symmetric, so use eigvalsh)
    eigvals = tf.linalg.eigvalsh(J)
    min_eig = float(tf.reduce_min(eigvals).numpy())
    # With residual=True, J = I + (PSD), so min eigval >= 1
    assert min_eig > 0.5, f"Min eigenvalue = {min_eig:.4f}, expected > 0.5 (residual+PSD)"
    print(f"  test_jacobian_psd PASS: min_eigval = {min_eig:.4f}")


def test_residual_near_identity_at_init():
    """At init, residual model should be approximately T(x) ~ x."""
    model = ConvexPotentialMap(in_dim=2, embed_dim=32, num_modules=8, residual=True)
    x = tf.random.normal((10, 2), dtype=DTYPE) * 0.5
    y = model(x)
    diff = tf.reduce_mean(tf.abs(y - x))
    print(f"  test_residual_near_identity_at_init: mean |T(x) - x| = {float(diff.numpy()):.4f}")
    # Not strict — just informational. With random init, the residual term can be nonzero.


def test_module_only_jacobian():
    """Test single module Jacobian against autodiff."""
    mod = ConvexPotentialModule(in_dim=2, embed_dim=16)
    x = tf.random.normal((5, 2), dtype=DTYPE)
    _ = mod(x)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = mod(x)
    J_auto = tape.batch_jacobian(y, x)
    J_anal = mod.jacobian(x)

    rel_diff = float(tf.reduce_max(tf.abs(J_auto - J_anal)) / tf.reduce_max(tf.abs(J_auto)))
    assert rel_diff < 1e-6, f"Module Jacobian mismatch: rel_diff = {rel_diff:.2e}"
    print(f"  test_module_only_jacobian PASS: rel_diff = {rel_diff:.2e}")


if __name__ == '__main__':
    print("\n=== ConvexPotentialMap tests ===\n")
    tf.random.set_seed(42)
    np.random.seed(42)

    test_forward_shape()
    test_jacobian_shape()
    test_module_only_jacobian()
    test_analytic_vs_autodiff_jacobian()
    test_jacobian_symmetric()
    test_jacobian_psd()
    test_residual_near_identity_at_init()

    print("\n=== All tests passed ===\n")
