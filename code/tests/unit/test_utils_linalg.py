"""
Tests for src/utils/linalg.py — numerically stable linear algebra operations.

# Run and save results:
#   .venv/bin/python -m pytest tests/test_utils_linalg.py -v 2>&1 | tee tests/test_utils_linalg_results.txt
"""

import pytest
import numpy as np
import tensorflow as tf

from src.utils.linalg import (
    safe_cholesky, safe_solve, log_det, safe_inv, safe_log_abs_det,
    symmetrize, matrix_sqrt,
    graph_safe_inv, graph_safe_log_abs_det,
    graph_safe_log_abs_det_fast, graph_safe_log_abs_det_svd,
    to_numpy,
)


# ============================================================================
# safe_cholesky — T1.1 to T1.4
# ============================================================================

class TestSafeCholesky:

    def test_well_conditioned(self):
        """T1.1 — L @ L^T == A for diagonal PD matrix."""
        A = tf.constant(np.diag([4.0, 9.0, 16.0]), dtype=tf.float64)
        L = safe_cholesky(A, jitter=0.0, adaptive=False)
        np.testing.assert_allclose(
            (L @ tf.transpose(L)).numpy(), A.numpy(), rtol=1e-5
        )
        np.testing.assert_allclose(
            tf.linalg.diag_part(L).numpy(), [2.0, 3.0, 4.0], rtol=1e-10
        )

    def test_adaptive_jitter_scales(self):
        """T1.2 — Adaptive jitter succeeds for both small and large matrices."""
        A_small = tf.constant(np.diag([1e-4, 1e-4]), dtype=tf.float64)
        A_large = tf.constant(np.diag([1e4, 1e4]), dtype=tf.float64)
        L_small = safe_cholesky(A_small)
        L_large = safe_cholesky(A_large)
        assert np.all(np.isfinite(L_small.numpy()))
        assert np.all(np.isfinite(L_large.numpy()))

    def test_near_singular(self):
        """T1.3 — Rank-1 (not PD) matrix: jitter rescues Cholesky."""
        A = tf.constant([[1.0, 1.0], [1.0, 1.0]], dtype=tf.float64)
        L = safe_cholesky(A, jitter=1e-6, adaptive=False)
        assert np.all(np.isfinite(L.numpy()))

    def test_batch_input(self):
        """T1.4 — Batch (3D) tensor processed correctly."""
        A_batch = tf.stack([
            tf.constant(np.diag([1.0, 2.0]), dtype=tf.float64),
            tf.constant(np.diag([3.0, 4.0]), dtype=tf.float64),
        ])
        L_batch = safe_cholesky(A_batch)
        assert L_batch.shape == (2, 2, 2)
        assert np.all(np.isfinite(L_batch.numpy()))


# ============================================================================
# safe_solve — T1.5 to T1.8
# ============================================================================

class TestSafeSolve:

    def test_vector_rhs(self):
        """T1.5 — Solve A x = b with vector RHS."""
        A = tf.constant([[2.0, 0.0], [0.0, 4.0]], dtype=tf.float64)
        b = tf.constant([6.0, 8.0], dtype=tf.float64)
        x = safe_solve(A, b)
        np.testing.assert_allclose(x.numpy(), [3.0, 2.0], rtol=1e-5)

    def test_matrix_rhs(self):
        """T1.6 — Solve A X = B with matrix RHS."""
        A = tf.constant(np.diag([2.0, 4.0]), dtype=tf.float64)
        B = tf.eye(2, dtype=tf.float64)
        X = safe_solve(A, B)
        np.testing.assert_allclose(
            X.numpy(), np.diag([0.5, 0.25]), rtol=1e-5
        )

    def test_cholesky_matches_default(self):
        """T1.7 — Cholesky method matches default for PD input."""
        A = tf.constant([[4.0, 1.0], [1.0, 3.0]], dtype=tf.float64)
        b = tf.constant([5.0, 7.0], dtype=tf.float64)
        x_default = safe_solve(A, b, method='default')
        x_cholesky = safe_solve(A, b, method='cholesky')
        np.testing.assert_allclose(
            x_default.numpy(), x_cholesky.numpy(), rtol=1e-4
        )

    def test_output_shape(self):
        """T1.8 — Vector in → vector out; matrix in → matrix out."""
        A = tf.constant([[2.0, 0.0], [0.0, 4.0]], dtype=tf.float64)
        b = tf.constant([6.0, 8.0], dtype=tf.float64)
        B = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float64)

        x = safe_solve(A, b)
        assert x.shape == (2,)

        X = safe_solve(A, B)
        assert X.shape == (2, 3)


# ============================================================================
# symmetrize — T1.9 to T1.10
# ============================================================================

class TestSymmetrize:

    def test_asymmetric_input(self):
        """T1.9 — (A + A^T)/2 gives correct symmetric result."""
        A = tf.constant([[1.0, 2.0], [4.0, 3.0]], dtype=tf.float64)
        S = symmetrize(A)
        np.testing.assert_allclose(
            S.numpy(), [[1.0, 3.0], [3.0, 3.0]], rtol=1e-6
        )
        np.testing.assert_allclose(S.numpy(), S.numpy().T, atol=1e-10)

    def test_already_symmetric(self):
        """T1.10 — Symmetric input passes through unchanged."""
        S = tf.constant(np.diag([5.0, 7.0]), dtype=tf.float64)
        np.testing.assert_allclose(symmetrize(S).numpy(), S.numpy(), rtol=1e-10)


# ============================================================================
# log_det — T1.11
# ============================================================================

class TestLogDet:

    def test_known_determinant(self):
        """T1.11 — log(det(diag(2,3,5))) = log(30)."""
        A = tf.constant(np.diag([2.0, 3.0, 5.0]), dtype=tf.float64)
        ld = log_det(A)
        np.testing.assert_allclose(ld.numpy(), np.log(30.0), rtol=1e-5)


# ============================================================================
# safe_inv — T1.12 to T1.13
# ============================================================================

class TestSafeInv:

    def test_round_trip(self):
        """T1.12 — A @ safe_inv(A) ≈ I."""
        A = tf.constant([[4.0, 1.0], [1.0, 3.0]], dtype=tf.float64)
        A_inv = safe_inv(A)
        product = (A @ A_inv).numpy()
        np.testing.assert_allclose(product, np.eye(2), atol=1e-5)

    def test_near_singular(self):
        """T1.13 — Near-singular matrix returns finite result."""
        A = tf.constant(np.eye(2) * 1e-8, dtype=tf.float64)
        A_inv = safe_inv(A)
        assert np.all(np.isfinite(A_inv.numpy()))


# ============================================================================
# safe_log_abs_det — T1.14 to T1.15
# ============================================================================

class TestSafeLogAbsDet:

    def test_matches_slogdet(self):
        """T1.14 — Matches tf.linalg.slogdet for regular PD matrix."""
        A = tf.constant([[3.0, 1.0], [1.0, 4.0]], dtype=tf.float64)
        _, expected = tf.linalg.slogdet(A)
        result = safe_log_abs_det(A, jitter=0.0)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)

    def test_negative_determinant(self):
        """T1.15 — Takes absolute value for negative determinant."""
        A = tf.constant([[-1.0, 0.0], [0.0, 2.0]], dtype=tf.float64)
        result = safe_log_abs_det(A)
        # det = -2, |det| = 2, but jitter shifts it slightly
        # With default jitter=1e-8: det = (-1+1e-8)*(2+1e-8) ≈ -2
        np.testing.assert_allclose(result.numpy(), np.log(2.0), atol=1e-5)


# ============================================================================
# matrix_sqrt — T1.16 to T1.17
# ============================================================================

class TestMatrixSqrt:

    def test_cholesky(self):
        """T1.16 — Cholesky sqrt: S @ S^T == A."""
        A = tf.constant([[4.0, 2.0], [2.0, 3.0]], dtype=tf.float64)
        S = matrix_sqrt(A, method='cholesky')
        reconstructed = (S @ tf.transpose(S)).numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), rtol=1e-5)

    def test_eig(self):
        """T1.17 — Eig sqrt: S @ S^T == A and S is symmetric."""
        A = tf.constant([[4.0, 2.0], [2.0, 3.0]], dtype=tf.float64)
        S = matrix_sqrt(A, method='eig')
        reconstructed = (S @ tf.transpose(S)).numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)
        np.testing.assert_allclose(S.numpy(), S.numpy().T, atol=1e-6)


# ============================================================================
# graph_safe_inv — T1.18 to T1.19
# ============================================================================

class TestGraphSafeInv:

    def test_matches_safe_inv(self):
        """T1.18 — pinv-based inverse matches standard inv for PD matrix."""
        A = tf.constant([[4.0, 1.0], [1.0, 3.0]], dtype=tf.float64)
        A_inv_safe = safe_inv(A)
        A_inv_graph = graph_safe_inv(A)
        np.testing.assert_allclose(
            A_inv_graph.numpy(), A_inv_safe.numpy(), rtol=1e-5
        )

    def test_near_singular(self):
        """T1.19 — Near-singular matrix returns finite result."""
        A = tf.constant(np.eye(2) * 1e-8, dtype=tf.float64)
        A_inv = graph_safe_inv(A)
        assert np.all(np.isfinite(A_inv.numpy()))


# ============================================================================
# graph_safe_log_abs_det — T1.20 to T1.21
# ============================================================================

class TestGraphSafeLogAbsDet:

    def test_forward_matches(self):
        """T1.20 — Forward pass matches safe_log_abs_det."""
        A = tf.constant([[3.0, 1.0], [1.0, 4.0]], dtype=tf.float64)
        result_safe = safe_log_abs_det(A, jitter=0.0)
        result_graph = graph_safe_log_abs_det(A, jitter=0.0)
        np.testing.assert_allclose(
            result_graph.numpy(), result_safe.numpy(), rtol=1e-5
        )

    def test_gradient_finite(self):
        """T1.21 — Custom gradient backward pass produces finite gradient.

        Gradient of log|det(A)| = A^{-T}. We verify it is finite and has
        the correct shape.
        """
        A = tf.Variable([[3.0, 1.0], [1.0, 4.0]], dtype=tf.float64)
        with tf.GradientTape() as tape:
            y = graph_safe_log_abs_det(A, jitter=1e-8)
        grad = tape.gradient(y, A)
        assert grad is not None
        assert grad.shape == (2, 2)
        assert np.all(np.isfinite(grad.numpy()))


# ============================================================================
# graph_safe_log_abs_det_fast — T1.22 to T1.23
# ============================================================================

class TestGraphSafeLogAbsDetFast:

    def test_forward_matches(self):
        """T1.22 — Forward matches safe_log_abs_det for regular matrix."""
        A = tf.constant([[3.0, 1.0], [1.0, 4.0]], dtype=tf.float64)
        result = graph_safe_log_abs_det_fast(A, jitter=0.0)
        expected = safe_log_abs_det(A, jitter=0.0)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)

    def test_nan_guard_zeros_gradient(self):
        """T1.23 — NaN input: gradient is zero (NaN guard fires).

        The NaN guard is designed for HMC: diverged leapfrog proposals produce
        NaN Jacobians. Those proposals get rejected anyway, so zero gradient
        is correct — it prevents the gradient from propagating NaN into the
        sampler state.
        """
        A_nan = tf.Variable(
            [[float('nan'), 0.0], [0.0, 1.0]], dtype=tf.float64
        )
        with tf.GradientTape() as tape:
            y = graph_safe_log_abs_det_fast(A_nan, jitter=1e-8)
        grad = tape.gradient(y, A_nan)
        assert grad is not None
        np.testing.assert_allclose(grad.numpy(), np.zeros((2, 2)), atol=1e-30)


# ============================================================================
# graph_safe_log_abs_det_svd — T1.24 to T1.25
# ============================================================================

class TestGraphSafeLogAbsDetSVD:

    def test_matches_slogdet(self):
        """T1.24 — Matches slogdet for regular PD matrix."""
        A = tf.constant([[3.0, 1.0], [1.0, 4.0]], dtype=tf.float64)
        _, expected = tf.linalg.slogdet(A)
        result = graph_safe_log_abs_det_svd(A, jitter=0.0)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)

    def test_negative_determinant(self):
        """T1.25 — SVD singular values are always positive, so handles negative det."""
        A = tf.constant([[-1.0, 0.0], [0.0, 2.0]], dtype=tf.float64)
        result = graph_safe_log_abs_det_svd(A)
        # det = -2, singular values = [2, 1+jitter], log(prod) ≈ log(2)
        np.testing.assert_allclose(result.numpy(), np.log(2.0), atol=1e-4)


# ============================================================================
# to_numpy — T1.26 to T1.27
# ============================================================================

class TestToNumpy:

    def test_tf_tensor(self):
        """T1.26 — Converts TF tensor to numpy ndarray."""
        t = tf.constant([1.0, 2.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_passthrough_numpy(self):
        """T1.27a — Numpy array passes through as same object."""
        a = np.array([1.0])
        assert to_numpy(a) is a

    def test_passthrough_scalar(self):
        """T1.27b — Python scalar passes through unchanged."""
        assert to_numpy(3.14) == 3.14


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
