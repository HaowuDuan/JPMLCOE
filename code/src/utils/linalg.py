"""Numerically stable linear algebra operations - TensorFlow version."""

import tensorflow as tf
import numpy as np


@tf.function
def safe_cholesky(A: tf.Tensor, jitter: float = 1e-10, adaptive: bool = True) -> tf.Tensor:
    """
    Compute Cholesky decomposition with optional adaptive regularization.
    
    Adaptive mode scales jitter by the average diagonal magnitude of A,
    matching MATLAB's philosophy of scale-dependent regularization.
    
    Args:
        A: Symmetric positive semi-definite matrix (..., n, n)
        jitter: Base regularization strength (default: 1e-10, matching MATLAB's 1e-14 with buffer)
        adaptive: If True, scale jitter by matrix trace (default: True)
    
    Returns:
        L: Lower triangular Cholesky factor such that L @ L^T ≈ A
    
    Example:
        >>> A = tf.constant([[100.0, 0.0], [0.0, 100.0]])
        >>> L = safe_cholesky(A, jitter=1e-10, adaptive=True)
        >>> # Effective jitter = 1e-10 * 100 = 1e-8
    """
    n = tf.shape(A)[-1]
    eye = tf.eye(n, dtype=A.dtype)
    
    if adaptive:
        # Scale jitter by average diagonal (trace / n)
        trace_A = tf.linalg.trace(A)
        n_float = tf.cast(n, A.dtype)
        avg_diag = trace_A / n_float
        
        # Scale jitter, with minimum = base jitter
        scaled_jitter = jitter * tf.maximum(avg_diag, 1.0)
    else:
        scaled_jitter = tf.cast(jitter, A.dtype)
    
    # Reshape for batch broadcasting: (...,) -> (..., 1, 1)
    scaled_jitter = tf.reshape(scaled_jitter, tf.concat([tf.shape(scaled_jitter), [1, 1]], axis=0))
    A_reg = A + eye * scaled_jitter
    return tf.linalg.cholesky(A_reg)


@tf.function
def safe_solve(A: tf.Tensor, b: tf.Tensor, method: str = 'default') -> tf.Tensor:
    """
    Solve linear system Ax = b with fallback strategies.

    Args:
        A: Coefficient matrix of shape (..., n, n)
        b: Right-hand side of shape (..., n) or (..., n, k)
        method: 'default', 'cholesky', or 'lstsq'

    Returns:
        x: Solution of same shape as b
    """
    # Check if b is a vector (needs extra dim) or already a matrix
    b_is_vector = (len(b.shape) < len(A.shape))
    
    if b_is_vector:
        b_rhs = b[..., tf.newaxis]
    else:
        b_rhs = b
    
    if method == 'cholesky':
        L = safe_cholesky(A)
        result = tf.linalg.cholesky_solve(L, b_rhs)
    elif method == 'lstsq':
        result = tf.linalg.lstsq(A, b_rhs, fast=False)
    else:
        # Default: use direct solve
        result = tf.linalg.solve(A, b_rhs)
    
    if b_is_vector:
        return result[..., 0]
    return result


@tf.function
def log_det(A: tf.Tensor) -> tf.Tensor:
    """
    Compute log determinant stably.

    Args:
        A: Positive definite matrix of shape (..., n, n)

    Returns:
        log|det(A)| of shape (...)
    """
    sign, logdet = tf.linalg.slogdet(A)
    tf.debugging.assert_positive(sign, message="log_det: matrix is not positive definite (sign <= 0)")
    return logdet


def safe_inv(A: tf.Tensor, jitter: float = 1e-10) -> tf.Tensor:
    """
    Compute matrix inverse with diagonal regularization.

    Prevents crashes when A is singular or near-singular (e.g. during
    HMC exploration when parameters take extreme values).

    NOT @tf.function: must run eagerly so MatrixInverse returns NaN/Inf
    instead of raising on near-singular inputs.

    Args:
        A: Matrix of shape (..., n, n)
        jitter: Regularization added to diagonal (default: 1e-10)

    Returns:
        A^{-1} of shape (..., n, n)
    """
    n = tf.shape(A)[-1]
    eye = tf.eye(n, dtype=A.dtype)
    return tf.linalg.inv(A + jitter * eye)


@tf.function
def safe_log_abs_det(M: tf.Tensor, jitter: float = 1e-8) -> tf.Tensor:
    """
    Compute log|det(M)| with regularization for backward pass stability.

    TF's gradient of det(M) uses MatrixInverse, which crashes in @tf.function
    when M is singular (eager mode silently returns NaN instead). Adding
    jitter*I ensures M is always invertible for the backward pass.

    Args:
        M: Matrix of shape (..., n, n). Need not be symmetric or PD.
        jitter: Regularization added to diagonal (default: 1e-8)

    Returns:
        log|det(M)| of shape (...)
    """
    n = tf.shape(M)[-1]
    eye = tf.eye(n, dtype=M.dtype)
    M_reg = M + jitter * eye
    return tf.math.log(tf.abs(tf.linalg.det(M_reg)))


# ---------------------------------------------------------------------------
# Graph-safe variants (for use inside tf.while_loop on GPU)
#
# tf.linalg.inv and the backward pass of tf.linalg.det/slogdet use the
# MatrixInverse op, which raises InvalidArgumentError on GPU in graph mode
# for singular/NaN inputs. These variants avoid MatrixInverse entirely.
# ---------------------------------------------------------------------------

def graph_safe_inv(A: tf.Tensor, jitter: float = 1e-10) -> tf.Tensor:
    """
    Like safe_inv but uses pinv (SVD-based) — safe in GPU graph mode.

    pinv never raises and equals inv for invertible matrices.

    Args:
        A: Matrix of shape (..., n, n)
        jitter: Regularization added to diagonal (default: 1e-10)

    Returns:
        A^{-1} of shape (..., n, n)
    """
    n = tf.shape(A)[-1]
    eye = tf.eye(n, dtype=A.dtype)
    return tf.linalg.pinv(A + jitter * eye)


# -- Custom gradient approach (recommended): fast forward, robust backward --

@tf.custom_gradient
def _graph_safe_log_abs_det_impl(M_reg):
    """log|det(M)| with robust backward pass.

    Forward: slogdet (fast LU, O(n^3) small constant).
    Backward: gradient of log|det(M)| is M^{-T}; uses pinv instead of inv
    to avoid MatrixInverse crash on GPU in graph mode.
    """
    sign, logabsdet = tf.linalg.slogdet(M_reg)
    def grad(dy):
        M_inv_T = tf.linalg.matrix_transpose(tf.linalg.pinv(M_reg))
        return dy[..., tf.newaxis, tf.newaxis] * M_inv_T
    return logabsdet, grad


@tf.function
def graph_safe_log_abs_det(M: tf.Tensor, jitter: float = 1e-8) -> tf.Tensor:
    """
    Like safe_log_abs_det but safe in GPU graph mode backward pass.

    Uses slogdet (fast LU) forward + pinv backward via custom gradient.

    Args:
        M: Matrix of shape (..., n, n). Need not be symmetric or PD.
        jitter: Regularization added to diagonal (default: 1e-8)

    Returns:
        log|det(M)| of shape (...)
    """
    n = tf.shape(M)[-1]
    eye = tf.eye(n, dtype=M.dtype)
    M_reg = M + jitter * eye
    return _graph_safe_log_abs_det_impl(M_reg)


# -- Sanitize + inv approach: fast inv with NaN guard, no pinv overhead --

@tf.custom_gradient
def _graph_safe_log_abs_det_fast_impl(M_reg):
    """log|det(M)| with NaN-guarded fast backward pass.

    Forward: slogdet (fast LU).
    Backward: detects NaN matrices (from diverged leapfrog), replaces them
    with identity before calling fast tf.linalg.inv. NaN matrices get zero
    gradient — HMC will reject these proposals anyway. Extra jitter (1e-6)
    ensures near-singular matrices are invertible.
    """
    sign, logabsdet = tf.linalg.slogdet(M_reg)
    def grad(dy):
        is_finite = tf.reduce_all(tf.math.is_finite(M_reg), axis=[-2, -1])
        n = tf.shape(M_reg)[-1]
        eye = tf.eye(n, dtype=M_reg.dtype)
        # Replace NaN matrices with identity (always invertible)
        M_safe = tf.where(
            is_finite[..., tf.newaxis, tf.newaxis], M_reg, eye
        )
        # Extra backward jitter for near-singular cases
        M_inv_T = tf.linalg.matrix_transpose(
            tf.linalg.inv(M_safe + 1e-6 * eye)
        )
        # Zero out gradient for NaN matrices
        M_inv_T = tf.where(
            is_finite[..., tf.newaxis, tf.newaxis],
            M_inv_T, tf.zeros_like(M_inv_T)
        )
        return dy[..., tf.newaxis, tf.newaxis] * M_inv_T
    return logabsdet, grad


@tf.function
def graph_safe_log_abs_det_fast(M: tf.Tensor, jitter: float = 1e-8) -> tf.Tensor:
    """
    Like graph_safe_log_abs_det but uses fast inv with NaN guard in backward.

    Sanitizes NaN matrices before calling tf.linalg.inv instead of using
    pinv (SVD) for all matrices. Near-original speed.

    Args:
        M: Matrix of shape (..., n, n). Need not be symmetric or PD.
        jitter: Regularization added to diagonal (default: 1e-8)

    Returns:
        log|det(M)| of shape (...)
    """
    n = tf.shape(M)[-1]
    eye = tf.eye(n, dtype=M.dtype)
    M_reg = M + jitter * eye
    return _graph_safe_log_abs_det_fast_impl(M_reg)


# -- SVD approach (not used): robust but ~4x slower due to full SVD --

@tf.function
def graph_safe_log_abs_det_svd(M: tf.Tensor, jitter: float = 1e-8) -> tf.Tensor:
    """
    log|det(M)| via SVD. Most robust but expensive (~4x slower than LU).

    SVD never raises and its gradient doesn't use MatrixInverse.
    Useful as fallback if the custom gradient approach has issues.

    Args:
        M: Matrix of shape (..., n, n). Need not be symmetric or PD.
        jitter: Regularization added to diagonal (default: 1e-8)

    Returns:
        log|det(M)| of shape (...)
    """
    n = tf.shape(M)[-1]
    eye = tf.eye(n, dtype=M.dtype)
    M_reg = M + jitter * eye
    s = tf.linalg.svd(M_reg, compute_uv=False)
    return tf.reduce_sum(tf.math.log(tf.maximum(s, 1e-30)), axis=-1)


@tf.function
def symmetrize(A: tf.Tensor) -> tf.Tensor:
    """
    Force matrix to be symmetric: A_sym = (A + A^T) / 2

    Args:
        A: Matrix of shape (..., n, n)

    Returns:
        Symmetric matrix of shape (..., n, n)
    """
    return 0.5 * (A + tf.linalg.matrix_transpose(A))


@tf.function
def matrix_sqrt(A: tf.Tensor, method: str = 'cholesky') -> tf.Tensor:
    """
    Compute matrix square root: sqrt(A) such that sqrt(A) @ sqrt(A)^T = A

    Args:
        A: Positive definite matrix of shape (..., n, n)
        method: 'cholesky' or 'eig' (eigenvalue decomposition)

    Returns:
        Matrix square root of shape (..., n, n)
    """
    if method == 'cholesky':
        return safe_cholesky(A)
    elif method == 'eig':
        # Eigenvalue decomposition
        eigvals, eigvecs = tf.linalg.eigh(A)
        eigvals = tf.maximum(eigvals, 1e-10)  # Ensure positive
        sqrt_eigvals = tf.sqrt(eigvals)
        # sqrt(A) = V @ diag(sqrt(λ)) @ V^T
        return eigvecs @ tf.linalg.diag(sqrt_eigvals) @ tf.linalg.matrix_transpose(eigvecs)
    else:
        raise ValueError(f"Unknown method: {method}")


def to_numpy(x):
    """Convert TF tensor to numpy; pass through if already numpy/scalar."""
    return x.numpy() if isinstance(x, tf.Tensor) else x
