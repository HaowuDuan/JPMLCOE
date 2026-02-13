"""Numerically stable linear algebra operations - TensorFlow version."""

import tensorflow as tf


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
        scaled_jitter = jitter
    
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
    if method == 'cholesky':
        L = safe_cholesky(A)
        return tf.linalg.cholesky_solve(L, b[..., tf.newaxis])[..., 0]
    elif method == 'lstsq':
        return tf.linalg.lstsq(A, b[..., tf.newaxis], fast=False)[..., 0]
    else:
        # Default: use direct solve
        return tf.linalg.solve(A, b[..., tf.newaxis])[..., 0]


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
    return logdet


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
