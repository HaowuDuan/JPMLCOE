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
