"""Numerically stable linear algebra operations."""

import numpy as np
from typing import Optional


def safe_cholesky(A: np.ndarray, jitter: float = 1e-6, max_tries: int = 5) -> np.ndarray:
    """
    Compute Cholesky decomposition with automatic regularization.

    Args:
        A: Symmetric positive definite matrix of shape (n, n)
        jitter: Initial regularization parameter
        max_tries: Maximum number of attempts with increasing jitter

    Returns:
        L: Lower triangular Cholesky factor

    Raises:
        ValueError: If decomposition fails after max_tries attempts
    """
    current_jitter = jitter

    for attempt in range(max_tries):
        try:
            if attempt == 0:
                # First try without jitter
                return np.linalg.cholesky(A)
            else:
                # Add jitter and retry
                A_reg = A + np.eye(A.shape[0]) * current_jitter
                return np.linalg.cholesky(A_reg)
        except np.linalg.LinAlgError:
            if attempt < max_tries - 1:
                current_jitter *= 10.0
            else:
                # Last resort: use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(A)
                eigvals = np.maximum(eigvals, jitter)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                return L

    raise ValueError(f"Cholesky decomposition failed after {max_tries} attempts")


def safe_solve(A: np.ndarray, b: np.ndarray, method: str = 'default') -> np.ndarray:
    """
    Solve linear system Ax = b with fallback strategies.

    Args:
        A: Coefficient matrix of shape (n, n)
        b: Right-hand side of shape (n,) or (n, k)
        method: 'default' (uses np.linalg.solve), 'cholesky', or 'lstsq'

    Returns:
        x: Solution of shape (n,) or (n, k)
    """
    if method == 'cholesky':
        try:
            L = safe_cholesky(A)
            # Solve L L^T x = b
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to least squares
            return np.linalg.lstsq(A, b, rcond=None)[0]
    elif method == 'lstsq':
        return np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            return np.linalg.lstsq(A, b, rcond=None)[0]


def log_det(A: np.ndarray, use_cholesky: bool = True) -> float:
    """
    Compute log determinant stably.

    Args:
        A: Positive definite matrix of shape (n, n)
        use_cholesky: If True, use Cholesky decomposition (faster and more stable)

    Returns:
        log|det(A)|
    """
    if use_cholesky:
        try:
            L = safe_cholesky(A)
            return 2.0 * np.sum(np.log(np.diag(L)))
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to slogdet
            sign, logdet = np.linalg.slogdet(A)
            if sign <= 0:
                raise ValueError("Matrix is not positive definite")
            return logdet
    else:
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            raise ValueError("Matrix is not positive definite")
        return logdet


def symmetrize(A: np.ndarray) -> np.ndarray:
    """
    Force matrix to be symmetric: A_sym = (A + A^T) / 2

    Args:
        A: Matrix of shape (n, n)

    Returns:
        Symmetric matrix of shape (n, n)
    """
    return 0.5 * (A + A.T)


def matrix_sqrt(A: np.ndarray, method: str = 'cholesky') -> np.ndarray:
    """
    Compute matrix square root: sqrt(A) such that sqrt(A) @ sqrt(A)^T = A

    Args:
        A: Positive definite matrix of shape (n, n)
        method: 'cholesky' or 'eig' (eigenvalue decomposition)

    Returns:
        Matrix square root of shape (n, n)
    """
    if method == 'cholesky':
        return safe_cholesky(A)
    elif method == 'eig':
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    else:
        raise ValueError(f"Unknown method: {method}")
