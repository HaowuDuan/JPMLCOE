"""Distribution utilities for stable probability computations."""

import numpy as np
from typing import Union, Optional


def log_gaussian_prob(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute log probability under multivariate Gaussian (numerically stable).

    log p(x) = -0.5 * [n*log(2π) + log|Σ| + (x-μ)^T Σ^(-1) (x-μ)]

    Args:
        x: Data point of shape (n,)
        mean: Mean of shape (n,)
        cov: Covariance of shape (n, n)

    Returns:
        Log probability (scalar)
    """
    diff = x - mean
    n = len(x)

    # Compute log determinant
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite")

    # Compute Mahalanobis distance
    try:
        mahalanobis = diff @ np.linalg.solve(cov, diff)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        mahalanobis = diff @ np.linalg.pinv(cov) @ diff

    return -0.5 * (n * np.log(2 * np.pi) + logdet + mahalanobis)


def log_sum_exp(log_values: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Compute log(sum(exp(log_values))) stably.

    Uses the log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Args:
        log_values: Array of log values
        axis: Axis along which to sum (None for all)

    Returns:
        log(sum(exp(log_values)))
    """
    max_log = np.max(log_values, axis=axis, keepdims=True)
    return np.squeeze(max_log + np.log(np.sum(np.exp(log_values - max_log), axis=axis, keepdims=True)))


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights in log-space: w_i / sum(w_j)

    Args:
        log_weights: Log weights of shape (N,) or (batch, N)

    Returns:
        Normalized weights (not in log space) of same shape
    """
    max_log = np.max(log_weights, axis=-1, keepdims=True)
    weights_unnorm = np.exp(log_weights - max_log)
    return weights_unnorm / np.sum(weights_unnorm, axis=-1, keepdims=True)


def multivariate_normal_sample(mean: np.ndarray, cov: np.ndarray,
                               rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
    """
    Sample from multivariate Gaussian with stable Cholesky decomposition.

    Args:
        mean: Mean of shape (d,)
        cov: Covariance of shape (d, d)
        rng: Random number generator
        n_samples: Number of samples

    Returns:
        Samples of shape (n_samples, d) or (d,) if n_samples=1
    """
    from .linalg import safe_cholesky

    d = len(mean)
    L = safe_cholesky(cov)

    # Sample standard normal
    z = rng.standard_normal((n_samples, d))

    # Transform to N(mean, cov)
    samples = mean + z @ L.T

    return samples if n_samples > 1 else samples[0]
