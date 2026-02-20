"""Distribution utilities for stable probability computations - TensorFlow version."""

import math
import tensorflow as tf
from typing import Optional, Tuple
from .linalg import safe_cholesky


@tf.function
def log_gaussian_prob(x: tf.Tensor, mean: tf.Tensor, cov: tf.Tensor) -> tf.Tensor:
    """
    Compute log probability under multivariate Gaussian (numerically stable).

    log p(x) = -0.5 * [n*log(2π) + log|Σ| + (x-μ)^T Σ^(-1) (x-μ)]

    Args:
        x: Data point of shape (..., n)
        mean: Mean of shape (..., n)
        cov: Covariance of shape (..., n, n)

    Returns:
        Log probability of shape (...)
    """
    diff = x - mean
    n = tf.cast(tf.shape(x)[-1], x.dtype)

    # Compute log determinant
    sign, logdet = tf.linalg.slogdet(cov)

    # Compute Mahalanobis distance using Cholesky
    L = safe_cholesky(cov)
    y = tf.linalg.triangular_solve(L, diff[..., tf.newaxis], lower=True)[..., 0]
    mahalanobis = tf.reduce_sum(y**2, axis=-1)

    return -0.5 * (n * tf.math.log(2.0 * tf.constant(math.pi, dtype=x.dtype)) + logdet + mahalanobis)


@tf.function
def log_sum_exp(log_values: tf.Tensor, axis: Optional[int] = None) -> tf.Tensor:
    """
    Compute log(sum(exp(log_values))) stably.

    Uses TensorFlow's built-in reduce_logsumexp.

    Args:
        log_values: Tensor of log values
        axis: Axis along which to sum (None for all)

    Returns:
        log(sum(exp(log_values)))
    """
    return tf.reduce_logsumexp(log_values, axis=axis)


@tf.function
def normalize_log_weights(log_weights: tf.Tensor, clip_range: Optional[Tuple[float, float]] = None) -> tf.Tensor:
    """
    Normalize weights in log-space: w_i / sum(w_j)

    Args:
        log_weights: Log weights of shape (..., N)
        clip_range: Optional (min, max) tuple to clip log_weights before exp()
                    e.g., (-30, 30) prevents overflow/underflow

    Returns:
        Normalized weights (not in log space) of same shape
    """
    max_log = tf.reduce_max(log_weights, axis=-1, keepdims=True)
    log_weights_normalized = log_weights - max_log

    if clip_range is not None:
        log_weights_normalized = tf.clip_by_value(
            log_weights_normalized, clip_range[0], clip_range[1]
        )

    weights_unnorm = tf.exp(log_weights_normalized)
    return weights_unnorm / tf.reduce_sum(weights_unnorm, axis=-1, keepdims=True)


@tf.function
def multivariate_normal_sample(mean: tf.Tensor, cov: tf.Tensor,
                                n_samples: int, seed: tf.Tensor) -> tf.Tensor:
    """
    Sample from multivariate Gaussian with stable Cholesky decomposition.

    Args:
        mean: Mean of shape (d,)
        cov: Covariance of shape (d, d)
        n_samples: Number of samples
        seed: Random seed for stateless sampling

    Returns:
        Samples of shape (n_samples, d)
    """
    d = tf.shape(mean)[0]
    L = tf.linalg.cholesky(cov)
    z = tf.random.stateless_normal([n_samples, d], seed=seed, dtype=mean.dtype)

    # Correct batch multiplication: z @ L^T
    return mean + tf.linalg.matmul(z, L, transpose_b=True)


def sample_particles_cholesky(
    initial_mean: tf.Tensor,
    initial_cov: tf.Tensor,
    n_particles: int,
    state_dim: int,
    seed: tf.Tensor,
    dtype=tf.float64
) -> tf.Tensor:
    """Sample N particles from N(mean, cov) via Cholesky decomposition."""
    L = safe_cholesky(initial_cov)
    z = tf.random.stateless_normal([n_particles, state_dim], seed=seed, dtype=dtype)
    return initial_mean + tf.linalg.matmul(z, L, transpose_b=True)


@tf.function
def compute_flow_weights(
    eta_1: tf.Tensor,
    eta_0: tf.Tensor,
    particles_prev: tf.Tensor,
    observation: tf.Tensor,
    model,
    prev_weights: Optional[tf.Tensor] = None,
    jacobians: Optional[tf.Tensor] = None,
    clip_range: Optional[Tuple[float, float]] = None
) -> tf.Tensor:
    """
    Compute particle weights for invertible flow filters with numerical stability.

    Optimized with vectorization when Q is state-independent (default).

    Weight formula: w_i ∝ p(z|η₁ⁱ) * p(η₁ⁱ|x_{k-1}ⁱ) * θ_i / p(η₀ⁱ|x_{k-1}ⁱ) * w_{k-1}ⁱ

    where:
    - η₁ⁱ: Flowed particle at λ=1
    - η₀ⁱ: Sampled particle at λ=0
    - θ_i: Product of Jacobian determinants (optional, for LEDH)
    - w_{k-1}ⁱ: Previous weight (optional, for sequential updates)

    Args:
        eta_1: Flowed particles at λ=1, shape (N, d)
        eta_0: Sampled particles at λ=0, shape (N, d)
        particles_prev: Particles from previous timestep, shape (N, d)
        observation: Current observation, shape (obs_dim,)
        model: StateSpaceModel with TensorFlow batch methods
        prev_weights: Previous weights, shape (N,). Defaults to uniform if None.
        jacobians: Jacobian determinants, shape (N,). Defaults to ones if None.
        clip_range: Optional range to clip log weights before exp(). None for no clipping (MATLAB behavior).

    Returns:
        Normalized weights, shape (N,)
    """
    n_particles = tf.shape(eta_1)[0]
    state_dim = tf.shape(eta_1)[1]

    if prev_weights is None:
        prev_weights = tf.ones(n_particles, dtype=eta_1.dtype) / tf.cast(n_particles, eta_1.dtype)

    if jacobians is None:
        jacobians = tf.ones(n_particles, dtype=eta_1.dtype)

    # Vectorized path (assumes state-independent Q)
    # 1. Batch transition means
    f_prev = model.state_transition_mean_batch(particles_prev)

    # 2. Single Q matrix
    Q = model.state_transition_cov_batch(particles_prev)

    # 3. Batch observation log-probs
    log_p_obs = model.log_observation_prob_batch(observation, eta_1)

    # 4. Vectorized log p(η₁ | x_{k-1})
    diff_1 = eta_1 - f_prev
    L_Q = safe_cholesky(Q)
    y_1 = tf.linalg.triangular_solve(L_Q, tf.transpose(diff_1), lower=True)
    y_1 = tf.transpose(y_1)

    log_p_eta1 = -0.5 * (
        tf.reduce_sum(y_1**2, axis=1) +
        2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Q))) +
        tf.cast(state_dim, eta_1.dtype) * tf.math.log(2.0 * tf.constant(math.pi, dtype=eta_1.dtype))
    )

    # 5. Vectorized log p(η₀ | x_{k-1})
    diff_0 = eta_0 - f_prev
    y_0 = tf.linalg.triangular_solve(L_Q, tf.transpose(diff_0), lower=True)
    y_0 = tf.transpose(y_0)

    log_p_eta0 = -0.5 * (
        tf.reduce_sum(y_0**2, axis=1) +
        2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Q))) +
        tf.cast(state_dim, eta_1.dtype) * tf.math.log(2.0 * tf.constant(math.pi, dtype=eta_1.dtype))
    )

    # 6. Combine in log space
    log_weights = (
        log_p_eta1 +
        log_p_obs +
        tf.math.log(tf.maximum(jacobians, 1e-300)) -
        log_p_eta0 +
        tf.math.log(tf.maximum(prev_weights, 1e-300))
    )

    # Replace NaN with -inf so those particles get zero weight
    log_weights = tf.where(tf.math.is_finite(log_weights), log_weights, tf.constant(-1e30, dtype=log_weights.dtype))

    # Normalize
    weights = normalize_log_weights(log_weights, clip_range=clip_range)

    # Check for weight collapse
    is_finite = tf.reduce_all(tf.math.is_finite(weights))
    uniform_weights = tf.ones(n_particles, dtype=eta_1.dtype) / tf.cast(n_particles, eta_1.dtype)
    def _warn_and_uniform():
        tf.print("WARNING: weight collapse detected, falling back to uniform weights")
        return uniform_weights
    weights = tf.cond(is_finite, lambda: weights, _warn_and_uniform)

    return weights
