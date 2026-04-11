"""KDE utilities for neural operator training.

Provides:
- Weighted Silverman bandwidth using effective sample size
- Evaluation of weighted/uniform KDE at query points
- Sampling from the uniform KDE q_h (for collocation points)

All ops are pure TensorFlow and differentiable through the bandwidth and
particle positions where appropriate.
"""

import tensorflow as tf
import math


# ============================================================================
# Bandwidth selection
# ============================================================================

def effective_sample_size(weights):
    """ESS = 1 / sum(w_i^2). Returns scalar tensor."""
    return 1.0 / tf.reduce_sum(weights ** 2)


def weighted_mean(particles, weights):
    """Weighted mean. particles: (N, d), weights: (N,). Returns (d,)."""
    return tf.reduce_sum(weights[:, None] * particles, axis=0)


def weighted_covariance(particles, weights):
    """Weighted covariance matrix. Returns (d, d)."""
    mu = weighted_mean(particles, weights)
    centered = particles - mu[None, :]
    cov = tf.einsum('n,ni,nj->ij', weights, centered, centered)
    return cov


def silverman_bandwidth_scalar(particles, weights, scale: float = 1.0):
    """Silverman's rule: scalar bandwidth using N_eff and weighted std.

    h = scale * (4/(d+2))^(1/(d+4)) * std * N_eff^(-1/(d+4))

    Returns a scalar h. Uses the average weighted standard deviation across
    dimensions. For per-dimension or full-covariance bandwidth, use
    silverman_bandwidth_matrix.

    The `scale` parameter lets you anneal: scale=1.0 starts (default Silverman),
    scale=0.5 makes it half, etc.
    """
    d = tf.cast(tf.shape(particles)[-1], particles.dtype)
    n_eff = effective_sample_size(weights)
    cov = weighted_covariance(particles, weights)
    # Average diagonal std (geometric mean would also work)
    std = tf.reduce_mean(tf.sqrt(tf.linalg.diag_part(cov)))
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    h = scale * factor * std * tf.pow(n_eff, -1.0 / (d + 4.0))
    return h


def silverman_bandwidth_matrix(particles, weights, scale: float = 1.0):
    """Full covariance bandwidth matrix.

    H = scale^2 * (4/(d+2))^(2/(d+4)) * Sigma * N_eff^(-2/(d+4))

    Returns a (d, d) matrix.
    """
    d = tf.cast(tf.shape(particles)[-1], particles.dtype)
    n_eff = effective_sample_size(weights)
    cov = weighted_covariance(particles, weights)
    factor = (4.0 / (d + 2.0)) ** (2.0 / (d + 4.0))
    H = (scale ** 2) * factor * cov * tf.pow(n_eff, -2.0 / (d + 4.0))
    return H


# ============================================================================
# KDE evaluation
# ============================================================================

def gaussian_log_kernel_diag(x_query, x_centers, h_diag):
    """log K(x_query - x_centers) for Gaussian kernel with diagonal covariance.

    x_query: (Q, d) query points
    x_centers: (N, d) kernel centers
    h_diag: (d,) per-dimension bandwidth (standard deviations)

    Returns: (Q, N) log-kernel values
    """
    d = tf.cast(tf.shape(x_query)[-1], x_query.dtype)
    # Pairwise differences: (Q, N, d)
    diff = x_query[:, None, :] - x_centers[None, :, :]
    # Scaled squared distance: (Q, N)
    sq_scaled = tf.reduce_sum((diff / h_diag[None, None, :]) ** 2, axis=-1)
    # log normalizer: -d/2 log(2pi) - sum log h_i
    log_norm = -0.5 * d * tf.math.log(2.0 * tf.constant(math.pi, dtype=x_query.dtype)) \
               - tf.reduce_sum(tf.math.log(h_diag))
    return log_norm - 0.5 * sq_scaled


def gaussian_log_kernel_scalar(x_query, x_centers, h):
    """log K(x_query - x_centers) for Gaussian kernel with isotropic h.

    h: scalar bandwidth
    Returns: (Q, N)
    """
    d_int = tf.shape(x_query)[-1]
    h_diag = tf.fill([d_int], h)
    return gaussian_log_kernel_diag(x_query, x_centers, h_diag)


def log_kde_weighted(x_query, particles, weights, h):
    """log p_h(x_query) where p_h(x) = sum_i w_i K_h(x - x_i).

    Uses logsumexp for numerical stability.

    x_query: (Q, d)
    particles: (N, d)
    weights: (N,) — must be normalized
    h: scalar bandwidth (will be broadcast as h_diag = h * ones(d))

    Returns: (Q,) log-density values
    """
    log_K = gaussian_log_kernel_scalar(x_query, particles, h)  # (Q, N)
    log_w = tf.math.log(weights + tf.constant(1e-30, dtype=weights.dtype))
    return tf.reduce_logsumexp(log_K + log_w[None, :], axis=-1)


def log_kde_uniform(x_query, particles, h):
    """log q_h(x_query) where q_h(x) = (1/N) sum_i K_h(x - x_i).

    Returns: (Q,)
    """
    N = tf.shape(particles)[0]
    log_K = gaussian_log_kernel_scalar(x_query, particles, h)  # (Q, N)
    log_uniform = -tf.math.log(tf.cast(N, x_query.dtype))
    return tf.reduce_logsumexp(log_K + log_uniform, axis=-1)


# ============================================================================
# Sampling from q_h (uniform-weighted KDE)
# ============================================================================

def sample_from_weighted_kde(particles, weights, h, n_samples, seed):
    """Sample n_samples points from p_h(x) = sum_i w_i K_h(x - x_i).

    Procedure:
    1. Pick particle index ~ Categorical(weights)
    2. Sample noise: xi ~ N(0, h^2 I)
    3. Output x_i + xi

    particles: (N, d)
    weights: (N,) normalized
    h: scalar bandwidth
    n_samples: int
    seed: (2,) int32 stateless seed

    Returns: (n_samples, d)
    """
    d = tf.shape(particles)[-1]
    keys = tf.random.experimental.stateless_split(seed, num=2)
    # Categorical sampling via Gumbel-max trick (stateless friendly)
    log_w = tf.math.log(weights + tf.constant(1e-30, dtype=weights.dtype))  # (N,)
    gumbel = -tf.math.log(
        -tf.math.log(
            tf.random.stateless_uniform(
                [n_samples, tf.shape(weights)[0]], seed=keys[0],
                dtype=particles.dtype, minval=1e-30, maxval=1.0,
            )
        )
    )
    idx = tf.argmax(log_w[None, :] + gumbel, axis=-1, output_type=tf.int32)
    centers = tf.gather(particles, idx)  # (n_samples, d)
    noise = tf.random.stateless_normal(
        [n_samples, d], seed=keys[1], dtype=particles.dtype
    )
    return centers + h * noise


def sample_from_uniform_kde(particles, h, n_samples, seed):
    """Sample n_samples points from q_h(x) = (1/N) sum_i K_h(x - x_i).

    Procedure:
    1. Pick particle index uniformly: i ~ Uniform{1..N}
    2. Sample noise: xi ~ N(0, h^2 I)
    3. Output x_i + xi

    particles: (N, d)
    h: scalar bandwidth
    n_samples: int
    seed: (2,) int32 stateless seed

    Returns: (n_samples, d)
    """
    N = tf.shape(particles)[0]
    d = tf.shape(particles)[-1]
    keys = tf.random.experimental.stateless_split(seed, num=2)
    # Pick indices uniformly
    idx = tf.random.stateless_uniform(
        [n_samples], seed=keys[0], minval=0, maxval=N, dtype=tf.int32
    )
    centers = tf.gather(particles, idx)  # (n_samples, d)
    # Add Gaussian noise
    noise = tf.random.stateless_normal(
        [n_samples, d], seed=keys[1], dtype=particles.dtype
    )
    return centers + h * noise
