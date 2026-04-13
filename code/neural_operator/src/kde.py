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


def particle_resolution_bandwidth_scalar(particles, scale: float = 5.0):
    """Bandwidth as a multiple of the mean nearest-neighbor distance.

    Independent of weights, independent of any density-estimation heuristic.
    Set by the actual particle spacing, which is the resolution at which
    the cloud can be smoothed without blurring out structure that the
    particles are physically able to represent.

    For 200 equally-spaced particles on [-5, 5], the mean nearest-neighbor
    distance is 10 / 199 ≈ 0.05, so `scale=5` gives `h ≈ 0.25`.

    For random particle clouds, the mean nearest-neighbor distance is the
    natural local resolution of the cloud, varying smoothly with cloud
    geometry but completely decoupled from how the weights happen to be
    distributed.

    Args:
        particles: (N, d) tensor.
        scale: multiplier on the mean nearest-neighbor distance. Default 5.

    Returns:
        scalar bandwidth h.
    """
    # Pairwise squared distances, mask the diagonal so a particle isn't its
    # own nearest neighbor.
    diff = particles[:, None, :] - particles[None, :, :]      # (N, N, d)
    sq_dist = tf.reduce_sum(diff * diff, axis=-1)             # (N, N)
    N = tf.shape(particles)[0]
    inf_diag = tf.linalg.diag(
        tf.fill([N], tf.constant(float('inf'), dtype=particles.dtype))
    )
    sq_dist = sq_dist + inf_diag
    nn_dist = tf.sqrt(tf.reduce_min(sq_dist, axis=-1))        # (N,)
    return scale * tf.reduce_mean(nn_dist)


def compute_bandwidth_scalar(particles, weights=None, policy: str = 'resolution',
                              scale: float = 1.0):
    """Dispatcher for the scalar KDE bandwidth.

    Two policies are supported:

    - 'resolution' (default): weight-independent bandwidth proportional to
      the mean nearest-neighbor particle distance. Right tool when the
      particles are not i.i.d. samples from a smooth density (e.g. particle
      filter outputs, equally-spaced grids, two-different-weight schemes on
      the same particles). Used by the neural operator path.

    - 'silverman': classical Silverman's rule using the unweighted source
      cloud spread and raw N. Right tool for density estimation when you
      believe the particles are i.i.d. samples from a smooth density and
      you want a good density estimate.

    The `scale` argument is an external multiplier on top of the policy's
    own default — used by the bandwidth annealing schedule, where
    `scale = h_scale` ranges over the schedule.

    Args:
        particles: (N, d) tensor.
        weights: (N,) tensor — only used by the silverman policy. Ignored
            by the resolution policy.
        policy: 'resolution' or 'silverman'.
        scale: external annealing multiplier (default 1.0 = no annealing).

    Returns:
        scalar bandwidth h.
    """
    if policy == 'resolution':
        # 5 * mean nearest-neighbor distance is the default for the
        # resolution policy; the external `scale` multiplies that base.
        return particle_resolution_bandwidth_scalar(particles, scale=5.0 * scale)
    elif policy == 'silverman':
        return silverman_bandwidth_scalar(particles, weights, scale=scale)
    else:
        raise ValueError(
            f"Unknown bandwidth policy {policy!r}. "
            f"Choose 'resolution' or 'silverman'."
        )


def silverman_bandwidth_scalar(particles, weights, scale: float = 1.0):
    """Silverman's rule: scalar bandwidth from the unweighted source cloud.

    h = scale * (4/(d+2))^(1/(d+4)) * std_unweighted * N^(-1/(d+4))

    The bandwidth is intentionally computed from the **unweighted** particle
    spread and the **raw** particle count, not from the weighted covariance
    and N_eff. The same bandwidth is used to smooth both the source KDE
    `p_h(x) = (1/N) sum_i K_h(x - x_i)` (uniform-weighted) and the target
    KDE `q_h(x) = sum_i w_i K_h(x - x_i)` (weighted), so it should describe
    the geometry of the particle cloud, not the weight skew.

    The `weights` argument is kept in the signature for backward
    compatibility but is intentionally unused — see history at
    `code/neural_operator/issues_to_be_addressed/2_quality_test.md` §7d for
    why the previous "weighted Silverman" form caused training instability.

    The `scale` parameter lets you anneal: scale=1.0 starts (default Silverman),
    scale=0.5 makes it half, etc.
    """
    del weights  # intentionally unused; bandwidth is a geometric property of the cloud
    d = tf.cast(tf.shape(particles)[-1], particles.dtype)
    N = tf.cast(tf.shape(particles)[0], particles.dtype)
    mu = tf.reduce_mean(particles, axis=0)
    centered = particles - mu[None, :]
    cov = tf.matmul(centered, centered, transpose_a=True) / N
    std = tf.reduce_mean(tf.sqrt(tf.linalg.diag_part(cov)))
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    h = scale * factor * std * tf.pow(N, -1.0 / (d + 4.0))
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
