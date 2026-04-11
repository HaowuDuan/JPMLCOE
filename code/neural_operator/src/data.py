"""Synthetic data generation for training the neural operator.

Generates random weighted particle clouds with varied:
- Spread (covariance scale)
- Multimodality (mixture components)
- ESS (skewness of weights)

Used for online training data.
"""

import tensorflow as tf
import math


def sample_random_cloud(n_particles, d, seed):
    """Sample one (particles, weights) example.

    Procedure:
    1. Sample number of mixture components (1, 2, or 3)
    2. Sample component means and per-dim stds
    3. Assign each particle to a component, sample its position
    4. Build position-correlated log-weights:
        log_w_total = log_w_pos + log_w_comp + log_w_noise
       where
        log_w_pos  = beta^T (x - mu_global)        (linear likelihood-style tilt)
        log_w_comp = bias per mixture component    (mode-level mass bias)
        log_w_noise = log Gamma(alpha)             (Dirichlet noise on top)
    5. Normalize via softmax-style stable exponentiation.

    The position-correlated terms are essential — particle filter weights come
    from a likelihood that depends on x, so the operator must learn how
    weights and positions interact. Pure independent Dirichlet noise (the old
    behavior) does not exercise this regime and the trained map collapses to
    near-identity on structured clouds.

    Returns: particles (N, d), weights (N,) normalized to sum to 1
    """
    keys = tf.random.experimental.stateless_split(seed, num=9)

    # 1. Number of mixture components: 1, 2, or 3 (uniform)
    n_components = tf.random.stateless_uniform(
        [], seed=keys[0], minval=1, maxval=4, dtype=tf.int32
    )

    # 2. Component parameters (always sample 3, then mask via component_ids)
    component_means = 3.0 * tf.random.stateless_normal(
        [3, d], seed=keys[1], dtype=tf.float64
    )
    log_stds = tf.random.stateless_uniform(
        [3, d], seed=keys[2],
        minval=tf.constant(math.log(0.2), dtype=tf.float64),
        maxval=tf.constant(math.log(2.0), dtype=tf.float64),
        dtype=tf.float64,
    )
    component_stds = tf.exp(log_stds)

    # 3. Assign particles to active components and sample positions
    component_ids = tf.random.stateless_uniform(
        [n_particles], seed=keys[3], minval=0, maxval=n_components, dtype=tf.int32
    )
    chosen_means = tf.gather(component_means, component_ids)  # (N, d)
    chosen_stds = tf.gather(component_stds, component_ids)    # (N, d)
    z = tf.random.stateless_normal([n_particles, d], seed=keys[4], dtype=tf.float64)
    particles = chosen_means + chosen_stds * z

    # 4a. Linear likelihood-style tilt: log_w_pos = beta^T (x - mu_global)
    # Centering avoids degenerate over-tilt when the cloud is far from origin.
    mu_global = tf.reduce_mean(particles, axis=0)
    beta = 1.5 * tf.random.stateless_normal([d], seed=keys[5], dtype=tf.float64)
    log_w_pos = tf.linalg.matvec(particles - mu_global[None, :], beta)  # (N,)

    # 4b. Per-component weight bias (some modes carry more posterior mass)
    component_log_w = 2.0 * tf.random.stateless_normal(
        [3], seed=keys[6], dtype=tf.float64
    )
    log_w_comp = tf.gather(component_log_w, component_ids)  # (N,)

    # 4c. Dirichlet noise on top, with log-uniform alpha in [0.1, 5]
    log_alpha = tf.random.stateless_uniform(
        [], seed=keys[7],
        minval=tf.constant(math.log(0.1), dtype=tf.float64),
        maxval=tf.constant(math.log(5.0), dtype=tf.float64),
        dtype=tf.float64,
    )
    alpha = tf.exp(log_alpha)
    gammas = tf.random.stateless_gamma(
        shape=[n_particles], seed=keys[8], alpha=alpha, dtype=tf.float64
    )
    gammas = gammas + tf.constant(1e-30, dtype=tf.float64)  # safety against zeros
    log_w_noise = tf.math.log(gammas)

    # 5. Combine and normalize via stable softmax
    log_w_total = log_w_pos + log_w_comp + log_w_noise
    log_w_total = log_w_total - tf.reduce_max(log_w_total)
    weights_raw = tf.exp(log_w_total)
    weights = weights_raw / tf.reduce_sum(weights_raw)

    return particles, weights


def gaussian_to_gaussian_map_1d(x, mu_q, sigma_q, mu_p, sigma_p):
    """Analytic 1D OT map from N(mu_q, sigma_q^2) to N(mu_p, sigma_p^2).

    T(x) = mu_p + (sigma_p / sigma_q) * (x - mu_q)

    Args:
        x: (batch, 1)
    Returns:
        T(x): (batch, 1)
        J(x): (batch, 1, 1) — constant sigma_p/sigma_q
    """
    T = mu_p + (sigma_p / sigma_q) * (x - mu_q)
    J = tf.fill([tf.shape(x)[0], 1, 1], tf.cast(sigma_p / sigma_q, x.dtype))
    return T, J
