"""Weighted DeepSets encoder for particle clouds.

Encodes (particles, weights) into a fixed-size context vector c that
parameterizes the conditional ConvexPotentialMap trunk.

Architecture:
1. Per-particle token: t_i = [x_i, log w_i, w_i]  (dimension d+2)
2. Per-particle MLP: h_i = phi(t_i)  (dimension hidden)
3. Weighted pool: m = sum_i w_i * h_i  (dimension hidden)
4. Concatenate global summary stats: weighted mean, weighted covariance, log N, ESS
5. Project to context vector c (dimension context_dim)

Permutation-invariant in the particle axis (DeepSets property).
Handles variable N by design.
"""

import tensorflow as tf

from kde import (
    effective_sample_size,
    weighted_mean,
    weighted_covariance,
)


class WeightedDeepSetsEncoder(tf.keras.Model):
    """Permutation-invariant encoder for weighted particle clouds.

    Args:
        in_dim: state dimension d
        hidden_dims: list of hidden layer sizes for per-particle MLP
        context_dim: output context vector dimension
    """

    def __init__(self, in_dim: int, hidden_dims=(64, 64), context_dim: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.context_dim = context_dim

        # Per-particle MLP: input is (d + 2), output is hidden_dims[-1]
        token_dim = in_dim + 2  # [x_i (d), log w_i (1), w_i (1)]
        layers = []
        prev = token_dim
        for h in hidden_dims:
            layers.append(tf.keras.layers.Dense(h, activation='relu'))
            prev = h
        self.particle_mlp = tf.keras.Sequential(layers)

        # Summary stat dim: weighted mean (d) + flat upper-triangular cov (d*(d+1)/2)
        # + log N (1) + log ESS (1) + log h (1)
        summary_dim = in_dim + (in_dim * (in_dim + 1)) // 2 + 3

        # Final projection: pooled (hidden) + summary (summary_dim) -> context
        self.output_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(context_dim, activation='relu'),
            tf.keras.layers.Dense(context_dim),
        ])

    def call(self, particles, weights, h_kde=None):
        """Encode (particles, weights) into context vector.

        Args:
            particles: (N, d) tensor
            weights: (N,) tensor, normalized to sum to 1
            h_kde: optional scalar bandwidth (added to context as info)

        Returns:
            c: (context_dim,) context vector
        """
        N = tf.cast(tf.shape(particles)[0], particles.dtype)
        log_weights = tf.math.log(weights + tf.constant(1e-30, dtype=weights.dtype))

        # Build per-particle tokens: [x_i, log w_i, w_i]
        tokens = tf.concat([
            particles,
            log_weights[:, None],
            weights[:, None],
        ], axis=-1)  # (N, d+2)

        # Per-particle MLP
        h_features = self.particle_mlp(tokens)  # (N, hidden)

        # Weighted pool
        pooled = tf.reduce_sum(weights[:, None] * h_features, axis=0)  # (hidden,)

        # Summary stats
        mu = weighted_mean(particles, weights)  # (d,)
        cov = weighted_covariance(particles, weights)  # (d, d)
        # Upper-triangular flatten (including diagonal)
        d = self.in_dim
        triu_indices = tf.constant(
            [(i, j) for i in range(d) for j in range(i, d)], dtype=tf.int32
        )
        cov_flat = tf.gather_nd(cov, triu_indices)  # (d*(d+1)/2,)

        log_N = tf.math.log(N)
        ess = effective_sample_size(weights)
        log_ess = tf.math.log(ess + tf.constant(1e-10, dtype=particles.dtype))
        if h_kde is None:
            h_kde = tf.constant(0.0, dtype=particles.dtype)
        log_h = tf.math.log(tf.maximum(h_kde, tf.constant(1e-10, dtype=particles.dtype)))

        summary = tf.concat([
            mu,
            cov_flat,
            tf.stack([log_N, log_ess, log_h]),
        ], axis=0)  # (summary_dim,)

        # Concatenate pooled features and summary
        combined = tf.concat([pooled, summary], axis=0)  # (hidden + summary_dim,)

        # Project to context
        c = self.output_proj(combined[None, :])[0]  # (context_dim,)
        return c
