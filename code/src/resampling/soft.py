import tensorflow as tf

@tf.function
def soft_resample(particles: tf.Tensor, weights: tf.Tensor, alpha: float,
                 seed: tf.Tensor) -> tuple:
    """
    Soft resampling (TensorFlow implementation).

    Implements the differentiable resampling strategy from the PF-net paper.
    Instead of sampling from the original weights 'w', it samples from a
    mixture distribution 'q' (blend of 'w' and uniform). The resulting
    particles are re-weighted using importance sampling to maintain gradients.

    q(k) = alpha * w_k + (1 - alpha) * (1/N)
    w'_k = w_k / q(k)

    Args:
        particles: Particle positions of shape (N, state_dim)
        weights: Normalized weights of shape (N,) - must sum to 1
        alpha: Trade-off parameter (0.0 to 1.0)
               - alpha=1.0: Hard resampling (standard PF)
               - alpha=0.0: Uniform sampling (Importance Sampling)
        seed: Random seed tensor of shape (2,)

    Returns:
        Tuple of (resampled_particles, new_weights) where:
        - resampled_particles: (N, state_dim)
        - new_weights: (N,) Normalized importance weights for the next step
    """
    N = tf.shape(particles)[0]
    N_float = tf.cast(N, tf.float32)
    alpha_f = tf.cast(alpha, tf.float32)

    # 1. Compute mixture distribution q(k)
    # q mixes the current belief with a uniform distribution
    uniform_dist = tf.ones_like(weights) / N_float
    q_weights = alpha_f * weights + (1.0 - alpha_f) * uniform_dist

    # 2. Perform systematic sampling based on q_weights
    cumsum = tf.cumsum(q_weights)
    
    # Generate systematic samples: u + k/N
    u = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0/N_float)
    u_vals = u + tf.range(N, dtype=tf.float32) / N_float

    # Find indices using the q_weights cumulative sum
    indices = tf.searchsorted(cumsum, u_vals, side='right')
    indices = tf.clip_by_value(indices, 0, N - 1)

    # 3. Resample particles
    resampled_particles = tf.gather(particles, indices)

    # 4. Compute new importance weights: w' = p(k) / q(k)
    # We retrieve the original weights and q_values for the selected particles
    selected_weights = tf.gather(weights, indices)
    selected_q = tf.gather(q_weights, indices)

    # Avoid division by zero (though q is strictly positive if alpha < 1)
    epsilon = 1e-8
    new_weights = selected_weights / (selected_q + epsilon)

    # Normalize weights to sum to 1 for the next iteration
    new_weights = new_weights / tf.reduce_sum(new_weights)

    return resampled_particles, new_weights
