"""Systematic resampling for particle filters (TensorFlow)."""
import tensorflow as tf

@tf.function
def systematic_resample(particles: tf.Tensor, weights: tf.Tensor,
                       seed: tf.Tensor) -> tf.Tensor:
    """
    Systematic resampling (TensorFlow implementation).

    This is the preferred resampling method as it:
    - Has lower variance than multinomial resampling
    - Maintains particle diversity better
    - Is deterministic given the random seed

    Algorithm:
    1. Generate a single random number u ~ Uniform[0, 1/N]
    2. Create evenly spaced points: u + k/N for k = 0, ..., N-1
    3. Use cumulative sum of weights to select particles

    Args:
        particles: Particle positions of shape (N, state_dim)
        weights: Normalized weights of shape (N,) - must sum to 1
        seed: Random seed tensor of shape (2,) for stateless sampling

    Returns:
        Resampled particles of shape (N, state_dim)
    """
    N = tf.shape(particles)[0]
    N_float = tf.cast(N, tf.float32)

    # Compute cumulative sum of weights
    cumsum = tf.cumsum(weights)

    # Generate systematic samples: u + k/N for k = 0, ..., N-1
    # where u ~ Uniform[0, 1/N]
    u = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0/N_float)
    u_vals = u + tf.range(N, dtype=tf.float32) / N_float

    # Find indices: for each u_val, find the smallest index where cumsum[idx] >= u_val
    # Using searchsorted (binary search)
    indices = tf.searchsorted(cumsum, u_vals, side='right')

    # Clip indices to valid range [0, N-1]
    indices = tf.clip_by_value(indices, 0, N - 1)

    # Resample particles
    resampled_particles = tf.gather(particles, indices)

    return resampled_particles


@tf.function
def systematic_resample_with_weights(particles: tf.Tensor, weights: tf.Tensor,
                                    seed: tf.Tensor) -> tuple:
    """
    Systematic resampling that also returns uniform weights.

    Args:
        particles: Particle positions of shape (N, state_dim)
        weights: Normalized weights of shape (N,)
        seed: Random seed tensor of shape (2,)

    Returns:
        Tuple of (resampled_particles, uniform_weights) where:
        - resampled_particles: (N, state_dim)
        - uniform_weights: (N,) all equal to 1/N
    """
    resampled_particles = systematic_resample(particles, weights, seed)

    N = tf.shape(particles)[0]
    N_float = tf.cast(N, tf.float32)
    uniform_weights = tf.ones(N, dtype=tf.float32) / N_float

    return resampled_particles, uniform_weights
