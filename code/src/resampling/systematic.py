"""Systematic resampling for particle filters (TensorFlow)."""
import tensorflow as tf
from .types import ResampleResult


def systematic_resample(particles: tf.Tensor, weights: tf.Tensor,
                       seed: tf.Tensor) -> ResampleResult:
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
        ResampleResult with particles, uniform weights, and ancestor indices
    """
    N = tf.shape(particles)[0]
    N_float = tf.cast(N, weights.dtype)

    # Compute cumulative sum of weights
    cumsum = tf.cumsum(weights)

    # Generate systematic samples: u + k/N for k = 0, ..., N-1
    # where u ~ Uniform[0, 1/N]
    u = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0/N_float, dtype=weights.dtype)
    u_vals = u + tf.range(N, dtype=weights.dtype) / N_float

    # Find indices: for each u_val, find the smallest index where cumsum[idx] >= u_val
    # Using searchsorted (binary search)
    indices = tf.searchsorted(cumsum, u_vals, side='right')

    # Clip indices to valid range [0, N-1]
    indices = tf.clip_by_value(indices, 0, N - 1)

    # Resample particles
    resampled_particles = tf.gather(particles, indices)

    # Uniform weights after resampling
    uniform_weights = tf.ones(N, dtype=weights.dtype) / N_float

    return ResampleResult(
        particles=resampled_particles,
        weights=uniform_weights,
        ancestor_indices=indices,
        transport_matrix=None,
    )
