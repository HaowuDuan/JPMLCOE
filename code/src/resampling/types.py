"""Resampling result types."""
from typing import NamedTuple, Optional
import tensorflow as tf


class ResampleResult(NamedTuple):
    """Standard return type for all resampling methods.

    Fields:
        particles: Resampled particle positions, shape (N, state_dim)
        weights: Post-resampling weights, shape (N,)
        ancestor_indices: Integer tensor of ancestor indices, shape (N,).
            For index-based methods (systematic, soft), this maps each new
            particle to its ancestor: new_particle[i] = old_particles[indices[i]].
            None for transport-based methods (OT entropy).
        transport_matrix: Transport matrix T, shape (N, N).
            For OT-based methods, resampled_particles = T @ old_particles.
            None for index-based methods.
    """
    particles: tf.Tensor
    weights: tf.Tensor
    ancestor_indices: Optional[tf.Tensor] = None
    transport_matrix: Optional[tf.Tensor] = None
