"""Map-based resampler using a trained neural operator.

The neural operator is trained separately under code/neural_operator/. This
module is the inference-side bridge: it accepts an already-built model that
exposes the duck-type interface

    c = model.encode(particles, weights, h_kde=h)
    T_x, J_x = model.transport_with_jacobian(x, c)

and wraps it in the standard ResampleResult interface used by the filters.

The model is loaded once at filter construction time (NOT inside the traced
call), then `__call__` runs a single forward pass to produce mapped particles
and per-particle Jacobians. Output weights are uniform.
"""

import tensorflow as tf

from .types import ResampleResult


def _silverman_bandwidth_scalar(particles, weights, scale=1.0):
    """Inlined Silverman scalar bandwidth.

    Kept self-contained here so this module has no import dependency on the
    neural_operator training package. Must match the formula used during
    training (see code/neural_operator/src/kde.py).
    """
    d = tf.cast(tf.shape(particles)[-1], particles.dtype)
    n_eff = 1.0 / tf.reduce_sum(weights ** 2)
    mu = tf.reduce_sum(weights[:, None] * particles, axis=0)
    centered = particles - mu[None, :]
    cov = tf.einsum('n,ni,nj->ij', weights, centered, centered)
    std = tf.reduce_mean(tf.sqrt(tf.linalg.diag_part(cov)))
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    return scale * factor * std * tf.pow(n_eff, -1.0 / (d + 4.0))


class NeuralOperatorResampler:
    """Map-based resampler driven by a trained neural operator.

    Returns mapped particles, uniform weights, and per-particle Jacobians
    suitable for LEDH local covariance transport.

    Use as:
        resampler = NeuralOperatorResampler(model)
        result = resampler(particles, weights, seed)
    """

    # Marker so callers can branch without isinstance() if they prefer
    returns_local_jacobians = True

    def __init__(self, model, h_scale: float = 1.0):
        """
        Args:
            model: A trained neural operator exposing
                .encode(particles, weights, h_kde=h)
                .transport_with_jacobian(x, c)
            h_scale: scalar multiplier on the Silverman bandwidth used at
                inference. 1.0 matches the final h_scale used during training.
        """
        self.model = model
        self.h_scale = h_scale
        # Freeze: resampler should not update weights at inference time
        if hasattr(model, 'trainable'):
            model.trainable = False

    def __call__(self, particles, weights, seed=None):
        """Apply the neural operator to one weighted cloud.

        Args:
            particles: (N, d) tensor
            weights: (N,) normalized
            seed: ignored (kept for API parity with stochastic resamplers)

        Returns:
            ResampleResult with:
              particles = T(x_i),
              weights = uniform,
              local_jacobians = J(x_i),
              transport_matrix = None,
              ancestor_indices = None.
        """
        del seed  # deterministic forward pass

        h = _silverman_bandwidth_scalar(particles, weights, scale=self.h_scale)
        c = self.model.encode(particles, weights, h_kde=h)
        T_x, J_x = self.model.transport_with_jacobian(particles, c)

        N = tf.shape(particles)[0]
        N_float = tf.cast(N, particles.dtype)
        uniform_weights = tf.ones(N, dtype=particles.dtype) / N_float

        return ResampleResult(
            particles=T_x,
            weights=uniform_weights,
            ancestor_indices=None,
            transport_matrix=None,
            local_jacobians=J_x,
        )
