"""Conditional SMC with Bootstrap Particle Filter.

Simple CSMC for Particle Gibbs x-step. Uses bootstrap PF proposals
(transition density) with one reference particle pinned to the previous
iteration's trajectory.

This is Phase 1 — simpler and cheaper than LEDH CSMC. Use when:
- Validating the PGibbs framework
- Low-dimensional models where bootstrap PF has adequate ESS
- Speed matters (no flow loop, no Jacobians)

For high-dimensional models or strong nonlinearity, use
ledh_invertible_csmc.py (Phase 2).
"""

import tensorflow as tf
import numpy as np
import math
from typing import Optional

from ...core.model_base import StateSpaceModel
from ...utils.distributions import sample_particles_cholesky
from ...utils.linalg import safe_inv
from ...resampling import systematic_resample
from ...resampling.diagnosis import effective_sample_size as ess_tf


class BootstrapConditionalSMC:
    """
    Conditional SMC with bootstrap PF proposals for Particle Gibbs x-step.

    All N particles use the transition density as proposal.
    Weight = p(y_t | x_t^i) (likelihood only, since proposal = transition).
    One particle (the reference, index N-1) is pinned to the reference trajectory.
    Ancestor sampling optionally improves mixing.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 500,
        resample_threshold: float = 0.5,
        ancestor_sampling: bool = True,
    ):
        self.model = model
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.ancestor_sampling = ancestor_sampling

        self.ref_idx = n_particles - 1
        self.rng_key = tf.constant([42, 0], dtype=tf.int32)

    def _next_seed(self):
        keys = tf.random.experimental.stateless_split(self.rng_key, num=2)
        self.rng_key = keys[0]
        return keys[1]

    def run(
        self,
        observations: tf.Tensor,
        reference_trajectory: tf.Tensor,
        seed: int = 42,
    ) -> tf.Tensor:
        """
        Run Conditional SMC and return a sampled trajectory.

        Args:
            observations: (T, obs_dim)
            reference_trajectory: (T+1, state_dim) — x*_{0:T}
                                  where [0] is x_0 and [t] is x_t for t=1,...,T
            seed: random seed

        Returns:
            new_trajectory: (T+1, state_dim)
        """
        self.rng_key = tf.constant([seed, 0], dtype=tf.int32)

        T = observations.shape[0]
        N = self.n_particles
        sd = self.state_dim

        # Genealogy storage
        particles_history = []
        ancestors_history = []

        # --- Initialize (t=0) ---
        mu_0 = tf.cast(self.model.mu_0, self.dtype)
        Sigma_0 = tf.cast(self.model.Sigma_0, self.dtype)

        seed_init = self._next_seed()
        free_particles = sample_particles_cholesky(
            mu_0, Sigma_0, N - 1, sd, seed_init, self.dtype
        )
        x_ref_0 = tf.cast(tf.reshape(reference_trajectory[0], [1, sd]), self.dtype)
        particles = tf.concat([free_particles, x_ref_0], axis=0)
        weights = tf.ones(N, dtype=self.dtype) / tf.cast(N, self.dtype)

        particles_history.append(tf.identity(particles))

        # --- Filter loop (t=1,...,T) ---
        for t in range(T):
            if hasattr(self.model, 't'):
                self.model.t = t + 1

            y_t = observations[t]
            x_ref_t = reference_trajectory[t + 1]

            # 1. Resample (with reference protection)
            ess = ess_tf(weights)
            if ess < self.resample_threshold * N:
                particles, weights, ancestors = self._resample_csmc(
                    particles, weights, x_ref_t, t + 1
                )
            else:
                ancestors = tf.range(N, dtype=tf.int32)

            ancestors_history.append(ancestors)

            # 2. Propagate
            seed_prop = self._next_seed()
            particles_new = self.model.state_transition_batch(
                particles, seed_prop, t=t + 1
            )

            # Pin reference to x*_t
            particles = tf.concat([
                particles_new[:self.ref_idx],
                x_ref_t[tf.newaxis, :]
            ], axis=0)

            # 3. Weight: w_t ∝ w_{t-1} * p(y_t | x_t)
            log_obs = self.model.log_observation_prob_batch(y_t, particles)
            log_w = tf.math.log(
                tf.maximum(weights, tf.constant(1e-300, dtype=self.dtype))
            ) + log_obs
            weights = tf.nn.softmax(log_w)

            particles_history.append(tf.identity(particles))

        # --- Sample output trajectory ---
        trajectory = self._sample_trajectory(
            particles_history, ancestors_history, weights
        )

        return trajectory

    def _resample_csmc(self, particles, weights, x_ref_next, t):
        """Resample with reference protection + ancestor sampling."""
        N = self.n_particles
        seed = self._next_seed()

        result = systematic_resample(particles, weights, seed=seed)
        ancestors = (
            result.ancestor_indices
            if result.ancestor_indices is not None
            else tf.range(N, dtype=tf.int32)
        )

        # Reference's ancestor
        if self.ancestor_sampling and t > 0:
            ref_ancestor = self._ancestor_sample(particles, weights, x_ref_next, t)
        else:
            ref_ancestor = tf.constant(self.ref_idx, dtype=tf.int32)

        ancestors = tf.concat([ancestors[:self.ref_idx], [ref_ancestor]], axis=0)
        particles = tf.gather(particles, ancestors)
        weights = tf.ones(N, dtype=self.dtype) / tf.cast(N, self.dtype)

        return particles, weights, ancestors

    def _ancestor_sample(self, particles, weights, x_ref_next, t):
        """Ancestor sampling: w_j * p(x*_t | x_{t-1}^j).

        For each candidate ancestor j:
          log w_{t|T}^j = log w_{t-1}^j + log N(x*_t; f(x_{t-1}^j, t), Q)
        """
        Q = self.model.process_noise_cov
        Q_inv = safe_inv(Q)
        f_prev = self.model.state_transition_mean_batch(particles, t=t)

        diff = x_ref_next[tf.newaxis, :] - f_prev
        sd_float = tf.cast(self.state_dim, self.dtype)
        log_det_Q = tf.linalg.slogdet(Q)[1]
        mahal = tf.reduce_sum(diff * tf.linalg.matvec(Q_inv, diff), axis=-1)
        log_trans = -0.5 * (
            sd_float * tf.math.log(tf.constant(2.0 * math.pi, dtype=self.dtype))
            + log_det_Q
            + mahal
        )

        log_w = tf.math.log(
            tf.maximum(weights, tf.constant(1e-300, dtype=self.dtype))
        ) + log_trans

        seed = self._next_seed()
        idx = tf.random.stateless_categorical(log_w[tf.newaxis, :], 1, seed=seed)
        return tf.cast(idx[0, 0], tf.int32)

    def _sample_trajectory(self, particles_history, ancestors_history, final_weights):
        """Sample one trajectory by tracing ancestry backward.

        1. Sample k_T ~ Categorical(w_T)
        2. Trace back: k_t = ancestors[t+1][k_{t+1}]
        3. Extract: x_t = particles_history[t][k_t]
        """
        T = len(ancestors_history)

        seed = self._next_seed()
        k = tf.random.stateless_categorical(
            tf.math.log(final_weights)[tf.newaxis, :], 1, seed=seed
        )
        k = tf.cast(k[0, 0], tf.int32)

        trajectory_indices = [k]
        for t in reversed(range(T)):
            k = ancestors_history[t][k]
            trajectory_indices.insert(0, k)

        trajectory = tf.stack([
            particles_history[t][trajectory_indices[t]]
            for t in range(T + 1)
        ])

        return trajectory
