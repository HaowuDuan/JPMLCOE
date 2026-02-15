"""LEDH Invertible Particle Flow Filter with bimodal sign-flip correction.

Extends LEDHParticleFlowFilter with a post-flow Metropolis sign-flip step
for models whose observation function has discrete symmetry (e.g. y = x^2/20).

The standard LEDH flow derives from a Gaussian (unimodal) posterior approximation.
When the true posterior is bimodal (e.g. both +x and -x produce the same observation),
the flow herds all particles to a single mode. The sign-flip correction exploits the
transition dynamics to stochastically reassign particles to the correct mode.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Callable, Dict, Any

from .ledh_invertible import LEDHParticleFlowFilter
from ...core.model_base import StateSpaceModel
from ...utils.flow_params import compute_flow_params_batch
from ...utils.distributions import compute_flow_weights
from ..kalman.batched_ekf import batched_ekf_update
from ...resampling.diagnosis import effective_sample_size as ess_tf


class LEDHInvertibleBimodal(LEDHParticleFlowFilter):
    """
    LEDH Invertible with post-flow sign-flip correction for bimodal posteriors.

    After the standard LEDH flow, each particle independently proposes flipping
    its sign. The flip is accepted with probability:

        flip_prob_i = p(-eta_1_i | x_{k-1,i}) / [p(eta_1_i | x_{k-1,i}) + p(-eta_1_i | x_{k-1,i})]

    This exploits the fact that while h(x) = h(-x) (observation symmetry),
    the transition f(x,t) is sign-sensitive, carrying information about the
    correct mode.

    The flip happens before weight computation, so compute_flow_weights
    automatically accounts for the flipped positions.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        n_lambda_steps: int = 29,
        regularization: float = 1e-8,
        resample_threshold: float = 0.5,
        resampling_method: Optional[Callable] = None,
        resampling_config: Optional[Dict[str, Any]] = None,
        weight_clip_range: Optional[float] = None,
        debug_mode: bool = False,
        flip_fraction: float = 1.0,
        **filter_kwargs
    ):
        """
        Args:
            All args from LEDHParticleFlowFilter, plus:
            flip_fraction: Fraction of particles eligible for sign-flip (0.0 to 1.0).
                          1.0 = all particles can flip (default).
                          0.5 = only half the particles are candidates.
                          Reducing this adds a conservative bias toward the flow's mode.
        """
        super().__init__(
            model=model,
            n_particles=n_particles,
            n_lambda_steps=n_lambda_steps,
            regularization=regularization,
            resample_threshold=resample_threshold,
            resampling_method=resampling_method,
            resampling_config=resampling_config,
            weight_clip_range=weight_clip_range,
            debug_mode=debug_mode,
            **filter_kwargs
        )
        self.flip_fraction = flip_fraction

    def update(self, y: tf.Tensor):
        """Update step: standard LEDH flow + sign-flip correction + weight computation."""
        R = self.model.observation_noise_cov

        eta_1 = self.eta_0.value()
        eta_bar = self.eta_bar_0.value()

        lambda_val = tf.constant(0.0, dtype=self.dtype)
        log_theta = tf.zeros(self.n_particles, dtype=self.dtype)

        # Cache R_inv (constant across timesteps)
        if self.R_inv_cache is None:
            self.R_inv_cache = tf.linalg.inv(R)
        R_inv = self.R_inv_cache
        regularization_tf = tf.constant(self.regularization, dtype=self.dtype)

        particle_covs_tf = self.particle_covs.value()
        eta_bar_0_tf = self.eta_bar_0.value()

        I_sd = tf.eye(self.state_dim, dtype=self.dtype)

        # --- Flow loop (identical to parent) ---
        for j in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[j]
            lambda_val = lambda_val + d_lambda

            A_batch, b_batch = compute_flow_params_batch(
                self.model, eta_bar, lambda_val, y, particle_covs_tf,
                R, R_inv, eta_bar_0_tf, self.state_dim, regularization_tf
            )

            drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
            eta_bar = eta_bar + d_lambda * drift_bar

            drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
            eta_1 = eta_1 + d_lambda * drift_1

            M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
            log_det_M = tf.math.log(tf.abs(tf.linalg.det(M_batch)))
            log_theta = log_theta + log_det_M

        # Normalize Jacobians
        max_log_theta = tf.reduce_max(log_theta)
        log_theta = log_theta - max_log_theta
        theta = tf.exp(log_theta)

        # --- Sign-flip correction ---
        eta_1 = self._sign_flip(eta_1)

        self.particles.assign(eta_1)

        # --- Weight computation (identical to parent) ---
        weights_new = compute_flow_weights(
            eta_1=eta_1,
            eta_0=self.eta_0.value(),
            particles_prev=self.particles_prev.value(),
            observation=y,
            model=self.model,
            prev_weights=self.weights.value(),
            jacobians=theta,
            clip_range=self.weight_clip_range
        )
        self.weights.assign(weights_new)

        self.weights_history.append(self.weights.value())

        # Log-likelihood
        log_likelihood = self.model.log_observation_prob_batch(y, eta_1)
        max_ll = tf.reduce_max(log_likelihood)
        log_lik = max_ll + tf.math.log(tf.reduce_mean(tf.exp(log_likelihood - max_ll)))
        self.log_likelihoods.append(log_lik)

        # Update per-particle covariances via batched EKF
        _, cov_updated = batched_ekf_update(
            self.model, self.eta_bar_0.value(), self.particle_covs.value(), y
        )
        self.particle_covs.assign(cov_updated)

        # ESS and resampling
        ess = ess_tf(self.weights.value())
        self.ess_history.append(ess)

        if ess < self.resample_threshold * self.n_particles:
            self._resample()
            self.resampled_at.append(len(self.ess_history) - 1)
            particles_np = self.particles.numpy()
            n_unique = len(np.unique(particles_np, axis=0))
            self.n_unique_particles.append(n_unique)

    def _sign_flip(self, eta_1: tf.Tensor) -> tf.Tensor:
        """
        Propose sign-flipping each particle based on transition probability.

        For each particle i:
          p_plus  = N(eta_1_i; f(x_{k-1,i}), Q)
          p_minus = N(-eta_1_i; f(x_{k-1,i}), Q)
          flip_prob = p_minus / (p_plus + p_minus) = sigmoid(log_p_minus - log_p_plus)

        With probability flip_prob, flip eta_1_i -> -eta_1_i.
        """
        # Transition mean and covariance
        f_prev = self.model.state_transition_mean_batch(self.particles_prev.value())
        Q = self.model.state_transition_cov_batch(self.particles_prev.value())
        Q_inv = tf.linalg.inv(Q)

        # Log transition probabilities (up to shared normalizing constant)
        diff_plus = eta_1 - f_prev
        diff_minus = -eta_1 - f_prev

        log_p_plus = -0.5 * tf.reduce_sum(
            diff_plus * tf.linalg.matvec(Q_inv, diff_plus), axis=1
        )
        log_p_minus = -0.5 * tf.reduce_sum(
            diff_minus * tf.linalg.matvec(Q_inv, diff_minus), axis=1
        )

        # Flip probability via numerically stable sigmoid
        flip_prob = tf.nn.sigmoid(log_p_minus - log_p_plus)

        # Apply flip_fraction: only a subset of particles are candidates
        if self.flip_fraction < 1.0:
            # Mask out particles that aren't eligible
            seed_mask = tf.constant([self.seed_counter, 1], dtype=tf.int32)
            self.seed_counter += 1
            eligible = tf.cast(
                tf.random.stateless_uniform([self.n_particles], seed=seed_mask, dtype=self.dtype)
                < self.flip_fraction,
                self.dtype
            )
            flip_prob = flip_prob * eligible

        # Stochastic flip decision
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1
        u = tf.random.stateless_uniform([self.n_particles], seed=seed, dtype=self.dtype)
        flip_mask = tf.cast(u < flip_prob, self.dtype)

        # Flip sign: x -> x * (1 - 2*mask), i.e. x -> -x where mask=1
        eta_1 = eta_1 * (1.0 - 2.0 * flip_mask[:, tf.newaxis])

        return eta_1
