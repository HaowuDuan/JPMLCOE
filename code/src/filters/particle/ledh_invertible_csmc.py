"""Conditional SMC with LEDH Invertible Particle Flow.

CSMC variant of LEDHParticleFlowFilter for use in Particle Gibbs.

Key modifications from ledh_invertible.py:
1. Accepts a reference trajectory x*_{0:T} that one particle is pinned to
2. Reference particle skips the LEDH flow (stays at x*_t)
3. Reference particle always survives resampling
4. Ancestor sampling for better mixing (optional)
5. Stores full particle genealogy for trajectory reconstruction

Weight formulas:
  - Free particles (i = 1,...,N-1):
      w_t^i ∝ w_{t-1}^i * p(y_t | η_1^i) * p(η_1^i | x_{t-1}^{a_i}) / p(η_0^i | x_{t-1}^{a_i}) * |det J^i|
  - Reference particle (i = N):
      w_t^N ∝ w_{t-1}^N * p(y_t | x_t^*)
    No flow → no Jacobian, no transition density ratio.
"""

import tensorflow as tf
import numpy as np
import math
from typing import Optional, Dict, Any

from ...core.model_base import StateSpaceModel
from ..kalman.batched_ekf import batched_ekf_predict, batched_ekf_update
from ...utils.flow_params import compute_flow_params_batch
from ...utils.distributions import compute_flow_weights, sample_particles_cholesky
from ...utils.linalg import safe_log_abs_det, safe_inv
from ...utils.constants import FlowScheduleConfig
from ...utils.resampling_config import resolve_resampling
from ...resampling.diagnosis import effective_sample_size as ess_tf


class LEDHConditionalSMC:
    """
    Conditional SMC with LEDH particle flow for Particle Gibbs x-step.

    N-1 "free" particles use LEDH flow (better proposals).
    1 "reference" particle is pinned to the reference trajectory (ensures ergodicity).
    Ancestor sampling optionally resamples the reference's history for better mixing.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 500,
        n_lambda_steps: int = 15,
        regularization: float = 1e-8,
        resample_threshold: float = 0.5,
        resampling_method=None,
        resampling_config: Optional[Dict[str, Any]] = None,
        weight_clip_range: Optional[float] = None,
        ancestor_sampling: bool = True,
        flow_config: FlowScheduleConfig = FlowScheduleConfig(),
    ):
        self.model = model
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.regularization = regularization
        self.resample_threshold = resample_threshold
        self.weight_clip_range = (-weight_clip_range, weight_clip_range) if weight_clip_range is not None else None
        self.ancestor_sampling = ancestor_sampling
        self.flow_config = flow_config

        # Resolve resampling method and config
        self.resampling_method, self.resampling_method_name, self.resampling_config = (
            resolve_resampling(resampling_method, resampling_config)
        )

        # Reference particle is always the last one (index N-1)
        self.ref_idx = n_particles - 1

        # RNG
        self.rng_key = tf.constant([42, 0], dtype=tf.int32)

        # Cache
        self.R_inv_cache = None

        self._generate_lambda_steps()

    def _next_seed(self):
        keys = tf.random.experimental.stateless_split(self.rng_key, num=2)
        self.rng_key = keys[0]
        return keys[1]

    def _generate_lambda_steps(self):
        q = self.flow_config.geometric_ratio
        epsilon_1 = (1 - q) / (1 - q**self.n_lambda_steps)
        lambda_steps_np = epsilon_1 * q**np.arange(self.n_lambda_steps)
        self.lambda_steps = tf.constant(lambda_steps_np, dtype=self.dtype)

    def run(
        self,
        observations: tf.Tensor,
        reference_trajectory: tf.Tensor,
        seed: int = 42,
    ) -> tf.Tensor:
        """
        Run Conditional SMC with LEDH flow and return a sampled trajectory.

        Args:
            observations: (T, obs_dim) observations y_{1:T}
            reference_trajectory: (T+1, state_dim) reference trajectory x*_{0:T}
                                  where [0] is x_0 and [t] is x_t for t=1,...,T
            seed: random seed

        Returns:
            new_trajectory: (T+1, state_dim) sampled trajectory from CSMC output
        """
        self.rng_key = tf.constant([seed, 0], dtype=tf.int32)
        self.R_inv_cache = None

        T = observations.shape[0]
        N = self.n_particles
        sd = self.state_dim

        # Storage for genealogy (also accessible to filter() method)
        particles_history = []  # List of (N, sd) tensors
        ancestors_history = []  # List of (N,) int tensors
        weights_history = []    # List of (N,) weight tensors per observation
        ess_history = []        # ESS per observation
        log_likelihoods = []    # Log-likelihood per observation

        # --- Initialize (t=0) ---
        particles, weights, particle_covs = self._initialize(reference_trajectory[0])
        particles_history.append(tf.identity(particles))

        # --- Filter loop (t=1,...,T) ---
        for t in range(T):
            if hasattr(self.model, 't'):
                self.model.t = t + 1

            y_t = observations[t]
            x_ref_t = reference_trajectory[t + 1]  # x*_t (1-indexed in trajectory)

            # 1. Resample (with reference protection + optional ancestor sampling)
            ess = ess_tf(weights)
            if ess < self.resample_threshold * N:
                particles, weights, particle_covs, ancestors = self._resample_csmc(
                    particles, weights, particle_covs,
                    x_ref_t, reference_trajectory, t + 1
                )
            else:
                ancestors = tf.range(N, dtype=tf.int32)

            ancestors_history.append(ancestors)

            # 2. Predict (EKF + stochastic transition)
            particles_prev = tf.identity(particles)
            eta_bar_0, particle_covs = self._predict(particles, particle_covs, t + 1)

            # Stochastic transition for free particles
            seed = self._next_seed()
            eta_0_all = self.model.state_transition_batch(particles_prev, seed, t=t + 1)

            # Pin reference to x*_t
            eta_0 = tf.concat([eta_0_all[:self.ref_idx], x_ref_t[tf.newaxis, :]], axis=0)

            # eta_bar_0 for reference: use model mean from reference's ancestor
            # (needed for flow params even though we won't flow the reference)

            # 3. LEDH Flow (free particles only)
            eta_1, log_theta_free = self._flow_free_particles(
                eta_0, eta_bar_0, particle_covs, y_t
            )

            # 4. Weight and log-likelihood
            weights, log_lik_t = self._compute_weights(
                eta_1, eta_0, particles_prev, y_t, weights,
                log_theta_free, x_ref_t
            )

            # 5. EKF update for covariances
            _, cov_updated = batched_ekf_update(
                self.model, eta_bar_0, particle_covs, y_t
            )
            particle_covs = cov_updated

            # Store particles and diagnostics
            particles = eta_1
            particles_history.append(tf.identity(particles))
            weights_history.append(tf.identity(weights))
            ess_history.append(float(ess_tf(weights).numpy()))

            # Log-likelihood from importance-weighted estimate
            log_likelihoods.append(float(log_lik_t.numpy()))

        # Store as instance attributes (accessible to filter())
        self._particles_history = particles_history
        self._ancestors_history = ancestors_history
        self._weights_history = weights_history
        self._ess_history = ess_history
        self._log_likelihoods = log_likelihoods

        # --- Sample output trajectory ---
        trajectory = self._sample_trajectory(
            particles_history, ancestors_history, weights
        )

        return trajectory

    def _initialize(self, x_ref_0):
        """Initialize particles: N-1 from prior, 1 pinned to reference."""
        N = self.n_particles
        sd = self.state_dim

        # Initial distribution
        mu_0 = tf.cast(self.model.mu_0, self.dtype)
        Sigma_0 = tf.cast(self.model.Sigma_0, self.dtype)

        # Sample N-1 free particles
        seed = self._next_seed()
        free_particles = sample_particles_cholesky(
            mu_0, Sigma_0, N - 1, sd, seed, self.dtype
        )

        # Pin reference to x*_0
        x_ref_0 = tf.cast(tf.reshape(x_ref_0, [1, sd]), self.dtype)
        particles = tf.concat([free_particles, x_ref_0], axis=0)

        # Uniform weights
        weights = tf.ones(N, dtype=self.dtype) / tf.cast(N, self.dtype)

        # Per-particle covariances (all start identical)
        particle_covs = tf.tile(
            tf.expand_dims(Sigma_0, 0), [N, 1, 1]
        )

        return particles, weights, particle_covs

    def _predict(self, particles, particle_covs, t):
        """Batched EKF predict for covariances + deterministic means."""
        eta_bar_0, cov_pred = batched_ekf_predict(
            self.model, particles, particle_covs
        )
        return eta_bar_0, cov_pred

    def _flow_free_particles(self, eta_0, eta_bar_0, particle_covs, y):
        """Apply LEDH flow to free particles only. Reference stays pinned."""
        N = self.n_particles
        R = self.model.observation_noise_cov

        if self.R_inv_cache is None:
            self.R_inv_cache = safe_inv(R)
        R_inv = self.R_inv_cache

        regularization_tf = tf.constant(self.regularization, dtype=self.dtype)
        I_sd = tf.eye(self.state_dim, dtype=self.dtype)

        # Split: free particles vs reference
        eta_1_free = eta_0[:self.ref_idx]  # (N-1, sd)
        eta_bar_free = eta_bar_0[:self.ref_idx]
        covs_free = particle_covs[:self.ref_idx]
        eta_bar_0_free = tf.identity(eta_bar_free)  # Save initial for flow params

        lambda_val = tf.constant(0.0, dtype=self.dtype)
        log_theta = tf.zeros(self.ref_idx, dtype=self.dtype)

        for j in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[j]
            lambda_val = lambda_val + d_lambda

            # Flow params for free particles only
            A_batch, b_batch = compute_flow_params_batch(
                self.model, eta_bar_free, lambda_val, y, covs_free,
                R, R_inv, eta_bar_0_free, self.state_dim, regularization_tf
            )

            # Euler step for eta_bar (deterministic)
            drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar_free) + b_batch
            eta_bar_free = eta_bar_free + d_lambda * drift_bar

            # Euler step for eta_1 (stochastic particles)
            drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1_free) + b_batch
            eta_1_free = eta_1_free + d_lambda * drift_1

            # Jacobian accumulation
            M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
            log_det_M = safe_log_abs_det(M_batch)
            log_theta = log_theta + log_det_M

        # Normalize Jacobians
        max_log_theta = tf.reduce_max(log_theta)
        log_theta = log_theta - max_log_theta

        # Reassemble: free particles (flowed) + reference (unchanged)
        x_ref = eta_0[self.ref_idx:self.ref_idx + 1]  # (1, sd)
        eta_1 = tf.concat([eta_1_free, x_ref], axis=0)

        return eta_1, log_theta

    def _compute_weights(self, eta_1, eta_0, particles_prev, y, prev_weights,
                         log_theta_free, x_ref_t):
        """Compute weights with heterogeneous proposals.

        Free particles: full LEDH weight (likelihood * transition ratio * Jacobian)
        Reference: likelihood only (no flow, no Jacobian)

        Returns:
            Tuple of (normalized_weights, log_marginal_lik)
        """
        N = self.n_particles

        # --- Free particles: use compute_flow_weights ---
        theta_free = tf.exp(log_theta_free)

        free_weights, free_log_lik = compute_flow_weights(
            eta_1=eta_1[:self.ref_idx],
            eta_0=eta_0[:self.ref_idx],
            particles_prev=particles_prev[:self.ref_idx],
            observation=y,
            model=self.model,
            prev_weights=prev_weights[:self.ref_idx],
            jacobians=theta_free,
            clip_range=self.weight_clip_range
        )

        # --- Reference particle: likelihood only ---
        log_obs_ref = self.model.log_observation_prob_batch(
            y, x_ref_t[tf.newaxis, :]
        )  # (1,)
        log_w_ref = tf.math.log(
            tf.maximum(prev_weights[self.ref_idx:], tf.constant(1e-300, dtype=self.dtype))
        ) + log_obs_ref

        # --- Log-likelihood: logsumexp over all unnormalized weights ---
        # free_log_lik = log(Σ_free w̃_i), log_w_ref = log(w̃_ref)
        # total = log(Σ_free w̃_i + w̃_ref)
        log_lik = tf.reduce_logsumexp(
            tf.stack([free_log_lik, log_w_ref[0]])
        )

        # --- Combine and normalize ---
        # Free weights are already normalized among themselves; we need to
        # renormalize including the reference.
        # Recompute in log space for proper normalization.
        # Use the unnormalized free weights instead.
        log_free = tf.math.log(
            tf.maximum(free_weights, tf.constant(1e-300, dtype=self.dtype))
        )
        log_all = tf.concat([log_free, log_w_ref], axis=0)
        weights = tf.nn.softmax(log_all)

        # Safety check
        is_finite = tf.reduce_all(tf.math.is_finite(weights))
        uniform = tf.ones(N, dtype=self.dtype) / tf.cast(N, self.dtype)
        weights = tf.cond(is_finite, lambda: weights, lambda: uniform)

        return weights, log_lik

    def _resample_csmc(self, particles, weights, particle_covs,
                       x_ref_next, reference_trajectory, t):
        """Resample with reference protection and optional ancestor sampling.

        The reference particle (index N-1) always survives.
        With ancestor sampling, we resample the reference's ancestor from:
          w_j * p(x*_t | x_{t-1}^j, theta)
        """
        N = self.n_particles
        seed = self._next_seed()

        # Resample all N particles
        result = self.resampling_method(
            particles, weights, seed=seed, **self.resampling_config
        )

        # Get ancestor indices from ResampleResult
        if result.ancestor_indices is not None:
            ancestors = result.ancestor_indices
        elif result.transport_matrix is not None:
            ancestors = tf.cast(tf.argmax(result.transport_matrix, axis=1), tf.int32)
        else:
            ancestors = tf.range(N, dtype=tf.int32)

        # Force reference particle's ancestor
        if self.ancestor_sampling and t > 0:
            # Ancestor sampling: resample a_N from w_j * p(x*_t | x_{t-1}^j)
            ref_ancestor = self._ancestor_sample(
                particles, weights, x_ref_next, t
            )
        else:
            # Standard: reference's ancestor is itself
            ref_ancestor = tf.constant(self.ref_idx, dtype=tf.int32)

        # Replace last ancestor index
        ancestors = tf.concat([ancestors[:self.ref_idx], [ref_ancestor]], axis=0)

        # Gather resampled particles and covariances
        new_particles = tf.gather(particles, ancestors)
        new_covs = tf.gather(particle_covs, ancestors)

        # Uniform weights after resampling
        new_weights = tf.ones(N, dtype=self.dtype) / tf.cast(N, self.dtype)

        return new_particles, new_weights, new_covs, ancestors

    def _ancestor_sample(self, particles, weights, x_ref_next, t):
        """Compute ancestor sampling weights and sample.

        For each candidate ancestor j:
          w_{t|T}^j ∝ w_{t-1}^j * p(x*_t | x_{t-1}^j, theta)

        The transition density p(x*_t | x_{t-1}^j) = N(x*_t; f(x_{t-1}^j, t), Q)
        """
        N = self.n_particles

        # Transition means from all particles
        f_prev = self.model.state_transition_mean_batch(particles, t=t)  # (N, sd)

        # Transition covariance
        Q = self.model.process_noise_cov  # (sd, sd)
        Q_inv = safe_inv(Q)

        # log p(x*_t | x_{t-1}^j) = log N(x*_t; f(x_{t-1}^j), Q)
        diff = x_ref_next[tf.newaxis, :] - f_prev  # (N, sd)

        # For 1D: log p = -0.5 * (log(2*pi*Q) + diff^2/Q)
        # General: use Mahalanobis distance
        sd_float = tf.cast(self.state_dim, self.dtype)
        log_det_Q = tf.linalg.slogdet(Q)[1]
        # (N, sd) @ (sd, sd) -> (N, sd), then sum -> (N,)
        mahal = tf.reduce_sum(diff * tf.linalg.matvec(Q_inv, diff), axis=-1)
        log_trans = -0.5 * (sd_float * tf.math.log(
            tf.constant(2.0 * math.pi, dtype=self.dtype)
        ) + log_det_Q + mahal)

        # Ancestor sampling weights
        log_w = tf.math.log(
            tf.maximum(weights, tf.constant(1e-300, dtype=self.dtype))
        ) + log_trans

        # Sample
        seed = self._next_seed()
        idx = tf.random.stateless_categorical(
            log_w[tf.newaxis, :], 1, seed=seed
        )
        return tf.cast(idx[0, 0], tf.int32)

    def _sample_trajectory(self, particles_history, ancestors_history, final_weights):
        """Sample one trajectory from the CSMC output by tracing ancestry.

        1. Sample k_T ~ Categorical(w_T)
        2. Trace back: k_t = ancestors[t+1][k_{t+1}]
        3. Extract: x_t = particles_history[t][k_t]
        """
        T = len(ancestors_history)

        # Sample final particle index
        seed = self._next_seed()
        k = tf.random.stateless_categorical(
            tf.math.log(final_weights)[tf.newaxis, :], 1, seed=seed
        )
        k = tf.cast(k[0, 0], tf.int32)

        # Trace backward
        trajectory_indices = [k]
        for t in reversed(range(T)):
            k = ancestors_history[t][k]
            trajectory_indices.insert(0, k)

        # Extract trajectory (T+1 entries: x_0, x_1, ..., x_T)
        trajectory = tf.stack([
            particles_history[t][trajectory_indices[t]]
            for t in range(T + 1)
        ])

        return trajectory

    def _generate_prior_trajectory(self, T, seed):
        """Generate a reference trajectory by sampling from the prior.

        x_0 ~ p(x_0), x_t ~ p(x_t | x_{t-1}) for t = 1, ..., T
        """
        rng_key = tf.constant([seed, 0], dtype=tf.int32)
        keys = tf.random.experimental.stateless_split(rng_key, num=T + 1)

        x_0 = self.model.sample_initial_state(keys[0])
        traj_list = [x_0]
        x = x_0
        for t in range(T):
            if hasattr(self.model, 't'):
                self.model.t = t + 1
            x = self.model.sample_state_transition(x, keys[t + 1])
            traj_list.append(x)

        return tf.stack(traj_list)  # (T+1, state_dim)

    def filter(self, observations, random_seed=None, progress_callback=None):
        """Standalone filtering: run CSMC with a prior reference trajectory.

        Generates a reference trajectory from the prior, runs one CSMC sweep,
        and returns FilterResult with per-timestep weighted means and covariances.

        Compatible with run_experiment.py infrastructure.
        """
        from ...core.types import FilterResult

        if isinstance(observations, np.ndarray):
            obs_tf = tf.constant(observations, dtype=self.dtype)
        else:
            obs_tf = tf.cast(observations, self.dtype)

        T = obs_tf.shape[0]
        seed = random_seed if random_seed is not None else 42

        # Generate reference trajectory from prior
        ref_trajectory = self._generate_prior_trajectory(T, seed)

        # Run CSMC sweep (populates self._particles_history, self._weights_history)
        self.run(obs_tf, ref_trajectory, seed=seed + 1)

        # Compute per-timestep means and covariances from particle cloud
        means_list = []
        covs_list = []

        for t in range(T):
            # particles at time t+1 (after observing y_t), weights from y_t
            particles_t = self._particles_history[t + 1]  # (N, sd)
            weights_t = self._weights_history[t]           # (N,)

            # Weighted mean
            mean_t = tf.reduce_sum(
                weights_t[:, tf.newaxis] * particles_t, axis=0
            )

            # Weighted covariance
            diff = particles_t - mean_t[tf.newaxis, :]
            cov_t = tf.einsum(
                'ni,nj->ij',
                weights_t[:, tf.newaxis] * diff,
                diff
            )

            means_list.append(mean_t.numpy())
            covs_list.append(cov_t.numpy())

        means_np = np.stack(means_list)  # (T, sd)
        covs_np = np.stack(covs_list)    # (T, sd, sd)

        total_ll = sum(self._log_likelihoods) if self._log_likelihoods else None

        return FilterResult(
            means=means_np,
            covs=covs_np,
            log_likelihood=total_ll,
            log_likelihoods=np.array(self._log_likelihoods) if self._log_likelihoods else None,
            ess=np.array(self._ess_history) if self._ess_history else None,
            weights_history=None,
            resampled_at=None,
            metadata={
                'filter_type': 'LEDHConditionalSMC',
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'ancestor_sampling': self.ancestor_sampling,
                'resampling_method': self.resampling_method_name,
            }
        )
