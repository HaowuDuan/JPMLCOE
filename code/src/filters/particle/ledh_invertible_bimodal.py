"""LEDH Invertible Particle Flow Filter with look-ahead sign correction.

Extends LEDHParticleFlowFilter with a look-ahead sign correction step
for models whose observation function has discrete symmetry (e.g. y = x^2/20).

The standard LEDH flow derives from a Gaussian (unimodal) posterior approximation.
When the true posterior is bimodal (e.g. both +x and -x produce the same observation),
the flow herds all particles to a single mode. The look-ahead correction uses future
observations to determine the correct sign, exploiting the asymmetry of the transition
dynamics f(x,t) != -f(-x,t).
"""

import tensorflow as tf
import numpy as np
import time
from typing import Optional, Callable, Dict, Any

from .ledh_invertible import LEDHParticleFlowFilter
from ...core.model_base import StateSpaceModel
from ...core.types import FilterResult
from ...utils.flow_params import compute_flow_params_batch
from ...utils.distributions import compute_flow_weights
from ...utils.linalg import safe_log_abs_det, safe_inv
from ..kalman.batched_ekf import batched_ekf_update
from ..kalman.batched_ukf import batched_ukf_update
from ...resampling.diagnosis import effective_sample_size as ess_tf


class LEDHInvertibleBimodal(LEDHParticleFlowFilter):
    """
    LEDH Invertible with look-ahead sign correction for bimodal posteriors.

    After the standard LEDH flow, each particle's sign is evaluated by
    propagating +eta_1 and -eta_1 deterministically through K future timesteps
    and scoring against future observations. The sign that better predicts
    future observations is selected.

    This avoids the chicken-and-egg problem of the transition-based flip
    (which depends on particles_prev being correct) by using future
    observations as independent evidence.
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
        lookahead_steps: int = 1,
        **filter_kwargs
    ):
        """
        Args:
            All args from LEDHParticleFlowFilter, plus:
            lookahead_steps: Number of future observations to use for sign
                           correction (K). 0 = disabled, 1-3 recommended.
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
        self.lookahead_steps = lookahead_steps

    def filter(self, observations: np.ndarray,
               initial_mean: Optional[np.ndarray] = None,
               initial_cov: Optional[np.ndarray] = None,
               random_seed: Optional[int] = None,
               progress_callback: Optional[Callable[[int, int, float], None]] = None) -> FilterResult:
        """Run filter with look-ahead sign correction.

        Overrides parent to pass future observations and timestep index to update().
        """
        self.initialize(initial_mean, initial_cov, random_seed)
        T = len(observations)

        # Pre-convert observations to TF once
        obs_tf = tf.constant(observations, dtype=self.dtype)

        for t in range(T):
            t0 = time.perf_counter()

            # Set model time for Kitagawa's cos(1.2*t) term
            self.predict(t=t + 1)

            # Pass future observations for look-ahead (up to K steps)
            if self.lookahead_steps > 0 and t + 1 < T:
                y_future = obs_tf[t + 1: t + 1 + self.lookahead_steps]
            else:
                y_future = None
            self.update(obs_tf[t], y_future=y_future, current_t=t + 1)

            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
            if progress_callback is not None:
                progress_callback(t, T, time.perf_counter() - t0)

        resampling_rate = len(self.resampled_at) / T if T > 0 else 0.0

        # Convert accumulated TF tensors to numpy once
        means_np = tf.stack(self.means).numpy()
        covs_np = tf.stack(self.covs).numpy()
        log_liks_tf = tf.stack(self.log_likelihoods) if self.log_likelihoods else None
        ess_np = tf.stack(self.ess_history).numpy()
        weights_np = tf.stack(self.weights_history).numpy()

        return FilterResult(
            means=means_np,
            covs=covs_np,
            log_likelihood=float(tf.reduce_sum(log_liks_tf).numpy()) if log_liks_tf is not None else None,
            log_likelihoods=log_liks_tf.numpy() if log_liks_tf is not None else None,
            ess=ess_np,
            weights_history=weights_np,
            resampled_at=self.resampled_at,
            n_unique=np.array(self.n_unique_particles) if self.n_unique_particles else None,
            metadata={
                'filter_type': 'LEDHInvertibleBimodal',
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'resampling_method': self.resampling_method_name,
                'resampling_rate': resampling_rate,
                'lookahead_steps': self.lookahead_steps
            }
        )

    def update(self, y: tf.Tensor, y_future=None, current_t=None):
        """Update step: standard LEDH flow + look-ahead sign correction + weight computation."""
        R = self.model.observation_noise_cov

        eta_1 = self.eta_0.value()
        eta_bar = self.eta_bar_0.value()

        lambda_val = tf.constant(0.0, dtype=self.dtype)
        log_theta = tf.zeros(self.n_particles, dtype=self.dtype)

        # Cache R_inv (constant across timesteps)
        if self.R_inv_cache is None:
            self.R_inv_cache = safe_inv(R)
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
            log_det_M = safe_log_abs_det(M_batch)
            log_theta = log_theta + log_det_M

        # Normalize Jacobians
        max_log_theta = tf.reduce_max(log_theta)
        log_theta = log_theta - max_log_theta
        theta = tf.exp(log_theta)

        # --- Look-ahead sign correction ---
        if y_future is not None and tf.shape(y_future)[0] > 0:
            eta_1 = self._lookahead_sign_correction(eta_1, y_future, current_t)

        self.particles.assign(eta_1)

        # --- Weight computation (identical to parent) ---
        weights_new, log_lik = compute_flow_weights(
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

        # Log-likelihood from importance-weighted estimate
        self.log_likelihoods.append(log_lik + max_log_theta)

        # Update per-particle covariances via batched EKF/UKF
        if self.filter_type == 'ukf':
            _, cov_updated = batched_ukf_update(
                self.model, self.eta_bar_0.value(), self.particle_covs.value(), y,
                self._ukf_weights_mean, self._ukf_weights_cov,
                self._ukf_lambda, self.state_dim
            )
        else:
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

    def _lookahead_sign_correction(self, eta_1: tf.Tensor, y_future: tf.Tensor,
                                   current_t: int) -> tf.Tensor:
        """Score +eta_1 vs -eta_1 against future observations.

        Propagates both sign candidates deterministically through the transition
        dynamics for K steps and scores against actual future observations.
        The sign with better predictive score is selected stochastically.

        Args:
            eta_1: Flowed particles at lambda=1, shape (N, state_dim)
            y_future: Future observations, shape (K_avail, obs_dim)
            current_t: Current timestep index (1-based, as used by model.t)

        Returns:
            eta_1 with signs corrected, shape (N, state_dim)
        """
        K_avail = tf.shape(y_future)[0]

        # Two candidate trajectories: positive and negative sign
        x_plus = eta_1       # (N, sd)
        x_minus = -eta_1     # (N, sd)

        log_score_plus = tf.zeros(self.n_particles, dtype=self.dtype)
        log_score_minus = tf.zeros(self.n_particles, dtype=self.dtype)

        # Save and restore model.t to avoid side effects
        saved_t = getattr(self.model, 't', None)

        Q = self.model.process_noise_cov        # (sd, sd)
        R = self.model.observation_noise_cov     # (od, od)

        for k in range(self.lookahead_steps):
            if k >= K_avail:
                break

            # Propagate one step: x_{t+k+1} = f(x_{t+k}, t+k+1)
            t_future = current_t + k + 1
            x_plus = self.model.state_transition_mean_batch(x_plus, t=t_future)
            x_minus = self.model.state_transition_mean_batch(x_minus, t=t_future)

            # Predicted observations: h(x_{t+k+1})
            y_pred_plus = self.model.observation_function_batch(x_plus)    # (N, od)
            y_pred_minus = self.model.observation_function_batch(x_minus)  # (N, od)

            # Predictive variance: H @ Q @ H^T + R per particle
            H_plus = self.model.observation_jacobian_batch(x_plus)    # (N, od, sd)
            H_minus = self.model.observation_jacobian_batch(x_minus)

            HQHt_plus = tf.einsum('nij,jk,nlk->nil', H_plus, Q, H_plus)   # (N, od, od)
            HQHt_minus = tf.einsum('nij,jk,nlk->nil', H_minus, Q, H_minus)
            var_pred_plus = HQHt_plus + tf.expand_dims(R, 0)    # (N, od, od)
            var_pred_minus = HQHt_minus + tf.expand_dims(R, 0)

            # Score: log N(y_future[k]; y_pred, var_pred)
            # For 1D obs: use diagonal elements directly
            y_k = y_future[k]  # (od,)
            diff_plus = y_k - y_pred_plus[:, 0]     # (N,) for 1D obs
            diff_minus = y_k - y_pred_minus[:, 0]

            vp = var_pred_plus[:, 0, 0]    # (N,)
            vm = var_pred_minus[:, 0, 0]

            log_score_plus += -0.5 * (diff_plus ** 2 / vp + tf.math.log(vp))
            log_score_minus += -0.5 * (diff_minus ** 2 / vm + tf.math.log(vm))

        # Restore model.t
        if saved_t is not None:
            self.model.t = saved_t

        # Flip probability: sigmoid(log_score_minus - log_score_plus)
        # If minus scores better, flip_prob > 0.5
        flip_prob = tf.nn.sigmoid(log_score_minus - log_score_plus)

        # Stochastic flip
        seed = self._next_seed()
        u = tf.random.stateless_uniform([self.n_particles], seed=seed, dtype=self.dtype)
        flip_mask = tf.cast(u < flip_prob, self.dtype)

        # Flip sign: x -> x * (1 - 2*mask), i.e. x -> -x where mask=1
        eta_1 = eta_1 * (1.0 - 2.0 * flip_mask[:, tf.newaxis])

        return eta_1
