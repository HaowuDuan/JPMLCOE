"""LEDH Particle Flow Filter optimized for HMC gradient computation.

Inherits from LEDHParticleFlowFilter. Only log_marginal_likelihood_tf() is
overridden — the filter() method is unchanged.

Optimizations:
1. Full @tf.function compilation: entire filter loop (T timesteps × 29 flow
   steps) wrapped in tf.while_loop inside @tf.function. GradientTape records
   it as one op; backward pass runs entirely in C++.
2. Optional stop_gradient on resampling: eliminates Sinkhorn backward.
   When enabled, a cheaper resampling method can be used (e.g. systematic).
3. No diagnostic tracking in the HMC path: no weights_history, ess_history,
   numpy calls, or list appends inside the tape scope.
"""

import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, Optional

from .ledh_invertible import LEDHParticleFlowFilter
from ..kalman.batched_ekf import batched_ekf_predict, batched_ekf_update
from ...utils.flow_params import compute_flow_params_batch
from ...utils.distributions import compute_flow_weights
from ...utils.linalg import (
    safe_log_abs_det, safe_inv,
    graph_safe_log_abs_det, graph_safe_log_abs_det_fast, graph_safe_inv,
)
from ...resampling import systematic_resample, soft_resample, ot_entropy_resample
from ...resampling.diagnosis import effective_sample_size as ess_tf


class LEDHParticleFlowFilterHMC(LEDHParticleFlowFilter):
    """
    LEDH Particle Flow Filter optimized for HMC gradient computation.

    The filter() method inherits from the parent and works identically.
    Only log_marginal_likelihood_tf() is overridden for HMC performance.

    Key design: the compiled filter assumes that model.observation_jacobian_batch
    and model.observation_function_batch are parameter-independent (true for
    Kitagawa: x/10 and x^2/20). For models where these depend on trainable
    parameters, the @tf.function trace would be stale after parameter updates.
    """

    def __init__(
        self,
        *args,
        stop_gradient_resampling: bool = True,
        hmc_resampling_method: Optional[str] = None,
        hmc_resampling_config: Optional[Dict[str, Any]] = None,
        always_resample: bool = False,
        eager_mode: bool = False,
        on_timestep: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            stop_gradient_resampling: If True, apply tf.stop_gradient after
                resampling in the HMC path. Eliminates gradient through the
                resampling transport plan. When True, you can safely switch
                to cheaper non-differentiable resampling (e.g. systematic).
            hmc_resampling_method: Override resampling method for HMC path.
                Options: 'systematic', 'soft', 'ot_entropy'.
                If None and stop_gradient_resampling=True: defaults to 'systematic'.
                If None and stop_gradient_resampling=False: uses parent's method.
            hmc_resampling_config: Config dict for HMC resampling method.
                If None and using parent's method: uses parent's config.
                If None and using different method: empty config.
            always_resample: If True, set resample threshold to N so every
                timestep resamples. Eliminates tf.cond branch-switching
                discontinuity that causes HMC step size collapse. Use with
                soft resampling (near-no-op when ESS is already high).
            eager_mode: If True, run without @tf.function / tf.while_loop.
                Errors appear instantly. GradientTape still works (just slower).
            on_timestep: Optional callback(t, log_lik_t, ess, max_log_theta)
                called at each timestep in the eager path. Use for diagnostics
                without polluting production logs.
            *args, **kwargs: Passed to LEDHParticleFlowFilter.__init__
        """
        super().__init__(*args, **kwargs)
        self.stop_gradient_resampling = stop_gradient_resampling
        self.always_resample = always_resample
        self.eager_mode = eager_mode
        self.on_timestep = on_timestep

        # When always_resample, set threshold to N so ESS < N always triggers.
        if always_resample:
            self.resample_threshold = 1.0

        # Set up HMC-specific resampling
        method_map = {
            'systematic': systematic_resample,
            'ot_entropy': ot_entropy_resample,
            'soft': soft_resample,
        }

        if hmc_resampling_method is not None:
            self._hmc_resampling_method = method_map.get(
                hmc_resampling_method, systematic_resample
            )
            self._hmc_resampling_name = hmc_resampling_method
            self._hmc_resampling_config = {}
            if hmc_resampling_config is not None:
                for key, value in hmc_resampling_config.items():
                    if isinstance(value, (int, np.integer)):
                        self._hmc_resampling_config[key] = int(value)
                    elif isinstance(value, (float, np.floating)):
                        self._hmc_resampling_config[key] = float(value)
                    else:
                        self._hmc_resampling_config[key] = value
        elif stop_gradient_resampling:
            # No need for expensive OT when gradient is cut
            self._hmc_resampling_method = systematic_resample
            self._hmc_resampling_name = 'systematic'
            self._hmc_resampling_config = {}
        else:
            # Use parent's method (preserve full gradient through resampling)
            self._hmc_resampling_method = self.resampling_method
            self._hmc_resampling_name = self.resampling_method_name
            self._hmc_resampling_config = self.resampling_config

        # Build compiled filter (traced on first call)
        if not self.eager_mode:
            self._compiled_filter = self._build_compiled_filter()

    def _build_compiled_filter(self):
        """Build @tf.function wrapping the entire filter loop.

        Uses tf.while_loop over T timesteps. The flow loop (n_lambda_steps)
        is inlined as a Python for-loop and unrolled once during tracing.
        All state is carried as tensors through the while_loop — no
        tf.Variable.assign inside the compiled function.

        Captures by closure: model, lambda_steps, n_lambda_steps, state_dim,
        resample_threshold, n_particles, weight_clip_range,
        stop_gradient_resampling, resampling method/config.
        """
        model = self.model
        n_flow_steps = self.n_lambda_steps
        lambda_steps = self.lambda_steps
        sd = self.state_dim
        resample_thresh = tf.constant(
            self.resample_threshold * self.n_particles, dtype=self.dtype
        )
        clip_range = self.weight_clip_range
        stop_grad = self.stop_gradient_resampling
        resample_fn = self._hmc_resampling_method
        resample_cfg = self._hmc_resampling_config
        uses_transport_matrix = (self._hmc_resampling_name == 'ot_entropy')
        always_resample = self.always_resample

        # Bypass nested @tf.function — inline into this trace so model
        # attributes resolve to symbolic tensors and gradients flow correctly
        # through the full computation.
        _ekf_predict = batched_ekf_predict.python_function
        _ekf_update = batched_ekf_update.python_function
        _flow_params = compute_flow_params_batch.python_function
        _flow_weights = compute_flow_weights.python_function
        # Graph mode: use graph_safe_log_abs_det (custom gradient with pinv
        # backward) to avoid MatrixInverse crash on GPU in graph mode.
        _log_abs_det = graph_safe_log_abs_det_fast.python_function

        # Capture param names at build time (works for any model)
        param_names = list(self.model.trainable_param_names)

        @tf.function
        def compiled_filter(observations, particles, weights, covs,
                            R, R_inv, regularization, seed_start,
                            param_values):
            # Set model params to symbolic tensors — graph reads current HMC values
            for i, name in enumerate(param_names):
                setattr(model, name, param_values[i])
            T = tf.shape(observations)[0]

            def cond(t, _particles, _weights, _covs, _seed, _log_lik):
                return t < T

            def body(t, particles, weights, covs, seed_ctr, log_lik):
                # Set model time (symbolic tensor — traced once, evaluated per iter)
                model.t = t + 1

                # --- Predict ---
                particles_prev = particles
                eta_bar_0, covs = _ekf_predict(model, particles, covs)

                seed_tf = tf.stack([seed_ctr, tf.constant(0, dtype=tf.int32)])
                seed_ctr = seed_ctr + 1
                eta_0 = model.state_transition_batch(particles_prev, seed_tf)

                # --- Flow loop (unrolled 29 steps) ---
                y = observations[t]
                eta_1 = eta_0
                eta_bar = eta_bar_0
                log_theta = tf.zeros([tf.shape(particles)[0]], dtype=particles.dtype)
                lambda_val = tf.constant(0.0, dtype=particles.dtype)
                I_sd = tf.eye(sd, dtype=particles.dtype)

                for j in range(n_flow_steps):
                    d_lambda = lambda_steps[j]
                    lambda_val = lambda_val + d_lambda

                    A_batch, b_batch = _flow_params(
                        model, eta_bar, lambda_val, y, covs,
                        R, R_inv, eta_bar_0, sd, regularization
                    )

                    drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
                    eta_bar = eta_bar + d_lambda * drift_bar

                    drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
                    eta_1 = eta_1 + d_lambda * drift_1

                    M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
                    log_det_M = _log_abs_det(M_batch)
                    log_theta = log_theta + log_det_M

                # Normalize Jacobians
                max_log_theta = tf.reduce_max(log_theta)
                log_theta = log_theta - max_log_theta
                theta = tf.exp(log_theta)

                particles = eta_1

                # --- Weights and log-likelihood ---
                weights, log_lik_step = _flow_weights(
                    eta_1=eta_1, eta_0=eta_0,
                    particles_prev=particles_prev,
                    observation=y, model=model,
                    prev_weights=weights,
                    jacobians=theta,
                    clip_range=clip_range
                )
                log_lik = log_lik + log_lik_step

                # --- EKF covariance update ---
                _, covs = _ekf_update(model, eta_bar_0, covs, y)

                # --- Resampling ---
                if always_resample:
                    # Always resample: no tf.cond, uniform computation graph.
                    # Soft resampling is a near-no-op when ESS is high.
                    rseed = tf.stack([seed_ctr, tf.constant(0, dtype=tf.int32)])
                    result = resample_fn(
                        particles, weights, seed=rseed, **resample_cfg
                    )
                    particles = result.particles
                    weights = result.weights
                    if uses_transport_matrix:
                        covs = tf.einsum('ij,jkl->ikl', result.transport_matrix, covs)
                    else:
                        covs = tf.gather(covs, result.ancestor_indices)
                    seed_ctr = seed_ctr + 1
                    if stop_grad:
                        particles = tf.stop_gradient(particles)
                        weights = tf.stop_gradient(weights)
                        covs = tf.stop_gradient(covs)
                else:
                    # Conditional resampling (original): tf.cond creates
                    # a discontinuity that can cause HMC step size collapse.
                    ess = ess_tf(weights)

                    def do_resample():
                        rseed = tf.stack([seed_ctr, tf.constant(0, dtype=tf.int32)])
                        result = resample_fn(
                            particles, weights, seed=rseed, **resample_cfg
                        )
                        new_p = result.particles
                        new_w = result.weights
                        if uses_transport_matrix:
                            T = result.transport_matrix
                            new_covs = tf.einsum('ij,jkl->ikl', T, covs)
                        else:
                            new_covs = tf.gather(covs, result.ancestor_indices)
                        if stop_grad:
                            new_p = tf.stop_gradient(new_p)
                            new_w = tf.stop_gradient(new_w)
                            new_covs = tf.stop_gradient(new_covs)
                        return new_p, new_w, new_covs, seed_ctr + 1

                    def no_resample():
                        return particles, weights, covs, seed_ctr

                    particles, weights, covs, seed_ctr = tf.cond(
                        ess < resample_thresh,
                        do_resample,
                        no_resample
                    )

                return t + 1, particles, weights, covs, seed_ctr, log_lik

            initial_state = (
                tf.constant(0, dtype=tf.int32),
                particles,
                weights,
                covs,
                seed_start,
                tf.constant(0.0, dtype=particles.dtype),
            )

            final_state = tf.while_loop(
                cond, body, initial_state,
                parallel_iterations=1
            )

            return final_state[5]  # total_log_lik

        return compiled_filter

    def initialize(self, *args, **kwargs):
        """Initialize filter state, also resetting R_inv cache.

        The parent caches R_inv but never resets it. For HMC, R changes
        between proposals (model params change), so the cache must be cleared.
        """
        super().initialize(*args, **kwargs)
        self.R_inv_cache = None

    def log_marginal_likelihood_tf(self, observations, seed=None):
        """
        HMC-optimized log marginal likelihood.

        Compiled path: entire filter loop runs as a single @tf.function
        with tf.while_loop. GradientTape records one op.

        Eager path: Python for-loop with tf.Variable.assign. Slow but
        errors appear instantly. Use for debugging.
        """
        random_seed = int(seed[0].numpy()) if seed is not None else 42
        self.initialize(random_seed=random_seed)

        R = self.model.observation_noise_cov
        regularization_tf = tf.constant(self.regularization, dtype=self.dtype)

        if self.eager_mode:
            R_inv = safe_inv(R)
            return self._run_eager(
                observations, R, R_inv, regularization_tf
            )

        # Graph mode: use graph_safe_inv (pinv) to avoid MatrixInverse crash
        R_inv = graph_safe_inv(R)

        # Gather trainable params as plain tensors so @tf.function treats them
        # as tensor inputs (not constants baked in at trace time).
        # tf.identity converts tf.Variable -> plain tensor while keeping the
        # gradient chain intact.
        param_values = []
        for name in self.model.trainable_param_names:
            val = getattr(self.model, name)
            val = tf.identity(val) if isinstance(val, tf.Tensor) else tf.constant(float(val), dtype=self.dtype)
            param_values.append(val)
        param_values = tf.stack(param_values)

        return self._compiled_filter(
            observations,
            self.particles.value(),
            self.weights.value(),
            self.particle_covs.value(),
            R, R_inv, regularization_tf,
            self.rng_key[0],
            param_values
        )

    def _run_eager(self, observations, R, R_inv, regularization_tf):
        """Eager fallback: Python for-loop with Variable.assign."""
        # Bypass @tf.function to avoid autograph issues (e.g. isinstance in
        # _as_sigma) and ensure GradientTape tracks ops eagerly.
        _ekf_predict = batched_ekf_predict.python_function
        _ekf_update = batched_ekf_update.python_function
        _flow_params = compute_flow_params_batch.python_function
        _flow_weights = compute_flow_weights.python_function
        _log_abs_det = safe_log_abs_det.python_function

        T = observations.shape[0]
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in range(T):
            if hasattr(self.model, 't'):
                self.model.t = t + 1

            # --- Predict ---
            self.particles_prev.assign(self.particles.value())

            eta_bar_0_tf, cov_pred_tf = _ekf_predict(
                self.model, self.particles.value(), self.particle_covs.value()
            )
            self.particle_covs.assign(cov_pred_tf)
            self.eta_bar_0.assign(eta_bar_0_tf)

            seed_tf = self._next_seed()
            eta_0_tf = self.model.state_transition_batch(
                self.particles_prev.value(), seed_tf, t=t + 1
            )
            self.eta_0.assign(eta_0_tf)

            # --- Flow loop ---
            y = observations[t]
            eta_1 = self.eta_0.value()
            eta_bar = self.eta_bar_0.value()
            log_theta = tf.zeros([self.n_particles], dtype=self.dtype)
            lambda_val = tf.constant(0.0, dtype=self.dtype)
            I_sd = tf.eye(self.state_dim, dtype=self.dtype)

            for j in range(self.n_lambda_steps):
                d_lambda = self.lambda_steps[j]
                lambda_val = lambda_val + d_lambda

                A_batch, b_batch = _flow_params(
                    self.model, eta_bar, lambda_val, y,
                    self.particle_covs.value(),
                    R, R_inv, self.eta_bar_0.value(),
                    self.state_dim, regularization_tf
                )

                drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
                eta_bar = eta_bar + d_lambda * drift_bar

                drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
                eta_1 = eta_1 + d_lambda * drift_1

                M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
                log_det_M = _log_abs_det(M_batch)
                log_theta = log_theta + log_det_M

            # Normalize Jacobians
            max_log_theta = tf.reduce_max(log_theta)
            log_theta = log_theta - max_log_theta
            theta = tf.exp(log_theta)

            self.particles.assign(eta_1)

            # --- Weights and log-likelihood ---
            weights_new, log_lik = _flow_weights(
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
            total_log_lik = total_log_lik + log_lik

            if self.on_timestep is not None:
                ess_val = float(ess_tf(self.weights.value()).numpy())
                self.on_timestep(
                    t=t + 1,
                    log_lik_t=float(log_lik.numpy()),
                    ess=ess_val,
                    max_log_theta=float(max_log_theta.numpy()),
                )

            # --- EKF covariance update ---
            _, cov_updated = _ekf_update(
                self.model, self.eta_bar_0.value(),
                self.particle_covs.value(), y
            )
            self.particle_covs.assign(cov_updated)

            # --- Resampling ---
            ess = ess_tf(self.weights.value())
            if ess < self.resample_threshold * self.n_particles:
                self._resample_hmc()

        return total_log_lik

    def _resample_hmc(self):
        """Resample with configurable method and optional stop_gradient.

        Used by the eager path only. The compiled path inlines resampling
        into the tf.while_loop body via tf.cond.
        """
        seed = self._next_seed()

        result = self._hmc_resampling_method(
            self.particles.value(),
            self.weights.value(),
            seed=seed,
            **self._hmc_resampling_config
        )

        if result.transport_matrix is not None:
            # OT resampling: transport covs via T @ covs (differentiable)
            T = result.transport_matrix
            self.particle_covs.assign(
                tf.einsum('ij,jkl->ikl', T, self.particle_covs.value())
            )
        elif result.ancestor_indices is not None:
            # Index-based resampling (systematic, soft)
            self.particle_covs.assign(
                tf.gather(self.particle_covs.value(), result.ancestor_indices)
            )
        else:
            raise ValueError(
                "ResampleResult has neither ancestor_indices nor transport_matrix"
            )
        self.particles.assign(result.particles)
        self.weights.assign(result.weights)

        if self.stop_gradient_resampling:
            self.particles.assign(tf.stop_gradient(self.particles.value()))
            self.weights.assign(tf.stop_gradient(self.weights.value()))
            self.particle_covs.assign(tf.stop_gradient(self.particle_covs.value()))
