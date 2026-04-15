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
from ..kalman.batched_ukf import batched_ukf_predict, batched_ukf_update
from ...utils.flow_params import compute_flow_params_batch
from ...utils.distributions import compute_flow_weights
from ...utils.linalg import (
    safe_log_abs_det, safe_inv,
    graph_safe_log_abs_det, graph_safe_log_abs_det_fast, graph_safe_log_abs_det_xla,
    graph_safe_inv,
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

        # Pre-compute UKF weights if needed (before compiled filter build)
        if self.filter_type == 'ukf' and not hasattr(self, '_ukf_weights_mean'):
            from ..kalman.batched_ukf import compute_ukf_weights
            alpha = self.ukf_params.get('alpha', 1e-3)
            beta = self.ukf_params.get('beta', 2.0)
            kappa = self.ukf_params.get('kappa', 0.0)
            self._ukf_weights_mean, self._ukf_weights_cov, self._ukf_lambda = (
                compute_ukf_weights(self.state_dim, alpha, beta, kappa, self.dtype)
            )

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
        use_ukf = (self.filter_type == 'ukf')
        if use_ukf:
            _batched_predict_fn = batched_ukf_predict.python_function
            _batched_update_fn = batched_ukf_update.python_function
            _ukf_wm = self._ukf_weights_mean
            _ukf_wc = self._ukf_weights_cov
            _ukf_lam = self._ukf_lambda
        else:
            _batched_predict_fn = batched_ekf_predict.python_function
            _batched_update_fn = batched_ekf_update.python_function
        _flow_params = compute_flow_params_batch.python_function
        _flow_weights = compute_flow_weights.python_function
        # Graph mode: use the XLA-compatible variant (QR forward + NaN-guarded
        # fast inv backward). slogdet has no XLA_CPU_JIT kernel so the _fast
        # variant fails to compile under jit_compile=True; QR works.
        _log_abs_det = graph_safe_log_abs_det_xla.python_function

        # Capture param names and per-particle R flag at build time
        param_names = list(self.model.trainable_param_names)
        has_per_particle_R = hasattr(self.model, 'observation_cov_batch')

        @tf.function(reduce_retracing=True)
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
                if use_ukf:
                    eta_bar_0, covs = _batched_predict_fn(
                        model, particles, covs,
                        _ukf_wm, _ukf_wc, _ukf_lam, sd
                    )
                else:
                    eta_bar_0, covs = _batched_predict_fn(model, particles, covs)

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

                    # Per-particle R for state-dependent obs noise (e.g. SV2D)
                    if has_per_particle_R:
                        R_flow = model.observation_cov_batch(eta_bar)  # (N, od, od)
                        R_flow = tf.maximum(R_flow, tf.constant(1e-30, dtype=R_flow.dtype))
                        R_inv_flow = 1.0 / R_flow  # safe for od=1
                    else:
                        R_flow = R
                        R_inv_flow = R_inv

                    A_batch, b_batch = _flow_params(
                        model, eta_bar, lambda_val, y, covs,
                        R_flow, R_inv_flow, eta_bar_0, sd, regularization
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
                log_lik = log_lik + log_lik_step + max_log_theta

                # --- Covariance update ---
                if use_ukf:
                    _, covs = _batched_update_fn(
                        model, eta_bar_0, covs, y,
                        _ukf_wm, _ukf_wc, _ukf_lam, sd
                    )
                else:
                    _, covs = _batched_update_fn(model, eta_bar_0, covs, y)

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
                parallel_iterations=1,
                maximum_iterations=T,
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

        Eager path: Python for-loop with plain tensors (no tf.Variable).
        Slower but errors appear instantly and gradients flow correctly.

        No tf.Variable allocation — initial state is computed as plain
        tensors from model's live properties, so gradients flow through
        Sigma_0 and sample_initial_state_batch for all models.
        """
        # Normalize seed as [2] int32 tensor — no .numpy() call
        seed_tf = seed if seed is not None else tf.constant([42, 0], dtype=tf.int32)
        seed_tf = tf.cast(seed_tf, tf.int32)
        keys = tf.random.experimental.stateless_split(seed_tf, num=2)
        filter_key = keys[0]
        init_seed = keys[1]

        # Plain tensor initial state — no tf.Variable, no initialize()
        particles = self.model.sample_initial_state_batch(
            self.n_particles, init_seed
        )
        initial_cov = self.model.Sigma_0
        particle_covs = tf.tile(
            tf.expand_dims(initial_cov, 0), [self.n_particles, 1, 1]
        )
        weights = (tf.ones(self.n_particles, dtype=self.dtype)
                   / tf.cast(self.n_particles, self.dtype))

        R = self.model.observation_noise_cov
        regularization_tf = tf.constant(self.regularization, dtype=self.dtype)

        if self.eager_mode:
            R_inv = safe_inv(R)
            return self._run_eager(
                observations, R, R_inv, regularization_tf,
                particles, particle_covs, weights, filter_key
            )

        # Graph mode: use graph_safe_inv (pinv) to avoid MatrixInverse crash
        R_inv = graph_safe_inv(R)

        # Gather trainable params as plain tensors so @tf.function treats them
        # as tensor inputs (not constants baked in at trace time).
        # tf.identity converts tf.Variable -> plain tensor while keeping the
        # gradient chain intact.
        param_names = list(self.model.trainable_param_names)
        param_values = []
        for name in param_names:
            val = getattr(self.model, name)
            val = tf.identity(val) if isinstance(val, tf.Tensor) else tf.constant(float(val), dtype=self.dtype)
            param_values.append(val)
        param_values = tf.stack(param_values)

        # Save original model attributes — compiled_filter mutates them via
        # setattr to bind to graph-local tensors during tracing. Without
        # restoration, subsequent calls would read stale graph tensors.
        saved_attrs = {name: getattr(self.model, name) for name in param_names}
        try:
            return self._compiled_filter(
                observations,
                particles,
                weights,
                particle_covs,
                R, R_inv, regularization_tf,
                filter_key[0],
                param_values
            )
        finally:
            for name, val in saved_attrs.items():
                setattr(self.model, name, val)

    def value_and_grad_tf(self, observations, param_dict, seed=None):
        """XLA-compatible value-and-gradient pass.

        Computes log_marginal_likelihood and its gradient w.r.t. the supplied
        parameters in a SINGLE jit_compile=True wrapper. By keeping the
        GradientTape inside the XLA region, the backward TensorList stays
        inside XLA — no boundary crossing (the limitation that blocks
        external GradientTape + jit_compile=True on the outer filter).

        Args:
            observations: (T, obs_dim) tensor.
            param_dict: dict {name: scalar tf.Tensor} of parameters to
                differentiate w.r.t. The values are watched and used to
                update the model before evaluating the filter.
            seed: optional [2] int32 tensor.

        Returns:
            (log_lik, grads_dict) where grads_dict has the same keys as
            param_dict.
        """
        # Sort to make iteration order deterministic
        names = sorted(param_dict.keys())
        values = tuple(param_dict[n] for n in names)

        # Save original Python-side attrs to restore at the end
        saved_attrs = {n: getattr(self.model, n) for n in names}

        @tf.function(jit_compile=True)
        def _value_and_grad(*vs):
            with tf.GradientTape() as tape:
                for v in vs:
                    tape.watch(v)
                # Bind watched tensors as model attributes so the gradient
                # chain flows through them.
                for n, v in zip(names, vs):
                    setattr(self.model, n, v)
                ll = self.log_marginal_likelihood_tf(observations, seed=seed)
            grads = tape.gradient(ll, list(vs))
            return ll, grads

        try:
            ll, grads = _value_and_grad(*values)
        finally:
            for n, val in saved_attrs.items():
                setattr(self.model, n, val)
        return ll, dict(zip(names, grads))

    def _run_eager(self, observations, R, R_inv, regularization_tf,
                   particles, particle_covs, weights, rng_key):
        """Eager fallback: Python for-loop with plain tensors.

        All state (particles, weights, particle_covs) is passed as plain
        tensors and reassigned in-place (Python rebinding, not
        tf.Variable.assign). This preserves the gradient chain for
        tf.GradientTape.
        """
        if self.filter_type == 'ukf':
            _eager_predict = batched_ukf_predict.python_function
            _eager_update = batched_ukf_update.python_function
        else:
            _eager_predict = batched_ekf_predict.python_function
            _eager_update = batched_ekf_update.python_function
        _flow_params = compute_flow_params_batch.python_function
        _flow_weights = compute_flow_weights.python_function
        _log_abs_det = safe_log_abs_det.python_function

        # Use the rng_key passed in (not self.rng_key) to keep everything
        # as plain tensors.
        def _next_seed_plain(key):
            keys = tf.random.experimental.stateless_split(key, num=2)
            return keys[0], keys[1]

        T = observations.shape[0]
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in range(T):
            if hasattr(self.model, 't'):
                self.model.t = t + 1

            # --- Predict ---
            particles_prev = particles

            if self.filter_type == 'ukf':
                eta_bar_0, cov_pred = _eager_predict(
                    self.model, particles, particle_covs,
                    self._ukf_weights_mean, self._ukf_weights_cov,
                    self._ukf_lambda, self.state_dim
                )
            else:
                eta_bar_0, cov_pred = _eager_predict(
                    self.model, particles, particle_covs
                )
            particle_covs = cov_pred

            rng_key, seed_tf = _next_seed_plain(rng_key)
            eta_0 = self.model.state_transition_batch(
                particles_prev, seed_tf, t=t + 1
            )

            # --- Flow loop ---
            y = observations[t]
            eta_1 = eta_0
            eta_bar = eta_bar_0
            log_theta = tf.zeros([self.n_particles], dtype=self.dtype)
            lambda_val = tf.constant(0.0, dtype=self.dtype)
            I_sd = tf.eye(self.state_dim, dtype=self.dtype)

            has_pp_R = hasattr(self.model, 'observation_cov_batch')

            for j in range(self.n_lambda_steps):
                d_lambda = self.lambda_steps[j]
                lambda_val = lambda_val + d_lambda

                # Per-particle R for state-dependent obs noise (e.g. SV2D)
                if has_pp_R:
                    R_flow = self.model.observation_cov_batch(eta_bar)
                    R_flow = tf.maximum(R_flow, tf.constant(1e-30, dtype=R_flow.dtype))
                    R_inv_flow = 1.0 / R_flow
                else:
                    R_flow = R
                    R_inv_flow = R_inv

                A_batch, b_batch = _flow_params(
                    self.model, eta_bar, lambda_val, y,
                    particle_covs,
                    R_flow, R_inv_flow, eta_bar_0,
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

            particles = eta_1

            # --- Weights and log-likelihood ---
            weights_new, log_lik = _flow_weights(
                eta_1=eta_1,
                eta_0=eta_0,
                particles_prev=particles_prev,
                observation=y,
                model=self.model,
                prev_weights=weights,
                jacobians=theta,
                clip_range=self.weight_clip_range
            )
            weights = weights_new
            total_log_lik = total_log_lik + log_lik + max_log_theta

            if self.on_timestep is not None:
                ess_val = float(ess_tf(weights).numpy())
                self.on_timestep(
                    t=t + 1,
                    log_lik_t=float(log_lik.numpy()),
                    ess=ess_val,
                    max_log_theta=float(max_log_theta.numpy()),
                )

            # --- Covariance update ---
            if self.filter_type == 'ukf':
                _, cov_updated = _eager_update(
                    self.model, eta_bar_0,
                    particle_covs, y,
                    self._ukf_weights_mean, self._ukf_weights_cov,
                    self._ukf_lambda, self.state_dim
                )
            else:
                _, cov_updated = _eager_update(
                    self.model, eta_bar_0,
                    particle_covs, y
                )
            particle_covs = cov_updated

            # --- Resampling ---
            ess = ess_tf(weights)
            if ess < self.resample_threshold * self.n_particles:
                rng_key, rseed = _next_seed_plain(rng_key)
                result = self._hmc_resampling_method(
                    particles, weights, seed=rseed,
                    **self._hmc_resampling_config
                )
                if result.transport_matrix is not None:
                    T_mat = result.transport_matrix
                    particle_covs = tf.einsum('ij,jkl->ikl', T_mat, particle_covs)
                elif result.ancestor_indices is not None:
                    particle_covs = tf.gather(particle_covs, result.ancestor_indices)
                particles = result.particles
                weights = result.weights
                if self.stop_gradient_resampling:
                    particles = tf.stop_gradient(particles)
                    weights = tf.stop_gradient(weights)
                    particle_covs = tf.stop_gradient(particle_covs)

        return total_log_lik

    # NOTE: _resample_hmc() removed — resampling is now inlined in
    # _run_eager() to avoid tf.Variable.assign() which severs gradients.
