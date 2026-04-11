"""Bootstrap Particle Filter optimized for HMC gradient computation.

Inherits from ParticleFilterTF. Adds log_marginal_likelihood_tf() for
differentiable parameter inference via HMC.

Much simpler than LEDH — no flow loop, no Jacobian accumulation.
Per timestep: propagate → weight → normalize → resample.
"""

import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, Optional

from .bootstrap_pf_tf import ParticleFilterTF
from ...resampling import systematic_resample, soft_resample, ot_entropy_resample
from ...resampling.diagnosis import effective_sample_size as ess_tf


class BootstrapPFHMC(ParticleFilterTF):
    """
    Bootstrap Particle Filter optimized for HMC gradient computation.

    The filter() method inherits from the parent and works identically.
    Only log_marginal_likelihood_tf() is added for HMC performance.
    """

    def __init__(
        self,
        *args,
        stop_gradient_resampling: bool = True,
        hmc_resampling_method: Optional[str] = None,
        hmc_resampling_config: Optional[Dict[str, Any]] = None,
        always_resample: bool = False,
        eager_mode: bool = False,
        **kwargs
    ):
        """
        Args:
            stop_gradient_resampling: If True, apply tf.stop_gradient after
                resampling in the HMC path.
            hmc_resampling_method: Override resampling method for HMC path.
            hmc_resampling_config: Config dict for HMC resampling method.
            always_resample: If True, set resample threshold to N so every
                timestep resamples. Eliminates tf.cond branch-switching
                discontinuity that causes HMC step size collapse. Use with
                soft resampling (near-no-op when ESS is already high).
            eager_mode: If True, run without @tf.function.
        """
        super().__init__(*args, **kwargs)
        self.stop_gradient_resampling = stop_gradient_resampling
        self.always_resample = always_resample
        self.eager_mode = eager_mode

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
            self._hmc_resampling_method = systematic_resample
            self._hmc_resampling_name = 'systematic'
            self._hmc_resampling_config = {}
        else:
            self._hmc_resampling_method = self.resampling_method
            self._hmc_resampling_name = self.resampling_method_name
            self._hmc_resampling_config = self.resampling_config

        if not self.eager_mode:
            self._compiled_filter = self._build_compiled_filter()

    def _build_compiled_filter(self):
        """Build @tf.function wrapping the entire bootstrap PF loop.

        Uses tf.while_loop over T timesteps. All state is carried as
        tensors through the while_loop — no tf.Variable.assign inside
        the compiled function.

        Captures by closure: model, n_particles, state_dim,
        resample_threshold, stop_gradient_resampling, resampling
        method/config.
        """
        model = self.model
        sd = self.state_dim
        n_particles = self.n_particles
        resample_thresh = tf.constant(
            self.resample_threshold * self.n_particles, dtype=self.dtype
        )
        stop_grad = self.stop_gradient_resampling
        resample_fn = self._hmc_resampling_method
        resample_cfg = self._hmc_resampling_config
        uses_transport_matrix = (self._hmc_resampling_name == 'ot_entropy')
        always_resample = self.always_resample

        param_names = list(self.model.trainable_param_names)

        @tf.function(reduce_retracing=True)
        def compiled_filter(observations, particles, weights,
                            seed_start, param_values):
            # Set model params to symbolic tensors
            for i, name in enumerate(param_names):
                setattr(model, name, param_values[i])
            T = tf.shape(observations)[0]

            def cond(t, _particles, _weights, _seed, _log_lik):
                return t < T

            def body(t, particles, weights, seed_ctr, log_lik):
                # --- Propagate ---
                seed_tf = tf.stack([seed_ctr, tf.constant(0, dtype=tf.int32)])
                seed_ctr = seed_ctr + 1
                particles = model.state_transition_batch(particles, seed_tf)

                # --- Weight ---
                y = observations[t]
                log_obs = model.log_observation_prob_batch(y, particles)
                log_weights = tf.math.log(weights + tf.constant(1e-30, dtype=weights.dtype)) + log_obs

                # --- Log-likelihood ---
                log_lik_t = tf.reduce_logsumexp(log_weights)
                log_lik = log_lik + log_lik_t

                # --- Normalize ---
                weights = tf.nn.softmax(log_weights)

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
                    seed_ctr = seed_ctr + 1
                    if stop_grad:
                        particles = tf.stop_gradient(particles)
                        weights = tf.stop_gradient(weights)
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
                        if stop_grad:
                            new_p = tf.stop_gradient(new_p)
                            new_w = tf.stop_gradient(new_w)
                        return new_p, new_w, seed_ctr + 1

                    def no_resample():
                        return particles, weights, seed_ctr

                    particles, weights, seed_ctr = tf.cond(
                        ess < resample_thresh,
                        do_resample,
                        no_resample
                    )

                return t + 1, particles, weights, seed_ctr, log_lik

            initial_state = (
                tf.constant(0, dtype=tf.int32),
                particles,
                weights,
                seed_start,
                tf.constant(0.0, dtype=particles.dtype),
            )

            final_state = tf.while_loop(
                cond, body, initial_state,
                parallel_iterations=1,
                maximum_iterations=T,
            )

            return final_state[4]  # total_log_lik

        return compiled_filter

    def log_marginal_likelihood_tf(self, observations, seed=None):
        """
        HMC-optimized log marginal likelihood.

        Compiled path: entire filter loop runs as a single @tf.function
        with tf.while_loop. GradientTape records one op.

        Eager path: Python for-loop with no compilation. Slow but
        errors appear instantly. Use for debugging.
        """
        # Normalize seed as [2] int32 tensor — no .numpy() call
        seed_tf = seed if seed is not None else tf.constant([42, 0], dtype=tf.int32)
        seed_tf = tf.cast(seed_tf, tf.int32)

        # Initialize particles as plain tensors
        keys = tf.random.experimental.stateless_split(seed_tf, num=2)
        rng_key = keys[0]
        init_seed = keys[1]
        particles = self.model.sample_initial_state_batch(self.n_particles, init_seed)
        weights = tf.ones(self.n_particles, dtype=self.dtype) / tf.cast(
            self.n_particles, self.dtype
        )

        if self.eager_mode:
            return self._run_eager(observations, particles, weights, rng_key)

        # Gather trainable params as plain tensors
        param_names = list(self.model.trainable_param_names)
        param_values = []
        for name in param_names:
            val = getattr(self.model, name)
            val = tf.identity(val) if isinstance(val, tf.Tensor) else tf.constant(
                float(val), dtype=self.dtype
            )
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
                rng_key[0],
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
        inside XLA — no boundary crossing.

        See LEDHParticleFlowFilterHMC.value_and_grad_tf for full details.
        """
        names = sorted(param_dict.keys())
        values = tuple(param_dict[n] for n in names)
        saved_attrs = {n: getattr(self.model, n) for n in names}

        @tf.function(jit_compile=True)
        def _value_and_grad(*vs):
            with tf.GradientTape() as tape:
                for v in vs:
                    tape.watch(v)
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

    def _run_eager(self, observations, particles, weights, rng_key):
        """Eager fallback: Python for-loop."""
        T = observations.shape[0]
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in range(T):
            # Split seed
            keys = tf.random.experimental.stateless_split(rng_key, num=2)
            rng_key = keys[0]
            step_seed = keys[1]

            # Propagate
            particles = self.model.state_transition_batch(particles, step_seed)

            # Weight
            y = observations[t]
            log_obs = self.model.log_observation_prob_batch(y, particles)
            log_weights = tf.math.log(weights + tf.constant(1e-30, dtype=weights.dtype)) + log_obs

            # Log-likelihood
            log_lik_t = tf.reduce_logsumexp(log_weights)
            total_log_lik = total_log_lik + log_lik_t

            # Normalize
            weights = tf.nn.softmax(log_weights)

            # Resample
            ess = ess_tf(weights)
            if ess < self.resample_threshold * self.n_particles:
                keys = tf.random.experimental.stateless_split(rng_key, num=2)
                rng_key = keys[0]
                rseed = keys[1]

                result = self._hmc_resampling_method(
                    particles, weights, seed=rseed,
                    **self._hmc_resampling_config
                )
                particles = result.particles
                weights = result.weights

                if self.stop_gradient_resampling:
                    particles = tf.stop_gradient(particles)
                    weights = tf.stop_gradient(weights)

        return total_log_lik
