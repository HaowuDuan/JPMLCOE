"""Main HMC/NUTS runner for parameter inference with differentiable filters."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from typing import Any, Dict, Optional, Type
import warnings

from .types import ParameterSpec, DPFResult
from .parameter_handler import ParameterHandler
from .differentiable_model import DifferentiableModel


class DPFRunner:
    """
    Differentiable Filter runner for parameter inference via HMC/NUTS.

    Architecture:
        1. DifferentiableModel wraps model, tracks trainable params
        2. ParameterHandler manages bijector transforms (constrained <-> unconstrained)
        3. Filter.log_marginal_likelihood_tf() computes log p(y|theta) in TF
        4. _negative_log_posterior() sets model params via setattr, then calls filter
        5. HMC/NUTS differentiates through log posterior via tf.GradientTape

    Key: NO tf.py_function (breaks gradients), NO tf.Variable.assign (severs chain).
    Parameters flow as plain tensors: unconstrained -> bijector -> setattr -> model -> filter.
    """

    def __init__(
        self,
        base_model: Any,
        filter_class: Type,
        filter_kwargs: Dict[str, Any],
        param_specs: Dict[str, ParameterSpec],
        sampler: str = 'hmc'
    ):
        """
        Initialize DPF runner.

        Args:
            base_model: State-space model instance
            filter_class: Filter class (e.g., ExtendedKalmanFilter)
            filter_kwargs: Keyword arguments for filter initialization
            param_specs: Dictionary of parameter specifications
            sampler: 'nuts' or 'hmc'
        """
        self.base_model = base_model
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        self.sampler = sampler.lower()

        # Wrap model: tracks trainable params, delegates everything else
        trainable_param_names = list(param_specs.keys())
        self.diff_model = DifferentiableModel(base_model, trainable_param_names)

        # Bijectors and priors (use model dtype for consistency)
        model_dtype = getattr(base_model, 'dtype', tf.float32)
        self.param_handler = ParameterHandler(param_specs, dtype=model_dtype)

        # Create filter ONCE with the wrapped model
        self.filter_obj = self.filter_class(self.diff_model, **self.filter_kwargs)

        self._observations_tf = None

    def _negative_log_posterior(self, unconstrained_params: tf.Tensor) -> tf.Tensor:
        """
        Compute -log p(theta | y) = -log p(y | theta) - log p(theta).

        NOT @tf.function — either runs eagerly or is traced by TFP internally.
        Uses setattr to update model params, preserving the gradient chain.
        """
        # 1. Bijectors: unconstrained -> constrained
        constrained_params = self.param_handler.constrain(unconstrained_params)

        # 2. setattr on model — gradient chain preserved
        self.diff_model.update_parameters(constrained_params)

        # 3. Filter forward pass (entirely in TF)
        seed = tf.constant([42, 0], dtype=tf.int32)
        log_likelihood = self.filter_obj.log_marginal_likelihood_tf(
            self._observations_tf, seed=seed
        )

        # 4. Log prior with Jacobian adjustment
        log_prior = self.param_handler.log_prior(constrained_params)

        return -(log_likelihood + log_prior)

    def run_inference(
        self,
        observations: np.ndarray,
        num_samples: int = 1000,
        num_burnin: int = 500,
        step_size: float = 0.01,
        num_leapfrog_steps: int = 10,
        adaptation_rate: float = 0.8,
        target_accept_prob: float = 0.75,
        seed: Optional[int] = None,
        max_tree_depth: int = 10
    ) -> DPFResult:
        """Run HMC or NUTS to sample from posterior p(theta | y)."""
        dtype = getattr(self.base_model, 'dtype', tf.float32)
        self._observations_tf = tf.constant(observations, dtype=dtype)

        def target_log_prob_fn(unconstrained_params):
            return -self._negative_log_posterior(unconstrained_params)

        # Choose sampler
        if self.sampler == 'nuts':
            print(f"Using NUTS sampler (max_tree_depth={max_tree_depth})")
            inner_kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                max_tree_depth=max_tree_depth
            )
        else:
            print(f"Using HMC sampler (num_leapfrog_steps={num_leapfrog_steps})")
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps
            )

        # Adaptive step size
        num_adaptation_steps = int(adaptation_rate * num_burnin)
        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob
        )

        if seed is not None:
            tf.random.set_seed(seed)

        def trace_fn(_, pkr):
            return {
                'is_accepted': pkr.inner_results.is_accepted,
                'step_size': pkr.new_step_size
            }

        total_steps = num_burnin + num_samples
        print(f"Running {self.sampler.upper()}: {num_burnin} burn-in + {num_samples} sampling = {total_steps} total steps")

        # Manual loop (identical to sample_chain internally) with progress tracking
        current_state = self.param_handler.unconstrained_init
        kernel_results = adaptive_kernel.bootstrap_results(current_state)

        samples_list = []
        is_accepted_list = []
        step_size_list = []
        step_times = []

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            for i in range(total_steps):
                t0 = time.perf_counter()
                current_state, kernel_results = adaptive_kernel.one_step(
                    current_state, kernel_results
                )
                dt = time.perf_counter() - t0
                step_times.append(dt)

                accepted = bool(kernel_results.inner_results.is_accepted.numpy())
                cur_step_size = float(kernel_results.new_step_size.numpy())

                is_accepted_list.append(accepted)
                step_size_list.append(cur_step_size)

                phase = "burn-in" if i < num_burnin else "sample"
                idx = i - num_burnin + 1 if i >= num_burnin else i + 1
                phase_total = num_samples if i >= num_burnin else num_burnin

                # Current param values
                constrained = self.param_handler.constrain(current_state)
                param_str = ", ".join(f"{n}={float(v.numpy()):.4f}" for n, v in constrained.items())

                # ETA
                avg_dt = np.mean(step_times[-10:])  # rolling average of last 10
                remaining = total_steps - (i + 1)
                eta = avg_dt * remaining

                # Running acceptance rate
                accept_rate = np.mean(is_accepted_list[-min(50, len(is_accepted_list)):])

                print(f"  [{phase} {idx}/{phase_total}] "
                      f"{dt:.1f}s | accept={accept_rate:.0%} | step_size={cur_step_size:.4f} | "
                      f"{param_str} | ETA {eta:.0f}s")

                if i >= num_burnin:
                    samples_list.append(current_state.numpy().copy())

        samples_unconstrained = tf.constant(np.stack(samples_list), dtype=current_state.dtype)
        trace_is_accepted = tf.constant(is_accepted_list[num_burnin:])
        trace_step_sizes = tf.constant(step_size_list[num_burnin:])

        # Post-process
        samples_constrained = self._transform_samples(samples_unconstrained)
        diagnostics = self._compute_diagnostics(
            samples_constrained, trace_is_accepted, trace_step_sizes
        )
        summary = self._compute_summary(samples_constrained)
        self.diff_model.restore_parameters()

        print(f"{self.sampler.upper()} complete!")

        return DPFResult(
            samples=samples_constrained,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                'model_type': self.base_model.__class__.__name__,
                'filter_type': self.filter_class.__name__,
                'sampler': self.sampler,
                'num_samples': num_samples,
                'num_burnin': num_burnin,
                'num_observations': len(observations)
            }
        )

    # Backward compat
    run_hmc = run_inference

    def _transform_samples(self, samples_unconstrained):
        num_samples = samples_unconstrained.shape[0]
        result = {name: [] for name in self.param_handler.param_names}
        for i in range(num_samples):
            constrained = self.param_handler.constrain(samples_unconstrained[i])
            for name, value in constrained.items():
                result[name].append(float(value.numpy()))
        return {name: np.array(vals) for name, vals in result.items()}

    def _compute_diagnostics(self, samples, is_accepted, step_sizes):
        diagnostics = {}
        dtype = getattr(self.base_model, 'dtype', tf.float32)
        diagnostics['acceptance_rate'] = float(
            tf.reduce_mean(tf.cast(is_accepted, dtype)).numpy()
        )
        diagnostics['final_step_size'] = float(step_sizes[-1].numpy())

        # Suppress complex64->float32 warning from FFT-based autocorrelation in ESS
        import logging
        tf_logger = logging.getLogger('tensorflow')
        prev_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)

        ess_dict = {}
        for name, s in samples.items():
            param_tf = tf.constant(s[np.newaxis, :], dtype=dtype)
            ess = tfp.mcmc.effective_sample_size(param_tf)
            ess_dict[name] = float(ess.numpy()[0])
        diagnostics['ess'] = ess_dict

        rhat_dict = {}
        for name, s in samples.items():
            if len(s) >= 4:
                mid = len(s) // 2
                chains = tf.constant([s[:mid], s[mid:2*mid]], dtype=dtype)
                rhat_dict[name] = float(tfp.mcmc.potential_scale_reduction(chains).numpy())
            else:
                rhat_dict[name] = np.nan
        diagnostics['rhat'] = rhat_dict

        tf_logger.setLevel(prev_level)

        return diagnostics

    def _compute_summary(self, samples):
        summary = {}
        for name, s in samples.items():
            summary[name] = {
                'mean': float(np.mean(s)), 'std': float(np.std(s)),
                'median': float(np.median(s)),
                'q5': float(np.percentile(s, 5)), 'q25': float(np.percentile(s, 25)),
                'q75': float(np.percentile(s, 75)), 'q95': float(np.percentile(s, 95)),
                'min': float(np.min(s)), 'max': float(np.max(s))
            }
        return summary
