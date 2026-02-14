"""Main HMC/NUTS runner for parameter inference with differentiable filters."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
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

        # Bijectors and priors
        self.param_handler = ParameterHandler(param_specs)

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

        print(f"Running {self.sampler.upper()} with {num_samples} samples, "
              f"{num_burnin} burn-in...")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            samples_unconstrained, trace_results = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin,
                current_state=self.param_handler.unconstrained_init,
                kernel=adaptive_kernel,
                trace_fn=trace_fn
            )

        # Post-process
        samples_constrained = self._transform_samples(samples_unconstrained)
        diagnostics = self._compute_diagnostics(
            samples_constrained, trace_results['is_accepted'], trace_results['step_size']
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
        diagnostics['acceptance_rate'] = float(
            tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy()
        )
        diagnostics['final_step_size'] = float(step_sizes[-1].numpy())

        ess_dict = {}
        for name, s in samples.items():
            param_tf = tf.constant(s[np.newaxis, :], dtype=tf.float32)
            ess = tfp.mcmc.effective_sample_size(param_tf)
            ess_dict[name] = float(ess.numpy()[0])
        diagnostics['ess'] = ess_dict

        rhat_dict = {}
        for name, s in samples.items():
            if len(s) >= 4:
                mid = len(s) // 2
                chains = tf.constant([s[:mid], s[mid:2*mid]], dtype=tf.float32)
                rhat_dict[name] = float(tfp.mcmc.potential_scale_reduction(chains).numpy())
            else:
                rhat_dict[name] = np.nan
        diagnostics['rhat'] = rhat_dict

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
