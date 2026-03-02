"""Standalone Metropolis-Hastings runner for parameter inference.

Uses tfp.mcmc.RandomWalkMetropolis with the exact log-posterior
(no gradients needed). Works with any filter that provides
log_marginal_likelihood_tf() — Kalman filter gives exact MH,
particle filter gives PMMH-like behavior but with fixed seeds.

Key difference from PMMHRunner:
  - Uses ParameterHandler bijectors (proposal in unconstrained space)
  - Uses DifferentiableModel wrapper for consistent interface
  - Step-by-step loop with progress printing (matches DPFRunner style)
  - Fixed seed per evaluation (deterministic likelihood surface)

Key difference from DPFRunner (HMC):
  - No GradientTape, no tf.function
  - tfp.mcmc.RandomWalkMetropolis instead of HMC/NUTS
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from typing import Any, Dict, Optional, Type

from .types import ParameterSpec, DPFResult
from .parameter_handler import ParameterHandler
from .differentiable_model import DifferentiableModel


class MHRunner:
    """Standalone Metropolis-Hastings for parameter inference via random walk."""

    def __init__(
        self,
        base_model: Any,
        filter_class: Type,
        filter_kwargs: Dict[str, Any],
        param_specs: Dict[str, ParameterSpec],
    ):
        self.base_model = base_model
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs

        trainable_param_names = list(param_specs.keys())
        self.diff_model = DifferentiableModel(base_model, trainable_param_names)

        model_dtype = getattr(base_model, 'dtype', tf.float32)
        self.param_handler = ParameterHandler(param_specs, dtype=model_dtype)

        self.filter_obj = self.filter_class(self.diff_model, **self.filter_kwargs)
        self._observations_tf = None

    def _log_posterior(self, unconstrained_params: tf.Tensor) -> tf.Tensor:
        """Compute log p(theta | y) = log p(y | theta) + log p(theta)."""
        constrained_params = self.param_handler.constrain(unconstrained_params)
        self.diff_model.update_parameters(constrained_params)

        seed = tf.constant([42, 0], dtype=tf.int32)
        log_likelihood = self.filter_obj.log_marginal_likelihood_tf(
            self._observations_tf, seed=seed
        )

        log_prior = self.param_handler.log_prior(constrained_params)
        return log_likelihood + log_prior

    def run_inference(
        self,
        observations: np.ndarray,
        num_samples: int = 2000,
        num_burnin: int = 1000,
        proposal_std: float = 0.1,
        seed: Optional[int] = None,
    ) -> DPFResult:
        """Run MH to sample from posterior p(theta | y).

        Args:
            observations: (T,) or (T, obs_dim) observation array.
            num_samples: Post-burn-in samples.
            num_burnin: Burn-in iterations.
            proposal_std: Proposal scale in unconstrained space.
            seed: Random seed.
        """
        dtype = getattr(self.base_model, 'dtype', tf.float32)
        self._observations_tf = tf.constant(observations, dtype=dtype)

        if seed is not None:
            tf.random.set_seed(seed)

        total_steps = num_burnin + num_samples
        print(f"Using MH sampler (proposal_std={proposal_std})")
        print(f"Running: {num_burnin} burn-in + {num_samples} sampling "
              f"= {total_steps} total steps")

        def target_log_prob_fn(unconstrained_params):
            return self._log_posterior(unconstrained_params)

        inner_kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=proposal_std),
        )

        current_state = self.param_handler.unconstrained_init
        kernel_results = inner_kernel.bootstrap_results(current_state)

        # Initial diagnostic
        lp = target_log_prob_fn(current_state)
        constrained = self.param_handler.constrain(current_state)
        param_str = ", ".join(
            f"{n}={float(v.numpy()):.4f}" for n, v in constrained.items()
        )
        print(f"  Initial: log_post={float(lp.numpy()):.4f}, {param_str}")

        samples_list = []
        is_accepted_list = []
        step_times = []
        trace_log = []

        for i in range(total_steps):
            t0 = time.perf_counter()
            current_state, kernel_results = inner_kernel.one_step(
                current_state, kernel_results
            )
            dt = time.perf_counter() - t0

            accepted = bool(kernel_results.is_accepted.numpy())
            is_accepted_list.append(accepted)
            step_times.append(dt)

            # Progress
            phase = "burn-in" if i < num_burnin else "sample"
            idx = i - num_burnin + 1 if i >= num_burnin else i + 1
            phase_total = num_samples if i >= num_burnin else num_burnin

            constrained = self.param_handler.constrain(current_state)
            param_str = ", ".join(
                f"{n}={float(v.numpy()):.4f}" for n, v in constrained.items()
            )
            avg_dt = np.mean(step_times[-10:])
            eta = avg_dt * (total_steps - i - 1)
            accept_rate = np.mean(
                is_accepted_list[-min(50, len(is_accepted_list)):]
            )

            print(f"  [{phase} {idx}/{phase_total}] "
                  f"{dt:.1f}s | accept={accept_rate:.0%} | "
                  f"proposal_std={proposal_std:.4f} | "
                  f"{param_str} | ETA {eta:.0f}s")

            trace_log.append({
                'step': i, 'phase': phase, 'step_in_phase': idx,
                'dt': dt, 'accept_rate': float(accept_rate),
                'step_size': proposal_std,
                **{n: float(v.numpy()) for n, v in constrained.items()},
            })

            if i >= num_burnin:
                samples_list.append(current_state.numpy().copy())

        return self._finalize(
            samples_list, is_accepted_list, step_times,
            num_burnin, num_samples, proposal_std,
            dtype, observations, trace_log
        )

    def _finalize(self, samples_list, is_accepted_list, step_times,
                  num_burnin, num_samples, proposal_std,
                  dtype, observations, trace_log):
        """Post-process samples and compute diagnostics."""
        import logging

        samples_unconstrained = tf.constant(
            np.stack(samples_list),
            dtype=dtype if not isinstance(dtype, np.dtype) else tf.float32,
        )

        # Transform to constrained space
        samples_constrained = {name: [] for name in self.param_handler.param_names}
        for i in range(samples_unconstrained.shape[0]):
            constrained = self.param_handler.constrain(samples_unconstrained[i])
            for name, value in constrained.items():
                samples_constrained[name].append(float(value.numpy()))
        samples_constrained = {
            name: np.array(vals) for name, vals in samples_constrained.items()
        }

        # Diagnostics
        post_burnin_accepted = is_accepted_list[num_burnin:]
        acceptance_rate = float(np.mean(post_burnin_accepted))

        tf_logger = logging.getLogger('tensorflow')
        prev_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)

        ess_dict = {}
        for name, s in samples_constrained.items():
            param_tf = tf.constant(s[:, np.newaxis], dtype=dtype)
            ess = tfp.mcmc.effective_sample_size(param_tf)
            ess_dict[name] = float(ess.numpy()[0])

        rhat_dict = {}
        for name, s in samples_constrained.items():
            if len(s) >= 4:
                mid = len(s) // 2
                chains = tf.constant([s[:mid], s[mid:2 * mid]], dtype=dtype)
                rhat_dict[name] = float(
                    tfp.mcmc.potential_scale_reduction(chains).numpy()
                )
            else:
                rhat_dict[name] = np.nan

        tf_logger.setLevel(prev_level)

        diagnostics = {
            'acceptance_rate': acceptance_rate,
            'final_step_size': proposal_std,
            'ess': ess_dict,
            'rhat': rhat_dict,
        }

        # Summary
        summary = {}
        for name, s in samples_constrained.items():
            summary[name] = {
                'mean': float(np.mean(s)), 'std': float(np.std(s)),
                'median': float(np.median(s)),
                'q5': float(np.percentile(s, 5)),
                'q25': float(np.percentile(s, 25)),
                'q75': float(np.percentile(s, 75)),
                'q95': float(np.percentile(s, 95)),
                'min': float(np.min(s)), 'max': float(np.max(s)),
            }

        self.diff_model.restore_parameters()

        step_times_arr = np.array(step_times)
        timing = {
            'step_times': step_times_arr.tolist(),
            'total_time_seconds': float(np.sum(step_times_arr)),
            'mean_step_time': float(np.mean(step_times_arr)),
            'median_step_time': float(np.median(step_times_arr)),
            'min_step_time': float(np.min(step_times_arr)),
            'max_step_time': float(np.max(step_times_arr)),
            'burnin_time_seconds': float(np.sum(step_times_arr[:num_burnin])),
            'sampling_time_seconds': float(np.sum(step_times_arr[num_burnin:])),
        }

        print(f"MH complete!")

        return DPFResult(
            samples=samples_constrained,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                'model_type': self.base_model.__class__.__name__,
                'filter_type': self.filter_class.__name__,
                'sampler': 'mh',
                'num_samples': num_samples,
                'num_burnin': num_burnin,
                'proposal_std': proposal_std,
                'num_observations': len(observations)
                    if hasattr(observations, '__len__') else 0,
                'timing': timing,
                'trace': trace_log or [],
            },
        )
