"""Particle Marginal Metropolis-Hastings (PMMH) for parameter inference.

Implements the PMMH algorithm from Andrieu, Doucet & Holenstein (2010).
No gradients needed — uses the particle filter's marginal likelihood estimate
directly in a Metropolis-Hastings random walk.

Key differences from HMC-based DPFRunner:
  - No DifferentiableModel wrapper (no gradient chain needed)
  - Fresh random seed each particle filter call (stochasticity is part of the algorithm)
  - On rejection, keep the OLD likelihood estimate (don't re-evaluate)
  - Simple random walk proposal in constrained space
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from typing import Any, Dict, Optional, Type
from dataclasses import dataclass

from .types import ParameterSpec, DPFResult


class PMMHRunner:
    """
    PMMH sampler for parameter inference in state-space models.

    Algorithm (Andrieu et al. 2010, Section 2.4.2):
      1. Propose theta* = theta + eps, eps ~ N(0, Sigma_proposal)
      2. Run particle filter with theta* using a FRESH random seed
         -> get log p_hat(y | theta*)
      3. Accept with probability:
           min(1, p_hat(y|theta*) * p(theta*) / (p_hat(y|theta) * p(theta)))
      4. On rejection, keep old theta AND old likelihood estimate
    """

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
        self.param_specs = param_specs
        self.param_names = list(param_specs.keys())

        # Build priors and initial values
        self.priors = {name: spec.prior for name, spec in param_specs.items()}
        self.init_values = {name: spec.init_value for name, spec in param_specs.items()}
        self.constraints = {name: spec.constraint for name, spec in param_specs.items()}

        self.dtype = getattr(base_model, 'dtype', tf.float32)

    def _set_model_params(self, param_values: Dict[str, float]):
        """Set parameters on the model."""
        for name, value in param_values.items():
            setattr(self.base_model, name, value)

    def _log_prior(self, param_values: Dict[str, float]) -> float:
        """Compute log prior for constrained parameters."""
        lp = 0.0
        for name, value in param_values.items():
            lp += float(self.priors[name].log_prob(value).numpy())
        return lp

    def _log_likelihood(self, observations_tf: tf.Tensor, seed: tf.Tensor) -> float:
        """Run particle filter and return log marginal likelihood estimate."""
        filter_obj = self.filter_class(self.base_model, **self.filter_kwargs)
        ll = filter_obj.log_marginal_likelihood_tf(observations_tf, seed=seed)
        return float(ll.numpy())

    def _propose(self, current: Dict[str, float], proposal_std: Dict[str, float],
                 rng: np.random.Generator) -> Dict[str, float]:
        """Propose new parameters via random walk in constrained space."""
        proposed = {}
        for name in self.param_names:
            val = current[name] + rng.normal(0, proposal_std[name])
            # Reflect at boundaries for positive constraint
            if self.constraints[name] == 'positive':
                val = abs(val) if val <= 0 else val
            proposed[name] = val
        return proposed

    def run_inference(
        self,
        observations: np.ndarray,
        num_samples: int = 1000,
        num_burnin: int = 500,
        proposal_std: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> DPFResult:
        """Run PMMH to sample from posterior p(theta | y).

        Args:
            observations: Array of observations, shape (T,) or (T, obs_dim).
            num_samples: Number of post-burn-in samples to collect.
            num_burnin: Number of burn-in iterations.
            proposal_std: Dict of proposal standard deviations per parameter.
                          Defaults to 5% of init_value for each parameter.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        observations_tf = tf.constant(observations, dtype=self.dtype)
        total_steps = num_burnin + num_samples

        # Default proposal std: 5% of initial value (following Andrieu et al.)
        if proposal_std is None:
            proposal_std = {
                name: max(0.01, abs(self.init_values[name]) * 0.05)
                for name in self.param_names
            }

        print(f"Using PMMH sampler")
        print(f"Running: {num_burnin} burn-in + {num_samples} sampling "
              f"= {total_steps} total steps")
        print(f"Proposal std: {proposal_std}")

        # Initialize at starting values
        current_params = dict(self.init_values)
        self._set_model_params(current_params)

        # Evaluate initial log-posterior (with first fresh seed)
        pf_seed = tf.constant([rng.integers(0, 2**31), 0], dtype=tf.int32)
        current_ll = self._log_likelihood(observations_tf, pf_seed)
        current_lp = self._log_prior(current_params)
        current_log_post = current_ll + current_lp

        param_str = ", ".join(f"{n}={v:.4f}" for n, v in current_params.items())
        print(f"  Initial: log_lik={current_ll:.4f}, log_prior={current_lp:.4f}, "
              f"log_post={current_log_post:.4f}")
        print(f"  Params: {param_str}")

        samples_list = []
        is_accepted_list = []
        step_times = []

        for i in range(total_steps):
            t_start = time.perf_counter()

            # 1. Propose new parameters
            proposed_params = self._propose(current_params, proposal_std, rng)

            # 2. Set proposed params and evaluate likelihood with FRESH seed
            self._set_model_params(proposed_params)
            pf_seed = tf.constant([rng.integers(0, 2**31), 0], dtype=tf.int32)
            proposed_ll = self._log_likelihood(observations_tf, pf_seed)
            proposed_lp = self._log_prior(proposed_params)
            proposed_log_post = proposed_ll + proposed_lp

            # 3. Metropolis accept/reject
            log_alpha = proposed_log_post - current_log_post
            log_u = np.log(rng.uniform())
            accepted = log_u < log_alpha

            if accepted:
                current_params = proposed_params
                current_ll = proposed_ll
                current_lp = proposed_lp
                current_log_post = proposed_log_post
            else:
                # Restore old params on model
                self._set_model_params(current_params)

            dt = time.perf_counter() - t_start
            step_times.append(dt)
            is_accepted_list.append(bool(accepted))

            if i >= num_burnin:
                samples_list.append(dict(current_params))

            # Progress
            phase = "burn-in" if i < num_burnin else "sample"
            idx = i - num_burnin + 1 if i >= num_burnin else i + 1
            phase_total = num_samples if i >= num_burnin else num_burnin

            recent = is_accepted_list[-min(50, len(is_accepted_list)):]
            accept_rate = np.mean(recent)

            avg_dt = np.mean(step_times[-10:])
            eta = avg_dt * (total_steps - i - 1)

            param_str = ", ".join(
                f"{n}={current_params[n]:.4f}" for n in self.param_names
            )
            print(f"  [{phase} {idx}/{phase_total}] "
                  f"{dt:.1f}s | accept={accept_rate:.0%} | "
                  f"ll={current_ll:.2f} | {param_str} | ETA {eta:.0f}s")

        # Build result
        samples_dict = {name: np.array([s[name] for s in samples_list])
                        for name in self.param_names}

        diagnostics = self._compute_diagnostics(
            samples_dict, is_accepted_list[num_burnin:]
        )
        summary = self._compute_summary(samples_dict)

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

        print(f"PMMH complete!")

        return DPFResult(
            samples=samples_dict,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                'model_type': self.base_model.__class__.__name__,
                'filter_type': self.filter_class.__name__,
                'sampler': 'pmmh',
                'num_samples': num_samples,
                'num_burnin': num_burnin,
                'proposal_std': proposal_std,
                'num_observations': len(observations),
                'timing': timing,
            }
        )

    def _compute_diagnostics(self, samples, is_accepted_post_burnin):
        diagnostics = {}
        diagnostics['acceptance_rate'] = float(np.mean(is_accepted_post_burnin))
        diagnostics['final_step_size'] = 0.0  # no step size in PMMH

        # ESS and R-hat via TFP
        import logging
        tf_logger = logging.getLogger('tensorflow')
        prev_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)

        ess_dict = {}
        for name, s in samples.items():
            param_tf = tf.constant(s[np.newaxis, :], dtype=self.dtype)
            ess = tfp.mcmc.effective_sample_size(param_tf)
            ess_dict[name] = float(ess.numpy()[0])
        diagnostics['ess'] = ess_dict

        rhat_dict = {}
        for name, s in samples.items():
            if len(s) >= 4:
                mid = len(s) // 2
                chains = tf.constant([s[:mid], s[mid:2*mid]], dtype=self.dtype)
                rhat_dict[name] = float(
                    tfp.mcmc.potential_scale_reduction(chains).numpy()
                )
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
                'q5': float(np.percentile(s, 5)),
                'q25': float(np.percentile(s, 25)),
                'q75': float(np.percentile(s, 75)),
                'q95': float(np.percentile(s, 95)),
                'min': float(np.min(s)), 'max': float(np.max(s)),
            }
        return summary
