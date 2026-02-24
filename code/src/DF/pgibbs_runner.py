"""Particle Gibbs + HMC runner for joint parameter and state inference.

Alternates two steps:
  1. theta-step: HMC on closed-form log p(theta | x_{0:T}, y_{1:T})
     - Smooth Gaussian sum, no particle filter, no gradient cliffs
  2. x-step: Conditional SMC to sample x_{0:T} | theta, y_{1:T}
     - Bootstrap CSMC or LEDH CSMC, with ancestor sampling
     - No gradients needed

See hmc_dpf_notes.md for full mathematical derivation.
"""

import tensorflow as tf
import numpy as np
import math
import time
from typing import Any, Dict, Optional, Type

from .types import ParameterSpec, DPFResult
from .parameter_handler import ParameterHandler
from .differentiable_model import DifferentiableModel


class PGibbsRunner:
    """
    Particle Gibbs + HMC for joint inference of parameters and latent states.

    The theta-step uses HMC on the closed-form conditional posterior:
        log p(theta | x, y) = sum_t log p(x_t | x_{t-1}, theta)
                             + sum_t log p(y_t | x_t, theta)
                             + log prior(theta)

    The x-step uses Conditional SMC (bootstrap or LEDH) to sample
    a new trajectory given the current theta.
    """

    def __init__(
        self,
        base_model: Any,
        csmc_class: Type,
        csmc_kwargs: Dict[str, Any],
        param_specs: Dict[str, ParameterSpec],
        init_filter_class: Optional[Type] = None,
        init_filter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            base_model: StateSpaceModel instance
            csmc_class: CSMC filter class (BootstrapConditionalSMC or LEDHConditionalSMC)
            csmc_kwargs: kwargs for CSMC construction (excluding model)
            param_specs: Dict[str, ParameterSpec] for trainable parameters
            init_filter_class: Optional filter class for initial trajectory
                              (defaults to bootstrap PF if None)
            init_filter_kwargs: kwargs for init filter
        """
        self.base_model = base_model
        self.csmc_class = csmc_class
        self.csmc_kwargs = csmc_kwargs
        self.init_filter_class = init_filter_class
        self.init_filter_kwargs = init_filter_kwargs or {}

        self.dtype = getattr(base_model, 'dtype', tf.float64)

        # Wrap model for differentiable theta-step
        trainable_param_names = list(param_specs.keys())
        self.diff_model = DifferentiableModel(base_model, trainable_param_names)

        # Bijectors and priors
        self.param_handler = ParameterHandler(param_specs, dtype=self.dtype)

        # Create CSMC filter
        self.csmc = self.csmc_class(model=self.diff_model, **self.csmc_kwargs)

    def _theta_log_posterior(
        self,
        unconstrained_params: tf.Tensor,
        trajectory: tf.Tensor,
        observations: tf.Tensor,
    ) -> tf.Tensor:
        """
        Closed-form log p(theta | x_{0:T}, y_{1:T}).

        No particle filter. Just a sum of Gaussian log-densities.

        Args:
            unconstrained_params: (num_params,) in unconstrained space
            trajectory: (T+1, state_dim) — x_0, x_1, ..., x_T
            observations: (T, obs_dim) — y_1, ..., y_T

        Returns:
            Scalar log-posterior value
        """
        # 1. Transform to constrained space
        constrained = self.param_handler.constrain(unconstrained_params)
        self.diff_model.update_parameters(constrained)

        T = observations.shape[0]
        log_prob = tf.constant(0.0, dtype=self.dtype)

        # 2. Initial state term: log p(x_0)
        x_0 = trajectory[0]
        mu_0 = self.diff_model.mu_0
        Sigma_0 = self.diff_model.Sigma_0
        log_prob = log_prob + self._log_gaussian(x_0, mu_0, Sigma_0)

        # 3. Transition terms: sum_t log p(x_t | x_{t-1}, theta)
        Q = self.diff_model.process_noise_cov  # (sd, sd), depends on theta
        for t in range(1, T + 1):
            x_prev = trajectory[t - 1]
            x_curr = trajectory[t]
            mean = self.diff_model.state_transition_mean(x_prev, t=t)
            log_prob = log_prob + self._log_gaussian(x_curr, mean, Q)

        # 4. Observation terms: sum_t log p(y_t | x_t, theta)
        for t in range(T):
            x_t = trajectory[t + 1]
            y_t = observations[t]
            log_obs = self.diff_model.log_observation_prob(y_t, x_t)
            log_prob = log_prob + log_obs

        # 5. Prior + Jacobian adjustment
        log_prior = self.param_handler.log_prior(constrained)
        log_prob = log_prob + log_prior

        return log_prob

    def _log_gaussian(self, x, mean, cov):
        """Log probability of multivariate Gaussian N(x; mean, cov)."""
        diff = x - mean
        d = tf.cast(tf.shape(x)[-1], self.dtype)
        sign, logdet = tf.linalg.slogdet(cov)
        L = tf.linalg.cholesky(cov)
        y = tf.linalg.triangular_solve(L, diff[..., tf.newaxis], lower=True)[..., 0]
        mahal = tf.reduce_sum(y**2, axis=-1)
        return -0.5 * (d * tf.math.log(
            tf.constant(2.0 * math.pi, dtype=self.dtype)
        ) + logdet + mahal)

    def _theta_step_mh(
        self,
        q: tf.Tensor,
        trajectory: tf.Tensor,
        observations: tf.Tensor,
        proposal_std: float,
    ):
        """
        One random-walk Metropolis-Hastings step for theta (unconstrained space).

        Symmetric proposal: q' = q + N(0, proposal_std^2 I).
        Reproduces the PMCMC paper's theta-step (Andrieu et al. 2010).

        Returns:
            new_q, accepted (bool), accept_prob (float)
        """
        # Propose
        eps = tf.random.normal(q.shape, dtype=self.dtype) * proposal_std
        q_prop = q + eps

        # Log posterior at current and proposed
        lp_current = self._theta_log_posterior(q, trajectory, observations)
        lp_proposed = self._theta_log_posterior(q_prop, trajectory, observations)

        # MH accept/reject (symmetric proposal → ratio = target ratio)
        log_alpha = tf.minimum(
            tf.constant(0.0, dtype=self.dtype),
            lp_proposed - lp_current
        )
        accept_prob = tf.exp(log_alpha)
        u = tf.random.uniform([], dtype=self.dtype)
        accepted = bool((u < accept_prob).numpy())

        if accepted:
            return q_prop, True, float(accept_prob.numpy())
        else:
            return q, False, float(accept_prob.numpy())

    def _theta_step_hmc(
        self,
        q: tf.Tensor,
        trajectory: tf.Tensor,
        observations: tf.Tensor,
        step_size: float,
        num_leapfrog_steps: int,
        grad_clip_norm: float,
    ):
        """
        One HMC step for theta given fixed trajectory.

        Returns:
            new_q: proposed (and possibly accepted) unconstrained params
            accepted: bool
            accept_prob: float
        """
        # Current log-posterior and gradient
        with tf.GradientTape() as tape:
            tape.watch(q)
            lp = self._theta_log_posterior(q, trajectory, observations)
        grad_lp = tape.gradient(lp, q)
        if grad_lp is None:
            grad_lp = tf.zeros_like(q)
        grad_lp = tf.where(tf.math.is_finite(grad_lp), grad_lp, tf.zeros_like(grad_lp))

        eps = tf.constant(step_size, dtype=self.dtype)

        # Draw momentum
        p = tf.random.normal(q.shape, dtype=self.dtype)

        # Leapfrog with gradient clipping
        q_prop = tf.identity(q)
        p_prop = tf.identity(p)

        # Half step for momentum
        g = tf.clip_by_norm(grad_lp, grad_clip_norm)
        p_prop = p_prop + 0.5 * eps * g

        for _ in range(num_leapfrog_steps):
            q_prop = q_prop + eps * p_prop
            with tf.GradientTape() as tape:
                tape.watch(q_prop)
                lp_step = self._theta_log_posterior(q_prop, trajectory, observations)
            g = tape.gradient(lp_step, q_prop)
            if g is None:
                g = tf.zeros_like(q_prop)
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            g = tf.clip_by_norm(g, grad_clip_norm)
            p_prop = p_prop + eps * g

        # Final half step (undo last full, do half)
        with tf.GradientTape() as tape:
            tape.watch(q_prop)
            lp_final = self._theta_log_posterior(q_prop, trajectory, observations)
        g = tape.gradient(lp_final, q_prop)
        if g is None:
            g = tf.zeros_like(q_prop)
        g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        g = tf.clip_by_norm(g, grad_clip_norm)
        p_prop = p_prop - eps * g  # undo last full
        p_prop = p_prop + 0.5 * eps * g  # half step

        # Metropolis accept/reject
        H_start = -lp + 0.5 * tf.reduce_sum(p**2)
        H_end = -lp_final + 0.5 * tf.reduce_sum(p_prop**2)
        dH = H_end - H_start
        log_alpha = tf.minimum(tf.constant(0.0, dtype=self.dtype), -dH)
        accept_prob = tf.exp(log_alpha)

        u = tf.random.uniform([], dtype=self.dtype)
        accepted = bool((u < accept_prob).numpy())

        if accepted:
            return q_prop, True, float(accept_prob.numpy())
        else:
            return q, False, float(accept_prob.numpy())

    def _x_step_csmc(self, observations, trajectory, seed):
        """Run CSMC to sample new trajectory given current theta."""
        # Update model params to current values (already set by theta-step)
        new_trajectory = self.csmc.run(
            observations=observations,
            reference_trajectory=trajectory,
            seed=seed,
        )
        return new_trajectory

    def _initialize_trajectory(self, observations, seed=42):
        """Get initial trajectory by running a bootstrap PF and sampling one path.

        If init_filter_class is provided, uses that. Otherwise, uses the CSMC
        filter itself with a dummy reference trajectory drawn from the prior.
        """
        T = observations.shape[0]
        sd = self.base_model.state_dim

        if self.init_filter_class is not None:
            # Use provided filter for initialization
            init_filter = self.init_filter_class(
                model=self.diff_model, **self.init_filter_kwargs
            )
            # Run filter and extract trajectory
            from ..core.types import FilterResult
            result = init_filter.filter(
                observations.numpy() if isinstance(observations, tf.Tensor) else observations,
            )
            # Use the filtered means as initial trajectory
            # (T, sd) for means, prepend x_0 = mu_0
            means = result.means  # (T, sd)
            x_0 = self.base_model.mu_0.numpy().reshape(1, sd)
            trajectory = np.concatenate([x_0, means], axis=0)  # (T+1, sd)
            return tf.constant(trajectory, dtype=self.dtype)

        # Default: sample trajectory from prior (x_0 ~ p(x_0), x_t ~ p(x_t|x_{t-1}))
        rng_key = tf.constant([seed, 0], dtype=tf.int32)
        keys = tf.random.experimental.stateless_split(rng_key, num=T + 1)

        x_0 = self.base_model.sample_initial_state(keys[0])
        trajectory_list = [x_0]
        x = x_0
        for t in range(T):
            if hasattr(self.base_model, 't'):
                self.base_model.t = t + 1
            x = self.base_model.sample_state_transition(x, keys[t + 1])
            trajectory_list.append(x)

        return tf.stack(trajectory_list)  # (T+1, sd)

    def run_inference(
        self,
        observations: np.ndarray,
        num_samples: int = 1000,
        num_burnin: int = 500,
        theta_sampler: str = 'hmc',
        hmc_steps_per_iter: int = 5,
        hmc_step_size: float = 0.05,
        hmc_num_leapfrog: int = 10,
        target_accept_prob: float = 0.8,
        grad_clip_norm: float = 100.0,
        mh_proposal_std: float = 0.1,
        mh_steps_per_iter: int = 1,
        seed: int = 42,
    ) -> DPFResult:
        """
        Run Particle Gibbs inference with HMC or MH theta-step.

        Args:
            observations: (T, obs_dim) observation array
            num_samples: number of post-burnin samples to collect
            num_burnin: number of burn-in iterations
            theta_sampler: 'hmc' for HMC theta-step, 'mh' for random-walk MH
            hmc_steps_per_iter: HMC steps per Gibbs iteration (theta-step)
            hmc_step_size: initial HMC step size
            hmc_num_leapfrog: leapfrog steps per HMC step
            target_accept_prob: target acceptance rate for step size adaptation
            grad_clip_norm: gradient clip norm for HMC
            mh_proposal_std: proposal std in unconstrained space (MH only)
            mh_steps_per_iter: MH steps per Gibbs iteration
            seed: random seed

        Returns:
            DPFResult with posterior samples and diagnostics
        """
        tf.random.set_seed(seed)
        obs_tf = tf.constant(observations, dtype=self.dtype)
        total_iters = num_burnin + num_samples

        steps_per_iter = (hmc_steps_per_iter if theta_sampler == 'hmc'
                          else mh_steps_per_iter)

        sampler_label = theta_sampler.upper()
        print(f"PGibbs + {sampler_label}: {num_burnin} burn-in + {num_samples} "
              f"sampling = {total_iters} total iterations")
        if theta_sampler == 'hmc':
            print(f"  HMC: {hmc_steps_per_iter} steps/iter, "
                  f"{hmc_num_leapfrog} leapfrog, step_size={hmc_step_size}")
        else:
            print(f"  MH: {mh_steps_per_iter} steps/iter, "
                  f"proposal_std={mh_proposal_std}")

        # Initialize theta
        q = self.param_handler.unconstrained_init

        # Set initial model parameters
        constrained = self.param_handler.constrain(q)
        self.diff_model.update_parameters(constrained)

        # Initialize trajectory
        print("Initializing trajectory...")
        trajectory = self._initialize_trajectory(obs_tf, seed=seed)
        print(f"  Initial trajectory shape: {trajectory.shape}")

        # Dual averaging for step size adaptation
        log_eps = tf.math.log(tf.constant(hmc_step_size, dtype=self.dtype))
        log_eps_bar = tf.constant(0.0, dtype=self.dtype)
        H_bar = tf.constant(0.0, dtype=self.dtype)
        mu = tf.math.log(tf.constant(10.0 * hmc_step_size, dtype=self.dtype))
        gamma = tf.constant(0.05, dtype=self.dtype)
        t0 = tf.constant(10.0, dtype=self.dtype)
        kappa = tf.constant(0.75, dtype=self.dtype)
        num_adapt = int(0.8 * num_burnin)

        # Storage
        samples_list = []
        theta_accepted_list = []
        step_size_list = []
        step_times = []
        trace_log = []

        for iteration in range(total_iters):
            t_start = time.perf_counter()
            cur_eps = float(tf.exp(log_eps).numpy())

            # === THETA-STEP: HMC or MH on closed-form log p(theta | x, y) ===
            n_theta_accepted = 0
            for step_i in range(steps_per_iter):
                if theta_sampler == 'hmc':
                    q, accepted, accept_prob = self._theta_step_hmc(
                        q, trajectory, obs_tf,
                        step_size=cur_eps,
                        num_leapfrog_steps=hmc_num_leapfrog,
                        grad_clip_norm=grad_clip_norm,
                    )
                else:
                    q, accepted, accept_prob = self._theta_step_mh(
                        q, trajectory, obs_tf,
                        proposal_std=mh_proposal_std,
                    )
                if accepted:
                    n_theta_accepted += 1

                # Update model parameters after each step
                constrained = self.param_handler.constrain(q)
                self.diff_model.update_parameters(constrained)

            theta_accept_rate = n_theta_accepted / steps_per_iter
            theta_accepted_list.append(theta_accept_rate)

            # === X-STEP: CSMC to sample new trajectory ===
            csmc_seed = seed + iteration + 1
            trajectory = self._x_step_csmc(obs_tf, trajectory, csmc_seed)

            dt = time.perf_counter() - t_start
            step_times.append(dt)

            # Step size adaptation (dual averaging, HMC only)
            if theta_sampler == 'hmc':
                m = tf.constant(float(iteration + 1), dtype=self.dtype)
                if iteration < num_adapt:
                    w = tf.constant(1.0, dtype=self.dtype) / (m + t0)
                    H_bar = (1.0 - w) * H_bar + w * (target_accept_prob - accept_prob)
                    log_eps = mu - tf.sqrt(m) / gamma * H_bar
                    m_kappa = tf.pow(m, -kappa)
                    log_eps_bar = m_kappa * log_eps + (1.0 - m_kappa) * log_eps_bar
                elif iteration == num_adapt:
                    log_eps = log_eps_bar
                cur_eps = float(tf.exp(log_eps).numpy())

            step_size_list.append(cur_eps if theta_sampler == 'hmc'
                                  else mh_proposal_std)

            # Print progress
            phase = "burn-in" if iteration < num_burnin else "sample"
            idx = iteration - num_burnin + 1 if iteration >= num_burnin else iteration + 1
            phase_total = num_samples if iteration >= num_burnin else num_burnin

            param_str = ", ".join(
                f"{n}={float(v.numpy()):.4f}" for n, v in constrained.items()
            )

            avg_dt = np.mean(step_times[-10:])
            remaining = total_iters - (iteration + 1)
            eta = avg_dt * remaining

            recent_accept = np.mean(
                theta_accepted_list[-min(50, len(theta_accepted_list)):]
            )

            step_info = (f"step_size={cur_eps:.4f}" if theta_sampler == 'hmc'
                        else f"proposal_std={mh_proposal_std:.4f}")
            print(f"  [{phase} {idx}/{phase_total}] "
                  f"{dt:.1f}s | theta_accept={recent_accept:.0%} | "
                  f"{step_info} | {param_str} | ETA {eta:.0f}s",
                  flush=True)

            row = {
                'step': iteration,
                'phase': phase,
                'step_in_phase': idx,
                'dt': dt,
                'accept_rate': float(recent_accept),
                'step_size': cur_eps if theta_sampler == 'hmc' else mh_proposal_std,
            }
            for n, v in constrained.items():
                row[n] = float(v.numpy())
            trace_log.append(row)

            # Collect post-burnin samples
            if iteration >= num_burnin:
                samples_list.append(q.numpy().copy())

        # Finalize
        return self._finalize(
            samples_list, theta_accepted_list, step_size_list,
            step_times, num_burnin, num_samples, observations,
            theta_sampler, trace_log=trace_log
        )

    def _finalize(self, samples_list, accepted_list, step_size_list,
                  step_times, num_burnin, num_samples, observations,
                  theta_sampler='hmc', trace_log=None):
        """Post-process samples and compute diagnostics."""
        import tensorflow_probability as tfp
        import logging

        samples_unconstrained = tf.constant(np.stack(samples_list), dtype=self.dtype)

        # Transform to constrained space
        samples_constrained = {}
        for name in self.param_handler.param_names:
            samples_constrained[name] = []
        for i in range(samples_unconstrained.shape[0]):
            constrained = self.param_handler.constrain(samples_unconstrained[i])
            for name, value in constrained.items():
                samples_constrained[name].append(float(value.numpy()))
        samples_constrained = {
            name: np.array(vals) for name, vals in samples_constrained.items()
        }

        # Summary statistics
        summary = {}
        for name, s in samples_constrained.items():
            summary[name] = {
                'mean': float(np.mean(s)), 'std': float(np.std(s)),
                'median': float(np.median(s)),
                'q5': float(np.percentile(s, 5)),
                'q25': float(np.percentile(s, 25)),
                'q75': float(np.percentile(s, 75)),
                'q95': float(np.percentile(s, 95)),
                'min': float(np.min(s)), 'max': float(np.max(s))
            }

        # Diagnostics
        post_burnin_accept = accepted_list[num_burnin:]
        acceptance_rate = float(np.mean(post_burnin_accept)) if post_burnin_accept else 0.0

        # ESS
        tf_logger = logging.getLogger('tensorflow')
        prev_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)

        ess_dict = {}
        for name, s in samples_constrained.items():
            if len(s) >= 10:
                param_tf = tf.constant(s[np.newaxis, :], dtype=self.dtype)
                ess_val = float(tfp.mcmc.effective_sample_size(param_tf).numpy()[0])
                ess_dict[name] = ess_val if np.isfinite(ess_val) else 'insufficient sample size'
            else:
                ess_dict[name] = 'insufficient sample size'

        # R-hat
        rhat_dict = {}
        for name, s in samples_constrained.items():
            if len(s) >= 4:
                mid = len(s) // 2
                chains = tf.constant([s[:mid], s[mid:2*mid]], dtype=self.dtype)
                rhat_val = float(tfp.mcmc.potential_scale_reduction(chains).numpy())
                rhat_dict[name] = rhat_val if np.isfinite(rhat_val) else 'insufficient sample size'
            else:
                rhat_dict[name] = 'insufficient sample size'

        tf_logger.setLevel(prev_level)

        diagnostics = {
            'acceptance_rate': acceptance_rate,
            'final_step_size': step_size_list[-1] if step_size_list else 0.0,
            'ess': ess_dict,
            'rhat': rhat_dict,
        }

        step_times_arr = np.array(step_times)
        timing = {
            'total_time_seconds': float(np.sum(step_times_arr)),
            'mean_step_time': float(np.mean(step_times_arr)),
            'step_times': step_times_arr.tolist(),
        }

        self.diff_model.restore_parameters()

        print(f"\nPGibbs + {theta_sampler.upper()} complete!")
        print(f"  Theta acceptance: {acceptance_rate:.1%}")

        return DPFResult(
            samples=samples_constrained,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                'model_type': self.base_model.__class__.__name__,
                'filter_type': self.csmc_class.__name__,
                'sampler': 'pgibbs',
                'theta_sampler': theta_sampler,
                'num_samples': num_samples,
                'num_burnin': num_burnin,
                'num_observations': len(observations),
                'timing': timing,
                'trace': trace_log or [],
            }
        )
