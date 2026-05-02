"""Main HMC/NUTS/PMMH runner for parameter inference with differentiable filters."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from typing import Any, Callable, Dict, Optional, Type
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
        sampler: str = 'hmc',
        on_grad: Optional[Callable] = None,
        mass_vector: Optional[list] = None,
    ):
        """
        Initialize DPF runner.

        Args:
            base_model: State-space model instance
            filter_class: Filter class (e.g., ExtendedKalmanFilter)
            filter_kwargs: Keyword arguments for filter initialization
            param_specs: Dictionary of parameter specifications
            sampler: 'nuts', 'hmc', or 'custom_hmc'
            on_grad: Optional callback(step, nlp, grad) called after each
                gradient evaluation. Useful for diagnostics without polluting
                production logs.
            mass_vector: Optional diagonal mass matrix as a list of floats,
                one per parameter. Enables PreconditionedHamiltonianMonteCarlo
                with momentum p ~ N(0, diag(mass_vector)). Large m_i means
                smaller effective leapfrog step in dimension i — use for
                steep/narrow posterior directions.
        """
        self.base_model = base_model
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        self.sampler = sampler.lower()
        self.on_grad = on_grad
        self._grad_step = 0
        self.mass_vector = mass_vector

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

        nlp = -(log_likelihood + log_prior)
        return nlp

    def _value_and_grad(self, q):
        """Compute log-posterior and its gradient at q."""
        with tf.GradientTape() as tape:
            tape.watch(q)
            nlp = self._negative_log_posterior(q)
        grad = tape.gradient(nlp, q)
        # Replace None/NaN gradient with zeros
        if grad is None:
            tf.print("  [grad] WARNING: grad is None, using zeros")
            grad = tf.zeros_like(q)
        n_bad = tf.reduce_sum(tf.cast(~tf.math.is_finite(grad), tf.int32))
        tf.print("  [grad] nlp=", nlp, " |grad|=", tf.norm(grad),
                 " grad=", grad, " q=", q, " n_bad=", n_bad)
        if n_bad > 0:
            tf.print("  [grad] WARNING: replacing", n_bad, "non-finite grads with zeros")
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))

        if self.on_grad is not None:
            self.on_grad(step=self._grad_step, nlp=float(nlp.numpy()), grad=grad)
            self._grad_step += 1

        return -nlp, -grad  # return log_prob and grad_log_prob

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
        max_tree_depth: int = 10,
        grad_clip_norm: float = 100.0,
        step_count_smoothing: int = 10,
        pre_warmup_map_steps: int = 0,
        pre_warmup_map_lr: float = 0.01,
    ) -> DPFResult:
        """Run HMC or NUTS to sample from posterior p(theta | y).

        Args:
            grad_clip_norm: Max gradient L2 norm for custom_hmc sampler.
        """
        dtype = getattr(self.base_model, 'dtype', tf.float32)
        self._observations_tf = tf.constant(observations, dtype=dtype)

        if self.sampler == 'custom_hmc':
            return self._run_custom_hmc(
                num_samples=num_samples,
                num_burnin=num_burnin,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                target_accept_prob=target_accept_prob,
                grad_clip_norm=grad_clip_norm,
                seed=seed,
                dtype=dtype,
            )

        def target_log_prob_fn(unconstrained_params):
            return -self._negative_log_posterior(unconstrained_params)

        # Build momentum distribution for diagonal mass matrix (if specified)
        momentum_distribution = None
        if self.mass_vector is not None:
            scale_diag = tf.constant(
                [float(m) ** 0.5 for m in self.mass_vector], dtype=dtype
            )
            momentum_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(len(self.mass_vector), dtype=dtype),
                scale_diag=scale_diag,
            )
            print(f"  mass_vector={self.mass_vector}  (p ~ N(0, diag(m)))")

        # If step_size is a Python list (per-axis), convert to tensor matching state shape.
        # TFP expects either a scalar or a single tensor (not a Python list of scalars).
        if isinstance(step_size, (list, tuple)):
            step_size = tf.constant(step_size, dtype=dtype)

        # Choose sampler
        if self.sampler == 'nuts':
            print(f"Using NUTS sampler (max_tree_depth={max_tree_depth})")
            if momentum_distribution is not None:
                inner_kernel = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=step_size,
                    max_tree_depth=max_tree_depth,
                    momentum_distribution=momentum_distribution,
                )
            else:
                inner_kernel = tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=step_size,
                    max_tree_depth=max_tree_depth,
                )
        else:
            print(f"Using HMC sampler (num_leapfrog_steps={num_leapfrog_steps})")
            if momentum_distribution is not None:
                inner_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=step_size,
                    num_leapfrog_steps=num_leapfrog_steps,
                    momentum_distribution=momentum_distribution,
                )
            else:
                inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=step_size,
                    num_leapfrog_steps=num_leapfrog_steps,
                )

        # Adaptive step size
        # shrinkage_target=initial_step_size anchors DA at the initial value
        # (TFP takes log internally, so pass the value not its log).
        # step_count_smoothing (t0) damps DA's response to early accept signals;
        # default 10 matches Hoffman-Gelman/Stan; raise to ~100 for cliffy targets
        # where 100% accept persists for many warmup iterations.
        num_adaptation_steps = int(adaptation_rate * num_burnin)
        adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob,
            shrinkage_target=tf.constant(step_size, dtype=dtype),
            step_count_smoothing=step_count_smoothing,
        )

        if seed is not None:
            tf.random.set_seed(seed)

        total_steps = num_burnin + num_samples
        print(f"Running {self.sampler.upper()}: {num_burnin} burn-in + {num_samples} sampling = {total_steps} total steps")

        current_state = self.param_handler.unconstrained_init

        # Optional MAP pre-warmup: gradient ascent on log-posterior with Adam.
        # Moves chain to high-density region before HMC starts so that the
        # surface is locally sharp and DA does not run away on 100% accept.
        # Disabled when pre_warmup_map_steps == 0 (default).
        if pre_warmup_map_steps > 0:
            print(f"  [pre-warmup MAP] {pre_warmup_map_steps} Adam steps, lr={pre_warmup_map_lr}")
            map_var = tf.Variable(current_state, dtype=current_state.dtype)
            map_opt = tf.optimizers.Adam(learning_rate=pre_warmup_map_lr)
            for _i in range(pre_warmup_map_steps):
                with tf.GradientTape() as _tape:
                    _tape.watch(map_var)
                    _nlp = -target_log_prob_fn(map_var)
                _g = _tape.gradient(_nlp, map_var)
                if _g is None or not bool(tf.reduce_all(tf.math.is_finite(_g)).numpy()):
                    print(f"  [pre-warmup MAP] non-finite grad at step {_i}, stopping early")
                    break
                map_opt.apply_gradients([(_g, map_var)])
                if (_i + 1) % max(1, pre_warmup_map_steps // 5) == 0:
                    print(f"  [pre-warmup MAP step {_i+1}/{pre_warmup_map_steps}] "
                          f"lp={float(-_nlp.numpy()):.4f}, |grad|={float(tf.norm(_g).numpy()):.4f}, "
                          f"q={map_var.numpy()}")
            current_state = tf.constant(map_var.numpy(), dtype=current_state.dtype)
            print(f"  [pre-warmup MAP] done. Starting HMC from q={current_state.numpy()}")

        # One-time gradient diagnostic before HMC loop
        with tf.GradientTape() as _tape:
            _tape.watch(current_state)
            _lp = target_log_prob_fn(current_state)
        _grad = _tape.gradient(_lp, current_state)
        print(f"  [grad check] lp={float(_lp.numpy()):.4f}, "
              f"|grad|={float(tf.norm(_grad).numpy()) if _grad is not None else 'None'}, "
              f"grad={_grad.numpy() if _grad is not None else 'None'}")

        kernel_results = adaptive_kernel.bootstrap_results(current_state)

        samples_list = []
        is_accepted_list = []
        step_size_list = []
        step_times = []
        trace_log = []

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
                _ss_arr = kernel_results.new_step_size.numpy()
                if np.ndim(_ss_arr) == 0:
                    cur_step_size = float(_ss_arr)
                else:
                    cur_step_size = [float(x) for x in np.atleast_1d(_ss_arr)]

                is_accepted_list.append(accepted)
                step_size_list.append(cur_step_size)

                self._print_progress(
                    i, num_burnin, num_samples, total_steps,
                    dt, step_times, cur_step_size, current_state,
                    is_accepted_list, trace_log=trace_log
                )

                if i >= num_burnin:
                    samples_list.append(current_state.numpy().copy())

        return self._finalize(
            samples_list, is_accepted_list, step_size_list,
            step_times, num_burnin, num_samples, num_leapfrog_steps,
            current_state.dtype, observations, trace_log=trace_log
        )

    # Backward compat
    run_hmc = run_inference

    def run_map(
        self,
        observations: np.ndarray,
        num_steps: int = 200,
        learning_rate: float = 0.01,
        optimizer: str = 'adam',
        random_seed: bool = False,
        seed: int = 42,
        print_every: int = 10,
    ) -> DPFResult:
        """Find MAP estimate via Adam/SGD optimization.

        Minimizes -log p(theta | y) = -log p(y|theta) - log p(theta).
        Useful as a fast diagnostic to verify gradients point toward true params.

        Args:
            random_seed: If True, use a different PF seed each step (stochastic
                gradient, averages over many steps). If False, use fixed seed
                (deterministic surface, standard optimization).
        """
        dtype = getattr(self.base_model, 'dtype', tf.float32)
        self._observations_tf = tf.constant(observations, dtype=dtype)

        q = tf.Variable(self.param_handler.unconstrained_init, dtype=dtype)

        # Warmup → constant → cosine decay LR schedule
        from .lr_schedules import WarmupConstantCosineDecay
        lr_schedule = WarmupConstantCosineDecay(
            peak_lr=learning_rate, total_steps=num_steps,
        )

        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(lr_schedule)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(lr_schedule)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'adam' or 'sgd'.")

        def _current_learning_rate() -> float:
            lr = opt.learning_rate
            if callable(lr):
                lr = lr(opt.iterations)
            try:
                return float(tf.convert_to_tensor(lr).numpy())
            except (TypeError, ValueError):
                return float(learning_rate)

        print(f"Running MAP ({optimizer}, lr={learning_rate}, steps={num_steps}, "
              f"warmup={lr_schedule.warmup_steps}, decay_start={lr_schedule.warmup_steps + lr_schedule.constant_steps}, "
              f"random_seed={random_seed})")

        best_loss = float('inf')
        best_q = q.numpy().copy()
        loss_history = []
        log_likelihood_history = []
        log_prior_history = []
        grad_norm_history = []
        learning_rate_history = []
        param_history = {name: [] for name in self.param_handler.param_names}
        grad_history = {name: [] for name in self.param_handler.param_names}
        step_times = []

        for step in range(num_steps):
            t0 = time.perf_counter()

            with tf.GradientTape() as tape:
                constrained = self.param_handler.constrain(q)
                self.diff_model.update_parameters(constrained)
                pf_seed = tf.constant(
                    [seed, step] if random_seed else [seed, 0],
                    dtype=tf.int32,
                )
                ll = self.filter_obj.log_marginal_likelihood_tf(
                    self._observations_tf, seed=pf_seed
                )
                lp = self.param_handler.log_prior(constrained)
                loss = -(ll + lp)

            grad = tape.gradient(loss, q)
            if grad is None:
                grad = tf.zeros_like(q)
            grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))

            q_eval = q.numpy().copy()
            constrained_vals = {
                name: float(val.numpy()) for name, val in constrained.items()
            }
            grad_vals = grad.numpy()
            grad_norm = float(np.linalg.norm(grad_vals))
            lr_val = _current_learning_rate()

            opt.apply_gradients([(grad, q)])

            dt = time.perf_counter() - t0
            step_times.append(dt)

            loss_val = float(loss.numpy())
            loss_history.append(loss_val)
            log_likelihood_history.append(float(ll.numpy()))
            log_prior_history.append(float(lp.numpy()))
            grad_norm_history.append(grad_norm)
            learning_rate_history.append(lr_val)

            for i, name in enumerate(self.param_handler.param_names):
                param_history[name].append(constrained_vals[name])
                grad_history[name].append(float(grad_vals[i]))

            if loss_val < best_loss:
                best_loss = loss_val
                best_q = q_eval

            if step % print_every == 0 or step == num_steps - 1:
                param_str = ", ".join(
                    f"{n}={v:.4f}" for n, v in constrained_vals.items()
                )
                avg_dt = np.mean(step_times[-10:])
                eta = avg_dt * (num_steps - step - 1)
                print(f"  [step {step}/{num_steps}] loss={loss_val:.2f} | "
                      f"ll={float(ll.numpy()):.2f} | |grad|={grad_norm:.2f} | "
                      f"{param_str} | {dt:.1f}s | ETA {eta:.0f}s")

        # Build DPFResult from MAP point
        best_constrained = self.param_handler.constrain(
            tf.constant(best_q, dtype=dtype)
        )
        samples = {
            name: np.array([float(val.numpy())])
            for name, val in best_constrained.items()
        }

        summary = {}
        for name in self.param_handler.param_names:
            trace = np.array(param_history[name])
            last_n = trace[-max(1, num_steps // 5):]
            map_estimate = float(samples[name][0])
            summary[name] = {
                'map': map_estimate,
                'mean': float(np.mean(last_n)),
                'std': float(np.std(last_n)),
                'median': float(np.median(last_n)),
                'q5': float(np.percentile(last_n, 5)),
                'q25': float(np.percentile(last_n, 25)),
                'q75': float(np.percentile(last_n, 75)),
                'q95': float(np.percentile(last_n, 95)),
                'min': float(np.min(last_n)),
                'max': float(np.max(last_n)),
            }

        step_times_arr = np.array(step_times)
        diagnostics = {
            'final_loss': float(loss_history[-1]) if loss_history else float('nan'),
            'best_loss': float(best_loss),
            'final_log_likelihood': float(log_likelihood_history[-1])
                if log_likelihood_history else float('nan'),
            'final_log_prior': float(log_prior_history[-1])
                if log_prior_history else float('nan'),
            'final_grad_norm': float(grad_norm_history[-1])
                if grad_norm_history else float('nan'),
            'loss_history': loss_history,
            'log_likelihood_history': log_likelihood_history,
            'log_prior_history': log_prior_history,
            'grad_norm_history': grad_norm_history,
            'learning_rate_history': learning_rate_history,
            'converged': bool(
                np.std(loss_history[-max(1, num_steps // 10):]) < 1.0
            ),
        }

        self.diff_model.restore_parameters()

        print(f"MAP complete! Best loss={best_loss:.2f}")
        for name, val in best_constrained.items():
            print(f"  {name} = {float(val.numpy()):.4f}")

        return DPFResult(
            samples=samples,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                'model_type': self.base_model.__class__.__name__,
                'filter_type': self.filter_class.__name__,
                'sampler': 'map',
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'num_steps': num_steps,
                'random_seed': random_seed,
                'num_observations': len(observations)
                    if hasattr(observations, '__len__') else 0,
                'timing': {
                    'total_time_seconds': float(np.sum(step_times_arr)),
                    'mean_step_time': float(np.mean(step_times_arr)),
                    'step_times': step_times_arr.tolist(),
                },
                'param_history': param_history,
                'grad_history': grad_history,
            },
        )

    def _run_custom_hmc(
        self,
        num_samples,
        num_burnin,
        step_size,
        num_leapfrog_steps,
        target_accept_prob,
        grad_clip_norm,
        seed,
        dtype,
    ):
        """Custom HMC with gradient clipping for noisy particle filter gradients.

        Uses gradient norm clipping to prevent leapfrog trajectories from
        diverging when the particle filter likelihood surface has cliffs.
        Dual averaging adapts step_size during burn-in.
        """
        if seed is not None:
            tf.random.set_seed(seed)

        total_steps = num_burnin + num_samples
        print(f"Using Custom HMC (leapfrog={num_leapfrog_steps}, "
              f"grad_clip={grad_clip_norm})")
        print(f"Running: {num_burnin} burn-in + {num_samples} sampling "
              f"= {total_steps} total steps")

        q = self.param_handler.unconstrained_init

        # Evaluate at starting point
        lp, grad_lp = self._value_and_grad(q)
        print(f"  Initial: lp={lp.numpy():.4f}, |grad|={tf.norm(grad_lp).numpy():.1f}, "
              f"grad={grad_lp.numpy()}")

        # Dual averaging state for step size adaptation
        # (Hoffman & Gelman, 2014, Algorithm 5)
        log_eps = tf.math.log(tf.constant(step_size, dtype=dtype))
        log_eps_bar = tf.constant(0.0, dtype=dtype)
        H_bar = tf.constant(0.0, dtype=dtype)
        mu = tf.math.log(tf.constant(10.0 * step_size, dtype=dtype))
        gamma = tf.constant(0.05, dtype=dtype)
        t0 = tf.constant(10.0, dtype=dtype)
        kappa = tf.constant(0.75, dtype=dtype)
        num_adapt = int(0.8 * num_burnin)

        samples_list = []
        is_accepted_list = []
        step_size_list = []
        step_times = []
        trace_log = []
        n_accepted = 0

        for i in range(total_steps):
            t_start = time.perf_counter()
            eps = tf.exp(log_eps)

            # Draw momentum
            p = tf.random.normal(q.shape, dtype=dtype)
            current_lp, current_grad = self._value_and_grad(q)

            # Leapfrog integration with gradient clipping
            q_prop = tf.identity(q)
            p_prop = tf.identity(p)

            # Half step for momentum
            grad_clipped = tf.clip_by_norm(current_grad, grad_clip_norm)
            p_prop = p_prop + 0.5 * eps * grad_clipped

            for _ in range(num_leapfrog_steps):
                # Full step for position
                q_prop = q_prop + eps * p_prop
                # Full step for momentum (except at end)
                _, g = self._value_and_grad(q_prop)
                g = tf.clip_by_norm(g, grad_clip_norm)
                p_prop = p_prop + eps * g

            # Undo last full momentum step, do half step instead
            _, g = self._value_and_grad(q_prop)
            g = tf.clip_by_norm(g, grad_clip_norm)
            p_prop = p_prop - eps * g  # undo
            p_prop = p_prop + 0.5 * eps * g  # half step

            # Compute energies
            prop_lp, _ = self._value_and_grad(q_prop)
            H_start = -current_lp + 0.5 * tf.reduce_sum(p**2)
            H_end = -prop_lp + 0.5 * tf.reduce_sum(p_prop**2)
            dH = H_end - H_start
            log_alpha = tf.minimum(tf.constant(0.0, dtype=dtype), -dH)
            accept_prob = tf.exp(log_alpha)

            # Metropolis accept/reject
            u = tf.random.uniform([], dtype=dtype)
            accepted = bool((u < accept_prob).numpy())
            if accepted:
                q = q_prop
                n_accepted += 1

            dt = time.perf_counter() - t_start

            # Dual averaging step size adaptation (during burn-in)
            m = tf.constant(float(i + 1), dtype=dtype)
            if i < num_adapt:
                w = tf.constant(1.0, dtype=dtype) / (m + t0)
                H_bar = (1.0 - w) * H_bar + w * (target_accept_prob - accept_prob)
                log_eps = mu - tf.sqrt(m) / gamma * H_bar
                m_kappa = tf.pow(m, -kappa)
                log_eps_bar = m_kappa * log_eps + (1.0 - m_kappa) * log_eps_bar
            elif i == num_adapt:
                log_eps = log_eps_bar  # fix step size after adaptation

            cur_eps = float(tf.exp(log_eps).numpy())
            is_accepted_list.append(accepted)
            step_size_list.append(cur_eps)
            step_times.append(dt)

            self._print_progress(
                i, num_burnin, num_samples, total_steps,
                dt, step_times, cur_eps, q, is_accepted_list,
                trace_log=trace_log
            )

            if i >= num_burnin:
                samples_list.append(q.numpy().copy())

        return self._finalize(
            samples_list, is_accepted_list, step_size_list,
            step_times, num_burnin, num_samples, num_leapfrog_steps,
            dtype, self._observations_tf.numpy(), trace_log=trace_log
        )

    def _print_progress(self, i, num_burnin, num_samples, total_steps,
                        dt, step_times, cur_step_size, current_state,
                        is_accepted_list, trace_log=None):
        """Print progress for HMC step."""
        phase = "burn-in" if i < num_burnin else "sample"
        idx = i - num_burnin + 1 if i >= num_burnin else i + 1
        phase_total = num_samples if i >= num_burnin else num_burnin

        constrained = self.param_handler.constrain(current_state)
        param_str = ", ".join(
            f"{n}={float(v.numpy()):.4f}" for n, v in constrained.items()
        )

        avg_dt = np.mean(step_times[-10:])
        remaining = total_steps - (i + 1)
        eta = avg_dt * remaining

        accept_rate = np.mean(
            is_accepted_list[-min(50, len(is_accepted_list)):]
        )

        if isinstance(cur_step_size, list):
            ss_str = "[" + ",".join(f"{x:.4f}" for x in cur_step_size) + "]"
        else:
            ss_str = f"{cur_step_size:.4f}"
        print(f"  [{phase} {idx}/{phase_total}] "
              f"{dt:.1f}s | accept={accept_rate:.0%} | step_size={ss_str} | "
              f"{param_str} | ETA {eta:.0f}s")

        if trace_log is not None:
            row = {
                'step': i,
                'phase': phase,
                'step_in_phase': idx,
                'dt': dt,
                'accept_rate': float(accept_rate),
                'step_size': cur_step_size,
            }
            for n, v in constrained.items():
                row[n] = float(v.numpy())
            trace_log.append(row)

    def _finalize(self, samples_list, is_accepted_list, step_size_list,
                  step_times, num_burnin, num_samples, num_leapfrog_steps,
                  dtype, observations, trace_log=None):
        """Post-process samples and compute diagnostics."""
        samples_unconstrained = tf.constant(
            np.stack(samples_list),
            dtype=dtype if not isinstance(dtype, np.dtype) else tf.float32
        )
        trace_is_accepted = tf.constant(is_accepted_list[num_burnin:])
        trace_step_sizes = tf.constant(step_size_list[num_burnin:])

        samples_constrained = self._transform_samples(samples_unconstrained)
        diagnostics = self._compute_diagnostics(
            samples_constrained, trace_is_accepted, trace_step_sizes
        )
        summary = self._compute_summary(samples_constrained)
        self.diff_model.restore_parameters()

        print(f"{self.sampler.upper()} complete!")

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
            'mean_time_per_gradient_eval': float(
                np.mean(step_times_arr) / max(num_leapfrog_steps, 1)
            ),
        }

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
                'num_leapfrog_steps': num_leapfrog_steps,
                'num_observations': len(observations)
                    if hasattr(observations, '__len__') else 0,
                'timing': timing,
                'trace': trace_log or [],
            }
        )

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
        _final_ss = step_sizes[-1].numpy()
        if np.ndim(_final_ss) == 0:
            diagnostics['final_step_size'] = float(_final_ss)
        else:
            diagnostics['final_step_size'] = [float(x) for x in np.atleast_1d(_final_ss)]

        # Suppress complex64->float32 warning from FFT-based autocorrelation in ESS
        import logging
        tf_logger = logging.getLogger('tensorflow')
        prev_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)

        ess_dict = {}
        for name, s in samples.items():
            param_tf = tf.constant(s[:, np.newaxis], dtype=dtype)
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
