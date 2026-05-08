"""Stan-style windowed HMC runner with iterative mass matrix + step adaptation.

Implements the algorithmic recipe from `to_be_addressed/stan_hmc_runner_plan.md`
(verified against Stan source `stan/mcmc/windowed_adaptation.hpp`,
`var_adaptation.hpp`, `covar_adaptation.hpp`, develop branch).

Phases:
  1. Initial buffer        — DA tunes step at identity M, no M update.
  2. Adaptation windows    — collect samples, end of window: estimate M from
                             window samples, run FindReasonableEpsilon at the
                             new M to seed DA, fresh DA state.
  3. Terminal buffer       — M locked, finalize step.
  4. Sampling phase        — fixed (M, ε), no adaptation.

This runner addresses the failure mode documented in
`to_be_addressed/FindReasonableEpsilon.md`: the offline single-pass mass tuning
(`hmc_runner.py` with a fixed `mass_vector`) crashed because DA started in a
regime where random momentum dominated gradient signal, drove the step into the
OT-backward-singular zone, and the OT MatrixSolve crashed at burn-in step 19.
Stan avoids this by re-running FindReasonableEpsilon every time M changes.

NOT a replacement for `hmc_runner.py`. The two coexist; `runner: stan` in YAML
selects this one.

Fixed-trajectory resonance warning
----------------------------------

This is a fixed-L HMC implementation (no NUTS-style dynamic termination). For
Gaussian-like posteriors, the per-axis dynamics are sinusoidal at frequency
``ω_d = sqrt(P_d / M_d)`` and the integration time is ``τ_d = L * ε * ω_d``.
When ``τ_d`` lands near ``2π * k`` for any integer ``k``, that axis becomes
sticky (``cos(τ_d) ≈ 1`` returns the position to itself) and ESS collapses on
that axis even when other axes mix fine. Symptom: chain mean for one parameter
biased toward the warmup-end position while other parameters converge cleanly.

Stan avoids this with NUTS (dynamic L per iteration). Static-L HMC has to
dodge resonance explicitly. Practical guidance:

- Avoid round values of L if you can (L=11, 13, 17 instead of L=10, 20).
- Jitter L per iteration if you really need fixed-trajectory HMC. (Not
  implemented in v1.)
- If a chain looks sticky on one axis with the right M-tuning, suspect
  resonance before suspecting a code bug. Print
  ``τ = num_leapfrog_steps * eps * sqrt((1/posterior_var) / M)`` per axis
  and check whether any τ is near ``2π * k``.
"""

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .types import ParameterSpec, DPFResult
from .parameter_handler import ParameterHandler
from .differentiable_model import DifferentiableModel


# =============================================================================
# 1. Metric abstraction (Section "1. Metric abstraction" in plan)
# =============================================================================

class Metric:
    """Abstract HMC metric. v1 only has DiagonalMetric."""

    def estimate_from_samples(self, samples: tf.Tensor,
                              shrinkage_alpha: float = 5.0,
                              shrinkage_target: float = 1e-3) -> "Metric":
        raise NotImplementedError

    def build_momentum_distribution(self) -> tfp.distributions.Distribution:
        raise NotImplementedError

    def kinetic_energy(self, p: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def step_for_leapfrog(self, eps: tf.Tensor) -> tf.Tensor:
        """Step size to use inside SimpleLeapfrogIntegrator. For diagonal M,
        leapfrog is `q += eps * M^-1 p`, so we just pass `eps`; M is handled by
        the momentum distribution."""
        raise NotImplementedError


class DiagonalMetric(Metric):
    """Diagonal mass matrix. M is a 1-D tensor of size dim."""

    def __init__(self, M: tf.Tensor):
        self.M = tf.cast(M, tf.float32) if M.dtype != tf.float32 else M

    def estimate_from_samples(self, samples: tf.Tensor,
                              shrinkage_alpha: float = 5.0,
                              shrinkage_target: float = 1e-3) -> "DiagonalMetric":
        """Stan's exact diagonal shrinkage formula (var_adaptation.hpp):

            var_shrunk = (n / (n + alpha)) * var
                         + shrinkage_target * (alpha / (n + alpha)) * 1

        - alpha (default 5.0): Stan's shrinkage constant.
        - shrinkage_target (default 1e-3): numerical safety floor (NOT a prior).

        Uses unbiased sample variance (Bessel's correction) to match Stan's
        Welford output. M = 1 / var_shrunk.
        """
        samples = tf.convert_to_tensor(samples)
        n = tf.cast(tf.shape(samples)[0], samples.dtype)
        if n < 2:
            warnings.warn("DiagonalMetric.estimate_from_samples called with n<2; "
                          "returning identity.")
            return DiagonalMetric(tf.ones(samples.shape[1], dtype=samples.dtype))

        mean = tf.reduce_mean(samples, axis=0)
        var = tf.reduce_sum((samples - mean) ** 2, axis=0) / (n - 1.0)

        alpha = tf.cast(shrinkage_alpha, samples.dtype)
        target = tf.cast(shrinkage_target, samples.dtype)
        var_shrunk = ((n / (n + alpha)) * var
                      + target * (alpha / (n + alpha)))
        return DiagonalMetric(M=1.0 / var_shrunk)

    def build_momentum_distribution(self) -> tfp.distributions.Distribution:
        # Conventional HMC: p ~ N(0, M). Then K(p) = 0.5 * p^T M^-1 p, and
        # leapfrog updates q with eps * M^-1 * p. This matches Stan's "metric"
        # convention where M is the inverse posterior covariance.
        return tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.M),
            scale_diag=tf.sqrt(self.M),
        )

    def kinetic_energy(self, p: tf.Tensor) -> tf.Tensor:
        return 0.5 * tf.reduce_sum(p * p / self.M)

    def step_for_leapfrog(self, eps: tf.Tensor) -> tf.Tensor:
        return eps


@dataclass
class WarmupInit:
    """Initial state for warmup. For Mode A (default), call `default_init`.
    For Mode B (v2, shared M after chain 1), pass adapted (metric, eps_init)
    here and set skip_windows_below to skip slow adaptation windows."""

    metric: Metric
    eps_init: float = 1.0
    skip_windows_below: int = 0


def default_warmup_init(dim: int, dtype=tf.float32) -> WarmupInit:
    """Mode A default: identity diagonal metric, eps_init=1.0 (gets overridden
    by FindReasonableEpsilon at the start of warmup)."""
    return WarmupInit(metric=DiagonalMetric(tf.ones(dim, dtype=dtype)),
                      eps_init=1.0)


# =============================================================================
# 2. find_reasonable_epsilon (Section "2." in plan)
# =============================================================================

def _one_leapfrog_trajectory(target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
                              q: tf.Tensor, p: tf.Tensor, eps: float,
                              metric: Metric, num_leapfrog: int
                              ) -> Tuple[tf.Tensor, tf.Tensor, float, bool]:
    """Run one HMC trajectory of `num_leapfrog` leapfrog steps.

    Returns (q_final, p_final, dh, finite). `dh` is the change in Hamiltonian
    H = -log_p + 0.5 p^T M^-1 p; `finite` is True if all values are finite.

    Wraps the body in try/except to catch graph-level exceptions raised by
    things like the OT backward MatrixSolve when the transport plan is sharp
    enough to make K singular. On a raised exception, returns the input q
    (no move), inf dh, finite=False — so FindReasonableEpsilon's hard-bracket
    logic and the warmup loop's finite check both fire correctly. Without
    the catch, an OT crash propagates and kills the entire chain.
    """
    try:
        eps_t = tf.cast(eps, q.dtype)

        def potential_and_grad(q_):
            with tf.GradientTape() as tape:
                tape.watch(q_)
                log_p = target_log_prob_fn(q_)
            grad_log_p = tape.gradient(log_p, q_)
            if grad_log_p is None:
                grad_log_p = tf.zeros_like(q_)
            # Replace non-finite gradients with zeros; the finite check at
            # the end of the trajectory will catch the issue at H level.
            grad_log_p = tf.where(tf.math.is_finite(grad_log_p),
                                   grad_log_p, tf.zeros_like(grad_log_p))
            return log_p, grad_log_p

        log_p0, grad0 = potential_and_grad(q)
        H0 = -log_p0 + metric.kinetic_energy(p)

        p_curr = p + 0.5 * eps_t * grad0  # half kick
        q_curr = q

        for i in range(num_leapfrog):
            # Drift: q += eps * M^-1 p
            q_curr = q_curr + eps_t * (p_curr / metric.M)
            # Full kick (except last step gets a half kick)
            log_p_i, grad_i = potential_and_grad(q_curr)
            if i == num_leapfrog - 1:
                p_curr = p_curr + 0.5 * eps_t * grad_i
            else:
                p_curr = p_curr + eps_t * grad_i

        log_p_final, _ = potential_and_grad(q_curr)
        H_final = -log_p_final + metric.kinetic_energy(p_curr)
        dh = H_final - H0

        finite = bool(
            tf.reduce_all(tf.math.is_finite(q_curr)).numpy()
            and tf.reduce_all(tf.math.is_finite(p_curr)).numpy()
            and tf.math.is_finite(H_final).numpy()
        )
        return q_curr, p_curr, float(dh.numpy()) if finite else float('inf'), finite

    except (tf.errors.InvalidArgumentError, tf.errors.InternalError,
            tf.errors.NotFoundError, ValueError) as e:
        # OT backward MatrixSolve singular, Cholesky failure, or similar
        # numerical exception inside the graph. Treat as a non-finite
        # trajectory: chain stays at q, dh=inf, finite=False. The caller
        # (FindReasonableEpsilon, warmup loop) then halves eps / rejects
        # the proposal, so the chain self-recovers.
        return q, p, float('inf'), False


def find_reasonable_epsilon(target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
                             q: tf.Tensor, metric: Metric, num_leapfrog: int,
                             eps_init: float = 1.0, target_alpha: float = 0.5,
                             max_iters: int = 50,
                             seed: Optional[tf.Tensor] = None) -> float:
    """Hoffman-Gelman 2014 Algorithm 4 with hard-bracket failure handling.

    If a leapfrog at eps produces non-finite ΔH, that eps becomes a hard upper
    bound on the search. Subsequent proposals only halve from the highest known
    finite eps.
    """
    if seed is None:
        seed = tf.constant([42, 0], dtype=tf.int32)

    momentum_dist = metric.build_momentum_distribution()
    # Stateless sampling for reproducibility: we just need one momentum sample.
    p0 = tf.reshape(momentum_dist.sample(seed=seed), q.shape)

    upper_bracket = float('inf')
    eps = float(eps_init)
    direction = None  # +1 to grow, -1 to shrink; set on first finite alpha

    for it in range(max_iters):
        _, _, dh, finite = _one_leapfrog_trajectory(
            target_log_prob_fn, q, p0, eps, metric, num_leapfrog
        )
        if not finite:
            upper_bracket = min(upper_bracket, eps)
            eps = eps / 2.0
            continue

        # alpha = min(1, exp(-dh))
        if dh > 0:
            log_alpha = -dh
        else:
            log_alpha = 0.0
        alpha = math.exp(min(log_alpha, 0.0))

        if direction is None:
            direction = +1 if alpha > target_alpha else -1

        if direction == +1 and alpha > target_alpha:
            new_eps = eps * 2.0
            if new_eps >= upper_bracket:
                # Would cross the failure boundary; halt at midpoint.
                eps = (eps + upper_bracket) / 2.0
                break
            eps = new_eps
        elif direction == -1 and alpha < target_alpha:
            eps = eps / 2.0
        else:
            break  # crossed target_alpha

    return eps


# =============================================================================
# 3. Dual averaging (Section "3." in plan)
# =============================================================================

@dataclass
class DualAveragingState:
    log_avg_step: float
    log_step: float
    error_sum: float
    iter: int
    mu: float                            # log_shrinkage_target
    gamma: float = 0.05                  # exploration_shrinkage
    t0: float = 10.0                     # step_count_smoothing
    kappa: float = 0.75                  # decay_rate (for averaged step)


def fresh_da_state(eps_init: float, da_shrinkage_factor: float = 10.0
                   ) -> DualAveragingState:
    """Stan/Hoffman-Gelman default: anchor at log(10*eps_init), not log(eps_init).

    The runaway protection in this runner comes from windowed re-starting (each
    adaptation window resets DA after FindReasonableEpsilon picks a sane
    eps_init for the new metric), not from anchor manipulation.
    """
    return DualAveragingState(
        log_avg_step=math.log(eps_init),
        log_step=math.log(eps_init),
        error_sum=0.0,
        iter=0,
        mu=math.log(da_shrinkage_factor * eps_init),
    )


def dual_averaging_step(state: DualAveragingState, accept_prob: float,
                        target_accept: float) -> DualAveragingState:
    """One step of dual averaging update."""
    state.iter += 1
    state.error_sum += target_accept - accept_prob
    state.log_step = state.mu - (
        math.sqrt(state.iter) / (state.gamma * (state.iter + state.t0))
        * state.error_sum
    )
    decay = state.iter ** (-state.kappa)
    state.log_avg_step = (decay * state.log_step
                          + (1.0 - decay) * state.log_avg_step)
    return state


# =============================================================================
# 4. Window schedule (Section "4." in plan)
# =============================================================================

def window_schedule(num_warmup: int, abs_buffer_init: int = 75,
                    abs_buffer_term: int = 50, abs_window_base: int = 25,
                    short_warmup_threshold: int = 150,
                    skip_metric_threshold: int = 20) -> List[int]:
    """Stan-style schedule. Verified against stan/mcmc/windowed_adaptation.hpp.

    Returns [init_buffer, win1, win2, ..., term_buffer].

      - num_warmup < 20: tiny, step-only single window (Stan also returns early).
      - 20 <= num_warmup < 150: short, fallback 15/75/10 with single adaptation
        window. Integer truncation matches Stan's `unsigned int` cast.
      - num_warmup >= 150: long, Stan absolute 75/25/50 with doubling windows.
    """
    if num_warmup < skip_metric_threshold:
        warnings.warn(
            f"num_warmup={num_warmup} < {skip_metric_threshold}; "
            f"skipping metric adaptation entirely (step-only)."
        )
        return [num_warmup]

    if num_warmup < short_warmup_threshold:
        # Stan's short-warmup fallback with INTEGER TRUNCATION
        buffer_init = int(num_warmup * 0.15)
        buffer_term = int(num_warmup * 0.10)
        middle = num_warmup - buffer_init - buffer_term
        if middle < 1:
            return [num_warmup]
        return [buffer_init, middle, buffer_term]

    # Long warmup: Stan's absolute numbers with doubling windows.
    middle = num_warmup - abs_buffer_init - abs_buffer_term
    windows: List[int] = []
    win_size = abs_window_base
    while sum(windows) + win_size <= middle:
        windows.append(win_size)
        win_size *= 2
    if sum(windows) < middle:
        windows[-1] += middle - sum(windows)
    return [abs_buffer_init] + windows + [abs_buffer_term]


# =============================================================================
# 5. Diagnostics dataclasses (Section "7." in plan)
# =============================================================================

@dataclass
class WindowSummary:
    """Per-window diagnostic record. Counts are PER-WINDOW, not cumulative."""
    window_idx: int
    n_iter: int
    accept_rate: float
    final_eps: float
    metric_M: List[float]
    n_divergences_in_window: int
    median_dh: float
    max_dh: float
    n_nonfinite_in_window: int
    is_adapt_window: bool


@dataclass
class WarmupResult:
    q: tf.Tensor
    metric: Metric
    eps: float
    n_divergences: int
    median_dh: float
    max_dh: float
    n_nonfinite: int
    window_summaries: List[WindowSummary]


# =============================================================================
# 6. HMC step with diagnostics (Section "5." support)
# =============================================================================

def hmc_step_with_diagnostics(target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
                               q: tf.Tensor, metric: Metric, eps: float,
                               num_leapfrog: int, seed: tf.Tensor
                               ) -> Tuple[tf.Tensor, float, float, bool]:
    """One HMC step. Returns (q_new, accept_prob, dh, finite).

    Performs metropolis accept/reject; accept_prob is the actual probability
    used in the Bernoulli decision (0.0 if dh non-finite).
    """
    momentum_dist = metric.build_momentum_distribution()
    p0 = tf.reshape(momentum_dist.sample(seed=seed), q.shape)

    q_new, _p_new, dh, finite = _one_leapfrog_trajectory(
        target_log_prob_fn, q, p0, eps, metric, num_leapfrog
    )

    if not finite:
        return q, 0.0, dh, False

    # Metropolis accept/reject. Use a stateless uniform draw, derived seed.
    accept_prob = math.exp(min(-dh, 0.0))
    accept_seed = tf.random.experimental.stateless_split(seed, num=2)[1]
    u = float(tf.random.stateless_uniform(shape=(), seed=accept_seed,
                                           minval=0.0, maxval=1.0).numpy())
    if u < accept_prob:
        return q_new, accept_prob, dh, True
    else:
        return q, accept_prob, dh, True


# =============================================================================
# 7. Stan warmup orchestrator (Section "5." in plan)
# =============================================================================

def stan_warmup(target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
                q0: tf.Tensor, num_warmup: int, num_leapfrog: int,
                target_accept: float = 0.8,
                shrinkage_alpha: float = 5.0,
                shrinkage_target: float = 1e-3,
                da_shrinkage_factor: float = 10.0,
                divergence_threshold: float = 1000.0,
                min_window_samples: int = 10,
                warmup_init: Optional[WarmupInit] = None,
                schedule_kwargs: Optional[Dict[str, Any]] = None,
                seed: Optional[tf.Tensor] = None,
                progress_callback: Optional[Callable[[int, int, float, float], None]] = None
                ) -> WarmupResult:
    """Run Stan-style windowed warmup. Returns a WarmupResult with locked
    (M, eps) for the sampling phase plus per-window diagnostics."""

    dim = int(tf.size(q0).numpy())
    if warmup_init is None:
        warmup_init = default_warmup_init(dim, dtype=q0.dtype)

    schedule = window_schedule(num_warmup, **(schedule_kwargs or {}))
    if seed is None:
        seed = tf.constant([42, 0], dtype=tf.int32)

    metric = warmup_init.metric
    # Seed the step with FindReasonableEpsilon at the initial metric.
    eps = find_reasonable_epsilon(target_log_prob_fn, q0, metric, num_leapfrog,
                                   eps_init=warmup_init.eps_init, seed=seed)
    da_state = fresh_da_state(eps, da_shrinkage_factor=da_shrinkage_factor)

    q = q0
    accept_history: List[float] = []
    dh_history: List[float] = []
    finite_history: List[bool] = []
    window_summaries: List[WindowSummary] = []
    iteration = 0

    for window_idx, window_size in enumerate(schedule):
        is_init_buffer = (window_idx == 0)
        is_term_buffer = (window_idx == len(schedule) - 1)
        # In a single-window schedule (tiny), the only window is treated as
        # init_buffer (step-only, no M update).
        if len(schedule) == 1:
            is_init_buffer = True
            is_term_buffer = False
        is_adapt_window = not (is_init_buffer or is_term_buffer)

        window_samples: List[tf.Tensor] = []

        for _ in range(window_size):
            iteration += 1
            iter_seed = tf.random.experimental.stateless_split(
                seed, num=iteration + 1)[-1]
            q_new, accept, dh, finite = hmc_step_with_diagnostics(
                target_log_prob_fn, q, metric, eps, num_leapfrog, seed=iter_seed
            )
            q = q_new
            accept_history.append(accept)
            dh_history.append(dh)
            finite_history.append(finite)
            if (not finite) or abs(dh) > divergence_threshold:
                pass  # counted in window summary below
            da_state = dual_averaging_step(da_state, accept, target_accept)
            eps = math.exp(da_state.log_step)
            if is_adapt_window:
                window_samples.append(q_new)
            if progress_callback is not None:
                progress_callback(iteration, num_warmup, eps, accept)

        # End of window — record per-window counts (NOT cumulative).
        window_dh = dh_history[-window_size:]
        window_accept = accept_history[-window_size:]
        n_div = sum(1 for d in window_dh
                    if (not np.isfinite(d)) or abs(d) > divergence_threshold)
        n_nonfin = sum(1 for d in window_dh if not np.isfinite(d))
        finite_dh = [abs(d) for d in window_dh if np.isfinite(d)]
        median_dh = float(np.median(finite_dh)) if finite_dh else float('inf')
        max_dh = float(np.max(finite_dh)) if finite_dh else float('inf')

        window_summaries.append(WindowSummary(
            window_idx=window_idx,
            n_iter=window_size,
            accept_rate=float(np.mean(window_accept)),
            final_eps=eps,
            metric_M=[float(x) for x in metric.M.numpy().tolist()]
                if isinstance(metric, DiagonalMetric) else [],
            n_divergences_in_window=n_div,
            median_dh=median_dh,
            max_dh=max_dh,
            n_nonfinite_in_window=n_nonfin,
            is_adapt_window=is_adapt_window,
        ))

        if is_adapt_window and len(window_samples) >= min_window_samples:
            samples_tensor = tf.stack(window_samples, axis=0)
            metric = metric.estimate_from_samples(
                samples_tensor, shrinkage_alpha=shrinkage_alpha,
                shrinkage_target=shrinkage_target,
            )
            # Re-seed step at the new metric; reset DA.
            eps = find_reasonable_epsilon(target_log_prob_fn, q, metric,
                                            num_leapfrog, eps_init=eps,
                                            seed=seed)
            da_state = fresh_da_state(eps, da_shrinkage_factor=da_shrinkage_factor)
        elif is_adapt_window:
            warnings.warn(
                f"window {window_idx} had {len(window_samples)} samples "
                f"< min_window_samples={min_window_samples}; "
                f"skipping metric update.")

    # Lock eps at the DA-averaged value (Stan convention).
    final_eps = math.exp(da_state.log_avg_step)

    finite_dh_all = [abs(d) for d in dh_history if np.isfinite(d)]
    return WarmupResult(
        q=q,
        metric=metric,
        eps=final_eps,
        n_divergences=sum(1 for d in dh_history
                          if (not np.isfinite(d)) or abs(d) > divergence_threshold),
        median_dh=float(np.median(finite_dh_all)) if finite_dh_all else float('inf'),
        max_dh=float(np.max(finite_dh_all)) if finite_dh_all else float('inf'),
        n_nonfinite=sum(1 for d in dh_history if not np.isfinite(d)),
        window_summaries=window_summaries,
    )


# =============================================================================
# 8. Sampling phase (Section "6." in plan)
# =============================================================================

def sample_phase(target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
                 q0: tf.Tensor, metric: Metric, eps: float,
                 num_samples: int, num_leapfrog: int,
                 divergence_threshold: float = 1000.0,
                 seed: Optional[tf.Tensor] = None,
                 progress_callback: Optional[Callable[[int, int, float, float], None]] = None
                 ) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """Sample at fixed (metric, eps). Returns (samples, diagnostics)."""
    if seed is None:
        seed = tf.constant([1729, 0], dtype=tf.int32)

    samples = []
    accepts = []
    dhs = []
    energies = []
    q = q0

    for it in range(num_samples):
        iter_seed = tf.random.experimental.stateless_split(seed, num=it + 2)[-1]
        q_new, accept, dh, finite = hmc_step_with_diagnostics(
            target_log_prob_fn, q, metric, eps, num_leapfrog, seed=iter_seed
        )
        q = q_new
        samples.append(q)
        accepts.append(accept)
        dhs.append(dh)

        # E = U(q) + K(p_freshly_resampled) — for E-BFMI we need energy at
        # the start of each iteration (after momentum resampling, before
        # leapfrog). For the diagnostic, use H at acceptance time.
        log_p = target_log_prob_fn(q).numpy()
        energies.append(float(-log_p))  # potential at q_new; kinetic averages out

        if progress_callback is not None:
            progress_callback(it + 1, num_samples, eps, accept)

    samples_tensor = tf.stack(samples, axis=0)

    # E-BFMI: Stan's formula = mean((E_t - E_{t-1})^2) / var(E)
    energies_arr = np.asarray(energies)
    if len(energies_arr) > 1 and np.var(energies_arr) > 0:
        delta_e_sq = np.mean(np.diff(energies_arr) ** 2)
        e_bfmi = float(delta_e_sq / np.var(energies_arr))
    else:
        e_bfmi = float('nan')

    n_div = sum(1 for d in dhs
                if (not np.isfinite(d)) or abs(d) > divergence_threshold)
    finite_dhs = [abs(d) for d in dhs if np.isfinite(d)]

    diagnostics = {
        "acceptance_rate": float(np.mean(accepts)),
        "n_divergences_sampling": int(n_div),
        "median_dh_sampling": float(np.median(finite_dhs)) if finite_dhs else float('inf'),
        "max_dh_sampling": float(np.max(finite_dhs)) if finite_dhs else float('inf'),
        "n_nonfinite_sampling": sum(1 for d in dhs if not np.isfinite(d)),
        "e_bfmi": e_bfmi,
    }
    return samples_tensor, diagnostics


# =============================================================================
# 9. Runner class (Section "7. StanDPFRunner" in plan)
# =============================================================================

class StanDPFRunner:
    """DPFRunner with Stan-style windowed warmup + sampling. Mirrors the
    interface of `DPFRunner` from `hmc_runner.py` so `run_dpf_experiment.py`
    can dispatch to either based on the YAML `runner` field."""

    def __init__(self, base_model: Any, filter_class: Type,
                 filter_kwargs: Dict[str, Any],
                 param_specs: Dict[str, ParameterSpec]):
        self.base_model = base_model
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        self.param_specs = param_specs

        trainable_param_names = list(param_specs.keys())
        self.diff_model = DifferentiableModel(base_model, trainable_param_names)

        model_dtype = getattr(base_model, 'dtype', tf.float32)
        self.param_handler = ParameterHandler(param_specs, dtype=model_dtype)

        self.filter_obj = filter_class(self.diff_model, **filter_kwargs)
        self._observations_tf = None

    def _negative_log_posterior(self, unconstrained_params: tf.Tensor) -> tf.Tensor:
        constrained_params = self.param_handler.constrain(unconstrained_params)
        self.diff_model.update_parameters(constrained_params)
        seed = tf.constant([42, 0], dtype=tf.int32)  # PF seed identical between warmup and sampling
        log_likelihood = self.filter_obj.log_marginal_likelihood_tf(
            self._observations_tf, seed=seed
        )
        log_prior = self.param_handler.log_prior(constrained_params)
        return -(log_likelihood + log_prior)

    def run_inference(
        self,
        observations: np.ndarray,
        num_samples: int = 400,
        num_warmup: int = 1000,
        num_leapfrog_steps: int = 5,  # avoid round numbers (10, 20) for Gaussian-like posteriors; see module docstring re: fixed-trajectory resonance
        target_accept_prob: float = 0.8,
        shrinkage_alpha: float = 5.0,
        shrinkage_target: float = 1e-3,
        da_shrinkage_factor: float = 10.0,
        divergence_threshold: float = 1000.0,
        min_window_samples: int = 10,
        abs_buffer_init: int = 75,
        abs_buffer_term: int = 50,
        abs_window_base: int = 25,
        short_warmup_threshold: int = 150,
        skip_metric_threshold: int = 20,
        seed: int = 42,
    ) -> DPFResult:
        dtype = getattr(self.base_model, 'dtype', tf.float32)
        self._observations_tf = tf.constant(observations, dtype=dtype)

        def target_log_prob_fn(unconstrained_params):
            return -self._negative_log_posterior(unconstrained_params)

        q0 = self.param_handler.unconstrained_init
        rng_seed = tf.constant([seed, 0], dtype=tf.int32)

        print(f"  [stan_warmup] num_warmup={num_warmup}, num_leapfrog={num_leapfrog_steps}, "
              f"target_accept={target_accept_prob}")

        t0 = time.perf_counter()

        # Per-iteration progress hook for warmup
        def warmup_cb(it, total, eps, accept):
            if it == 1 or it % max(1, total // 20) == 0:
                print(f"    [warmup {it}/{total}] eps={eps:.4f} accept={accept:.0%}")

        warmup_result = stan_warmup(
            target_log_prob_fn, q0, num_warmup, num_leapfrog_steps,
            target_accept=target_accept_prob,
            shrinkage_alpha=shrinkage_alpha,
            shrinkage_target=shrinkage_target,
            da_shrinkage_factor=da_shrinkage_factor,
            divergence_threshold=divergence_threshold,
            min_window_samples=min_window_samples,
            schedule_kwargs={
                'abs_buffer_init': abs_buffer_init,
                'abs_buffer_term': abs_buffer_term,
                'abs_window_base': abs_window_base,
                'short_warmup_threshold': short_warmup_threshold,
                'skip_metric_threshold': skip_metric_threshold,
            },
            seed=rng_seed,
            progress_callback=warmup_cb,
        )

        warmup_time = time.perf_counter() - t0
        print(f"  [stan_warmup done] eps={warmup_result.eps:.4f}, "
              f"n_divergences={warmup_result.n_divergences}, "
              f"wall={warmup_time:.0f}s")
        if isinstance(warmup_result.metric, DiagonalMetric):
            print(f"    final M = {warmup_result.metric.M.numpy()}")

        # Sampling phase
        def sample_cb(it, total, eps, accept):
            if it == 1 or it % max(1, total // 10) == 0:
                print(f"    [sample {it}/{total}] eps={eps:.4f} accept={accept:.0%}")

        t1 = time.perf_counter()
        samples_tensor, sampling_diag = sample_phase(
            target_log_prob_fn, warmup_result.q, warmup_result.metric,
            warmup_result.eps, num_samples, num_leapfrog_steps,
            divergence_threshold=divergence_threshold,
            seed=rng_seed,
            progress_callback=sample_cb,
        )
        sampling_time = time.perf_counter() - t1
        print(f"  [sampling done] accept={sampling_diag['acceptance_rate']:.0%}, "
              f"e_bfmi={sampling_diag['e_bfmi']:.3f}, "
              f"n_divergences={sampling_diag['n_divergences_sampling']}, "
              f"wall={sampling_time:.0f}s")

        # Map unconstrained samples back to constrained (per-parameter) space
        samples_np = samples_tensor.numpy()
        per_param = {name: [] for name in self.param_handler.param_names}
        for i in range(samples_np.shape[0]):
            constrained = self.param_handler.constrain(samples_np[i])
            for name, value in constrained.items():
                per_param[name].append(float(value.numpy()))
        per_param = {name: np.array(vals) for name, vals in per_param.items()}

        # Summary statistics per parameter
        summary = {}
        for name, vals in per_param.items():
            summary[name] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "median": float(np.median(vals)),
                "q5": float(np.quantile(vals, 0.05)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q95": float(np.quantile(vals, 0.95)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

        diagnostics = {
            "acceptance_rate": sampling_diag["acceptance_rate"],
            "final_step_size": warmup_result.eps,
            "final_metric_diag": (
                [float(x) for x in warmup_result.metric.M.numpy().tolist()]
                if isinstance(warmup_result.metric, DiagonalMetric) else None
            ),
            "n_divergences_warmup": warmup_result.n_divergences,
            "n_divergences_sampling": sampling_diag["n_divergences_sampling"],
            "median_dh_warmup": warmup_result.median_dh,
            "max_dh_warmup": warmup_result.max_dh,
            "median_dh_sampling": sampling_diag["median_dh_sampling"],
            "max_dh_sampling": sampling_diag["max_dh_sampling"],
            "n_nonfinite_warmup": warmup_result.n_nonfinite,
            "n_nonfinite_sampling": sampling_diag["n_nonfinite_sampling"],
            "e_bfmi": sampling_diag["e_bfmi"],
            "warmup_windows": [
                {
                    "window_idx": w.window_idx,
                    "n_iter": w.n_iter,
                    "accept_rate": w.accept_rate,
                    "final_eps": w.final_eps,
                    "metric_M": w.metric_M,
                    "n_divergences_in_window": w.n_divergences_in_window,
                    "median_dh": w.median_dh,
                    "max_dh": w.max_dh,
                    "n_nonfinite_in_window": w.n_nonfinite_in_window,
                    "is_adapt_window": w.is_adapt_window,
                }
                for w in warmup_result.window_summaries
            ],
            "warmup_wall_seconds": warmup_time,
            "sampling_wall_seconds": sampling_time,
        }

        return DPFResult(
            samples=per_param,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                "model_type": type(self.base_model).__name__,
                "filter_type": type(self.filter_obj).__name__,
                "sampler": "stan_hmc",
                "num_samples": num_samples,
                "num_warmup": num_warmup,
                "num_leapfrog_steps": num_leapfrog_steps,
                "num_observations": int(observations.shape[0]),
                "target_accept_prob": target_accept_prob,
                "runner": "stan",
            },
        )
