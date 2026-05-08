# Plan — `stan_hmc_runner.py`: Stan-style windowed adaptation

Status: revised after codex review (round 1). Awaiting user approval.

## What this fixes

The pipeline failure documented in `to_be_addressed/FindReasonableEpsilon.md`:
- Set M = diag(1/var(q)) from a previous chain's samples.
- Set initial ε to the previous chain's adapted value (no rescaling).
- DA started in a regime where random momentum (`|p| ~ sqrt(M)`) dominated gradient signal.
- DA misread 100% accept as "step too small," grew step exponentially, OT backward crashed.

Stan avoids this by re-running FindReasonableEpsilon every time M changes, so DA always starts at a sane ε for the current metric. Single-pass offline tuning skips this step.

## Module location and integration

New file: `code/src/DF/stan_hmc_runner.py`. Sits alongside the existing `hmc_runner.py`. Does not replace it.

YAML field in `dpf.hmc:`:
```yaml
dpf:
  hmc:
    runner: stan          # default 'tfp' for existing behavior; 'stan' switches
    num_warmup: 1000      # works for any value; Stan-style short-warmup fallback applies
    # Windowed adaptation knobs (Stan's actual logic: absolute 75/25/50 when they fit,
    # else 15%/75%/10% fractional fallback. Below num_warmup=20, metric adaptation is skipped.)
    abs_buffer_init: 75
    abs_buffer_term: 50
    abs_window_base: 25
    short_warmup_threshold: 150    # below this, switch to fractional 15%/75%/10%
                                    # 150 = 75 + 25 + 50 (Stan's exact threshold from set_window_params)
    skip_metric_threshold: 20      # below this, no M adaptation, step-only
    min_window_samples: 10         # skip M update for windows with < this many samples (was 4)
    # Metric regularization (Stan's exact formula from var_adaptation.hpp)
    metric_shrinkage_alpha: 5.0    # Stan default; controls how fast n_samples beats the floor
    metric_shrinkage_target: 1e-3  # Stan default; small numerical floor (NOT a Bayesian prior)
    metric_use_accepted_only: false  # use all states in window (Stan default)
    # FindReasonableEpsilon failure policy
    find_eps_max_iters: 50
    find_eps_init: 1.0
    # DA anchor (Stan/Hoffman-Gelman default)
    da_shrinkage_factor: 10.0      # mu = log(da_shrinkage_factor * eps_init)
    # Diagnostics
    track_divergences: true
    divergence_dh_threshold: 1000.0  # |ΔH| above this counted as a divergence
```

`run_dpf_experiment.py` dispatches between `DPFRunner` (existing) and `StanDPFRunner` (new) based on the `runner` field.

**PF-seed policy (codex review point 10):** the PF seed (`[42, 0]`) is identical between warmup and sampling phase. The adapted (M, ε) is then calibrated to the same target surface that sampling sees.

## Architecture — components

### 1. `Metric` abstraction with `WarmupInit` hook (codex round-2 point 8)

```python
class Metric(ABC):
    """Abstract interface for HMC metric. v1 has DiagonalMetric only."""
    @abstractmethod
    def estimate_from_samples(self, samples: tf.Tensor,
                               shrinkage_alpha: float = 5.0,
                               shrinkage_target: float = 1e-3) -> 'Metric': ...
    @abstractmethod
    def build_momentum_distribution(self) -> tfd.Distribution: ...
    @abstractmethod
    def kinetic_energy(self, p: tf.Tensor) -> tf.Tensor: ...
    @abstractmethod
    def as_step_size_scaling(self) -> tf.Tensor: ...  # for FindReasonableEpsilon

@dataclass
class WarmupInit:
    """Warmup starting state. Mode A: identity Metric, fresh DA. Mode B (v2):
    pass chain-1's adapted (metric, eps) here to skip slow adaptation windows."""
    metric: Metric
    eps_init: float
    skip_windows_below: int = 0     # v2 Mode B: number of windows to skip
                                      # (chain-1's M is sufficient)

class DiagonalMetric(Metric):
    """Diagonal mass matrix, v1 implementation."""
    def __init__(self, M: tf.Tensor):  # 1-D tensor of size dim
        self.M = M

    def estimate_from_samples(self, samples, shrinkage_alpha=5.0,
                                shrinkage_target=1e-3):
        """Stan's exact diagonal shrinkage formula (var_adaptation.hpp):

            var_shrunk = (n / (n + alpha)) * var
                         + shrinkage_target * (alpha / (n + alpha)) * 1

        - alpha (default 5.0): Stan's "shrinkage constant" — controls how fast
          the estimator trusts the sample variance over the floor target.
        - shrinkage_target (default 1e-3): the small positive value var is
          pulled toward when n is small. This is a NUMERICAL SAFETY FLOOR,
          not a Bayesian prior — Stan uses 1e-3 to keep var bounded away
          from zero, not to encode informed prior belief.

        M = 1 / var_shrunk per axis. No additional clamp/floor needed because
        the formula already lower-bounds var at 1e-3 * (alpha / (n + alpha)).
        """
        # Sample variance with N-1 (Bessel's correction) to match Stan's Welford
        # output. tf.math.reduce_variance returns biased variance (divides by N),
        # so we compute unbiased variance manually:
        n = tf.cast(tf.shape(samples)[0], samples.dtype)
        mean = tf.reduce_mean(samples, axis=0)
        var = tf.reduce_sum((samples - mean) ** 2, axis=0) / (n - 1.0)

        var_shrunk = ((n / (n + shrinkage_alpha)) * var
                      + shrinkage_target * (shrinkage_alpha / (n + shrinkage_alpha)))
        return DiagonalMetric(M=1.0 / var_shrunk)

    def build_momentum_distribution(self):
        return tfd.MultivariateNormalDiag(loc=tf.zeros_like(self.M),
                                          scale_diag=tf.sqrt(self.M))

    def kinetic_energy(self, p):
        return 0.5 * tf.reduce_sum(p**2 / self.M)
```

**V2 dense extension foreshadow.** `DenseMetric.estimate_from_samples` should use Stan's exact dense regularization (`covar_adaptation.hpp`):

```python
# Stan's dense shrinkage formula:
covar_shrunk = (n / (n + 5.0)) * sample_covar + 1e-3 * (5.0 / (n + 5.0)) * I
```

Same shrinkage constant (5.0) and target (1e-3) as the diagonal case, but pulls toward `1e-3 * I` instead of a vector of `1e-3`. Then:
- store `Σ = covar_shrunk` and Cholesky `L_Σ` (not `Σ⁻¹` directly),
- compute kinetic energy as `0.5 * p^T Σ⁻¹ p` via the Cholesky factor (`solve(L_Σ L_Σ^T, p)`),
- sample momentum as `p = L_Σ^{-T} z, z ~ N(0, I)` so `p ~ N(0, Σ⁻¹)`.

The `1e-3 * I` regularization is what keeps `Σ` PSD and Cholesky-decomposable even at low sample counts. No extra jitter beyond Stan's regularization is needed.

### 2. `find_reasonable_epsilon(...)` with hard upper bracket (codex review point 6)

```python
def find_reasonable_epsilon(target_log_prob_fn, q, metric, num_leapfrog,
                             eps_init=1.0, target_alpha=0.5, max_iters=50,
                             leapfrog_adapter=None) -> float:
    """Hoffman-Gelman 2014 Algorithm 4, with hard-bracket failure handling.

    If a leapfrog at eps produces non-finite ΔH (OT crash, gradient NaN),
    eps becomes a hard upper bound for the rest of the search. Subsequent
    proposals only halve from the highest known-finite eps.
    """
    upper_bracket = float('inf')  # set to first failing eps
    eps = eps_init
    for _ in range(max_iters):
        log_alpha = compute_log_alpha(target_log_prob_fn, q, eps, metric, num_leapfrog)
        if not np.isfinite(log_alpha):
            upper_bracket = min(upper_bracket, eps)
            eps = eps / 2.0
            continue
        alpha = np.exp(min(log_alpha, 0.0))
        # Initialize direction on first finite alpha
        if direction not yet set:
            direction = +1 if alpha > target_alpha else -1
        if direction == +1 and alpha > target_alpha:
            new_eps = eps * 2.0
            if new_eps >= upper_bracket:
                # Would cross failure boundary; halt at midpoint
                eps = (eps + upper_bracket) / 2.0
                break
            eps = new_eps
        elif direction == -1 and alpha < target_alpha:
            eps = eps / 2.0
        else:
            break  # crossed target_alpha
    return eps
```

Implementation uses TFP's `SimpleLeapfrogIntegrator` (codex review point 2) wrapped in a thin adapter `_one_leapfrog_trajectory(q, p, eps, metric, num_leapfrog)`. Adapter insulates against `tfp.mcmc.internal.*` API changes.

### 3. `DualAveragingState` and `dual_averaging_step` (codex review point 3)

Implemented from scratch — ~50 lines.

```python
@dataclass
class DualAveragingState:
    log_avg_step: float
    log_step: float
    error_sum: float
    iter: int
    mu: float                          # log_shrinkage_target
    gamma: float = 0.05                # exploration_shrinkage
    t0: float = 10.0                   # step_count_smoothing
    kappa: float = 0.75                # decay_rate (for averaged step)

def dual_averaging_step(state, accept_prob, target_accept):
    state.iter += 1
    state.error_sum += target_accept - accept_prob
    eta = 1.0 / (state.iter + state.t0)
    state.log_step = state.mu - sqrt(state.iter) / (state.gamma * (state.iter + state.t0)) * state.error_sum
    decay = state.iter ** (-state.kappa)
    state.log_avg_step = decay * state.log_step + (1 - decay) * state.log_avg_step
    return state

def fresh_da_state(eps_init, da_shrinkage_factor=10.0):
    """Stan/Hoffman-Gelman default: anchor at log(10*eps_init), not log(eps_init).

    The runaway protection in this runner comes from windowed re-starting (each
    adaptation window resets DA after FindReasonableEpsilon picks a sane eps_init
    for the new metric), not from the anchor. So the standard anchor applies.
    """
    return DualAveragingState(
        log_avg_step=log(eps_init),
        log_step=log(eps_init),
        error_sum=0.0,
        iter=0,
        mu=log(da_shrinkage_factor * eps_init),  # Stan default: 10*eps_init
    )
```

### 4. `window_schedule(num_warmup, abs_buffer_init, abs_buffer_term, abs_window_base, short_warmup_threshold, skip_metric_threshold, min_window_samples) -> list[int]`

Stan's actual scheme. Three regimes:

- **Long warmup (`num_warmup >= 150`):** absolute Stan numbers `[75, 25, 50, 100, 200, 500, ..., 50]` — buffer_init=75, doubling adaptation windows, buffer_term=50. The 150 cutoff is exactly `75 + 25 + 50`, the minimum that fits Stan's three stages.
- **Short warmup (`20 <= num_warmup < 150`):** Stan's fallback `[15% init, 75% middle (one big window), 10% term]` with **integer truncation** (Stan source uses `0.15 * num_warmup` cast to `unsigned int`). Single adaptation window absorbs the entire middle so M is estimated once from a bigger sample base instead of fragmenting.
- **Tiny warmup (`num_warmup < skip_metric_threshold`, default 20):** step-only single window, no M adaptation. Stan also does this.

```python
def window_schedule(num_warmup, abs_buffer_init=75, abs_buffer_term=50,
                    abs_window_base=25, short_warmup_threshold=150,
                    skip_metric_threshold=20) -> list[int]:
    """Stan-style schedule. Verified against `stan/mcmc/windowed_adaptation.hpp`.

    Returns [init_buffer, win1, win2, ..., term_buffer]. For long warmups, uses
    Stan's absolute 75/25/50 numbers with doubling adaptation windows. For short
    warmups, falls back to 15/75/10 with a single adaptation window so M is
    estimated once from a bigger sample base instead of fragmenting. For tiny
    warmups, skips M adaptation entirely.

    Stan's exact thresholds:
      - tiny: num_warmup < 20 (Stan's set_window_params returns early)
      - short: 20 <= num_warmup < 150 (Stan's fallback triggers when 75+25+50 don't fit)
      - long: num_warmup >= 150 (default Stan absolute numbers fit)
    """
    if num_warmup < skip_metric_threshold:
        warnings.warn(
            f"num_warmup={num_warmup} < {skip_metric_threshold}; "
            f"skipping metric adaptation entirely (step-only).")
        return [num_warmup]

    if num_warmup < short_warmup_threshold:
        # Stan's short-warmup fallback: 15/75/10 with INTEGER TRUNCATION
        # (Stan source: `0.15 * num_warmup` cast to unsigned int).
        buffer_init = int(num_warmup * 0.15)
        buffer_term = int(num_warmup * 0.10)
        middle = num_warmup - buffer_init - buffer_term
        if middle < 1:
            return [num_warmup]
        return [buffer_init, middle, buffer_term]

    # Long warmup: Stan's absolute numbers with doubling windows.
    middle = num_warmup - abs_buffer_init - abs_buffer_term
    windows = []
    win_size = abs_window_base
    while sum(windows) + win_size <= middle:
        windows.append(win_size)
        win_size *= 2
    if sum(windows) < middle:
        windows[-1] += middle - sum(windows)  # last window absorbs the remainder
    return [abs_buffer_init] + windows + [abs_buffer_term]
```

Examples (recomputed with integer truncation and the 150 cutoff):

| `num_warmup` | regime | init | windows | term | total |
|---|---|---|---|---|---|
| 15   | tiny | (single window, no M) | --- | --- | 15 |
| 50   | short | 7 | [38] | 5 | 50 (one big adaptation window) |
| 100  | short | 15 | [75] | 10 | 100 (one big adaptation window) |
| 150  | long | 75 | [25] | 50 | 150 (Stan absolute, exactly fits) |
| 200  | long | 75 | [25, 50] | 50 | 200 |
| 500  | long | 75 | [25, 50, 100, 200] | 50 | 500 |
| 1000 | long | 75 | [25, 50, 100, 200, 500] | 50 | 1000 (Stan classic) |
| 2000 | long | 75 | [25, 50, 100, 200, 1525] | 50 | 2000 (last window absorbs remainder) |

The `min_window_samples` check is enforced in the orchestrator: any window with fewer samples than the minimum (default 10) simply skips the M update (logs a warning, keeps the previous metric). This is a project-specific safety net; Stan does not have an analogous knob.

### 5. `StanWarmupState` and `stan_warmup` (revised orchestrator)

```python
@dataclass
class WindowSummary:
    """Per-window diagnostic record. All counts are PER-WINDOW, not cumulative."""
    window_idx: int
    n_iter: int
    accept_rate: float
    final_eps: float
    metric_M: tf.Tensor
    n_divergences_in_window: int    # per-window (not cumulative across all warmup)
    median_dh: float
    max_dh: float
    n_nonfinite_in_window: int

@dataclass
class StanWarmupState:
    q: tf.Tensor
    metric: Metric
    eps: float
    da_state: DualAveragingState
    window_samples: list
    accept_history: list
    dh_history: list
    n_divergences: int

def stan_warmup(target_log_prob_fn, q0, num_warmup, num_leapfrog, ...) -> WarmupResult:
    schedule = window_schedule(num_warmup, ...)
    metric = DiagonalMetric(tf.ones(dim))   # identity
    eps = find_reasonable_epsilon(target, q0, metric, num_leapfrog)
    state = StanWarmupState(q=q0, metric=metric, eps=eps,
                            da_state=fresh_da_state(eps),
                            window_samples=[], accept_history=[],
                            dh_history=[], n_divergences=0)
    window_summaries = []

    for window_idx, window_size in enumerate(schedule):
        is_init_buffer = (window_idx == 0)
        is_term_buffer = (window_idx == len(schedule) - 1)
        is_adapt_window = not (is_init_buffer or is_term_buffer)

        for _ in range(window_size):
            q_new, accept, dh, finite = hmc_step_with_diagnostics(
                target, state.q, state.metric, state.eps, num_leapfrog)
            state.q = q_new
            state.accept_history.append(accept)
            state.dh_history.append(dh)
            if not finite or abs(dh) > divergence_threshold:
                state.n_divergences += 1
            state.da_state = dual_averaging_step(state.da_state, accept, target_accept)
            state.eps = exp(state.da_state.log_step)
            if is_adapt_window:
                state.window_samples.append(q_new)

        # End of window — record per-window counts (not cumulative)
        n_div_in_window = sum(
            1 for d in state.dh_history[-window_size:]
            if (not np.isfinite(d)) or abs(d) > divergence_threshold
        )
        n_nonfinite_in_window = sum(
            1 for d in state.dh_history[-window_size:] if not np.isfinite(d)
        )
        window_summaries.append(WindowSummary(
            window_idx=window_idx,
            n_iter=window_size,
            accept_rate=np.mean(state.accept_history[-window_size:]),
            final_eps=state.eps,
            metric_M=state.metric.M.numpy().copy(),
            n_divergences_in_window=n_div_in_window,
            median_dh=np.median([abs(d) for d in state.dh_history[-window_size:]
                                 if np.isfinite(d)]),
            max_dh=np.max([abs(d) for d in state.dh_history[-window_size:]
                           if np.isfinite(d)]) if any(np.isfinite(d) for d in state.dh_history[-window_size:]) else float('inf'),
            n_nonfinite_in_window=n_nonfinite_in_window,
        ))

        if is_adapt_window:
            samples_tensor = tf.stack(state.window_samples)
            state.metric = state.metric.estimate_from_samples(
                samples_tensor, shrinkage_alpha, shrinkage_target)
            state.eps = find_reasonable_epsilon(target, state.q, state.metric, num_leapfrog,
                                                 eps_init=state.eps)
            state.da_state = fresh_da_state(state.eps)
            state.window_samples = []

    # Lock metric and eps for sampling phase
    return WarmupResult(q=state.q, metric=state.metric,
                        eps=exp(state.da_state.log_avg_step),  # use averaged step
                        n_divergences=state.n_divergences,
                        window_summaries=window_summaries)
```

### 6. `sample_phase(...)` — fixed (M, ε)

Uses `tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo` with `momentum_distribution = metric.build_momentum_distribution()`, `step_size = eps`. No adaptation. Track divergences over the sampling phase too.

### 7. `StanDPFRunner.run_inference(...) -> DPFResult`

Mirrors `DPFRunner.run_inference` interface. Calls `stan_warmup` then `sample_phase`. Adds the warmup window summaries to `result.diagnostics['warmup_windows']` for inspection.

## Diagnostics (codex review point 7)

Per chain, recorded in `DPFResult.diagnostics`:
- `n_divergences_warmup`, `n_divergences_sampling` — counts of |ΔH| > threshold or non-finite.
- `dh_summary_warmup`, `dh_summary_sampling` — median, max, fraction non-finite.
- `warmup_windows` — list of `WindowSummary` records.
- `e_bfmi` — Energy-based BFMI, Stan's actual definition: `mean((E_t - E_{t-1})^2) / var(E)` where `E = U(q) + K(p)` per iteration. Computed on post-warmup samples per chain. Values < 0.3 flag potential energy-conservation problems.
- `final_metric` — adapted M.
- `final_eps` — adapted ε (averaged from DA).
- `acceptance_rate` — sampling-phase accept rate (existing).

## Multi-chain modes

For v1: **Mode A only** — independent warmup per chain. Codex confirms this is the right v1 default.

Mode B (shared M after chain 1) is deferred to v2.

## Failure policy (codex review point 10, comprehensive)

- **Schedule has three regimes** — long (Stan absolute 75/25/50), short (15/75/10 fallback), tiny (step-only). No hard `num_warmup` error; the schedule picks the appropriate regime.
- **`num_warmup < skip_metric_threshold` (default 20):** step-only single window, no M adaptation. Verified against Stan source (`windowed_adaptation.hpp` returns early from `set_window_params` and emits "WARNING: No estimation is performed for num_warmup < 20"; `adaptation_window()` then returns false for all counters, so `learn_variance`/`learn_covariance` never adds samples).
- **`skip_metric_threshold <= num_warmup < short_warmup_threshold` (200):** single big adaptation window in the middle (75% of warmup). M is estimated once from a substantial sample base, not fragmented.
- **Window has < `min_window_samples` (default 10):** skip M update for that window, log warning, keep previous metric. With the regime-based schedule, this rarely triggers.
- **Near-zero variance on any axis:** Stan's shrinkage formula `(n/(n+5))*var + 1e-3*(5/(n+5))` already lower-bounds the result at `1e-3 * (5/(n+5))`, so M is always finite. No extra clamp needed.
- **FindReasonableEpsilon hits non-finite ΔH:** that ε becomes hard upper bracket. Search continues strictly below.
- **FindReasonableEpsilon exhausts max_iters:** log warning, return current eps (best estimate so far).
- **DA produces NaN/inf step during warmup:** halt warmup, log error, return current state with warning. Don't crash.
- **Sampling phase divergences > 5% of iterations:** flag in diagnostics. Don't abort — chain may still be useful with reduced confidence.
- **PF returns non-finite log-likelihood:** treat as proposal rejection (skip update), log occurrence.

## Test plan

Unit tests:
1. `test_find_reasonable_epsilon_gaussian`: 1D N(0, σ²) target, eps_init=10, converges to eps ≈ 0.4-0.6.
2. `test_find_reasonable_epsilon_with_failure`: target that throws non-finite at large eps; FindReasonableEpsilon respects upper bracket.
3. `test_diagonal_metric_estimate`: synthetic samples from known diag-cov, recovered M close to ground truth with shrinkage.
4. `test_diagonal_metric_low_count`: < 4 samples → returns identity M.
5. `test_window_schedule_long`: num_warmup=1000 matches Stan's classic 75/[25,50,100,200,500]/50 exactly.
6. `test_window_schedule_short`: num_warmup=100 returns single big adaptation window (15%/75%/10%).
7. `test_window_schedule_tiny`: num_warmup=15 returns step-only single window with warning.
8. `test_window_schedule_sums`: total sums to num_warmup over a wide range (15, 50, 100, 200, 500, 1000, 2000, 5000).
9. `test_dual_averaging_state_basic`: DA on a known-stationary target converges to the right step. Anchor at log(10*eps_init).

Integration tests:
8. `test_stan_warmup_lg`: 1D LG, full warmup → final M close to inverse posterior var.
9. `test_stan_vs_tfp_lg_4chain`: 4 chains stan_hmc vs current tfp runner. R-hat, ESS, posterior should match within MC error. Stan version should have ≥ comparable ESS.

Stress tests:
10. `test_stan_rb_4chain`: range-bearing 4-chain. Compare to existing axisstep+l10. Stan version may converge with smaller num_leapfrog because M is properly tuned.
11. `test_stan_sv2d`: SV2D — does FindReasonableEpsilon's upper-bracket policy avoid the OT backward crash?

## Risks (revised)

1. **TFP-internal API drift.** `tfp.mcmc.internal.leapfrog_integrator.SimpleLeapfrogIntegrator` is unstable. Adapter mitigates but doesn't eliminate.
2. **Per-window TF retracing.** Each window may trigger a fresh trace if state shapes change. Need to pre-compile or use shape-stable accumulators.
3. **OT-backward stability under FindReasonableEpsilon.** Mitigated by hard upper bracket.
4. **Effort estimate.** Codex pushed to 4-6 days for production-quality. Plan adopts that estimate.
5. **Diagnostic infrastructure.** Adding divergence/E-BFMI tracking requires custom hooks into the leapfrog loop. Not free — adds ~50-100 lines.
6. **Multi-chain orchestration.** Each chain runs full warmup independently. Sequential on CUDA → 4× warmup cost. May want a YAML knob to allow shorter warmup for chains 2-N if user accepts the slight diagnostic cost (Mode B).

## Effort (revised, codex review point 8)

- ~600-900 LOC including diagnostics infrastructure.
- **Prototype:** 2-3 days (stan_warmup + sample_phase, single-chain, identity M only).
- **v1 production-ready:** 4-6 focused days (full Metric abstraction, all failure policies, diagnostics, tests, config plumbing).

## Order of implementation

1. `Metric` abstraction + `DiagonalMetric`.
2. `find_reasonable_epsilon` with hard-bracket policy + tests.
3. `DualAveragingState` + `dual_averaging_step` + tests.
4. `window_schedule` + tests.
5. `hmc_step_with_diagnostics` (returns ΔH, accept, finite flag).
6. `stan_warmup` orchestrator.
7. `sample_phase`.
8. `StanDPFRunner` integration with `run_dpf_experiment.py` (YAML field).
9. Diagnostics: per-window summaries, E-BFMI, divergence counting.
10. Integration tests on 1D LG.
11. Stress tests on RB and SV2D.

## Out of scope for v1

- Full (non-diagonal) mass matrix (`DenseMetric` is a v2 extension via the abstraction).
- Riemannian HMC.
- NUTS replacement.
- Mode B (shared M across chains).
- Stan's "term_buffer can be longer than buffer_init" variants. We use Stan's standard sizes.

## Open questions for user / codex (after round-2 revision)

1. **`num_warmup` default.** Plan uses 1000. Schedule regime-switches automatically, so any value works. Confirm 1000 default.
2. **`shrinkage_alpha`.** Plan uses 5.0 (codex round-2 confirmed reasonable for diagonal v1, no need to scale with dimension).
3. **`divergence_dh_threshold` = 1000.** Codex round-2 said defensible default, keep configurable, also report median/max/nonfinite rate alongside.
4. **PF-seed policy:** identical between warmup and sampling. Codex round-2 confirmed.
5. **Mode B (shared M)** — `WarmupInit` hook added to the abstraction in v1; full implementation deferred to v2. Confirm.

## Round-2 changes summary

- **Schedule:** abandoned pure fractional approach. Now uses Stan's actual three-regime logic (absolute Stan numbers / 15-75-10 fallback / step-only). Verified against `stan/mcmc/windowed_adaptation.hpp`.
- **`min_window_samples`:** 4 → 10 (engineering safety floor — Stan does not have an analogous knob).
- **DA anchor:** `mu = log(eps_init)` → `mu = log(10 * eps_init)` (standard Hoffman-Gelman / Stan).
- **E-BFMI:** corrected to Stan's actual formula `mean((E_t − E_{t-1})²) / var(E)`.
- **Per-window divergences:** fixed orchestrator bug where the count was cumulative instead of per-window. Pure project bookkeeping, not a Stan-conformance issue.
- **`WarmupInit` hook:** added to `Metric` abstraction so v2 Mode B (shared M) can plug in without rewriting warmup loop.

## Round-3 corrections (verified against Stan source)

- **Diagonal shrinkage formula corrected.** Was `var_shrunk = (n*var + α)/(n + α)` with α=5.0 — pulled var toward 1.0, which is **1000× more aggressive than Stan**. Now matches `stan/mcmc/var_adaptation.hpp`:

  ```cpp
  var = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0)) * Ones;
  ```

  Pulls var toward `1e-3` (a numerical floor), not 1.0. The `1e-3` is Stan's regularization target (numerical safety), not a Bayesian prior.

- **Dense v2 foreshadow corrected.** Was generic "Cholesky-safe jitter." Now references Stan's exact `stan/mcmc/covar_adaptation.hpp`:

  ```cpp
  covar = (n / (n + 5.0)) * covar + 1e-3 * (5.0 / (n + 5.0)) * Identity;
  ```

  Same shrinkage constant and target, applied to the dense covariance.

- **Tiny warmup behavior re-affirmed.** Verified `stan/mcmc/windowed_adaptation.hpp` does skip metric estimation for `num_warmup < 20` (early return from `set_window_params` leaves `adapt_*_buffer_=0`, so `adaptation_window()` is always false → estimator never sees samples). My plan's behavior matches.

- **Removed `var_floor` knob.** Stan's regularization formula already lower-bounds `var_shrunk` at `1e-3 * (alpha / (n + alpha))`. No need for an extra max() floor.
