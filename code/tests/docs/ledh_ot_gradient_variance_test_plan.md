# LEDH-OT Gradient Variance — Test Implementation Plan

**Status:** Draft — waiting for user approval before any file is created or run.
**Scope:** Diagnose (not fix) the shared root cause of two observed bugs on the 2D stochastic-volatility model with the LEDH + OT-entropy filter:
1. **MAP + LEDH-OT + SV2D:** per-step gradient norm spikes from ~10 to 50–100+ under `random_seed=True` while the loss at those same steps is unremarkable.
2. **HMC + LEDH-OT + SV2D:** step size adapter collapses because occasional large gradients send leapfrog trajectories off-manifold → proposals rejected → dual averaging shrinks step size → never recovers.

**Constraint:** no production code is modified. All work happens under `code/tests/`.

---

## 1. The hypothesis in plain words

When the filter resamples particles, it uses entropy-regularized optimal transport (OT). The *forward* pass is stable. The *backward* pass — implemented in `_sinkhorn_implicit_vjp` at `code/src/resampling/ot_entropy.py:400–463` — ends with this line:

```python
sol = tf.linalg.solve(K, rhs[:, tf.newaxis])[:, 0]
```

`K` is a `(2N−1) × (2N−1) = 1999 × 1999` matrix assembled from the actual row/column sums `a, b` of the converged transport matrix `T`. When weights are roughly uniform, `a` and `b` are healthy, `K` is well-conditioned, and `sol` is small and stable.

**When a random seed produces even a few near-zero weights, the corresponding diagonal entries of `diag(a)` or `diag(b[:-1])` inside `K` become tiny, `K` becomes nearly singular, and `tf.linalg.solve` returns huge values.** Those huge values feed directly into `grad_p = R − T·U`, `grad_a = −ε·u`, `grad_b = −ε·v`, and propagate back to `dparticles` and `dlog_weights`. The forward `T` looks normal, so the *loss* looks normal, but the *gradient* at that step explodes.

**One sentence:** intermittent, seed-dependent weight skewness drives `cond(K)` through the roof, the unregularized `solve` amplifies it, and the result is a rare large tail in the gradient distribution — which shows up as MAP spikes and as HMC step-size collapse.

**Why both symptoms come from this:**
- MAP/Adam sees the outlier directly as a spike in `|grad|`. Momentum partially averages it out, so parameters still move in the right direction, just slowly and noisily.
- HMC sees the same outlier as an exploding leapfrog trajectory → rejection → the step-size adapter shrinks → next draw has similar rare-blowup probability → collapse.

---

## 2. Test strategy — three tests, three yes/no questions

| # | Question | Answered by | Needs GPU? |
|---|----------|-------------|------------|
| 1 | Can `_sinkhorn_implicit_vjp` blow up when fed bad (skewed-weight) inputs, in isolation? | Unit test with hand-constructed `T` | No |
| 2 | At the moments the *real pipeline* spikes, is `_sinkhorn_implicit_vjp` actually receiving bad inputs? | Short MAP run with an in-test monkeypatched `_sinkhorn_implicit_vjp` that logs `cond(K)` and `||sol||` every call | Yes (to reproduce the CUDA/float32 failure mode) |
| 3 | If we turn off the backward path through OT resampling, do the spikes disappear? | Same short MAP run with `filter.stop_gradient_resampling=true` via Hydra override, compared against baseline | Yes |

**Ordering rationale:**
- Test 1 is the cheapest and most targeted. If it says "`_sinkhorn_implicit_vjp` is always stable", the hypothesis is dead and we redirect before spending GPU time.
- Test 2 is the smoking-gun test. Per-step correlation between `|grad|` spikes and `cond(K)` spikes is direct causal evidence.
- Test 3 is the control. It removes the suspect and checks whether the symptom follows. If spikes vanish when the backward path is cut, the hypothesis is confirmed end-to-end.

All three together give: *the function can explode → it does explode in practice at the spike moments → removing it removes the spikes*. Any one of them alone is insufficient.

**Per the user's `one_fix_at_a_time` rule**, we run these sequentially, not in parallel. Each test's result decides whether the next one is worth running.

---

## 3. Test 1 — Sinkhorn VJP conditioning under controlled skewness

### 3.1 File
`code/tests/unit/test_sinkhorn_vjp_conditioning.py` (new file)

### 3.2 Imports and dependencies
```python
# Production imports (read-only — NOT modified)
from src.resampling.ot_entropy import (
    _sinkhorn_implicit_vjp,
    compute_cost_matrix,
    sinkhorn_iteration,
    compute_transport_matrix_from_potentials,
)
from tests.hmc._gradient_test_utils import save_result, reset_results
```

### 3.3 What it does, step by step
For each combination of:
- `N ∈ {100, 500, 1000}` — particle counts
- `dtype ∈ {float32, float64}` — precision
- `skewness ∈ {"uniform", "mild", "heavy", "pathological"}` — weight distribution

do the following:
1. **Build weights** for the chosen skewness level. Concrete recipes:
   - `uniform`: `w_i = 1/N`.
   - `mild`: weights drawn from `Dirichlet(α=1.0)` — moderate variance.
   - `heavy`: weights drawn from `Dirichlet(α=0.1)` — a few particles dominate.
   - `pathological`: one particle at `1 − (N−1)·1e−6`, others at `1e−6` each — extreme corner case the hypothesis predicts should explode.
2. **Build particles** from `N(0, I_d)` with `d=2` (matches 2D SV state dim).
3. **Run the forward Sinkhorn** using the same code path the production filter uses:
   - Compute scaled particles (centered, divided by `std * sqrt(d) + 1e-8`).
   - `cost = compute_cost_matrix(scaled, scaled)`
   - `(alpha, beta, _) = sinkhorn_iteration(log_weights, cost, epsilon=0.5, ..., max_iter=100, threshold=1e-3)`
   - `T = compute_transport_matrix_from_potentials(scaled, alpha, beta, 0.5, log_weights)`
4. **Build a neutral upstream gradient** `dT`:
   - `dT = tf.random.normal(shape=(N,N), stddev=1.0)` with a fixed seed.
   - Unit-normalize so `||dT||_F = 1` — this makes gradient magnitudes across cases directly comparable (the output norm is a condition-number-like quantity).
5. **Apply the same `P = T / N_float`, `dP = dT * N_float` rescaling** that the production custom-gradient wrapper uses (`ot_entropy.py:553–554`) — otherwise we are not testing the actual VJP path.
6. **Call `_sinkhorn_implicit_vjp(dP, P, epsilon=0.5)`** — the function under test.
7. **Measure inside the function's code path** (done by re-implementing the same `K` construction in the test, not by touching the production function):
   - `cond_K = np.linalg.cond(K.numpy())`
   - `sol_norm = np.linalg.norm(sol)`
   - `rhs_norm = np.linalg.norm(rhs)`
   - `amplification = sol_norm / max(rhs_norm, 1e-30)` — the per-call amplification factor.
8. **Measure the output**:
   - `grad_p_norm = tf.norm(grad_p)`
   - `grad_b_norm = tf.norm(grad_b)`
   - `grad_out_norm = sqrt(grad_p_norm**2 + grad_b_norm**2)`
   - `output_amplification = grad_out_norm / ||dT||` (= `grad_out_norm / 1.0` with unit input).

### 3.4 Record schema (one row per (N, dtype, skewness) combination)
```json
{
  "test": "sinkhorn_vjp_conditioning",
  "N": 1000,
  "dtype": "float32",
  "skewness": "pathological",
  "epsilon": 0.5,
  "cond_K": 1.8e9,
  "sol_norm": 4.2e5,
  "rhs_norm": 1.3e0,
  "amplification": 3.2e5,
  "grad_p_norm": 1.1e6,
  "grad_b_norm": 2.4e5,
  "grad_out_norm": 1.12e6,
  "output_amplification": 1.12e6,
  "forward_T_min": 1e-42,
  "forward_T_max": 9.9e-4,
  "sinkhorn_converged": true,
  "passed": false
}
```
Results saved to `code/tests/unit/results/test_sinkhorn_vjp_conditioning.json` via `save_result`.

### 3.5 Pass/fail criteria

| Condition | Expected `output_amplification` | Flag |
|-----------|--------------------------------|------|
| uniform, any N, any dtype | < 10 | pass |
| mild, any N, any dtype | < 100 | pass |
| heavy, N ≤ 500, float64 | < 1e3 | pass |
| heavy, N = 1000, float32 | ??? (this is the interesting cell) | measure |
| pathological, any config | > 1e4 (hypothesis) | flag "hypothesis consistent" |

The unit test itself should **not** assert-fail on the interesting cells — it should record the numbers. Assertions only on the "uniform should be small" sanity check, so the test fails loudly if the hypothesis is wrong in the opposite direction (stable even on pathological inputs).

### 3.6 Interpretation matrix

| `cond_K` at (N=1000, float32, heavy) | `cond_K` at (N=1000, float64, heavy) | Interpretation |
|--------------------------------------|--------------------------------------|----------------|
| Huge | Small | It's a precision problem → test at float64 as workaround |
| Huge | Huge | It's a conditioning problem independent of precision → need to regularize `K` |
| Small | Small | Hypothesis is wrong → go back to search |
| Small | Huge | Bizarre, indicates a bug in the test itself |

### 3.7 Runtime estimate
- CPU only, no GPU needed.
- Each Sinkhorn run is ~50 iterations on a 1000×1000 matrix → seconds.
- Full grid: 3 × 2 × 4 = 24 cases × ~10s each ≈ 4 minutes wall time.
- Test write time: ~20 minutes.

### 3.8 What is **not** in scope for Test 1
- This test does not run a filter. It does not run MAP. It does not hit the GPU.
- This test does not modify any production file.
- This test reconstructs `K` once for measurement purposes; that is local re-implementation inside the test, not a change to `_sinkhorn_implicit_vjp`.

---

## 4. Test 2 — End-to-end attribution via monkeypatched `_sinkhorn_implicit_vjp`

### 4.1 File
`code/tests/filters/test_ledh_ot_grad_attribution.py` (new file)

### 4.2 What "monkeypatch" means here — and why it's not touching production
Test file replaces the symbol `src.resampling.ot_entropy._sinkhorn_implicit_vjp` in the running Python process with a *wrapper* that forwards every call to the original function and, on each call, records:
- the step index of the enclosing MAP loop,
- `cond(K)`,
- `||sol||` and `||rhs||`,
- `||grad_p||`, `||grad_b||`.

The wrapper is installed in `setUp` / fixture and removed in `tearDown`. The production file on disk is unchanged. The production function's behavior is unchanged (the wrapper calls it unchanged). This is a standard `unittest.mock.patch`-style instrumentation pattern.

**Caveat we need to resolve before writing this test:** `_sinkhorn_implicit_vjp` is called inside a `tf.custom_gradient` backward function (`ot_entropy.py:549` and `:603`), which may be traced inside a `tf.function` + `jit_compile=True` via the LEDH filter's `compiled_filter`. If the function is fully XLA-compiled, monkeypatching won't work — the wrapper's Python-level logging would be stripped out of the compiled graph. **This is an open question I need to resolve before writing Test 2.**

Two fallbacks if monkeypatching inside the compiled graph fails:
- **Fallback A:** temporarily disable JIT for the diagnostic run by passing `eager_mode: true` in the filter config (already a supported knob in `ledh_invertible_hmc.yaml`). This is a config change, not a production code change.
- **Fallback B:** run Test 2 outside the filter — feed the filter's per-step transport matrices into a standalone test harness and measure there. This loses end-to-end context but preserves attribution.

I will investigate feasibility before writing Test 2, and report back which approach works before asking for approval to run it.

### 4.3 Experimental setup
**Config:** a shortened copy of `code/configs/dpf/map/stochastic_volatility_2d/ledh_ot_sigma2.yaml`, saved as `code/tests/filters/fixtures/ledh_ot_sigma2_short.yaml`:
- `filter.n_particles: 200` (down from 1000 for speed)
- `filter.n_lambda_steps: 29` (unchanged)
- `data.T: 50` (down from 200 for speed)
- `dpf.map.num_steps: 30`
- `dpf.map.random_seed: true`
- `filter.eager_mode: true` (if needed for monkeypatching — see §4.2)

**Procedure:**
1. `reset_results(__file__)`.
2. Install monkeypatch on `_sinkhorn_implicit_vjp`.
3. Run the 30-step MAP loop via `DPFRunner.run_map` — calling the same code path the user's failing run uses.
4. On each MAP step, record:
   - `step`, `loss`, `ll`, `grad_norm_sigma2` (from the runner's existing history lists),
   - from the monkeypatch: per-resample-event `cond_K`, `sol_norm`, `amplification`. A single MAP step may call `_sinkhorn_implicit_vjp` multiple times (once per time step where resampling fires), so we record an array of these per MAP step.
5. Remove monkeypatch.
6. Save full per-step record to JSON.

### 4.4 Record schema (one row per MAP step)
```json
{
  "test": "ledh_ot_grad_attribution",
  "step": 14,
  "loss": 479.2,
  "ll": -477.6,
  "grad_norm_sigma2": 112.7,
  "n_resample_events": 23,
  "max_cond_K": 4.8e8,
  "mean_cond_K": 1.2e6,
  "max_amplification": 8.1e4,
  "mean_amplification": 3.2e2,
  "per_event_cond_K": [1.1e3, 4.8e8, 2.3e4, ...],
  "per_event_amplification": [8.2, 8.1e4, 1.9e1, ...]
}
```
Saved to `code/tests/filters/results/test_ledh_ot_grad_attribution.json`.

### 4.5 Pass/fail / interpretation criteria
After the run, compute Spearman rank correlation between `grad_norm_sigma2` and `max_cond_K` across the 30 MAP steps.

| Correlation | Interpretation |
|-------------|----------------|
| ρ > 0.7 | Smoking gun. Spikes *are* the Sinkhorn VJP blowing up. |
| 0.3 < ρ ≤ 0.7 | Partial — `_sinkhorn_implicit_vjp` is a contributor but not the only one. Go back and look for co-contributors. |
| ρ ≤ 0.3 | Hypothesis wrong. Look elsewhere (flow Jacobian backward, log-weights clipping, etc.). |

Additional sanity check: the top-5 worst `grad_norm_sigma2` steps should, under the hypothesis, be the same steps (or a superset) as the top-5 worst `max_cond_K` steps.

### 4.6 Runtime estimate
- Needs GPU (to reproduce the CUDA/float32 failure mode).
- 30 MAP steps × 8s/step × (eager-mode slowdown, maybe 2-3×) ≈ 10–15 min GPU time.
- Write time: ~1 hour, plus upfront JIT-compatibility investigation (~30 min).

### 4.7 What is **not** in scope
- Not modifying any file under `code/src/`.
- Not modifying `hmc_runner.py` to add new logging hooks — all logging goes through the in-test monkeypatch.

---

## 5. Test 3 — Control: turn off the suspect and re-run

### 5.1 File
`code/tests/filters/test_ledh_ot_spike_elimination.py` (new file)

### 5.2 What it does
Runs the same short MAP config from §4.3 twice:
- **Run A (baseline):** current config, `filter.stop_gradient_resampling: false`. Should reproduce the spike pattern.
- **Run B (suspect off):** `filter.stop_gradient_resampling: true`. Gradients no longer flow through OT resampling; `_sinkhorn_implicit_vjp` is not called in the backward pass. If the hypothesis is right, the spikes should vanish.

Both runs use the same `data.seed=42` for the underlying observations but `random_seed=true` for the per-step PF seed, so the estimator-noise distribution is comparable between runs.

### 5.3 Config overrides (Hydra, no code change)
```
filter.stop_gradient_resampling=true
```
applied via Hydra override. No new config file required, and no `.yaml` under `code/configs/` is modified.

### 5.4 Metrics collected per run
From `result.diagnostics`:
- `grad_norm_history` (30 entries)
- `loss_history` (30 entries)
- `param_history['sigma2']` (30 entries)

Computed:
- `max(grad_norm_history)`, `std(grad_norm_history)`, `p95(grad_norm_history)`
- Count of steps with `grad_norm > 30` (a rough spike threshold)
- Final sigma2 trajectory shape

### 5.5 Record schema (one row per run)
```json
{
  "test": "ledh_ot_spike_elimination",
  "run": "baseline",   // or "stop_gradient_true"
  "grad_norm_max": 112.7,
  "grad_norm_std": 22.1,
  "grad_norm_p95": 58.3,
  "n_spikes_above_30": 6,
  "loss_std": 14.3,
  "sigma2_final": 1.52,
  "grad_norm_history": [...],
  "loss_history": [...]
}
```
Saved to `code/tests/filters/results/test_ledh_ot_spike_elimination.json`.

### 5.6 Pass/fail / interpretation criteria

| Baseline `n_spikes_above_30` | Stop-grad `n_spikes_above_30` | Interpretation |
|------------------------------|------------------------------|----------------|
| ≥ 3 | 0 | **Hypothesis confirmed.** The OT backward path is the spike source. |
| ≥ 3 | 1-2 | Mostly confirmed — OT dominates but a minor second source exists. |
| ≥ 3 | ≥ 3 | **Hypothesis wrong or incomplete.** Spikes are NOT from OT resampling. Redirect. |
| 0 | n/a | Baseline didn't reproduce — rerun with fresh seeds or check configuration. |

### 5.7 Runtime estimate
- Needs GPU.
- 2 × 30 MAP steps × 8s/step ≈ 8–10 min GPU time total.
- Write time: ~30 minutes.

### 5.8 Important caveat
Even if Test 3 says "spikes disappear with `stop_gradient_resampling=true`", that is a **diagnostic**, not a fix. Stopping the gradient through resampling is known to bias HMC (per the comments in `hmc_invertible_hmc.py` and git history). This test confirms causation, it does not prescribe the production change.

---

## 6. Existing overlap to check before writing

Before creating any of the three new test files, I will read:
- `code/tests/hmc/test_ot_gradient_standalone.py`
- `code/tests/hmc/test_sinkhorn_convergence.py`
- `code/tests/unit/test_resampling.py`
- `code/tests/hmc/_gradient_test_utils.py`
- `code/tests/hmc/sv2d_diagnostics/` contents

to confirm Tests 1 and 2 don't duplicate existing test harnesses. If an existing test already measures `cond(K)` or `_sinkhorn_implicit_vjp` output norms, I will extend it rather than duplicate. I will report any overlap before starting.

---

## 7. Execution order and gates

Per `feedback_one_fix_at_a_time.md` and `feedback_no_unilateral_decisions.md`, each step below waits for explicit user approval:

1. **Gate 1 — plan approval.** User reads this document and says "yes, write Test 1" (or objects).
2. **Write Test 1.** Show file to user before running.
3. **Run Test 1 on CPU.** Show JSON output and interpretation to user. User decides whether Test 2 is justified.
4. **Investigate Test 2 JIT/monkeypatch feasibility.** Report back whether in-graph monkeypatching works, whether eager-mode fallback is needed, or whether we take Fallback B.
5. **Write Test 2.** Show file to user before running.
6. **Run Test 2 on GPU (office).** Show JSON output and correlation analysis.
7. **Write Test 3.** Show file to user before running.
8. **Run Test 3 on GPU (office).** Show JSON output and before/after comparison.
9. **Synthesize findings.** Report the full causal chain (or the falsification) and wait for user instruction on next step (which will be to decide whether and how to *fix* the production code — a separate task, separate approval).

Nothing in the `code/src/` tree is touched at any point in this plan.

---

## 8. Open questions for user

1. **Location of this plan file:** I put it at `code/tests/docs/ledh_ot_gradient_variance_test_plan.md`. OK? Or do you want it at `code/` root, or somewhere else?
2. **Test 2 GPU access:** Test 2 and Test 3 need the office machine. Is office available now, or should I structure the plan so Test 1 can run locally while we wait?
3. **SV1D as a control:** should Test 3 also run once on SV1D + LEDH-OT as a control, to check whether the spikes are specific to the 2D state or apply to 1D too? This adds ~5 minutes of GPU time and would strengthen the diagnosis, but it's scope-creep from the original question.
4. **Codex re-review:** once Tests 1–3 are written (before running), do you want Codex to review the *test code* for neutrality/correctness? Per `feedback_no_unilateral_decisions.md`, I think the answer is yes — but confirm.

---

## 9. Summary

Three tests answer three yes/no questions that together prove or disprove one specific hypothesis: **the backward pass through entropy-regularized OT resampling blows up when a random PF seed produces skewed particle weights, and this is the shared root cause of MAP gradient spikes and HMC step-size collapse on LEDH-OT + SV2D.**

No production code is modified. All findings are written to JSON under `code/tests/*/results/`. Each step waits for user approval before proceeding.
