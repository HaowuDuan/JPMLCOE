# Consolidated Action Plan — Codebase Issues

## Context

Cross-referencing `codebase_analysis.md`, `some_issue.md`, the LEDH-vs-MATLAB analysis, `EKF_LINEARIZATION_POINT_ISSUE.md`, and `RESAMPLING_REFACTOR_PLAN.md` to produce a single prioritized action list. Items are categorized as DONE, HAS PLAN, or TODO.

---

## A. DONE (completed this session)

| # | Issue | File(s) | What was done |
|---|-------|---------|---------------|
| A1 | `P_reg` dead code (codebase_analysis 1.1) | `flow_params.py:71,80,91` | `P` → `P_reg` on all 3 lines |
| A2 | Hardcoded `tf.float32` identity (codebase_analysis 1.6) | `flow_params.py:89,174` | Infer from `P.dtype` / `P_b.dtype` |
| A3 | EKF simple form loses PD | `batched_ekf.py:107-116` | Joseph form: `(I-KH)P(I-KH)^T + KRK^T` |
| A4 | `tf.linalg.inv(S)` in EKF | `batched_ekf.py:98-102` | `safe_cholesky` + `cholesky_solve` |
| A5 | Float32 pipeline (LEDH path) | `ledh_invertible.py`, `acoustic_tracking_full.py`, `flow_params.py`, `batched_ekf.py`, `systematic.py`, `soft.py` | Configurable dtype, default `tf.float64` |

**Double-check before next experiment run:**
- Run LEDH on acoustic tracking to confirm no dtype mismatch errors at runtime (float64 propagation end-to-end).
- Verify `flow_params.py` `regularization=None` default works under `@tf.function` tracing (the LEDH filter always passes a tensor, so the `None` branch only executes if the unbatched version is called without the argument).

---

## B. HAS PLAN / REPORT (separate documents, not started yet)

| # | Issue | Document |
|---|-------|----------|
| B1 | EKF linearization point — `m_i^+` discarded, next predict uses flowed particles instead of EKF means | `EKF_LINEARIZATION_POINT_ISSUE.md` |
| B2 | Resampling O(N^2) index hack — systematic/soft/OT should return ancestor indices via `ResampleResult` | `RESAMPLING_REFACTOR_PLAN.md` |

---

## C. TODO — Bugs (from `codebase_analysis.md`)

### C1. HIGH: `distributions.py` — Hardcoded pi truncates float64

**File:** `code/src/utils/distributions.py`, lines 33, 166, 177
**Problem:** `tf.constant(3.14159265359, ...)` has only 11 significant figures. `math.pi` has 16.
**Fix:** `import math` at top; replace all 3 with `tf.constant(math.pi, dtype=...)`.
**Impact on LEDH:** This is in `compute_flow_weights` and `log_gaussian_prob`, both called every timestep. With float64 now enabled, the truncation matters.

### C2. MEDIUM: `distributions.py` — Bare `tf.linalg.cholesky` without safeguard

**File:** `code/src/utils/distributions.py`, line 29
**Problem:** `L = tf.linalg.cholesky(cov)` can fail if cov is near-singular. `safe_cholesky` exists and handles this.
**Fix:** `from .linalg import safe_cholesky`; replace `tf.linalg.cholesky(cov)` with `safe_cholesky(cov)`.

### C3. MEDIUM: `distributions.py` — Silent fallback to uniform weights

**File:** `code/src/utils/distributions.py`, lines 192-195
**Problem:** When any weight is NaN/Inf, all weights are silently replaced with 1/N. No warning logged. Hides numerical collapse.
**Fix:** Add `tf.print("WARNING: weight collapse detected, falling back to uniform")` inside the fallback branch of `tf.cond`. This prints during graph execution.

### C4. MEDIUM: `two_sensor_bearing.py` — Batch log-prob missing bearing wrap

**File:** `code/src/models/two_sensor_bearing.py`, line ~338
**Problem:** `diff = observation - means` without `tf.atan2(tf.sin(diff), tf.cos(diff))` wrapping. Single-particle version wraps correctly.
**Fix:** Add `diff = tf.atan2(tf.sin(diff), tf.cos(diff))` after `diff = observation - means`.

### C5. MEDIUM: `lorenz96.py` — Undefined attribute `observed_dims`

**File:** `code/src/models/lorenz96.py`, line 242
**Problem:** `self.observed_dims` should be `self.obs_indices`. Runtime crash.
**Fix:** `self.observed_dims` → `self.obs_indices`.

### C6. MEDIUM: `lorenz96.py` — Batch transition does only 1 RK4 step

**File:** `code/src/models/lorenz96.py`, line 233
**Problem:** `state_transition_mean_batch` does 1 step; single-particle version loops `obs_interval` times.
**Fix:** Add the `obs_interval` loop to the batch method.

### C7. LOW: `linalg.py` — `log_det` discards sign

**File:** `code/src/utils/linalg.py`, lines 93-94
**Problem:** `sign` from `slogdet` is discarded. For SPD matrices this is always +1 so it's benign, but the docstring says "positive definite" — if non-PD input is ever passed, the result is silently wrong.
**Fix:** Either assert PD or return `sign * logdet` and rename to `signed_log_det`.

---

## D. TODO — Needs investigation (from `codebase_analysis.md`)

### D1. HIGH: `ot_entropy.py` — `tf.stop_gradient` blocks gradient flow

**File:** `code/src/resampling/ot_entropy.py`, lines 395, 399
**Problem:** Centering/scaling of particles uses `tf.stop_gradient`, blocking gradients through the transport plan normalization.
**Action:** Investigate whether this was intentional for stability. If the Sinkhorn iterations converge fine without `stop_gradient`, remove it. If removing causes NaN gradients, keep it and document why.

### D2. HIGH: `ot_entropy.py` — Softmin order of operations

**File:** `code/src/resampling/ot_entropy.py`, line 75
**Problem:** Code does `f - cost_matrix / epsilon` (divides only cost by epsilon). Docstring says `(f - cost_matrix) / epsilon`.
**Action:** Check whether the Sinkhorn potential updates absorb the `1/epsilon` factor. If potentials `f` are stored in "epsilon-scaled" form (i.e., `f` is really `f/epsilon`), then `f - cost_matrix/epsilon` is correct. Need to trace the Sinkhorn loop to verify. If incorrect, fix to `(f - cost_matrix) / epsilon`.

### D3. CRITICAL: `hmc_runner.py` — `tf.py_function` breaks differentiability

**File:** `code/src/DF/hmc_runner.py`, lines 57-101
**Problem:** `tf.py_function` inside `@tf.function` breaks the computation graph. HMC uses numerical gradients only.
**Action:** This is an architectural issue. Investigate feasibility of making the filter pipeline run entirely inside `tf.function`. Likely requires eliminating all Python-side state in the filter. Defer to a separate effort.

---

## E. Quality of life (from `some_issue.md`)

| # | Item | Status | Notes |
|---|------|--------|-------|
| E1 | Suppress TF/Metal info messages | DONE | `os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')` at top of `run_experiment.py`; config-driven via `tf_log_level` in `config.yaml` |
| E2 | Progress tracking callback | DONE | `progress_callback: Optional[Callable[[int, int, float], None]]` added to 7 filter `filter()` methods; wired via `show_progress: true` in `config.yaml` |
| E3 | StochasticEDH MPS reproducibility | TODO | Force CPU when `device=auto` for StochasticEDH |

---

## Recommended execution order

**Immediate (before next experiment):**
1. Double-check A1-A5 by running LEDH on acoustic tracking (runtime dtype test)
2. C1 — hardcoded pi (3 line changes, high impact now that float64 is on)
3. C2 — safe_cholesky in distributions.py (1 line change)

**Next batch:**
4. C3 — weight collapse warning
5. C5 — lorenz96 attribute name fix (crash fix)
6. C6 — lorenz96 batch transition loop
7. C4 — two_sensor_bearing batch wrapping

**Investigation needed:**
8. D1 — ot_entropy stop_gradient (needs gradient testing)
9. D2 — ot_entropy softmin (needs Sinkhorn trace)
10. D3 — hmc_runner architecture (large scope, defer)

**When convenient:**
11. C7 — log_det sign
12. E3 — StochasticEDH MPS reproducibility
13. B1 — EKF linearization point (implement from report)
14. B2 — Resampling refactor (implement from plan)
