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

## B. HAS PLAN / REPORT

| # | Issue | Document | Status |
|---|-------|----------|--------|
| B1 | EKF linearization point — `m_i^+` discarded, next predict uses flowed particles instead of EKF means | `EKF_LINEARIZATION_POINT_ISSUE.md` | Not started |
| B2 | Resampling O(N^2) index hack — systematic/soft/OT return ancestor indices via `ResampleResult` | `RESAMPLING_REFACTOR_PLAN.md` | **DONE** — `ResampleResult` NamedTuple created; systematic/soft return `ancestor_indices`, OT returns `transport_matrix`; LEDH uses direct gather instead of O(N^2) nearest-neighbor |

---

## C. DONE — Bugs (from `codebase_analysis.md`)

| # | Issue | File(s) | What was done |
|---|-------|---------|---------------|
| C1 | Hardcoded pi truncates float64 | `distributions.py` | `import math`; all 3 occurrences → `tf.constant(math.pi, dtype=...)` |
| C2 | Bare `tf.linalg.cholesky` | `distributions.py` | `from .linalg import safe_cholesky`; replaced in `log_gaussian_prob` |
| C3 | Silent uniform weight fallback | `distributions.py` | Added `tf.print("WARNING: weight collapse...")` in `tf.cond` fallback |
| C4 | Batch log-prob missing bearing wrap | `two_sensor_bearing.py` | Added `diff = tf.atan2(tf.sin(diff), tf.cos(diff))` in `log_observation_prob_batch` |
| C5 | `self.observed_dims` undefined | `lorenz96.py` | `self.observed_dims` → `self.obs_indices` |
| C6 | Batch transition 1 RK4 step | `lorenz96.py` | Added `obs_interval` loop to `state_transition_mean_batch` |
| C7 | `log_det` discards sign | `linalg.py` | Added `tf.debugging.assert_positive(sign)` for PD enforcement |

---

## D. DONE — Investigation results (from `codebase_analysis.md`)

### D1. `ot_entropy.py` — `tf.stop_gradient` on centering/scaling → INTENTIONAL

**File:** `code/src/resampling/ot_entropy.py`, lines 395, 399
**Finding:** The `tf.stop_gradient` is intentional for numerical stability. The `@tf.custom_gradient` decorator on the resampling function already handles the backward pass correctly. The `stop_gradient` prevents gradients from flowing through the centering/scaling normalization (which is purely a numerical conditioning step), while the custom gradient provides the mathematically correct Jacobian. No change needed.

### D2. `ot_entropy.py` — Softmin order of operations → CODE CORRECT, DOCSTRING WRONG

**File:** `code/src/resampling/ot_entropy.py`, line 75
**Finding:** Code `f - cost_matrix / epsilon` is correct. The Sinkhorn potentials `f` are stored in epsilon-scaled form (called with `log_w + beta/eps`), so `f` already absorbs the `1/epsilon` factor. Only `cost_matrix` needs dividing by epsilon. The docstring's `(f_j - C_ij) / epsilon` has wrong parenthesization but the code is correct. No code change needed.

### D3. `hmc_runner.py` — `tf.py_function` architecture → DEFERRED

**File:** `code/src/DF/hmc_runner.py`, lines 57-101
**Status:** Architectural issue requiring significant refactor to make the filter pipeline run entirely inside `tf.function`. Deferred to a separate effort as planned.

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
