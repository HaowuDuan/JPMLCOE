# Fix HMC Graph-Mode Crashes & Add Eager Debug Mode

## Context

The `LEDHParticleFlowFilterHMC` subclass wraps the 29-step flow loop in `@tf.function` for HMC performance. This forces **graph-mode execution**, where TensorFlow raises `InvalidArgumentError` on singular matrices instead of silently returning NaN (eager-mode behavior). We've hit this crash three times so far, each requiring a 10-minute graph tracing cycle to discover. This plan fixes all remaining unsafe operations at once and adds an eager-mode toggle so future issues can be caught in seconds.

---

## Fix 1: Unsafe `tf.linalg.cholesky(Q)` in `compute_flow_weights`

**File:** [distributions.py:161](src/utils/distributions.py)

**Problem:** `compute_flow_weights()` is `@tf.function` (line 103) and called from the HMC tape path ([ledh_invertible_hmc.py:222](src/filters/particle/ledh_invertible_hmc.py)). It calls raw `tf.linalg.cholesky(Q)` where `Q = sigma_V^2`. During HMC leapfrog, sigma_V can approach 0, making Q singular. Graph-mode Cholesky will crash.

**Fix:** Replace `tf.linalg.cholesky(Q)` with `safe_cholesky(Q)` (from `utils/linalg.py`).

```python
# distributions.py — add import
from .linalg import safe_cholesky

# Line 161: replace
L_Q = tf.linalg.cholesky(Q)
# with
L_Q = safe_cholesky(Q)
```

`safe_cholesky` adds adaptive jitter (`jitter * max(avg_diag, 1.0) * I`), ensuring Q is always PD.

**Impact on standalone filtering:** None. `safe_cholesky` behaves identically to `tf.linalg.cholesky` when Q is already PD. The jitter (1e-10 scaled) is negligible for well-conditioned Q.

---

## Fix 2: Confirm `safe_inv` stays without `@tf.function`

**File:** [linalg.py:98](src/utils/linalg.py)

**Current state:** Already correct — `safe_inv` has no `@tf.function` decorator and the docstring explains why.

**Why this is safe for standalone filtering:** `safe_inv` is called from `update()` in `ledh_invertible.py` at line 222. This runs eagerly during `filter()`. No behavioral change.

**Why it must NOT have `@tf.function`:** It's called from `log_marginal_likelihood_tf()` at line 177 of `ledh_invertible_hmc.py`, which is eager context (outside the compiled flow loop). If it had `@tf.function`, TF would compile it separately and `MatrixInverse` would crash on singular R.

**No change needed.** Just confirming this is correct.

---

## Fix 3: Add `eager_mode` flag for debugging

**File:** [ledh_invertible_hmc.py](src/filters/particle/ledh_invertible_hmc.py)

**Problem:** Every bug requires 10 min of graph tracing before the error appears. The user needs a way to validate the forward pass instantly.

**Approach:** Add `eager_mode: bool = False` parameter to `__init__`. When True, `_build_compiled_flow()` returns a plain Python function (no `@tf.function`). The flow loop runs eagerly — errors appear immediately, and `GradientTape` still works (just slower).

Changes to `__init__`:
```python
def __init__(self, *args, eager_mode: bool = False, ...):
    ...
    self.eager_mode = eager_mode
    self._compiled_flow = self._build_compiled_flow()
```

Changes to `_build_compiled_flow`:
```python
def _build_compiled_flow(self):
    n_steps = self.n_lambda_steps

    def flow_loop(model, eta_0, eta_bar, particle_covs, eta_bar_0,
                  y, R, R_inv, lambda_steps, state_dim, regularization):
        # ... identical loop body ...
        return eta_1, eta_bar, log_theta

    if self.eager_mode:
        return flow_loop          # plain Python function
    else:
        return tf.function(flow_loop)  # compiled graph
```

**Config activation** (no code changes needed to switch):
```bash
python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc \
  filter.eager_mode=true dpf.hmc.num_samples=5 dpf.hmc.num_burnin=2 data.T=10
```

---

## Audit: No other unsafe operations in HMC path

| Operation | File | Line | Status |
|-----------|------|------|--------|
| `safe_log_abs_det(M_batch)` | ledh_invertible_hmc.py | 142 | Safe (jitter) |
| `safe_cholesky(S)` | flow_params.py | 232 | Safe (jitter) |
| `tf.linalg.cholesky_solve(L_S, H_batch)` | flow_params.py | 233 | Safe (uses L from safe_cholesky) |
| `safe_inv(R)` | ledh_invertible_hmc.py | 177 | Safe (eager, jitter) |
| `safe_cholesky(S)` | batched_ekf.py | 101 | Safe (jitter) |
| `tf.linalg.cholesky(Q)` | distributions.py | 161 | **UNSAFE — Fix 1** |

All other raw `tf.linalg.inv`/`cholesky` calls are in model files (acoustic_tracking, range_bearing, etc.) that are NOT in the HMC path.

---

## Files to modify

| File | Change |
|------|--------|
| `code/src/utils/distributions.py` | Add `from .linalg import safe_cholesky`, replace line 161 |
| `code/src/filters/particle/ledh_invertible_hmc.py` | Add `eager_mode` parameter and branching |

**Files NOT modified:** `linalg.py` (already correct), `ledh_invertible.py` (parent untouched), configs (eager_mode defaults to False).

---

## Verification

1. **Quick smoke test (eager mode, ~30s):**
   ```bash
   cd code && python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc \
     filter.eager_mode=true dpf.hmc.num_samples=2 dpf.hmc.num_burnin=1 data.T=5
   ```
   Should complete without `InvalidArgumentError`. Any crash appears instantly.

2. **Compiled mode test (~15 min including tracing):**
   ```bash
   cd code && python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc \
     dpf.hmc.num_samples=10 dpf.hmc.num_burnin=5
   ```
   Should trace the graph (~10 min), then run 15 HMC steps. No crashes.

3. **Standalone filtering (regression check):**
   Run the parent class filter to confirm `safe_cholesky` in `compute_flow_weights` doesn't change results.
