# Full @tf.function Compilation of HMC Filter Loop

## Context

Compiling only the inner flow loop (29 steps) gave zero speedup because the bottleneck is the outer `for t in range(T)` Python loop (T=100 timesteps). Each timestep creates ~20 tape entries for EKF predict/update, weights, resampling, etc. — totaling ~2000 eager tape entries that dominate the backward pass.

**Goal**: Wrap the entire filter loop (100 timesteps × 29 flow steps) in a single `@tf.function` using `tf.while_loop`. The backward pass runs entirely in C++ with one PartitionedCall op.

**Safety**: Only `ledh_invertible_hmc.py` is modified. Parent class, model code, and all utility functions stay untouched.

---

## Approach

Replace `tf.Variable.assign` loop with a purely functional `tf.while_loop` where all state (particles, weights, covs, seed) is carried as tensors through the loop.

### What changes

**File: `code/src/filters/particle/ledh_invertible_hmc.py`**

1. **`__init__`**: Build compiled filter via `_build_compiled_filter()` instead of `_build_compiled_flow()`

2. **`_build_compiled_filter()`**: New method. Returns a `@tf.function` containing:
   - `tf.while_loop` over T timesteps
   - Loop body inlines: EKF predict → stochastic transition → flow loop (29 steps, unrolled) → weights → log-lik → EKF update → conditional resampling
   - All state passed as loop-carried tensors: `(t, particles, weights, covs, seed_counter, total_log_lik)`
   - No `tf.Variable.assign` inside the function

3. **`log_marginal_likelihood_tf()`**: Rewritten to:
   - Call `self.initialize()` (creates Variables, runs eagerly — unchanged)
   - Extract initial tensor values from Variables
   - Call `self._compiled_filter(observations, particles, weights, covs, ...)`
   - Return the scalar `total_log_lik`

4. **Eager mode fallback**: When `eager_mode=True`, run the same functional logic but without `@tf.function` / `tf.while_loop` (plain Python for-loop). This replaces the current Variable.assign-based eager path.

### Key design decisions

| Issue | Solution |
|-------|----------|
| `self.model.t = t + 1` | Set inside while_loop body. `t` is symbolic tensor. Model methods use `tf.cast(self.t, dtype)` which traces correctly. |
| `self.seed_counter += 1` | Carry as int32 tensor in loop state. Increment functionally. |
| `if ess < threshold: resample()` | Replace with `tf.cond(ess < threshold, do_resample, no_op)`. Both return `(particles, weights, covs, seed)`. |
| `tf.Variable.assign` | Eliminated. All state is tensors passed through while_loop. |
| Flow loop (29 steps) | Inlined into while_loop body as Python for-loop. Unrolled once during tracing. |
| Model params (sigma_V, sigma_W) | Accessed via `self.model.sigma_V` — read at execution time, not trace time. Graph picks up new HMC proposals correctly. |

### What stays the same

- `initialize()` — still creates tf.Variables (needed by parent's `filter()`)
- `_resample_hmc()` — kept for the eager path
- Parent class `LEDHParticleFlowFilter` — completely untouched
- `batched_ekf.py`, `flow_params.py`, `distributions.py`, `linalg.py` — untouched
- `kitagawa.py` — untouched (already uses tf.cast/tf.cos for tensor-compatible t)
- Config files — untouched

---

## Expected performance

| Metric | Current (eager outer loop) | Compiled (tf.while_loop) |
|--------|---------------------------|--------------------------|
| Tape entries | ~2000 | 1 |
| Tracing time | ~10 min (flow only) | ~2-5 min (full loop, traced once) |
| Per-step time (post-trace) | ~750s | Target: ~5-50s |
| Backward pass | Python-dispatched per-op | C++ graph differentiation |

---

## Verification

1. **Eager smoke test** (instant, catches logic errors):
   ```bash
   cd code && python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc \
     filter.eager_mode=true dpf.hmc.num_samples=2 dpf.hmc.num_burnin=1 data.T=5
   ```

2. **Compiled quick test** (~2-5 min tracing):
   ```bash
   cd code && python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc \
     filter.n_lambda_steps=5 dpf.hmc.num_samples=3 dpf.hmc.num_burnin=2
   ```
   Compare step 1 time (includes tracing) vs steps 2-3 (post-trace). Steps 2-3 should be dramatically faster than 750s.

3. **Full run** (production):
   ```bash
   cd code && python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc \
     dpf.hmc.num_samples=10 dpf.hmc.num_burnin=5
   ```
