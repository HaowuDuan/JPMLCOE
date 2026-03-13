# XLA JIT Compilation Plan (`jit_compile=True`)

## Background

The codebase currently uses `@tf.function` (graph mode) extensively but never enables XLA via `jit_compile=True`. XLA provides additional optimization beyond graph tracing:
- **Kernel fusion**: multiple ops become a single GPU kernel launch
- **Memory layout optimization**: reduces data movement
- **Dead code elimination**: removes unused computation paths

This plan groups all `@tf.function` sites by priority and explains **why** each is or isn't a good candidate.

---

## Phase 1: Critical Hot Paths (Highest Impact)

These functions are called in the innermost loops of the flow filters — they execute `n_lambda_steps` (e.g., 29) times **per timestep**, so even small per-call speedups compound significantly.

### 1.1 `compute_flow_params()` — `src/utils/flow_params.py:8`
- **Why**: Called 29x per timestep per particle. Contains observation_jacobian + safe_solve + matmul + matvec. Fusing these into a single kernel eliminates ~5 kernel launches per call × 29 steps.
- **XLA-safe**: Yes — all ops (matmul, solve, matvec) are XLA-compatible. `regularization` parameter is a Python-level branch resolved at trace time.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 1.2 `compute_flow_params_batch()` — `src/utils/flow_params.py:100`
- **Why**: Batched variant used in per-particle LEDH. Same hot-path reasoning.
- **XLA-safe**: Yes — batched matmul + safe_solve.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 1.3 `compute_flow_params_batch_stochastic()` — `src/utils/flow_params.py:167`
- **Why**: Stochastic EDH variant, same inner-loop frequency.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 1.4 `compute_flow_params_global()` — `src/utils/flow_params.py:273`
- **Why**: Global EDH variant.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 1.5 `compute_flow_weights()` — `src/utils/distributions.py:117`
- **Why**: Called every update step. Contains safe_cholesky + triangular_solve + reduce_sum + tf.cond (NaN fallback). Kernel fusion helps substantially.
- **XLA-safe**: Yes — `tf.cond` is XLA-compatible, all linalg ops are supported.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

---

## Phase 2: Batched Kalman Operations (Medium Impact)

These are called once per timestep in LEDH filters but operate on `(n_particles, state_dim, state_dim)` batches, so XLA's batched kernel fusion helps.

### 2.1 `batched_ekf_predict()` — `src/filters/kalman/batched_ekf.py:13`
- **Why**: Batched over n_particles. Contains state_transition_mean_batch + state_jacobian_batch + batched matmul + symmetrize. XLA can fuse the matmul chain.
- **XLA-safe**: Yes — pure batched linalg.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 2.2 `batched_ekf_update()` — `src/filters/kalman/batched_ekf.py:61`
- **Why**: Same reasoning — observation_jacobian_batch + cholesky_solve + Joseph form update.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 2.3 `batched_ukf_predict()` — `src/filters/kalman/batched_ukf.py:43`
- **Why**: UKF sigma-point propagation batched over particles.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 2.4 `batched_ukf_update()` — `src/filters/kalman/batched_ukf.py:75`
- **Why**: Same reasoning.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 2.5 `batched_ukf_sigma_points()` — `src/filters/kalman/batched_ukf.py:135`
- **Why**: Generates sigma points for all particles at once.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

---

## Phase 3: Core Linear Algebra Utilities (Low-Medium Impact)

These are called by the hot-path functions above. When the caller has `jit_compile=True`, XLA will already inline these. Adding `jit_compile=True` here ensures they're also compiled when called standalone.

### 3.1 `safe_cholesky()` — `src/utils/linalg.py:7`
- **Why**: Called from compute_flow_weights, batched_ekf, etc. When called outside a jit-compiled caller, this ensures it's still compiled.
- **XLA-safe**: Yes — cholesky + trace + reshape + eye.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 3.2 `safe_solve()` — `src/utils/linalg.py:48`
- **Why**: Called from compute_flow_params. Method parameter is a Python string resolved at trace time (not dynamic).
- **XLA-safe**: Yes — cholesky_solve or solve or lstsq, all XLA-ok.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 3.3 `symmetrize()` — `src/utils/linalg.py:285`
- **Why**: Trivial (transpose + add), but called frequently. Compilation eliminates kernel launch overhead.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 3.4 `matrix_sqrt()` — `src/utils/linalg.py:299`
- **Why**: Used in UKF sigma-point generation.
- **XLA-safe**: Yes — cholesky or eigh path, both supported.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 3.5 `graph_safe_log_abs_det_fast()` — `src/utils/linalg.py:240`
- **Why**: Used in flow filters for Jacobian log-determinant. Custom gradient avoids MatrixInverse.
- **XLA-safe**: Yes — slogdet forward + NaN-guarded inv backward.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 3.6 `graph_safe_log_abs_det()` — `src/utils/linalg.py:186`
- **Why**: Alternative variant using pinv in backward pass.
- **XLA-safe**: Yes — slogdet + pinv (SVD-based).
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

### 3.7 `graph_safe_log_abs_det_svd()` — `src/utils/linalg.py:263`
- **Why**: Most robust variant. SVD is expensive but XLA handles it.
- **XLA-safe**: Yes.
- **Change**: `@tf.function` → `@tf.function(jit_compile=True)`

---

## Phase 4: Distribution & Resampling Utilities (Low Impact)

Small functions with simple ops. XLA benefit is marginal since they're already fast, but compilation is safe and removes per-call overhead.

### 4.1 `log_gaussian_prob()` — `src/utils/distributions.py:9`
- **XLA-safe**: Yes. **Change**: Add `jit_compile=True`

### 4.2 `log_sum_exp()` — `src/utils/distributions.py:38`
- **XLA-safe**: Yes. **Change**: Add `jit_compile=True`

### 4.3 `normalize_log_weights()` — `src/utils/distributions.py:55`
- **XLA-safe**: Yes. **Change**: Add `jit_compile=True`

### 4.4 `multivariate_normal_sample()` — `src/utils/distributions.py:80`
- **XLA-safe**: Yes. **Change**: Add `jit_compile=True`

### 4.5 `effective_sample_size()` — `src/resampling/diagnosis.py:11`
- **XLA-safe**: Yes. **Change**: Add `jit_compile=True`

### 4.6 `normalize_log_weights()` — `src/resampling/diagnosis.py:29`
- **XLA-safe**: Yes. **Change**: Add `jit_compile=True`

### 4.7 `normalize_weights()` — `src/resampling/diagnosis.py:47`
- **XLA-safe**: Yes. **Change**: Add `jit_compile=True`

---

## Phase 5: ODE Solvers (Low Impact)

Thin wrappers — XLA benefit is small but safe.

### 5.1 `euler_step()` — `src/utils/ode_solvers.py:7`
- **XLA-safe**: Yes, provided the drift function `f` is also XLA-compatible (it is, since it's a model method traced into the graph).
- **Change**: Add `jit_compile=True`

### 5.2 `rk4_step()` — `src/utils/ode_solvers.py:24`
- **XLA-safe**: Yes, same caveat.
- **Change**: Add `jit_compile=True`

### 5.3 `euler_maruyama_step()` — `src/utils/ode_solvers.py:56`
- **XLA-safe**: Yes.
- **Change**: Add `jit_compile=True`

### 5.4 `integrate_ode()` — `src/utils/ode_solvers.py:87`
- **XLA-safe**: Yes — uses `tf.range` loop (graph control flow).
- **Change**: Add `jit_compile=True`

---

## Phase 6: OT Resampling (Needs Testing)

### 6.1 Component functions — `src/resampling/ot_entropy.py`
- `compute_cost_matrix` (line 21), `softmin` (line 45), `compute_diameter` (line 92), `stabilize_marginals` (line 124): All XLA-safe individually.
- **Change**: Add `jit_compile=True` to these 4.

### 6.2 Sinkhorn loops — `src/resampling/ot_entropy.py`
- `sinkhorn_step_forward` (line 234), `sinkhorn_fwd_backward` (line 333): Use `tf.while_loop` internally for convergence. XLA supports `tf.while_loop` but the dynamic iteration count may cause issues on some backends.
- **Change**: Add `jit_compile=True` but **test carefully** — if compilation fails or produces wrong results, revert these two.

---

## DO NOT COMPILE — With Reasons

### `log_det()` — `src/utils/linalg.py:83`
- **Why not**: Contains `tf.debugging.assert_positive` which raises runtime errors. XLA cannot handle assertions — they either get silently dropped or cause compilation failure.
- **Action**: Leave as `@tf.function` (no `jit_compile`).

### `safe_log_abs_det()` — `src/utils/linalg.py:121`
- **Why not**: Uses `tf.linalg.det` whose backward pass calls `MatrixInverse`. On CUDA, `MatrixInverse` crashes in compiled mode for singular/near-singular matrices. This is the known source of weight collapse on CUDA.
- **Action**: Leave as-is. Prefer `graph_safe_log_abs_det_fast` in calling code.

### `safe_inv()` — `src/utils/linalg.py:99`
- **Why not**: Explicitly designed to run eagerly (comment says "NOT @tf.function"). Returns NaN/Inf in eager mode instead of raising — needed for graceful fallback.
- **Action**: Do not add any decorator.

### `graph_safe_inv()` — `src/utils/linalg.py:151`
- **Why not**: Not decorated by design — uses `linalg.pinv` (expensive SVD). Adding compilation wouldn't help because it's called rarely and pinv compilation time is high.
- **Action**: Leave undecorated.

### `_negative_log_posterior()` — `src/DF/hmc_runner.py`
- **Why not**: Runs eagerly by design to preserve gradient chain with TFP's HMC. Comment says "NOT @tf.function — either runs eagerly or is traced by TFP internally."
- **Action**: Do not touch.

### `_negative_log_posterior()` — `src/DF/mh_runner.py`
- **Why not**: Same reasoning — "No GradientTape, no tf.function" by design.
- **Action**: Do not touch.

### `filter_tf()` — `src/filters/particle/bootstrap_pf_tf.py:117` (main filter loop)
- **Why not**: Uses `tf.unique` at line ~198 whose output shape is data-dependent. XLA requires all shapes to be statically known or bounded. Also uses `dynamic_size=True` TensorArray.
- **Action**: Leave as `@tf.function`. If XLA is desired later, replace `tf.unique` with a fixed-shape alternative (e.g., sort-based unique count).

### EKF/UKF `_predict_step`, `_update_step` methods — `extended_kalman.py`, `unscented_kalman.py`
- **Why not**: Already have `reduce_retracing=True` and are called from outer loops that may change shapes (different models, different state dims). Adding `jit_compile=True` would cause recompilation for each new shape, which is expensive. The batched variants (Phase 2) are the right place for XLA.
- **Action**: Leave as `@tf.function(reduce_retracing=True)`.

### `kalman.py` filter methods
- **Why not**: Comment says "NOT decorated with @tf.function — runs eagerly or is traced by TFP for HMC." These are entry points that TFP traces; adding jit_compile would conflict.
- **Action**: Do not touch.

### HMC filter loops — `ledh_invertible_hmc.py`, `bootstrap_pf_hmc.py`
- **Why not**: Complex control flow with `tf.while_loop` + `GradientTape` nesting + `tf.cond`. XLA may fail to compile or produce incorrect gradients. These are also traced by TFP's HMC kernel.
- **Action**: Do not add `jit_compile`. The inner functions they call (compute_flow_params, etc.) will already be XLA-compiled from Phase 1, giving indirect benefit.

### Model `@tf.function` methods — `range_bearing.py`, `acoustic_tracking_full.py`, `two_sensor_bearing.py`
- **Why not necessary**: These are small methods (single matmul, single atan2, etc.) that are always called from within a larger traced function. XLA on the caller already inlines and fuses them. Adding `jit_compile=True` to each individually would just cause redundant compilation.
- **Action**: Leave as `@tf.function`. They benefit indirectly when callers are jit-compiled.

### Particle filter drift functions — `edh_flow.py:169`, `edh_invertible.py:173,178`, `edh_flow_global.py:161`
- **Why not necessary**: Trivial ops (one matmul + add). Always called within the filter's flow integration loop which benefits from the caller's compilation. Standalone jit overhead not worth it.
- **Action**: Leave as `@tf.function`.

---

## Implementation Order & Testing Strategy

1. **Start with Phase 1** (5 functions). Run existing experiments on MPS first to verify:
   - No compilation errors
   - Results match eager/graph mode (compare summary.json outputs)
   - Speedup measurement (time the filter loop)

2. **Then Phase 2** (5 functions). Same verification.

3. **Phases 3-5** can be done in one batch — low risk.

4. **Phase 6** (OT resampling) last — test the Sinkhorn loop carefully.

## Potential Issues to Watch

- **First-run compilation time**: XLA compiles on first call. Subsequent calls are fast. If n_particles or state_dim changes between runs, recompilation occurs.
- **Numerical differences**: XLA may reorder floating-point operations. Compare RMSE/ESS values before and after. The MPS vs CUDA weight collapse issue could be affected.
- **Memory**: XLA may use more memory during compilation. Monitor GPU memory on RTX 3090.
- **Fallback**: If any function causes XLA compilation failure, simply remove `jit_compile=True` from that function — graph mode will be used automatically.
