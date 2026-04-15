# HMC Pipeline Issues — Memory Leak, JIT, & Gradient Instability

Date: 2026-04-08, updated 2026-04-10

## Summary

Smoke tests show 13-18GB memory increases for moderate experiments. JIT/XLA compilation is not working end-to-end. Single HMC step takes 51s for SV2D LEDH with 500 particles + 200 timesteps. Gradient explodes after a few leapfrog steps, causing step size to collapse to zero.

## Status of fixes attempted in 2026-04-10 session

| ID | Issue | Status |
|---|---|---|
| B2 | tf.Variable creation per likelihood call | **FIXED** — `initialize()` removed from `log_marginal_likelihood_tf`, plain tensors built via `sample_initial_state_batch` |
| B3 | Gradients severed through `Sigma_0` (LEDH only) | **FIXED** — same fix as B2; `Sigma_0` now flows through tensor path |
| C2 (partial) | `.numpy()` seed extraction | **FIXED** — both filters use `[2]` int32 tensor + `stateless_split` |
| C4 | No `reduce_retracing=True` | **FIXED** — added to both compiled filters |
| (new) | Stale graph tensors leaked into model attrs after compiled call | **FIXED** — `try/finally` save/restore wrapping `_compiled_filter` |
| (new, related to A3) | XLA-incompatible `slogdet` in Jacobian custom_gradient | **MITIGATED** — added new function `graph_safe_log_abs_det_xla` (QR forward + same NaN-guarded backward) in `linalg.py`. Used by LEDH+HMC path. Old `_fast` variant untouched. |
| (new) | `tf.while_loop` missing `maximum_iterations` (XLA backward fails) | **FIXED** — added to LEDH outer loop, BPF outer loop, and Sinkhorn iteration |
| C1 | XLA end-to-end with `jit_compile=True` | **HIT FUNDAMENTAL TF LIMITATION** — see new section E |

All fixes are additive / no behavioral change to graph mode (verified: existing gradient tests pass). The XLA effort hit two distinct walls — see section E.

---

## A. Gradient Instability (explains step-size collapse)

### A1. LEDH observation model wrong for SV2D (HIGH)

- `stochastic_volatility_2d.py:138,255` — SV2D has multiplicative (state-dependent) observation noise
- `ledh_invertible_hmc.py:362`, `flow_params.py:219` — LEDH flow path assumes additive noise with constant R
- Model exposes `has_non_additive_obs_noise=True` but LEDH ignores it
- Result: flow doesn't move particles according to volatility; sigma2 only hits through weight correction → sharp likelihood cliffs
- **Fix**: Make LEDH flow aware of state-dependent observation noise, or use a different filter for SV2D.

### A2. EKF covariance wrong for SV2D (HIGH)

- `batched_ekf.py:89` — uses `model.observation_cov(means[0])` once and applies same R to all particles
- `stochastic_volatility_2d.py:185` — `observation_cov(x) = exp(x_2)`, so one particle's volatility sets every particle's update
- Feeds into `A_batch` / Jacobian terms → strong source of gradient instability
- **Fix**: Compute per-particle observation covariance.

### A3. Jacobian backward explodes on near-singular states (MEDIUM)

- `linalg.py:219,228` — `graph_safe_log_abs_det_fast()` uses direct inverse `M^{-T}` in custom gradient
- When `M = I + d_lambda * A` gets near-singular during 29-step flow, inverse spikes
- Only NaN guard, no clamping
- Matches "few leapfrog steps then gradient blows up" symptom
- **Fix**: Add condition-number clamping or regularization to Jacobian backward.

### A4. OT backward is ill-conditioned (HIGH)

- `ot_entropy.py:436,448` — `_sinkhorn_implicit_vjp()` solves dense `(2N-1) x (2N-1)` linear system per resample
- At N=500 that's 999x999, becomes ill-conditioned when transport plan is sharp
- Another direct path to exploding gradients, separate from LEDH Jacobian
- **Fix**: Regularize the implicit VJP solve or increase Sinkhorn epsilon.

### A5. Resampling creates non-smooth likelihood surface (HIGH)

- `ledh_invertible_hmc.py:52,314`, `bootstrap_pf_hmc.py:33,179` — `always_resample` defaults to False
- Code switches branches on `ESS < threshold` → piecewise-defined likelihood
- HMC integrates across discontinuities → gradient artifacts
- **Fix**: Set `always_resample=True` for HMC runs, or use soft resampling.

### A6. grad_clip_norm does nothing on TFP HMC path (HIGH)

- `hmc_runner.py:138,148,175` — `grad_clip_norm` accepted but only used in `custom_hmc` branch
- Standard `tfp.mcmc.HamiltonianMonteCarlo` has no gradient clipping
- **Fix**: Wrap `target_log_prob_fn` to clip gradients via custom gradient, or add clipping inside the filter.

---

## B. Memory Leak

### B1. Full differentiable OT/LEDH tape across all timesteps (HIGH — primary memory driver)

- `ledh_ot_sigma2.yaml:19,23` — `stop_gradient_resampling: false` keeps full OT gradients
- With LEDH, OT also transports `covs`, so backward propagates through transported covariance tensors
- 200 timesteps × 29 lambda steps × 500 particles × dense Sinkhorn backward = massive tape
- This is the main explanation for 13-18GB memory growth, not just tf.Variable creation
- **Fix**: Use `stop_gradient_resampling: true` if full OT gradient isn't needed, or reduce T.

### B2. tf.Variable creation per likelihood call (HIGH)

- `ledh_invertible_hmc.py:359` — `log_marginal_likelihood_tf()` calls `initialize()` every evaluation
- `ledh_invertible.py:156` — allocates fresh `tf.Variable`s for particles, weights, covariances
- Variables are trainable by default → accumulate in TF's variable registry
- **Fix**: Initialize once, reuse. Mark `trainable=False`.

### B3. Dropped gradients through initial distribution (MEDIUM)

- `ledh_invertible.py:137,141,156` — `initialize()` converts `mu_0`/`Sigma_0` through NumPy before sampling
- For SV2D, `Sigma_0` depends on `a2` and `sigma2`, so derivatives through initial state are cut
- BPF does not have this defect (samples from live tensor model path)
- **Fix**: Keep initialization in TF tensor path.

---

## C. JIT / Performance

### C1. XLA not end-to-end (HIGH)

- `ledh_invertible_hmc.py:185`, `bootstrap_pf_hmc.py:116` — `@tf.function` without `jit_compile=True`
- LEDH uses `.python_function` on helpers with `jit_compile=True`, bypassing their XLA wrappers
  - `batched_ekf.py:13`, `flow_params.py:8`, `distributions.py:117`, `linalg.py:71`
- **Fix**: Add `jit_compile=True` to outer function, remove `.python_function` calls.

### C2. Eager `.numpy()` blocks full compilation (HIGH)

- `ledh_invertible_hmc.py:359`, `bootstrap_pf_hmc.py:214` — `seed[0].numpy()` forces eager
- `hmc_runner.py:242,250,251,263` — `.numpy()` every HMC iteration for logging/sync
- XLA log message is misleading — only inner helper gets XLA'd, outer loop stays Python/eager
- **Fix**: Use stateless TF seed splitting. Gate host sync behind a verbose flag.

### C3. OT cost matrix computed twice (MEDIUM)

- `ot_entropy.py:588,598,363` — `ot_entropy_resample()` computes cost matrix, then `compute_transport_matrix_from_potentials()` recomputes it
- Wasted O(N^2) per resample
- **Fix**: Pass cost matrix through.

### C4. No input signature / retracing (MEDIUM)

- Neither compiled filter has `input_signature` or `reduce_retracing=True`
- Different observation lengths → new `ConcreteFunction` cache entries
- **Fix**: Add input signature with `None` time dimension.

---

## D. Config Issues (now fixed)

- ~~1000 particles~~ → reduced to 500 across all HMC configs
- ~~num_steps inconsistent~~ → all MAP configs set to 300
- ~~target_accept_prob 0.9 on range_bearing~~ → lowered to 0.75

---

## E. XLA end-to-end — investigated 2026-04-10, blocked

### What we tried

Goal: enable `@tf.function(jit_compile=True)` on `compiled_filter` so the entire LEDH/BPF + OT pipeline runs as one XLA computation (potentially 2-5x speedup).

Test infrastructure: `code/tests/jit/test_jit_compile.py` (probe + speedup) and `code/tests/jit/test_jit_value_and_grad.py` (value+grad wrapper variant). Both use a test-only `_xla_recompile()` helper or new `value_and_grad_tf()` method — no production call sites changed.

### Wall 1: `tf.linalg.slogdet` not implemented for XLA_CPU_JIT

- `linalg.py:218` (`_graph_safe_log_abs_det_fast_impl`) calls `tf.linalg.slogdet`
- XLA_CPU_JIT has no kernel for `LogMatrixDeterminant`
- **Fixed** by adding `graph_safe_log_abs_det_xla` (QR forward + same backward) and switching the LEDH HMC call site

### Wall 2: TensorList crossing XLA/TF boundary (gradient via external `GradientTape`)

- When `jit_compile=True` is on the outer compiled_filter, TF builds a backward graph through `tf.while_loop` that requires a TensorList for stashing intermediate states
- That TensorList must cross from XLA-compiled forward to non-XLA backward
- TF documents this as unsupported: `Support for TensorList crossing the XLA/TF boundary is not implemented`
- **Forward-only XLA works**: measured **1.3x speedup** at T=20, N=200 (`speedup.ledh_ot.forward` in test results)

### Wall 3: `Max_grad/Sum` requires compile-time constant (gradient via internal `GradientTape`)

- Workaround for Wall 2: put `GradientTape` inside another `@tf.function(jit_compile=True)` so the whole forward+backward stays in one XLA region
- Prototype: new method `value_and_grad_tf()` added to both filter classes (additive only — production code untouched)
- New error: `gradients/Max_grad/Sum` requires compile-time constant axis
- Source: `tf.reduce_max(log_theta)` in LEDH compiled_filter at line 245 (and a similar one in `compute_flow_weights` at `distributions.py:68`)
- **Mathematically fixable** with `tf.stop_gradient(tf.reduce_max(...))` (logsumexp trick — gradient is identical) but requires touching production hot-path code
- **Not applied** — out of scope for this session's "no production behavior changes" constraint

### Why option (a) — internal GradientTape — wasn't pursued further

Option (a) requires either:
1. Adding `stop_gradient` on `max_log_theta` in production `compiled_filter` (mathematically safe via LSE trick but touches hot path)
2. AND/OR refactoring `hmc_runner.py` to call `value_and_grad_tf()` instead of doing external `GradientTape`

Both were explicitly out of scope ("don't touch working code" / "the runner refactor is unacceptable").

### What stays in the repo from this XLA arc

| File | What's there | Active? |
|---|---|---|
| `linalg.py` | `graph_safe_log_abs_det_xla` (new function) | Used by LEDH HMC path; old `_fast` variant untouched |
| `ledh_invertible_hmc.py` | `value_and_grad_tf()` method (new, additive) | **Unused in production**; only the JIT test calls it |
| `bootstrap_pf_hmc.py` | `value_and_grad_tf()` method (new, additive) | **Unused in production**; only the JIT test calls it |
| `tests/jit/test_jit_compile.py` | XLA probe + forward speedup (1.3x) | Documents the wall |
| `tests/jit/test_jit_value_and_grad.py` | Value+grad wrapper test | Documents Wall 3 |
| Backups: `*_old.py` for `ledh_invertible_hmc`, `bootstrap_pf_hmc`, `hmc_runner` | Pre-session snapshots | For easy revert |

### Expected vs measured speedup

- **Forward-only XLA**: 1.3x measured. Modest. Doesn't justify the engineering effort to make gradient work.
- **Forward+backward XLA**: would need to clear Wall 3 (stop_gradient surgery on production code) AND refactor runner (option a). Not pursued.

### Conclusion

XLA end-to-end on the HMC pipeline is blocked by a chain of TF/XLA limitations that compound with the existing architecture. The 1.3x forward-only speedup is not worth the integration cost. **Recommend dropping the XLA effort and focusing on the gradient instability issues (A1-A6) which have much larger expected impact** (e.g., fixing the SV2D step-size collapse from `target_accept_prob` adaptation killing the chain).

---

## Priority Fix Order (revised after 2026-04-10)

1. ~~B2: Stop creating tf.Variable per likelihood call~~ — **DONE**
2. ~~C2: Replace `.numpy()` seed extraction~~ — **DONE**
3. ~~C4: Add reduce_retracing~~ — **DONE**
4. ~~Stale tensor leak from setattr inside compiled_filter~~ — **DONE** (try/finally save/restore)
5. Fix LEDH observation model for SV2D (A1, A2) — correctness, biggest expected win
6. Add gradient clipping to TFP HMC path (A6) — prevents step-size collapse on SV2D
7. Set `always_resample=True` for HMC configs (A5) — smooth likelihood surface
8. Investigate `stop_gradient_resampling` for HMC (B1) — primary memory driver, separate investigation deferred
9. Regularize Jacobian backward (A3) and OT implicit VJP (A4)
10. ~~XLA end-to-end (C1)~~ — **BLOCKED, dropped**
