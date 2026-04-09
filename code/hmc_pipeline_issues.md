# HMC Pipeline Issues — Memory Leak, JIT, & Gradient Instability

Date: 2026-04-08, updated 2026-04-09

## Summary

Smoke tests show 13-18GB memory increases for moderate experiments. JIT/XLA compilation is not working end-to-end. Single HMC step takes 51s for SV2D LEDH with 500 particles + 200 timesteps. Gradient explodes after a few leapfrog steps, causing step size to collapse to zero.

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

## Priority Fix Order

1. Fix LEDH observation model for SV2D (A1, A2) — correctness
2. Add gradient clipping to TFP HMC path (A6) — prevents step-size collapse
3. Set `always_resample=True` for HMC (A5) — smooth likelihood surface
4. Reduce tape footprint: `stop_gradient_resampling` or reduce T (B1)
5. Stop creating tf.Variable per likelihood call (B2)
6. Add `jit_compile=True`, remove `.python_function` (C1)
7. Replace `.numpy()` seed extraction (C2)
8. Regularize Jacobian backward (A3) and OT implicit VJP (A4)
