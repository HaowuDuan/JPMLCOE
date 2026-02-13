# Performance Optimization Plan

## Context

The `PERFORMANCE_DIAGNOSIS.md` identified bottlenecks across the particle flow filter codebase. Several quick wins have already been applied (parallel_iterations=10, removed explicit tf.constant in loops, vectorized resampling, precomputed lambda schedule in edh_flow/ledh_invertible). The **dominant remaining bottleneck** is the per-particle sequential computation in LEDH filters: 29 lambda steps x N particles, each calling `compute_flow_params` individually. This plan addresses all remaining optimizations in dependency order.

---

## Phase 1: Quick Wins (no dependencies)

### 1A. Precompute step sizes in `edh_invertible.py`
**File:** `code/src/filters/particle/edh_invertible.py`
- `_generate_step_sizes()` (line 181-188) is called every `update()` (line 243) but returns a constant
- Move computation into `__init__()`, store as `self.lambda_steps` (same pattern as `ledh_invertible.py` line 122-129)
- Replace `step_sizes = self._generate_step_sizes()` at line 243 with `self.lambda_steps`

### 1B. Pre-allocate tf.Variables in `ledh_invertible.py` predict()
**File:** `code/src/filters/particle/ledh_invertible.py`
- Lines 184, 192, 198 create new `tf.Variable` objects every timestep
- Pre-allocate in `initialize()` with zero tensors, then use `.assign()` in `predict()`
- Same fix for `edh_invertible.py` lines 203, 225, 237

---

## Phase 2: Batch Jacobian Interface on Models

This unlocks Phases 3 and 4. Add three new batch methods to `model_base.py` and implement efficient overrides on TF-based models.

### 2A. Add default methods to `model_base.py`
**File:** `code/src/core/model_base.py`

Add after line 144:
- `observation_jacobian_batch(particles)` -> `(N, obs_dim, state_dim)` — default: `tf.map_fn` over `observation_jacobian`
- `observation_function_batch(particles)` -> `(N, obs_dim)` — default: `tf.map_fn` over `observation_function`
- `state_jacobian_batch(particles)` -> `(N, state_dim, state_dim)` — default: `tf.map_fn` over `state_jacobian`

### 2B. Efficient overrides for TF-based models

**`LinearGaussianModel`** (`code/src/models/linear_gaussian.py`):
- `observation_jacobian_batch`: broadcast constant `self.H` to `(N, obs_dim, state_dim)`
- `observation_function_batch`: `particles @ H.T` -> `(N, obs_dim)`
- `state_jacobian_batch`: broadcast constant `self.F`

**`RangeBearingModel`** (`code/src/models/range_bearing.py`):
- `observation_jacobian_batch`: vectorized `dx`, `dy`, `range` computation across all N particles, stack into `(N, 2, 2)`
- `observation_function_batch`: vectorized `[sqrt(dx^2+dy^2), atan2(dy,dx)]` -> `(N, 2)`

**`TwoSensorBearingOnlyModel`** (`code/src/models/two_sensor_bearing.py`):
- `observation_jacobian_batch`: vectorized for both sensors simultaneously
- `observation_function_batch`: vectorized `[atan2(dy1,dx1), atan2(dy2,dx2)]`

**`AcousticTrackingFullModel`** (`code/src/models/acoustic_tracking_full.py`):
- `observation_jacobian_batch`: vectorize across particles (sensor loop is compile-time constant under @tf.function)
- `observation_function_batch`: vectorize amplitude computation across particles

---

## Phase 3: Batch flow params + Batch EKF

### 3A. Add `compute_flow_params_batch()` to `flow_params.py`
**File:** `code/src/utils/flow_params.py`

New function alongside existing `compute_flow_params`. Returns `A_batch: (N, state_dim, state_dim)`, `b_batch: (N, state_dim)`.

**Signature handles two LEDH variants:**
- `linearization_points: (N, state_dim)` — always per-particle
- `P`: either `(N, sd, sd)` per-particle (LEDH invertible) or `(sd, sd)` global (LEDH flow) — broadcast if 2D
- `eta_bar_0`: either `(N, sd)` per-particle (LEDH invertible) or `(sd,)` global (LEDH flow / EDH) — broadcast if 1D
- `R`, `R_inv`: `(od, od)` — always shared, broadcast

Math (all batched TF ops):
```
# P is (N,sd,sd) or (sd,sd) — expand if needed: P_b = P[None] if 2D else P
H_batch = model.observation_jacobian_batch(points)         # (N, od, sd)
HP = H_batch @ P_b                                         # (N, od, sd)
HPH = HP @ transpose(H_batch)                              # (N, od, od)
S = lambda * HPH + R[None]                                 # (N, od, od)
L_S = safe_cholesky(S)                                     # (N, od, od)
S_inv_H = cholesky_solve(L_S, H_batch)                     # (N, od, sd)
A = -0.5 * P_b @ transpose(H_batch) @ S_inv_H             # (N, sd, sd)
h_batch = model.observation_function_batch(points)         # (N, od)
e = h_batch - einsum('nij,nj->ni', H_batch, points)       # (N, od)
# b computation mirrors flow_params.py lines 88-93, using einsum for batch
```

Key: TF's `cholesky`, `cholesky_solve`, `matmul` all support leading batch dims natively. `safe_cholesky` in `linalg.py` already supports `(..., n, n)` batch dims.

### 3B. Vectorize `batched_ekf_predict()`
**File:** `code/src/filters/kalman/batched_ekf.py`

Replace `tf.map_fn` with true batched matmul:
```
mean_pred = model.state_transition_mean_batch(means)          # (N, sd)
F_batch = model.state_jacobian_batch(means)                   # (N, sd, sd)
Q = model.state_transition_cov_batch(means)                   # (sd, sd)
cov_pred = F_batch @ covs @ F_batch^T + Q[None]               # (N, sd, sd)
```

### 3C. Vectorize `batched_ekf_update()`
**File:** `code/src/filters/kalman/batched_ekf.py`

Replace `tf.map_fn` with batched ops:
```
H_batch = model.observation_jacobian_batch(means)              # (N, od, sd)
y_pred = model.observation_function_batch(means)               # (N, od)
R = model.observation_cov(means[0])                            # (od, od)
S = H_batch @ covs @ H_batch^T + R[None]                      # (N, od, od)
K = covs @ H_batch^T @ inv(S)                                 # (N, sd, od)
mean_updated = means + einsum('nij,nj->ni', K, innovation)    # (N, sd)
cov_updated = (I[None] - K @ H_batch) @ covs                  # (N, sd, sd)
```

---

## Phase 4: Rewire LEDH Filters to Use Batch Ops

### 4A. Rewrite `ledh_invertible.py` update() flow loop
**File:** `code/src/filters/particle/ledh_invertible.py`

Replace the `tf.map_fn` + `_update_single_particle` (lines 259-287) with:
```python
for j in range(self.n_lambda_steps):
    d_lambda = self.lambda_steps[j]
    lambda_val = lambda_val + d_lambda

    # ONE batch call for all N particles (per-particle P_i from batched EKF)
    A_batch, b_batch = compute_flow_params_batch(
        self.model, eta_bar, lambda_val, y, particle_covs_tf,
        R, R_inv, eta_bar_0_tf, self.state_dim, regularization_tf
    )

    # Vectorized Euler steps
    drift = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
    eta_bar = eta_bar + d_lambda * drift

    drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
    eta_1 = eta_1 + d_lambda * drift_1

    # Vectorized log-det
    M = I[None] + d_lambda * A_batch                     # (N, sd, sd)
    log_theta = log_theta + tf.math.log(tf.abs(tf.linalg.det(M)))
```

This turns 29 x N sequential ops into 29 batched ops. Remove dead code `_update_single_particle()`.

**Note on P**: Each particle maintains its own independent EKF covariance P_i (line 214: `P_i = particle_covs_tf[i]`). The batch version takes `P_batch: (N, sd, sd)` — the full per-particle covariance tensor from the batched EKF. The `flow_params.py` docstring should be updated to reflect that P can be per-particle for LEDH.

### 4B. Rewrite `ledh_flow.py` _flow_step_euler()
**File:** `code/src/filters/particle/ledh_flow.py`

Replace the `tf.map_fn` at line 256 with:
```python
# P is global (sd,sd) from single EKF — compute_flow_params_batch broadcasts it
A_batch, b_batch = compute_flow_params_batch(
    self.model, particles, lambda_val, y, P, R, R_inv,
    eta_bar_0, self.state_dim, regularization_tf
)
drift = tf.einsum('nij,nj->ni', A_batch, particles) + b_batch

# Clip drift norms (vectorized)
drift_norms = tf.norm(drift, axis=1, keepdims=True)
scale = tf.minimum(1.0, 100.0 / (drift_norms + 1e-10))
drift = drift * scale

particles_new = particles + drift * d_lambda
```

Remove dead code `_compute_drift_single()`.

---

## Phase 5: Additional Optimizations (independent, lower priority)

### 5A. Cache R_inv in `ledh_invertible.py`
Currently `R_inv = tf.linalg.inv(R)` is computed every `update()` call (line 251). Cache it like `edh_invertible.py` does (lines 231-234).

### 5B. Cache R_tf in `ledh_flow.py`
`R_tf = tf.constant(self.model.observation_noise_cov, dtype=tf.float32)` is recreated every `update()` (line 311). Cache in `__init__` or first use.

---

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `code/src/core/model_base.py` | Add `observation_jacobian_batch`, `observation_function_batch`, `state_jacobian_batch` |
| `code/src/models/linear_gaussian.py` | Add vectorized overrides for batch methods |
| `code/src/models/range_bearing.py` | Add vectorized overrides for batch methods |
| `code/src/models/two_sensor_bearing.py` | Add vectorized overrides for batch methods |
| `code/src/models/acoustic_tracking_full.py` | Add vectorized overrides for batch methods |
| `code/src/utils/flow_params.py` | Add `compute_flow_params_batch()` |
| `code/src/filters/kalman/batched_ekf.py` | Replace `tf.map_fn` with batched matmul in both functions |
| `code/src/filters/particle/ledh_invertible.py` | Rewrite flow loop to batch, pre-allocate Variables, cache R_inv |
| `code/src/filters/particle/ledh_flow.py` | Replace `tf.map_fn` drift with batch, cache R_tf |
| `code/src/filters/particle/edh_invertible.py` | Precompute step sizes, pre-allocate Variables |

## Verification

After each phase, run an existing experiment (e.g., range_bearing) to confirm:
- Filter produces numerically consistent results (means, log-likelihood)
- No runtime errors or shape mismatches
- Measurable wall-clock speedup for Phases 3-4
