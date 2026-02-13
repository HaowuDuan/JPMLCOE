# Performance Diagnosis Report

Comprehensive analysis of performance bottlenecks in the `code/src/` folder.
Primary concern: `ledh_invertible` is extremely slow.

---

## 1. LEDH Invertible (`ledh_invertible.py`) — CRITICAL

### 1.1 Flow loop: nested Python loop (lines 300-331)

**The dominant bottleneck.** The `update()` method has:

```
for j in range(n_lambda_steps):          # 29 iterations
    for i in range(n_particles):          # 500 iterations  (via tf.map_fn)
        compute_flow_params(...)           # per-particle Jacobian + matrix ops
        euler_step(...)                    # 2x per particle (eta_bar, eta_1)
```

**Cost per observation:** 29 x 500 = **14,500 sequential particle updates.**
Each involves:
- `model.observation_jacobian(eta_bar[i])` — per-particle Jacobian
- `safe_solve(S, H)` — Cholesky solve of (obs_dim x obs_dim) system
- 4 matrix-matrix multiplies (state_dim x state_dim)
- 5 matrix-vector multiplies
- 2 Euler steps

Even though the inner loop uses `tf.map_fn` (line 318), it runs with
`parallel_iterations=1`, so it executes **sequentially**. The `tf.map_fn`
avoids retracing overhead vs raw Python loop, but does not parallelize.

**Suggestion:** Vectorize the inner particle loop into batched TF operations.
For each lambda step, instead of processing particles one at a time:
- Compute `observation_jacobian` for all N particles at once → `H_batch: (N, obs_dim, state_dim)`
- Compute `A_batch: (N, state_dim, state_dim)` and `b_batch: (N, state_dim)` via batched matmul
- Apply Euler step to all particles simultaneously: `eta = eta + dt * (A_batch @ eta + b_batch)`
  using `tf.einsum('nij,nj->ni', A_batch, eta) + b_batch`
- Compute batch log-det: `tf.linalg.det(I + dt * A_batch)` → `(N,)`

This would turn the 29 x 500 sequential ops into 29 batched ops.
**Requires:** Adding `observation_jacobian_batch(particles)` to models.

### 1.2 `batched_ekf.py` uses `tf.map_fn` (lines 47-54, 111-118)

Both `batched_ekf_predict` and `batched_ekf_update` use `tf.map_fn`
over particle index, which runs **sequentially** (not parallel).

For the linear model (state_transition = F, observation = H), this is
wasteful because F, Q, H, R are all constant across particles:
- Predict: `mean_pred = means @ F^T`, `cov_pred = F @ covs @ F^T + Q` (batched matmul)
- Update: `S = H @ covs @ H^T + R`, `K = covs @ H^T @ S_inv` (batched)

TF's `tf.linalg` ops like `matmul`, `inv`, `solve` all support leading
batch dimensions natively. No `tf.map_fn` needed.

**Suggestion:** Replace `tf.map_fn` with true batched linear algebra
using `tf.matmul` with batch dimensions. For nonlinear models, you'd
still need `tf.map_fn` for the Jacobian, but the matrix arithmetic
(the expensive part) can be batched.

### 1.3 `tf.constant()` inside loops (lines 304-309)

```python
d_lambda_tf = tf.constant(d_lambda, dtype=tf.float32)
lambda_val_tf = tf.constant(lambda_val, dtype=tf.float32)
eta_bar_tf = tf.constant(eta_bar, dtype=tf.float32)
eta_1_tf = tf.constant(eta_1, dtype=tf.float32)
```

Each `tf.constant()` creates a new tensor node on every iteration.
The `eta_bar` and `eta_1` conversions are especially costly for
`(500, state_dim)` tensors, 29 times per observation.

**Suggestion:** Use `tf.Variable` and `.assign()` instead, or keep
everything as tensors throughout the loop (avoid numpy round-trips).

### 1.4 Resampling index matching (lines 370-374)

```python
for i in range(self.n_particles):
    dists = np.sum((particles_np - resampled_particles_np[i])**2, axis=1)
    idx = np.argmin(dists)
```

This is O(N^2) nearest-neighbor search in Python.
With N=500, that's 250,000 distance computations per resample event.

**Suggestion:** Use `cdist` from scipy or vectorized numpy broadcasting:
`dists = np.sum((particles_np[None,:,:] - resampled_particles_np[:,None,:])**2, axis=-1)`
then `indices = np.argmin(dists, axis=1)`. Single vectorized call.

---

## 2. LEDH Flow (`ledh_flow.py`) — MODERATE

### 2.1 Per-particle drift via `tf.map_fn` (line 267)

```python
drift = tf.map_fn(compute_drift_for_particle, particles, dtype=tf.float32)
```

Each particle's drift requires:
- `compute_flow_params()` → Jacobian + 4 matrix ops + Cholesky solve
- Total: 29 lambda steps x N map_fn calls = 14,500 sequential ops

Same core problem as LEDH invertible: per-particle Jacobian computation
inside a Python loop.

**Suggestion:** Same as 1.1 — add batched `observation_jacobian` and
vectorize `compute_flow_params` to process all particles at once.

### 2.2 Drift norm capping (lines 223-228)

```python
drift_norm = tf.norm(particle_drift)
scale = tf.minimum(1.0, max_drift_norm / (drift_norm + 1e-10))
particle_drift = particle_drift * scale
```

Per-particle norm + conditional scaling adds overhead inside `tf.map_fn`.
If vectorized, this becomes a single batched norm operation.

---

## 3. EDH Invertible (`edh_invertible.py`) — MODERATE

### 3.1 Flow loop is already efficient for EDH

The EDH flow loop (lines 264-289) computes `A, b` once per lambda step
(at the ensemble mean), then applies the same drift to all particles via
batched `_compute_drift_batch`. This is the right design:
**29 calls** to `compute_flow_params`, not 29 x 500.

No optimization needed here.

### 3.2 `_generate_step_sizes` called every update (line 262)

```python
step_sizes = self._generate_step_sizes()
```

This creates a new tensor of step sizes on every `update()` call.
The step sizes are constant.

**Suggestion:** Precompute once in `__init__` and store as `self.step_sizes`.

---

## 4. EDH Flow (`edh_flow.py`) — MODERATE

### 4.1 `compute_flow_params` called every lambda step (line 293)

Same structure as EDH invertible — one `compute_flow_params` call per
lambda step at the ensemble mean. 29 calls per observation. Efficient.

### 4.2 `tf.constant()` conversions inside loop (line 262-263)

```python
lambda_val_tf = tf.constant(lambda_val, dtype=tf.float32)
```

Minor: Python float → TF constant on every iteration.

**Suggestion:** Pre-compute cumulative lambda schedule as a TF tensor.

---

## 5. Bootstrap PF (`bootstrap_pf_tf.py`) — GOOD

### 5.1 Already well-vectorized

The entire `filter_tf` method is a single `@tf.function` (line 128).
Uses batched model methods (`state_transition_batch`,
`log_observation_prob_batch`). No per-particle Python loops.

**This is the performance target** that other filters should aspire to.

---

## 6. Stochastic EDH (`stochastic_edh.py`) — MODERATE

### 6.1 Same structure as EDH flow

Uses `_compute_drift` (batched) and `euler_step` per lambda step.
Score correction is precomputed before the loop.
29 calls per observation. Reasonable.

### 6.2 Optimal schedule solver (lines 86-158)

When `schedule_mu > 0`, the BVP shooting method runs:
- Up to 6 forward integrations for bracket widening
- 40 bisection iterations x 500 Euler steps each = 20,000 steps

This is a one-time-per-observation cost but can be significant.

**Suggestion:** Cache the optimal schedule if the model is time-invariant
(P and H don't change much between observations). Recompute only when
`||P_new - P_cached|| > tol`.

---

## 7. Kernel Flow PF (`kernel_flow.py`) — MODERATE

### 7.1 Per-particle Python loops

`predict()` and `initialize()` use Python `for i in range(n_particles)`
loops with individual `sample_state_transition` calls.

**Suggestion:** Use `state_transition_batch` like bootstrap PF does.

### 7.2 Scipy `linalg.solve` in numpy

The update step uses scipy/numpy linear algebra, not TF.
This is acceptable for small particle counts (N=20 typical for kernel PF)
but doesn't leverage GPU if available.

---

## 8. `flow_params.py` (`compute_flow_params`) — MODERATE

### 8.1 Single-particle interface

```python
def compute_flow_params(model, linearization_point, lambda_val, ...)
```

Processes one linearization point at a time. When called inside
LEDH's per-particle loop, this means N separate calls.

**Suggestion:** Add a `compute_flow_params_batch` that takes:
- `linearization_points: (N, state_dim)`
- `P_batch: (N, state_dim, state_dim)`
- `eta_bar_0_batch: (N, state_dim)`

And returns `A_batch: (N, state_dim, state_dim)`, `b_batch: (N, state_dim)`.

The math is identical but uses batched TF ops:
- `H_batch = model.observation_jacobian_batch(points)` → `(N, obs_dim, state_dim)`
- `HPH_batch = H_batch @ P_batch @ transpose(H_batch)` → batched matmul
- `S_batch = lambda * HPH_batch + R` → broadcast R
- `A_batch = -0.5 * P_batch @ transpose(H_batch) @ solve(S_batch, H_batch)`
- etc.

---

## 9. `ode_solvers.py` — GOOD

### 9.1 Already supports batched inputs

`euler_step` works with `(N, state_dim)` tensors natively.
No issues here.

---

## 10. Cross-cutting Issues

### 10.1 Repeated TF/numpy boundary crossings

Several filters repeatedly convert between TF tensors and numpy arrays
within inner loops:

| File | Operation | Frequency |
|------|-----------|-----------|
| `ledh_invertible.py` | `tf.constant(eta_bar)` | 29x per obs |
| `ledh_invertible.py` | `self.particle_covs` numpy array indexed in TF loop | 29x500 per obs |
| `edh_invertible.py` | `epsilon_j.numpy()` | 29x per obs |
| `edh_flow.py` | `tf.constant(lambda_val)` | 29x per obs |

Each TF↔numpy crossing has overhead (~10-100 microseconds). At 14,500
crossings per observation and 20 observations, this adds up.

**Suggestion:** Keep all loop variables as TF tensors. Convert to numpy
only at the end of `update()` for storage.

### 10.2 `tf.Variable` allocation

Several filters create new `tf.Variable` objects every `predict()`/`update()`:

```python
eta_1 = tf.Variable(self.eta_0.value(), dtype=tf.float32)    # ledh_invertible
eta_bar = tf.Variable(self.eta_bar_0.value(), dtype=tf.float32)
self.particles_prev = tf.Variable(self.particles.value(), dtype=tf.float32)
```

`tf.Variable` allocation involves GPU memory management overhead.

**Suggestion:** Pre-allocate `tf.Variable`s in `__init__` or `initialize()`,
then use `.assign()` in the loop. Or use plain tensors (not Variables)
when mutation via `.assign()` isn't needed.

### 10.3 Missing `@tf.function` on hot paths

The `update()` methods of `ledh_invertible`, `edh_invertible`, `edh_flow`,
and `ledh_flow` are **not** decorated with `@tf.function`. This means
every TF op is executed eagerly with Python dispatch overhead.

The bootstrap PF wraps its entire loop in `@tf.function` (line 128),
which is why it's fast.

**Suggestion:** Wrapping the entire `update()` in `@tf.function` would
eliminate Python dispatch overhead. However, this requires all operations
inside to be TF-compatible (no numpy, no Python side effects like
`list.append`). The flow loop is the best candidate for wrapping in a
separate `@tf.function`.

---

## Summary: Priority Ranking

| Priority | File | Issue | Impact | Effort |
|----------|------|-------|--------|--------|
| **P0** | `ledh_invertible.py` | Nested loop: 29 x 500 sequential `compute_flow_params` | ~100x slowdown vs batched | High |
| **P1** | `ledh_invertible.py` | `tf.map_fn` with `parallel_iterations=1` | Sequential execution | Medium |
| **P1** | `batched_ekf.py` | `tf.map_fn` instead of batched linalg | ~5-10x slower than batched | Medium |
| **P2** | `ledh_flow.py` | `tf.map_fn` per-particle drift | Same as P0 for LEDH flow | High |
| **P2** | `flow_params.py` | Single-particle interface only | Blocks P0 fix | Medium |
| **P3** | All flow filters | `tf.constant()` inside loops | ~1-5% overhead | Low |
| **P3** | All flow filters | TF/numpy boundary crossings | ~1-5% overhead | Low |
| **P3** | `edh_invertible.py` | `_generate_step_sizes` per update | Minor | Trivial |
| **P4** | `ledh_invertible.py` | O(N^2) resampling index match | Occasional (only on resample) | Low |
| **P4** | `kernel_flow.py` | Python loops for predict/initialize | Low N (20 typical) | Low |

### Quick wins (no workflow change):
1. Set `parallel_iterations=10` (or higher) in `tf.map_fn` calls — allows TF to parallelize
2. Precompute `step_sizes` tensor in `__init__` for `edh_invertible`
3. Pre-allocate `tf.Variable`s instead of creating new ones per timestep
4. Remove `tf.constant()` wrappers inside loops — keep tensors as tensors

### Medium effort (no workflow change):
1. Replace `tf.map_fn` in `batched_ekf.py` with true batched matmul
2. Vectorize resampling index matching with numpy broadcasting

### High effort (adds new model interface):
1. Add `observation_jacobian_batch(particles)` to model base class
2. Add `compute_flow_params_batch()` to `flow_params.py`
3. Vectorize the inner particle loop in LEDH invertible/flow update
