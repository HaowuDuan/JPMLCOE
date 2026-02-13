# Refactoring Guide: Performance Optimization

Step-by-step instructions organized from easiest to hardest.
Each step is independent — you can apply them in any order.

Items marked with **[UNCERTAIN]** may not work as described and need testing.

---

## Tier 1: Quick Wins (no logic change, no new interfaces)

These changes are purely mechanical and cannot break correctness.

---

### 1.1 Bump `parallel_iterations` in `tf.map_fn`

**File:** `src/filters/particle/ledh_invertible.py`, line 326

**Why it helps:** `parallel_iterations=1` forces TF to process particles
one after another. Increasing it allows TF to run multiple particle updates
concurrently (overlapping GPU kernel launches or CPU threads). Since each
particle's computation is independent within a lambda step, this is safe.

**Before:**
```python
results = tf.map_fn(
    process_particle,
    tf.range(self.n_particles),
    fn_output_signature=(...),
    parallel_iterations=1  # Sequential execution to match original behavior
)
```

**After:**
```python
results = tf.map_fn(
    process_particle,
    tf.range(self.n_particles),
    fn_output_signature=(...),
    parallel_iterations=10
)
```

**[UNCERTAIN]** The speedup depends on hardware. On CPU, TF may not actually
parallelize these because the GIL and op scheduling. On GPU, it can overlap
kernel launches. Try values 10, 50, and `self.n_particles` and benchmark.
The particle updates are mathematically independent per lambda step, so
correctness is not affected — but numerical results may differ slightly due
to floating-point non-associativity.

---

### 1.2 Precompute step sizes in `edh_invertible.py`

**File:** `src/filters/particle/edh_invertible.py`

**Why it helps:** `_generate_step_sizes()` is called on every `update()`
(line 262) but returns the same values every time. Moving it to `__init__`
saves one tensor allocation per observation.

**Step 1:** Add to `__init__`, after line 134 (after `self.debug_info` block):
```python
self.step_sizes = self._generate_step_sizes()
```

**Step 2:** In `update()`, line 262, change:
```python
# Before
step_sizes = self._generate_step_sizes()

# After
step_sizes = self.step_sizes
```

---

### 1.3 Pre-convert `lambda_steps` to TF tensor in `ledh_invertible.py`

**File:** `src/filters/particle/ledh_invertible.py`

**Why it helps:** `self.lambda_steps` is a numpy array. Inside the flow loop,
`d_lambda = self.lambda_steps[j]` returns a Python float, which then gets
wrapped in `tf.constant(d_lambda)` on line 304. Converting once in `__init__`
avoids 29 `tf.constant()` calls per observation.

**Step 1:** In `_generate_lambda_steps()` (line 131-136), change:
```python
# Before
lambda_steps_np = epsilon_1 * q**np.arange(self.n_lambda_steps)
self.lambda_steps = lambda_steps_np

# After
lambda_steps_np = epsilon_1 * q**np.arange(self.n_lambda_steps)
self.lambda_steps = lambda_steps_np
self.lambda_steps_tf = tf.constant(lambda_steps_np, dtype=tf.float32)
# Also precompute cumulative sum for lambda_val
self.lambda_cumsum_tf = tf.cumsum(self.lambda_steps_tf)
```

**Step 2:** In `update()`, replace lines 300-305:
```python
# Before
for j in range(self.n_lambda_steps):
    d_lambda = self.lambda_steps[j]
    lambda_val += d_lambda
    d_lambda_tf = tf.constant(d_lambda, dtype=tf.float32)
    lambda_val_tf = tf.constant(lambda_val, dtype=tf.float32)

# After
for j in range(self.n_lambda_steps):
    d_lambda_tf = self.lambda_steps_tf[j]
    lambda_val_tf = self.lambda_cumsum_tf[j]
```

This eliminates 29 x 2 = 58 `tf.constant()` calls per observation.

---

### 1.4 Remove redundant `tf.constant()` wrapping on tensors already in TF

**File:** `src/filters/particle/ledh_invertible.py`, lines 307-309

**Why it helps:** After each lambda step, `eta_bar` and `eta_1` are already
TF tensors (returned by `tf.map_fn`). Wrapping them in `tf.constant()` copies
the data from TF → numpy → TF, which is wasteful.

**Before:**
```python
# Convert current states to tensors for map_fn
eta_bar_tf = tf.constant(eta_bar, dtype=tf.float32)
eta_1_tf = tf.constant(eta_1, dtype=tf.float32)
```

**After:** Check if `eta_bar` is already a tensor. On the first iteration,
`eta_bar` is set from `self.eta_bar_0.value()` (line 286) which IS a tensor.
After each iteration, it's `results[0]` (line 329) which is also a tensor.
So these `tf.constant()` calls are redundant.

```python
# Just use them directly — they're already tensors
eta_bar_tf = eta_bar
eta_1_tf = eta_1
```

Then update the `process_particle` closure and `_update_single_particle`
to use `eta_bar_tf` and `eta_1_tf` (which are the same object now).

Similarly line 297:
```python
# Before
eta_bar_0_tf = tf.constant(self.eta_bar_0.value(), dtype=tf.float32)

# After — .value() already returns a tensor
eta_bar_0_tf = self.eta_bar_0.value()
```

**Why this works:** `tf.constant()` on an existing tensor silently copies
data through numpy. Since the data is already float32, the copy is pure waste.

---

### 1.5 Avoid `tf.Variable` allocation per timestep

**File:** `src/filters/particle/ledh_invertible.py`, lines 212-213, 232, 237-238

**Why it helps:** `tf.Variable(...)` allocates GPU/CPU memory and registers
the variable. When called every `predict()` and `update()` (T times), this
is T unnecessary allocations.

**Step 1:** In `initialize()`, add after line 183:
```python
# Pre-allocate Variables (reuse across timesteps)
self.particles_prev = tf.Variable(tf.zeros_like(particles_tf), dtype=tf.float32)
self.eta_bar_0 = tf.Variable(tf.zeros([self.n_particles, self.state_dim], dtype=tf.float32))
self.eta_0 = tf.Variable(tf.zeros([self.n_particles, self.state_dim], dtype=tf.float32))
```

**Step 2:** In `predict()`, change:
```python
# Before (line 213)
self.particles_prev = tf.Variable(self.particles.value(), dtype=tf.float32)
# ...
# Before (line 232)
self.eta_bar_0 = tf.Variable(eta_bar_0_tf, dtype=tf.float32)
# ...
# Before (line 238)
self.eta_0 = tf.Variable(eta_0_tf, dtype=tf.float32)

# After
self.particles_prev.assign(self.particles.value())
# ...
self.eta_bar_0.assign(eta_bar_0_tf)
# ...
self.eta_0.assign(eta_0_tf)
```

Apply the same pattern in `edh_invertible.py` lines 220, 242, 256.

---

### 1.6 Same quick wins for `edh_invertible.py`

Apply the same changes from 1.3-1.5 to `edh_invertible.py`:

- Precompute `self.step_sizes` in `__init__` (see 1.2)
- In `update()` line 276, `tf.constant(lambda_val)` is inside a `tf.range`
  loop — use precomputed cumulative lambda tensor instead
- Line 256: `tf.Variable(self.eta_bar_0)` → pre-allocate, use `.assign()`
- Line 220: `tf.Variable(self.particles.value())` → pre-allocate, use `.assign()`

---

## Tier 2: Medium Effort (improve batched_ekf, vectorize resampling)

These change internal implementations but not the external interface.

---

### 2.1 Replace `tf.map_fn` in `batched_ekf.py` with true batched ops

**File:** `src/filters/kalman/batched_ekf.py`

**Why it helps:** `tf.map_fn` processes particles sequentially (one EKF
predict/update per loop iteration). For models where F, Q, H, R are
state-independent (constant), the matrix arithmetic can be vectorized
into single batched `tf.matmul` calls that process all N particles at once.

For the linear model: F, Q, H, R are stored as class attributes and don't
depend on `x`. So `state_jacobian(x)` always returns `self.F`, etc.

**Replace `batched_ekf_predict` with:**

```python
@tf.function
def batched_ekf_predict(model, means, covs):
    """
    Batched EKF prediction for N particles.

    Args:
        means: (N, state_dim)
        covs:  (N, state_dim, state_dim)
    Returns:
        mean_pred: (N, state_dim)
        cov_pred:  (N, state_dim, state_dim)
    """
    # Get F and Q from first particle (they're state-independent)
    F = model.state_jacobian(means[0])        # (d, d)
    Q = model.state_transition_cov(means[0])  # (d, d)

    # Batched mean prediction: mean_pred_i = F @ mean_i for all i
    # means: (N, d), F: (d, d) → mean_pred: (N, d)
    mean_pred = tf.linalg.matvec(F, means)    # broadcasts F over batch dim

    # Batched covariance prediction: cov_pred_i = F @ cov_i @ F^T + Q
    # covs: (N, d, d), F: (d, d)
    # F @ covs: (N, d, d) — use tf.matmul with broadcast
    F_covs = tf.matmul(F, covs)               # (d,d) @ (N,d,d) → broadcasts to (N,d,d)
    cov_pred = tf.matmul(F_covs, F, transpose_b=True) + Q

    # Symmetrize
    cov_pred = 0.5 * (cov_pred + tf.linalg.matrix_transpose(cov_pred))

    return mean_pred, cov_pred
```

**[UNCERTAIN]** This only works when F and Q are state-independent. For
nonlinear models (range_bearing, two_sensor_bearing), `state_jacobian(x)`
returns different F for each x. You need a fallback:

```python
# Check if model has constant Jacobian
F_0 = model.state_jacobian(means[0])
F_1 = model.state_jacobian(means[1])
is_constant = tf.reduce_all(tf.equal(F_0, F_1))
```

Or simply: check `isinstance(model, LinearGaussianModel)` outside the
`@tf.function` and dispatch to the batched vs map_fn path.

**Replace `batched_ekf_update` similarly:**

```python
@tf.function
def batched_ekf_update(model, means, covs, observation):
    H = model.observation_jacobian(means[0])  # (obs_dim, state_dim)
    R = model.observation_cov(means[0])       # (obs_dim, obs_dim)

    # y_pred: (N, obs_dim)
    y_pred = tf.linalg.matvec(H, means)
    innovation = observation - y_pred          # (N, obs_dim) via broadcast

    # S_i = H @ cov_i @ H^T + R
    H_covs = tf.matmul(H, covs)               # (obs,state) @ (N,state,state) → (N,obs,state)
    S = tf.matmul(H_covs, H, transpose_b=True) + R  # (N,obs,obs)

    # K_i = cov_i @ H^T @ S_i^{-1}
    covs_HT = tf.matmul(covs, H, transpose_b=True)   # (N, state, obs)
    S_inv = tf.linalg.inv(S)                           # (N, obs, obs)
    K = tf.matmul(covs_HT, S_inv)                     # (N, state, obs)

    # mean_updated_i = mean_i + K_i @ innovation_i
    # innovation: (N, obs) → expand to (N, obs, 1)
    innovation_col = tf.expand_dims(innovation, -1)    # (N, obs, 1)
    mean_update = tf.squeeze(tf.matmul(K, innovation_col), axis=-1)  # (N, state)
    mean_updated = means + mean_update

    # cov_updated_i = (I - K_i @ H) @ cov_i
    I = tf.eye(model.state_dim, dtype=tf.float32)
    KH = tf.matmul(K, tf.broadcast_to(H, [tf.shape(K)[0], H.shape[0], H.shape[1]]))
    cov_updated = tf.matmul(I - KH, covs)
    cov_updated = 0.5 * (cov_updated + tf.linalg.matrix_transpose(cov_updated))

    return mean_updated, cov_updated
```

**[UNCERTAIN]** The `tf.matmul` broadcasting rules for `(d,d) @ (N,d,d)` need
verification. TF docs say: "The inputs must, following any transpositions, be
tensors of rank >= 2 where the inner 2 dimensions specify valid matrix
multiplication dimensions, and any further outer dimensions specify matching
batch size." So `(d,d) @ (N,d,d)` should broadcast to `(N,d,d)`. **Test
this with a small example first.** If TF doesn't broadcast `(d,d)` to
`(N,d,d)`, you'll need `tf.broadcast_to(F, [N, d, d])` explicitly.

---

### 2.2 Vectorize resampling index matching

**File:** `src/filters/particle/ledh_invertible.py`, lines 370-374

**Why it helps:** The current code is O(N^2) with a Python loop.
Vectorized numpy computes all pairwise distances in one call.

**Before:**
```python
indices = []
for i in range(self.n_particles):
    dists = np.sum((particles_np - resampled_particles_np[i])**2, axis=1)
    idx = np.argmin(dists)
    indices.append(idx)
```

**After:**
```python
# Vectorized: (N, 1, state_dim) - (1, N, state_dim) → (N, N) distances
dists = np.sum(
    (resampled_particles_np[:, None, :] - particles_np[None, :, :]) ** 2,
    axis=-1
)
indices = np.argmin(dists, axis=1).tolist()
```

This replaces N Python iterations with one numpy call.
For N=500 and state_dim=4, this is ~500x faster.

Same change applies to `edh_invertible.py` if it has similar code (it doesn't —
EDH invertible uses direct index-based resampling, so only LEDH needs this).

---

## Tier 3: High Impact (vectorize the flow loop)

This is the change that will make LEDH invertible competitive with EDH.

---

### 3.1 Add `observation_jacobian_batch` to model base class

**File:** `src/core/model_base.py`

**Why it's needed:** The flow loop calls `model.observation_jacobian(x)` per
particle. To vectorize, we need a version that takes `(N, state_dim)` and
returns `(N, obs_dim, state_dim)`.

**Add to `StateSpaceModel` after line 144:**

```python
def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
    """Compute observation Jacobians for batch. Default: tf.map_fn."""
    return tf.map_fn(self.observation_jacobian, particles,
                     fn_output_signature=tf.TensorSpec(
                         [self.obs_dim, self.state_dim], tf.float32))

def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
    """Compute h(x) for batch. Default: tf.map_fn."""
    return tf.map_fn(self.observation_function, particles,
                     fn_output_signature=tf.TensorSpec(
                         [self.obs_dim], tf.float32))
```

**Then override in each model for efficiency.**

**For `linear_gaussian.py`** — trivial since H is constant:
```python
def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
    """H is constant — broadcast to (N, obs_dim, state_dim)."""
    N = tf.shape(particles)[0]
    return tf.broadcast_to(self.H, [N, self.ny, self.nx])

def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
    """h(x) = H @ x for all particles."""
    return particles @ tf.transpose(self.H)  # (N, obs_dim)
```

**For `range_bearing.py`** — Jacobian depends on x:
```python
def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
    """Vectorized Jacobian for all particles."""
    dx = particles[:, 0] - self.sensor_pos[0]  # (N,)
    dy = particles[:, 1] - self.sensor_pos[1]  # (N,)
    range_val = tf.maximum(tf.sqrt(dx**2 + dy**2), 1e-10)  # (N,)

    # Build (N, 2, 2) Jacobian batch
    H00 = dx / range_val
    H01 = dy / range_val
    H10 = -dy / range_val**2
    H11 = dx / range_val**2

    row0 = tf.stack([H00, H01], axis=-1)  # (N, 2)
    row1 = tf.stack([H10, H11], axis=-1)  # (N, 2)
    return tf.stack([row0, row1], axis=-2) # (N, 2, 2)

def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
    dx = particles[:, 0] - self.sensor_pos[0]
    dy = particles[:, 1] - self.sensor_pos[1]
    range_val = tf.sqrt(dx**2 + dy**2)
    bearing = tf.atan2(dy, dx)
    return tf.stack([range_val, bearing], axis=-1)
```

**For `two_sensor_bearing.py`** — same pattern, two bearings.

**[UNCERTAIN]** For `acoustic_tracking_full.py`, the Jacobian has more complex
structure. You'll need to read that model's `observation_jacobian` method
and vectorize it. The default `tf.map_fn` fallback in the base class will
still work, just slower.

---

### 3.2 Add `compute_flow_params_batch` to `flow_params.py`

**File:** `src/utils/flow_params.py`

**Why it helps:** This is the core of the LEDH inner loop. Currently called
once per particle per lambda step (500 x 29 = 14,500 times per observation).
A batched version processes all 500 particles in one call per lambda step
(29 calls total).

**Add after the existing `compute_flow_params` function:**

```python
@tf.function
def compute_flow_params_batch(
    model,
    linearization_points: tf.Tensor,   # (N, state_dim)
    lambda_val: tf.Tensor,             # scalar
    observation: tf.Tensor,            # (obs_dim,)
    P_batch: tf.Tensor,                # (N, state_dim, state_dim)
    R: tf.Tensor,                      # (obs_dim, obs_dim)
    R_inv: tf.Tensor,                  # (obs_dim, obs_dim)
    eta_bar_0_batch: tf.Tensor,        # (N, state_dim)
    state_dim: int,
    regularization: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched version of compute_flow_params.

    Returns:
        A_batch: (N, state_dim, state_dim)
        b_batch: (N, state_dim)
    """
    N = tf.shape(linearization_points)[0]

    # Step 1: Batch Jacobians — (N, obs_dim, state_dim)
    H_batch = model.observation_jacobian_batch(linearization_points)

    # Step 2: Batch h(x) — (N, obs_dim)
    h_batch = model.observation_function_batch(linearization_points)

    # Step 3: Regularize P if needed
    if regularization > 0.0:
        trace_P = tf.linalg.trace(P_batch)           # (N,)
        state_dim_f = tf.cast(state_dim, P_batch.dtype)
        reg_strength = regularization * (trace_P / state_dim_f)  # (N,)
        eye = tf.eye(state_dim, dtype=P_batch.dtype)
        P_batch = P_batch + reg_strength[:, None, None] * eye

    # Step 4: HPH^T — batched matrix multiply
    # H_batch: (N, obs, state), P_batch: (N, state, state)
    HP = tf.matmul(H_batch, P_batch)                          # (N, obs, state)
    HPH = tf.matmul(HP, H_batch, transpose_b=True)            # (N, obs, obs)

    # Step 5: S = lambda * HPH + R
    S_batch = lambda_val * HPH + R                             # (N, obs, obs) broadcast R

    # Step 6: Solve S @ X = H → S_inv_H, shape (N, obs, state)
    # Using Cholesky for stability
    L_S = safe_cholesky_batch(S_batch)                         # (N, obs, obs)
    S_inv_H = tf.linalg.cholesky_solve(L_S, H_batch)          # (N, obs, state)

    # Step 7: A = -0.5 * P @ H^T @ S_inv_H
    # P @ H^T: (N, state, state) @ (N, state, obs) = (N, state, obs)
    PHT = tf.matmul(P_batch, H_batch, transpose_b=True)       # (N, state, obs)
    A_batch = -0.5 * tf.matmul(PHT, S_inv_H)                  # (N, state, state)

    # Step 8: e = h(x) - H @ x
    # H_batch @ linearization_points: (N, obs, state) @ (N, state) → need einsum
    Hx = tf.einsum('nij,nj->ni', H_batch, linearization_points)  # (N, obs)
    e_batch = h_batch - Hx                                     # (N, obs)

    # Step 9: b = (I + 2*lambda*A) @ [(I + lambda*A) @ P @ H^T @ R_inv @ (z - e) + A @ eta_bar_0]
    I = tf.eye(state_dim, dtype=tf.float32)                    # (state, state)

    I_plus_lA = I + lambda_val * A_batch                       # (N, state, state)
    I_plus_2lA = I + 2 * lambda_val * A_batch                  # (N, state, state)

    # P @ H^T @ R_inv: (N, state, obs) @ (obs, obs) = (N, state, obs)
    PHT_Rinv = tf.matmul(PHT, R_inv)                           # (N, state, obs), R_inv broadcasts

    # (I + lambda*A) @ P @ H^T @ R_inv: (N, state, state) @ (N, state, obs) = (N, state, obs)
    term1_mat = tf.matmul(I_plus_lA, PHT_Rinv)                # (N, state, obs)

    # (z - e): observation is (obs,), e is (N, obs) → (N, obs)
    z_minus_e = observation - e_batch                           # (N, obs)

    # term1_mat @ (z - e): (N, state, obs) @ (N, obs) → use einsum
    term1 = tf.einsum('nij,nj->ni', term1_mat, z_minus_e)     # (N, state)

    # A @ eta_bar_0: (N, state, state) @ (N, state) → einsum
    term2 = tf.einsum('nij,nj->ni', A_batch, eta_bar_0_batch) # (N, state)

    # (I + 2*lambda*A) @ (term1 + term2)
    combined = term1 + term2                                    # (N, state)
    b_batch = tf.einsum('nij,nj->ni', I_plus_2lA, combined)   # (N, state)

    return A_batch, b_batch
```

**You also need `safe_cholesky_batch`** — add to `linalg.py`:

```python
@tf.function
def safe_cholesky_batch(A: tf.Tensor, jitter: float = 1e-10) -> tf.Tensor:
    """Cholesky for batched matrices (N, n, n)."""
    n = tf.shape(A)[-1]
    eye = tf.eye(n, dtype=A.dtype)
    trace_A = tf.linalg.trace(A)               # (N,)
    n_float = tf.cast(n, A.dtype)
    avg_diag = trace_A / n_float               # (N,)
    scaled_jitter = jitter * tf.maximum(avg_diag, 1.0)  # (N,)
    A_reg = A + scaled_jitter[:, None, None] * eye
    return tf.linalg.cholesky(A_reg)
```

**Why this is correct:** The math is identical to the single-particle version.
Every operation is the same formula, just with a leading batch dimension N.
`tf.matmul` and `tf.einsum` handle the batch dimension natively.

**[UNCERTAIN]** The `safe_cholesky_batch` may need more robust error handling.
If any single S_i is ill-conditioned, the whole batch Cholesky fails. You
might need to add a larger jitter or use `tf.linalg.solve` instead of
`cholesky_solve` as a fallback. Test on the range_bearing model first,
as its nonlinear observation can produce ill-conditioned S matrices.

---

### 3.3 Rewrite LEDH invertible `update()` to use batched flow params

**File:** `src/filters/particle/ledh_invertible.py`

**Why it helps:** This is THE change that eliminates the 14,500-iteration
bottleneck. Instead of `29 * 500` sequential calls, you get `29` batched
calls. Each batched call processes all 500 particles in parallel using
TF's vectorized linear algebra.

**Replace the flow loop in `update()` (lines 299-331) with:**

```python
from ...utils.flow_params import compute_flow_params_batch

# ... (keep lines 280-298 unchanged) ...

# Flow loop — BATCHED
for j in range(self.n_lambda_steps):
    d_lambda_tf = self.lambda_steps_tf[j]
    lambda_val_tf = self.lambda_cumsum_tf[j]

    # Batched flow params: all particles at once
    A_batch, b_batch = compute_flow_params_batch(
        self.model,
        eta_bar,                   # (N, state_dim)
        lambda_val_tf,             # scalar
        y_tf,                      # (obs_dim,)
        particle_covs_tf,          # (N, state_dim, state_dim)
        R,                         # (obs_dim, obs_dim)
        R_inv,                     # (obs_dim, obs_dim)
        eta_bar_0_tf,              # (N, state_dim)
        self.state_dim,
        regularization_tf
    )

    # Batched Euler step: eta = eta + dt * (A @ eta + b)
    # A_batch: (N, d, d), eta_bar: (N, d) → drift: (N, d)
    drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
    eta_bar = eta_bar + d_lambda_tf * drift_bar

    drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
    eta_1 = eta_1 + d_lambda_tf * drift_1

    # Batched Jacobian determinant: det(I + dt * A_i) for all i
    I = tf.eye(self.state_dim, dtype=tf.float32)
    M_batch = I + d_lambda_tf * A_batch             # (N, d, d)
    log_det_batch = tf.math.log(tf.abs(tf.linalg.det(M_batch)))  # (N,)
    log_theta = log_theta + log_det_batch
```

**What changed:**
1. `compute_flow_params` (per-particle) → `compute_flow_params_batch` (all N)
2. `euler_step` (per-particle) → `tf.einsum` (all N, one op)
3. `tensor_scatter_nd_update` (per-particle) → direct tensor assignment
4. `tf.linalg.det` (per-particle) → batched `tf.linalg.det` on `(N,d,d)`

**Remove** the `_update_single_particle` method and the `tf.map_fn` call
entirely — they are no longer needed.

**[UNCERTAIN]** `tf.linalg.det` on `(N, d, d)` batched tensors is supported
by TF. However, for very large state dimensions, `tf.linalg.det` may be
numerically unstable. An alternative is `tf.linalg.slogdet` which returns
`(sign, logdet)` — you already need log-det, so `slogdet` is more natural:
```python
_, log_det_batch = tf.linalg.slogdet(M_batch)
```
This is more numerically stable than `log(abs(det(...)))`.

---

### 3.4 Same change for `ledh_flow.py`

**File:** `src/filters/particle/ledh_flow.py`

The `_flow_step_euler` method (line 230) uses `tf.map_fn` over particles.
Apply the same batched pattern:

**Before (line 261-267):**
```python
def compute_drift_for_particle(particle):
    return self._compute_drift_single(
        particle, lambda_val, y, P, R, R_inv, eta_bar_0, regularization_tf
    )
drift = tf.map_fn(compute_drift_for_particle, particles, dtype=tf.float32)
```

**After:**
```python
# Use batched flow params
A_batch, b_batch = compute_flow_params_batch(
    self.model,
    particles,              # (N, state_dim) — each particle is its own linearization
    lambda_val, y,
    # P is GLOBAL (same for all particles) — tile to (N, d, d)
    tf.broadcast_to(P, [tf.shape(particles)[0], self.state_dim, self.state_dim]),
    R, R_inv,
    tf.broadcast_to(eta_bar_0, [tf.shape(particles)[0], self.state_dim]),
    self.state_dim,
    regularization_tf
)
drift = tf.einsum('nij,nj->ni', A_batch, particles) + b_batch
```

**Key difference from LEDH invertible:** In LEDH flow, `P` is the
GLOBAL covariance (same for all particles), so you broadcast it.
In LEDH invertible, `P_i` is per-particle, so it's already `(N, d, d)`.

**[UNCERTAIN]** The `tf.broadcast_to` for P and eta_bar_0 allocates
memory for N copies. For large state_dim and large N, this could be
wasteful. An alternative is to modify `compute_flow_params_batch` to
accept either `(N, d, d)` or `(d, d)` for P, and broadcast internally.
But start with the simple approach and optimize later if needed.

---

## Tier 4: Optional / Minor

---

### 4.1 Use `tf.linalg.slogdet` instead of `log(abs(det(...)))` everywhere

**Files:** `ledh_invertible.py` line 276, `edh_invertible.py` if applicable

**Before:**
```python
log_det_M_i = tf.math.log(tf.abs(tf.linalg.det(M_i)))
```

**After:**
```python
_, log_det_M_i = tf.linalg.slogdet(M_i)
```

**Why:** `slogdet` is numerically stable for near-singular matrices.
`det` can overflow/underflow for large matrices, then `log(abs(0))` = -inf.

---

### 4.2 Cache `R_inv` across timesteps in LEDH invertible

**File:** `src/filters/particle/ledh_invertible.py`, line 292

**Why it helps:** `R_inv = tf.linalg.inv(R)` is computed on every `update()`.
If R is constant (typical), this is redundant.

**Before:**
```python
R_inv = tf.linalg.inv(R)
```

**After:** Add to `__init__` or `initialize()`:
```python
self._R_inv_cache = None
```

In `update()`:
```python
if self._R_inv_cache is None:
    self._R_inv_cache = tf.linalg.inv(R)
R_inv = self._R_inv_cache
```

EDH invertible already does this (line 250-253). Align LEDH to match.

---

## Order of Implementation

Recommended order to maximize value while minimizing risk:

1. **Tier 1 (1.1-1.6)** — Apply all quick wins first. Run experiments to
   confirm nothing breaks. Benchmark.

2. **Tier 2.2** — Vectorize resampling. Simple and isolated.

3. **Tier 2.1** — Batched EKF. Test on 1d_linear first (where F/H/Q/R
   are constant). Then test on range_bearing.

4. **Tier 3.1** — Add batch Jacobian methods to models. Start with
   linear_gaussian (trivial), then range_bearing. Test each model
   independently by comparing output of `observation_jacobian_batch(X)`
   vs `tf.stack([observation_jacobian(X[i]) for i])`.

5. **Tier 3.2** — Add `compute_flow_params_batch`. Test by comparing
   output against N calls to `compute_flow_params` with same inputs.

6. **Tier 3.3** — Rewrite LEDH invertible update. This is the payoff.
   Run 1d_linear_ledh_invertible experiment and compare RMSE + timing
   against the old version.

7. **Tier 3.4** — Same for LEDH flow.

At each step, compare numerical output (means, covs, RMSE) against the
un-optimized version to confirm correctness before proceeding.
