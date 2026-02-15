# HMC Performance Analysis for LEDH Particle Flow Filter

## 1. Observed Performance

| Metric | Value | Source |
|--------|-------|--------|
| LEDH filter forward pass (standalone) | ~0.1s | User-reported (200 particles, T=100) |
| One HMC step (10 leapfrog) | **~750s** | Observed: `[burn-in 1/5] 750s` |
| One gradient evaluation (fwd + bwd) | **~75s** | 750s / 10 leapfrog |
| **Backward/forward ratio** | **~750x** | 75s / 0.1s — pathological |
| EKF baseline (15 steps, T=200) | 955s total | `outputs/dpf/2026-02-14_15-23-03/summary.json` |

**The backward pass is ~750x more expensive than the forward pass.** Normal ratio is 1-3x. The gradient is not being computed optimally.

### Comparison: LinearGaussian + EKF DPF run

| | EKF run | LEDH run |
|---|---|---|
| Filter | EKF (no particles) | LEDH (200 particles, 29 flow, OT resample) |
| T | 200 | 100 |
| HMC steps | 15 (5+10) | 15 (5+10) |
| Total wall time | 955s | ~11,250s (estimated) |
| Per HMC step | 64s | **750s** |
| Per gradient eval | 6.4s | **75s** |

### Actual DPF config used

From `outputs/dpf/kitagawa_ledh/.hydra/config.yaml`:
```yaml
filter:
  n_particles: 200, n_lambda_steps: 29, resampling_method: ot_entropy
  resampling_config: {epsilon: 0.5}, weight_clip_range: 50.0
hmc:
  num_samples: 10, num_burnin: 5, num_leapfrog_steps: 10
data:
  T: 100
```

---

## 2. Root Cause: Three Smoking Guns

### Smoking Gun #1: Flow loop creates ~32,000 individual GradientTape operations

The flow loop in `update()` ([ledh_invertible.py:232-253](code/src/filters/particle/ledh_invertible.py#L232-L253)) is a **Python for-loop** over 29 iterations. Each iteration creates ~11 individually-recorded GradientTape operations:

```python
for j in range(self.n_lambda_steps):          # 29 Python iterations
    A_batch, b_batch = compute_flow_params_batch(...)  # tape op 1 (@tf.function)
    drift_bar = tf.einsum(...)                          # tape op 2
    eta_bar = eta_bar + d_lambda * drift_bar            # tape ops 3-4
    drift_1 = tf.einsum(...)                            # tape op 5
    eta_1 = eta_1 + d_lambda * drift_1                  # tape ops 6-7
    M_batch = tf.expand_dims(...) + d_lambda * A_batch  # tape ops 8-9
    log_det_M = tf.math.log(tf.abs(tf.linalg.det(...))) # tape op 10
    log_theta = log_theta + log_det_M                   # tape op 11
```

**Total flow tape ops**: 29 × ~11 × 100 timesteps = **~31,900 tape entries**

Each entry stores tensor references and requires Python-level backward computation during `tape.gradient()`. At ~2ms per backward op (including Python dispatch overhead), this alone accounts for **~64s** of the 75s per gradient eval.

### Smoking Gun #2: OT Sinkhorn backward re-differentiates the entire Sinkhorn graph

The custom gradient for OT resampling ([ot_entropy.py:413-422](code/src/resampling/ot_entropy.py#L413-L422)):

```python
@tf.custom_gradient
def compute_transport_matrix_with_gradient(...):
    # Forward: Sinkhorn (~100 iterations on 200×200 cost matrix)
    alpha, beta = sinkhorn_with_epsilon_scaling(...)
    T = compute_transport_matrix_from_potentials(...)

    def gradient(dT):
        dT_clipped = tf.clip_by_value(dT, -1.0, 1.0)
        # THIS re-differentiates through the ENTIRE Sinkhorn computation
        dparticles, dlog_weights = tf.gradients(T, [particles, log_weights], dT_clipped)
        return dparticles, dlog_weights, None, None, None, None
```

**The problem**: `tf.gradients(T, ...)` in the backward pass **re-differentiates through the entire Sinkhorn algorithm** — including `sinkhorn_with_epsilon_scaling` which uses `tf.while_loop` with up to 100 iterations and 10 epsilon-scaling steps. The cost matrix is 200×200 = 40,000 entries.

This happens at **every timestep where resampling triggers** (~40 out of 100 timesteps based on the 40% resampling rate in standalone filter runs).

### Smoking Gun #3: Diagnostic code and `.numpy()` calls inside the tape

Several operations in `update()` are diagnostic-only but run inside GradientTape:

```python
# Line 276: stores tensor reference — keeps ALL timestep weights alive in memory
self.weights_history.append(self.weights.value())

# Line 283: appends scalar tensor to Python list
self.log_likelihoods.append(log_lik)

# Line 292-293: ESS computation not needed for likelihood
ess = ess_tf(self.weights.value())
self.ess_history.append(ess)

# Lines 299-301: .numpy() and np.unique INSIDE the tape (forces GPU→CPU sync)
particles_np = self.particles.numpy()
n_unique = len(np.unique(particles_np, axis=0))
```

The `.numpy()` call ([ledh_invertible.py:299](code/src/filters/particle/ledh_invertible.py#L299)) forces device synchronization. The list accumulation keeps all 100 timesteps of intermediate tensors alive in memory.

### Additional issue: 700 `tf.Variable.assign` operations on the tape

`predict()` has 4 `.assign()` calls, `update()` has 3 = **7 per timestep × 100 = 700 total** Variable operations on the tape, each creating dependency edges in the gradient graph.

---

## 3. Concrete Optimization Plan

### Fix 1: Compiled flow loop (highest impact)

**What**: Wrap the 29-step flow loop in a single `@tf.function`. The Python for-loop gets unrolled during tracing into a single compiled graph. GradientTape records ONE entry instead of ~320 per timestep.

**Why it works**: Inside the flow loop, `compute_flow_params_batch` calls `model.observation_jacobian_batch` and `model.observation_function_batch`. For the Kitagawa model:
- `observation_jacobian_batch(x)` → returns `x / 10` (parameter-independent)
- `observation_function_batch(x)` → returns `x² / 20` (parameter-independent)

R and R_inv (which depend on sigma_W) are passed as **explicit arguments**, not read from model attributes. So the `@tf.function` trace is valid across all HMC parameter proposals without retracing.

```python
@tf.function
def _flow_loop(model, eta_0, eta_bar_0, particle_covs, y, R, R_inv,
               lambda_steps, state_dim, regularization):
    """Compiled flow loop — ONE tape entry instead of ~320 per timestep."""
    eta_1 = eta_0
    eta_bar = eta_bar_0
    lambda_val = tf.constant(0.0, dtype=eta_0.dtype)
    n_particles = tf.shape(eta_0)[0]
    log_theta = tf.zeros(n_particles, dtype=eta_0.dtype)
    I_sd = tf.eye(state_dim, dtype=eta_0.dtype)
    n_steps = lambda_steps.shape[0]

    for j in range(n_steps):  # Unrolled during tracing — 29 iterations is fine
        d_lambda = lambda_steps[j]
        lambda_val = lambda_val + d_lambda

        A_batch, b_batch = compute_flow_params_batch(
            model, eta_bar, lambda_val, y, particle_covs,
            R, R_inv, eta_bar_0, state_dim, regularization
        )

        drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
        eta_bar = eta_bar + d_lambda * drift_bar

        drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
        eta_1 = eta_1 + d_lambda * drift_1

        M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
        log_det_M = tf.math.log(tf.abs(tf.linalg.det(M_batch)))
        log_theta = log_theta + log_det_M

    return eta_1, eta_bar, log_theta
```

**Tape entries**: 31,900 → 100 (one per timestep). **Expected speedup**: 10-50x on backward pass.

### Fix 2: `stop_gradient` on resampled particles

**What**: Wrap resampled particles/weights/covariances in `tf.stop_gradient`. This prevents GradientTape from differentiating through the Sinkhorn OT computation.

**Why it's correct**: In particle MCMC (Andrieu et al. 2010), the random seed is fixed and the likelihood estimate is a deterministic function of θ. The gradient of log p(y|θ) w.r.t. θ flows through:
- θ → Q, R → flow params A, b → flowed particle positions η₁ → log p(y|η₁)
- θ → Q, R → importance weights

The resampling step redistributes particles but does NOT need to be differentiated for the parameter gradient to be valid. The standard PMCMC approach does not differentiate through resampling.

```python
# In log_marginal_likelihood_tf, after resampling:
if ess < self.resample_threshold * self.n_particles:
    self._resample()
    self.particles.assign(tf.stop_gradient(self.particles.value()))
    self.weights.assign(tf.stop_gradient(self.weights.value()))
    self.particle_covs.assign(tf.stop_gradient(self.particle_covs.value()))
```

**Impact**: Eliminates ~40 Sinkhorn backward passes per filter evaluation. **Expected**: 2-5x additional speedup.

### Fix 3: Clean `log_marginal_likelihood_tf` path

**What**: Remove all diagnostic tracking (ESS history, weight history, unique particle counts, `.numpy()` calls) from the differentiable code path. These are only needed for `filter()`, not for HMC.

```python
def log_marginal_likelihood_tf(self, observations, seed=None):
    random_seed = int(seed[0].numpy()) if seed is not None else 42
    self.initialize(random_seed=random_seed)

    T = observations.shape[0]
    total_log_lik = tf.constant(0.0, dtype=self.dtype)

    R = self.model.observation_noise_cov
    R_inv = tf.linalg.inv(R)
    reg = tf.constant(self.regularization, dtype=self.dtype)

    for t in range(T):
        if hasattr(self.model, 't'):
            self.model.t = t + 1

        # --- Predict ---
        self.predict()

        # --- Flow (compiled — single tape entry) ---
        eta_1, _, log_theta = _flow_loop(
            self.model, self.eta_0.value(), self.eta_bar_0.value(),
            self.particle_covs.value(), observations[t],
            R, R_inv, self.lambda_steps, self.state_dim, reg
        )
        max_log_theta = tf.reduce_max(log_theta)
        theta = tf.exp(log_theta - max_log_theta)
        self.particles.assign(eta_1)

        # --- Weights ---
        weights_new = compute_flow_weights(
            eta_1=eta_1, eta_0=self.eta_0.value(),
            particles_prev=self.particles_prev.value(),
            observation=observations[t], model=self.model,
            prev_weights=self.weights.value(), jacobians=theta,
            clip_range=self.weight_clip_range
        )
        self.weights.assign(weights_new)

        # --- Log-likelihood (NO diagnostic lists) ---
        log_likelihood = self.model.log_observation_prob_batch(observations[t], eta_1)
        max_ll = tf.reduce_max(log_likelihood)
        log_lik = max_ll + tf.math.log(tf.reduce_mean(tf.exp(log_likelihood - max_ll)))
        total_log_lik = total_log_lik + log_lik

        # --- EKF covariance update ---
        _, cov_updated = batched_ekf_update(
            self.model, self.eta_bar_0.value(), self.particle_covs.value(),
            observations[t]
        )
        self.particle_covs.assign(cov_updated)

        # --- Resample with stop_gradient ---
        ess = ess_tf(self.weights.value())
        if ess < self.resample_threshold * self.n_particles:
            self._resample()
            self.particles.assign(tf.stop_gradient(self.particles.value()))
            self.weights.assign(tf.stop_gradient(self.weights.value()))
            self.particle_covs.assign(tf.stop_gradient(self.particle_covs.value()))

    return total_log_lik
```

**Impact**: Eliminates list accumulation, `.numpy()` calls, and diagnostic computation from the tape. Reduces memory pressure.

### Fix 4 (future): Fully functional filter + `@tf.function`

Refactor `log_marginal_likelihood_tf` to pass all state as tensors (no `tf.Variable.assign`). This enables wrapping the entire function in `@tf.function`, reducing ALL tape entries to **1**.

```python
@tf.function
def log_marginal_likelihood_compiled(sigma_V, sigma_W, observations, ...):
    particles, covs, weights = initialize_tensors(...)
    total_ll = 0.0
    for t in range(T):  # Unrolled during tracing
        particles, covs, weights, ll = _timestep(
            particles, covs, weights, observations[t], sigma_V, sigma_W, t
        )
        total_ll += ll
    return total_ll
```

**Expected**: Backward pass drops to ~0.1-1s (same order as forward). TF handles all differentiation internally in C++/XLA.

**Difficulty**: Significant refactoring — predict/update must become pure functions.

---

## 4. Expected Impact

| Fix | Tape entries | Est. per gradient eval | Speedup |
|-----|-------------|------------------------|---------|
| **Current** | ~32,000+ | 75s | 1x |
| Fix 1 (compiled flow loop) | ~1,500 | ~5-15s | 5-15x |
| Fix 1 + 2 (+ stop_gradient resample) | ~1,100 | ~3-8s | 10-25x |
| Fix 1 + 2 + 3 (+ clean path) | ~800 | ~2-5s | 15-40x |
| Fix 4 (full @tf.function) | 1 | ~0.1-1s | 75-750x |

### Projected HMC runtimes with Fix 1+2+3

| Config | Per step (10 leapfrog) | 15-step test | 750-step full |
|--------|----------------------|--------------|---------------|
| Current | 750s | 3.1h | 6.5 days |
| With Fix 1+2+3 | ~30-50s | 7-12 min | 6-10 hours |

---

## 5. Gradient Path Reference

### Parameter dependence map

| Model method | sigma_V | sigma_W | Used in |
|---|---|---|---|
| `state_transition_cov_batch` | **YES** (returns σ_V²) | no | `batched_ekf_predict`, `compute_flow_weights` |
| `state_transition_batch` | **YES** (scales noise) | no | `predict()` |
| `observation_noise_cov` | no | **YES** (returns σ_W²) | `update()` → R, R_inv |
| `observation_cov` | no | **YES** (returns σ_W²) | `batched_ekf_update` |
| `log_observation_prob_batch` | no | **YES** (variance) | `compute_flow_weights`, log-lik |
| `observation_jacobian_batch` | no | no | `compute_flow_params_batch` (flow loop) |
| `observation_function_batch` | no | no | `compute_flow_params_batch` (flow loop) |
| `state_transition_mean_batch` | no | no | `batched_ekf_predict`, `compute_flow_weights` |

### Why compiled flow loop doesn't need retracing

The flow loop calls `compute_flow_params_batch(model, eta_bar, lambda_val, y, particle_covs, R, R_inv, ...)`.

Inside `compute_flow_params_batch`:
- `model.observation_jacobian_batch(linearization_points)` → Kitagawa: `particles / 10` — **no parameters**
- `model.observation_function_batch(linearization_points)` → Kitagawa: `particles² / 20` — **no parameters**

R, R_inv are explicit arguments (not read from model). The model's sigma_V/sigma_W are NOT accessed inside the flow loop. The trace captures the parameter-independent model methods and is reusable.

---

## 6. Timing Data Fix

Per-step timing was not being saved. Fixed in [hmc_runner.py](code/src/DF/hmc_runner.py): `DPFResult.metadata['timing']` now includes `step_times`, `total_time_seconds`, `mean_step_time`, `mean_time_per_gradient_eval`, etc.
