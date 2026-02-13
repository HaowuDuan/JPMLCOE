# EKF Linearization Point Issue in LEDH Invertible Filter

## Summary

The LEDH filter's auxiliary per-particle EKF does not maintain its own mean chain across timesteps. The EKF updated mean `m_i^+` is computed but discarded, and the next timestep's EKF prediction uses the flowed particle position instead. This breaks the EKF mean/covariance consistency and diverges from Algorithm 1 (Li & Coates 2017).

---

## The Problem

### What should happen (per Algorithm 1)

Each particle `i` has an independent auxiliary EKF that maintains `(m_i, P_i)`:

```
Timestep k:
  EKF predict:  m_i^-  = f(m_i^{k-1,+}),   P_i^-  = F_i P_i^{k-1,+} F_i^T + Q
  Flow:         uses P_i^- and m_i^- (as η̄_0 and linearization point)
  EKF update:   m_i^+  = m_i^- + K_i(y - h(m_i^-)),   P_i^+ = (I - K_i H_i) P_i^-

Timestep k+1:
  EKF predict:  m_i^-  = f(m_i^{k,+}),   P_i^-  = F_i P_i^{k,+} F_i^T + Q
                         ^^^^^^^^^^^
                         Uses EKF's own updated mean from previous step
```

The EKF mean chain is: `m_0 → m_1^- → m_1^+ → m_2^- → m_2^+ → ...`

### What actually happens in the code

**`update()` at `ledh_invertible.py:279-282`** — EKF updated mean is discarded:

```python
_, cov_updated = batched_ekf_update(
    self.model, self.eta_bar_0.value(), self.particle_covs.value(), y
)
self.particle_covs.assign(cov_updated)  # Only covariance is kept
```

The `_` discards `mean_updated` (= `m_i^+`). The covariance `P_i^+` is stored correctly.

**`predict()` at `ledh_invertible.py:189-190`** — Next EKF predict uses flowed particles, not EKF means:

```python
eta_bar_0_tf, cov_pred_tf = batched_ekf_predict(
    self.model, self.particles.value(), self.particle_covs.value()
)
```

`self.particles.value()` at this point contains the flowed (and possibly resampled) particles `x_i^1` from the previous timestep — **not** the EKF updated mean `m_i^+`.

The actual chain is: `m_0 → m_1^- → (m_1^+ discarded) → f(x_1^flowed) → ...`

---

## Why This Matters

The flowed particle `x_i^1` and the EKF updated mean `m_i^+` incorporate the observation in fundamentally different ways:

| Quantity | How it incorporates observation |
|---|---|
| `m_i^+` (EKF mean) | Linear Kalman correction: `m^- + K(y - h(m^-))` |
| `x_i^1` (flowed particle) | Nonlinear ODE integration through the flow field |

These produce different values, especially when:
- The observation model `h(x)` is highly nonlinear
- The flow moves particles far from the EKF predicted mean
- The Kalman gain `K` and flow matrices `A, b` imply different corrections

### Consequences

1. **Jacobian mismatch**: The next step's `batched_ekf_predict` evaluates `F = ∂f/∂x` at the particle position `x_i^1` instead of at `m_i^+`. For nonlinear `f`, these Jacobians differ, producing different predicted covariances `P_i^-`.

2. **Covariance drift**: Over many timesteps, the EKF covariance `P_i` drifts from what it should be, because the linearization points for `F` and `H` are inconsistent with the mean trajectory the EKF "thinks" it's tracking.

3. **Flow quality degrades**: Since the flow equations use `P_i` directly, degraded covariances lead to degraded flow fields `A(λ), b(λ)`, which in turn produce worse particle positions.

---

## Affected Code Paths

| File | Lines | Issue |
|---|---|---|
| `ledh_invertible.py` | 279 | EKF updated mean discarded (`_`) |
| `ledh_invertible.py` | 189-190 | `batched_ekf_predict` called with `self.particles` instead of stored EKF means |
| `ledh_invertible.py` | 296-327 | `_resample()` does not resample EKF means (because they don't exist as state) |

---

## Proposed Fix

### 1. Add a `particle_means` state variable

Track the EKF means separately from the particle positions:

```python
# In __init__ or initialize():
self.particle_means = tf.Variable(
    tf.zeros([self.n_particles, self.state_dim], dtype=tf.float32)
)
```

Initialize to the same values as particles:

```python
# In initialize():
self.particle_means = tf.Variable(particles_tf, dtype=tf.float32)
```

### 2. Store the EKF updated mean in `update()`

```python
# ledh_invertible.py, line 279 — change from:
_, cov_updated = batched_ekf_update(
    self.model, self.eta_bar_0.value(), self.particle_covs.value(), y
)
self.particle_covs.assign(cov_updated)

# To:
mean_updated, cov_updated = batched_ekf_update(
    self.model, self.eta_bar_0.value(), self.particle_covs.value(), y
)
self.particle_covs.assign(cov_updated)
self.particle_means.assign(mean_updated)
```

### 3. Use EKF means (not particles) for EKF predict

```python
# ledh_invertible.py, line 189 — change from:
eta_bar_0_tf, cov_pred_tf = batched_ekf_predict(
    self.model, self.particles.value(), self.particle_covs.value()
)

# To:
eta_bar_0_tf, cov_pred_tf = batched_ekf_predict(
    self.model, self.particle_means.value(), self.particle_covs.value()
)
```

### 4. Resample EKF means alongside covariances

In `_resample()`, after obtaining ancestor indices:

```python
# After the resampling refactor (using ResampleResult):
self.particle_covs.assign(tf.gather(self.particle_covs.value(), indices))
self.particle_means.assign(tf.gather(self.particle_means.value(), indices))
```

---

## Verification Plan

1. **Unit test**: After EKF update, verify `particle_means` equals `batched_ekf_update` output (not particle positions).
2. **Consistency check**: At each timestep, log `||particle_means - particles||` to quantify divergence.
3. **Regression test**: Run the acoustic tracking experiment with both implementations. Compare:
   - RMSE against ground truth
   - ESS trajectory
   - Covariance trace over time (if P_i drift is happening, traces will diverge between old and new)
4. **Compare with MATLAB**: If MATLAB reference results are available, the fixed version should match more closely.

---

## Impact Assessment

- **Severity**: Medium-high for nonlinear models with large state dimensions.
  For nearly-linear models, `m_i^+` and `x_i^1` will be similar, so the impact is small. For the acoustic tracking model (nonlinear `h(x)` with 16-dim state, 25-dim observation), the impact could be significant.
- **Risk of fix**: Low. The change is additive (new state variable) and the EKF mean was already being computed — just discarded.
- **Interaction with resampling refactor**: The fix requires resampling `particle_means` with the same ancestor indices. This should be implemented alongside the `ResampleResult` refactor from `RESAMPLING_REFACTOR_PLAN.md`.
