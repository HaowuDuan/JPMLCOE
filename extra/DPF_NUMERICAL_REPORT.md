# DPF Numerical Instability Report

## Executive Summary

Two independent bugs cause 21/24 DPF+HMC experiments to fail:

1. **LEDH**: Float32 precision loss during 29-step Jacobian accumulation causes gradient explosion. The forward pass produces NaN/inf log-determinants, making the likelihood surface non-differentiable. **No amount of HMC tuning can fix this.**

2. **OT resampling**: The custom gradient in `compute_transport_matrix_with_gradient` hard-clips incoming gradients to [-1, 1] and uses `tf.stop_gradient` on converged Sinkhorn potentials, destroying the gradient signal. HMC receives near-zero or heavily biased gradients. **This is a code bug, not a precision issue.**

---

## Results Summary (24 experiments)

| # | Experiment | Accept | Accurate? | Time |
|---|-----------|--------|-----------|------|
| 1 | linear_gaussian_ledh_sys | 0% | NO | 160s |
| 2 | linear_gaussian_ledh_soft | 0% | NO | 162s |
| 3 | linear_gaussian_ledh_ot | 0% | NO | 191s |
| 4 | **linear_gaussian_bpf_sys** | **100%** | **YES** | 9s |
| 5 | linear_gaussian_bpf_soft | 50% | NO | 10s |
| 6 | linear_gaussian_bpf_ot | 0% | NO | 44s |
| 7 | cubic_sensor_ledh_sys | 50% | NO | 191s |
| 8 | cubic_sensor_ledh_soft | 50% | NO | 191s |
| 9 | cubic_sensor_ledh_ot | 0% | NO | 206s |
| 10 | cubic_sensor_bpf_sys | 0% | NO | 8s |
| 11 | cubic_sensor_bpf_soft | 50% | NO | 9s |
| 12 | cubic_sensor_bpf_ot | 0% | NO | 72s |
| 13 | kitagawa_ledh_sys | 0% | NO | 186s |
| 14 | kitagawa_ledh_soft | 0% | NO | 185s |
| 15 | kitagawa_ledh_ot | 0% | NO | 193s |
| 16 | **kitagawa_bpf_sys** | **100%** | **partial** | 8s |
| 17 | kitagawa_bpf_soft | 0% | NO | 9s |
| 18 | kitagawa_bpf_ot | 50% | NO | 76s |
| 19 | range_bearing_ledh_sys | 0% | NO | 550s |
| 20 | range_bearing_ledh_soft | 0% | NO | 543s |
| 21 | range_bearing_ledh_ot | 0% | NO | 547s |
| 22 | range_bearing_bpf_sys | 50% | NO | 282s |
| 23 | **range_bearing_bpf_soft** | **50%** | **YES** | 282s |
| 24 | range_bearing_bpf_ot | 0% | NO | 333s |

**Pattern**: LEDH 0/12, OT 0/8, BPF+systematic 2/4 work.

---

## Issue 1: LEDH Jacobian Accumulation

### What happens

The LEDH particle flow integrates particles through 29 lambda steps. At each step, it computes `M_i = I + dλ·A_i` and accumulates log-determinants:

```python
# ledh_invertible_hmc.py:187-204

for j in range(n_flow_steps):       # 29 iterations
    d_lambda = lambda_steps[j]
    A_batch, b_batch = _flow_params(...)

    M_batch = I + d_lambda * A_batch     # (N, sd, sd)
    log_det_M = _log_abs_det(M_batch)    # log|det(M_i)|
    log_theta = log_theta + log_det_M    # ACCUMULATE
```

After the loop, normalization happens:

```python
# ledh_invertible_hmc.py:207-209

max_log_theta = tf.reduce_max(log_theta)
log_theta = log_theta - max_log_theta
theta = tf.exp(log_theta)
```

### Why it blows up on CUDA (float32)

**The A matrix eigenvalue problem.** The flow matrix is:

```python
# flow_params.py:237
A = -0.5 * P @ H^T @ S^{-1} @ H
```

where `P` is the per-particle covariance (from the batched EKF), `H` is the observation Jacobian, and `S = λ·H·P·H^T + R`. The eigenvalues of `A` scale with the eigenvalues of `P`.

When `P` grows (poor EKF conditioning), the eigenvalues of `dλ·A` can approach or exceed 1 in magnitude. Then `det(I + dλ·A)` becomes very small or negative, and `log|det|` produces large negative values.

**Accumulation over 29 steps.** Each `log_det_M` contributes roughly O(1) to O(10) in magnitude. After 29 additions in float32:
- Individual values: ±0.1 to ±10
- Accumulated sum: can reach ±100s
- Float32 mantissa: 24 bits → ~7 decimal digits
- Relative precision at sum ≈ 100: only ~5 digits in individual terms

**The cascade:**
1. Accumulated `log_theta` values lose relative precision across particles
2. After normalization (`log_theta - max_log_theta`), small differences between particles are lost
3. `exp()` of these imprecise values → all particles get ~same weight or NaN
4. Gradient of `log|det|` requires `M^{-T}` (linalg.py:228-229), which amplifies the error
5. HMC leapfrog uses these gradients → parameters fly to ±inf → NaN

**Evidence from the logs:**

```
# linear_gaussian_ledh_hmc_soft — MAP optimization diverges:
[nlp] q= [1.90477431]     # reasonable
[nlp] q= [1.50762582]     # drifting
[nlp] q= [-0.10146296]    # wrong direction
[nlp] q= [-3.74681568]    # diverging
[nlp] q= [-8.22657585]    # gone
[nlp] q= [32225134]       # exploded

# cubic_sensor_ledh_hmc_sys — NaN appears:
[nlp] q= [-5.0032]
[nlp] q= [-59080.125 420407.469]   # inf
[nlp] q= [nan 840819.938]          # NaN
```

### Why it works on MPS (macOS)

MPS backend uses float64 internally for many operations. Float64 has 53-bit mantissa (~15 decimal digits), giving ~8 more digits of precision in the accumulation. The same accumulated log_theta ≈ 100 still has ~13 digits of precision in the differences between particles.

### The backward pass problem

```python
# linalg.py:208-237 — graph_safe_log_abs_det_fast (used at line 150 of ledh_invertible_hmc.py)

@tf.custom_gradient
def _graph_safe_log_abs_det_fast_impl(M_reg):
    sign, logabsdet = tf.linalg.slogdet(M_reg)       # Forward: OK
    def grad(dy):
        is_finite = tf.reduce_all(tf.math.is_finite(M_reg), ...)
        M_safe = tf.where(is_finite[...], M_reg, eye)  # Replace NaN with I
        M_inv_T = transpose(inv(M_safe + 1e-6 * eye))  # Backward: inv
        M_inv_T = tf.where(is_finite[...], M_inv_T, zeros)  # Zero grad for NaN
        return dy[...] * M_inv_T
    return logabsdet, grad
```

The NaN guard (replacing NaN matrices with identity, zeroing their gradient) is a band-aid. When `M_batch` has near-singular entries, `inv(M_safe + 1e-6 * I)` still produces large gradient magnitudes. Over 29 steps in the backward pass, these compound.

---

## Issue 2: OT Resampling Gradient

### What happens

The OT resampling uses Sinkhorn to compute a transport matrix `T`, then resamples via `T @ particles`. The gradient of the HMC log-likelihood must flow back through `T` to the model parameters.

### Bug 1: Aggressive gradient clipping (the main problem)

```python
# ot_entropy.py:368-424

@tf.custom_gradient
def compute_transport_matrix_with_gradient(particles, log_weights, ...):
    # Forward: compute T via Sinkhorn
    T = compute_transport_matrix_from_potentials(...)

    def gradient(dT):
        dT_clipped = tf.clip_by_value(dT, -1.0, 1.0)   # <-- KILLS GRADIENT
        dparticles, dlog_weights = tf.gradients(T, [particles, log_weights], dT_clipped)
        return dparticles, dlog_weights, None, None, None, None

    return T, gradient
```

The transport matrix `T` has shape `(N, N)` where N=200 particles. Each entry is O(1/N) = O(0.005). The incoming gradient `dT` from `T @ particles` has entries proportional to the particle values, which can easily exceed 1.0.

**Clipping `dT` to [-1, 1] destroys the gradient signal.** The clipped values bear no relation to the actual gradient direction. HMC receives a garbage gradient → proposes bad moves → 0% acceptance.

### Bug 2: Incomplete implicit differentiation

```python
# ot_entropy.py:195-210

# Run Sinkhorn to convergence
n_iter, alpha_final, beta_final, _ = tf.while_loop(...)

# "Implicit function theorem" step:
alpha_stop = tf.stop_gradient(alpha_final)     # <-- CUTS GRADIENT HISTORY
beta_stop = tf.stop_gradient(beta_final)

alpha_extra = softmin(eps, C^T, log_w + beta_stop / eps)
beta_extra = softmin(eps, C, uniform_log_w + alpha_extra / eps)
```

The idea is from Corenflos et al. 2021: at convergence, the fixed-point equation `α = softmin(ε, C^T, log_w + β/ε)` holds, so you can differentiate through one iteration using the implicit function theorem instead of unrolling all Sinkhorn iterations.

**However**, `tf.stop_gradient` on `alpha_final` and `beta_final` means gradients only flow through the single extrapolation step. This is a valid approximation IF the Sinkhorn has truly converged. But combined with the epsilon-scaling loop (which uses `tf.while_loop` with `maximum_iterations=10`), the potentials may not be fully converged, making the single-step gradient a poor approximation.

### Bug 3: Stop-gradient on centering/scaling

```python
# ot_entropy.py:395-401

mean = tf.reduce_mean(particles, axis=0, keepdims=True)
centered = particles - tf.stop_gradient(mean)           # stop_gradient

std = tf.math.reduce_std(particles)
scale_factor = tf.stop_gradient(std * tf.sqrt(dim) + 1e-8)  # stop_gradient
scaled = centered / scale_factor
```

The particle centering and scaling use `tf.stop_gradient`, so the Sinkhorn computation doesn't see how particles change with model parameters through the normalization constants. This introduces a bias in the gradient.

### Why systematic resampling works

Systematic resampling uses `tf.searchsorted` on cumulative weights → discrete ancestor indices → `tf.gather`. The gradient doesn't flow through resampling at all (inherently non-differentiable). But when `stop_gradient_resampling=false`, the HMC gradient only needs to flow through the **weight computation**, not through particle positions. For BPF, the weight computation is simply the observation likelihood — clean, simple, well-conditioned gradients.

### Why soft resampling partially works

Soft resampling computes `q = α·w + (1-α)/N`, resamples, then importance-corrects with `w' = w/q`. The gradient flows through the weight ratio, which is a smooth, simple function. No Sinkhorn, no transport matrix, no custom gradients.

---

## Gradient Path Summary

| Filter + Resampling | Gradient Path | Status | Root Cause |
|---------------------|--------------|--------|------------|
| BPF + systematic | weights only (simple) | **WORKS** | Clean gradient |
| BPF + soft | weights + ratio correction | Partial | Slightly noisier |
| BPF + OT | weights + Sinkhorn transport | **BROKEN** | Gradient clipping bug |
| LEDH + systematic | weights + 29-step Jacobian | **BROKEN** | Float32 precision |
| LEDH + soft | weights + 29-step Jacobian | **BROKEN** | Float32 precision |
| LEDH + OT | weights + Jacobian + Sinkhorn | **BROKEN** | Both bugs compound |

---

## Recommended Fixes

### Fix 1: LEDH — Force float64 for Jacobian accumulation

Minimal change: cast `log_theta`, `M_batch`, and `A_batch` to float64 inside the flow loop, cast back after normalization. This preserves float32 for the bulk of computation (particles, observations) while giving the numerically sensitive accumulation 15 digits of precision.

```python
# In the flow loop (ledh_invertible_hmc.py):
M_batch_64 = tf.cast(M_batch, tf.float64)
log_det_M_64 = _log_abs_det(M_batch_64)
log_theta_64 = log_theta_64 + log_det_M_64

# After loop:
log_theta = tf.cast(log_theta_64, particles.dtype)
```

Estimated impact: ~10-20% slower (only the det/inv is in float64, not the whole filter).

### Fix 2: OT — Remove gradient clipping

The gradient clipping at line 416 is actively harmful. Remove it:

```python
# ot_entropy.py:413-416
def gradient(dT):
    # dT_clipped = tf.clip_by_value(dT, -1.0, 1.0)  # REMOVE THIS
    dparticles, dlog_weights = tf.gradients(T, [particles, log_weights], dT)
    return dparticles, dlog_weights, None, None, None, None
```

If gradients are truly too large, use gradient norm clipping in the HMC integrator instead (which operates on the full gradient vector, preserving direction).

### Fix 3: OT — Remove stop_gradient on centering/scaling

```python
# ot_entropy.py:395-401
mean = tf.reduce_mean(particles, axis=0, keepdims=True)
centered = particles - mean                              # no stop_gradient

std = tf.math.reduce_std(particles)
scale_factor = std * tf.sqrt(dimension) + 1e-8           # no stop_gradient
scaled = centered / scale_factor
```

### Fix 4: Priority order

1. **Fix OT gradient clipping** (Bug 2, Fix 2) — clear code bug, easy fix, affects 8 experiments
2. **Fix LEDH float64** (Bug 1, Fix 1) — targeted precision fix, affects 12 experiments
3. **Fix OT centering** (Bug 2, Fix 3) — secondary improvement
4. **Re-run with more HMC steps** (20 burn-in + 50 samples) to get real diagnostics
