# HMC Gradient Explosion Diagnosis: Range-Bearing LEDH + OT

## Config
- `hmc/range_bearing/ledh_ot.yaml`
- LEDH + OT resampling, TFP HMC, compiled path (tf.while_loop)
- 500 particles, 15 flow steps, 50 timesteps, float64 on CUDA (RTX 3090)
- σ_range=σ_bearing=0.3 (initial), true=0.1

## Symptoms

**Before range floor fix (floor=1e-10):**
```
[grad check] lp=-45.1172, |grad|=448346131.4477, grad=[-1.91e+08, +4.06e+08]
```

**After range floor fix (floor=0.3):**
```
[grad check] lp=-61.4878, |grad|=279402.1146, grad=[+2.62e+05, +9.74e+04]
```

1. **Gradient magnitude O(10⁵)** after floor fix (was O(10⁸)) — should be O(10²-10³)
2. **Both gradients WRONG DIRECTION** — both positive, but true values 0.1 < 0.3 → should be negative
3. Step size collapses to 0.0000 (TFP dual averaging shrinks it to compensate)
4. Parameters stuck at 0.3, 0.3 — never explored
5. Massive Cholesky failures throughout all burn-in steps
6. **Issue is float64** — NOT a float32 precision problem
7. **Issue is range-bearing only** — cubic_sensor works fine with same setup
8. **Issue is OT resampling only** — other resampling methods (soft, systematic) work

## Root Cause Analysis

### PRIMARY: Exploding gradient through OT cost matrix across timesteps

**This is the dominant mechanism.** The LEDH+OT filter is a recurrent computation over T=50 timesteps, and the OT resampling gradient through the cost matrix creates exponential gradient growth — identical to the exploding gradient problem in RNNs.

At each timestep, OT resampling computes:
```
new_particles = T @ old_particles    (N×N transport matrix)
```

The transport matrix T depends on particle positions through the cost matrix:
```
T_ij ∝ exp((α_i + β_j - C_ij) / ε)
C_ij = ||x_i - x_j||² / 2
```

The gradient of T w.r.t. particle positions is:
```
∂T/∂particles ∝ T × (position_differences) / ε
```

With ε=0.5, the per-timestep Jacobian of the full resampling step has spectral norm:
```
||∂new_particles/∂old_particles|| ≈ 1 + O(||particles|| / ε) > 1
```

Across 50 timesteps, this compounds exponentially:
- Amplification 1.15×/step → 1.15^50 ≈ 1,000
- Amplification 1.5×/step → 1.5^50 ≈ 6 × 10⁸

**Why this only affects OT, not soft or systematic:**
- **OT**: gradient flows through the FULL N×N cost matrix, creating ∂T/∂positions path with 1/ε amplification
- **Soft**: gradient flows through weight-based soft assignment; no full cost matrix dependency on positions
- **Systematic + stop_gradient**: chain is cut at every timestep, no cross-timestep gradient flow

**Why the gradient direction is wrong:**
The cross-timestep OT gradient dominates and says "increase σ" because:
1. Larger σ → softer flow (smaller A) → particles move less aggressively
2. Less aggressive flow → more clustered particle cloud → lower OT transport cost
3. Lower transport cost → smoother T matrix → better gradient properties
4. This effect, amplified exponentially across 50 timesteps, overwhelms the direct likelihood gradient which correctly says "decrease σ"

### SECONDARY: R⁻¹ amplification in b vector (contributes ~74× per flow step)

The b vector uses R⁻¹ directly (not S⁻¹ which self-regulates):
```
b(λ) = (I + 2λA)·[(I + λA)·P·Hᵀ·R⁻¹·(z-e) + A·η̄₀]
```

Gradient amplification: ∂(R⁻¹)/∂σ = -2/σ³ = -74 at σ=0.3.

This amplification is **intrinsic to the LEDH flow equations** — cannot be eliminated by switching to Cholesky solve (same mathematical gradient). It affects all models, not just range-bearing, but is secondary to the OT cross-timestep issue.

Note: Reparameterizing from σ to precision τ=1/σ² would eliminate this factor (∂R⁻¹/∂τ = I), but the gradient would still be wrong-direction due to the OT amplification being the dominant effect.

### TERTIARY: Near-sensor particles blow up H (range-bearing specific)

The range-bearing observation Jacobian has bearing entries ∝ 1/r². The Hessian (used in backward pass) ∝ 1/r³. For particles near the sensor:

| r     | H_bearing | Hessian | After fix (r clamped to 0.3) |
|-------|-----------|---------|------------------------------|
| 1.4   | ~0.5      | ~0.4    | unchanged                    |
| 0.3   | ~11       | ~37     | clamped here                 |
| 0.1   | ~100      | ~1000   | → 11, 37                    |
| 0.01  | ~10000    | ~10⁶    | → 11, 37                    |

**Applied fix:** Range floor increased from 1e-10 to 0.3 in `range_bearing.py` (both single and batch Jacobian). This reduced gradient from O(10⁸) to O(10⁵) — a 1600× improvement — but did not fix the OT cross-timestep issue.

## What Works and What Doesn't

| Resampling | stop_gradient | Range-bearing HMC | Why |
|------------|--------------|-------------------|-----|
| Systematic | Yes          | ✅ Works           | No cross-timestep gradient |
| Soft       | No           | ✅ Works           | No cost matrix gradient path |
| OT         | Yes          | ✅ Works           | Chain cut at each timestep |
| OT         | No           | ❌ Broken          | Exponential gradient growth through cost matrix |

## Key Code Locations

| Location | What it does | Issue |
|----------|-------------|-------|
| `ot_entropy.py:422-453` | `_compute_transport_matrix` custom gradient | `tf.gradients(T, [particles, ...])` flows through cost matrix → exponential growth |
| `ot_entropy.py:354` | `log_T = (α + β - C) / ε` | Division by ε amplifies gradient of T w.r.t. C |
| `ot_entropy.py:343` | `cost_matrix = compute_cost_matrix(particles, particles)` | C depends on particle positions → gradient path |
| `flow_params.py:252` | `PHT_Rinv = P·Hᵀ·R_inv` | R_inv directly in gradient chain, 74× amplification |
| `range_bearing.py:231,282` | `range_val = max(range_val, 0.3)` | **FIXED** — was 1e-10, now 0.3 |
| `ledh_invertible_hmc.py:260` | `tf.cond(ess < resample_thresh, ...)` | Resampling within tf.while_loop — gradient chains across timesteps |

## Proposed Fixes

### Fix 1: Increase OT epsilon (quickest test)

Increase ε from 0.5 to 2.0 or 5.0. Larger ε → smoother T → smaller ∂T/∂particles → reduced per-timestep amplification.

**Trade-off:** Degrades resampling quality (T becomes more uniform/diffusive).

### Fix 2: Truncated backprop through time

Apply `stop_gradient` every K timesteps (e.g., K=5 or K=10). The gradient chain length is bounded to K instead of 50.

```python
if t % K == 0:
    particles = tf.stop_gradient(particles)
    weights = tf.stop_gradient(weights)
```

**Trade-off:** Loses long-range gradient information. Still has gradient for K timesteps.

### Fix 3: Detach OT cost matrix from gradient

In `_compute_transport_matrix`, stop the gradient through particle positions in the cost matrix:
```python
cost_matrix = compute_cost_matrix(tf.stop_gradient(particles), tf.stop_gradient(particles))
```
Keep gradient through `log_weights` (how filter weights affect T). Cut gradient through positions (how particle positions affect T via cost matrix).

**Trade-off:** Gradient still flows through weights → T → new positions. Only the cost matrix → T → new positions path is cut. This is the path causing exponential growth.

### Fix 4: Gradient normalization in cost matrix path

Scale the gradient through the cost matrix by 1/||grad|| to prevent amplification, while preserving direction.

### Recommendation

**Fix 3 (detach cost matrix)** is the most targeted. It cuts exactly the path causing exponential growth while preserving:
- Gradient through weights → T (how particle importance affects resampling)
- Gradient through flow dynamics → weights → T → positions
- Gradient through observation likelihood → positions
- Full OT transport quality (no ε change)

**Fix 1 (larger ε)** is the quickest test to confirm the diagnosis.
