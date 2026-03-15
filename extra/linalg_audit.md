# Linear Algebra Safety Audit

Full scan of `code/src/` for standalone (unprotected) linear algebra operations.

## Available Safe Wrappers (`utils/linalg.py`)

| Function | What it does |
|---|---|
| `safe_cholesky(A, jitter, adaptive)` | Cholesky with adaptive diagonal regularization |
| `safe_solve(A, b, method)` | Linear solve with Cholesky fallback |
| `safe_inv(A, jitter)` | Matrix inverse with diagonal jitter |
| `safe_log_abs_det(M, jitter)` | `log|det(M)|` with jitter for backward-pass stability |
| `log_det(A)` | Stable log-det via `tf.linalg.slogdet` |
| `symmetrize(A)` | Force symmetry: `(A + A^T) / 2` |
| `matrix_sqrt(A)` | Square root via `safe_cholesky` or eigendecomposition |

## Files Already Using Safe Wrappers

| File | Imports |
|---|---|
| `filters/kalman/extended_kalman.py` | `safe_cholesky`, `safe_solve`, `symmetrize` |
| `filters/kalman/batched_ekf.py` | `safe_cholesky`, `symmetrize` |
| `filters/kalman/unscented_kalman.py` | `safe_cholesky`, `symmetrize` |
| `filters/kalman/kalman.py` | `symmetrize` |
| `filters/particle/ledh_invertible_hmc.py` | `safe_log_abs_det`, `safe_inv` |
| `utils/distributions.py` | `safe_cholesky` |
| `utils/flow_params.py` | `safe_solve`, `safe_cholesky` |

---

## CRITICAL: Raw `det()` in Flow Jacobian Accumulation

These are the most dangerous operations in the codebase. The Jacobian log-determinant
is accumulated over 20-30 lambda steps; a single NaN or Inf corrupts the entire
weight and causes weight collapse. This is the **primary suspect for weight collapse on CUDA**.

### 1. `filters/particle/ledh_invertible.py:252`
```python
log_det_M = tf.math.log(tf.abs(tf.linalg.det(M_batch)))  # (N,)
```
- **Risk: CRITICAL** — No jitter. `det()` backward pass uses `MatrixInverse`, which
  crashes in `@tf.function` when `M` is singular. Accumulated over ~29 steps.
- **Fix:** Replace with `safe_log_abs_det(M_batch)`.

### 2. `filters/particle/ledh_invertible_bimodal.py:115`
```python
log_det_M = tf.math.log(tf.abs(tf.linalg.det(M_batch)))
```
- **Risk: CRITICAL** — Same as above (bimodal variant inherits the same flow loop).
- **Fix:** Replace with `safe_log_abs_det(M_batch)`.

### 3. `models/acoustic_tracking.py:378`
```python
log_det_2pi_R = tf.math.log(tf.linalg.det(2.0 * np.pi * R_tf))
```
- **Risk: HIGH** — No `tf.abs()`, no jitter. If `R` is near-singular, `det` can
  return 0 or negative (due to float rounding), making `log()` produce `-inf` or `NaN`.
- **Fix:** Replace with `log_det(2.0 * np.pi * R_tf)` (uses `slogdet`).

### 4. `models/acoustic_tracking.py:266` (NumPy path)
```python
log_det = np.log(np.linalg.det(2 * np.pi * self.R) + 1e-10)
```
- **Risk: MEDIUM** — Has `+1e-10` fallback, but `det` can still return negative for
  near-singular matrices, making `det + 1e-10` still negative.
- **Fix:** Use `np.linalg.slogdet` instead.

---

## HIGH: Raw `tf.linalg.inv()` Calls

All of these can produce NaN/Inf when the matrix is near-singular.
Using `safe_inv` adds diagonal jitter to prevent crashes.

### Particle Flow Filters (R_inv caching)

| File | Line | Code | Risk |
|---|---|---|---|
| `filters/particle/ledh_invertible.py` | 221 | `self.R_inv_cache = tf.linalg.inv(R)` | HIGH |
| `filters/particle/ledh_invertible_bimodal.py` | 89 | `self.R_inv_cache = tf.linalg.inv(R)` | HIGH |
| `filters/particle/ledh_flow.py` | 324 | `self.R_inv_cache = tf.linalg.inv(R_tf)` | HIGH |
| `filters/particle/edh_flow.py` | 218 | `R_inv_tf = tf.linalg.inv(R_tf)` | HIGH |
| `filters/particle/edh_flow_global.py` | 219 | `R_inv_tf = tf.linalg.inv(R_tf)` | HIGH |

**Note:** `R` is the observation noise covariance — typically well-conditioned and
set by the user. The risk here is moderate in practice but using `safe_inv` is cheap
insurance.

### Particle Flow Filters (P_inv, M_inv — dynamic matrices)

| File | Line | Code | Risk |
|---|---|---|---|
| `filters/particle/stochastic_edh.py` | 81 | `M_inv = tf.linalg.inv(M)` | HIGH |
| `filters/particle/stochastic_edh.py` | 117 | `J_prior = tf.linalg.inv(P)` | HIGH |
| `filters/particle/stochastic_edh.py` | 178 | `R_inv = tf.linalg.inv(R)` | HIGH |
| `filters/particle/stochastic_edh.py` | 192 | `P_inv = tf.linalg.inv(P)` | HIGH |
| `filters/particle/sde_local_correction.py` | 39 | `R_inv = tf.linalg.inv(R)` | HIGH |
| `filters/particle/sde_local_correction.py` | 55 | `P_inv = tf.linalg.inv(P)` | HIGH |
| `filters/particle/ledh_invertible_bimodal.py` | 180 | `Q_inv = tf.linalg.inv(Q)` | HIGH |

**Note:** `P` (predicted covariance) and `M = J_prior + β*J_meas` are computed
dynamically and can become near-singular during particle flow, especially when
particles spread or collapse. These are **higher risk** than `R_inv`.

### Cholesky-based inverse (edh_invertible)

| File | Line | Code | Risk |
|---|---|---|---|
| `filters/particle/edh_invertible.py` | 240-241 | `L = tf.linalg.cholesky(R)` then `tf.linalg.inv(L)^T @ tf.linalg.inv(L)` | MEDIUM |

**Note:** Computes `R_inv` via `chol(R)` then inverting `L`. More stable than raw
`inv(R)` but still uses raw `cholesky`. Should use `safe_cholesky` and prefer
`tf.linalg.cholesky_solve` instead of explicit `inv(L)`.

### Model files

| File | Line | Code | Risk |
|---|---|---|---|
| `models/acoustic_tracking.py` | 377 | `R_inv = tf.linalg.inv(R_tf)` | MEDIUM |
| `models/acoustic_tracking_full.py` | 355 | `R_inv = tf.linalg.inv(self.R)` | MEDIUM |
| `models/acoustic_tracking_full.py` | 360 | `R_inv = tf.linalg.inv(self.R)` | MEDIUM |

---

## MEDIUM: Raw `tf.linalg.cholesky()` Calls

These lack the adaptive jitter of `safe_cholesky`. If the matrix loses positive-
definiteness due to floating point drift, the raw call will crash or return NaN.

### For sampling (initial particle generation)

| File | Line | Context |
|---|---|---|
| `filters/particle/ledh_invertible.py` | 165 | `L = tf.linalg.cholesky(initial_cov_tf)` |
| `filters/particle/edh_invertible.py` | 170 | `L = tf.linalg.cholesky(initial_cov_tf)` |
| `filters/particle/edh_flow.py` | 130 | `L = tf.linalg.cholesky(initial_cov_tf)` |
| `filters/particle/edh_flow_global.py` | 132 | `L = tf.linalg.cholesky(initial_cov_tf)` |
| `filters/particle/ledh_flow.py` | 141 | `L = tf.linalg.cholesky(initial_cov_tf)` |

**Risk: LOW-MEDIUM** — `initial_cov` is user-supplied and typically well-conditioned.
But `safe_cholesky` is a drop-in replacement and costs nothing.

### In models (noise covariance decomposition)

| File | Lines | Context |
|---|---|---|
| `models/linear_gaussian.py` | 178, 289, 333 | Cholesky of `Sigma_0`, `R`, `Sigma_0` |
| `models/range_bearing.py` | 119, 126, 316, 342, 364 | Cholesky of `Sigma_0`, `Q`, `R`, `Q`, `Sigma_0` |
| `models/two_sensor_bearing.py` | 131, 346, 385 | Cholesky of `Sigma_0`, `R`, `Sigma_0` |
| `models/acoustic_tracking.py` | 354 | Cholesky of `Q_tf` |
| `models/acoustic_tracking_full.py` | 185, 195, 384, 406 | Cholesky of `Sigma_0`, `Q` |

**Risk: LOW** — These are fixed noise matrices from the model definition. Unlikely
to fail, but `safe_cholesky` provides cheap protection.

---

## LOW: NumPy Operations (Debug / Diagnostics Only)

These are used for debug logging or diagnostics, not in the computational graph.

| File | Lines | Operations |
|---|---|---|
| `filters/particle/edh_flow.py` | 247-248 | `np.linalg.eigvals`, `np.linalg.cond` |
| `filters/particle/edh_flow_global.py` | 248-249 | `np.linalg.eigvals`, `np.linalg.cond` |
| `filters/particle/ledh_flow.py` | 357-358 | `np.linalg.eigvals`, `np.linalg.cond` |
| `filters/kalman/kalman.py` | 232 | `np.linalg.cond` |
| `models/linear_gaussian.py` | 171 | `np.linalg.cholesky` (for initial sampling) |

**Risk: NONE** — Debug code; failures here don't affect filter output.

### NumPy fallback path in `models/acoustic_tracking.py`

| Line | Code |
|---|---|
| 259 | `np.linalg.cholesky(self.R)` |
| 260 | `np.linalg.solve(L, residual)` |
| 265 | `np.linalg.pinv(self.R)` |
| 266 | `np.log(np.linalg.det(2 * np.pi * self.R) + 1e-10)` |

**Risk: LOW** — This is a NumPy fallback path (inside `except LinAlgError`), so it
already handles the Cholesky failure case. Could improve the `det + 1e-10` idiom
but low priority.

---

## Operations That Are Already Safe

| File | Line | Why |
|---|---|---|
| `models/acoustic_tracking_full.py:350` | `tf.linalg.slogdet(2π·R)` | `slogdet` is stable |
| `models/range_bearing.py:225` | `tf.linalg.slogdet(2π·R)` | `slogdet` is stable |
| `models/two_sensor_bearing.py:255` | `tf.linalg.slogdet(2π·R)` | `slogdet` is stable |
| `models/range_bearing.py:229` | `tf.linalg.solve(R, diff_col)` | `solve` is stable for SPD |
| `models/two_sensor_bearing.py:257` | `tf.linalg.solve(R, diff_col)` | `solve` is stable for SPD |

---

## Action Plan

### Phase 1: Critical — Fix Jacobian Determinant (Weight Collapse Root Cause)

These are the most impactful changes. They directly address the weight collapse
observed on CUDA (RTX 3090) where float32 precision differs from MPS.

1. **`ledh_invertible.py:252`** — Replace raw `log(abs(det(M)))` with `safe_log_abs_det(M_batch)`
   - Add import: `from ...utils.linalg import safe_log_abs_det`
   - Change: `log_det_M = safe_log_abs_det(M_batch)`

2. **`ledh_invertible_bimodal.py:115`** — Same fix
   - Add import: `from ...utils.linalg import safe_log_abs_det`
   - Change: `log_det_M = safe_log_abs_det(M_batch)`

3. **`acoustic_tracking.py:378`** — Replace `log(det(2π·R))` with `slogdet`
   - Change to: `sign, log_det_2pi_R = tf.linalg.slogdet(2.0 * np.pi * R_tf)`

### Phase 2: High — Protect Dynamic Matrix Inverses

These matrices are computed during flow and can become ill-conditioned.

4. **`stochastic_edh.py`** — Replace all 4 raw `tf.linalg.inv()` with `safe_inv`
   - Add import: `from ...utils.linalg import safe_inv`
   - Lines 81, 117, 178, 192

5. **`sde_local_correction.py`** — Replace 2 raw `tf.linalg.inv()` with `safe_inv`
   - Add import: `from ...utils.linalg import safe_inv`
   - Lines 39, 55

6. **`ledh_invertible_bimodal.py:180`** — Replace `tf.linalg.inv(Q)` with `safe_inv`

### Phase 3: Medium — Protect Static Matrix Inverses (R_inv caching)

Lower risk since `R` is user-defined noise, but still worth protecting.

7. **`ledh_invertible.py:221`** — Replace with `safe_inv(R)`
8. **`ledh_invertible_bimodal.py:89`** — Replace with `safe_inv(R)`
9. **`ledh_flow.py:324`** — Replace with `safe_inv(R_tf)`
10. **`edh_flow.py:218`** — Replace with `safe_inv(R_tf)`
11. **`edh_flow_global.py:219`** — Replace with `safe_inv(R_tf)`
12. **`edh_invertible.py:240-241`** — Replace Cholesky+inv pattern with `safe_inv(R)`
13. **`acoustic_tracking.py:377`** — Replace with `safe_inv(R_tf)`
14. **`acoustic_tracking_full.py:355,360`** — Replace with `safe_inv(self.R)`

### Phase 4: Low — Replace Raw Cholesky in Particle Filters

15. Replace `tf.linalg.cholesky(initial_cov_tf)` with `safe_cholesky(initial_cov_tf)`
    in: `ledh_invertible.py:165`, `edh_invertible.py:170`, `edh_flow.py:130`,
    `edh_flow_global.py:132`, `ledh_flow.py:141`

### Phase 5: Skip — No Action Needed

- NumPy debug operations (`np.linalg.eigvals`, `np.linalg.cond`) — diagnostic only
- NumPy fallback in `acoustic_tracking.py:259-266` — already inside `except LinAlgError`
- Model Cholesky calls on fixed noise matrices — very low risk
- `tf.linalg.slogdet` calls — already stable
- `tf.linalg.solve` calls in models — used on well-conditioned SPD matrices

---

## Summary

| Priority | Count | Description |
|---|---|---|
| CRITICAL | 3 | Raw `det()` in Jacobian accumulation and log-likelihood |
| HIGH | 11 | Raw `inv()` on dynamic or cached matrices |
| MEDIUM | 5 | Raw `cholesky()` on initial covariances |
| LOW/SKIP | ~15 | Debug ops, fixed-matrix Cholesky, already-safe ops |

**Total standalone operations requiring attention: ~19**
**Already protected: ~7 files using safe wrappers correctly**
