# Plan: Improve Initial Conditions Across State Space Models

## Overview

Four targeted changes to three files, ordered from most impactful to least.

---

## Change 1 — Lorenz96: `mu_0` → spinup state (not equilibrium)

**File:** `src/models/lorenz96.py` (line 126)

**Problem:** `mu_0` returns `forcing * ones(1000)` (the unstable equilibrium), but `sample_initial_state` returns `spinup_state + noise` (on the chaotic attractor). These are completely different. Any filter initialized with `model.mu_0` starts far from the truth.

**Fix:** Change the `mu_0` property to return `self._spinup_state_tf` (already computed in `__init__`). Remove the now-unused `self._mu_0 = forcing * ones` assignment.

```python
@property
def mu_0(self) -> tf.Tensor:
    """Mean of initial distribution: the spinup state on the chaotic attractor."""
    return self._spinup_state_tf
```

**Impact:** Zero regression — no caller passes `mu_0` to the Lorenz96 constructor.

---

## Change 2 — AcousticTracking: consistent Gaussian sampling

**File:** `src/models/acoustic_tracking.py` (lines 146–150, 211–215)

**Problem:** `sample_initial_state` draws from `Uniform([10,30])` for position, but `mu_0=[20,20,0,0]` and `Sigma_0=diag([33.33,33.33,1,1])` describe a Gaussian. Inconsistency confuses filters that use `mu_0`/`Sigma_0` directly.

**Fix:**
1. In `__init__`, precompute `self._L_Sigma0 = tf.linalg.cholesky(self._Sigma_0)` after the `_Sigma_0` assignment.
2. Replace both `sample_initial_state` and `sample_initial_state_batch` with Cholesky Gaussian draws — matching the pattern in every other model.

```python
def sample_initial_state(self, seed):
    z = tf.random.stateless_normal([4], seed=seed, dtype=self.dtype)
    return self._mu_0 + tf.linalg.matvec(self._L_Sigma0, z)

def sample_initial_state_batch(self, n, seed):
    z = tf.random.stateless_normal([n, 4], seed=seed, dtype=self.dtype)
    return self._mu_0 + tf.linalg.matmul(z, self._L_Sigma0, transpose_b=True)
```

**Impact:** `AcousticTrackingModel` is not used by any active experiment config — zero regression.

---

## Change 3 — LinearGaussian: stationary `Sigma_0` when not provided

**File:** `src/models/linear_gaussian.py` (lines 113–116)

**Problem:** When `Sigma_0=None`, the model defaults to `I`. For F=[[0.95]], B=[[0.5]], the true stationary variance is `0.25/(1-0.95²) ≈ 2.56`, so `I` makes the filter overconfident.

**Fix:** When `Sigma_0=None`, solve the discrete Lyapunov equation `P = F P F^T + Q` via `scipy.linalg.solve_discrete_lyapunov`, following the exact same pattern as `StochasticVolatility2DModel`. Fall back to identity if F is unstable (spectral radius ≥ 1).

```python
# Add imports at top:
from scipy import linalg as scipy_linalg
import warnings

# Replace the Sigma_0=None branch:
if Sigma_0 is None:
    F_np = np.array(F, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)
    Q_np = B_np @ B_np.T
    spectral_radius = np.max(np.abs(np.linalg.eigvals(F_np)))
    if spectral_radius < 1.0:
        try:
            P0_np = scipy_linalg.solve_discrete_lyapunov(F_np, Q_np)
            P0_np = 0.5 * (P0_np + P0_np.T)
            self._Sigma_0 = tf.constant(P0_np, dtype=self.dtype)
        except Exception:
            warnings.warn("solve_discrete_lyapunov failed; falling back to identity.", RuntimeWarning)
            self._Sigma_0 = tf.eye(self.nx, dtype=self.dtype)
    else:
        warnings.warn(
            f"F unstable (spectral radius={spectral_radius:.4f}); using identity Sigma_0.",
            RuntimeWarning
        )
        self._Sigma_0 = tf.eye(self.nx, dtype=self.dtype)
else:
    self._Sigma_0 = tf.constant(Sigma_0, dtype=self.dtype)
```

**Impact:** All YAML configs explicitly pass `Sigma_0`, so they go through the `else` branch — zero regression. Only code that omits `Sigma_0` is affected (gets a better default).

---

## Change 4 — RangeBearing: default `mu_0` further from sensor

**File:** `src/models/range_bearing.py` (line 103)

**Problem:** Default `mu_0=[1,1]` is only 1.41 units from the sensor at origin. Bearing Jacobians are sensitive at short range.

**Fix:** Change default to `[5.0, 5.0]` (range ≈ 7.07 from origin). One-line change.

```python
mu_0_np = mu_0 if mu_0 is not None else np.array([5.0, 5.0])
```

Update docstring accordingly:
```
mu_0: Initial state mean [x_0, y_0]. If None, uses [5.0, 5.0]
      (range ≈ 7.07 from sensor at origin, giving stable Jacobian SNR).
```

**Impact:** Callers that rely on the default `[1,1]` will get a slightly different starting point. Low risk; verify range-bearing tests still pass.

---

## Models Left Unchanged

| Model | Reason |
|-------|--------|
| Kitagawa | `initial_var=5.0` is canonical (Andrieu et al. 2010) |
| StochasticVolatility 1D | Already uses exact stationary variance `sigma^2/(1-alpha^2)` |
| StochasticVolatility 2D | Already solves discrete Lyapunov equation |
| TwoSensorBearing | Anisotropic prior `[[1000,0],[0,2]]` is intentional (stiffness benchmark) |
| AcousticTrackingFull | Uses paper values exactly (Li & Coates Section V-A1) |
| CubicSensor | `initial_var=5.0` is close to stationary variance (~5.26) |

---

## Implementation Order

| Step | File | Complexity | Risk |
|------|------|------------|------|
| 1 | `lorenz96.py` — property swap | Trivial (1 line) | Minimal |
| 2 | `range_bearing.py` — default value | Trivial (1 line) | Minimal |
| 3 | `acoustic_tracking.py` — sampling refactor | Low (3 edits) | Low (model unused in experiments) |
| 4 | `linear_gaussian.py` — Lyapunov Sigma_0 | Medium (scipy import + conditional logic) | Low (conditional on Sigma_0=None) |

---

## Verification Checklist

After all changes:

1. `LinearGaussianModel(F=[[0.95]], B=[[0.5]], H=[[1.0]], D=[[0.3]])` → `Sigma_0 ≈ [[2.564]]`
2. `LinearGaussianModel(..., Sigma_0=[[1.0]])` → `Sigma_0 == [[1.0]]` (backward compat preserved)
3. `Lorenz96Model(state_dim=10).mu_0` is NOT `[8, 8, ..., 8]`
4. `AcousticTrackingModel().sample_initial_state_batch(1000, seed)` → sample mean ≈ `[20, 20, 0, 0]`, sample var ≈ `[33.33, 33.33, 1, 1]`
5. `RangeBearingModel().mu_0.numpy()` == `[5.0, 5.0]`
6. Run full test suite: `pytest code/tests/ -v`
