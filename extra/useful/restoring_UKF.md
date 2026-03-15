# Restoring UKF as Pluggable Global Filter in Flow Filters

## Context

The numpy backup (`extra/code_backup_np/`) had a `filter_type: str = 'ekf'` parameter in all flow filter constructors, allowing the user to choose EKF or UKF for covariance guidance. The TensorFlow rewrite removed this flexibility -- all flow filters now hardcode `ExtendedKalmanFilter`. This plan restores the pluggable design while respecting TensorFlow performance constraints.

## Architecture Overview

### What the global filter provides to flow filters

The global filter (EKF or UKF) provides two things:
1. **Predicted mean** (`eta_bar_0`): used as the linearization point and in b(lambda) computation
2. **Predicted covariance** (`P_{k|k-1}`): used in A(lambda) and b(lambda) flow equations

Both EKF and UKF expose these through the **same interface**: `.mean` (tf.Variable), `.cov` (tf.Variable), `.predict()`, `.update()`. The swap is straightforward for the global filter.

### What the global filter does NOT provide

The observation Jacobian `H = dh/dx` used in `compute_flow_params*()` is computed **directly from the model** (`model.observation_jacobian()`), NOT from the global filter. This means:
- Swapping EKF -> UKF as the global filter does **not** affect flow parameter computation
- The flow equations inherently require H -- this is a property of the Daum-Huang algorithm, not the filter choice

### Two covariance tracking mechanisms in the codebase

| Mechanism | Used by | What it does | UKF complexity |
|-----------|---------|-------------|----------------|
| **Single global filter** | edh_invertible, edh_flow, edh_flow_global, ledh_flow, stochastic_edh, stochastic_edh_paper, sde_local_correction | One EKF runs alongside particles, provides P and mean | **Low** -- just swap EKF for UKF instance |
| **Batched per-particle filter** | ledh_invertible (and its HMC/CSMC variants) | N separate EKF (mean,cov) pairs, one per particle, via `batched_ekf_predict/update` | **High** -- need new `batched_ukf.py` with batched sigma points |

---

## Step-by-Step Plan

### Step 1: Add `filter_type` parameter to all flow filter constructors

**Pattern** (from numpy backup):
```python
def __init__(self, model, ..., filter_type: str = 'ekf', ...):
    self.filter_type = filter_type
```

**Files to modify** (constructors only):

| File | Class | Current constructor | Change |
|------|-------|-------------------|--------|
| `src/filters/particle/edh_invertible.py` | `EDHParticleFlowFilter` | No filter_type param | Add `filter_type: str = 'ekf'` |
| `src/filters/particle/edh_flow.py` | `ExactDaumHuangFlow` | No filter_type param | Add `filter_type: str = 'ekf'` |
| `src/filters/particle/edh_flow_global.py` | `ExactDaumHuangFlowGlobal` | No filter_type param | Add `filter_type: str = 'ekf'` |
| `src/filters/particle/ledh_flow.py` | `LocalExactDaumHuangFlow` | No filter_type param | Add `filter_type: str = 'ekf'` |
| `src/filters/particle/ledh_invertible.py` | `LEDHParticleFlowFilter` | No filter_type param | Add `filter_type: str = 'ekf'` |
| `src/filters/particle/ledh_invertible_hmc.py` | (inherits LEDH) | Pass through `**filter_kwargs` | Ensure filter_type passes through |
| `src/filters/particle/ledh_invertible_csmc.py` | (inherits LEDH) | Pass through `**filter_kwargs` | Ensure filter_type passes through |
| `src/filters/particle/ledh_invertible_bimodal.py` | (inherits LEDH) | Pass through `**filter_kwargs` | Ensure filter_type passes through |

**Child classes that inherit** (no constructor change needed, they receive it via `**filter_kwargs` or `super().__init__()`):
- `stochastic_edh.py` (inherits from `edh_flow.py`)
- `stochastic_edh_paper.py` (inherits from `stochastic_edh.py`)
- `sde_local_correction.py` (inherits from `stochastic_edh.py`)

**Imports to add** in each modified file:
```python
from ..kalman.unscented_kalman import UnscentedKalmanFilter
```

### Step 2: Create a shared factory function for global filter instantiation

**New file: `src/filters/kalman/filter_factory.py`**

This avoids duplicating the if/elif/else logic in every flow filter.

```python
"""Factory function for creating Kalman filter instances."""

from .extended_kalman import ExtendedKalmanFilter
from .unscented_kalman import UnscentedKalmanFilter


def create_kalman_filter(filter_type: str, model, mean_0, Sigma_0, **kwargs):
    """
    Create a Kalman filter instance based on filter_type string.

    Args:
        filter_type: 'ekf' or 'ukf'
        model: StateSpaceModel instance
        mean_0: Initial mean (numpy array or tf.Tensor)
        Sigma_0: Initial covariance (numpy array or tf.Tensor)
        **kwargs: Additional kwargs (e.g., alpha, beta, kappa for UKF)

    Returns:
        Filter instance with .mean, .cov, .predict(), .update() interface
    """
    if filter_type == 'ekf':
        return ExtendedKalmanFilter(
            model, mean_0=mean_0, Sigma_0=Sigma_0,
            sample_initial_mean=False
        )
    elif filter_type == 'ukf':
        # Extract UKF-specific params, use defaults if not provided
        ukf_kwargs = {
            k: kwargs[k] for k in ('alpha', 'beta', 'kappa')
            if k in kwargs
        }
        return UnscentedKalmanFilter(
            model, mean_0=mean_0, Sigma_0=Sigma_0,
            sample_initial_mean=False,
            **ukf_kwargs
        )
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}. Use 'ekf' or 'ukf'.")
```

### Step 3: Replace hardcoded EKF instantiation in flow filters that use a single global filter

For each flow filter that uses a single global EKF, replace the hardcoded `ExtendedKalmanFilter(...)` call with `create_kalman_filter(self.filter_type, ...)`.

#### 3a. `edh_invertible.py` (lines 111-117)

**Current:**
```python
from ..kalman.extended_kalman import ExtendedKalmanFilter
# ...
def _create_filter(self, initial_mean, initial_cov):
    initial_mean_np = to_numpy(initial_mean)
    initial_cov_np = to_numpy(initial_cov)
    return ExtendedKalmanFilter(self.model, mean_0=initial_mean_np, Sigma_0=initial_cov_np,
                                sample_initial_mean=False)
```

**Change to:**
```python
from ..kalman.filter_factory import create_kalman_filter
# ...
def _create_filter(self, initial_mean, initial_cov):
    initial_mean_np = to_numpy(initial_mean)
    initial_cov_np = to_numpy(initial_cov)
    return create_kalman_filter(self.filter_type, self.model,
                                mean_0=initial_mean_np, Sigma_0=initial_cov_np)
```

No other changes needed -- the rest of the code uses `self.global_filter.predict()`, `.update()`, `.mean`, `.cov` which are the same for both EKF and UKF.

#### 3b. `edh_flow.py` (lines 139-151)

**Current:**
```python
from ..kalman.extended_kalman import ExtendedKalmanFilter
# ...
self.global_filter = ExtendedKalmanFilter(
    self.model, mean_0=ensemble_mean_np, Sigma_0=initial_cov_emp_np)
```

**Change to:**
```python
from ..kalman.filter_factory import create_kalman_filter
# ...
self.global_filter = create_kalman_filter(
    self.filter_type, self.model, mean_0=ensemble_mean_np, Sigma_0=initial_cov_emp_np)
```

#### 3c. `edh_flow_global.py` (lines 138-145)

Same pattern as 3b.

#### 3d. `ledh_flow.py` (lines 148-156)

Same pattern as 3b.

### Step 4: Handle the batched per-particle case (ledh_invertible.py)

This is the most complex part. `ledh_invertible.py` uses `batched_ekf_predict()` and `batched_ekf_update()` which are EKF-specific (they use Jacobians). A UKF equivalent needs batched sigma point operations.

#### 4a. Create `src/filters/kalman/batched_ukf.py`

**New file** implementing batched UKF predict and update for N (mean, cov) pairs simultaneously.

**Key design considerations for TensorFlow optimization:**

1. **Sigma point generation must be batched**: For N particles, each with `state_dim` d, we need `N * (2d+1)` sigma points. This is a `(N, 2d+1, d)` tensor.

2. **Use `tf.linalg.cholesky` on batched covariances**: `safe_cholesky` already supports batch dimensions `(N, d, d)`.

3. **Propagation through model functions**: Need `model.state_transition_mean_batch()` on the `(N*(2d+1), d)` reshaped sigma points, then reshape back to `(N, 2d+1, d)`.

4. **Weighted statistics**: Batched weighted mean and covariance computation using `tf.einsum`.

```python
"""
Batched UKF operations for per-particle filters.

Uses batched sigma point generation and propagation for N (mean, cov) pairs.
"""

import tensorflow as tf
from typing import Tuple
from ...utils.linalg import symmetrize, safe_cholesky, safe_inv


def _compute_ukf_weights(state_dim, alpha=1e-3, beta=2.0, kappa=0.0, dtype=tf.float64):
    """Pre-compute UKF weights (called once, reused across timesteps)."""
    n = state_dim
    lambda_ = alpha**2 * (n + kappa) - n
    W_m_0 = lambda_ / (n + lambda_)
    W_c_0 = W_m_0 + (1 - alpha**2 + beta)
    W_i = 1.0 / (2 * (n + lambda_))

    weights_mean = tf.constant(
        [W_m_0] + [W_i] * (2 * n), dtype=dtype
    )  # (2n+1,)
    weights_cov = tf.constant(
        [W_c_0] + [W_i] * (2 * n), dtype=dtype
    )  # (2n+1,)
    return weights_mean, weights_cov, lambda_


@tf.function
def _batched_sigma_points(means, covs, lambda_, state_dim):
    """
    Generate sigma points for N (mean, cov) pairs.

    Args:
        means: (N, d)
        covs: (N, d, d)
        lambda_: scalar
        state_dim: int d

    Returns:
        sigma_points: (N, 2d+1, d)
    """
    n = state_dim
    scale = tf.cast(n + lambda_, covs.dtype)
    sqrt_cov = safe_cholesky(scale * covs)  # (N, d, d)

    # Build sigma points: mean, mean + cols, mean - cols
    # means[:, tf.newaxis, :] -> (N, 1, d)
    center = means[:, tf.newaxis, :]  # (N, 1, d)

    # sqrt_cov columns: (N, d, d) -> transpose to (N, d, d)
    # We want the i-th column for each particle: sqrt_cov[:, :, i]
    # tf.linalg.matrix_transpose gives (N, d, d) with rows as columns
    cols = tf.linalg.matrix_transpose(sqrt_cov)  # (N, d, d) -> columns along axis=1

    positive = means[:, tf.newaxis, :] + cols  # (N, d, d) -- d sigma points
    negative = means[:, tf.newaxis, :] - cols  # (N, d, d)

    # Stack: (N, 2d+1, d)
    sigma_points = tf.concat([center, positive, negative], axis=1)
    return sigma_points


@tf.function
def batched_ukf_predict(model, means, covs, weights_mean, weights_cov, lambda_, state_dim):
    """
    Batched UKF prediction for N particles.

    Args:
        model: StateSpaceModel
        means: (N, d)
        covs: (N, d, d)
        weights_mean: (2d+1,) pre-computed UKF mean weights
        weights_cov: (2d+1,) pre-computed UKF covariance weights
        lambda_: scalar
        state_dim: int

    Returns:
        mean_pred: (N, d)
        cov_pred: (N, d, d)
    """
    N = tf.shape(means)[0]
    n_sigma = 2 * state_dim + 1

    # Generate sigma points: (N, 2d+1, d)
    sigma_pts = _batched_sigma_points(means, covs, lambda_, state_dim)

    # Reshape to (N*(2d+1), d) for batch model call
    flat_pts = tf.reshape(sigma_pts, [N * n_sigma, state_dim])

    # Propagate through state transition
    flat_pred = model.state_transition_mean_batch(flat_pts)  # (N*(2d+1), d)

    # Reshape back to (N, 2d+1, d)
    pred_pts = tf.reshape(flat_pred, [N, n_sigma, state_dim])

    # Weighted mean: sum_j w_m[j] * pred_pts[:, j, :]
    mean_pred = tf.einsum('j,njd->nd', weights_mean, pred_pts)  # (N, d)

    # Weighted covariance: sum_j w_c[j] * (pred_pts - mean_pred) outer product
    diff = pred_pts - mean_pred[:, tf.newaxis, :]  # (N, 2d+1, d)
    cov_pred = tf.einsum('j,nji,njk->nik', weights_cov, diff, diff)  # (N, d, d)

    # Add process noise Q
    Q = model.state_transition_cov_batch(means)
    Q = tf.cast(Q, covs.dtype)
    if len(Q.shape) == 2:
        cov_pred = cov_pred + tf.expand_dims(Q, 0)
    else:
        cov_pred = cov_pred + Q

    cov_pred = symmetrize(cov_pred)
    return mean_pred, cov_pred


@tf.function
def batched_ukf_update(model, means, covs, observation, weights_mean, weights_cov, lambda_, state_dim):
    """
    Batched UKF update for N particles.

    Args:
        model: StateSpaceModel
        means: (N, d) predicted means
        covs: (N, d, d) predicted covariances
        observation: (obs_dim,)
        weights_mean: (2d+1,)
        weights_cov: (2d+1,)
        lambda_: scalar
        state_dim: int

    Returns:
        mean_updated: (N, d)
        cov_updated: (N, d, d)
    """
    N = tf.shape(means)[0]
    n_sigma = 2 * state_dim + 1
    obs_dim = tf.shape(observation)[0]

    # Generate sigma points: (N, 2d+1, d)
    sigma_pts = _batched_sigma_points(means, covs, lambda_, state_dim)

    # Reshape to (N*(2d+1), d) for batch model call
    flat_pts = tf.reshape(sigma_pts, [N * n_sigma, state_dim])

    # Propagate through observation model
    flat_obs = model.observation_function_batch(flat_pts)  # (N*(2d+1), obs_dim)

    # Reshape back: (N, 2d+1, obs_dim)
    obs_pts = tf.reshape(flat_obs, [N, n_sigma, -1])

    # Predicted observation mean
    y_pred = tf.einsum('j,njm->nm', weights_mean, obs_pts)  # (N, obs_dim)

    # Innovation covariance S
    diff_y = obs_pts - y_pred[:, tf.newaxis, :]  # (N, 2d+1, obs_dim)
    S = tf.einsum('j,nji,njk->nik', weights_cov, diff_y, diff_y)  # (N, obs_dim, obs_dim)
    R = tf.cast(model.observation_cov(means[0]), covs.dtype)
    S = S + tf.expand_dims(R, 0)

    # Cross-covariance P_xy
    diff_x = sigma_pts - means[:, tf.newaxis, :]  # (N, 2d+1, d)
    P_xy = tf.einsum('j,nji,njk->nik', weights_cov, diff_x, diff_y)  # (N, d, obs_dim)

    # Kalman gain K = P_xy @ S^{-1}
    K = tf.matmul(P_xy, safe_inv(S))  # (N, d, obs_dim)

    # Innovation
    innovation = tf.expand_dims(observation, 0) - y_pred  # (N, obs_dim)

    # Update mean
    mean_updated = means + tf.einsum('nij,nj->ni', K, innovation)  # (N, d)

    # Update covariance: P - K @ S @ K^T
    KS = tf.matmul(K, S)
    K_T = tf.linalg.matrix_transpose(K)
    cov_updated = covs - tf.matmul(KS, K_T)
    cov_updated = symmetrize(cov_updated)

    return mean_updated, cov_updated
```

**TensorFlow performance considerations:**
- `_batched_sigma_points` is a single `@tf.function` that uses `safe_cholesky` on the full (N, d, d) batch -- no Python loop
- Sigma points are flattened to `(N*(2d+1), d)` for a single `model.state_transition_mean_batch()` call -- avoids N separate calls
- `tf.einsum` handles all weighted statistics in one operation
- Memory: For N=200 particles, d=3 state dims, this creates `200 * 7 = 1400` sigma points -- trivial

#### 4b. Modify `ledh_invertible.py` to support UKF

**Changes needed:**

1. Add `filter_type` parameter to constructor
2. Conditionally import and use `batched_ekf_*` or `batched_ukf_*` functions
3. If `filter_type == 'ukf'`, pre-compute UKF weights once in `initialize()`
4. Store UKF-specific parameters (alpha, beta, kappa, lambda_)

```python
# In __init__:
def __init__(self, ..., filter_type: str = 'ekf', ukf_params: dict = None, ...):
    self.filter_type = filter_type
    self.ukf_params = ukf_params or {}  # alpha, beta, kappa

# In initialize():
if self.filter_type == 'ukf':
    from ..kalman.batched_ukf import _compute_ukf_weights
    alpha = self.ukf_params.get('alpha', 1e-3)
    beta = self.ukf_params.get('beta', 2.0)
    kappa = self.ukf_params.get('kappa', 0.0)
    self._ukf_weights_mean, self._ukf_weights_cov, self._ukf_lambda = (
        _compute_ukf_weights(self.state_dim, alpha, beta, kappa, self.dtype)
    )

# In predict():
if self.filter_type == 'ekf':
    eta_bar_0_tf, cov_pred_tf = batched_ekf_predict(
        self.model, self.particles.value(), self.particle_covs.value()
    )
elif self.filter_type == 'ukf':
    eta_bar_0_tf, cov_pred_tf = batched_ukf_predict(
        self.model, self.particles.value(), self.particle_covs.value(),
        self._ukf_weights_mean, self._ukf_weights_cov, self._ukf_lambda, self.state_dim
    )

# In update() (batched_ekf_update call):
if self.filter_type == 'ekf':
    _, cov_updated = batched_ekf_update(
        self.model, self.eta_bar_0.value(), self.particle_covs.value(), y
    )
elif self.filter_type == 'ukf':
    _, cov_updated = batched_ukf_update(
        self.model, self.eta_bar_0.value(), self.particle_covs.value(), y,
        self._ukf_weights_mean, self._ukf_weights_cov, self._ukf_lambda, self.state_dim
    )
```

#### 4c. Modify LEDH HMC/CSMC/bimodal variants

These inherit from `LEDHParticleFlowFilter` and pass through `**filter_kwargs`. Verify that:
- `filter_type` is passed through to the parent `__init__`
- No hardcoded EKF references exist in these files

**Files to check:**
- `src/filters/particle/ledh_invertible_hmc.py`
- `src/filters/particle/ledh_invertible_csmc.py`
- `src/filters/particle/ledh_invertible_bimodal.py`

### Step 5: Update Hydra/YAML config support

If experiments use Hydra configs (`.yaml` files) to instantiate filters, add `filter_type` to the config schema.

**Files to check:**
- `src/experiments/run_experiment.py`
- `src/experiments/run_dpf_experiment.py`
- Any `conf/` YAML files

Add to filter config:
```yaml
filter:
  filter_type: ekf  # or ukf
  # UKF-specific (only used when filter_type=ukf):
  ukf_alpha: 0.001
  ukf_beta: 2.0
  ukf_kappa: 0.0
```

### Step 6: Update `__init__.py` exports

**File: `src/filters/kalman/__init__.py`**
- Export `create_kalman_filter` from the new factory module
- Export `batched_ukf_predict`, `batched_ukf_update`

### Step 7: Write tests

#### 7a. Unit test: `batched_ukf.py`

**New file: `tests/unit/test_batched_ukf.py`**

Test against the scalar UKF:
```python
def test_batched_ukf_predict_matches_scalar():
    """batched_ukf_predict with N=1 should match UnscentedKalmanFilter._predict_step."""

def test_batched_ukf_update_matches_scalar():
    """batched_ukf_update with N=1 should match UnscentedKalmanFilter._update_step."""

def test_batched_ukf_predict_shapes():
    """Verify output shapes for various N, state_dim combinations."""
```

#### 7b. Integration test: flow filter with UKF

**New file: `tests/filters/test_flow_filters_ukf.py`**

```python
def test_edh_with_ukf_runs():
    """EDH flow filter with filter_type='ukf' produces reasonable output."""

def test_ledh_invertible_with_ukf_runs():
    """LEDH invertible with filter_type='ukf' produces reasonable output."""

def test_ukf_vs_ekf_similar_results():
    """UKF and EKF produce similar (not necessarily identical) results on linear-Gaussian model."""
```

#### 7c. Verify existing tests still pass with default `filter_type='ekf'`

Run the full test suite to confirm no regressions:
```bash
cd code && python -m pytest tests/ -v
```

---

## Summary of Files to Create/Modify

### New files:
| File | Purpose |
|------|---------|
| `src/filters/kalman/filter_factory.py` | Shared factory function for EKF/UKF instantiation |
| `src/filters/kalman/batched_ukf.py` | Batched UKF predict/update for per-particle covariances |
| `tests/unit/test_batched_ukf.py` | Unit tests for batched UKF |
| `tests/filters/test_flow_filters_ukf.py` | Integration tests for flow filters with UKF |

### Modified files:
| File | Change |
|------|--------|
| `src/filters/particle/edh_invertible.py` | Add `filter_type` param, use `create_kalman_filter` |
| `src/filters/particle/edh_flow.py` | Add `filter_type` param, use `create_kalman_filter` |
| `src/filters/particle/edh_flow_global.py` | Add `filter_type` param, use `create_kalman_filter` |
| `src/filters/particle/ledh_flow.py` | Add `filter_type` param, use `create_kalman_filter` |
| `src/filters/particle/ledh_invertible.py` | Add `filter_type` + `ukf_params`, conditional batched EKF/UKF |
| `src/filters/particle/ledh_invertible_hmc.py` | Verify `filter_type` passes through |
| `src/filters/particle/ledh_invertible_csmc.py` | Verify `filter_type` passes through |
| `src/filters/particle/ledh_invertible_bimodal.py` | Verify `filter_type` passes through |
| `src/filters/kalman/__init__.py` | Add new exports |

### Files NOT modified (no changes needed):
| File | Why |
|------|-----|
| `src/utils/flow_params.py` | Flow params use `model.observation_jacobian*` directly, not the global filter |
| `src/filters/kalman/extended_kalman.py` | EKF implementation unchanged |
| `src/filters/kalman/unscented_kalman.py` | UKF implementation already exists |
| `src/filters/kalman/batched_ekf.py` | Batched EKF unchanged, batched UKF is a new file |
| `src/filters/particle/stochastic_edh.py` | Inherits from edh_flow, gets filter_type automatically |
| `src/filters/particle/stochastic_edh_paper.py` | Inherits, gets filter_type automatically |
| `src/filters/particle/sde_local_correction.py` | Inherits, gets filter_type automatically |

---

## Verification

1. **Unit tests**: `python -m pytest tests/unit/test_batched_ukf.py -v`
2. **Integration tests**: `python -m pytest tests/filters/test_flow_filters_ukf.py -v`
3. **Regression**: `python -m pytest tests/ -v` (full suite, all existing tests pass)
4. **Quick smoke test** on linear-Gaussian model:
   ```python
   from src.filters.particle.edh_invertible import EDHParticleFlowFilter
   model = LinearGaussianModel(...)
   filter_ekf = EDHParticleFlowFilter(model, filter_type='ekf')
   filter_ukf = EDHParticleFlowFilter(model, filter_type='ukf')
   # Both should produce similar RMSE on the same observation sequence
   ```

---

## Execution Order

1. **Step 2** first (create `filter_factory.py`) -- no dependencies
2. **Step 4a** next (create `batched_ukf.py`) -- no dependencies
3. **Step 1 + Step 3** together (add `filter_type` param and use factory in all global-filter flow filters)
4. **Step 4b** (modify `ledh_invertible.py` for batched UKF)
5. **Step 4c** (verify HMC/CSMC/bimodal pass-through)
6. **Step 6** (update exports)
7. **Step 5** (config support, if applicable)
8. **Step 7** (tests)
