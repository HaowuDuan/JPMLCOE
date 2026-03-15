# Plan: Global (Non-Relinearizing) EDH Flow Variant

## Context

The current `compute_flow_params` uses a relinearizing formula with `e = h(x_lin) - H·x_lin` correction, computing `b` with `(z - e)`. The global (non-relinearizing) version should use a different formula where `b` uses `z` directly — no `e` term, no `model.observation_function()` call. The existing `edh_flow_global.py` is currently a broken copy of `edh_flow.py` (still relinearizes). The SDE filter (`stochastic_edh.py`) already avoids relinearization but incorrectly calls `compute_flow_params` which computes the `e` term.

**Formulas:**
- Relinearizing: `b(λ) = (I + 2λA)[(I + λA)P H^T R^{-1} (z - e) + A η̄_0]` where `e = h(x_lin) - H·x_lin`
- Global: `b(λ) = (I + 2λA)[(I + λA)P H^T R^{-1} z + A η̄_0]` — uses `z` directly

## Changes

### 1. Add `compute_flow_params_global` to `flow_params.py`

Add after `compute_flow_params` (~line 98). Takes precomputed `H` instead of `model` + `linearization_point`:

```python
@tf.function
def compute_flow_params_global(
    H, lambda_val, observation, P, R, R_inv, eta_bar_0, state_dim, regularization=None
)
```

Differences from `compute_flow_params`:
- **No `model` param** — takes `H: tf.Tensor` directly (precomputed once by caller)
- **No `linearization_point` param** — H is already evaluated
- **Remove** `h_x = model.observation_function(...)` and `e = h_x - H @ linearization_point` (lines 86-87)
- **Change** `observation - e` → `observation` in the `term1` computation (line 93)
- A(λ) formula unchanged

### 2. Add `compute_flow_params_batch_global` to `flow_params.py`

Add after `compute_flow_params_batch` (~line 204). Takes precomputed `H_batch` instead of `model` + `linearization_points`:

```python
@tf.function
def compute_flow_params_batch_global(
    H_batch, lambda_val, observation, P, R, R_inv, eta_bar_0, state_dim, regularization=None
)
```

Differences from `compute_flow_params_batch`:
- **No `model` param** — takes `H_batch: (N, obs_dim, state_dim)` directly
- **No `linearization_points` param**
- **Remove** `h_batch = model.observation_function_batch(...)`, `Hx = ...`, `e_batch = ...` (lines 173-175)
- **Change** `z_minus_e` → just broadcast `observation` to `(1, obs_dim)` (line 188)

### 3. Fix `edh_flow_global.py` update method

- **Import**: `compute_flow_params` → `compute_flow_params_global`
- **Precompute H once** before flow loop: `H_fixed = self.model.observation_jacobian(eta_bar_0_tf)`
- **Remove line 234**: `eta_bar = eta_bar_0_tf` (no flowing mean needed)
- **Remove line 275**: `eta_bar = tf.reduce_mean(particles_flow, axis=0)` (no relinearization)
- **Replace** all `compute_flow_params(self.model, eta_bar, lambda_val, ...)` calls with `compute_flow_params_global(H_fixed, lambda_val, ...)`
- **Debug block** (line 243): `H = self.model.observation_jacobian(eta_bar)` → use `H_fixed`
- **Update docstrings** to describe global (non-relinearizing) variant

### 4. Fix `stochastic_edh.py` update method

- **Import**: `compute_flow_params` → `compute_flow_params_global`
- **Precompute H once** before flow loop (consolidate with existing `H_tf` on line 189):
  - Move `H_fixed = self.model.observation_jacobian(eta_bar_0)` before the `if q > 0` block
  - Use `H_fixed` in both the score correction and the flow params call
- **Replace** `compute_flow_params(self.model, eta_bar_0, homotopy_param, ...)` (line 217) with `compute_flow_params_global(H_fixed, homotopy_param, ...)`
- **Update docstring** (line 31): already says "Uses re-linearization at the flowing mean" — fix to say uses global/fixed linearization

### 5. Register in `__init__.py`

Add to `code/src/filters/particle/__init__.py`:
```python
from .edh_flow_global import ExactDaumHuangFlowglobal
```
And add `'ExactDaumHuangFlowglobal'` to `__all__`.

## Files to modify

- `code/src/utils/flow_params.py` — add `compute_flow_params_global` and `compute_flow_params_batch_global`
- `code/src/filters/particle/edh_flow_global.py` — fix update method to use global formula, no relinearization
- `code/src/filters/particle/stochastic_edh.py` — switch to `compute_flow_params_global`
- `code/src/filters/particle/__init__.py` — register `ExactDaumHuangFlowglobal`

## Verification

- For a **linear** observation model (h(x) = Hx), `edh_flow_global` and `edh_flow` should produce identical results (since e=0 for linear h)
- `stochastic_edh` with `diffusion_scale=0` should match `edh_flow_global` exactly
- Run existing acoustic tracking or bearing-only experiments comparing `edh_flow` vs `edh_flow_global` vs `stochastic_edh`
