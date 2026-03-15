# Fix LEDH Gradient Quality: Decouple sigma_W from Flow Jacobian

## Context

When running MAP optimization on Kitagawa with LEDH, both `sigma_V` and `sigma_W` receive nearly identical gradients (both pushing upward). `sigma_V` up is correct (true=3.162), but `sigma_W` should go **down** (true=1.0).

**Root cause**: `R = sigma_W^2` enters the flow parameter computation (`A`, `b`) via `S = λ*HPH + R`. Over 29 discrete Euler flow steps, the Jacobian accumulation `log_theta = Σ log|I + dλ*A|` creates a spurious gradient for `sigma_W` that overwhelms the correct gradient from `log p(y|η₁)`.

**Fix**: Apply `tf.stop_gradient` to `R` and `R_inv` when entering the flow parameter computation. The observation likelihood `log p(y|η₁)` still reads `sigma_W` directly via `model.log_observation_prob_batch`, so the correct gradient is preserved.

---

## Changes

### 1. `code/src/filters/particle/ledh_invertible_hmc.py`

#### a) Add parameter to `__init__` (line 48-54)

```python
def __init__(
    self,
    *args,
    stop_gradient_resampling: bool = True,
    stop_gradient_R_in_flow: bool = False,   # NEW
    hmc_resampling_method: Optional[str] = None,
    ...
):
```

Store it:
```python
self.stop_gradient_R_in_flow = stop_gradient_R_in_flow
```

#### b) Compiled path: `_build_compiled_filter()` (line 126+)

Capture the flag at build time (alongside existing `stop_grad` at line 146):
```python
stop_grad_R = self.stop_gradient_R_in_flow
```

Inside `compiled_filter()`, before the while_loop (after line 172):
```python
R_flow = tf.stop_gradient(R) if stop_grad_R else R
R_inv_flow = tf.stop_gradient(R_inv) if stop_grad_R else R_inv
```

In the flow loop body (line 202-204), change:
```python
# BEFORE:
A_batch, b_batch = _flow_params(
    model, eta_bar, lambda_val, y, covs,
    R, R_inv, eta_bar_0, sd, regularization
)
# AFTER:
A_batch, b_batch = _flow_params(
    model, eta_bar, lambda_val, y, covs,
    R_flow, R_inv_flow, eta_bar_0, sd, regularization
)
```

#### c) Eager path: `_run_eager()` (line 364+)

After `R` and `R_inv` are set (line 364-371), add:
```python
R_flow = tf.stop_gradient(R) if self.stop_gradient_R_in_flow else R
R_inv_flow = tf.stop_gradient(R_inv) if self.stop_gradient_R_in_flow else R_inv
```

In the flow loop (line 408-412), change `R` → `R_flow`, `R_inv` → `R_inv_flow`:
```python
A_batch, b_batch = _flow_params(
    self.model, eta_bar, lambda_val, y,
    self.particle_covs.value(),
    R_flow, R_inv_flow, self.eta_bar_0.value(),
    self.state_dim, regularization_tf
)
```

### 2. `code/configs/dpf/map/kitagawa/ledh.yaml`

Add `stop_gradient_R_in_flow: true` to the filter section:
```yaml
filter:
  _target_: src.filters.particle.ledh_invertible_hmc.LEDHParticleFlowFilterHMC
  n_particles: 1000
  n_lambda_steps: 29
  ...
  stop_gradient_R_in_flow: true    # NEW: decouple obs noise from flow Jacobian
```

---

## What is NOT changed

- `flow_params.py` — R is just a tensor input, no modification needed
- `distributions.py` — observation likelihood path untouched, sigma_W gradient preserved
- `bootstrap_pf_hmc.py` — BPF has no flow, not affected
- `batched_ekf.py` — secondary R path (R → EKF update → covs → next timestep flow); less critical, address later if needed

---

## Verification

1. **Kitagawa MAP**: `dpf=map/kitagawa/ledh` — sigma_W should decrease toward 1.0
2. **Linear Gaussian HMC**: LEDH configs — obs_noise_std should still converge near 1.0
3. **Gradient check**: sigma_V and sigma_W gradients should no longer be identical
