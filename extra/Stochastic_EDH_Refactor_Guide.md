# Stochastic EDH Flow: Step-by-Step Refactoring Guide

**Goal**: Rewrite `stochastic_edh.py` to inherit from `ExactDaumHuangFlow` (TensorFlow), replacing ~500 lines with ~60 lines.

---

## Why This Works

The paper (Dai & Daum, 2021, Theorem 3.1) proves that for Gaussian prior and linear measurement:

> The posterior distribution p(x, λ) is **identical** for ALL choices of Q (including Q=0).
> Q does not appear in the density function, the mean, or the covariance.

This means the Exact Flow (Q=0) and any stochastic flow (Q>0) produce the **same posterior**. The drift A(λ), b(λ) from the Exact Flow formulas (Eqs. 21-22) are already correct — we just add isotropic noise on top for numerical stability.

Concretely, the Exact Flow SDE with Q=0:
```
dx = [A(λ)x + b(λ)] dλ          (deterministic, edh_flow.py)
```

becomes:
```
dx = [A(λ)x + b(λ)] dλ + √q dW   (stochastic, same A and b)
```

The A, b are **unchanged**. The noise term √q·dW is additive and independent of the drift computation. That's why we can reuse `compute_flow_params` exactly as-is.

### Why NOT re-linearize H at intermediate steps

The deterministic `edh_flow.py` re-linearizes H at each λ step by updating `eta_bar = mean(particles)`. This is a practical heuristic for nonlinear problems.

For the stochastic flow, the paper's derivation (Theorem 2.1) assumes ∇log h is **linear in x**, meaning H is constant. Re-linearizing during the SDE evolution would:
1. Violate assumption (A1) under which the flow was derived
2. Invalidate the stability guarantees (Theorems 5.1-5.3)
3. Make Q state-dependent (through the re-linearized H), contradicting the requirement that Q is independent of x

So: `edh_flow.py` re-linearizes (ODE, heuristic, fine). `stochastic_edh.py` does NOT (SDE, theory-correct).

---

## Step-by-Step Instructions

### Step 1: Update imports

Replace the current imports with:

```python
import tensorflow as tf
import numpy as np
from typing import Optional, Callable, Dict, Any
from .edh_flow import ExactDaumHuangFlow
from ...utils.flow_params import compute_flow_params
from ...utils.ode_solvers import euler_step
```

**Why**: We inherit everything from `ExactDaumHuangFlow`. No need for NumPy flow_base, scipy, etc.

### Step 2: Slim down `__init__`

```python
class StochasticEDHFlow(ExactDaumHuangFlow):
    """
    Stochastic EDH particle flow filter (TensorFlow).

    Extends the deterministic Exact Flow with isotropic diffusion:
        dx = [A(λ)x + b(λ)]dλ + √(q·I) dW_λ

    A(λ) and b(λ) are identical to the Exact Flow (Eqs. 21-22).
    The noise stabilizes numerics without changing the posterior
    (Theorem 3.1, Dai & Daum 2021).

    Key difference from edh_flow: H is NOT re-linearized at
    intermediate λ steps (required by SDE stability theory).
    """

    def __init__(self, model, diffusion_scale: float = 0.001, **kwargs):
        super().__init__(model, **kwargs)
        self.diffusion_scale = diffusion_scale
        self.seed_counter = 0
```

**Why**: Everything else (`n_particles`, `n_lambda_steps`, `filter_type`, `resampling_method`, `debug_mode`, etc.) is handled by the parent. We only add `diffusion_scale`.

### Step 3: No need to override `initialize` or `predict`

The parent's `initialize` and `predict` already do exactly what we need:
- `initialize`: creates particles, sets up EKF/UKF, computes `predicted_cov`
- `predict`: updates EKF mean from ensemble, runs EKF predict, stores `eta_bar_0`, propagates particles

These are **identical** for deterministic and stochastic flows. The stochastic part only applies during the update (flow) step.

### Step 4: Override `update` — the only real change

```python
def update(self, y: np.ndarray):
    """
    Stochastic flow from λ=0 to λ=1.

    Same as ExactDaumHuangFlow.update() except:
    1. H is fixed at η̄_0 (no re-linearization)
    2. Euler-Maruyama integration (adds √q·dW noise)
    """
    # --- Setup (same as parent) ---
    observation = tf.constant(y, dtype=tf.float32)
    P_tf = tf.constant(self.predicted_cov, dtype=tf.float32)
    R_tf = tf.constant(self.model.observation_noise_cov, dtype=tf.float32)
    eta_bar_0_tf = tf.constant(self.eta_bar_0, dtype=tf.float32)
    R_inv_tf = tf.linalg.inv(R_tf)

    particles_flow = self.particles.value()
    lambda_val = 0.0

    # --- Flow loop ---
    for i in range(self.n_lambda_steps):
        d_lambda = self.lambda_steps[i]
        lambda_val += d_lambda

        lambda_val_tf = tf.constant(lambda_val, dtype=tf.float32)

        # Compute A(λ), b(λ) using Exact Flow formulas
        # KEY DIFFERENCE: linearize at eta_bar_0 (fixed), NOT at flowing mean
        A, b = compute_flow_params(
            self.model, eta_bar_0_tf, lambda_val_tf, observation,
            P_tf, R_tf, R_inv_tf, eta_bar_0_tf, self.state_dim
        )

        # Deterministic drift: same as parent
        d_lambda_tf = tf.constant(d_lambda, dtype=tf.float32)
        particles_flow = euler_step(
            particles_flow, self._compute_drift, d_lambda_tf, A, b
        )

        # SDE noise: √(q · dλ) · dW
        if self.diffusion_scale > 0:
            seed = tf.constant([self.seed_counter, i], dtype=tf.int32)
            noise = tf.random.stateless_normal(
                tf.shape(particles_flow), seed=seed, dtype=particles_flow.dtype
            )
            particles_flow = particles_flow + noise * tf.sqrt(
                tf.constant(self.diffusion_scale * d_lambda, dtype=tf.float32)
            )

    self.seed_counter += 1

    # --- Finalize (same as parent) ---
    self.particles.assign(particles_flow)
    self.global_filter.update(y)
```

**Why each part works**:

| Line | What it does | Why |
|------|-------------|-----|
| `eta_bar_0_tf` as linearization point | Fixes H at prior mean | Paper requires constant H for SDE stability (Assumption A1) |
| `compute_flow_params(...)` | Computes Exact Flow A, b | Theorem 4.1.1: Exact Flow = stochastic flow with Q=0. Same A, b. |
| `euler_step(...)` | Deterministic drift part | The [Ax + b]dλ term, identical to parent |
| `noise * sqrt(q * dλ)` | SDE diffusion part | Euler-Maruyama discretization of √Q dW where Q = qI |
| `stateless_normal` with seed | Reproducible noise | Each (step, timestep) gets a unique seed |

### Step 5: Remove the `StiffnessMitigationSolver` class

The optimal schedule solver can be kept as a separate utility if needed, but it should NOT be part of `StochasticEDHFlow`. If you want to support it later, pass `schedule_func` as a parent kwarg and modify the lambda schedule in the parent.

### Step 6: Delete methods that are no longer needed

These are all inherited from the parent and should NOT be in the child class:
- `initialize` — inherited
- `predict` — inherited
- `_compute_jacobian` — not used (parent calls `model.observation_jacobian` inside `compute_flow_params`)
- `_compute_covariance` — not used (parent uses EKF covariance)
- `_compute_drift_coefficients` — replaced by `compute_flow_params`
- `_compute_drift` — inherited (same `particles @ A.T + b`)
- `_get_schedule_values` — not needed (lambda schedule from parent's `_generate_lambda_steps`)

---

## Before and After Comparison

### Linearization point at each λ step

**Parent (`edh_flow.py`)**:
```python
# Line 293: re-linearizes at flowing mean
A, b = compute_flow_params(self.model, eta_bar, ...)
# Line 302: update linearization point
eta_bar = tf.reduce_mean(particles_flow, axis=0)
```

**Child (`stochastic_edh.py`)**:
```python
# Always linearize at η̄_0 (fixed)
A, b = compute_flow_params(self.model, eta_bar_0_tf, ...)
# No eta_bar update
```

### Integration step

**Parent**:
```python
particles_flow = euler_step(particles_flow, self._compute_drift, d_lambda_tf, A, b)
```

**Child**:
```python
particles_flow = euler_step(particles_flow, self._compute_drift, d_lambda_tf, A, b)
particles_flow = particles_flow + noise * sqrt(q * dλ)   # extra line
```

That's it. One extra line of noise, one change to the linearization point.

---

## Checklist

- [ ] New file imports `ExactDaumHuangFlow` from `.edh_flow`
- [ ] `__init__` calls `super().__init__` with all parent kwargs + adds `diffusion_scale`
- [ ] `update` uses `eta_bar_0_tf` (not flowing mean) as linearization point
- [ ] `update` adds `√(q·dλ)·N(0,I)` noise after each `euler_step`
- [ ] `update` uses `tf.random.stateless_normal` with proper seeds
- [ ] `update` calls `self.global_filter.update(y)` at the end
- [ ] No `_compute_drift_coefficients`, `_compute_covariance`, `_compute_jacobian` methods
- [ ] `StiffnessMitigationSolver` moved to separate file or deleted (see Section below)
- [ ] Config yaml updated: `_target_` points to new class, `diffusion_scale` parameter kept
- [ ] Tests still pass with `diffusion_scale=0.0` (should match parent behavior exactly)

---

## StiffnessMitigationSolver: Integrated (DONE)

### Paper Reference

Dai & Daum, "Stiffness Mitigation in Stochastic Particle Flow Filters" (arXiv:2107.04672, 2021), Section 3.

### Implementation

The solver has been **rewritten in TensorFlow** and integrated directly into `StochasticEDHFlow` as class methods. No separate class or scipy dependency.

**Keyword**: `schedule_mu` (float, default 0.0)
- `schedule_mu = 0.0` → linear schedule β=λ (Remark 3.1)
- `schedule_mu > 0` → solves BVP for optimal β(λ) at each `update()`

**Methods added to StochasticEDHFlow**:

| Method | What it does |
|--------|-------------|
| `_dkappa_dbeta()` | ∂κ/∂β via `tf.linalg.inv/trace` (Remark 3.2) |
| `_rk4_schedule_step()` | RK4 step for [β, β̇] ODE |
| `_shoot()` | Integrate to λ=1, return β(1) for shooting |
| `_compute_optimal_schedule()` | Bisection + final integration |

**Math**: The optimal schedule reparameterizes the flow. Since the general drift is F_opt(λ) = β̇(λ)·F_exact(β(λ)), using A_exact(β) with step size dβ is equivalent. The noise uses dλ (Brownian motion is in λ-time, not β-time).

**Config** (`stochastic_edh.yaml`):
```yaml
schedule_mu: 0.0  # 0.0 = linear, >0 = optimal stiffness-mitigating schedule
```

**Previous issues all resolved**: Missing imports (#7), beta clipping (#8), float cache (#9), narrow bracket (#10), disconnected solver (#11) — all eliminated in the TF rewrite.
