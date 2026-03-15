# Augmented UKF Implementation Plan (Algorithm 5.15)

## Background

The 2D Stochastic Volatility model has **non-additive observation noise**:

```
y_t = b · x₁ + exp(x₂/2) · v_t,    v_t ~ N(0, 1)
```

The observation noise `v` enters multiplicatively through `exp(x₂/2)`, making it state-dependent. Current EKF and standard UKF (Alg 5.14) treat observation noise as additive:

```
S = H P H^T + R(x)    ← R evaluated at a single point estimate
```

This misses the coupling between uncertainty in x₂ and observation noise scaling. The augmented UKF (Alg 5.15, Särkkä Ch. 5) solves this by creating sigma points in the joint (state, noise) space.

### Current Results on SV2D (RMSE)

| Filter | RMSE | Note |
|--------|------|------|
| Bootstrap PF (5000 particles) | 1.12 | Best, brute force |
| LEDH Invertible (500) | 1.44 | Proper importance weights |
| EDH Invertible (500) | 1.52 | Proper importance weights |
| EKF / UKF | 1.77 | Additive noise assumption |
| All flow variants | ~1.77 | Converge to Gaussian approx |

**Goal**: Augmented UKF should improve on 1.77 by properly propagating the x₂-noise interaction through sigma points.

---

## Algorithm: Augmented UKF Update (Alg 5.15)

**Prediction** stays the same as standard UKF (Alg 5.14) since process noise is additive for SV2D: `x_{t+1} = A x_t + e_t`.

**Update** uses augmented sigma points:

1. Define augmented state: x̃ = (x, r), n'' = n_x + n_r
2. Augmented mean: m̃ = (m_pred, 0)
3. Augmented covariance: P̃ = block_diag(P_pred, R_noise)
   - R_noise = I (covariance of raw noise v ~ N(0,1))
4. Generate 2n''+1 sigma points in augmented space
5. Split each sigma point: x-part and r-part
6. Propagate through **full** observation: Ŷ^(i) = h(X^(i), R^(i)) = b·X₁^(i) + exp(X₂^(i)/2)·R^(i)
7. Compute S, P_xy from sigma point statistics — **NO separate +R** (absorbed through augmentation)
8. Standard Kalman update: K, m_updated, P_updated

**Why this helps**: Sigma points explore different (x₂, r) combinations. When x₂ is large, the h(x, r) outputs have larger spread, naturally increasing S. When x₂ is small, the spread is smaller. This coupling is exactly what additive-R UKF misses.

---

## Implementation Plan

### Files to Modify

| File | Change |
|------|--------|
| `src/core/model_base.py` | Add `observation_function_with_noise(x, r)` and batch variant with additive default |
| `src/models/stochastic_volatility_2d.py` | Override with `b*x₁ + exp(x₂/2)*r` and `has_non_additive_obs_noise = True` |

### New Files

| File | Purpose |
|------|---------|
| `src/filters/kalman/augmented_ukf.py` | `AugmentedUnscentedKalmanFilter` class extending `UnscentedKalmanFilter` |
| `configs/filter/augmented_ukf.yaml` | Hydra filter config (same params as UKF: alpha, beta, kappa) |
| `configs/experiment/stochastic_volatility_2d/stochastic_volatility_2d_augmented_ukf.yaml` | Experiment config |

### Wiring Changes

| File | Change |
|------|--------|
| `src/filters/kalman/__init__.py` | Export `AugmentedUnscentedKalmanFilter` |
| `src/experiments/run_experiment.py` | Add `'AugmentedUnscentedKalmanFilter'` to EKF/UKF filter name checks |
| `run_stochastic_volatility_2d_filters.sh` | Add experiment to run list |

---

## Design Decisions

### 1. Inherit from `UnscentedKalmanFilter`

The augmented UKF only differs in the update step. By inheriting:
- `_predict_step()` is reused unchanged (additive process noise)
- `predict()`, `update()`, `filter()`, `reset()` are all inherited
- Only `_update_step()` is overridden with augmented sigma points

### 2. Precompute augmented weights in `__init__`

The augmented dimension n'' = n_x + n_r is fixed for a given model. Weights and lambda are computed once and cached, not recomputed every update step.

### 3. Model interface: `observation_function_with_noise(x, r)`

- Default in base class: `h(x) + r` (additive — backward compatible)
- Override in SV2D: `b*x₁ + exp(x₂/2)*r` (non-additive)
- Batch variant provided for efficiency
- Property `has_non_additive_obs_noise` for introspection (not used by the filter itself, but useful for diagnostics)

### 4. `observation_noise_cov` is the raw noise covariance

For SV2D, `observation_noise_cov = [[1]]` (the variance of v ~ N(0,1)). This is used as the R block in the augmented covariance P̃ = block_diag(P, R). The state-dependent effective variance exp(x₂) is captured automatically through sigma point propagation.

### 5. No `log_marginal_likelihood_tf` override (yet)

The parent's `log_marginal_likelihood_tf` inlines predict/update with standard UKF logic. For HMC parameter estimation, we'd need to override it with augmented update logic. Deferred — not needed for the SV2D experiment.

---

## Expected Outcome

The augmented UKF should produce RMSE between the standard UKF (~1.77) and the invertible particle filters (~1.44). The improvement comes from properly capturing that:
- Large x₂ → large observation noise → observation is less informative → less aggressive update
- Small x₂ → small observation noise → observation is more informative → stronger update

The standard UKF evaluates this at a single point; the augmented UKF samples multiple (x₂, r) combinations via sigma points.

**Limitation**: The augmented UKF still assumes Gaussian posteriors. For the SV2D model, the true posterior over x₂ is non-Gaussian (the observation informs x₂ through the residual *magnitude*, a second-order effect). So we don't expect it to match the particle filter. But it should close part of the gap.
