# Stochastic Volatility: Log-Space Mode (1D) + 2D Model Implementation

## What Was Done

### 1. Log-Space Mode for 1D SV Model

**File modified:** `src/models/stochastic_volatility.py`

**Problem:** The original 1D SV model has observation `y_t = β·exp(x_t/2)·v_t`. For EKF/UKF:
- `E[y|x] = 0` → observation Jacobian `H_x = 0`
- Kalman gain `K = P·0·S^{-1} = 0` → **no state correction**
- EKF/UKF run open-loop with no measurement updates (useless)

**Solution:** Log-squared observation transform (Kim, Shephard & Chib 1998):
```
z_t = log(y_t²) = log(β²) + x_t + log(v_t²)
```
where `log(v_t²) ~ log(χ²₁)` approximated as `N(ψ(1/2)+log(2), π²/2)` ≈ `N(-1.2704, 4.9348)`.

This makes the observation **linear in x** with additive (approximate) Gaussian noise:
- `h(x) = log(β²) + x + E[ε]` → `H_x = 1` (nonzero!)
- `R = π²/2` (constant, not state-dependent)

**Changes to `StochasticVolatilityModel`:**
- Constructor: added `log_space: bool = False` and `y_floor: float = 1e-8` params
- `observation_mean(x)`: returns `log(β²) + x + E[ε]` when `log_space=True`
- `observation_cov(x)`: returns `π²/2` (constant) when `log_space=True`
- `observation_jacobian(x)`: returns `[[1.0]]` when `log_space=True`
- `observation_function(x)`: same as `observation_mean` in log-space
- `log_observation_prob(y, x)`: transforms raw `y` to `log(y²)` internally
- `observation_noise_cov`: returns `π²/2` when `log_space=True`
- All batch methods updated similarly
- New method: `transform_observations(observations)` — transforms raw `y` to `log(y²)`

**Key design:** `sample_observation()` is unchanged (always returns raw `y_t`). The transform happens at filter time via `transform_observations()`. This keeps data generation clean.

**Approximation quality:** The `log(χ²₁)` distribution is skewed (not truly Gaussian). This introduces some bias but works well for HMC parameter inference where you need a reasonable likelihood approximation.

### 2. 2D Stochastic Volatility Model

**File created:** `src/models/stochastic_volatility_2d.py`

**Model:**
```
State:   x_{t+1} = A·x_t + e_t,  e_t ~ N(0, Σ)    (x ∈ R²)
Obs:     y_t = b·x_{1,t} + exp(x_{2,t}/2)·v_t      (y ∈ R¹)
```

- `x_1` = level component (enters observation mean)
- `x_2` = log-volatility component (enters observation variance)

**Why EKF/UKF work directly (no log-space trick needed):**
- `E[y|x] = b·x_1` → `H_x = [b, 0] ≠ 0`
- Kalman gain is nonzero → state correction works
- `x_2` information comes indirectly through state-dependent `R = exp(x_2)`

**Constructor parameters:**
- `a1, a2`: diagonal A entries (default: 0.95, 0.91)
- `sigma1, sigma2`: diagonal √Σ entries (default: 0.5, 1.0)
- `b`: observation coefficient (default: 1.0)
- `A`: full 2×2 matrix (overrides `a1, a2` if provided)
- `Sigma`: full 2×2 matrix (overrides `sigma1, sigma2` if provided)

**Stationarity:** Validates `max|eigenvalue(A)| < 1`. Computes stationary `P₀` via `scipy.linalg.solve_discrete_lyapunov(A, Σ)`.

**Sigma validation:** Checks eigenvalues of Σ are positive.

### 3. Experiment Runner Integration

**Files modified:**
- `src/experiments/run_experiment.py` — added observation transform hook after data generation
- `src/experiments/run_dpf_experiment.py` — same hook added

Both check `hasattr(model, 'transform_observations')` and apply the transform before passing observations to the filter.

### 4. Config Files Created

**Model configs:**
- `configs/model/stochastic_volatility_log.yaml` — 1D SV with `log_space: true`
- `configs/model/stochastic_volatility_2d.yaml` — 2D SV (diagonal defaults)

**Experiment configs:**
- `configs/experiment/stochastic_volatility/stochastic_volatility_ekf_log.yaml`
- `configs/experiment/stochastic_volatility/stochastic_volatility_ukf_log.yaml`
- `configs/experiment/stochastic_volatility_2d/stochastic_volatility_2d_ekf.yaml`
- `configs/experiment/stochastic_volatility_2d/stochastic_volatility_2d_ukf.yaml`
- `configs/experiment/stochastic_volatility_2d/stochastic_volatility_2d_pf.yaml`

### 5. Registration

**File modified:** `src/models/__init__.py` — added `StochasticVolatility2DModel` import and export.

### 6. Tests

**File created:** `tests/unit/test_sv_models.py`

Tests cover:
- Log-space mode: state methods unchanged, H_x=1, constant R, transform_observations, EKF/UKF run, gradient differentiability
- 2D SV: dimensions, Lyapunov equation, stationarity validation, observation mean/Jacobian/cov, batch methods, EKF/UKF run, gradient differentiability, full A/Sigma matrices

---

## How to Run

```bash
# Tests
cd code && python -m pytest tests/unit/test_sv_models.py -v

# 1D Log-space EKF experiment
cd code && python -m src.experiments.run_experiment \
  +experiment=stochastic_volatility/stochastic_volatility_ekf_log

# 2D SV EKF experiment
cd code && python -m src.experiments.run_experiment \
  +experiment=stochastic_volatility_2d/stochastic_volatility_2d_ekf
```

---

## How to Undo

### Revert 1D SV log-space changes:
```bash
git checkout -- src/models/stochastic_volatility.py
```

### Remove 2D SV model entirely:
```bash
rm src/models/stochastic_volatility_2d.py
# Then remove the import from src/models/__init__.py
```

### Remove experiment runner hooks:
```bash
git checkout -- src/experiments/run_experiment.py
git checkout -- src/experiments/run_dpf_experiment.py
```

### Remove all new config files:
```bash
rm configs/model/stochastic_volatility_log.yaml
rm configs/model/stochastic_volatility_2d.yaml
rm -rf configs/experiment/stochastic_volatility_2d/
rm configs/experiment/stochastic_volatility/stochastic_volatility_ekf_log.yaml
rm configs/experiment/stochastic_volatility/stochastic_volatility_ukf_log.yaml
```

### Remove tests:
```bash
rm tests/unit/test_sv_models.py
```

---

## Files Changed (Summary)

| File | Action | Description |
|------|--------|-------------|
| `src/models/stochastic_volatility.py` | Modified | Added `log_space` mode, `transform_observations()` |
| `src/models/stochastic_volatility_2d.py` | **New** | 2D SV model with level + log-vol |
| `src/models/__init__.py` | Modified | Added `StochasticVolatility2DModel` |
| `src/experiments/run_experiment.py` | Modified | Added observation transform hook |
| `src/experiments/run_dpf_experiment.py` | Modified | Added observation transform hook |
| `configs/model/stochastic_volatility_log.yaml` | **New** | 1D SV log-space config |
| `configs/model/stochastic_volatility_2d.yaml` | **New** | 2D SV model config |
| `configs/experiment/stochastic_volatility/stochastic_volatility_ekf_log.yaml` | **New** | EKF + log-space experiment |
| `configs/experiment/stochastic_volatility/stochastic_volatility_ukf_log.yaml` | **New** | UKF + log-space experiment |
| `configs/experiment/stochastic_volatility_2d/stochastic_volatility_2d_ekf.yaml` | **New** | 2D SV + EKF experiment |
| `configs/experiment/stochastic_volatility_2d/stochastic_volatility_2d_ukf.yaml` | **New** | 2D SV + UKF experiment |
| `configs/experiment/stochastic_volatility_2d/stochastic_volatility_2d_pf.yaml` | **New** | 2D SV + BPF experiment |
| `tests/unit/test_sv_models.py` | **New** | Tests for both models |
| `code/useful/sv_log_space_and_2d_implementation.md` | **New** | This document |

---

## Future Work (Not Implemented)

1. **EKF II / Augmented EKF** — mathematically equivalent to standard EKF for non-additive noise; only useful with augmented UKF
2. **Augmented UKF** — generates sigma points in `[x, q, r]` space; best accuracy for multiplicative noise models
3. **Statistical Linearization Filter (SLF)** — awaiting pseudocode
4. **Full (non-diagonal) A and Σ for HMC inference** — 2D SV model already supports full matrices via constructor args; HMC config for inferring off-diagonal entries needs `ParameterSpec` support for matrix parameters
