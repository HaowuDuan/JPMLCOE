# 2D Stochastic Volatility: HMC Parameter Estimation Plan

## Model

**State:** `x_t = [x_1t, x_2t]`
**Transition:** `x_{t+1} = A x_t + e_t`, `e_t ~ N(0, Sigma)` — diagonal: `A = diag(a1, a2)`, `Sigma = diag(sigma1^2, sigma2^2)`
**Observation:** `y_t = b * x_1t + exp(x_2t / 2) * v_t`, `v_t ~ N(0,1)`

**Identifiability:** `b` and `sigma1` have scale invariance (`b*x1` with `x1 ~ sigma1`). Fix `b=1.0`. Learnable parameters: `a1, a2, sigma1, sigma2` (4 max).

**Stationarity:** `|a_i| < 1`. Use interval constraint `(eps, 1-eps)` with `eps=0.001`.

**Stationary initial condition:** `P0 = diag(sigma1^2/(1-a1^2), sigma2^2/(1-a2^2))` — differentiable, recomputed from live parameters.

---

## Inference Pipeline

- **Sampler:** HMC (MAP for quick validation first)
- **Filter:** `LEDHParticleFlowFilterHMC` + OT resampling, `stop_gradient=false`
- **Files:**
  - Model: `code/src/models/stochastic_volatility_2d.py`
  - Filter: `code/src/filters/particle/ledh_invertible_hmc.py`
  - Runner: `code/src/DF/hmc_runner.py`
  - Param handler: `code/src/DF/parameter_handler.py`

---

## Phase 1: Model Refactoring

### Problem
Both SV models store pre-computed TF constants (`self._A`, `self._Sigma`, `self._alpha_tf`, etc.). The DPF pipeline updates `self.a1` via `setattr`, but methods read from the constants. Gradients don't flow.

### Fix
Remove all pre-computed constants. Every method computes from live scalar attributes (`self.a1`, `self.a2`, `self.sigma1`, `self.sigma2`, `self.b`). Single source of truth — no dual paths.

### 2D SV: Methods to change
| Method | Currently uses | Change to |
|--------|---------------|-----------|
| `state_transition_batch` | `self._A`, `self._L_Sigma` | `particles * [a1, a2] + z * [sigma1, sigma2]` |
| `state_transition_mean_batch` | `self._A` | `particles * [a1, a2]` |
| `state_transition_cov_batch` | `self._Sigma` | `diag([sigma1^2, sigma2^2])` |
| `log_observation_prob_batch` | `self._b` | `self.b` |
| `observation_jacobian_batch` | `self._b` | `self.b` |
| `observation_function_batch` | `self._b` | `self.b` |
| `sample_initial_state_batch` | `self._L_P0` | `z * [sigma1/sqrt(1-a1^2), sigma2/sqrt(1-a2^2)]` |
| `state_transition_mean` | `self._A` | `x * [a1, a2]` |
| `state_jacobian` | `self._A` | `diag([a1, a2])` |
| `state_transition_cov` | `self._Sigma` | `diag([sigma1^2, sigma2^2])` |
| `process_noise_cov` | `self._Sigma` | `diag([sigma1^2, sigma2^2])` |
| `Sigma_0` | `self._P0` | `diag([sigma1^2/(1-a1^2), sigma2^2/(1-a2^2)])` |
| `observation_mean` | `self._b` | `self.b` |
| `observation_jacobian` | `self._b` | `self.b` |
| `observation_function` | `self._b` | `self.b` |
| `observation_cov` | — | no change (uses `x[1]` only) |
| `log_observation_prob` | `self._b` | `self.b` |
| `sample_initial_state` | `self._L_P0` | compute from live params |
| `sample_state_transition` | `self._A`, `self._L_Sigma` | compute from live params |
| `sample_observation` | `self._b` | `self.b` |
| `observation_function_with_noise[_batch]` | `self._b` | `self.b` |
| `state_jacobian_batch` | `self._A` | `diag([a1, a2])` tiled |
| `observation_cov_corrected` | — | no change |

### 1D SV: Same refactoring
Remove `self._alpha_tf`, `self._sigma_tf`, `self._beta_tf`, `self._stationary_var`, `self._log_beta2`. All methods use `self.alpha`, `self.sigma`, `self.beta` directly.

### Validation
- Existing unit tests must pass (backward compatibility)
- New test: `tf.GradientTape` gradient of `log_observation_prob_batch` w.r.t. each parameter, compared to finite difference

---

## Phase 2: Configs (MAP first, then HMC)

### Priors
| Parameter | Constraint | Prior |
|-----------|-----------|-------|
| `a1` | `(0.001, 0.999)` | `Uniform(0, 1)` |
| `a2` | `(0.001, 0.999)` | `Uniform(0, 1)` |
| `sigma1` | `positive` | `LogNormal(loc=-0.7, scale=0.5)` |
| `sigma2` | `positive` | `LogNormal(loc=0.0, scale=0.5)` |

### True parameters
```
a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0 (fixed)
T=200, seed=42
```

### Difficulty ladder

| Level | Trainable | Config name | Notes |
|-------|-----------|-------------|-------|
| 0 | `sigma2` | `ledh_ot_sigma2` | Simplest: positive constraint, no stationarity |
| 1 | `a2` | `ledh_ot_a2` | Stationarity constraint, P0 depends on a2 |
| 2 | `a2, sigma2` | `ledh_ot_a2_sigma2` | Two params, both affect P0 |
| 3 | `a1, a2, sigma1, sigma2` | `ledh_ot_all` | Full inference, b=1 fixed |

Each level: MAP config first (fast), then HMC config.

### Filter settings
```yaml
filter: LEDHParticleFlowFilterHMC
n_particles: 1000
n_lambda_steps: 29
resampling_method: ot_entropy
resampling_config:
  epsilon: 0.5
weight_clip_range: 50.0
stop_gradient_resampling: false
```

### HMC settings
```yaml
sampler: hmc
num_samples: 500
num_burnin: 200
step_size: 0.001
num_leapfrog_steps: 5
target_accept_prob: 0.75
adaptation_rate: 0.8
grad_clip_norm: 100.0
```

---

## Phase 3: Run and Validate

1. MAP at each level — verify loss decreases toward true parameter
2. HMC at each level — verify posterior concentrates around true value
3. Diagnostics: acceptance rate, ESS, R-hat, trace plots
4. Scale up through the ladder

---

## Implementation Order

1. Refactor `StochasticVolatility2DModel` — remove constants, all methods dynamic
2. Refactor `StochasticVolatilityModel` (1D) — same
3. Run existing tests — must still pass
4. Add gradient flow test for both models
5. Create Level 0 MAP + HMC configs
6. Run Level 0, validate
7. Create Level 1-3 configs, run progressively
