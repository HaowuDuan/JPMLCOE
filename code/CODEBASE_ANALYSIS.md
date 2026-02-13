# Comprehensive Codebase Analysis: Differentiable Particle Filter Pipeline

**Date:** 2026-02-12
**Scope:** Full codebase analysis covering `code/src/` — filters, utils, models, resampling, experiments, and DF modules.

---

## Table of Contents

1. [Bugs and Problems in the Differentiable PF Pipeline](#1-bugs-and-problems)
2. [Performance Bottlenecks and Improvement Suggestions](#2-performance-bottlenecks)
3. [Preliminary Pytest Test Plan for Numerical Stability](#3-test-plan)

---

## 1. Bugs and Problems

### 1.1 CRITICAL: `flow_params.py` — Regularization Computed but Never Applied

**File:** `src/utils/flow_params.py`, Lines 57–71
**Severity:** CRITICAL

```python
# Lines 57-65: P_reg is computed...
if regularization > 0.0:
    reg_strength = regularization * (trace_P / state_dim_f)
    P_reg = P + reg_strength * tf.eye(state_dim, dtype=P.dtype)
else:
    P_reg = P

# Line 71: ...but P is used instead of P_reg!
HPH = H @ P @ tf.transpose(H)  # BUG: should be P_reg
```

**Impact:** LEDH filter's regularization parameter has zero effect. All downstream computations (A matrix, b vector) use unregularized P. This means LEDH can silently produce ill-conditioned matrices without the intended safeguard.

**Fix:** Change line 71 to `HPH = H @ P_reg @ tf.transpose(H)`. Also use `P_reg` on line 80 and 91.

---

### 1.2 CRITICAL: `hmc_runner.py` — `tf.py_function` Breaks Differentiability

**File:** `src/DF/hmc_runner.py`, Lines 57–101
**Severity:** CRITICAL

```python
@tf.function(reduce_retracing=True)
def _negative_log_posterior(self, unconstrained_params):
    ...
    log_likelihood = tf.py_function(update_and_run_filter, [], tf.float32)
    ...
```

**Impact:**
- `tf.py_function` executes in eager mode inside a tf.function, breaking the computation graph. No gradient flows through the filter output w.r.t. parameters.
- The `@tf.function(reduce_retracing=True)` decorator is functionally useless because `py_function` forces eager evaluation.
- HMC will use numerical (finite-difference) gradients only, not algorithmic gradients. This defeats the purpose of the "Differentiable Filter" framework.

**Fix:** Either (a) make the full filter pipeline run inside tf.function without py_function, or (b) remove the @tf.function decorator and explicitly acknowledge numerical gradients.

---

### 1.3 HIGH: `ot_entropy.py` — `tf.stop_gradient` Blocks Gradient Flow

**File:** `src/resampling/ot_entropy.py`, Lines 394–400
**Severity:** HIGH

```python
mean = tf.reduce_mean(particles, axis=0, keepdims=True)
centered = particles - tf.stop_gradient(mean)          # Blocks gradients!

std = tf.math.reduce_std(particles)
scale_factor = tf.stop_gradient(std * tf.sqrt(dimension) + 1e-8)  # Blocks gradients!
scaled = centered / scale_factor
```

**Impact:** Gradients from the transport matrix back through the particle normalization are blocked. For a method whose primary purpose is differentiable resampling, this severely limits gradient flow. The centering/scaling should be treated as part of the differentiable computation graph.

**Fix:** Remove `tf.stop_gradient` wrappers on both `mean` and `scale_factor`.

---

### 1.4 HIGH: `ot_entropy.py` — Softmin Order of Operations

**File:** `src/resampling/ot_entropy.py`, Line 75
**Severity:** HIGH

```python
# Docstring says: -epsilon * log(sum_j exp((f_j - C_ij) / epsilon))
# Code does:
temp_val = f - cost_matrix / epsilon   # = f - C/eps  (WRONG)
# Should be:
temp_val = (f - cost_matrix) / epsilon  # = (f - C)/eps  (CORRECT per docstring)
```

**Impact:** If the potentials `f` are not pre-scaled by `1/epsilon`, this computes the wrong softmin. The Sinkhorn convergence and transport matrix computation could be incorrect, leading to wrong resampled particle distributions.

**Note:** Verify whether the Sinkhorn loop's potential updates absorb the 1/epsilon factor. If potentials are in "epsilon-scaled" form, the current code may be intentionally correct but the docstring is misleading.

---

### 1.5 HIGH: `distributions.py` — Silent Fallback to Uniform Weights

**File:** `src/utils/distributions.py`, Lines 192–195
**Severity:** HIGH

```python
is_finite = tf.reduce_all(tf.math.is_finite(weights))
uniform_weights = tf.ones(n_particles, ...) / tf.cast(n_particles, ...)
weights = tf.cond(is_finite, lambda: weights, lambda: uniform_weights)
```

**Impact:** If ANY single weight is NaN/Inf, ALL weights are silently replaced with uniform (1/N). No warning is logged. This hides serious numerical problems — the user sees the filter "working" when it has actually collapsed. Weight collapse is a critical diagnostic signal that should be surfaced, not hidden.

**Suggestion:** Add `tf.debugging.assert_all_finite` in debug mode, or at minimum log a warning counter.

---

### 1.6 HIGH: `flow_params.py` — Hardcoded `tf.float32` for Identity Matrix

**File:** `src/utils/flow_params.py`, Line 89
**Severity:** HIGH

```python
I = tf.eye(state_dim, dtype=tf.float32)  # Should use P.dtype
```

**Impact:** If P, H, R are `tf.float64`, this creates a type mismatch. TensorFlow may auto-cast but the result loses precision. All downstream matrix operations in the b(lambda) computation will be affected.

**Fix:** `I = tf.eye(state_dim, dtype=P.dtype)`

---

### 1.7 HIGH: `distributions.py` — Hardcoded Pi with Insufficient Precision

**File:** `src/utils/distributions.py`, Lines 33, 166, 177
**Severity:** MEDIUM-HIGH

```python
tf.constant(3.14159265359, dtype=x.dtype)  # Only 11 significant figures
```

**Impact:** For `tf.float64`, this truncates pi (which has 16+ significant figures for float64). The error accumulates across log-probability computations over many particles and timesteps.

**Fix:** Use `tf.constant(math.pi, dtype=x.dtype)` (import `math` module).

---

### 1.8 MEDIUM: `two_sensor_bearing.py` — Batch Method Missing Bearing Wrapping

**File:** `src/models/two_sensor_bearing.py`
**Severity:** MEDIUM

The single-particle `log_observation_prob` wraps bearing differences to `[-pi, pi]` via `tf.atan2(tf.sin(diff), tf.cos(diff))`, but the batch version `log_observation_prob_batch` does NOT wrap bearings.

**Impact:** For particles near bearing discontinuity (near +/-pi), the batch method will compute incorrect log-probabilities, leading to wrong particle weights.

**Fix:** Add `diff = tf.atan2(tf.sin(diff), tf.cos(diff))` to the batch method.

---

### 1.9 MEDIUM: `lorenz96.py` — Undefined Attribute `observed_dims`

**File:** `src/models/lorenz96.py`, Line 242
**Severity:** MEDIUM

```python
observed_states = particles[:, self.observed_dims]  # AttributeError!
```

The attribute is defined as `self.obs_indices` (line 74), not `self.observed_dims`.

**Impact:** Runtime crash when calling `log_observation_prob_batch`.

**Fix:** Change `self.observed_dims` to `self.obs_indices`.

---

### 1.10 MEDIUM: `lorenz96.py` — `state_transition_mean_batch` Only Does 1 RK4 Step

**File:** `src/models/lorenz96.py`, Line 233
**Severity:** MEDIUM

The batch transition method performs only a single RK4 step, while the single-particle version integrates for `obs_interval` steps.

**Impact:** Incorrect state predictions when using batch methods (e.g., in `compute_flow_weights`).

**Fix:** Add the `obs_interval` loop to the batch method.

---

### 1.11 MEDIUM: `distributions.py` — Bare Cholesky Without Safeguard

**File:** `src/utils/distributions.py`, Line 29
**Severity:** MEDIUM

```python
L = tf.linalg.cholesky(cov)  # Can fail if cov is not SPD
```

**Impact:** If covariance matrix is numerically non-SPD (common after many filter steps), this throws a runtime error. The `safe_cholesky` utility exists but is not used here.

**Fix:** Replace with `L = safe_cholesky(cov)`.

---

### 1.12 LOW: `linalg.py` — `log_det` Ignores Determinant Sign

**File:** `src/utils/linalg.py`, Lines 79–80
**Severity:** LOW

```python
sign, logdet = tf.linalg.slogdet(A)
return logdet  # sign is discarded
```

If a matrix becomes non-SPD, the sign will be negative, but this is silently ignored.

---

## 2. Performance Bottlenecks

### 2.1 Filters

| Bottleneck | File | Impact | Suggestion |
|---|---|---|---|
| **Per-particle flow: N x n_lambda_steps compute_flow_params** | `ledh_invertible.py` | Dominant cost. Each particle requires its own Jacobian + matrix ops per flow step. For 500 particles x 29 steps = 14,500 calls. | Batch the Jacobian computation: `observation_jacobian_batch(particles)`. Cache H when linearization point is reused. |
| **tf.map_fn with parallel_iterations** | `ledh_invertible.py`, `batched_ekf.py` | Sequential processing of particles. `parallel_iterations=10` helped but doesn't fully parallelize. | Keep parallel_iterations=10 (verified improvement). Going higher may not help due to memory. |
| **Per-particle EKF objects** | `ledh_invertible.py` lines 363-374 | Creates/maintains per-particle covariance matrices. O(N * state_dim^2) memory. | Use `batched_ekf.py` for predict/update (already done). Consider shared covariance approximation for large N. |
| **UKF sigma point loop** | `unscented_kalman.py`, lines 146-148 | TensorArray loop inside @tf.function creates retracing overhead. | Replace with vectorized `tf.stack([mean + sqrt_cov[:, i] for i in range(n)])` outside @tf.function, or use pure tensor operations. |
| **Kalman matrix inversion** | `kalman.py`, line 158 | Uses `tf.linalg.inv(innovation_cov)` instead of solve. | Replace with `tf.linalg.solve(innovation_cov, H @ cov)` to avoid explicit inversion. |

### 2.2 Utils

| Bottleneck | File | Impact | Suggestion |
|---|---|---|---|
| **Nested matrix ops in b(lambda)** | `flow_params.py`, line 91 | Computes `(I + lambda*A) @ P @ H^T @ R_inv @ (z-e)` as full matrix chain. For state_dim=100, this is three 100x100 multiplications. | Factor as vector operations: compute `P @ H^T @ R_inv @ (z-e)` first (result is a vector), then apply `(I + lambda*A)`. |
| **Transpose-based triangular solve** | `distributions.py`, lines 160-161, 171-172 | `triangular_solve(L_Q, tf.transpose(diff))` transposes N x d to d x N, solves, transposes back. Memory-intensive. | For state-independent Q, this is acceptable. For large d, consider batch solves. |
| **Q Cholesky not cached** | `distributions.py`, line 159 | Cholesky of Q computed fresh each call to `compute_flow_weights`. Q is often constant across timesteps. | Cache L_Q if Q doesn't change. |

### 2.3 Models

| Bottleneck | File | Impact | Suggestion |
|---|---|---|---|
| **Default batch methods are Python loops** | `model_base.py`, lines 123-144 | `state_transition_batch`, `state_transition_mean_batch`, `log_observation_prob_batch` all loop over particles. | Override with vectorized TF implementations in each model. LinearGaussian does this well (good example). |
| **Lorenz96 finite-difference Jacobian** | `lorenz96.py` | Computes Jacobian by perturbing each dimension. O(state_dim^2) per call. For state_dim=1000, this is 1M operations. | Use analytical Jacobian for Lorenz96 (known closed form). |
| **AcousticTracking mixed NumPy/TF** | `acoustic_tracking.py` | NumPy methods can't be traced by tf.function, forcing eager execution. | Port to pure TensorFlow (acoustic_tracking_full.py is a good reference). |

### 2.4 Resampling

| Bottleneck | File | Impact | Suggestion |
|---|---|---|---|
| **OT entropy Sinkhorn iterations** | `ot_entropy.py` | Sinkhorn convergence can require 50-100 iterations. Each iteration is O(N^2) for the cost matrix. | Use epsilon-scaling (already implemented) with warm-starting from previous timestep's potentials. |
| **O(N^2) distance matrix** | `ot_entropy.py`, line 39 | `squared_cost_matrix` computes all pairwise distances. For N=1000, this is 1M entries. | Acceptable for N < 2000. For larger N, consider approximate OT methods. |

### 2.5 Experiments & DF

| Bottleneck | File | Impact | Suggestion |
|---|---|---|---|
| **HMC with py_function** | `hmc_runner.py` | Every HMC step runs the full filter in eager mode. No graph optimization, no parallel execution. | Refactor filter to run entirely inside tf.function (requires eliminating all Python-side state). |
| **No filter result caching** | `hmc_runner.py` | Each HMC step creates a new filter object. Initialization overhead repeated. | Reuse filter object, only update parameters between steps. |

---

## 3. Preliminary Pytest Test Plan for Numerical Stability

Focus: Key operations that can fail silently — matrix inversions, ODE integration, weight normalization, Jacobian computations.

### 3.1 `test_linalg.py` — Linear Algebra Utilities

```
test_safe_cholesky_well_conditioned
    - Input: known SPD matrix -> verify L @ L^T approx equals A

test_safe_cholesky_near_singular
    - Input: matrix with eigenvalues [1.0, 1e-12] -> verify no crash, L is lower triangular

test_safe_cholesky_adaptive_scaling
    - Input: SPD matrix with trace=1e6 -> verify jitter scales proportionally
    - Input: SPD matrix with trace=1e-6 -> verify jitter doesn't over-regularize

test_safe_solve_cholesky_accuracy
    - Input: known A, x -> compute b = Ax, solve for x -> verify ||x_hat - x|| < tol

test_safe_solve_ill_conditioned
    - Input: matrix with condition number 1e10 -> verify solution doesn't blow up

test_safe_solve_agrees_across_methods
    - Compare cholesky, lstsq, default methods on same system -> verify agreement

test_log_det_positive_definite
    - Input: known SPD matrix -> verify against np.linalg.slogdet

test_symmetrize_preserves_eigenvalues
    - Input: slightly asymmetric matrix -> verify eigenvalues of output match expected

test_matrix_sqrt_cholesky_vs_eig
    - Input: SPD matrix -> verify both methods produce S such that S @ S^T approx A
```

### 3.2 `test_flow_params.py` — Flow Parameter Computation

```
test_flow_params_linear_gaussian
    - Use LinearGaussian model -> compute A, b at lambda=0, 0.5, 1.0
    - Verify A is symmetric negative semi-definite (eigenvalues <= 0)
    - Verify ||b|| doesn't explode as lambda -> 1

test_flow_params_regularization_effect
    - Compute A, b with regularization=0 and regularization=0.01
    - Verify regularized A has smaller spectral radius (once P_reg bug is fixed)

test_flow_params_identity_at_lambda_zero
    - At lambda=0: A should be 0 matrix, b should be 0 vector
    - Verify within tolerance

test_flow_params_S_invertibility
    - Sweep lambda from 0 to 1 -> verify S = lambda*HPH + R remains SPD throughout
    - Check condition number doesn't exceed threshold

test_flow_params_dtype_consistency
    - Pass float64 inputs -> verify all outputs are float64 (catches hardcoded float32)
```

### 3.3 `test_distributions.py` — Weight and Probability Computations

```
test_log_gaussian_prob_standard_normal
    - Input: x=0, mean=0, cov=I -> verify log_prob equals known value

test_log_gaussian_prob_ill_conditioned_cov
    - Input: covariance with condition number 1e8 -> verify finite result

test_normalize_log_weights_extreme_values
    - Input: log_weights = [-1000, -999, -1001] -> verify finite normalized weights
    - Input: log_weights = [0, 0, 0] -> verify uniform output

test_normalize_log_weights_single_dominant
    - Input: log_weights = [0, -100, -100] -> verify dominant weight near 1.0

test_compute_flow_weights_uniform_flow
    - Identity flow (eta_1 == eta_0) with uniform prev_weights
    - Verify output weights are proportional to observation likelihood

test_compute_flow_weights_preserves_normalization
    - Any valid input -> verify sum(weights) == 1.0 within tolerance

test_compute_flow_weights_jacobian_sign
    - Input: negative jacobians -> verify behavior (should warn or handle)

test_compute_flow_weights_nan_handling
    - Input: one particle at NaN -> verify fallback to uniform (and ideally a warning)
```

### 3.4 `test_ode_solvers.py` — Integration Methods

```
test_euler_step_linear_system
    - dx/dt = Ax, known solution -> verify Euler approximation within O(dt) error

test_rk4_step_linear_system
    - dx/dt = Ax, known solution -> verify RK4 within O(dt^4) error

test_rk4_accuracy_vs_euler
    - Same system, same dt -> verify RK4 error << Euler error

test_euler_maruyama_mean_and_variance
    - Run many samples of OU process -> verify mean and variance match theory

test_euler_step_preserves_shape
    - Input: (d,) -> output (d,)
    - Input: (batch, d) -> output (batch, d)

test_integrate_ode_convergence
    - Halve dt -> verify error decreases at expected rate (O(dt) for Euler, O(dt^4) for RK4)
```

### 3.5 `test_resampling.py` — Resampling Methods

```
test_soft_resample_preserves_weighted_mean
    - Input: particles, weights -> verify weighted mean of output matches input

test_soft_resample_gradient_flows
    - Use tf.GradientTape -> verify gradient w.r.t. particles is not None/zero

test_ot_entropy_resample_preserves_mean
    - Input: particles, weights -> verify resampled mean approx matches weighted mean

test_ot_entropy_transport_matrix_properties
    - Verify T is doubly stochastic (rows and columns sum to expected values)
    - Verify T >= 0 elementwise

test_ot_entropy_gradient_flows
    - Use tf.GradientTape -> verify gradient w.r.t. particles is not None/zero
    - (Will fail until stop_gradient bug is fixed)

test_sinkhorn_convergence
    - Fixed cost matrix, known solution -> verify convergence within max_iter

test_squared_cost_matrix_non_negative
    - Random particles -> verify all entries >= 0
```

### 3.6 `test_models.py` — Model Consistency

```
test_jacobian_vs_finite_difference
    - For each model: compute analytical Jacobian and finite-difference Jacobian
    - Verify agreement within O(epsilon) tolerance
    - Critical for: range_bearing (atan2), acoustic_tracking_full, two_sensor_bearing

test_observation_mean_jacobian_consistency
    - For each model: verify observation_jacobian(x) == d(observation_mean)/dx
    - Use tf.GradientTape to compute automatic Jacobian

test_batch_vs_single_agreement
    - For each model method: verify batch(particles)[i] == single(particles[i])
    - Critical methods: log_observation_prob_batch, state_transition_mean_batch

test_range_bearing_jacobian_near_sensor
    - Place particle very close to sensor (r < 0.01)
    - Verify Jacobian doesn't produce Inf/NaN
    - Verify observation_mean doesn't produce Inf/NaN

test_two_sensor_bearing_wrapping
    - Place particle such that bearing is near +/- pi
    - Verify log_observation_prob and log_observation_prob_batch agree

test_model_dtypes
    - Verify all model outputs are consistent dtype (float32 or float64)
```

### 3.7 `test_filters_integration.py` — End-to-End Stability

```
test_ekf_on_linear_gaussian
    - Run EKF on LinearGaussian model -> verify matches Kalman filter exactly

test_edh_flow_preserves_particle_count
    - Run EDH flow -> verify N particles in == N particles out at each step

test_edh_invertible_weights_sum_to_one
    - Run EDH invertible -> verify weights sum to 1.0 at each timestep

test_ledh_invertible_weights_sum_to_one
    - Run LEDH invertible -> verify weights sum to 1.0 at each timestep

test_filter_on_trivial_model
    - Zero noise model -> verify filter mean converges to true state

test_filter_numerical_stability_long_sequence
    - Run filter for T=500 steps -> verify no NaN/Inf in means or covs
    - Check covariance stays SPD (all eigenvalues > 0)
```

### 3.8 `test_df_pipeline.py` — Differentiable Filter Framework

```
test_parameter_handler_bijector_roundtrip
    - constrain(unconstrain(x)) == x for all constraint types

test_parameter_handler_log_prior_jacobian
    - Verify log prior includes correct Jacobian adjustment
    - Use finite differences to verify

test_differentiable_model_parameter_update
    - Update a parameter -> verify model uses new value
    - Restore -> verify original value is back

test_negative_log_posterior_finite
    - Run with valid parameters -> verify output is finite scalar
    - Run with extreme parameters -> verify output is finite (no crash)
```

---

## Summary of Priority Actions

### Immediate (Fix Before Next Experiment Run)
1. Fix `flow_params.py` line 71: use `P_reg` instead of `P`
2. Fix `flow_params.py` line 89: use `P.dtype` instead of `tf.float32`

### High Priority (Fix Before Relying on Results)
3. Fix `ot_entropy.py` lines 395-400: remove `tf.stop_gradient`
4. Investigate `ot_entropy.py` line 75: verify softmin order of operations
5. Fix `two_sensor_bearing.py`: add bearing wrapping to batch log-prob
6. Fix `lorenz96.py`: rename `observed_dims` to `obs_indices`

### Medium Priority (Improve Robustness)
7. Fix `distributions.py` line 29: use `safe_cholesky`
8. Replace hardcoded pi with `math.pi` (lines 33, 166, 177)
9. Add warning when weights fall back to uniform
10. Fix `lorenz96.py` batch transition to use `obs_interval` steps

### Long Term (Architecture)
11. Refactor `hmc_runner.py` to eliminate `tf.py_function`
12. Add `observation_jacobian_batch` to model interface for LEDH performance
13. Write the pytest suite outlined in Section 3
