# Codebase Analysis Report

> Generated 2026-02-14. Covers all 57 Python source files under `code/src/`.

---

## Executive Summary

The codebase implements a Bayesian filtering framework with Kalman, particle, and differentiable particle flow filters. The core algorithms are correct and well-documented mathematically. However, the **DPF filter variants** (`edh_flow.py`, `edh_invertible.py`, `ledh_flow.py`, `ledh_invertible.py`, `stochastic_edh.py`) were written by copy-pasting from each other and never refactored. This manifests in three main problems:

1. **Massive code duplication** -- five nearly-identical copies of resampling config, initialization, lambda schedule, debug storage, and mean/cov estimation (~200 duplicated lines total)
2. **Performance bottlenecks** -- flow loops run as Python `for` loops calling TensorFlow ops, missing `tf.function` compilation where it matters most
3. **Readability** -- inconsistent APIs between `FlowFilterBase` subclasses vs standalone filters, dead code, magic numbers, and no shared base for the DPF-specific logic

The rest of the codebase (models, Kalman filters, resampling, linalg utilities) is in better shape, with localized issues around the `Kitagawa` model's mutable time-step state and the `model_base.py` batch-method defaults.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Code Duplication](#2-code-duplication-the-biggest-problem)
3. [Performance Bottlenecks](#3-performance-bottlenecks)
4. [Readability & Code Quality](#4-readability--code-quality)
5. [Design Issues](#5-design-issues)
6. [Model-Specific Issues](#6-model-specific-issues)
7. [Recommended Refactoring Priorities](#7-recommended-refactoring-priorities)

---

## 1. Architecture Overview

```
src/
  core/
    model_base.py          -- StateSpaceModel ABC (all models inherit from this)
    filter_base.py         -- FilterBase (Kalman filters)
    types.py               -- FilterResult dataclass
  models/                  -- 8 state-space models (linear_gaussian, kitagawa, lorenz96, ...)
  filters/
    kalman/                -- KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, batched_ekf
    particle/              -- 7 particle flow filters + FlowFilterBase
  resampling/              -- systematic, soft, ot_entropy
  utils/                   -- flow_params, distributions, linalg, ode_solvers, device
  experiments/             -- run_experiment.py, run_dpf_experiment.py, visualization.py
  DF/                      -- DPFRunner, ParameterHandler, HMC runner (differentiable framework)
```

**Two parallel filter hierarchies exist:**

| Hierarchy | Base | Filters | Has weights? | Has resampling? |
|-----------|------|---------|-------------|----------------|
| `FlowFilterBase` | `flow_base.py` | `edh_flow`, `ledh_flow`, `stochastic_edh` | No (equal 1/N) | No |
| Standalone | None | `edh_invertible`, `ledh_invertible` | Yes (importance) | Yes |

This split is the root cause of duplication -- the invertible filters reimplement everything from scratch.

---

## 2. Code Duplication (The Biggest Problem)

### 2.1 Resampling Configuration -- 2 copies (invertible filters only)

The resampling config block (string-to-function method map + config scalar conversion) appears in the two invertible filters that actually use resampling:

| File | Lines |
|------|-------|
| `ledh_invertible.py` | 56-81 |
| `edh_invertible.py` | 61-87 |

**Note:** `edh_flow.py`, `ledh_flow.py`, and `stochastic_edh.py` also contain this block but shouldn't -- they are equal-weight filters that never resample. The resampling code in those three files is dead code and should be removed.

**Verdict:** Not worth extracting a shared utility for just 2 filters. The dead resampling config in the 3 equal-weight filters should simply be deleted.

### 2.2 Particle Initialization -- 4 copies

Cholesky-based particle sampling from N(mu_0, Sigma_0):

- `ledh_invertible.py:154-164`
- `edh_invertible.py:164-175`
- `ledh_flow.py:130-146`
- `edh_flow.py:119-139`

The `FlowFilterBase` and invertible filters also have **incompatible `initialize()` signatures** -- `FlowFilterBase` takes `random_state: np.random.Generator`, invertible filters take `initial_mean, initial_cov, random_seed: int`.

### 2.3 `filter()` Method -- 2 copies

The `filter()` method in `ledh_invertible.py:381-428` and `edh_invertible.py:348-395` is virtually identical (loop, predict, update, mean/cov, stack, return FilterResult).

---

## 3. Performance Bottlenecks

### 3.1 Flow loops are sequential Python `for` loops

The flow loops in all filters are inherently sequential -- each lambda step depends on the previous step's particles. This cannot be parallelized.

Wrapping in `tf.function` would only save Python dispatch overhead (~microseconds per TF op), which is negligible compared to the actual matrix computations. **Not a real bottleneck for filtering.**

The one context where `tf.function` could help is during **HMC gradient computation** (inside `GradientTape`), where graph-mode can optimize the backward pass. This is a narrower concern limited to `log_marginal_likelihood_tf()` in the invertible filters.

### 3.2 HIGH: `model_base.py` default batch methods use Python loops

`model_base.py:123-144` -- All default batch methods (`state_transition_batch`, `log_observation_prob_batch`, etc.) loop over particles in Python:

```python
def state_transition_batch(self, particles, seed):
    N = particles.shape[0]
    seeds = tf.random.experimental.stateless_split(seed, num=N)
    return tf.stack([self.sample_state_transition(particles[i], seeds[i]) for i in range(N)])
```

With 1000 particles, this is 1000 individual TF calls instead of one vectorized call. Models that override these (linear_gaussian, kitagawa) are fine. Models that don't (stochastic_volatility) hit this.

### 3.3 MEDIUM: R_inv computation -- missing caching and no safe inverse

**Missing caching:** `edh_flow.py:218` and `stochastic_edh.py:178` compute `tf.linalg.inv(R)` every timestep. They even declare `self.R_inv_cache = None` in `__init__` but never use it. Other filters (`ledh_invertible.py:214-217`, `ledh_flow.py:322-324`, `edh_invertible.py:239-241`) correctly cache.

**No safe inverse:** All files use raw `tf.linalg.inv(R)` except `edh_invertible.py:240-241` which uses Cholesky-based inversion. Since R is a covariance matrix (SPD), all should use `safe_cholesky` from `utils/linalg.py` + `tf.linalg.cholesky_solve` for robustness.

Files to fix:
- `edh_flow.py:218` -- add caching + use safe inverse
- `stochastic_edh.py:178` -- add caching + use safe inverse
- `ledh_invertible.py:216` -- switch from `tf.linalg.inv` to safe Cholesky
- `ledh_flow.py:324` -- switch from `tf.linalg.inv` to safe Cholesky
- `sde_local_correction.py:39` -- same

### 3.4 LOW: `np.unique` forces mid-loop TF->numpy conversion

`ledh_invertible.py:294-296` and `edh_invertible.py:307-309`:
```python
particles_np = self.particles.numpy()
n_unique = len(np.unique(particles_np, axis=0))
```
All other data (means, covs, weights, ESS) correctly stays as TF tensors during the loop and converts once at the end via `tf.stack().numpy()`. This `np.unique` call forces a mid-loop numpy conversion for a diagnostic counter. **Fix:** Defer to post-loop -- store resampled particle snapshots as TF tensors and compute unique counts after filtering completes.

### 3.6 MEDIUM: `stochastic_edh.py` BVP shooting is expensive

`stochastic_edh.py:88-156` -- When `schedule_mu > 0`, `_compute_optimal_schedule` calls `_shoot()` via bisection (~42 shooting iterations, each integrating 500 Euler steps). This runs **per-timestep** inside `update()`. With T=100 timesteps, that's 42 * 500 * 100 = 2.1M Euler steps just for scheduling.

**Fix:** Cache the schedule if P and H don't change significantly between timesteps, or precompute for a grid of P values.

### 3.5 LOW: TF tensors accumulated in Python lists

All filters accumulate TF tensors in Python lists (`self.means.append(mean)`) then `tf.stack().numpy()` at the end. **Fix:** Pre-allocate `tf.TensorArray` of known size T, write by index during the loop, then `.stack().numpy()` once at the end.

---

## 4. Readability & Code Quality

### 4.1 Dead Code

| File | What | Lines |
|------|------|-------|
| `ledh_flow.py` | `_compute_drift_single()` method -- never called | 185-219 |
| `ledh_flow.py` | `from concurrent.futures import ThreadPoolExecutor` -- never used | 6 |
| `edh_flow.py` | `from ...utils.ode_solvers import rk4_step` -- `rk4` raises `NotImplementedError` | 9 |
| `kitagawa.py` | `sample_state_transition_tf`, `log_observation_prob_tf`, `sample_initial_state_batch_tf` -- legacy duplicates of existing methods | 397-432 |
| `kitagawa.py` | `observe()` -- identical to `observation_function()` which is identical to `observation_mean()` | 228-230 |

### 4.2 Magic Numbers

| File | Line | Value | What it means |
|------|------|-------|---------------|
| All 5 filters | `_generate_lambda_steps` | `q = 1.2` | Geometric ratio for exponential schedule, from the paper |
| `ledh_flow.py` | 264, 270 | `100.0`, `1000.0` | Drift clipping and particle clipping thresholds |
| `ledh_flow.py` | 264 | `1e-10` | Epsilon for norm clipping |
| `stochastic_edh.py` | 121-127 | `0.1, 20.0, 50.0` | Bisection bracket bounds for shooting |
| `stochastic_edh.py` | 89 | `500` | Number of ODE steps in shooting integration |
| `stochastic_edh.py` | 136 | `40` | Number of bisection iterations |

**Fix:** Define these as class constants or config parameters.

### 4.3 Inconsistent API Patterns

**Initialize signatures differ:**
- `FlowFilterBase.initialize(random_state: np.random.Generator)`
- `LEDHParticleFlowFilter.initialize(initial_mean, initial_cov, random_seed: int)`
- `EDHParticleFlowFilter.initialize(initial_mean, initial_cov, random_seed: int)`

**Filter signatures differ:**
- `FlowFilterBase.filter(observations, random_state, progress_callback)`
- `LEDHParticleFlowFilter.filter(observations, initial_mean, initial_cov, random_seed, progress_callback)`

**Type conversions scattered:**
The pattern `y.numpy() if isinstance(y, tf.Tensor) else y` appears in:
- `edh_invertible.py:317`
- `edh_flow.py:293`
- `ledh_flow.py:403`
- `stochastic_edh.py:252`

Should be a utility: `to_numpy(tensor)`.

### 4.4 `flow_params.py` is dense and hard to follow

The four functions in `flow_params.py` (370 total lines) are the mathematical heart of the framework but lack intermediate variable naming that maps to the paper's notation. For example, `flow_params.py:93`:

```python
term1 = tf.linalg.matvec((I + lambda_val * A) @ P_reg @ tf.transpose(H) @ R_inv, observation - e)
```

This is a single 90-character expression implementing equation (11) with 5 matrix operations chained. Breaking it up would help readability:

```python
PHT_Rinv = P_reg @ tf.transpose(H) @ R_inv   # P @ H^T @ R^{-1}
I_lA = I + lambda_val * A                      # (I + lambda * A)
term1 = tf.linalg.matvec(I_lA @ PHT_Rinv, z - e)
```

(The batched version at `flow_params.py:244-268` already does this correctly.)

### 4.5 Inconsistent `debug_mode` implementation

Debug storage is allocated in all 4 filters but only `edh_flow.py` and `ledh_flow.py` actually populate it in their `update()` methods. `ledh_invertible.py` and `edh_invertible.py` allocate the debug dict but never write to it.

### 4.6 `edh_invertible.py:251` uses `tf.range` in Python loop

```python
for j in tf.range(self.n_lambda_steps):
```

`tf.range` returns a TF tensor, but the `for` loop iterates over it in Python (eager mode). This is slower than `range()` because it creates TF tensors for each index. Should be `range(self.n_lambda_steps)` (as correctly done in `ledh_invertible.py:227`).

---

## 5. Design Issues

### 5.1 `compute_flow_weights` at `distributions.py:104-202` is a 100-line monolith

This function computes 6 different quantities (transition means, Q, obs log-probs, two Cholesky solves, log-weight combination). The two Cholesky solve blocks (lines 159-169 and 171-180) are nearly identical -- only `diff_1` vs `diff_0` differs. Extract a helper:

```python
def _log_transition_prob(diff, L_Q, state_dim):
    y = tf.linalg.triangular_solve(L_Q, tf.transpose(diff), lower=True)
    y = tf.transpose(y)
    return -0.5 * (tf.reduce_sum(y**2, axis=1) + 2.0 * ... + ...)
```

---

## 6. Model-Specific Issues

### 6.1 Kitagawa `self.t` global mutable state (`kitagawa.py:70`)

The Kitagawa model tracks the current time step in `self.t`, which is mutated by `sample_state_transition()` (line 139/144) and `sample_initial_state()` (line 127/133). This design:

- **Breaks parallelism**: `state_transition_batch()` at line 269-284 uses `self.t` which was set externally
- **Creates hidden coupling**: Callers must set `self.t` before calling batch methods (as done in `ledh_invertible.py:370-371`)
- **Is not thread-safe**: Two filters sharing a model would corrupt each other's time index

**Fix:** Make `t` an explicit parameter to all time-dependent methods. The `state_transition_mean_with_t()` at line 177 already shows the right API -- generalize it.

### 6.2 Kitagawa legacy TF methods (`kitagawa.py:397-432`)

Three methods with `_tf` suffix (`sample_state_transition_tf`, `log_observation_prob_tf`, `sample_initial_state_batch_tf`) duplicate the non-`_tf` versions. The comment says "Legacy TF methods (kept for backward compatibility)" -- but nothing in the codebase calls them. Safe to delete.

### 6.3 Kitagawa triple observation API (`kitagawa.py:199, 224, 228`)

Three methods return the same thing:
- `observation_mean(x)` -- required by base class
- `observation_function(x)` -- required by flow filters
- `observe(x)` -- required by kernel flow filter

**Fix:** The base class should define `observation_function = observation_mean` as default, and `observe` should be removed from individual models.

### 6.4 `model_base.py` -- `observation_function` and `observation_function_batch` not abstract

These are called by `compute_flow_params_batch` (line 240) but are not part of the abstract interface. If a model doesn't implement them, it fails at runtime with `AttributeError`, not at class definition time.

### 6.5 RNG design: use `tf.random.experimental.stateless_split` instead of counter increment

All filters use `self.seed_counter += 1` to produce seeds like `[42, 0]`, `[43, 0]`, `[44, 0]`, ... Neighboring integer seeds are not guaranteed to produce statistically independent random streams. The proper approach is `tf.random.experimental.stateless_split`, which uses hash-based splitting (Threefry/Philox) to guarantee independence.

**Two patterns to replace:**

*Pattern 1 -- Counter increment (8 files, ~20 call sites):*

| File | Lines |
|------|-------|
| `ledh_invertible.py` | 155-156, 199-200, 300-301 |
| `edh_invertible.py` | 165-166, 227-228, 321-322 |
| `ledh_flow.py` | 302-303 |
| `edh_flow.py` | 195-196 |
| `edh_flow_global.py` | 197-198 |
| `stochastic_edh.py` | 241, 248 |
| `sde_local_correction.py` | 101, 108 |

*Pattern 2 -- Numpy `random_state.integers()` to generate TF seeds (3 files + `flow_base.py`):*

| File | Lines |
|------|-------|
| `flow_base.py` | 77 |
| `ledh_flow.py` | 136 |
| `edh_flow.py` | 125 |
| `edh_flow_global.py` | 127 |

**Fix:** Store `self.seed` as a TF int32 tensor initialized from the user's input seed. Every time randomness is needed, split:

```python
# In __init__ or initialize():
self.seed = tf.constant([random_seed, 0], dtype=tf.int32)

# Each time randomness is needed:
seeds = tf.random.experimental.stateless_split(self.seed, num=2)
self.seed = seeds[0]
subkey = seeds[1]
z = tf.random.stateless_normal(shape, seed=subkey)
```

This eliminates the numpy RNG dependency and the counter hack, and guarantees independent streams.

---

## 7. Recommended Refactoring Priorities

### Priority 1: Fix Kitagawa time-step design

**Effort: ~half day. Impact: Correctness for batch/parallel operations.**

- Add `t` parameter to `state_transition_mean()`, `state_transition_batch()`, etc.
- Remove `self.t` mutation from sampling methods
- Pass `t` explicitly from the filter loop

### Priority 2: Clean up dead code and magic numbers

**Effort: ~2 hours. Impact: Readability.**

- Delete `ledh_flow._compute_drift_single`, unused imports, Kitagawa `_tf` methods
- Define constants: `LAMBDA_RATIO = 1.2`, `MAX_DRIFT_NORM = 100.0`, etc.
- Remove `observe()` aliases in models

### Priority 3: Cache `R_inv` consistently

**Effort: ~30 min. Impact: Minor speedup per timestep.**

`edh_flow.py:218` recomputes `tf.linalg.inv(R)` every timestep. Add `self.R_inv_cache` as already done in `edh_invertible.py`, `ledh_invertible.py`, and `ledh_flow.py`.

### Priority 4 (Optional): Unify filter hierarchies

**Effort: ~2 days. Impact: Long-term maintainability.**

Merge `FlowFilterBase` subclasses and invertible filters into a single hierarchy. The key difference (equal weights vs importance weights) can be a flag or strategy pattern rather than separate class trees.

---

## Summary Table

| Category | Count | Severity |
|----------|-------|----------|
| Duplicated code blocks across filters | 3 patterns (§2.1, 2.2, 2.3) | High |
| Performance bottlenecks | 6 identified (§3.1–3.6) | 1 High, 3 Medium, 2 Low |
| Dead code / unused imports | 5 instances (§4.1) | Low |
| Magic numbers | 6 locations (§4.2) | Low |
| API inconsistencies | 3 patterns (§4.3) | Medium |
| Design issues | 6 identified (§5.1, 6.1–6.5) | Medium |
