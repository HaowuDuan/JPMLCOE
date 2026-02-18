# DPF Codebase Refactor Plan

> Based on CODE_ANALYSIS.md (2026-02-14). Investigated 2026-02-17.
> Of 26 original issues: **4 fixed**, **3 partially fixed**, **19 still exist**.

---

## Priority 1: Extract Shared Utilities (§2.1, §2.2, §4.3)

**Goal:** Eliminate copy-pasted blocks across 5 filter files by extracting to shared helpers.

### 1A. Resampling config helper

**Problem:** Identical 25-line resampling-method-resolution + config-scalar-conversion block is copy-pasted in all 5 filters. Three of them (edh_flow, ledh_flow, stochastic_edh) are equal-weight filters that never resample — the code is dead there.

**Fix:** Create `src/utils/resampling_config.py` with a helper function. Delete dead code from equal-weight filters.

```python
# src/utils/resampling_config.py
import numpy as np
from ..resampling import systematic_resample, soft_resample, ot_entropy_resample

_METHOD_MAP = {
    'systematic': systematic_resample,
    'soft': soft_resample,
    'ot_entropy': ot_entropy_resample,
}

def resolve_resampling(resampling_method, resampling_config):
    """
    Resolve resampling method string/callable and sanitize config scalars.

    Returns:
        (method_fn, method_name, config_dict)
    """
    if isinstance(resampling_method, str):
        method_fn = _METHOD_MAP.get(resampling_method, systematic_resample)
        method_name = resampling_method
    elif resampling_method is not None:
        method_fn = resampling_method
        method_name = getattr(resampling_method, '__name__', 'custom')
    else:
        method_fn = systematic_resample
        method_name = 'systematic'

    config = {}
    if resampling_config is not None:
        for key, value in resampling_config.items():
            if isinstance(value, (int, np.integer)):
                config[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                config[key] = float(value)
            else:
                config[key] = value

    return method_fn, method_name, config
```

**Then in invertible filters (2 files):**
```python
from ...utils.resampling_config import resolve_resampling

# In __init__:
self.resampling_method, self.resampling_method_name, self.resampling_config = (
    resolve_resampling(resampling_method, resampling_config)
)
```

**In equal-weight filters (edh_flow, ledh_flow, stochastic_edh):**
Delete the entire resampling config block — they never call `self.resampling_method`.

**Why:** Eliminates 125 lines of duplication (25 lines × 5 files). Ensures any future resampling method only needs to be registered in one place.

### 1B. Particle initialization helper

**Problem:** Identical Cholesky-based particle sampling code in 4 files:
- `ledh_invertible.py:160-168`
- `edh_invertible.py:165-173`
- `ledh_flow.py:131-144`
- `edh_flow.py:124-133`

All do: `L = safe_cholesky(cov); z = stateless_normal; particles = mean + z @ L^T`.

**Fix:** Add to `src/utils/distributions.py`:

```python
def sample_particles_cholesky(
    initial_mean: tf.Tensor,
    initial_cov: tf.Tensor,
    n_particles: int,
    state_dim: int,
    seed: tf.Tensor,
    dtype=tf.float64
) -> tf.Tensor:
    """Sample N particles from N(mean, cov) via Cholesky decomposition."""
    L = safe_cholesky(initial_cov)
    z = tf.random.stateless_normal([n_particles, state_dim], seed=seed, dtype=dtype)
    return initial_mean + tf.linalg.matmul(z, L, transpose_b=True)
```

**Why:** Single source of truth for particle initialization. If we ever switch to e.g. SVD-based sampling for ill-conditioned covariances, we change it in one place.

### 1C. `to_numpy` utility

**Problem:** Pattern `y.numpy() if isinstance(y, tf.Tensor) else y` scattered across 5+ files.

**Fix:** Add to `src/utils/linalg.py` (or a new `src/utils/tensor_utils.py`):

```python
def to_numpy(x):
    """Convert TF tensor to numpy; pass through if already numpy/scalar."""
    return x.numpy() if isinstance(x, tf.Tensor) else x
```

**Why:** One-liner, but it's about consistency and grep-ability. Makes the intent obvious at every call site.

---

## Priority 2: Fix `filter()` Duplication (§2.3)

**Problem:** `ledh_invertible.py:387-434` and `edh_invertible.py:349-396` are virtually identical filter loops (init, predict/update, stack, return FilterResult). The only difference is the metadata `filter_type` string.

**Fix:** Extract to a shared mixin or base class method.

**Option A — Mixin (recommended, minimal disruption):**

```python
# src/filters/particle/invertible_filter_mixin.py
import time
import numpy as np
import tensorflow as tf
from typing import Optional, Callable
from ...core.types import FilterResult

class InvertibleFilterMixin:
    """Shared filter() and _estimate_mean_cov() for invertible flow filters."""

    def _estimate_mean_cov(self):
        """Weighted mean and covariance (TF ops)."""
        particles = self.particles.value()
        weights = self.weights.value()
        mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
        diff = particles - mean
        cov = tf.reduce_sum(
            weights[:, tf.newaxis, tf.newaxis] *
            tf.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )
        return mean, cov

    def filter(self, observations: np.ndarray,
               initial_mean=None, initial_cov=None,
               random_seed=None, progress_callback=None) -> FilterResult:
        """Run filter on sequence of observations."""
        self.initialize(initial_mean, initial_cov, random_seed)
        T = len(observations)
        obs_tf = tf.constant(observations, dtype=self.dtype)

        for t in range(T):
            t0 = time.perf_counter()
            self.predict()
            self.update(obs_tf[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
            if progress_callback is not None:
                progress_callback(t, T, time.perf_counter() - t0)

        resampling_rate = len(self.resampled_at) / T if T > 0 else 0.0
        means_np = tf.stack(self.means).numpy()
        covs_np = tf.stack(self.covs).numpy()
        log_liks_tf = tf.stack(self.log_likelihoods) if self.log_likelihoods else None
        ess_np = tf.stack(self.ess_history).numpy()
        weights_np = tf.stack(self.weights_history).numpy()

        return FilterResult(
            means=means_np,
            covs=covs_np,
            log_likelihood=float(tf.reduce_sum(log_liks_tf).numpy()) if log_liks_tf is not None else None,
            log_likelihoods=log_liks_tf.numpy() if log_liks_tf is not None else None,
            ess=ess_np,
            weights_history=weights_np,
            resampled_at=self.resampled_at,
            n_unique=np.array(self.n_unique_particles) if self.n_unique_particles else None,
            metadata={
                'filter_type': self.__class__.__name__,
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'resampling_method': self.resampling_method_name,
                'resampling_rate': resampling_rate
            }
        )
```

**Then:**
```python
class LEDHParticleFlowFilter(InvertibleFilterMixin):
    # Delete filter() and _estimate_mean_cov() — inherited from mixin
    ...

class EDHParticleFlowFilter(InvertibleFilterMixin):
    # Delete filter() and _estimate_mean_cov() — inherited from mixin
    ...
```

**Why:** Eliminates ~100 lines of identical code. Uses `self.__class__.__name__` for metadata so no per-class override needed. `LEDHParticleFlowFilterHMC` already inherits from `LEDHParticleFlowFilter`, so it gets this for free.

---

## Priority 3: Fix R_inv Caching (§3.3)

**Problem:** `edh_flow.py` declares `self.R_inv_cache = None` but never uses it — recomputes `safe_inv(R)` every timestep. `stochastic_edh.py` has no caching at all.

**Files to fix:**

### 3A. `edh_flow.py:232-233`

**Current:**
```python
# line 233
R_inv_tf = safe_inv(R_tf)
```

**Fix:**
```python
if self.R_inv_cache is None:
    self.R_inv_cache = safe_inv(R_tf)
R_inv_tf = self.R_inv_cache
```

### 3B. `stochastic_edh.py:190`

**Current:**
```python
R_inv = safe_inv(R)
```

**Fix:**
```python
if self.R_inv_cache is None:
    self.R_inv_cache = safe_inv(R)
R_inv = self.R_inv_cache
```

Note: `stochastic_edh.py` inherits from `edh_flow.py` which already declares `self.R_inv_cache = None`. So the field already exists.

**Why:** R is constant across timesteps (observation noise covariance). Computing its inverse once saves one matrix inversion per timestep. For high-dimensional observation spaces this matters; for 1D Kitagawa it's negligible but still good practice.

---

## Priority 4: Delete Dead Code (§4.1)

Simple deletions — no risk to functionality.

### 4A. `ledh_flow.py:6` — Remove unused import

```python
# DELETE this line:
from concurrent.futures import ThreadPoolExecutor
```

### 4B. `ledh_flow.py:186-220` — Remove dead `_compute_drift_single()`

This method is never called anywhere. The batched `_flow_step_euler()` is used instead.

```python
# DELETE the entire _compute_drift_single method (lines 186-220)
```

### 4C. `edh_flow.py:10` — Remove unused `rk4_step` import

```python
# CHANGE:
from ...utils.ode_solvers import euler_step, rk4_step
# TO:
from ...utils.ode_solvers import euler_step
```

**Why:** Dead code obscures the actual control flow and confuses readers. These are safe deletions — nothing calls them.

---

## Priority 5: Fix `tf.range` in Python Loop (§4.6)

**Problem:** `edh_invertible.py:252` uses `tf.range()` in a Python for-loop.

```python
# CURRENT (edh_invertible.py:252):
for j in tf.range(self.n_lambda_steps):

# FIX:
for j in range(self.n_lambda_steps):
```

**Why:** `tf.range()` creates a TF tensor, then Python iterates over it in eager mode, creating individual tensor objects for each index. `range()` uses plain Python integers — faster and cleaner. `ledh_invertible.py:233` already does this correctly.

---

## Priority 6: Fix Kitagawa `self.t` Mutable State (§6.1)

**Problem:** `kitagawa.py` stores time step as `self.t` (line 62), which is:
- Mutated by `sample_state_transition()` (line 104: `self.t += 1`)
- Set externally by filters: `self.model.t = t + 1` (ledh_invertible.py:377)
- Not thread-safe
- Creates hidden coupling between filters and models

**Fix:** Make `t` an explicit parameter everywhere. Keep `self.t` for backward compatibility but deprecate the mutation pattern.

### Step 1: Add `t` parameter to batch methods that need it

```python
# kitagawa.py — add t parameter with fallback to self.t
def state_transition_batch(self, particles, seed, t=None):
    sigma_V = _as_sigma(self.sigma_V, self.dtype)
    t_val = tf.cast(t if t is not None else self.t, self.dtype)
    x = particles[:, 0:1]
    mean = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
    w = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
    return mean + sigma_V * w

def state_transition_mean_batch(self, particles, t=None):
    t_val = tf.cast(t if t is not None else self.t, self.dtype)
    x = particles[:, 0:1]
    return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)

def state_transition_mean(self, x, t=None):
    t_val = tf.cast(t if t is not None else self.t, self.dtype)
    x0 = x[0] if len(x.shape) == 1 else x
    mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
    return tf.reshape(mean, tf.shape(x))
```

### Step 2: Update model_base.py signatures

```python
# model_base.py — add optional t parameter to base class signatures
def state_transition_batch(self, particles, seed, t=None):
    ...
def state_transition_mean_batch(self, particles, t=None):
    ...
```

### Step 3: Update filter loops to pass `t` explicitly

```python
# In filter loops (ledh_invertible.py, edh_invertible.py, etc.):
# CURRENT:
if hasattr(self.model, 't'):
    self.model.t = t + 1
self.predict()

# FIX:
self.predict(t=t+1)

# In predict():
def predict(self, t=None):
    seed = ...
    eta_0 = self.model.state_transition_batch(self.particles.value(), seed, t=t)
    ...
```

**Why:** Eliminates hidden state mutation. Makes the time-dependence explicit. Allows running multiple filters on the same model concurrently. The `t=None` fallback preserves backward compatibility so nothing breaks during migration.

---

## Priority 7: Fix RNG Counter Pattern (§6.5)

**Problem:** All particle filters use `self.seed_counter += 1` with `tf.constant([counter, 0])`. Neighboring integer seeds are not guaranteed to produce independent random streams.

**Current pattern (8+ files, ~20 call sites):**
```python
seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
self.seed_counter += 1
z = tf.random.stateless_normal(shape, seed=seed)
```

**Fix:** Use `tf.random.experimental.stateless_split` (already used in `model_base.py`):

```python
# In initialize():
self.rng_key = tf.constant([random_seed or 42, 0], dtype=tf.int32)

# Helper method (add to mixin or base):
def _next_seed(self):
    """Split RNG key, return subkey for use."""
    keys = tf.random.experimental.stateless_split(self.rng_key, num=2)
    self.rng_key = keys[0]
    return keys[1]

# Usage:
seed = self._next_seed()
z = tf.random.stateless_normal(shape, seed=seed)
```

**Files to update:** `ledh_invertible.py`, `edh_invertible.py`, `edh_flow.py`, `ledh_flow.py`, `stochastic_edh.py`, `sde_local_correction.py`, `ledh_invertible_hmc.py`

**Why:** Hash-based splitting (Threefry/Philox) guarantees statistically independent streams. The counter-increment approach produces correlated seeds (`[42,0]`, `[43,0]`, `[44,0]`...) which are adjacent inputs to the same hash — not a guaranteed problem in practice but theoretically unsound. This also eliminates the numpy RNG dependency in `flow_base.py:77`.

---

## Priority 8: Define Constants for Magic Numbers (§4.2)

**Problem:** Hardcoded values scattered across files with no explanation:
- `q = 1.2` in `_generate_lambda_steps()` (5 files)
- `100.0`, `1000.0`, `1e-10` in `ledh_flow.py` (drift/particle clipping)
- `0.1`, `20.0`, `50.0` bisection bounds in `stochastic_edh.py`
- `500` ODE steps, `40` bisection iterations in `stochastic_edh.py`

**Fix:** Define as class-level constants or constructor parameters.

```python
# In each filter or in a shared constants module:
LAMBDA_GEOMETRIC_RATIO = 1.2      # Paper's recommended exponential schedule ratio
MAX_DRIFT_NORM = 100.0            # Maximum drift magnitude per flow step
MAX_PARTICLE_NORM = 1000.0        # Maximum particle distance from origin
DRIFT_EPSILON = 1e-10             # Avoid division by zero in drift clipping

# In stochastic_edh.py __init__:
class StochasticEDHFlow(ExactDaumHuangFlow):
    BVP_N_STEPS = 500              # Euler steps for BVP shooting integration
    BVP_N_BISECTION = 40           # Bisection iterations for initial velocity
    BVP_BRACKET_LO = 0.1           # Initial lower bracket for bisection
    BVP_BRACKET_HI = 20.0          # Initial upper bracket for bisection
    BVP_BRACKET_HI_MAX = 50.0      # Extended upper bracket if initial fails
```

**Why:** Makes the code self-documenting. Anyone reading `_shoot(u0, n_steps=self.BVP_N_STEPS)` understands it's a tunable parameter, not an arbitrary magic number.

---

## Priority 9: Populate Debug Info in Invertible Filters (§4.5)

**Problem:** `ledh_invertible.py` and `edh_invertible.py` allocate `self.debug_info` dict with 11 keys but never write to any of them. The flow-based filters (`edh_flow.py`, `ledh_flow.py`) do populate theirs.

**Fix:** Either populate them or remove the dead allocation.

**Recommendation:** Remove the allocation. If debug info is needed for invertible filters, add it when actually implementing the diagnostics.

```python
# In ledh_invertible.py and edh_invertible.py __init__:
# CURRENT (lines 115-130):
if self.debug_mode:
    self.debug_info = { ... 11 keys ... }
else:
    self.debug_info = None

# FIX — just remove the block entirely, or simplify to:
self.debug_info = {} if self.debug_mode else None
```

**Why:** Allocating 11 empty lists that are never written to is misleading. It implies diagnostics are available when they're not.

---

## Priority 10: Simplify `observe()` Chain (§6.3)

**Problem:** Three methods return the same value:
- `model_base.py:136-138` defines `observe()` → delegates to `observation_function()`
- `kitagawa.py:160-161` defines `observation_function()` → delegates to `observation_mean()`
- Both `observation_mean` and `observation_function` are abstract in the base class

**Fix:** Remove `observe()` from `model_base.py` if nothing uses it, or add a comment explaining which interface each caller expects.

```python
# Check what calls observe():
# grep -r "\.observe(" shows it's used by kernel_flow.py
# So we keep it but add clarity:

# model_base.py:
def observe(self, x: tf.Tensor) -> tf.Tensor:
    """Alias for observation_function(). Used by KernelMappingPF."""
    return self.observation_function(x)
```

In `kitagawa.py`, the `observation_function()` override is fine — it delegates to `observation_mean()` which is the actual implementation. No change needed there since the base class requires `observation_function` to be implemented.

**Why:** Low priority but reduces confusion when reading the code. The real fix is documenting which interface each filter family expects.

---

## Priority 11: Cache BVP Schedule (§3.6)

**Problem:** `stochastic_edh.py:_compute_optimal_schedule()` runs per-timestep with 40 bisection iterations × 500 Euler steps = 20,000 ops per timestep. With T=100 timesteps, that's 2M Euler steps just for scheduling.

**Fix:** Cache the schedule when P and H don't change significantly.

```python
# In stochastic_edh.py:
def update(self, y):
    ...
    R_inv = ...

    # Cache optimal schedule — only recompute if P changed significantly
    if self.schedule_mu > 0:
        P_hash = float(tf.linalg.trace(P))  # Cheap proxy for "did P change?"
        if not hasattr(self, '_cached_schedule') or abs(P_hash - self._cached_P_hash) > 1e-6 * abs(self._cached_P_hash):
            self._cached_beta, self._cached_dbeta = self._compute_optimal_schedule(P, R_inv)
            self._cached_P_hash = P_hash
        beta_vals, dbeta_vals = self._cached_beta, self._cached_dbeta
```

**Why:** For models where P evolves slowly (common in practice), this eliminates redundant BVP solves. The trace is a cheap 1D summary — if trace(P) hasn't changed, the full matrix hasn't changed much either.

---

## Priority 12 (Optional): Refactor `compute_flow_weights` (§5.1)

**Problem:** 103-line function with two nearly-identical Cholesky solve blocks (lines 160-169 and 171-180).

**Fix:** Extract a helper for the log-Gaussian-prob computation:

```python
def _log_transition_prob(diff, L_Q, state_dim, dtype):
    """Log p(x|mu, Q) for batch of particles, given L_Q = cholesky(Q)."""
    y = tf.linalg.triangular_solve(L_Q, tf.transpose(diff), lower=True)
    y = tf.transpose(y)
    return -0.5 * (
        tf.reduce_sum(y**2, axis=1) +
        2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Q))) +
        tf.cast(state_dim, dtype) * tf.math.log(2.0 * tf.constant(math.pi, dtype=dtype))
    )
```

Then `compute_flow_weights` becomes:
```python
L_Q = safe_cholesky(Q)
log_p_eta1 = _log_transition_prob(eta_1 - f_prev, L_Q, state_dim, eta_1.dtype)
log_p_eta0 = _log_transition_prob(eta_0 - f_prev, L_Q, state_dim, eta_1.dtype)
```

**Why:** Reduces `compute_flow_weights` by ~15 lines and makes the symmetry between eta_1 and eta_0 computation obvious.

---

## Summary — Execution Order

| Step | Files Changed | Lines Removed | Lines Added | Risk |
|------|--------------|--------------|-------------|------|
| P1A: Resampling helper | 6 (new util + 5 filters) | ~125 | ~35 | Low |
| P1B: Particle init helper | 5 (util + 4 filters) | ~40 | ~15 | Low |
| P1C: to_numpy utility | 6 (util + 5 call sites) | ~5 | ~5 | None |
| P2: filter() mixin | 3 (new mixin + 2 filters) | ~100 | ~45 | Medium |
| P3: R_inv caching | 2 (edh_flow, stochastic_edh) | 0 | ~6 | None |
| P4: Dead code deletion | 2 (ledh_flow, edh_flow) | ~40 | 0 | None |
| P5: tf.range fix | 1 (edh_invertible) | 0 | 0 (change 1 word) | None |
| P6: Kitagawa self.t | 3+ (kitagawa, model_base, filters) | ~5 | ~20 | Medium |
| P7: RNG counter | 7 (all particle filters) | ~20 | ~30 | Medium |
| P8: Magic numbers | 5 (filters + stochastic_edh) | 0 | ~15 | None |
| P9: Debug info cleanup | 2 (invertible filters) | ~30 | ~2 | None |
| P10: observe() docs | 1 (model_base) | 0 | ~2 | None |
| P11: BVP cache | 1 (stochastic_edh) | 0 | ~8 | Low |
| P12: flow_weights refactor | 1 (distributions) | ~15 | ~12 | Low |

**Recommended batches:**
1. **Quick wins (P3, P4, P5, P8, P9, P10):** ~30 min, zero risk, immediate cleanup
2. **Shared utilities (P1A, P1B, P1C, P2):** ~2 hours, deduplicates ~250 lines
3. **Design fixes (P6, P7):** ~3 hours, requires updating all filter files + tests
4. **Performance (P11, P12):** ~1 hour, targeted optimizations
