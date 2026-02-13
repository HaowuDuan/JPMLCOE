# Resampling Module Refactoring Plan

## Problem Statement

The LEDH invertible filter maintains **per-particle state** (covariances `particle_covs` and optionally per-particle filter objects `particle_filters`). When resampling occurs, these must be duplicated/reassigned according to which original particle each resampled particle descends from — i.e., the **ancestor indices**.

The current resampling functions (`systematic_resample`, `soft_resample`, `ot_entropy_resample`) all compute or implicitly define ancestor relationships internally, but **only return resampled particles and weights** — they discard the indices.

This forces `ledh_invertible.py` lines 410–415 to reverse-engineer indices via an **O(N²) nearest-neighbor search**:

```python
# Current hack in ledh_invertible._resample()
for i in range(self.n_particles):
    dists = np.sum((particles_np - resampled_particles_np[i])**2, axis=1)
    idx = np.argmin(dists)
    indices.append(idx)
```

**Problems with this approach:**
1. **O(N²) cost** — dominant overhead during resampling
2. **Incorrect for OT resampling** — OT creates interpolated particles (`T @ particles`), so there is no single true ancestor. Nearest-neighbor matching silently assigns the wrong ancestor.
3. **Fragile** — if two particles are close in state space but have very different covariances, nearest-neighbor picks the wrong one
4. **Breaks encapsulation** — resampling logic is split between the resampling module and the filter

---

## Design: Unified `ResampleResult`

### Core Idea

All resampling functions return a **`ResampleResult` namedtuple** containing:

```python
ResampleResult = namedtuple('ResampleResult', [
    'particles',          # (N, d) — resampled particles
    'weights',            # (N,) — post-resampling weights
    'ancestor_indices',   # (N,) int or None — which original particle each new one came from
    'transport_matrix',   # (N, N) float or None — for OT-based methods
])
```

**Key distinction by method type:**

| Method | `ancestor_indices` | `transport_matrix` | How LEDH uses it |
|---|---|---|---|
| `systematic_resample` | `indices` (int) | `None` | `new_covs = covs[indices]` |
| `soft_resample` | `indices` (int) | `None` | `new_covs = covs[indices]` |
| `ot_entropy_resample` | `None` | `T` (float NxN) | `new_covs[i] = covs[argmax(T[i])]` or weighted mixture |

---

## File-by-File Changes

### 1. `src/resampling/types.py` (NEW FILE)

Create a lightweight result type:

```python
"""Resampling result types."""
from typing import NamedTuple, Optional
import tensorflow as tf


class ResampleResult(NamedTuple):
    """Standard return type for all resampling methods.

    Fields:
        particles: Resampled particle positions, shape (N, state_dim)
        weights: Post-resampling weights, shape (N,)
        ancestor_indices: Integer tensor of ancestor indices, shape (N,).
            For index-based methods (systematic, soft), this maps each new
            particle to its ancestor: new_particle[i] = old_particles[indices[i]].
            None for transport-based methods (OT entropy).
        transport_matrix: Transport matrix T, shape (N, N).
            For OT-based methods, resampled_particles = T @ old_particles.
            None for index-based methods.
    """
    particles: tf.Tensor
    weights: tf.Tensor
    ancestor_indices: Optional[tf.Tensor] = None
    transport_matrix: Optional[tf.Tensor] = None
```

### 2. `src/resampling/systematic.py`

**Change:** Return `ResampleResult` with `ancestor_indices`.

The indices already exist internally (line 41: `indices = tf.searchsorted(...)`). Currently discarded after `tf.gather`. Just include them in the return.

```python
# Before:
def systematic_resample(particles, weights, seed) -> tf.Tensor:
    ...
    indices = tf.searchsorted(cumsum, u_vals, side='right')
    indices = tf.clip_by_value(indices, 0, N - 1)
    resampled_particles = tf.gather(particles, indices)
    return resampled_particles

# After:
def systematic_resample(particles, weights, seed) -> ResampleResult:
    ...
    indices = tf.searchsorted(cumsum, u_vals, side='right')
    indices = tf.clip_by_value(indices, 0, N - 1)
    resampled_particles = tf.gather(particles, indices)
    uniform_weights = tf.ones(N, dtype=tf.float32) / N_float
    return ResampleResult(
        particles=resampled_particles,
        weights=uniform_weights,
        ancestor_indices=indices,
        transport_matrix=None,
    )
```

**Also:** Remove `systematic_resample_with_weights` — it becomes redundant since `ResampleResult` always includes weights.

### 3. `src/resampling/soft.py`

**Change:** Return `ResampleResult` with `ancestor_indices`.

Same situation — `indices` exists on line 47 but is discarded after gather.

```python
# Before:
def soft_resample(particles, weights, alpha, seed) -> tuple:
    ...
    indices = tf.searchsorted(cumsum, u_vals, side='right')
    indices = tf.clip_by_value(indices, 0, N - 1)
    resampled_particles = tf.gather(particles, indices)
    ...
    return resampled_particles, new_weights

# After:
def soft_resample(particles, weights, alpha, seed) -> ResampleResult:
    ...
    indices = tf.searchsorted(cumsum, u_vals, side='right')
    indices = tf.clip_by_value(indices, 0, N - 1)
    resampled_particles = tf.gather(particles, indices)
    ...
    return ResampleResult(
        particles=resampled_particles,
        weights=new_weights,
        ancestor_indices=indices,
        transport_matrix=None,
    )
```

### 4. `src/resampling/ot_entropy.py`

**Change:** Return `ResampleResult` with `transport_matrix`.

OT resampling creates interpolated particles (`T @ particles`), so there are no discrete ancestor indices. Return the transport matrix instead.

```python
# Before:
def ot_entropy_resample(particles, weights, epsilon, seed, ...) -> Tuple[tf.Tensor, tf.Tensor]:
    ...
    T = compute_transport_matrix_with_gradient(...)
    resampled_particles = T @ particles
    uniform_weights = tf.ones(N, dtype=tf.float32) / N_float
    return resampled_particles, uniform_weights

# After:
def ot_entropy_resample(particles, weights, epsilon, seed, ...) -> ResampleResult:
    ...
    T = compute_transport_matrix_with_gradient(...)
    resampled_particles = T @ particles
    uniform_weights = tf.ones(N, dtype=tf.float32) / N_float
    return ResampleResult(
        particles=resampled_particles,
        weights=uniform_weights,
        ancestor_indices=None,
        transport_matrix=T,
    )
```

### 5. `src/resampling/__init__.py`

**Change:** Export `ResampleResult`.

```python
from .types import ResampleResult
from .systematic import systematic_resample
from .soft import soft_resample
from .ot_entropy import ot_entropy_resample
from .diagnosis import effective_sample_size, normalize_log_weights

__all__ = [
    'ResampleResult',
    'systematic_resample',
    'soft_resample',
    'ot_entropy_resample',
    'effective_sample_size',
    'normalize_log_weights'
]
```

### 6. `src/filters/particle/ledh_invertible.py` — `_resample()`

**Change:** Replace O(N²) hack with direct index/transport usage.

```python
# Before (lines 387-431):
def _resample(self):
    seed = ...
    result = self.resampling_method(self.particles.value(), self.weights.value(), seed=seed, **self.resampling_config)

    # Handle different return types
    if isinstance(result, tuple):
        resampled_particles, new_weights = result
    else:
        resampled_particles = result
        new_weights = ...

    # O(N²) hack to find indices
    particles_np = self.particles.numpy()
    resampled_particles_np = resampled_particles.numpy()
    indices = []
    for i in range(self.n_particles):
        dists = np.sum((particles_np - resampled_particles_np[i])**2, axis=1)
        idx = np.argmin(dists)
        indices.append(idx)

    # Resample covs/filters using indices
    ...

# After:
def _resample(self):
    seed = ...
    result = self.resampling_method(self.particles.value(), self.weights.value(), seed=seed, **self.resampling_config)

    # Extract from ResampleResult (or legacy tuple)
    if hasattr(result, 'ancestor_indices'):
        # New ResampleResult interface
        resampled_particles = result.particles
        new_weights = result.weights

        if result.ancestor_indices is not None:
            # Index-based (systematic, soft): direct gather
            indices = result.ancestor_indices.numpy()
        elif result.transport_matrix is not None:
            # Transport-based (OT): use dominant ancestor per row
            T_np = result.transport_matrix.numpy()
            indices = np.argmax(T_np, axis=1)
        else:
            raise ValueError("ResampleResult has neither indices nor transport matrix")
    elif isinstance(result, tuple):
        # Legacy tuple interface (backward compat)
        resampled_particles, new_weights = result
        # Fallback to nearest-neighbor (legacy path)
        ...
    else:
        resampled_particles = result
        new_weights = tf.ones(self.n_particles, ...) / ...
        ...

    # Resample covariances
    new_covs = self.particle_covs[indices]  # vectorized numpy fancy indexing
    self.particle_covs = new_covs.copy()

    # Resample filters (if using UKF per-particle filters)
    if self.particle_filters is not None:
        resampled_particles_np = resampled_particles.numpy()
        new_filters = []
        for i in range(self.n_particles):
            filt = self._create_filter(resampled_particles_np[i], new_covs[i])
            new_filters.append(filt)
        self.particle_filters = new_filters

    self.particles.assign(resampled_particles)
    self.weights.assign(new_weights)
```

**Performance improvement:** O(N²) loop replaced by O(N) `np.argmax` (for OT) or O(1) direct index use (for systematic/soft).

### 7. `src/filters/particle/edh_invertible.py` — `_resample()`

**Change:** Update to handle `ResampleResult`. EDH doesn't need indices (no per-particle state), so only needs particles and weights.

```python
# Before:
def _resample(self):
    ...
    result = self.resampling_method(...)
    if isinstance(result, tuple):
        resampled_particles, new_weights = result
    else:
        resampled_particles = result
        new_weights = ...

# After:
def _resample(self):
    ...
    result = self.resampling_method(...)
    if hasattr(result, 'particles'):
        # ResampleResult
        resampled_particles = result.particles
        new_weights = result.weights
    elif isinstance(result, tuple):
        resampled_particles, new_weights = result
    else:
        resampled_particles = result
        new_weights = tf.ones(self.n_particles, ...) / ...

    self.particles.assign(resampled_particles)
    self.weights.assign(new_weights)
```

### 8. `src/filters/particle/bootstrap_pf_tf.py` — `_resample()`

**Change:** Same pattern as EDH — extract particles/weights from `ResampleResult`.

---

## Backward Compatibility

The `ResampleResult` is a `NamedTuple`, which means:
- It is iterable: `particles, weights = result[:2]` still works for legacy code
- It is indexable: `result[0]` returns particles
- `isinstance(result, tuple)` returns `True` — legacy `if isinstance(result, tuple)` checks still pass

However, legacy unpacking `particles, weights = result` will fail because `ResampleResult` has 4 fields. To handle this transition:
- Keep the `if hasattr(result, 'particles')` check in filters to detect new vs legacy format
- Once all filters are updated, remove the legacy branches

---

## OT Resampling: Covariance Resampling Strategy

For OT entropy resampling, the transport matrix T produces interpolated particles. For per-particle covariances, two options:

**Option A: Dominant ancestor (simple, recommended)**
```python
indices = np.argmax(T_np, axis=1)
new_covs = old_covs[indices]
```
- Assigns each new particle the covariance of its highest-weight ancestor
- Fast (O(N²) for argmax, but vectorized)
- Loses the interpolation information, but covariances are approximate anyway

**Option B: Weighted mixture (more accurate, more expensive)**
```python
# T_np shape: (N, N), old_covs shape: (N, d, d)
new_covs = np.einsum('ij,jkl->ikl', T_np, old_covs)
```
- Each new covariance is a weighted average of ancestor covariances
- Respects the transport plan fully
- O(N² * d²) cost, but vectorized

**Recommendation:** Start with Option A. Option B can be added later behind a config flag if needed.

---

## `@tf.function` Compatibility Note

`ResampleResult` is a Python NamedTuple. Inside `@tf.function`:
- Returning a NamedTuple from a tf.function works (TF traces named tuples as structured outputs)
- The `None` default values for `ancestor_indices` and `transport_matrix` work in graph mode since tf.function handles Python None as "not a tensor"
- The `ancestor_indices` field is an int32 tensor, while other fields are float32 — this is fine for NamedTuple returns

**Potential issue:** `systematic_resample` is currently decorated with `@tf.function`. Returning a NamedTuple with a `None` field from `@tf.function` works because TF treats it as a Python constant. Verify this doesn't cause retracing.

---

## Execution Order

1. Create `src/resampling/types.py` with `ResampleResult`
2. Update `src/resampling/systematic.py` — return `ResampleResult` with indices
3. Update `src/resampling/soft.py` — return `ResampleResult` with indices
4. Update `src/resampling/ot_entropy.py` — return `ResampleResult` with transport matrix
5. Update `src/resampling/__init__.py` — export `ResampleResult`
6. Update `src/filters/particle/edh_invertible.py` — handle `ResampleResult`
7. Update `src/filters/particle/bootstrap_pf_tf.py` — handle `ResampleResult`
8. Update `src/filters/particle/ledh_invertible.py` — replace O(N²) hack with direct index use
9. Remove `systematic_resample_with_weights` (redundant)
10. Run existing experiments to verify no regressions

---

## Summary

| What | Before | After |
|---|---|---|
| Systematic return | `particles` only | `ResampleResult(particles, weights, indices, None)` |
| Soft return | `(particles, weights)` | `ResampleResult(particles, weights, indices, None)` |
| OT entropy return | `(particles, weights)` | `ResampleResult(particles, weights, None, T)` |
| LEDH index recovery | O(N²) nearest-neighbor hack | O(1) direct use (systematic/soft) or O(N) argmax (OT) |
| Filter compatibility | `isinstance(result, tuple)` checks | `hasattr(result, 'particles')` + legacy fallback |
