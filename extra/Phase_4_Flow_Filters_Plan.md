# Phase 4: Flow Filters TensorFlow Migration Plan

## Summary
Convert remaining particle flow filters from NumPy to pure TensorFlow.

## Status: Phases 1-3 Complete ✅
- ✅ Phase 1: Core utilities (linalg, distributions, ode_solvers)
- ✅ Phase 2: Models (6 models converted)
- ✅ Phase 3: Kalman filters (KF, EKF, UKF)

## Phase 4: Flow Filters to Convert

### Priority 1: Invertible Flow Filters (Simplest)
These use `compute_flow_weights()` which is already TensorFlow-ready.

1. **edh_invertible.py** - EDH Invertible Particle Flow
   - Uses: compute_flow_weights, euler_step, EKF/UKF
   - Complexity: Medium (resampling, per-particle filters)
   - Lines: ~350

2. **ledh_invertible.py** - LEDH Invertible Particle Flow
   - Uses: compute_flow_weights, euler_step, safe_solve, EKF/UKF
   - Complexity: Medium (local linearization, per-particle filters)
   - Lines: ~400

### Priority 2: Standard Flow Filters
These use ODE integration with drift computation.

3. **edh_flow.py** - EDH Flow Filter
   - Uses: euler_step, EKF/UKF for covariance
   - Complexity: Medium (global drift, no per-particle filters)
   - Lines: ~300

4. **ledh_flow.py** - LEDH Flow Filter
   - Uses: euler_step, local linearization
   - Complexity: Medium-High (local per-particle drift)
   - Lines: ~350

### Priority 3: Stochastic Filter (Most Complex)
5. **stochastic_edh.py** - Stochastic EDH with Optimal Schedule
   - Uses: euler_maruyama_step, scipy.integrate.solve_ivp, scipy.optimize
   - Complexity: High (BVP solver, optimal schedules, SDE integration)
   - Lines: ~600
   - **Issue**: Requires scipy replacements (solve_ivp, root_scalar, CubicSpline)

### Priority 4: Kernel Flow Filters (Optional)
6. **kernel_flow.py** - Kernel-based flow
7. **kernel_local_flow.py** - Local kernel flow
   - Complexity: High (kernel computations, matrix operations)
   - **Decision**: Skip or defer (research code, not production critical)

## Migration Strategy

### Approach A: Incremental (Recommended)
Convert filters in priority order 1→2→3, testing after each.

**Effort Estimate:**
- Priority 1 (invertible): ~2-3 hours per filter = 4-6 hours
- Priority 2 (flow): ~2-4 hours per filter = 4-8 hours
- Priority 3 (stochastic): ~6-8 hours (scipy dependencies)
- **Total: 14-22 hours**

### Approach B: Hybrid (Pragmatic)
Convert Priority 1 & 2 only (skip stochastic_edh if scipy replacement is complex).

**Effort Estimate:** ~8-14 hours

### Approach C: Minimal (Quick Win)
Convert only Priority 1 (invertible filters).

**Effort Estimate:** ~4-6 hours

## Key Conversion Steps (Per Filter)

### 1. Update Imports
```python
import tensorflow as tf
import numpy as np  # Keep for API compatibility
```

### 2. Convert Initialization
- Convert model matrices/constants to `tf.constant`
- Use `tf.Variable` for mutable state (particles, weights)
- Initialize per-particle filters with TensorFlow models

### 3. Convert Core Methods
- Add `@tf.function` to:
  - `_compute_drift()` / `_compute_A_b()`
  - Flow integration loops
  - Weight computation
- Use `tf.map_fn` for per-particle operations
- Replace `np.random` with `tf.random.stateless_*`

### 4. Handle Dependencies
- **Already TensorFlow**: compute_flow_weights, euler_step, EKF/UKF
- **Need attention**: Resampling (convert to TensorFlow or keep NumPy)
- **Scipy issues**: stochastic_edh needs custom BVP solver or skip

### 5. Maintain Compatibility
- Accept/return NumPy arrays at public API (`filter()`, `initialize()`)
- Convert internally: `particles_tf = tf.constant(particles)`
- Convert back: `return particles_tf.numpy()`

## Technical Challenges

### Challenge 1: Per-Particle Filters (invertible variants)
**Current**: Each particle has its own EKF/UKF instance
**Solution**:
- Keep filter objects as-is (already TensorFlow after Phase 3)
- Use `tf.map_fn` or loop over particles for filter updates

### Challenge 2: Resampling
**Current**: Uses NumPy multinomial sampling
**Options**:
- A) Convert to `tf.random.categorical` (preferred)
- B) Keep NumPy resampling (convert to/from numpy at boundary)

### Challenge 3: Scipy Dependencies (stochastic_edh only)
**Current**: Uses `scipy.integrate.solve_ivp`, `scipy.optimize.root_scalar`
**Options**:
- A) Implement TensorFlow BVP solver (complex, ~8+ hours)
- B) Keep NumPy/scipy for optimal schedule computation only
- C) Skip stochastic_edh conversion

## Recommendations

### Option 1: Full Conversion (All Filters)
- Convert Priority 1-3 (skip kernel filters)
- Implement TensorFlow BVP solver for stochastic_edh
- **Time**: 14-22 hours
- **Value**: Complete TensorFlow migration

### Option 2: Pragmatic Conversion (Most Filters)
- Convert Priority 1 & 2 (invertible + standard flow)
- Skip stochastic_edh or hybrid approach (NumPy scipy for BVP)
- **Time**: 8-14 hours
- **Value**: 80% of filters converted, acceptable hybrid for 1 filter

### Option 3: Quick Win (Core Filters Only)
- Convert Priority 1 only (invertible filters)
- These are most commonly used and cleanest to convert
- **Time**: 4-6 hours
- **Value**: Key filters converted, foundation for future work

## My Recommendation: **Option 2 - Pragmatic Conversion**

**Rationale:**
1. Invertible filters are critical and straightforward to convert
2. Standard flow filters complete the core functionality
3. Stochastic_edh is research code with complex scipy dependencies
4. Acceptable to have hybrid approach for optimal schedule computation
5. Kernel filters are not production-critical

**Deliverables:**
- 4 filters fully TensorFlow: edh_invertible, ledh_invertible, edh_flow, ledh_flow
- 1 hybrid filter: stochastic_edh (TensorFlow core, NumPy/scipy for BVP)
- All filters GPU-accelerated via `@tf.function`
- Clean, maintainable codebase

## Next Steps
1. Review and approve plan
2. Start with edh_invertible.py (simplest invertible filter)
3. Test thoroughly before proceeding
4. Continue with ledh_invertible.py
5. Move to flow filters (edh_flow, ledh_flow)
6. Decide on stochastic_edh approach

## Questions for Review
1. Which option do you prefer (1, 2, or 3)?
2. Is stochastic_edh critical for your use case?
3. Should I implement TensorFlow BVP solver or accept hybrid?
4. Are kernel filters used in production?
