# Plan: TF-Native Cleanup — Remove NumPy Legacy from Codebase

## Context

The codebase evolved from a pure NumPy prototype (`code_backup_np/`) to a TensorFlow-based project. Most filters and some models were properly ported to TF, but several models and one filter (`KernelMappingPF`) retain NumPy-only code. This creates:
- Type mismatches (models returning `np.ndarray` fed to filters expecting `tf.Tensor`)
- Inconsistent APIs (`observe()` vs `observation_function()`, `initial_state_mean` vs `mu_0`)
- `KernelMappingPF` using `scipy.linalg` while every other filter uses `tf.linalg`
- Missing abstract methods on the base class that are implicitly required

**Goal:** Make all models and filters consistently TF-native internally, following the pattern of `RangeBearingModel` and `TwoSensorBearingOnlyModel` (the gold-standard TF models).

**Boundary contract stays the same:** `filter(observations: np.ndarray) -> FilterResult(means: np.ndarray, ...)` — numpy at the public edges, TF internally.

---

## Phase 0: Base Class — Add Missing Abstract Methods

**File:** `src/core/model_base.py`

The base class defines the contract. Currently `observation_function`, `observation_noise_cov`, `process_noise_cov`, `mu_0`, and `Sigma_0` are NOT abstract but are called by filters and the experiment runner. Make them explicit:

- Add `observation_function(x: tf.Tensor) -> tf.Tensor` as abstract method
- Add `observation_noise_cov` as abstract property -> `tf.Tensor`
- Add `process_noise_cov` as abstract property -> `tf.Tensor`
- Add `mu_0` as abstract property -> `tf.Tensor`
- Add `Sigma_0` as abstract property -> `tf.Tensor`
- Add `observe(x)` as default method delegating to `observation_function(x)` (backward compat for any external callers)

---

## Phase 1: Lorenz96Model — Full TF Port (biggest change)

**File:** `src/models/lorenz96.py`

Currently 100% numpy. No `dtype` param. Takes `np.random.Generator`, returns `np.ndarray`.

Changes:
- Add `dtype` param (default `tf.float64` for chaotic precision)
- Convert `H`, `R`, `Q` to `tf.constant` in constructor
- Add `mu_0` property (climatological mean as TF tensor)
- Add `Sigma_0` property (`initial_spread^2 * I` as TF tensor)
- Port `_lorenz96_tendency` to TF: use `tf.roll(x, k, axis=-1)` (equivalent to `np.roll`)
- Port `sample_initial_state(seed)`, `sample_state_transition(x, seed)`, `sample_observation(x, seed)` to TF — keep numpy spinup cached as `tf.constant` (one-time cost)
- Port all deterministic methods (`state_transition_mean`, `observation_mean`, `observation_jacobian`, etc.) to TF
- Port `log_observation_prob` to TF
- Port batch methods — `_lorenz96_tendency` with `axis=-1` handles `(N, state_dim)` batches natively
- Add `observation_function`, `observation_noise_cov`, `process_noise_cov` properties
- Import `rk4_step` from `src/utils/ode_solvers.py` (already TF-native) instead of numpy version
- Remove numpy-only helpers: `_get_spinup_state(rng)`, `integrate_steps(x, n_steps, rng)`, `spinup(n_steps, rng)`

---

## Phase 2: StochasticVolatilityModel — TF Port

**File:** `src/models/stochastic_volatility.py`

Currently numpy primary with `_tf` suffix duplicates. Has `dtype` but ignores it.

Changes:
- Remove `TF_AVAILABLE` guard (TF is a hard dependency)
- Store `alpha`, `sigma`, `beta` as `tf.constant` with proper dtype
- Make all core methods TF-native (remove numpy implementations)
- Delete `_tf` suffix methods — their logic becomes the primary methods
- Add `mu_0` -> `tf.zeros([1])`, `Sigma_0` -> stationary variance
- Convert `observation_noise_cov`, `process_noise_cov` to return `tf.Tensor`
- Add `observation_function` returning `tf.zeros_like(x)`
- Port batch methods from numpy to TF

---

## Phase 3: AcousticTrackingModel — TF Port

**File:** `src/models/acoustic_tracking.py`

Currently numpy primary with `_tf` suffix duplicates. Uses `initial_state_mean`/`initial_state_cov` (numpy) instead of `mu_0`/`Sigma_0`.

Changes:
- Remove `TF_AVAILABLE` guard
- Convert `F`, `Q`, `R`, `sensor_positions` to `tf.constant`
- Replace `initial_state_mean`/`initial_state_cov` properties with `mu_0`/`Sigma_0` returning TF tensors
- Make all core methods TF-native
- Delete `_tf` suffix methods
- Port batch methods from numpy to TF
- Add `observation_function` method

---

## Phase 4: KitagawaModel — Remove Dual-Mode

**File:** `src/models/kitagawa.py`

Has TF support throughout but every method has `isinstance`/`hasattr` checks for numpy fallback.

Changes:
- Remove numpy branches from all methods (keep only TF paths)
- Remove numpy helper methods: `_f`, `_df_dx`, `_h`, `_dh_dx`
- Remove `isinstance(x, tf.Tensor)` / `hasattr(rng, 'standard_normal')` guards
- Remove `_tf` suffix legacy methods
- Already has `mu_0`, `Sigma_0`, `observation_noise_cov`, `observation_function` — just clean up their implementations

---

## Phase 5: LinearGaussianModel — Minor Cleanup

**File:** `src/models/linear_gaussian.py`

Mostly clean. Has `hasattr(seed, 'standard_normal')` checks in sampling methods for a numpy path that `generate_data` never actually calls.

Changes:
- Remove numpy branches from `sample_initial_state`, `sample_state_transition`, `sample_observation`
- Inline `_sample_*_tf` methods into the primary methods

---

## Phase 6: KernelMappingPF — Convert from numpy/scipy to TF

**File:** `src/filters/particle/kernel_flow.py`

The only filter using numpy/scipy internally. Depends on all models being TF-native (Phases 1-5).

Changes:
- Remove `from scipy import linalg`
- Store particles as `tf.Variable` (not `np.ndarray`)
- Convert `_compute_localization_matrix` result to `tf.constant`
- Port `_compute_prior_stats` to TF (`tf.reduce_mean`, `tf.matmul`, Schur product)
- Port `_update_matrix` inner loop:
  - Replace `model.observe(p) for p in particles` loop with `model.observation_function_batch(particles)`
  - Replace `model.observation_jacobian(particles[i])` loop with `model.observation_jacobian_batch(particles)`
  - Replace `np.einsum` -> `tf.einsum`
  - Replace `scipy.linalg.pinv/inv` -> `tf.linalg.pinv/inv`
  - Replace `scipy.linalg.norm` -> `tf.norm`
  - Replace all `np.exp/sum/mean` -> `tf.exp/reduce_sum/reduce_mean`
- Port `_update_scalar` — same pattern
- Port `initialize` to use `model.sample_initial_state_batch(n, seed)`
- Port `predict` to use `model.state_transition_batch(particles, seed)`
- `filter()` converts TF outputs to numpy for `FilterResult` (same as all other filters)
- `get_mean()`/`get_covariance()` return numpy for FilterResult compat

---

## Phase 7: Remove `tf.constant()` Wrappers in Flow Filters

After all models return `tf.Tensor` from `observation_noise_cov`, the `tf.constant()` wrappers become redundant and harmful (they freeze values, blocking gradient flow for HMC).

**Files:**
- `src/filters/particle/edh_flow.py:215` — `tf.constant(self.model.observation_noise_cov, ...)` -> `tf.cast(self.model.observation_noise_cov, self.dtype)`
- `src/filters/particle/edh_flow_global.py:216` — same
- `src/filters/particle/stochastic_edh.py:177` — same
- `src/filters/particle/sde_local_correction.py:38` — same

---

## Execution Order

```
Phase 0 (base class)
  |
  +-- Phases 1-5 (models, can be done in parallel)
  |
  v
Phase 6 (KernelMappingPF -- depends on TF models)
  |
  v
Phase 7 (tf.constant wrapper cleanup)
```

---

## Verification

After each phase, run:
```bash
cd code && python -m pytest tests/ -x -v
```

After Phase 4 specifically (Kitagawa cleanup):
```bash
python -m pytest tests/filters/particle/test_hmc_gradient.py -v
```

After Phase 6 (KernelMappingPF), run a kernel flow experiment:
```bash
cd code && python -m src.experiments.run_experiment +experiment=lorenz96/lorenz96_kernel_matrix
```

Final smoke test — run a representative experiment for each model:
```bash
python -m src.experiments.run_experiment +experiment=stochastic_volatility/stochastic_volatility_kernel_scalar
```

---

## Files NOT Changed

- `src/core/types.py` — `FilterResult` stays as `np.ndarray` (public API)
- `src/models/range_bearing.py` — already TF-native
- `src/models/two_sensor_bearing.py` — already TF-native
- `src/models/acoustic_tracking_full.py` — already TF-native
- `src/models/utils.py` (`generate_data`) — already uses TF seeds, no changes needed
- `src/experiments/run_experiment.py` — works as-is after models have `mu_0`/`Sigma_0`
- `src/utils/ode_solvers.py` — already TF-native
- All Kalman filters, all other particle/flow filters (already TF internally)
