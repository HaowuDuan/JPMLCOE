# Experiment Workflow Analysis & Cleanup Report

## 1. Current Data Flow Overview

```
main(cfg)
  |
  +-> run_filter_experiment(cfg)
  |     |
  |     +-> Determine observe_initial flag (lines 59-63)
  |     +-> generate_data(model, T, rng, observe_initial)
  |     +-> Instantiate filter (lines 68-129, complex branching)
  |     +-> filter_obj.filter(observations)  ->  FilterResult
  |     +-> Compute RMSE, performance metrics
  |     +-> Return results dict
  |
  +-> save_results(results, output_dir)
  +-> plot (visualization.py)
```

---

## 2. The `observe_initial` / `predict_then_update_filters` Problem

### 2.1 What the Code Does (lines 59-63, run_experiment.py)

```python
predict_then_update_filters = {
    'KernelMappingPF',
    'ExactDaumHuangFlow', 'LocalExactDaumHuangFlow',
}
observe_initial = filter_target not in predict_then_update_filters
```

This flag controls **data generation**, not the filter itself. It produces two completely different ground-truth sequences for different filter types:

### 2.2 Two Different Data Generation Paths (models/utils.py)

**Path A: `observe_initial=True`** (EKF, UKF, Kalman, Bootstrap PF, EDH Invertible, Stochastic EDH)
```
states[0] = sample_initial_state()        # x_0
observations[0] = observe(states[0])      # y_0 = h(x_0) + noise

for t in 1..T-1:
    states[t] = transition(states[t-1])   # x_t
    observations[t] = observe(states[t])  # y_t
```
- **Result**: `states` = [x_0, x_1, ..., x_{T-1}], `observations` = [y_0, y_1, ..., y_{T-1}]
- states[0] IS the initial condition

**Path B: `observe_initial=False`** (EDH Flow, LEDH Flow, KernelMappingPF)
```
current = sample_initial_state()          # x_0 (DISCARDED, not stored)

for t in 0..T-1:
    current = transition(current)          # x_{t+1}
    states[t] = current
    observations[t] = observe(current)
```
- **Result**: `states` = [x_1, x_2, ..., x_T], `observations` = [y_1, y_2, ..., y_T]
- **x_0 is lost** -- it was sampled and immediately thrown away

### 2.3 Why This Is Bad

1. **Different ground truths for different filters**: You cannot fairly compare RMSE between EDH Flow and EKF because they are evaluated on different state sequences. The EKF's RMSE includes the initial state, the flow filter's does not.

2. **The terminology is misleading**: `observe_initial` sounds like it controls whether we observe the initial state, but it actually controls whether the initial state **exists in the output at all**. `predict_then_update_filters` is also confusing -- ALL filters in this codebase do predict-then-update in their `filter()` loop.

3. **Leaky abstraction**: Data generation should not know about filter implementation details. The same ground truth should be used regardless of which filter is run.

### 2.4 What ACTUALLY Differs Between Filters

Every filter in this codebase follows the same loop structure:

```python
# ALL filters (flow_base.py:144-146, kalman.py:193-195, bootstrap_pf.py:124-129, edh_invertible.py:412-417):
for t in range(T):
    self.predict()
    self.update(observations[t])
    mean, cov = self._estimate_mean_cov()
    means.append(mean)
```

The difference is NOT predict-then-update vs update-then-predict. The actual difference is:
- **Flow filters** (EDH Flow, LEDH Flow): Initialize particles from prior, then at each step predict particles through dynamics, then flow them toward the observation. They don't have an initial "update at t=0" because their first step is always predict.
- **EKF/UKF/Kalman**: Also do predict-then-update. The Kalman does `predict() -> update()` at t=0, meaning the initial mean/cov goes through a prediction step before the first observation is incorporated.
- **Bootstrap PF**: Same predict-then-update loop.

The real semantic difference is not in the filter loop but in whether the filter produces an estimate **at the initial condition** (before any transition). Currently, **none of the filters produce an estimate at t=0 before the first predict step**. They all start with predict.

### 2.5 Better Terminology Suggestion

Instead of `observe_initial` / `predict_then_update_filters`, the concept should be:

- **`include_initial_state`**: Whether to include x_0 in the returned states array
- Or better yet: **always include x_0** in ground truth, and handle the alignment in the filter/visualization layer

---

## 3. State Vector Indexing: The Core Confusion

### 3.1 Current Behavior

| Scenario | `states` array | `means` (filter output) | Alignment |
|----------|---------------|------------------------|-----------|
| `observe_initial=True` | [x_0, x_1, ..., x_{T-1}] | [m_0, m_1, ..., m_{T-1}] where m_t = result after predict(t) + update(y_t) | states[t] vs means[t] -- but m_0 is NOT the initial condition; it's the result of predicting FROM x_0 then updating with y_0 |
| `observe_initial=False` | [x_1, x_2, ..., x_T] | [m_1, m_2, ..., m_T] | states[t] vs means[t] -- x_0 is lost entirely |

### 3.2 What This Means for the Kalman Filter (observe_initial=True)

```
Initial: mean = mean_0, cov = Sigma_0      (these are the initial conditions)

t=0: predict() -> mean = F @ mean_0        (predicted to t=1 conceptually)
     update(y_0) -> mean = ...             (but updating with y_0 which is at t=0!)
     means[0] = updated mean
```

**This is problematic.** The Kalman filter predicts from its initial state (which represents t=0) to get a prediction for t=1, then updates with y_0 which is an observation of x_0. There's a **time mismatch**: the predicted state is at t=1 but the observation is at t=0.

Actually, looking more carefully: since `states[0] = x_0` and `observations[0] = h(x_0) + noise`, and the Kalman starts with `mean_0` as its prior for x_0, then:
- `predict()` at t=0: propagates mean_0 to get a prior for x_1
- `update(y_0)` at t=0: updates this x_1 prior with y_0 which is from x_0

**This IS a mismatch.** The filter should either:
- Update first with y_0 (at x_0), then predict to x_1, or
- Start by predicting to x_1, then update with y_1

### 3.3 What This Means for Flow Filters (observe_initial=False)

```
Initial: particles ~ p(x_0)

t=0: predict() -> particles propagated to represent x_1
     update(y_0) -> y_0 = h(x_1) + noise (since observe_initial=False, observations are shifted)
     means[0] = posterior estimate of x_1
```

This is **consistent** because `observations[0]` corresponds to `states[0] = x_1`, and the filter predicts to x_1 before updating with the observation of x_1.

### 3.4 The Root Issue

The `observe_initial` flag was invented to paper over a fundamental design decision: **should the filter loop be predict-then-update or update-then-predict?**

For `observe_initial=True` filters (EKF, PF, etc.), the current code has a subtle bug or at least a conceptual inconsistency:
- At t=0, the filter predicts from x_0 to x_1, then updates with y_0 (observation of x_0)
- This means means[0] is a corrupted estimate -- it used a prediction step that moved the state to x_1 but then updated with data from x_0

**However**, this may still "work" in practice because the filter is initialized near x_0 and the first predict step with `states[0] = x_0` and `observations[0] = h(x_0)` can still produce reasonable estimates. But it's not mathematically clean.

### 3.5 Recommended Clean Design

**Option: Always generate `states = [x_0, x_1, ..., x_T]` and `observations = [y_1, y_2, ..., y_T]`**

This is the standard filtering convention:
- x_0 ~ p(x_0): initial state
- x_t = f(x_{t-1}) + noise, for t = 1, ..., T
- y_t = h(x_t) + noise, for t = 1, ..., T

Then every filter runs:
```python
# Initialize with prior for x_0
for t in range(T):  # t = 0, 1, ..., T-1 indexing observations
    predict()       # propagate prior from x_t to x_{t+1}
    update(y[t])    # update with y_{t+1}
    store means[t]  # posterior estimate of x_{t+1}
```

The filter output `means` has shape (T, state_dim) and represents estimates of x_1, ..., x_T.
The ground truth for comparison is `states[1:]` (x_1, ..., x_T).

For plotting, prepend the initial condition:
```python
# In visualization:
full_states = np.vstack([x_0[np.newaxis, :], states[1:]])    # or just states which is [x_0, ..., x_T]
full_means  = np.vstack([initial_mean[np.newaxis, :], means]) # prepend filter's initial condition
full_covs   = np.vstack([initial_cov[np.newaxis, :, :], covs])
```

This way:
- All filters use the **same data**
- RMSE is computed on the **same state sequence**
- The plot shows x_0 through x_T with the filter's initial condition visible

---

## 4. Visualization: Initial State Prepending

### 4.1 Current Status: NO prepending happens

In `visualization.py`, both `plot_filter_results()` (line 42) and `plot_high_dim_results()` (line 261):

```python
time = np.arange(T)
ax.plot(time, states[:, i], ...)  # plots states as-is
ax.plot(time, means[:, i], ...)   # plots means as-is
```

**The initial conditions (x_0, mean_0, Sigma_0) are never added to the front of the arrays before plotting.**

### 4.2 What This Means

- For `observe_initial=True`: The plot starts at the first predict-then-update result, not at the true initial condition. You never see how far the filter's initial guess was from the truth.
- For `observe_initial=False`: The plot starts at x_1 (after the first transition). x_0 is completely absent.

### 4.3 Recommendation

After filtering, prepend the initial conditions before passing to visualization:

```python
# After filtering:
initial_state = states[0] if observe_initial else x_0_saved  # need to save x_0
initial_mean  = filter_initial_mean  # mean_0 used to initialize the filter
initial_cov   = filter_initial_cov   # Sigma_0 used to initialize the filter

# Prepend for visualization
full_states = np.vstack([initial_state[np.newaxis, :], states])  # if states starts at x_1
full_means  = np.vstack([initial_mean[np.newaxis, :], means])
full_covs   = np.concatenate([initial_cov[np.newaxis, :, :], covs], axis=0)

# Pass to visualization
plot_filter_results(full_states, observations, full_means, full_covs, ...)
```

This way the plot shows t=0 to t=T, the viewer can see the initial uncertainty, and the initial convergence behavior of the filter is visible.

---

## 5. Differences Between `code/` (TensorFlow) and `code_backup_np/` (NumPy)

### 5.1 Files That Are Identical
- `run_experiment.py` -- **identical** in both versions
- `visualization.py` -- **identical** in both versions
- `models/utils.py` (generate_data) -- **identical**
- `core/types.py` (FilterResult) -- **identical**
- `particle_base.py` -- **identical**

### 5.2 Key Differences in Filter Implementations

| Component | `code/` (TF) | `code_backup_np/` (NumPy) |
|-----------|-------------|-------------------------|
| `flow_base.py` predict() | Uses `tf.constant`, `_make_tf_seed()`, TF model API | Uses `np.random.default_rng()`, pure NumPy model API |
| `flow_base.py` initialize() | Uses `_make_tf_seed()`, stores as np.array | Uses `np.random.default_rng()`, threaded particle sampling |
| `flow_base.py` _estimate_mean_cov() | `self.particles.numpy()` conversion needed | Direct `self.particles` access |
| `edh_flow.py` | TF Variables for particles, `tf.function` decorated, `state_transition_batch()` | N/A (identical flow_base handles it) |
| `edh_invertible.py` | Has `_create_filter()` creating EKF with TF | Same structure but EKF is NumPy-based |
| `extended_kalman.py` | `tf.Variable`, `tf.function`, `tf.linalg.*` | Pure NumPy (in backup) |

### 5.3 The TF -> NumPy API Mismatch Problem

The current `code/` version's `flow_base.py` calls:
```python
seed = self._make_tf_seed()
x = tf.constant(self.particles[i], dtype=tf.float32)
new_particles.append(np.asarray(self.model.sample_state_transition(x, seed)))
```

This means `flow_base.py` in `code/` expects the model's `sample_state_transition()` to accept TensorFlow tensors and TF seeds. But if the models have been updated to use NumPy APIs, this will break.

The backup's `flow_base.py` uses:
```python
self.model.sample_state_transition(self.particles[i], self.random_state)
```
Which is the clean NumPy API.

### 5.4 What Broke

The `flow_base.py` in `code/` was partially converted to TF but still has the fundamental structure from the NumPy version. The `predict()` method:
- Creates TF seeds via `_make_tf_seed()`
- Wraps particles in `tf.constant`
- Calls model methods that may or may not accept TF inputs

Meanwhile, `edh_flow.py` in `code/` overrides `predict()` with a proper TF implementation using `state_transition_batch()`, so the base class `predict()` is never actually called for EDH Flow. But other filters inheriting from `FlowFilterBase` would hit the broken base class predict().

---

## 6. Suggestions to Clean Up `run_experiment.py`

### Problem 1: The 60-line filter instantiation block (lines 68-129)

Currently there are 4 branches for filter construction:
1. EKF/UKF with tracking models (state_dim % 4 == 0) -- special perturbed initialization
2. EKF/UKF with non-tracking models
3. Plain KalmanFilter (no model needed)
4. Everything else (pass model)

**Suggestion**: Move initialization logic into the filter classes themselves or into a factory function.

```python
# Clean version:
def create_filter(cfg, model, rng):
    """Factory function that handles all filter-specific initialization."""
    filter_name = cfg.filter._target_.split('.')[-1]

    if filter_name in ('ExtendedKalmanFilter', 'UnscentedKalmanFilter'):
        initial_mean, initial_cov = sample_perturbed_initial(model, rng)
        return hydra.utils.instantiate(cfg.filter, model=model,
                                        mean_0=initial_mean, Sigma_0=initial_cov)
    elif filter_name == 'KalmanFilter':
        return hydra.utils.instantiate(cfg.filter)
    else:
        return hydra.utils.instantiate(cfg.filter, model=model)
```

### Problem 2: The `observe_initial` hack (lines 56-66)

**Suggestion**: Remove entirely. Always generate data the same way:
```python
states, observations, initial_state = generate_data(model, T=T, rng=rng)
# states = [x_1, ..., x_T], observations = [y_1, ..., y_T], initial_state = x_0
```

### Problem 3: Filter dispatch for `filter()` call (lines 140-149)

```python
if filter_name in ['ExactDaumHuangFlow', 'LocalExactDaumHuangFlow']:
    result = filter_obj.filter(observations, random_state=rng)
else:
    result = filter_obj.filter(observations)
```

**Suggestion**: All filters should accept the same signature. Either all accept `random_state` (and ignore it if not needed), or handle it internally during construction.

### Problem 4: Diagnostics extraction boilerplate (lines 156-167)

```python
pf_diagnostics = {}
if hasattr(result, 'log_likelihoods') and result.log_likelihoods is not None:
    pf_diagnostics['log_likelihoods'] = result.log_likelihoods
# ... 5 more nearly identical blocks
```

**Suggestion**: `FilterResult` is a dataclass. Just iterate over its fields:
```python
diagnostic_fields = ['log_likelihoods', 'ess', 'weights_history', 'resampled_at', 'n_unique']
pf_diagnostics = {f: getattr(result, f) for f in diagnostic_fields if getattr(result, f, None) is not None}
```

### Problem 5: Hardcoded model-specific logic (lines 76-106)

The perturbed initial state logic is hardcoded for tracking models (state_dim % 4 == 0, std=10 for positions, std=1 for velocities, bounds [0,40]x[0,40]).

**Suggestion**: Move this into the model class:
```python
class AcousticTrackingModel:
    def sample_perturbed_initial(self, true_state, rng):
        """Model-specific perturbed initialization for EKF/UKF."""
        ...
```

### Problem 6: The `_is_valid_initial_state()` function (lines 16-33)

This is acoustic-tracking-specific validation hardcoded as a top-level utility function.

**Suggestion**: Move into the model as `model.is_valid_state(state)`.

### Problem 7: Performance tracking mixed with filtering logic (lines 134-190)

60 lines of tracemalloc/psutil code interleaved with the core algorithm.

**Suggestion**: Use a context manager:
```python
with PerformanceTracker() as perf:
    result = filter_obj.filter(observations)
metadata['performance'] = perf.get_metrics(T)
```

### Problem 8: Plotting dispatch in main() (lines 298-332)

35 lines of model-specific plotting logic (state_dim == 16, obs_spacing, plot_indices).

**Suggestion**: Let the model define its plot configuration:
```python
plot_config = model.get_plot_config()  # returns dict with plot_indices, obs_spacing, etc.
plot_results(states, observations, means, covs, save_path, **plot_config)
```

---

## 7. Proposed Clean `run_experiment.py` Structure

```python
def run_filter_experiment(cfg):
    rng = np.random.default_rng(cfg.seed)
    model = hydra.utils.instantiate(cfg.model)

    # 1. Generate data (always the same, no observe_initial flag)
    T = cfg.get('T', 100)
    initial_state, states, observations = generate_data(model, T=T, rng=rng)
    # initial_state = x_0, states = [x_1, ..., x_T], observations = [y_1, ..., y_T]

    # 2. Create filter (factory handles all special cases)
    filter_obj = create_filter(cfg, model, initial_state, rng)

    # 3. Run filter with performance tracking
    with PerformanceTracker() as perf:
        result = filter_obj.filter(observations)

    # 4. Compute metrics
    rmse = np.sqrt(np.mean((result.means - states)**2))

    # 5. Return clean results
    return ExperimentResult(
        initial_state=initial_state,
        states=states,
        observations=observations,
        filter_result=result,
        rmse=rmse,
        performance=perf.get_metrics(T),
        config=cfg
    )
```

---

## 8. Summary of Key Findings

| Issue | Current State | Impact | Fix |
|-------|--------------|--------|-----|
| `observe_initial` flag | Generates different ground truths per filter | Unfair RMSE comparison, confusing | Remove; always generate same data |
| Initial state in plots | Never prepended | Missing t=0 in visualizations | Prepend x_0 and mean_0 before plotting |
| Filter instantiation | 60-line if/elif chain | Hard to maintain, add new filters | Factory function or model-defined init |
| TF/NumPy mismatch in flow_base.py | Base class uses TF seeds, subclass overrides | Fragile, breaks if base predict() called | Decide on one API (recommend NumPy) |
| Performance tracking | Inline tracemalloc/psutil code | Clutters core logic | Context manager |
| Model-specific hardcoding | Tracking bounds, state_dim%4 | Breaks for new models | Move into model classes |
| Diagnostics extraction | 5 repetitive hasattr checks | Boilerplate | Iterate dataclass fields |
| Plotting dispatch | 35-line if/elif in main() | Grows with each model | Model-defined plot config |
