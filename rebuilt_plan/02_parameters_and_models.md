# Phase 02 — Parameters & Models

**Objective**: explicit parameter trees (not mutable model attributes), capability protocols instead of a monolithic base class, and Linear Gaussian model as the first concrete implementation with a learnable noise parameter.

**Fixes from current code**: `DifferentiableModel.update_parameters(setattr)` is the primary anti-pattern — it mutates model state inside gradient tape. Replacement: pure functional parameter merge.

## Deliverable files

```
jpml_tf/
  parameters/
    __init__.py
    schema.py                    # ParameterTree, explicit trainable flags, trainable_names(), frozen_names()
    transforms.py                # softplus/sigmoid/identity bijectors as pure fns
    merge.py                     # partition_params, constrain, merge_params (all pure)
    priors.py                    # PriorSpec -> tfp.distributions
  models/
    __init__.py
    base.py                      # StateSpaceModel protocol + capability protocols
    registry.py                  # @register_model decorator
    linear_gaussian.py           # LG model; opt-in HasGaussianTransitionNoise, HasGaussianObservationNoise
  tests/
    test_02_parameters.py
    test_02_models.py
```

## Parameter tree design

A parameter tree is a nested dict mapping param names to scalar/tensor values:

```python
Params = dict[str, tf.Tensor | "Params"]  # recursive

# Example for LG:
lg_params_unconstrained = {
    "obs_noise_std_raw": tf.constant(0.5),   # pre-softplus
}

lg_params_constrained = {
    "obs_noise_std": tf.constant(1.0),        # post-softplus
}

lg_fixed = {
    "F": tf.constant([[0.9]]),
    "B": tf.constant([[1.0]]),
    "H": tf.constant([[1.0]]),
}
```

`ParameterSchema` is the source of truth for which leaves are trainable:

```python
class ParameterSchema(BaseModel):
    parameters: list[ParameterSpec]

    def trainable_names(self) -> list[str]: ...
    def frozen_names(self) -> list[str]: ...
```

The runtime `fixed` tree returned by `partition_params` is derived from `ParameterSpec.trainable == False`, not from a separate config bucket.

## Pure functions (no model state mutation)

```python
def partition_params(
    spec: ParameterSchema, params: Params
) -> tuple[Params, Params]:
    """Return (trainable_unconstrained, fixed), using ParameterSpec.trainable."""

def constrain(spec: ParameterSchema, unconstrained: Params) -> Params:
    """Apply bijector per param; returns constrained tree."""

def merge_params(constrained: Params, fixed: Params) -> Params:
    """Return a single tree for the filter."""

def log_prior(spec: ParameterSchema, constrained: Params) -> tf.Tensor:
    """Sum of log-priors + log-jacobians."""
```

**No method on the model mutates anything.** The model is a plain `tf.Module` or NamedTuple whose `call()` methods take `params` as an argument.

## Model protocol (minimal, capability-based)

```python
class StateSpaceModel(Protocol):
    state_dim: int
    obs_dim: int
    dtype: tf.DType

    def sample_initial_state(
        self, params: Params, seed: tf.Tensor
    ) -> tf.Tensor: ...

    def sample_transition(
        self, state: tf.Tensor, params: Params, seed: tf.Tensor
    ) -> tf.Tensor: ...

    def log_observation_prob(
        self, obs: tf.Tensor, state: tf.Tensor, params: Params
    ) -> tf.Tensor: ...
```

**Opt-in capability protocols** (filters check `isinstance` before using):

```python
class HasGaussianObservationNoise(Protocol):
    def observation_function(self, state, params) -> tf.Tensor: ...
    def observation_noise_cov(self, state, params) -> tf.Tensor: ...  # may depend on state!
    def observation_jacobian(self, state, params) -> tf.Tensor: ...

class HasAnalyticInitialMoments(Protocol):
    def initial_mean(self, params) -> tf.Tensor: ...
    def initial_cov(self, params) -> tf.Tensor: ...

class HasStateDependentObsCov(Protocol):
    """Required by LEDH+SV2D. Flags that obs_cov varies per particle."""
    def observation_noise_cov_batch(
        self, states: tf.Tensor, params: Params
    ) -> tf.Tensor: ...  # shape (N, obs_dim, obs_dim)
```

## Linear Gaussian model (first concrete)

`models/linear_gaussian.py`:

```python
@register_model(family="linear_gaussian", algorithm="default")
class LinearGaussianModel(StateSpaceModel):
    # Implements: sample_initial_state, sample_transition, log_observation_prob
    # Also implements: HasGaussianObservationNoise, HasGaussianTransitionNoise,
    #                  HasAnalyticInitialMoments
    # Does NOT implement: HasStateDependentObsCov
    pass
```

**Learnable parameter**: `obs_noise_std` (positive). Softplus bijector. LogNormal prior with mode at truth. Linear Gaussian matrices such as `F`, `B`, and `H` may remain frozen in the runtime `fixed` tree, but that frozen status is derived from explicit `trainable=False` schema entries or from `model_constants`, not from container membership.

## Gate tests

`tests/test_02_parameters.py`:
1. `test_partition_merge_roundtrip`: `merge(constrain(partition(spec, full)[0]), partition(spec, full)[1]) == full` for LG spec.
2. `test_log_prior_finite_at_init`: LogNormal prior at `initial_value` gives finite log-prob.
3. `test_transform_gradients_finite`: `tf.GradientTape` on `constrain(unconstrained)` gives non-NaN, non-zero gradient.
4. `test_partition_uses_explicit_trainable_flag`: toggling one parameter's `trainable` field in `ParameterSchema` changes the output of `partition_params(spec, params)` while leaving `merge_params(constrain(trainable), fixed)` numerically equal to the original merged parameter tree.

`tests/test_02_models.py`:
5. `test_lg_sample_reproducible`: two calls with same seed give identical state/observation sequences.
6. `test_lg_log_obs_gradient_wrt_param`: `grad log_observation_prob wrt obs_noise_std` is finite, non-zero.
7. `test_capability_protocol_dispatch`: `isinstance(lg_model, HasStateDependentObsCov) == False`; `HasGaussianObservationNoise == True`.
8. `test_registry_lookup`: `get_model(family="linear_gaussian", algorithm="default")` returns the class; `algorithm="bogus"` raises.

## Pass criteria

All 8 tests pass. Gradient of `log p(y | x; θ)` w.r.t. learnable `obs_noise_std` is numerically verified against finite-difference within `1e-5` relative error.

## Risks

- `tf.Module` auto-tracking of `tf.Variable` can silently re-introduce mutable state. Mitigation: models hold ONLY `tf.constant` for structural model constants; parameter values flow in through function args and the explicit `trainable` flag controls partitioning.
- Capability protocol dispatch adds runtime isinstance checks in hot loops. Mitigation: filters resolve capabilities at build time, not per-step.

## Estimated effort

2–3 days. Parameter plumbing is conceptually simple but easy to get wrong. Protocol design should be reviewed before implementation.

## Prerequisite

Phase 01 complete (startup, scenario schema, result sink).
