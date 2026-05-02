# Phase 03: Kalman Reference Filter and Parameterized Log-Likelihood Gradient

## 1. Goal

Build the exact Linear Gaussian Kalman reference path in TensorFlow and require a learnable noise-scale gradient through `kalman_loglik_apply`, validating the `StateSpaceModel`, `ParameterSchema`, partition/merge, and differentiable likelihood path before stochastic filters are introduced.

## 2. What Gets Built

Files and modules owned or updated by this phase:

- `jpml_tf/filters/__init__.py` - imports concrete filter modules so their registry decorators run.
- `jpml_tf/core/registries.py` - provides `FILTER_REGISTRY` and `register_filter`.
- `jpml_tf/filters/common.py` - generic result containers and log-likelihood accumulation helpers from Phase 02; no predict/update vocabulary appears in common utilities.
- `jpml_tf/filters/kalman.py` - exact Kalman prediction, update, full filtering scan, and differentiable log-likelihood for models that implement Gaussian capability protocols.
- `jpml_tf/models/linear_gaussian.py` - updated, if not already done in Phase 02, so the gate model exposes at least one learnable Linear Gaussian noise scale through `ParameterSchema` without mutating model attributes.
- `tests/test_03_kalman_filter.py` - the single gate test file for this phase.

Shared result types:

```python
# jpml_tf/core/registries.py
FILTER_REGISTRY: dict[str, FilterBuilder]

def register_filter(name: str, *, schema: type[BaseModel]) -> Callable[[FilterBuilder], FilterBuilder]: ...

# jpml_tf/filters/common.py
class FilterRunResult(NamedTuple):
    estimates: PyTree
    loglik: tf.Tensor          # shape ()
    diagnostics: Mapping[str, tf.Tensor]

class LikelihoodResult(NamedTuple):
    loglik: tf.Tensor          # shape ()
    aux: Mapping[str, tf.Tensor]
```

Kalman registers as one concrete filter instance:

```python
@register_filter("kalman", schema=KalmanFilterParams)
def build_kalman_filter(params: KalmanFilterParams) -> FilterAlgorithm: ...
```

Kalman state and outputs:

```python
# jpml_tf/filters/kalman.py
class KalmanCarry(NamedTuple):
    mean: tf.Tensor            # shape (state_dim,)
    cov: tf.Tensor             # shape (state_dim, state_dim)
    loglik: tf.Tensor          # shape ()

class KalmanStepOutput(NamedTuple):
    pred_mean: tf.Tensor       # shape (state_dim,)
    pred_cov: tf.Tensor        # shape (state_dim, state_dim)
    filt_mean: tf.Tensor       # shape (state_dim,)
    filt_cov: tf.Tensor        # shape (state_dim, state_dim)
    innovation: tf.Tensor      # shape (obs_dim,)
    innovation_cov: tf.Tensor  # shape (obs_dim, obs_dim)
    loglik_increment: tf.Tensor # shape ()
```

Specific functions:

```python
def kalman_predict(
    model: StateSpaceModel,            # must implement HasGaussianTransitionNoise
    params: Params,
    mean: tf.Tensor,
    cov: tf.Tensor,
    t: int,
) -> tuple[tf.Tensor, tf.Tensor]: ...

def kalman_update(
    model: StateSpaceModel,            # must implement HasGaussianObservationNoise
    params: Params,
    pred_mean: tf.Tensor,
    pred_cov: tf.Tensor,
    y: tf.Tensor,
    t: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: ...

def kalman_filter_apply(
    model: StateSpaceModel,            # must implement HasClosedFormPrior + Gaussian noise capabilities
    params: Params,
    observations: tf.Tensor,           # shape (T, obs_dim)
) -> FilterRunResult: ...

def kalman_loglik_apply(
    model: StateSpaceModel,
    params: Params,
    observations: tf.Tensor,
) -> LikelihoodResult: ...
```

`kalman_filter_apply` is the deterministic exact filter path. It returns filtered means/covariances in `FilterRunResult.estimates`, the total Gaussian log-likelihood, and fixed-shape diagnostics. `kalman_loglik_apply` is the differentiable likelihood-family entry point; it may share the same internal scan but returns `LikelihoodResult` for future MAP and HMC paths in Phase 06.

Required learnable-parameter plumbing for the gate:

```python
schema = ParameterSchema(
    parameters={
        "log_obs_noise_scale": ParameterSpec(
            name="log_obs_noise_scale",
            shape=(),
            init=0.0,
            trainable=True,
            transform=TransformKind.positive,
            prior=PriorSpec(family="normal", params={"loc": 0.0, "scale": 1.0}),
        ),
        ...
    }
)

learnable, frozen = partition_params(schema, params)  # derived from ParameterSpec.trainable
constrained = constrain(schema, learnable_unconstrained)
params = merge_params(schema, constrained, frozen)
```

The gate must include at least one learnable Linear Gaussian noise scale that changes `transition_cov` or `observation_cov`. The preferred first target is an observation-noise scale because its effect on the Kalman innovation covariance and log-likelihood is direct. The model must consume the merged constrained `params` argument as a `tf.GradientTape`-traced input; it must not assign the learned value onto a model attribute.

Expected tensor shapes for the 2D/1D Linear Gaussian gate:

```text
observations:                  (T, 1), float64
initial_mean:                  (2,)
initial_cov:                   (2, 2)
filtered_means:                (T, 2)
filtered_covs:                 (T, 2, 2)
predicted_means:               (T, 2)
predicted_covs:                (T, 2, 2)
innovations:                   (T, 1)
innovation_covs:               (T, 1, 1)
loglik:                        ()
unconstrained_learnable_noise: ()
loglik_grad:                   ()
```

This phase does not introduce `FastFilterSpec` or `DiffLikelihoodSpec` Pydantic classes yet unless the implementation needs them as thin static wrappers. The gate validates the core deterministic filter and the learnable-parameter gradient path together. Full config-driven filter dispatch belongs to the Phase 08 experiment runner.

## 3. What Gets Tested and Acceptance Criteria

Gate test file: `tests/test_03_kalman_filter.py`

- `test_single_step_predict_update_matches_manual_formula(result_sink)` constructs the 2D/1D Linear Gaussian model from Phase 02 with at least one learnable noise scale, runs one prediction/update, and compares predicted mean, predicted covariance, innovation, innovation covariance, filtered mean, filtered covariance, and log-likelihood increment against a manual TF formula using the same constrained parameter value. Recommended test-code tolerances: absolute `1e-12` for means/covariances and `1e-10` for log-likelihood.
- `test_full_filter_shapes_dtypes_and_loglik(result_sink)` runs `kalman_filter_apply` on a fixed small observation sequence and asserts all output shapes and dtypes listed above, including scalar `float64` log-likelihood.
- `test_loglik_path_matches_filter_loglik(result_sink)` runs `kalman_loglik_apply` and `kalman_filter_apply` on the same model, merged params, and observations and asserts equal total log-likelihood within absolute tolerance `1e-12`.
- `test_kalman_filter_tffunction_matches_eager(result_sink)` wraps `kalman_filter_apply` in `@tf.function(reduce_retracing=True)` and asserts matching filtered means, filtered covariances, and log-likelihood within test-code tolerances. Re-running with the same `T` must not re-trace.
- `test_kalman_filter_jit_compile_matches_eager(result_sink)` wraps the same function in `@tf.function(jit_compile=True)` and asserts matching outputs at relative `1e-5`. Implementations must use `tf.linalg.cholesky` + `tf.linalg.cholesky_solve`, not `tf.linalg.slogdet` or `tf.linalg.inv`, both because XLA does not support `slogdet` (Wall 1 in `hmc_pipeline_issues.md` §E) and because explicit inverses degrade gradient stability.
- `test_kalman_loglik_grad_through_learnable_noise_is_finite_nonzero(result_sink)` builds an unconstrained learnable tensor for the noise scale, constrains it through `ParameterSchema`, merges it with frozen params, calls `kalman_loglik_apply` under `tf.GradientTape`, and gates `tape.gradient` of the scalar log-likelihood with respect to the unconstrained noise tensor. The gradient must be finite and non-zero by a test-code threshold such as `abs(grad) > 1e-8`.
- `test_partition_merge_constrain_path_used_by_loglik(result_sink)` perturbs only the learnable unconstrained noise value, rebuilds constrained/merged params through `partition_params`, `constrain`, and `merge_params`, and asserts the Kalman log-likelihood changes by more than `1e-8`.
- `test_grad_matches_central_finite_difference(result_sink)` compares the autodiff gradient against a centered finite-difference gradient at `h=1e-4`, with relative error below `1e-4`.

Gate-pass condition:

- `pytest tests/test_03_kalman_filter.py` passes.
- Every test saves a JSON result via the Phase 01 `result_sink` fixture.
- The exact single-step formulas match the implementation within tolerances.
- Eager, `tf.function`, and `tf.function(jit_compile=True)` outputs agree.
- The filter and differentiable likelihood entry points return the same total log-likelihood.
- The learnable-noise gradient through `kalman_loglik_apply` is finite and non-zero, and matches finite difference.
- The gate exercises `ParameterSchema` partition, constrain, and merge before calling the likelihood; direct model-attribute mutation is not an accepted implementation.

No minimal alternative is accepted for this phase. The gradient-through-filter case is the gate because this rebuild must validate abstractions against the hardest case they are expected to support, not only against static closed-form matrices.

## 4. What the Reader Needs to Understand

### Key Concepts

Kalman is the exact reference path for Linear Gaussian models. It sits in both worlds: the fast deterministic filter returns state estimates and log-likelihood, while the differentiable likelihood path returns a log-likelihood for gradient-facing samplers. This phase implements both views over one exact recurrence.

The filter is a pure scan over explicit state. This directly addresses the `diagnosis.md` `Control Flow and Mutation` problems where existing filters maintain mutable `tf.Variable` state, Python history lists, and object-local state. The Kalman carry contains only tensors: mean, covariance, and accumulated log-likelihood.

The model remains responsible for local dynamics and observation functions. The filter calls the `StateSpaceModel` capability methods from Phase 02. It does not mutate model parameters and does not assume Linear Gaussian parameters live as object attributes.

The learnable-noise gradient is part of the Kalman gate, not a sampler concern deferred to later. This phase proves that a learnable parameter can be partitioned, constrained, merged, consumed by the model, scanned through the filter, and differentiated end-to-end.

Closed-form ground truth is still used, but it is not enough. Before stochastic particle filters add Monte Carlo variance, the TF rebuild needs one deterministic filter whose outputs can be checked to tight tolerances under eager execution, `tf.function`, `jit_compile=True`, and `tf.GradientTape`.

### Invariants Established

- Kalman filtering is implemented as pure functions over `(model, params, observations)`.
- The time loop uses a fixed-shape recurrence compatible with `tf.scan` or `tf.while_loop` with `maximum_iterations` set.
- Filter state is explicit in `KalmanCarry`; no mutable filter object stores state. No `tf.Variable` is created inside the compiled function.
- `kalman_filter_apply` and `kalman_loglik_apply` agree on total log-likelihood.
- A learnable Linear Gaussian noise scale flows through `ParameterSchema` partition/constrain/merge into the model covariance path.
- `tf.GradientTape().gradient(kalman_loglik_apply, unconstrained_noise)` is finite and non-zero on the gate dataset.
- Eager, `tf.function`, and `tf.function(jit_compile=True)` Kalman outputs match within test-code tolerances.
- All Kalman tensors run under the global float64 policy established in Phase 01.
- Every gate test saves results through the Phase 01 helper.

### Tricky Bits and Rationale

Covariance updates must be numerically symmetric enough for tight gates. The implementation may use the Joseph form internally if needed, but the public contract is still exact Kalman predict/update. If Joseph form is chosen, tests should compare against the same mathematically equivalent update within tolerance, not require bitwise equality.

The log-likelihood increment must be computed with `tf.linalg.cholesky` and `tf.linalg.cholesky_solve`, not matrix inverse. This follows the numerical-discipline concerns in `diagnosis.md` and avoids unstable inverse-based formulas that degrade further under `tf.GradientTape`. Use twice the sum of log-diagonals of the Cholesky factor for `log|S|`; do not call `tf.linalg.slogdet` because XLA does not support it (Wall 1 in `hmc_pipeline_issues.md` §E).

The gate dataset must be chosen so the learnable-noise gradient is not accidentally near zero. A symmetric or perfectly matched observation sequence can make a mathematically valid gradient too small for a useful gate. The test data should be fixed, deterministic, and intentionally off-model enough to exercise the noise parameter.

The Kalman path must not write diagnostics to disk, print progress, or call `.numpy()` inside compiled code (recall C2 in `hmc_pipeline_issues.md`). Diagnostics are returned as tensors in the result NamedTuple.

`kalman_loglik_apply` is separate from `kalman_filter_apply` even if both call the same internal scan. This keeps the two algorithm families explicit.

A small Cholesky ridge (`eps * trace(S) / d * I`) is acceptable to keep updates numerically stable at near-singular S; document the eps choice as part of the filter's params schema. This same pattern repeats in EKF (used inside LEDH in Phase 05) and in the LEDH update step — establishing it here pays off twice.

### Alternatives Considered

Gating only static Linear Gaussian matrices is rejected. It would test the easiest case and could still allow the rebuild to fail when parameters are learnable, which is the failure mode diagnosed under `Parameter Handling`.

Starting with bootstrap PF is rejected because stochastic variance would make it harder to separate model-interface bugs from Monte Carlo behavior. Linear Gaussian Kalman gives the closed-form reference needed first; the exact reference must still include the learnable-parameter gradient path.

Implementing EKF/UKF in this phase is rejected. They require nonlinear Jacobian behavior and approximation-specific gates. Linear Gaussian Kalman gives the closed-form reference needed first; EKF/UKF can build on the same result types and Jacobian helpers later (the EKF Jacobian path is exercised inside LEDH in Phase 05).

Adding scenario-driven filter dispatch is rejected because this phase is pure-core filtering, not experiment glue. Config-to-filter construction belongs to Phase 08.

### Locked Design Decisions Realized

- Decision 1: the shared state-space core is used by a deterministic filter and a differentiable likelihood entry point without forcing one to hide the other.
- Decision 4: tests save JSON results through the Phase 01 helper.
- Decision 6: Jacobian helpers come from the model abstraction and default to autodiff via `tf.GradientTape` unless a model override exists.
- `design.md` section 3 `Parameter and ParameterSchema`: the gate validates partitioning, constraining, merging, and differentiating through a learnable parameter.
- `design.md` section 3 `Filter`: implements the Kalman family as both `FilterRunResult` and `LikelihoodResult` paths.
- `diagnosis.md` `Parameter Handling`: replaces mutable model attributes and duplicated sampler/filter parameter plumbing with explicit pytrees in a differentiable log-likelihood.
- `diagnosis.md` `Control Flow and Mutation`: replaces mutable filter state and Python history lists with explicit carries and fixed-shape outputs.
- `hmc_pipeline_issues.md` §E Wall 1: avoids `tf.linalg.slogdet` to keep the path XLA-compatible from Phase 03 forward.

### TF Function / JIT Boundary Decisions

`kalman_predict`, `kalman_update`, `kalman_filter_apply`, and `kalman_loglik_apply` must be compatible with `@tf.function(reduce_retracing=True, jit_compile=True)` when the model/static metadata is treated as Python-side static structure and tensors are traced values. The time recursion is expressed with `tf.scan` (or `tf.while_loop` with `maximum_iterations` bound to `T`), not a Python list accumulator.

`tf.GradientTape` must see the unconstrained learnable noise value as a traced tensor. Pydantic validation and schema construction remain outside any `tf.function`; `constrain`, `merge_params`, and `kalman_loglik_apply` are pure TF-compatible functions inside the differentiated path.

Config parsing, result saving, JSON writing, plotting, and timing remain outside `tf.function`. The gate only tests compiled/differentiated numerical functions and host-side JSON result persistence through the fixture.

## 5. Dependencies

This phase depends on:

- Phase 01 for package scaffold, scenario schema, and the JSON gate-test result sink.
- Phase 02 for `ParameterSchema`, `StateSpaceModel`, capability protocols, and `LinearGaussianModel`.

Later dependencies:

- Bootstrap and LEDH particle-filter gates in Phase 05 will compare stochastic estimates against this exact Kalman reference for Linear Gaussian cases where applicable.
- Differentiable particle likelihood gates will reuse the parameter partition/constrain/merge gradient pattern established here, then apply it to the harder LEDH Jacobian-accumulation path.
- HMC and MAP samplers in Phase 06 will use `kalman_loglik_apply` as the first deterministic differentiable objective. Their gates also include the SV2D-derived hard likelihood from Phase 05.
- EKF (used per-particle inside LEDH, Phase 05) reuses the Cholesky discipline and Jacobian-via-`tf.GradientTape` conventions introduced here.
- From Phase 05 onward, abstraction gates must include the hardest case the abstraction is expected to support. For this project that means LEDH particle flow with 29-step Jacobian accumulation in float64 for particle-filter and differentiable-likelihood gates, and the Linear Gaussian finite-N particle-likelihood sampler gate defined in Phase 06.
