# Target JAX Architecture

This document uses `codebase_map.md` and `diagnosis.md` as context. The user decisions in the Pass 3 prompt are constraints: two algorithm families are first-class, conditional resampling is the default for non-gradient runs, global `float64` is the default, every test saves JSON results through a uniform helper, configs are whole-run scenario files with no Hydra defaults-list composition, and Jacobians default to `jax.jacfwd` / `jax.jacrev` with optional analytic overrides.

## 1. Goals and Non-Goals

### Goals

1. Arbitrary state-space models. A model should define transition, observation, prior, log-density, and optional analytic hooks without being tied to one filter, sampler, or parameter-inference path.
2. Arbitrary parameters. Any model parameter, noise parameter, flow parameter, or sampler-exposed scalar should be representable in a typed parameter schema, not hardcoded in a runner.
3. Per-parameter learnable flag. The architecture must explicitly partition learnable and frozen parameters. This replaces the current implicit YAML/model coupling diagnosed under "Parameter Handling / Trainable versus frozen parameters are inferred from YAML coupling."
4. Full diagnostic control. Users choose what metrics are collected, at what frequency, and where they are written. Diagnostics are not embedded in hot paths, addressing "Effectful Code Inside or Near Hot Paths / Logging, callbacks, timing, and `.numpy()` are interleaved with computation."
5. Full gradient-clipping control. Gradient clipping is configured outside training loops, supports global and per-parameter policies, and applies uniformly to MAP and any custom gradient-based sampler path.

### Non-Goals

- No TensorFlow compatibility layer. The rebuild should not preserve `tf.Variable`, TensorFlow Probability objects, or TensorFlow-specific XLA workarounds.
- No Hydra defaults-list composition. Configs are self-contained scenario files; there are no compositional config groups.
- No per-experiment dtype toggling in compiled code. `jax_enable_x64` is enabled once at startup and `float64` is the critical-path default.
- No single forced filter interface that hides algorithm-family differences. Non-differentiable fast PF and differentiable likelihood filters share a state-space core but have distinct interfaces.
- No hardcoded Jacobian requirement. Analytic Jacobians are optional overrides; autodiff is the default.
- No object mutation as a parameter-passing mechanism. The diagnosis found mutable model attributes and compiled closures around model objects; the JAX design uses explicit pytrees.
- No file I/O, progress bars, JSON writes, plotting, or wall-clock timing inside JIT-compiled kernels.
- No silent config fallbacks for algorithm names, resampling policies, priors, or constraints.

## 2. Module Layout

Proposed tree:

```text
jpmljax/
  __init__.py
  startup.py
  config/
    schema.py
    load.py
    scenarios/
  core/
    types.py
    model.py
    params.py
    random.py
    registries.py
  models/
    linear_gaussian.py
    range_bearing.py
    stochastic_volatility.py
    stochastic_volatility_2d.py
    kitagawa.py
  filters/
    common.py
    analytic_kalman.py
    bootstrap_particle.py
    multistep_flow.py
  resamplers/
    policies.py
    systematic.py
    stratified.py
    multinomial.py
    soft.py
    ot.py
  samplers/
    base.py
    blackjax_hmc.py
    optax_map.py
    random_walk_mh.py
  mcp/
    server.py
    tools.py
    schemas.py
  optim/
    clipping.py
    schedules.py
    optax_builders.py
  diagnostics/
    metrics.py
    events.py
    sinks.py
    test_results.py
  experiments/
    run_filter.py
    run_inference.py
    materialize.py
  tests/
    conftest.py
    helpers/
```

- `startup.py` - user-facing startup hook; enables `jax_enable_x64` before importing compiled kernels.
- `config/` - user-facing scenario config parsing and validation; no Hydra composition.
- `core/` - pure core abstractions: model protocol, capability protocols, registries, parameter schema, common pytrees, PRNG conventions.
- `models/` - pure-core model definitions; each model exposes parameter schema and pure functions.
- `filters/` - concrete registered filter instances; shared interfaces live in `core/`.
- `resamplers/` - concrete registered resampler instances plus policy specs.
- `samplers/` - concrete registered sampler instances plus user-facing orchestration wrappers.
- `mcp/` - user-facing MCP server shell for Claude Desktop control; delegates validation, execution, and result lookup to `config/`, `experiments/`, and `diagnostics/`.
- `optim/` - pure-core gradient clipping and Optax builder utilities.
- `diagnostics/` - host-side diagnostic specs and sinks plus pure metric calculations.
- `experiments/` - experiment glue: load config, build pytrees, call compiled kernels, write outputs.
- `tests/` - test suite using a uniform JSON result sink.

## 3. Core Abstractions

### StateSpaceModel

Purpose: represent arbitrary state-space models without mutable model objects. This addresses `diagnosis.md` findings "Model and Data Boundaries / Models combine simulation, deterministic filtering methods, and log-probability methods" and "Default batch methods are Python-loop based."

Interface sketch:

```python
State = Array[shape=(state_dim,)]
Obs = Array[shape=(obs_dim,)]
Params = PyTree[str, Array]

class StateSpaceModel(eqx.Module):
    name: str
    state_dim: int
    obs_dim: int
    schema: ParameterSchema
    noise_spec: NoiseSpec
    static: ModelStatic

    def sample_initial(self, params: Params, key: Key) -> State: ...
    def transition_fn(self, x: State, u: PyTree, theta: Params, key: Key) -> State: ...
    def observation_fn(self, x: State, theta: Params, key: Key) -> Obs: ...
    def transition_logprob(self, x_next: State, x: State, u: PyTree, theta: Params) -> Array[()]: ...
    def observation_logprob(self, y: Obs, x: State, theta: Params) -> Array[()]: ...

class NoiseSpec(BaseModel):
    structure: Literal["additive", "multiplicative", "arbitrary"]
    params: dict[str, Any] = {}

class HasGaussianTransitionNoise(Protocol):
    def transition_mean(self, params: Params, x: State, u: PyTree, t: int) -> State: ...
    def transition_cov(self, params: Params, x: State, u: PyTree, t: int) -> Array: ...

class HasGaussianObservationNoise(Protocol):
    def observation_mean(self, params: Params, x: State, t: int) -> Obs: ...
    def observation_cov(self, params: Params, x: State, t: int) -> Array: ...

class HasClosedFormPrior(Protocol):
    def initial_mean(self, params: Params) -> State: ...
    def initial_cov(self, params: Params) -> Array: ...

class HasAnalyticJacobian(Protocol):
    def transition_jacobian_override(self, params: Params, x: State, t: int) -> Array | None: ...
    def observation_jacobian_override(self, params: Params, x: State, t: int) -> Array | None: ...
```

Batching is not a method contract. Particle batching is done by `jax.vmap(model.transition_fn, in_axes=(0, None, None, 0))` or equivalent wrappers. `noise_spec` is a structural declaration for how noise enters the state-space model; it is not a Gaussian guarantee. A model can declare additive noise and still not implement Gaussian moment capabilities. Gaussian moment methods, closed-form priors, and analytic Jacobians are opt-in capabilities; filters and samplers check capability protocols before using them. Jacobian defaults use `jax.jacfwd` / `jax.jacrev`, with analytic overrides only where a model explicitly provides `HasAnalyticJacobian`.

Design goals satisfied: arbitrary SSMs, arbitrary parameters, optional analytic Jacobians. JIT boundary: model functions are called inside filter `scan`s and are traced as part of the filter kernel; static metadata is held as static Equinox leaves.

### Parameter and ParameterSchema

Purpose: replace mutable model-attribute parameter handling from `diagnosis.md` sections "Learnable parameters are mutable model attributes" and "Samplers and filters duplicate parameter plumbing."

Interface sketch:

```python
class TransformKind(Enum):
    identity
    positive
    unit_interval
    bounded

class PriorSpec(BaseModel):
    family: Literal["normal", "lognormal", "uniform", "student_t", ...]
    params: dict[str, float]

class ParameterSpec(BaseModel):
    name: str
    shape: tuple[int, ...] = ()
    init: Any
    true: Any | None = None
    learnable: bool
    transform: TransformKind
    bounds: tuple[float, float] | None = None
    prior: PriorSpec | None = None

class ParameterSchema(BaseModel):
    parameters: dict[str, ParameterSpec]
```

First milestone scope is exactly the prior families and parameter transforms currently implemented in the TensorFlow code, as cataloged under `diagnosis.md` "Parameter Handling." New priors and transforms are extension points after parity, not first-milestone requirements.

Pure functions:

```python
partition_params(schema, params) -> (learnable_tree, frozen_tree)
merge_params(schema, learnable_tree, frozen_tree) -> params
constrain(schema, unconstrained_tree) -> constrained_tree
unconstrain(schema, constrained_tree) -> unconstrained_tree
log_prior(schema, constrained_tree) -> scalar
log_abs_det_jacobian(schema, unconstrained_tree) -> scalar
```

Design goals satisfied: arbitrary parameters and per-parameter learnable flags. JIT boundary: `constrain`, `merge_params`, `log_prior`, and Jacobian correction are pure JAX functions inside sampler kernels. Pydantic validation happens outside JIT.

### Filter

Purpose: share the state-space model core while keeping non-differentiable fast filters and differentiable likelihood filters as separate first-class families. This explicitly follows user decision 1 and fixes the diagnosis finding "The base filter interface does not cover the differentiable likelihood protocol."

Shared types:

```python
class FilterRunResult(NamedTuple):
    estimates: PyTree
    loglik: Array[()]
    diagnostics: PyTree[str, Array]

class LikelihoodResult(NamedTuple):
    loglik: Array[()]
    aux: PyTree[str, Array]
```

Non-gradient fast family:

```python
# jpmljax/core/registries.py
FilterBuilder = Callable[[dict[str, Any]], FilterAlgorithm]
FILTER_REGISTRY: dict[str, FilterBuilder] = {}

def register_filter(name: str, *, schema: type[BaseModel]) -> Callable[[FilterBuilder], FilterBuilder]: ...

class FastFilterSpec(BaseModel):
    kind: str
    params: dict[str, Any] = {}
    resampling: ResamplingPolicy | None
    diagnostics: DiagnosticSpec

fast_filter_apply(
    spec: FastFilterSpec,
    model: StateSpaceModel,
    params: Params,
    observations: Array[T, obs_dim],
    key: Key,
) -> FilterRunResult
```

Differentiable likelihood family:

```python
class DiffLikelihoodSpec(BaseModel):
    kind: str
    params: dict[str, Any] = {}
    resampling: GradientResamplingPolicy
    diagnostics: DiagnosticSpec

diff_loglik_apply(
    spec: DiffLikelihoodSpec,
    model: StateSpaceModel,
    params: Params,
    observations: Array[T, obs_dim],
    key: Key,
) -> LikelihoodResult

class MultiStepFlowFilter(Protocol):
    n_flow_steps: int
    def flow_step(carry, substep: int, key: Key) -> tuple[Any, FlowStepResult]: ...
    def apply_flow(model, params, particles, observation, key, params_dict) -> FlowApplyResult: ...

class FlowStepResult(NamedTuple):
    particles: PyTree
    log_abs_det_jacobian: Array
    aux: PyTree[str, Array]

class FlowApplyResult(NamedTuple):
    particles: PyTree
    accumulated_logdet: Array
    diagnostics: PyTree[str, Array]
```

Concrete filters register builders and per-kind Pydantic parameter schemas in `FILTER_REGISTRY`; `kind` is validated against that registry at config-load/build time. Adding a filter is one new registration entry in a concrete module, not a framework enum edit. Analytic filters check Gaussian-moment capabilities before using them. Multi-step flow filters implement the generic `MultiStepFlowFilter` contract with `n_flow_steps`; named flow algorithms are registered instances of that contract.

Design goals satisfied: arbitrary SSMs, diagnostic control, separate algorithm families. JIT boundary: `fast_filter_apply` and `diff_loglik_apply` are individually jitted with algorithm kind and static dimensions as static config. Time loops use `lax.scan`; particles use `vmap`.

### Sampler

Purpose: provide a shared orchestration shape while letting registered sampler algorithms specialize around different transition kernels.

Interface sketch:

```python
# jpmljax/core/registries.py
SamplerBuilder = Callable[[dict[str, Any]], SamplerAlgorithm]
SAMPLER_REGISTRY: dict[str, SamplerBuilder] = {}

def register_sampler(name: str, *, schema: type[BaseModel]) -> Callable[[SamplerBuilder], SamplerBuilder]: ...

class SamplerSpec(BaseModel):
    kind: str
    params: dict[str, Any] = {}
    num_warmup: int | None
    num_samples: int | None
    diagnostics: DiagnosticSpec
    clipping: ClipSpec | None

class SamplerResult(NamedTuple):
    samples: PyTree
    summary: PyTree
    diagnostics: PyTree[str, Array]

build_logdensity(
    model: StateSpaceModel,
    schema: ParameterSchema,
    frozen_params: Params,
    likelihood_spec: DiffLikelihoodSpec,
    observations: Array,
) -> Callable[[UnconstrainedParams, Key], tuple[Array[()], Aux]]

class SamplerAlgorithm(Protocol):
    def init(self, position: PyTree, objective, params: dict[str, Any], key: Key) -> SamplerState: ...
    def step(self, state: SamplerState, key: Key) -> tuple[SamplerState, SamplerInfo]: ...
    def run(self, init_position: PyTree, objective, key: Key, params: dict[str, Any]) -> SamplerResult: ...
```

Concrete samplers register builders and per-kind Pydantic parameter schemas in `SAMPLER_REGISTRY`; `kind` is validated against that registry at config-load/build time. Gradient-based samplers consume differentiable likelihoods; non-gradient samplers can consume non-gradient log densities without being forced through a gradient-specific interface.

Design goals satisfied: arbitrary parameters, gradient-clipping control, diagnostic control. JIT boundary: registered sampler step functions are jitted where applicable; long chains or optimization trajectories use `lax.scan`; host wrappers handle summaries and result writing.

### Diagnostic

Purpose: replace embedded prints, callbacks, and `.numpy()` conversions diagnosed under "Effectful Code Inside or Near Hot Paths."

Interface sketch:

```python
class MetricSpec(BaseModel):
    name: str
    source: Literal["filter", "sampler", "optimizer", "resampler"]
    reducer: Literal["last", "mean", "histogram", "full_trace"]
    frequency: int = 1

class ParticleHistorySpec(BaseModel):
    enabled: bool = True
    reducer: Literal["full", "summary"] = "full"
    sink: Literal["json", "npy", "none"] = "npy"

class DiagnosticSpec(BaseModel):
    enabled: bool
    metrics: list[MetricSpec]
    sinks: list[SinkSpec]
    particle_history: ParticleHistorySpec = ParticleHistorySpec(enabled=True)

class DiagnosticEvent(NamedTuple):
    step: Array[()]
    metrics: dict[str, Array]

collect_metrics(carry, output, spec) -> PyTree[str, Array]
```

Particle-history logging is default-on. Compiled functions return diagnostic arrays as aux outputs. Host-side sinks implement JSON, CSV, NumPy, stdout, or no-op writing.

Design goals satisfied: full diagnostic control. JIT boundary: metric calculation can be inside JIT if it is pure and shape-stable; sink writing is always outside JIT.

### Optimizer and Gradient-Clipping Hook

Purpose: make clipping configurable outside training loops, replacing ad hoc clipping and gradient zeroing from the current DPF runner.

Interface sketch:

```python
class ClipSpec(BaseModel):
    mode: Literal["none", "global_norm", "per_parameter_norm", "value", "nan_to_zero"]
    global_norm: float | None = None
    per_parameter: dict[str, float] = {}
    value: float | None = None
    apply_to: Literal["grad", "momentum", "both"] = "grad"

build_gradient_transform(optimizer_spec, clip_spec) -> optax.GradientTransformation
clip_gradients(grads: Params, params: Params, spec: ClipSpec) -> tuple[Params, ClipMetrics]
```

Registered optimization algorithms use Optax directly. Custom gradient-based samplers can use the same `clip_gradients` function inside their transition if their registered semantics allow it; third-party MCMC kernels should normally run unclipped unless the sampler config explicitly selects a clipped experimental transition.

Design goals satisfied: full gradient-clipping control. JIT boundary: clipping is a pure tree transformation inside MAP or custom sampler steps. Clip metrics are returned as diagnostics, not printed.

### Resampling Policy

Purpose: make conditional versus smooth/always resampling first-class, fixing the diagnosis finding "Tensor-dependent branch points create non-smooth or non-JIT-safe paths."

Interface sketch:

```python
# jpmljax/core/registries.py
ResamplerBuilder = Callable[[dict[str, Any]], Resampler]
RESAMPLER_REGISTRY: dict[str, ResamplerBuilder] = {}

def register_resampler(name: str, *, schema: type[BaseModel]) -> Callable[[ResamplerBuilder], ResamplerBuilder]: ...

class ResamplingPolicy(BaseModel):
    mode: Literal["conditional", "always", "never", "schedule"]
    method: str
    ess_threshold: float | None = 0.5
    params: dict[str, Any] = {}

class GradientResamplingPolicy(BaseModel):
    mode: Literal["always", "smooth", "never"]
    method: str
    stop_gradient: bool = False
    params: dict[str, Any] = {}

class Resampler(Protocol):
    family: Literal["discrete_ancestor", "differentiable_transport", "identity"]
    def resample(self, particles, log_weights, key, params) -> ResampleOutput: ...

resample(policy, particles, weights, key, aux_state) -> ResampleOutput
```

Fast non-gradient PF defaults to `mode="conditional"` with ESS threshold. Gradient-facing filters must choose `always`, `smooth`, or `never`; conditional ESS resampling is not the default in differentiable likelihoods. Concrete methods register builders and per-kind schemas in `RESAMPLER_REGISTRY`; `method` is validated against that registry at config-load/build time.

Design goals satisfied: full diagnostic control and separate algorithm-family control. JIT boundary: policy mode and method are static. ESS comparisons use `lax.cond` in fast filters; differentiable filters avoid discontinuous conditional policy by default.

### Test-Result Sink

Purpose: enforce the user decision that every test saves numerical results to JSON through a uniform helper. This updates the uneven discipline diagnosed under "Test Design / Tests mix assertions, diagnostics, saved artifacts, and scripts."

Interface sketch:

```python
class TestCaseResult(BaseModel):
    schema_version: str
    test_id: str
    kind: Literal["gate", "smoke", "diagnostic", "benchmark"]
    status: Literal["pass", "fail", "skip", "xfail"]
    metrics: dict[str, float | int | bool | str | list]
    tolerances: dict[str, float] = {}
    config_digest: str | None = None
    artifacts: dict[str, str] = {}

def save_result(test_file: str, case: TestCaseResult, *, sink: TestResultSink | None = None) -> None: ...
def reset_results(test_file: str, *, sink: TestResultSink | None = None) -> None: ...
```

Design goals satisfied: full diagnostic control and uniform test contract. JIT boundary: none. Test result saving is host-only and should be pytest-integrated.

## 4. How the Five Design Goals Are Realized

### Arbitrary SSMs

The `StateSpaceModel` abstraction requires only pure functions over state, inputs, parameters, keys, and a `noise_spec`, plus static metadata. Gaussian moments, closed-form priors, and analytic Jacobians are capability protocols rather than base requirements. Particle batching is provided by `vmap`; time recursion is provided by filter-level `scan`. This avoids the current large object protocol diagnosed under "Model and Data Boundaries" and removes Python-loop batch fallbacks.

Concrete enforcement: every model exposes `schema`, `state_dim`, `obs_dim`, and the pure functions in section 3. A model cannot require mutable internal state to run.

### Arbitrary Parameters

`ParameterSchema` owns all parameter names, shapes, transforms, priors, initial values, true values, and learnability flags. Samplers receive unconstrained learnable pytrees; filters receive full constrained parameter pytrees.

Concrete enforcement: `build_logdensity` is the only place that merges learnable and frozen params for inference. No filter or model mutates parameter values.

### Per-Parameter Learnable Flag

Each `ParameterSpec` has `learnable: bool`. `partition_params` and `merge_params` are mandatory paths for sampler construction. This directly replaces current YAML coupling between `model`, `dpf.trainable_params`, and `data.true_params`.

Concrete enforcement: a scenario config is invalid if a trainable parameter has no transform/prior policy where required, or if a sampler references a name absent from the schema.

### Full Diagnostic Control

`DiagnosticSpec` selects metrics, frequency, reducers, and sinks. Core kernels return diagnostic arrays; host sinks serialize them. No core filter prints, writes files, or calls callbacks.

Concrete enforcement: experiment runners pass `DiagnosticSpec` into filters and samplers, then materialize returned diagnostics through `diagnostics/sinks.py`.

### Full Gradient-Clipping Control

`ClipSpec` is independent of concrete sampler or optimizer implementations. Registered optimization algorithms compose clipping through Optax transformations. Custom gradient-based samplers use the same pure `clip_gradients` hook only when their registered semantics make that statistical change explicit.

Concrete enforcement: training loops do not contain hardcoded clipping thresholds. They receive a `ClipSpec` and return `ClipMetrics`.

## 5. Dependency Choices

### HMC / NUTS: Pick BlackJAX

Recommendation: BlackJAX for HMC/NUTS. BlackJAX exposes low-level kernels around a user-provided `logdensity_fn`, with state/info step APIs suitable for explicit `jit`, `scan`, `vmap`, and custom diagnostics. Its NUTS docs show direct kernel construction with `state = nuts.init(position)` and `new_state, info = nuts.step(rng_key, state)` (official docs: https://blackjax.readthedocs.io/en/latest/autoapi/blackjax/mcmc/nuts/index.html).

Losing alternatives: NumPyro and hand-rolled HMC. NumPyro has a mature high-level MCMC interface and supports potential functions, but it brings a probabilistic-programming model layer and MCMC object orchestration that is more than this rebuild needs (official docs: https://num.pyro.ai/en/0.16.1/mcmc.html). Hand-rolled HMC keeps maximum control but repeats risky adaptation, mass-matrix, divergence, and tree-building work. BlackJAX is the better fit for explicit SSM likelihoods and custom result plumbing.

### Model Containers / Pytree Framework: Pick Equinox

Recommendation: Equinox. It represents parameterized objects as pytrees and provides filtered JAX transformations where array leaves are dynamic and non-array leaves are static. This maps directly to the diagnosis requirement to separate static model structure from dynamic arrays (official docs: https://docs.kidger.site/equinox/api/transformations/).

Losing alternatives: Flax NNX and plain pytrees. Flax NNX is powerful, but its graph/state split is aimed at stateful neural network modules and is more machinery than pure SSM kernels need (official docs: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html). Plain JAX pytrees minimize dependencies but require more boilerplate for static-vs-dynamic partitioning, model ergonomics, and filtered transforms. Equinox gives the needed structure without reintroducing mutable model attributes.

### Optimizer: Pick Optax

Recommendation: Optax. It is built around pure gradient transformations with explicit optimizer state, and it already provides global norm clipping, adaptive clipping, value clipping, schedules, and transformation chaining (official docs: https://optax.readthedocs.io/en/stable/api/transformations.html).

Losing alternatives: hand-rolled Adam/SGD and library-specific optimizer stacks. Hand-rolled optimizers would repeat schedule and clipping logic already needed by the design. Equinox does not replace Optax; it composes with it. Optax is the obvious fit for MAP and gradient-transformation hooks.

### Shape/Type Assertions: Pick Chex for Tests and Development Guards

Recommendation: Chex, used in tests and optional debug validation, not inside every hot kernel. Chex provides array shape/rank/device assertions and dataclass helpers (official docs: https://chex.readthedocs.io/en/latest/api.html).

Losing alternatives: raw `assert`, jaxtyping-only annotations, and no assertions. Raw asserts are uneven and often stripped or host-only. Type annotations help humans but do not validate runtime shape. No assertions would repeat current ambiguity around model/filter contracts. Chex should be kept out of production hot loops unless guarded by debug config.

### Config System: Pick Pydantic v2 + YAML Loader

Recommendation: Pydantic v2 models are the source of truth for validating one whole-run scenario config. On-disk human-edited scenario files are YAML. YAML and JSON loaders both build the same Pydantic model, and MCP tools exchange JSON over the wire using JSON schema auto-generated from that model. Pydantic gives explicit schemas, validation errors, serialization, and JSON schema generation without requiring config composition. Its docs describe model validation and structured fields as primary use cases (official docs: https://docs.pydantic.dev/latest/concepts/models/).

Algorithm config sections use open registry keys rather than closed Pydantic unions. A section contains `kind: str` plus `params: dict[str, Any]`; the registry entry for that `kind` supplies the per-kind Pydantic parameter schema used at load/build time.

Losing alternatives: Hydra, ml_collections, fiddle, and draccus/pyrallis. Hydra is ruled out by the no defaults-list composition constraint and is explicitly implicated in the config-layering diagnosis. ml_collections `ConfigDict` is common in JAX projects and has immutable `FrozenConfigDict`, but it is weaker for external YAML validation and discriminated unions (official docs: https://ml-collections.readthedocs.io/en/stable/config_dict.html). Fiddle is oriented toward object graph construction, which conflicts with explicit pure function boundaries. Draccus/pyrallis is close to the dataclass scenario-config style, but Pydantic v2 has stronger validation and schema tooling for a growing config surface.

## 6. Diagnostic + Gradient-Clipping Plug-In Design

Diagnostics are configured through `DiagnosticSpec`, not embedded in model, filter, or sampler code.

Interface sketch:

```python
class SinkSpec(BaseModel):
    kind: Literal["json", "csv", "npy", "stdout", "none"]
    path: str | None = None
    flush_every: int | None = None

class DiagnosticSpec(BaseModel):
    enabled: bool = True
    metrics: list[MetricSpec]
    sinks: list[SinkSpec]
    frequency: int = 1

class DiagnosticBundle(NamedTuple):
    arrays: dict[str, Array]
    metadata: dict[str, Any]
```

Compiled code returns `DiagnosticBundle.arrays`. Host code materializes it:

```python
result = run_compiled(...)
write_diagnostics(result.diagnostics, config.diagnostics.sinks)
```

Gradient clipping is built before the optimization loop:

```python
clip_spec = scenario.optim.clipping
tx = optax.chain(
    build_clip_transform(clip_spec),
    build_optimizer_transform(scenario.optim),
)
```

For per-parameter clipping, the `ClipSpec.per_parameter` map is compiled into a mask tree aligned with `ParameterSchema`. For HMC, clipping is not applied to BlackJAX NUTS/HMC by default; if a scenario requests clipping, it must select an explicit sampler kind such as `custom_hmc_clipped` so the statistical change is visible in the config.

## 7. JIT Strategy

Startup:

- `startup.py` calls `jax.config.update("jax_enable_x64", True)` before importing computational modules.
- Scenario config parsing, Pydantic validation, path setup, and output directory creation stay outside JIT.

Fast non-gradient filters:

- JIT boundary: `fast_filter_apply(spec_static, model_static, params, observations, key)`.
- Static: algorithm kind, dimensions, resampling method, diagnostic metric selection, fixed scan lengths if required.
- Traced: params, observations, PRNG key, filter carry arrays.
- `lax.scan` replaces time loops.
- `vmap` runs particle transitions, log weights, observation evaluations, and per-particle Jacobian defaults where needed.
- Conditional ESS resampling uses `lax.cond`; this family does not promise smooth gradients through resampling.

Differentiable likelihood filters:

- JIT boundary: `diff_loglik_apply(spec_static, model_static, params, observations, key)`.
- Static: differentiable resampling policy, particle count, flow step count, metric selection.
- Traced: params, observations, PRNG key, carry arrays.
- `lax.scan` handles time; inner flow steps use `scan` if step count should remain compact or static unrolling if profiling proves it faster.
- No host callbacks, file writes, or Python conditionals on traced values.

Samplers:

- HMC/NUTS: BlackJAX one-step kernel is jitted; full chains use `lax.scan`; multiple chains use `vmap` over initial states and keys. Across devices, `pmap` or sharding can be added later.
- MAP: one train step is jitted; full optimization can use `lax.scan` when diagnostics are fixed-shape, or a host loop for interactive diagnostics.
- MH: proposal and accept/reject are `lax.scan` compatible. It can consume non-gradient log-density.

Must stay outside JIT:

- Config parsing and validation.
- Enabling `jax_enable_x64`.
- Filesystem I/O, JSON/CSV/NPY writing, plotting.
- Progress bars and stdout logging.
- Dynamic import or scenario registry resolution.
- Test result saving.
- Wall-clock timing, except explicit benchmark wrappers using `block_until_ready()`.

## 8. Testing Discipline

Interfaces are built from generic protocols and open registries; they never name specific algorithms or models. Gate tests pull the hardest existing concrete fixtures from `jpmljax/config/scenarios/` and the TF reference code; concrete names belong in tests and fixtures only. Adding a new algorithm or model is a registration entry plus a fixture update, never an interface change.

Every test uses the uniform `save_result` helper. This is a contract, not optional instrumentation.

Test tolerances live in test code, not in scenario configs. Saved JSON may record the tolerances applied by the test for auditability, but scenario YAML is not the source of expected-test thresholds.

Pytest wiring:

```python
@pytest.fixture
def result_sink(request) -> TestResultSink:
    sink = JsonTestResultSink(root="tests/results", test_file=request.node.path)
    sink.reset_once_per_file()
    return sink

def test_case(result_sink):
    metrics = run_case(...)
    result_sink.save(TestCaseResult(..., metrics=metrics, status="pass"))
    assert metrics["rel_err"] <= tolerance
```

Enforcement options:

- Tests that produce numerical metrics must accept `result_sink`.
- A pytest hook records whether `save_result` was called for each test marked `@pytest.mark.numerical`.
- Tests declare expected schema through `TestCaseResult.kind` and `schema_version`.

Gate tests versus smoke tests:

- Gate tests are deterministic, small, assert tolerances, and save compact JSON metrics.
- Every gate test for a generic abstraction must exercise that abstraction against the hardest concrete instance currently in the project. Concrete instance names are allowed and expected inside tests. Interfaces remain generic.
- Smoke tests may run longer, still assert minimal health conditions, and save richer diagnostics.
- Diagnostic tests can be marked separately, may save traces, and should not be required for fast CI unless selected.
- Benchmarks are separate from correctness tests and must call `block_until_ready()` before timing.

Expected JSON shape:

```json
{
  "schema_version": "jax-ssm-test-v1",
  "file": "test_name.py",
  "cases": [
    {
      "test_id": "string",
      "kind": "gate",
      "status": "pass",
      "metrics": {},
      "tolerances": {"metric_name": 1e-6},
      "config_digest": "optional",
      "artifacts": {}
    }
  ]
}
```

## 9. Migration Strategy

Port by abstraction completeness, not model simplicity. Each new abstraction is added with the hardest existing concrete fixture that exercises it:

1. Runtime and test foundation: global float64 startup, PRNG key conventions, scenario validation, and uniform test result sink.
2. Parameter/model protocol: validate transforms, priors, learnable/frozen partitioning, non-additive `noise_spec`, nonlinear autodiff Jacobians, and a concrete Gaussian-capability reference model.
3. Analytic filter capability: gate the registered analytic filter instance against a learnable-parameter log-likelihood gradient path.
4. Resampling capability: gate all current discrete resamplers plus at least one current differentiable resampler through the same registry-backed policy surface.
5. Particle/flow filter capability: gate the registered multi-step flow filter with the current hard 29-step Jacobian-accumulation fixture in float64, not only a bootstrap particle filter.
6. Differentiable likelihood capability: gate autodiff versus finite-difference agreement through the multi-step flow Jacobian-accumulation path.
7. Sampler/optimizer capability: gate registered gradient-based samplers and optimizers on the current hard high-dimensional nonlinear likelihood target, not only the analytic Gaussian objective.
8. MCP/runner capability: round-trip and execute a real hard scenario through the same generic scenario schema and registries.
9. Model coverage: port all current model configs under `code/configs/model/`: `1d_linear_gaussian`, `1d_linear_gaussian_tiny_obs_noise`, `2d_linear_gaussian`, `2d_linear_gaussian_tiny_obs_noise`, `5d_linear_gaussian`, `5d_linear_gaussian_partial_strong`, `5d_linear_gaussian_partial_weak`, `acoustic_tracking_full`, `cubic_sensor`, `kitagawa`, `lorenz96`, `range_bearing`, `stochastic_volatility`, `stochastic_volatility_2d`, `stochastic_volatility_log`, and `two_sensor_bearing`.

Hard-instance checklist for reviewers: model/protocol gates include multiplicative-noise fixture, nonlinear-Jacobian finite-difference check, full prior/transform enumeration, and deeply nested parameter pytrees; config/MCP gates include a real hard scenario fixture; particle/likelihood gates include the current multi-step flow Jacobian-accumulation fixture.

Drop or do not port:

- TensorFlow object wrappers, `tf.Variable` filter state, and `setattr` parameter mutation from `diagnosis.md` "Parameter Handling."
- Hydra defaults-list composition and grouped config directory structure from "Config Layering"; scenario configs replace it.
- TF-specific XLA workarounds such as `.python_function` bypasses, TensorFlow custom-gradient structure, and `.numpy()` host conversions in hot paths.
- Script-local `sys.path` manipulation in tests.
- `hmc_runner_old.py`-style compatibility paths unless a specific numerical comparison requires them.

Issues fixed by architecture:

- Numeric dtype drift is removed by global `float64` startup and no critical-path dtype toggles.
- Mutable filter state is removed by pure carries and `scan`.
- Resampling branch policy is explicit by algorithm family.
- Diagnostic side effects move outside JIT.
- Test artifact saving becomes uniform.

TF-only issues that vanish by not porting TF:

- TensorFlow Probability distribution dtype casting.
- TensorFlow `tf.Variable.assign` gradient-chain concerns.
- TensorFlow graph/eager split and `.numpy()` diagnostics inside GradientTape paths.
- XLA-specific TensorList and `tf.function` boundary workarounds.

## 10. Open Questions for the User

None from the current locked constraints.

## 11. MCP Server Interface

The repo ships an MCP server under `jpmljax/mcp/` so Claude Desktop can control the pipeline through a thin user-facing shell. The MCP layer contains no model, filter, sampler, optimizer, diagnostic, or validation business logic. It delegates to `config/` for schema loading and validation, `experiments/` for scenario execution, and `diagnostics/test_results.py` for saved test-result access.

Minimum tool surface:

```python
list_scenario_configs() -> list[ScenarioConfigSummary]

validate_scenario_config(
    payload: dict | str,
    format: Literal["yaml", "json"],
) -> ValidationResult

run_scenario(
    scenario: dict,
    output_root: str | None = None,
) -> RunScenarioResult

list_test_results(
    root: str | None = None,
) -> list[TestResultSummary]

fetch_test_result(
    result_id: str,
) -> dict

get_scenario_config_json_schema() -> dict
```

Tool contracts:

- `list_scenario_configs` reads the configured scenario directory and returns names, paths, model kind, filter kind, sampler kind, and short descriptions where present.
- `validate_scenario_config` accepts YAML text or JSON objects, builds the same Pydantic scenario model used by CLI and Python entry points, and returns structured validation errors.
- `run_scenario` calls the same `experiments/` runner as the CLI, returns the output directory, summary stats, config digest, and paths to saved diagnostics.
- `list_test_results` enumerates JSON files produced through the uniform `save_result` contract.
- `fetch_test_result` returns one saved test-result JSON without interpreting its metrics.
- `get_scenario_config_json_schema` returns the Pydantic-generated JSON schema used by the MCP JSON wire format.

The MCP gate test must round-trip the hardest existing concrete scenario fixture through `validate_scenario_config` and `run_scenario`; currently that fixture is the SV2D LEDH scenario adapted from the TF code. The MCP schema and tool signatures remain generic; the test fixture carries the concrete model and algorithm names.

Config interaction:

- On-disk scenario files are YAML.
- MCP wire payloads are JSON.
- Both YAML and JSON inputs are validated against the same Pydantic v2 model.
- JSON schema is generated from that Pydantic model; there is no separate MCP schema definition.

JIT boundary:

- MCP tools stay entirely outside JIT.
- MCP tools can trigger compiled runners through `experiments/`, but never call model/filter/sampler kernels directly.
- Long-running `run_scenario` reports host-side status and returns only after the runner has materialized outputs.
