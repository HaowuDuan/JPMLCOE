# Design-Pressure Audit v2: Abstract API vs Concrete Gate Instances

This audit corrects the prior audit by separating two concerns:

- Abstractions must remain model-agnostic and algorithm-agnostic.
- Gate tests must apply design pressure using the hardest concrete instances currently known.

## Part A: Abstraction Leaks

### `design.md`

#### Finding A1: `StateSpaceModel` assumes Gaussian covariance APIs as universal model methods

Severity: **MAJOR**

Location: `design.md` section 3 `StateSpaceModel`, interface methods `transition_cov`, `observation_cov`, and `log_observation_prob`.

Leak: The interface quietly privileges models whose process and observation noise can be represented as covariance matrices. That fits additive Gaussian models and analytic filters, but not a universal state-space model with multiplicative noise or arbitrary noise structure.

Generic replacement: Keep state-space dynamics generic and move Gaussian moments into optional capability protocols:

```python
class StateSpaceModel(eqx.Module):
    state_dim: int
    obs_dim: int
    schema: ParameterSchema
    noise_spec: NoiseSpec
    def sample_initial(params, key) -> State: ...
    def sample_transition(params, x, key, t) -> State: ...
    def transition_mean(params, x, t) -> State | NotAvailable: ...
    def observation(params, x, key, t) -> Obs: ...
    def log_observation_prob(params, y, x, t) -> Array: ...

class GaussianMomentModel(Protocol):
    def initial_moments(params) -> tuple[State, Array]: ...
    def transition_moments(params, x, t) -> tuple[State, Array]: ...
    def observation_moments(params, x, t) -> tuple[Obs, Array]: ...
```

Analytic Gaussian filters require `GaussianMomentModel`; particle filters require sampling/log-prob capabilities. The base SSM does not require covariance methods.

#### Finding A2: `FilterSpec.kind` bakes concrete algorithm names into the interface

Severity: **MAJOR**

Location: `design.md` section 3 `Filter`, `FastFilterSpec.kind: Literal["bootstrap_pf", "ledh_flow", "kalman", "ekf", "ukf"]` and `DiffLikelihoodSpec.kind: Literal["bootstrap_pf_diff", "ledh_diff", "kalman_loglik"]`.

Leak: The abstraction requires framework code changes to add a new filter algorithm. It also mixes filter family with concrete algorithm name.

Generic replacement: Split family-level interface from concrete algorithm registry:

```python
class FilterSpec(BaseModel):
    family: Literal["particle", "analytic", "flow"]
    algorithm: str
    settings: dict[str, Any]

class FilterAlgorithm(Protocol):
    family: FilterFamily
    def apply(model, params, observations, key, settings) -> FilterRunResult: ...

class DifferentiableLikelihoodAlgorithm(Protocol):
    family: FilterFamily
    def loglik(model, params, observations, key, settings) -> LikelihoodResult: ...
```

Concrete instances such as current filters register under `algorithm`, but the interface remains family-generic.

#### Finding A3: Flow support is named by one algorithm instead of a generic multi-step flow contract

Severity: **MAJOR**

Location: `design.md` section 3 `Filter` and section 7 `JIT Strategy`, phrases `ledh_flow`, `ledh_diff`, and `flow step count`.

Leak: The design recognizes flow filters but only through named concrete algorithms. It does not define a generic multi-step flow transform contract that can support any Jacobian-accumulating flow.

Generic replacement: Add a model-agnostic and algorithm-agnostic flow protocol without naming the concrete flow algorithm:

```python
class FlowStepResult(NamedTuple):
    particles: PyTree
    log_abs_det_jacobian: Array  # shape (N,)
    aux: PyTree[str, Array]

class ParticleFlowAlgorithm(Protocol):
    def flow_step(carry, t, substep, key) -> tuple[carry, FlowStepResult]: ...
    def apply_flow(model, params, particles, observation, key, settings) -> FlowApplyResult: ...

class FlowApplyResult(NamedTuple):
    particles: PyTree
    accumulated_logdet: Array    # shape (N,)
    diagnostics: PyTree[str, Array]
```

Gate tests can use the current hardest concrete flow instance, but the interface should say only multi-step flow with accumulated log-determinants.

#### Finding A4: `SamplerSpec.kind` bakes concrete sampler names into the interface

Severity: **MAJOR**

Location: `design.md` section 3 `Sampler`, `SamplerSpec.kind: Literal["hmc", "nuts", "map", "mh"]`, plus concrete functions `run_hmc`, `run_map`, and `run_mh`.

Leak: The sampler abstraction is a list of current algorithms rather than a generic sampler/optimizer protocol. Adding a new sampler requires extending core interface names.

Generic replacement: Use a generic algorithm registry with shared input/output contracts:

```python
class SamplerSpec(BaseModel):
    family: Literal["gradient_mcmc", "optimizer", "random_walk", "custom"]
    algorithm: str
    settings: dict[str, Any]

class SamplerAlgorithm(Protocol):
    def init(position, logdensity, settings, key) -> SamplerState: ...
    def step(state, key) -> tuple[SamplerState, SamplerInfo]: ...
    def run(initial_position, logdensity, key, settings) -> SamplerResult: ...
```

Concrete HMC/MAP/MH implementations live as registered algorithms, not as the interface itself.

#### Finding A5: Resampling methods are closed over current names

Severity: **MAJOR**

Location: `design.md` section 3 `Resampling Policy`, `method: Literal["systematic", "stratified", "multinomial", "soft", "ot"]`; `step4.md` section 2 mirrors this for core stochastic resamplers.

Leak: The policy interface is closed over the current method list. Adding a new resampler is a framework schema change instead of registering a new method.

Generic replacement: Keep family/method contracts separate:

```python
class ResamplingPolicy(BaseModel):
    mode: Literal["conditional", "always", "never", "schedule"]
    method: str
    family: Literal["discrete_ancestor", "differentiable_transport", "identity"]
    ess_threshold: float | None = 0.5
    params: dict[str, Any] = {}

class Resampler(Protocol):
    family: ResamplingFamily
    def resample(particles, log_weights, key, params) -> ResampleOutput: ...
```

The current resamplers should be registered concrete instances. The abstract output must cover both ancestor-based and transport-based methods, for example with optional `ancestors` and `transport_matrix` fields.

#### Finding A6: Module layout names concrete algorithms in core locations

Severity: **MINOR**

Location: `design.md` section 2 `Module Layout`, paths such as `filters/kalman.py`, `filters/ledh.py`, `samplers/hmc.py`, `samplers/map.py`, and `resampling/ot.py`.

Leak: Concrete filenames are not inherently wrong, but the layout could imply core framework extension requires editing named top-level modules.

Generic replacement: Keep concrete implementation modules, but add registry layers such as `filters/registry.py`, `samplers/registry.py`, and `resampling/registry.py`. The universal API imports protocols and registry entry points, not concrete algorithm modules.

### `step1a.md`

No abstraction leak found. The phase is process-level startup and package scaffold. It does not encode a model, filter, sampler, or resampling API.

### `step1b.md`

#### Finding A7: Config schema is generic enough but too unstructured to express algorithm capabilities

Severity: **MINOR**

Location: `step1b.md` section 2 `What Gets Built`, `ModelConfig.kind: str`, `FilterConfig.kind: str`, `SamplerConfig.kind: str | None`, and generic `settings` dicts.

Leak: This does not bake in concrete names, which is good. The risk is the opposite: it does not distinguish family/capability from concrete algorithm string, so later code may overload `kind` with both.

Generic replacement: Use generic capability fields without naming concrete algorithms:

```python
class ModelConfig(BaseModel):
    family: str | None = None
    name: str
    parameters: dict[str, Any]

class FilterConfig(BaseModel):
    family: Literal["particle", "analytic", "flow"]
    algorithm: str
    settings: dict[str, Any]

class SamplerConfig(BaseModel):
    family: Literal["gradient_mcmc", "optimizer", "random_walk", "none"]
    algorithm: str | None
    settings: dict[str, Any]
```

### `step1c.md`

No abstraction leak found. The result sink is generic test infrastructure. Its `artifacts: dict[str, str]` field is generic and does not name model or algorithm instances.

### `step2.md`

#### Finding A8: `StateSpaceModel` protocol repeats covariance-method assumption

Severity: **MAJOR**

Location: `step2.md` section 2 `What Gets Built`, `StateSpaceModel` methods `initial_cov`, `transition_cov`, and `observation_cov`.

Leak: Same as `design.md` A1. The step turns covariance methods into base protocol requirements, privileging additive Gaussian/moment-available models.

Generic replacement: In `jpmljax/core/model.py`, make the base model sampling/log-prob protocol generic and define moment/covariance protocols as optional capabilities required only by analytic filters.

#### Finding A9: `LinearGaussianModel` is correctly concrete, but the step risks teaching the base API around it

Severity: **MINOR**

Location: `step2.md` section 2 `What Gets Built`, `LinearGaussianStatic` and `LinearGaussianModel`.

Leak: The concrete model is allowed. The issue is not the concrete instance; it is the surrounding base protocol if it mirrors Linear Gaussian covariance structure.

Generic replacement: Keep `LinearGaussianModel` as the first concrete instance, but explicitly state that it implements an optional `GaussianMomentModel` capability in addition to the generic `StateSpaceModel`.

### `step3.md`

#### Finding A10: Concrete Kalman phase is acceptable, but `filters/common.py` must not become Kalman-shaped

Severity: **MINOR**

Location: `step3.md` section 2 `What Gets Built`, `FilterRunResult`, `LikelihoodResult`, `KalmanCarry`, and `KalmanStepOutput`.

Leak: `KalmanCarry` and `KalmanStepOutput` are concrete and acceptable. The potential leak would be if `FilterRunResult.estimates` or `diagnostics` is documented or implemented as always containing means/covariances.

Generic replacement: Keep `FilterRunResult.estimates: PyTree` and `diagnostics: PyTree[str, Array]` generic. Put Kalman-specific means/covariances under a concrete Kalman output schema, not in the universal result type.

### `step4.md`

#### Finding A11: Resampling interface is closed over concrete method names

Severity: **MAJOR**

Location: `step4.md` section 2 `What Gets Built`, `ResamplingPolicy.method: Literal["systematic", "stratified", "multinomial", "soft", "ot"]`, plus individual method-specific function names.

Leak: Method-specific functions are fine as concrete implementations, but the policy schema should not be closed over current methods if new resamplers must be addable without framework API changes.

Generic replacement: Keep concrete functions for the current resamplers, but make the universal dispatch use `method: str`, `family`, and a registry. Gate tests can still enumerate current method registrations.

## Part B: Gate-Test Coverage Gaps

### `design.md`

#### Finding B1: Testing discipline does not state hard-instance gate rule

Severity: **MAJOR**

Location: `design.md` section 8 `Testing Discipline`.

What's missing: The generic test rule that abstraction gates must use the hardest current concrete instance. Generic capability not under test: high-dimensional nonlinear likelihood, multi-step flow Jacobian accumulation, differentiable likelihood gradient agreement, real scenario round-trip.

Recommended test addition: Add a section-level rule: gate tests for each abstraction must name the hardest current fixture. Current concrete fixtures should include SV2D for nonlinear model/sampler pressure, the current flow particle filter with 29-step Jacobian accumulation for flow/differentiable-likelihood pressure, and real SV2D flow scenario YAML for config/MCP pressure.

#### Finding B2: Migration plan still sequences simple gates before hard gates

Severity: **MAJOR**

Location: `design.md` section 9 `Migration Strategy`, items 2 through 6.

What's missing: The generic capabilities are not pressure-tested early: particle-flow carry shape, multi-step logdet accumulation, high-dimensional constrained parameter pytrees, and differentiable likelihood gradients.

Recommended test addition: Rewrite migration gates so the first particle-flow/differentiable-likelihood gate uses the current hardest flow fixture with 29 steps in `float64`, and the sampler gate uses the current hardest nonlinear model fixture. Do not put those names into interfaces; put them into gate definitions.

### `step1a.md`

No gate-test coverage gap found. Startup x64 is infrastructure, and its gate directly supports the hard cases by enforcing process-level `float64`.

### `step1b.md`

#### Finding B3: Config loader gate uses a minimal Linear Gaussian fixture only

Severity: **MAJOR**

Location: `step1b.md` section 2 `linear_gaussian_gate.yaml` and section 3 `test_load_minimal_yaml_scenario`.

What's missing: Generic capability not under test: whole-run scenario schema can represent a real nested model/filter/resampling/sampler/diagnostics configuration. Hardest current concrete instance: SV2D plus flow particle filter scenario with 29 flow steps, resampling, diagnostics, and output settings.

Recommended test addition: Add a second fixture such as `jpmljax/config/scenarios/sv2d_flow_gate.yaml`. The gate should validate it as YAML and JSON, assert family/algorithm/settings fields survive round-trip, and assert JSON schema includes enough structure for model parameters, filter settings, resampling, diagnostics, and run knobs.

### `step1c.md`

#### Finding B4: Result sink gate does not exercise external artifact references

Severity: **MINOR**

Location: `step1c.md` section 3 `tests/test_test_results.py`.

What's missing: Generic capability not under test: hard numerical tests can save large arrays externally and reference them from JSON. Hardest current concrete instance: flow particle histories and per-flow-step Jacobian/logdet traces.

Recommended test addition: Add a gate case that writes a dummy external artifact path under `artifacts` and asserts the JSON stores a stable relative reference without embedding large arrays.

### `step2.md`

#### Finding B5: Model gate allows empty dynamic params

Severity: **MAJOR**

Location: `step2.md` section 2 line `Linear Gaussian dynamic parameters may be empty in the first gate`; section 3 Option A recommends static matrices only.

What's missing: Generic capability not under test: model functions consume dynamic `params` and respond to learnable/frozen partitioning. Hardest current concrete instance: SV2D parameter tree; nearest immediate fixture: Linear Gaussian with learnable noise scale, already required by updated `step3`.

Recommended test addition: In `step2`, require at least one dynamic parameter consumed by model methods. Test that changing the constrained parameter changes a covariance or log-prob output without mutating model attributes.

#### Finding B6: State-space gate does not exercise multiplicative or non-additive noise

Severity: **MAJOR**

Location: `step2.md` section 3 `tests/test_linear_gaussian_model.py` and section 4 `Linear Gaussian is the reference model`.

What's missing: Generic capability not under test: base SSM can represent additive, multiplicative, or arbitrary noise structure. Hardest current concrete instance: SV2D stochastic volatility, because volatility dynamics create nonlinear/noise-coupled behavior.

Recommended test addition: Add a later but blocking hard-model gate before filters/samplers rely on the protocol: a model implementing the same generic `StateSpaceModel` protocol with multiplicative or nonlinear noise. Use the current SV2D implementation as the fixture.

#### Finding B7: Jacobian gate only tests constant linear Jacobians

Severity: **MAJOR**

Location: `step2.md` section 3 `test_default_jacobians_match_F_and_H`.

What's missing: Generic capability not under test: autodiff Jacobians through nonlinear model functions and repeated Jacobian/logdet accumulation. Hardest current concrete instance: flow particle filter with 29-step Jacobian accumulation.

Recommended test addition: Add a nonlinear Jacobian gate before flow filters. The concrete fixture should eventually be the current 29-step flow path with autodiff-vs-finite-difference agreement through accumulated logdet.

#### Finding B8: Transform/prior gate lacks current-family enumeration and hard parameter trees

Severity: **MAJOR**

Location: `step2.md` section 3 `test_constrain_unconstrain_roundtrip` and `test_log_prior_returns_scalar_float64`; section 4 says tests avoid extreme values.

What's missing: Generic capability not under test: every current transform/prior family works on scalar and structured pytrees, including learnable/frozen mixes and near-boundary finite values. Hardest current concrete instance: current SV2D HMC/MAP parameter schema.

Recommended test addition: Enumerate all transform/prior families currently implemented in the TF code, gate scalar and pytree leaves, include near-boundary finite cases, and include one SV2D-shaped parameter schema fixture.

### `step3.md`

#### Finding B9: Updated Kalman gate is compliant for its concrete analytic-filter phase

Severity: **MINOR**

Location: `step3.md` sections 2 and 3.

What's missing: Nothing for the scope of exact analytic Linear Gaussian filtering. The phase names a concrete analytic filter and now gates learnable-parameter gradient through it. This is a concrete gate, not an abstraction leak.

Recommended test addition: No change needed for this phase. Later generic filter-family gates still need the hard flow fixture, as already stated in `step3.md` section 5.

### `step4.md`

#### Finding B10: Discrete-resampler gate is strong for current discrete methods, but differentiable resamplers remain ungated

Severity: **MAJOR**

Location: `step4.md` section 3 gates systematic, stratified, and multinomial; section 2 says smooth/OT differentiable kernels are not implemented here.

What's missing: Generic capability not under test: differentiable resampling or transport-style output contract. Hardest current concrete instances: soft and OT resamplers used by differentiable particle-filter likelihoods.

Recommended test addition: In the differentiable-likelihood/resampling step, gate soft and OT methods through the same generic `ResampleOutput` contract with gradient checks. Keep their names in tests/registrations, not in the universal interface.

## Part C: Cross-Cutting

### Finding C1: Interfaces need registries; gates need concrete fixtures

Severity: **MAJOR**

Files: `design.md`, `step1b.md`, `step4.md`.

Issue: Several interfaces use `Literal[...]` over concrete algorithm names. The corrected pattern is: universal API uses family/capability plus `algorithm: str` registry key; tests enumerate today’s concrete registered algorithms.

Recommendation: Add registries for filters, samplers, resamplers, and models. Gate tests should assert current concrete registrations exist and pass hard fixtures.

### Finding C2: Base SSM needs noise capability separation

Severity: **MAJOR**

Files: `design.md`, `step2.md`, `step3.md`.

Issue: The base model protocol is too close to analytic Gaussian filtering. Kalman can require Gaussian moment capabilities, but particle filters and nonlinear models should not.

Recommendation: Split base SSM sampling/log-prob protocol from optional Gaussian moment/Jacobian capabilities. Kalman tests gate the Gaussian capability; SV2D/flow tests gate generic nonlinear sampling/log-prob/Jacobian behavior.

### Finding C3: Hard-instance pressure is now present in `step3`/`step4` dependencies but not in `design.md`/`step2`

Severity: **MAJOR**

Files: `design.md`, `step2.md`, `step3.md`, `step4.md`.

Issue: `step3.md` and `step4.md` now state the hardest-case principle for later gates, but `design.md` section 8/9 and `step2.md` still permit easy-case-first validation.

Recommendation: Update `design.md` testing/migration sections and revise `step2.md` to require dynamic parameters and to block downstream PF/sampler abstractions until a nonlinear/multiplicative hard model gate validates the generic SSM protocol.

### Finding C4: Config/MCP gates must use concrete hard scenarios without hardcoding them into schema interfaces

Severity: **MAJOR**

Files: `design.md`, `step1b.md`, `step1c.md`.

Issue: Schema/MCP interfaces should stay generic, but their gates must validate a real hard scenario. Current config gate uses a minimal Linear Gaussian fixture.

Recommendation: Add hard scenario fixtures as test data and validate round-trips through generic schema fields. Do not add `sv2d` or a specific flow name to schema types except as registry data in fixture files.

## Part D: Explicitly NOT Findings

- Prior audit D2 recommendation to add `LEDH` Jacobian accumulation as a first-class named abstraction is rescinded. Correct finding: add a generic multi-step flow/logdet accumulation protocol; use the current LEDH case only as a gate fixture.
- Prior audit D6 recommendation to state `SV2D` as mandatory inside sampler design is rescinded. Correct finding: sampler interfaces must support high-dimensional nonlinear likelihoods generically; SV2D belongs in sampler gate tests.
- Prior audit language implying `StateSpaceModel` should name hard concrete models is rescinded. Correct finding: the base SSM should be even more generic, with `noise_spec`/capability protocols; hard models are concrete tests.
- Prior audit pressure to add concrete scenario names into MCP tool signatures is rescinded. Correct finding: MCP tool signatures stay generic; MCP tests should round-trip a current hard scenario fixture.
- Prior audit framing that all current model names must appear in core interfaces is rescinded. Concrete model names belong in registries, scenario data, and migration/test fixtures, not universal APIs.
