# Phase 06: TFP Samplers, Preconditioned HMC/NUTS, and MAP

## 1. Goal

Build the registry-backed TensorFlow sampler layer: preconditioned TFP HMC/NUTS for posterior sampling and a MAP runner backed by `tf.keras.optimizers` or `tfp.optimizer`, then gate it with a 4-chain HMC run on a Linear Gaussian finite-N particle-filter likelihood and a MAP convergence run on the exact Kalman log-likelihood from Phase 03.

## 2. What Gets Built

Files and modules owned by this phase:

- `jpml_tf/core/registries.py` - extends the shared registry module with `SAMPLER_REGISTRY`, `register_sampler`, `build_sampler`, and per-algorithm schema lookup.
- `jpml_tf/samplers/__init__.py` - imports concrete sampler modules so their registration decorators run.
- `jpml_tf/samplers/base.py` - generic sampler protocols, state containers, run results, and objective wrappers.
- `jpml_tf/samplers/preconditioning.py` - diagonal mass-matrix containers, flatten/unflatten helpers for parameter trees, and posterior-variance-to-mass-matrix conversion.
- `jpml_tf/samplers/tfp_mcmc.py` - registered TFP-backed HMC and NUTS algorithms using `HamiltonianMonteCarlo`, `NoUTurnSampler`, `DualAveragingStepSizeAdaptation`, `SimpleStepSizeAdaptation`, and `TransformedTransitionKernel` where constrained-state sampling is requested.
- `jpml_tf/samplers/map.py` - registered MAP runner using `tf.keras.optimizers` and optionally `tfp.optimizer.lbfgs_minimize` for deterministic local refinement.
- `jpml_tf/optim/clipping.py` - global and per-parameter clipping helpers built outside sampler loops and shared by MAP or any explicitly clipped experimental sampler.
- `jpml_tf/experiments/logdensity.py` - host-side builders that combine parameter transforms, priors, trainable-flag-derived frozen-parameter merges, and either particle or Kalman likelihoods into sampler-facing objectives.
- `tests/test_06_hmc_and_map.py` - the single gate test file for this phase.

Generic sampler registry and protocols:

```python
# jpml_tf/core/registries.py
SAMPLER_REGISTRY: dict[str, SamplerBuilder] = {}

def register_sampler(name: str, *, schema: type[BaseModel]) -> Callable[[SamplerBuilder], SamplerBuilder]: ...
def build_sampler(name: str, params: dict[str, Any]) -> SamplerAlgorithm: ...
def sampler_schema(name: str) -> type[BaseModel]: ...
```

```python
# jpml_tf/samplers/base.py
class LogDensityResult(NamedTuple):
    logdensity: tf.Tensor                   # shape ()
    aux: Mapping[str, tf.Tensor]

class SamplerState(NamedTuple):
    position: Params
    kernel_state: Any
    step: tf.Tensor                         # shape (), int32
    aux: Mapping[str, tf.Tensor]

class SamplerInfo(NamedTuple):
    accept_prob: tf.Tensor | None
    logdensity: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]

class SamplerRunResult(NamedTuple):
    samples: Params
    summary: Mapping[str, tf.Tensor]
    diagnostics: Mapping[str, tf.Tensor]

class MAPState(NamedTuple):
    position: Params
    optimizer_state: Any
    objective: tf.Tensor                    # shape ()
    grad_norm: tf.Tensor                    # shape ()
    step: tf.Tensor                         # shape (), int32

class SamplerAlgorithm(Protocol):
    family: Literal["mcmc", "optimizer"]

    def run(
        self,
        init_position: Params,
        objective_fn: Callable[[Params], LogDensityResult],
        seed: tf.Tensor,                    # shape (2,), int32
    ) -> SamplerRunResult: ...
```

Objective-builder and preconditioning utilities:

```python
# jpml_tf/experiments/logdensity.py
def build_particle_logdensity(
    *,
    model: StateSpaceModel,
    schema: ParameterSchema,
    frozen_params: Params,                 # derived from ParameterSpec.trainable == False
    likelihood: DifferentiableLikelihoodAlgorithm,
    observations: tf.Tensor,
    seed: tf.Tensor,
    policy: GradientResamplingPolicy,
) -> Callable[[Params], LogDensityResult]: ...

def build_kalman_logdensity(
    *,
    model: StateSpaceModel,
    schema: ParameterSchema,
    frozen_params: Params,                 # derived from ParameterSpec.trainable == False
    observations: tf.Tensor,
) -> Callable[[Params], LogDensityResult]: ...

# jpml_tf/samplers/preconditioning.py
class DiagonalMassMatrix(NamedTuple):
    diag: tf.Tensor                         # shape (event_size,)
    inv_diag: tf.Tensor                     # shape (event_size,)
    event_size: tf.Tensor                   # shape (), int32

def flatten_learnable_params(params: Params) -> tuple[tf.Tensor, TreeDef]: ...
def unflatten_learnable_params(vector: tf.Tensor, treedef: TreeDef) -> Params: ...
def diagonal_mass_matrix_from_variance(
    variance_tree: Params,
    *,
    floor: float = 1e-6,
) -> DiagonalMassMatrix: ...
```

Clipping hooks for optimizer-style paths:

```python
# jpml_tf/optim/clipping.py
class ClipSpec(BaseModel):
    mode: Literal["none", "global_norm", "per_parameter_norm", "value", "nan_to_zero"]
    global_norm: float | None = None
    per_parameter: dict[str, float] = {}
    value: float | None = None
    apply_to: Literal["grad", "momentum", "both"] = "grad"

class ClipMetrics(NamedTuple):
    global_norm_before: tf.Tensor
    global_norm_after: tf.Tensor
    per_parameter_norms_before: Mapping[str, tf.Tensor]
    per_parameter_norms_after: Mapping[str, tf.Tensor]

def clip_gradients(
    grads: Params,
    params: Params,
    schema: ParameterSchema,
    spec: ClipSpec,
) -> tuple[Params, ClipMetrics]: ...
```

Concrete registered sampler instances:

```python
@register_sampler("tfp_hmc", schema=TFPHMCParams)
def build_tfp_hmc(params: TFPHMCParams) -> SamplerAlgorithm: ...

@register_sampler("tfp_nuts", schema=TFPNUTSParams)
def build_tfp_nuts(params: TFPNUTSParams) -> SamplerAlgorithm: ...

@register_sampler("adam_map", schema=MAPOptimizerParams)
def build_adam_map(params: MAPOptimizerParams) -> SamplerAlgorithm: ...
```

Concrete kernel builders:

```python
# jpml_tf/samplers/tfp_mcmc.py
def build_hmc_kernel(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    params: TFPHMCParams,
    mass_matrix: DiagonalMassMatrix,
) -> tfp.mcmc.TransitionKernel: ...

def build_nuts_kernel(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    params: TFPNUTSParams,
    mass_matrix: DiagonalMassMatrix,
) -> tfp.mcmc.TransitionKernel: ...

def run_mcmc_chains(
    kernel: tfp.mcmc.TransitionKernel,
    init_state: tf.Tensor,
    seed: tf.Tensor,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
) -> SamplerRunResult: ...

# jpml_tf/samplers/map.py
def run_map(
    init_position: Params,
    objective_fn: Callable[[Params], LogDensityResult],
    optimizer: tf.keras.optimizers.Optimizer,
    num_steps: int,
    clip_spec: ClipSpec | None,
    seed: tf.Tensor,
) -> SamplerRunResult: ...
```

Expected hard-gate tensor shapes:

```text
unconstrained_position:        learnable parameter tree or flattened vector, float64
frozen_params:                 frozen parameter tree, float64
particle_logdensity:           (), float64
kalman_logdensity:             (), float64
logdensity_grad:               same tree or vector shape as unconstrained_position
mass_matrix_diag:              (event_size,), float64
hmc_samples:                   (4, num_kept, event_size) or tree with leading axes (4, num_kept, ...)
nuts_samples:                  short-chain tensor/tree with chain and sample axes
map_position:                  same tree structure as unconstrained_position
clip_metrics:                  scalar/global metrics plus per-parameter maps
```

The HMC hard gate uses a finite-N particle-filter likelihood on Linear Gaussian data, not the exact Kalman likelihood, because the sampler layer must be proven against the likelihood family it will actually see in particle-filter inference. The MAP hard gate uses the exact Kalman objective from Phase 03 so optimizer convergence and clipping are validated on a deterministic reference target.

## 3. What Gets Tested and Acceptance Criteria

Gate test file: `tests/test_06_hmc_and_map.py`

- `test_sampler_registry_builds_hmc_nuts_and_map_instances(result_sink)` imports `jpml_tf.samplers`, asserts `SAMPLER_REGISTRY` contains `tfp_hmc`, `tfp_nuts`, and `adam_map`, validates each params payload through its schema, and builds all three instances through the registry only.
- `test_particle_and_kalman_logdensity_builders_are_finite_float64(result_sink)` builds a Linear Gaussian particle-logdensity objective for HMC and an exact Kalman logdensity objective for MAP, then asserts both return scalar `float64` values and finite gradients with respect to the selected learnable parameter.
- `test_diagonal_mass_matrix_from_linear_gaussian_posterior_variance(result_sink)` computes a posterior-variance estimate for the Linear Gaussian gate model, converts it to a `DiagonalMassMatrix`, and asserts positive finite diagonal entries, correct flattened event size, and floor-regularized inverse masses.
- `test_tfp_hmc_four_chain_linear_gaussian_particle_likelihood_mixes(result_sink)` runs 4 HMC chains on the finite-N Linear Gaussian particle likelihood with fixed stateless seeds, diagonal mass matrix preconditioning, and dual-averaging adaptation. The gate asserts `max_rhat <= 1.05`, `min_effective_sample_size >= max(20, 0.1 * total_kept_draws)`, finite acceptance diagnostics, and no nonfinite logdensity evaluations after warmup.
- `test_tfp_nuts_short_chain_executes_on_same_objective(result_sink)` runs a short NUTS chain on the same objective, asserts finite samples, finite adapted step size, bounded tree depth, and correct summary shapes. NUTS is registered and health-checked here even though the strict convergence gate is carried by HMC.
- `test_map_runner_converges_on_kalman_loglik_and_returns_clip_metrics(result_sink)` runs the registered MAP runner on the exact Kalman objective from Phase 03 with a clip policy built outside the loop, asserts the negative log-posterior decreases, asserts the final constrained parameter is closer to the known truth than the initialization, and asserts nonempty `ClipMetrics`.
- `test_global_and_per_parameter_clipping_live_outside_the_map_loop(result_sink)` constructs `ClipSpec` before the optimization step, applies one update, and asserts the post-clip global norm respects the configured threshold while at least one selected parameter leaf also respects its per-parameter threshold.
- `test_frozen_parameters_are_not_mutated_by_hmc_or_map(result_sink)` snapshots frozen parameter leaves before a short HMC run and a short MAP run, then asserts they remain unchanged. The sampler layer may update learnable unconstrained position only; it may not mutate merged or model-owned parameters.

Gate-pass condition:

- `pytest tests/test_06_hmc_and_map.py` passes.
- Every test saves a JSON result through the Phase 01 `result_sink` fixture.
- The HMC hard gate uses 4 chains on a Linear Gaussian finite-N particle-filter likelihood and meets the R-hat and ESS thresholds above.
- The MAP hard gate converges on the exact Kalman objective from Phase 03 and returns clipping diagnostics from externally configured policies.
- HMC and NUTS are built through `SAMPLER_REGISTRY` and use TFP kernels rather than custom leapfrog code.
- The diagonal mass matrix is positive, finite, and derived from posterior-variance information instead of an ad hoc constant scale.
- No sampler or optimizer writes parameter values onto the model object or mutates frozen parameters.

No Kalman-only minimal alternative is accepted for this phase. The sampler layer must be proven on a particle-filter likelihood for MCMC and on the exact Kalman likelihood for deterministic MAP optimization.

## 4. What the Reader Needs to Understand

### Key Concepts

The sampler layer consumes objectives, not models or filters directly. `build_particle_logdensity` and `build_kalman_logdensity` are the only bridges from parameter schema plus likelihood plus priors into sampler-facing callables. This prevents the duplicated parameter plumbing diagnosed in the current TF code.

The phase deliberately uses two objective classes. HMC is gated on a finite-N particle-filter likelihood because the whole point is to validate sampling on the noisy-but-deterministic objective family that downstream particle inference will expose. MAP is gated on the exact Kalman objective because optimizer correctness is easier to isolate on a deterministic reference target.

Preconditioning is part of the sampler contract, not a tuning afterthought. A diagonal mass matrix estimated from posterior variance changes both sampler stability and the meaning of step size. The implementation therefore gives mass-matrix construction its own module and gate rather than burying it inside one HMC builder.

Gradient clipping is configuration, not hidden control flow. For MAP it is built outside the loop and returned as auditable metrics. Standard HMC and NUTS are left unclipped by default because clipping changes the transition kernel; if clipped MCMC is ever added, it must be an explicitly named registered algorithm.

### Invariants Established

- Sampler construction is registry-backed; concrete algorithms enter through `SAMPLER_REGISTRY` only.
- HMC and NUTS use TFP kernels; MAP uses `tf.keras.optimizers` or `tfp.optimizer`.
- Parameter transforms, priors, and frozen-parameter merges live in `experiments/logdensity.py`, not inside sampler implementations, and the trainable versus frozen split comes from explicit schema flags rather than config buckets.
- The HMC hard gate runs 4 chains against a finite-N particle-filter likelihood and reports convergence summaries on kept draws.
- The MAP hard gate runs on the exact Kalman objective from Phase 03 and returns clip metrics from externally configured policies.
- Diagonal preconditioning is explicit, finite, and positive.
- Frozen parameters remain unchanged across both sampler families.
- Global `float64` remains the only critical-path dtype.

### Tricky Bits and Rationale

The particle-filter objective used by HMC must be deterministic across repeated evaluations at one parameter value. That means fixed stateless seeds or a common-random-numbers policy. Re-sampling new particle randomness at every leapfrog gradient evaluation destroys the Hamiltonian interpretation and makes R-hat or ESS results meaningless.

The resampling policy for the HMC objective must remain branch-stable. Conditional ESS switching is forbidden because it recreates the `tf.cond` discontinuity already documented in `extra/HMC_DPF_NOTES.md`. The recommended gate objective uses an always-resample gradient policy; if the chosen resampler still stops the resampling gradient, that choice must be explicit in the policy object rather than hidden in the sampler.

Posterior variance and mass matrix are not interchangeable without care. A mass matrix scales momentum; a posterior covariance describes position geometry. The implementation therefore stores both `diag` and `inv_diag`, applies a floor before inversion, and keeps flatten/unflatten logic near the preconditioner so chain state and parameter tree stay aligned.

MAP clipping needs clear boundaries. The clipping transform is created before the loop, applied to raw gradients or optimizer momentum according to `ClipSpec`, and recorded as metrics. Thresholds must not be embedded in optimizer step bodies, and test tolerances must live in test code rather than scenario configs.

`TransformedTransitionKernel` is supported for constrained-state sampling when a TFP-facing constrained parameterization is required, but the default gate path still samples on the unconstrained parameter tree built by Phase 02. That keeps one transform story across HMC, NUTS, and MAP while leaving room for a TFP-native constrained path when it is genuinely useful.

### Alternatives Considered

Gating HMC only on the exact Kalman objective is rejected because it would validate the easiest possible geometry and leave particle-filter objective handling unproven.

Using a dense mass matrix in the first sampler phase is rejected because the parameter spaces in current scope are still low to medium dimensional and the diagonal preconditioner already retires the main scaling bug without adding fragile covariance adaptation.

Letting each sampler rebuild parameter transforms and priors is rejected because it recreates the current TF split between runner, wrapper, and filter parameter logic.

Embedding clipping thresholds inside optimizer step code is rejected because the user requirement is explicit external clipping control.

### Locked Design Decisions Realized

- The family-generic registry rule is realized for samplers: the interface is a generic sampler protocol, while `tfp_hmc`, `tfp_nuts`, and `adam_map` are registration data.
- The hard gate in `00_overview.md` is realized exactly: 4-chain HMC convergence is checked on a Linear Gaussian particle-filter likelihood, not a toy target.
- The exact Kalman differentiable likelihood from Phase 03 is reused directly for the MAP gate rather than reimplemented.
- Explicit parameter trees remain the only dynamic parameter representation. No sampler writes constrained values into model attributes.
- The Phase 04 rule against conditional resampling in gradient paths is preserved for sampler-facing particle objectives.
- Global `float64` and the uniform JSON result sink remain mandatory.

### JIT Boundary Decisions

Objective functions built in `experiments/logdensity.py` should be wrapped in `@tf.function(reduce_retracing=True)` and may use `jit_compile=True` when the inner likelihood path supports it. The sampler layer treats those objective callables as compiled numerical kernels and does not add file I/O or host conversions around them.

One HMC or NUTS transition step and the MAP train step should each be `tf.function` compatible. TFP's internal leapfrog and tree-building control flow already uses `tf.while_loop`; the sampler module owns only the host-side assembly of kernels, adaptation wrappers, and summaries.

Registry lookup, Pydantic validation, prior schema loading, result saving, artifact writing, and chain-summary JSON serialization stay outside `tf.function`. R-hat and ESS summary extraction can remain host-side in this phase; Phase 07 turns those diagnostics into first-class modules.

## 5. Dependencies

This phase depends on:

- Phase 01 for package scaffold, scenario fixtures, and the JSON result sink.
- Phase 02 for `ParameterSchema`, partition/constrain/merge, prior evaluation, and explicit parameter trees.
- Phase 03 for the exact Kalman differentiable likelihood used by the MAP gate and for exact Linear Gaussian posterior-variance reference information.
- Phase 04 for gradient-path resampling policies and any differentiable or stop-gradient resampling method selected by the particle objective.
- Phase 05 for the particle-filter and LEDH differentiable likelihood interfaces consumed by HMC/NUTS objectives.

Later dependencies:

- Phase 07 diagnostics consume chain outputs, step-size traces, acceptance traces, and MAP iterates generated here.
- Phase 08 migration ports the old HMC and MAP config families onto this registry-backed sampler layer and verifies their tests use the uniform result sink.
