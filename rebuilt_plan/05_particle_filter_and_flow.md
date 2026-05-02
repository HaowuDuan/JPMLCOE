# Phase 05: Particle Filter Baseline and LEDH Differentiable Flow Likelihood

## 1. Goal

Build both particle-filter families promised by the overview: a registered baseline bootstrap particle filter for forward-only filtering, and a registered localized Exact Daum-Huang (LEDH) differentiable likelihood path with 29-step Jacobian accumulation on the SV2D hard fixture, including state-dependent observation covariance `R_i = f(x_i)` and gradient-safe always-resample or smooth resampling from Phase 04.

## 2. What Gets Built

Files and modules owned by this phase:

- `jpml_tf/filters/base.py` - extends the filter contracts with a forward particle-filter protocol, a differentiable likelihood protocol, and a generic multi-step flow protocol. Interfaces stay family-generic; concrete names remain registry keys.
- `jpml_tf/filters/common.py` - adds particle carries, per-time-step outputs, flow-step outputs, log-space normalization helpers, and a checkpoint wrapper for the multi-step flow body when memory pressure requires recomputation.
- `jpml_tf/filters/particle.py` - the registered bootstrap particle filter used on the non-gradient path and consuming `ResamplingPolicy` from Phase 04.
- `jpml_tf/filters/ledh.py` - the registered LEDH flow implementation used on the gradient path, including per-particle local linearization, multi-step Jacobian accumulation, and differentiable log-likelihood assembly.
- `jpml_tf/models/stochastic_volatility_2d.py` - concrete hard-case model if not already available from Phase 02. It must implement `StateSpaceModel` plus `HasStateDependentObsCov`, and it must expose batched Jacobian helpers or allow autodiff fallback.
- `tests/test_05_particle_filter_and_flow.py` - the single gate test file for this phase.

Generic protocols and registry-backed contracts:

```python
# jpml_tf/filters/base.py
class ParticleFilterAlgorithm(Protocol):
    family: Literal["particle", "flow_particle"]
    input_signature: tuple[tf.TensorSpec, ...]

    def apply(
        self,
        model: StateSpaceModel,
        params: Params,
        observations: tf.Tensor,            # shape (T, obs_dim)
        seed: tf.Tensor,                    # shape (2,), int32
        policy: ResamplingPolicy,
    ) -> FilterRunResult: ...

class DifferentiableLikelihoodAlgorithm(Protocol):
    family: Literal["particle", "flow_particle"]
    input_signature: tuple[tf.TensorSpec, ...]

    def loglik(
        self,
        model: StateSpaceModel,
        params: Params,
        observations: tf.Tensor,            # shape (T, obs_dim)
        seed: tf.Tensor,                    # shape (2,), int32
        policy: GradientResamplingPolicy,
    ) -> LikelihoodResult: ...

class MultiStepFlowFilter(Protocol):
    n_flow_steps: int

    def flow_step(
        self,
        carry: "FlowCarry",
        substep: tf.Tensor,                 # shape (), int32
        step_seed: tf.Tensor,               # shape (2,), int32
    ) -> tuple["FlowCarry", "FlowStepResult"]: ...

    def apply_flow(
        self,
        model: StateSpaceModel,
        params: Params,
        particles: tf.Tensor,               # shape (N, state_dim)
        particle_covs: tf.Tensor,           # shape (N, state_dim, state_dim)
        observation: tf.Tensor,             # shape (obs_dim,)
        t: tf.Tensor,                       # shape (), int32
        seed: tf.Tensor,                    # shape (2,), int32
    ) -> "FlowApplyResult": ...
```

Shared particle and flow result containers:

```python
# jpml_tf/filters/common.py
class ParticleFilterCarry(NamedTuple):
    particles: tf.Tensor                    # shape (N, state_dim)
    log_weights: tf.Tensor                  # shape (N,)
    loglik: tf.Tensor                       # shape ()
    seed: tf.Tensor                         # shape (2,), int32
    aux: Mapping[str, tf.Tensor]

class ParticleStepOutput(NamedTuple):
    particles: tf.Tensor                    # shape (N, state_dim)
    log_weights: tf.Tensor                  # shape (N,)
    ess: tf.Tensor                          # shape ()
    loglik_increment: tf.Tensor             # shape ()
    resampled: tf.Tensor                    # shape (), bool

class FlowLinearization(NamedTuple):
    jacobians: tf.Tensor                    # shape (N, obs_dim, state_dim)
    obs_covs: tf.Tensor                     # shape (N, obs_dim, obs_dim)
    drift_mats: tf.Tensor                   # shape (N, state_dim, state_dim)
    drift_vecs: tf.Tensor                   # shape (N, state_dim)

class FlowCarry(NamedTuple):
    particles: tf.Tensor                    # shape (N, state_dim)
    particle_covs: tf.Tensor                # shape (N, state_dim, state_dim)
    logdet: tf.Tensor                       # shape (N,)
    lambda_value: tf.Tensor                 # shape ()
    diagnostics: Mapping[str, tf.Tensor]

class FlowStepResult(NamedTuple):
    particles: tf.Tensor                    # shape (N, state_dim)
    particle_covs: tf.Tensor                # shape (N, state_dim, state_dim)
    step_logdet: tf.Tensor                  # shape (N,)
    obs_covs: tf.Tensor                     # shape (N, obs_dim, obs_dim)
    any_nonfinite: tf.Tensor                # shape (), bool

class FlowApplyResult(NamedTuple):
    particles: tf.Tensor                    # shape (N, state_dim)
    particle_covs: tf.Tensor                # shape (N, state_dim, state_dim)
    accumulated_logdet: tf.Tensor           # shape (N,)
    flow_logdet_per_step: tf.Tensor         # shape (n_flow_steps, N)
    obs_covs_per_step: tf.Tensor            # shape (n_flow_steps, N, obs_dim, obs_dim)
    diagnostics: Mapping[str, tf.Tensor]

def maybe_checkpoint_flow_body(
    fn: Callable[..., tuple[FlowCarry, FlowStepResult]],
    *,
    enabled: bool,
) -> Callable[..., tuple[FlowCarry, FlowStepResult]]: ...
```

Concrete registered filter instances:

```python
@register_filter("bootstrap_particle", schema=BootstrapParticleParams)
def build_bootstrap_particle(params: BootstrapParticleParams) -> ParticleFilterAlgorithm: ...

@register_filter("ledh_ot", schema=LEDHFlowParams)
def build_ledh_flow(params: LEDHFlowParams) -> DifferentiableLikelihoodAlgorithm: ...
```

Core functions:

```python
# jpml_tf/filters/particle.py
def bootstrap_particle_filter_apply(
    model: StateSpaceModel,
    params: Params,
    observations: tf.Tensor,
    seed: tf.Tensor,
    policy: ResamplingPolicy,
) -> FilterRunResult: ...

# jpml_tf/filters/ledh.py
def compute_ledh_linearization(
    model: HasStateDependentObsCov,
    params: Params,
    particles: tf.Tensor,                   # shape (N, state_dim)
    particle_covs: tf.Tensor,               # shape (N, state_dim, state_dim)
    observation: tf.Tensor,
    lambda_value: tf.Tensor,
    t: tf.Tensor,
) -> FlowLinearization: ...

def ledh_apply_flow(
    model: StateSpaceModel,
    params: Params,
    particles: tf.Tensor,
    particle_covs: tf.Tensor,
    observation: tf.Tensor,
    t: tf.Tensor,
    seed: tf.Tensor,
    flow_params: LEDHFlowParams,
) -> FlowApplyResult: ...

def ledh_loglik_apply(
    model: StateSpaceModel,
    params: Params,
    observations: tf.Tensor,
    seed: tf.Tensor,
    policy: GradientResamplingPolicy,
) -> LikelihoodResult: ...
```

The LEDH implementation must satisfy the Phase 02 capability split:

```python
class HasStateDependentObsCov(Protocol):
    def observation_noise_cov_batch(
        self,
        states: tf.Tensor,                  # shape (N, state_dim)
        params: Params,
    ) -> tf.Tensor: ...                    # shape (N, obs_dim, obs_dim)
```

This is not optional for the hard gate. The SV2D LEDH path must consume `R_i = f(x_i)` at every flow step. Treating `R` as one constant matrix for the full particle cloud is rejected.

Expected hard-gate tensor shapes:

```text
T:                            fixed small gate length, e.g. 8 to 12
N:                            fixed particle count, e.g. 32 to 128
state_dim:                    SV2D state dimension
obs_dim:                      SV2D observation dimension
observations:                 (T, obs_dim), float64
particles:                    (N, state_dim), float64
particle_covs:                (N, state_dim, state_dim), float64
log_weights:                  (N,), float64
flow_step_count:              29 exactly on the hard gate
flow_logdet_per_step:         (T, 29, N), float64
obs_covs_per_step:            (T, 29, N, obs_dim, obs_dim), float64
accumulated_logdet:           (T, N), float64
likelihood_loglik:            (), float64
transport_jacobian:           (state_dim, state_dim) for the selected particle/check
autodiff_loglik_grad:         selected learnable leaf or scalar tensor, float64
```

The Jacobian accumulation must be log-space and stepwise normalized. Multiplying raw determinants across 29 steps is rejected because it recreates the overflow and underflow failure documented in `extra/PFPF_Numerical_Instability_Fixes.md`.

## 3. What Gets Tested and Acceptance Criteria

Gate test file: `tests/test_05_particle_filter_and_flow.py`

- `test_filter_registry_builds_bootstrap_and_ledh_instances(result_sink)` imports `jpml_tf.filters`, asserts `FILTER_REGISTRY` contains both `bootstrap_particle` and `ledh_ot`, validates each params payload through its schema, and builds both algorithms from the registry only.
- `test_bootstrap_particle_filter_contract_and_phase04_resampling(result_sink)` runs `bootstrap_particle_filter_apply` on a small nonlinear fixture with a conditional policy from Phase 04 and asserts scalar `float64` log-likelihood, explicit particle/log-weight carry shapes, valid ESS values, and resampling behavior that matches the registered Phase 04 policy rather than a private branch.
- `test_ledh_sv2d_29_step_state_dependent_r_flow_runs_float64(result_sink)` runs the SV2D hard fixture with `n_flow_steps == 29`, asserts `flow_logdet_per_step` shape `(T, 29, N)`, asserts `obs_covs_per_step` shape `(T, 29, N, obs_dim, obs_dim)`, and asserts that at at least one timestep the particle-specific observation covariance differs across particles by more than `1e-8`. Every returned flow tensor must be `float64`.
- `test_ledh_tffunction_matches_eager(result_sink)` wraps `ledh_loglik_apply` in `@tf.function(reduce_retracing=True)` and asserts eager and traced total log-likelihood, accumulated logdet summaries, and selected diagnostics agree within absolute tolerance `1e-9` and relative tolerance `1e-6` at `float64`.
- `test_ledh_forward_jit_compile_smoke(result_sink)` wraps forward-only `ledh_apply_flow` and `bootstrap_particle_filter_apply` in `@tf.function(jit_compile=True)` and asserts execution succeeds on the gate fixture. The requirement here is XLA-lowerable forward computation; full backward-through-OT XLA is not the hard gate.
- `test_ledh_multistep_jacobian_matches_central_finite_difference(result_sink)` selects one timestep, one particle, and one learnable scalar or one selected state-coordinate perturbation, computes the transported-state Jacobian through all 29 flow steps with `tf.GradientTape().jacobian`, compares it with a centered finite-difference Jacobian, and gates maximum absolute error below `1e-4` and relative error below `5e-3`.
- `test_ledh_loglik_gradient_is_finite_nonzero_and_stepwise_finite(result_sink)` computes `tf.GradientTape` gradient of the scalar LEDH log-likelihood with respect to at least one learnable SV2D parameter, asserts every checked gradient leaf is finite, asserts at least one selected scalar gradient magnitude exceeds `1e-8`, and asserts `any_nonfinite` stays false at every flow step and timestep.
- `test_ledh_gradient_policy_rejects_conditional_resampling(result_sink)` attempts to build or run the LEDH differentiable path with `GradientResamplingPolicy(mode="conditional", ...)` and asserts validation or capability failure. Accepted modes for the gradient path are `always`, `smooth`, or `never`; conditional ESS-switching is explicitly forbidden here.

Gate-pass condition:

- `pytest tests/test_05_particle_filter_and_flow.py` passes.
- Every test saves a JSON result through the Phase 01 `result_sink` fixture.
- The hard gate is the SV2D LEDH fixture with exactly 29 flow steps in `float64`.
- The LEDH implementation consumes state-dependent `R_i = f(x_i)`; constant `R` substitution is not accepted.
- Autodiff Jacobian and finite-difference Jacobian agree through the full 29-step flow path within test-code tolerance.
- The LEDH log-likelihood gradient is finite and non-zero on the hard fixture.
- No NaN or Inf appears at any flow step, Jacobian step, or accumulated logdet step on the hard gate.
- The differentiable path uses `GradientResamplingPolicy` from Phase 04 and never conditional ESS resampling.

No bootstrap-only minimal alternative is accepted for this phase. The baseline particle filter is required, but the abstraction is not considered complete until the LEDH hard gate passes.

## 4. What the Reader Needs to Understand

### Key Concepts

This phase is intentionally two-stage. The baseline bootstrap PF proves the generic particle-filter result contract, scan structure, seed discipline, and Phase 04 resampling integration on the non-gradient path. The LEDH path then applies full design pressure by differentiating through a multi-step particle flow on the current hardest particle-based fixture.

The differentiable likelihood path is not just "particle filtering plus gradients." The Jacobian terms are part of the scalar likelihood itself. If `flow_logdet_per_step` is treated as diagnostics only, the sampler-facing objective in Phase 06 will be wrong even if the state estimates look plausible.

LEDH is local, not global. Each particle carries its own covariance approximation and its own observation covariance `R_i`. That is why Phase 02 split `HasStateDependentObsCov` out as an opt-in capability. The hard fixture is exactly the case that invalidates a one-size-fits-all observation covariance.

The explicit seed rule remains non-negotiable. Every stochastic transition, resampling call, and finite-difference comparison uses an explicit stateless `[seed, step]` key. The finite-difference gate must hold all random choices fixed across `theta + h` and `theta - h`; otherwise the comparison mostly measures Monte Carlo noise.

### Invariants Established

- Forward-only particle filtering and differentiable particle-flow likelihoods share the same registry surface and generic result containers.
- The baseline PF uses `ResamplingPolicy` from Phase 04 and does not implement private resampling branches.
- The LEDH differentiable path uses `GradientResamplingPolicy` from Phase 04 and rejects conditional ESS switching.
- The SV2D hard gate uses `n_flow_steps == 29` and runs in global `float64`.
- LEDH consumes particle-specific observation covariances `R_i = f(x_i)` through the model capability, not a constant `R`.
- Per-step and accumulated Jacobian terms are explicit fixed-shape tensors.
- No model attributes are mutated; parameters are passed in as explicit trees built upstream from Phase 02.
- The Jacobian accumulation path is `tf.function` compatible and can be wrapped with recomputation if memory is the bottleneck.

### Tricky Bits and Rationale

The main numerical trap is where to regularize. The existing TF notes show that regularizing the innovation covariance `S` directly changes the LEDH equations, while regularizing the particle covariance `P` preserves the intended math. The rebuild therefore regularizes `P` before forming local flow matrices and uses Cholesky-based solves instead of explicit inverses.

The Jacobian accumulation must be log-space and stepwise normalized. The old TF code multiplied determinant magnitudes directly and overflowed or underflowed across long flows. The new plan accumulates `log|det(M_i)|` per step and allows max-subtraction normalization after each lambda step. Weight clipping after max-normalization is rejected; it changes the particle ranking rather than merely stabilizing numerics.

`R_i` is state-dependent on the hard gate, so the implementation cannot precompute one `R_inv` for the full cloud. It may still cache per-particle Cholesky factors within a given substep, and it should compute them once per particle per lambda step rather than repeatedly inside inner algebra.

Memory can become the limiting factor before flops do. A 29-step flow over `T` timesteps, `N` particles, and per-particle covariance matrices creates a large autodiff tape. The design therefore locks a recomputation hook in `filters/common.py` via `tf.recompute_grad`. The default gate can run without checkpointing on the reduced fixture, but the implementation must not paint itself into a corner where checkpointing requires interface surgery later.

Diagnostics stay inside return values, not host side effects. The filter must not write artifacts, print progress, or call `.numpy()` inside `tf.function`. If the gate needs particle histories or per-step logdet traces, those are tensors returned in `LikelihoodResult.aux` and saved through the Phase 01 sink on the host side.

### Alternatives Considered

Gating only the bootstrap PF is rejected because it would validate the easy contract while leaving the multi-step flow, Jacobian accumulation, and state-dependent covariance path unproven.

Allowing conditional ESS resampling in the differentiable path is rejected because it recreates the `tf.cond` likelihood discontinuity already diagnosed in `extra/HMC_DPF_NOTES.md`. Phase 06 samplers should inherit a smooth or at least branch-stable objective, not rediscover that failure mode.

Treating `R` as constant inside LEDH is rejected because it passes toy tests and fails the concrete hard fixture the rebuild is meant to support.

Accumulating raw determinant products is rejected because it reproduces the overflow path already observed in the current TF code. The rebuild must retire that failure mode at the plan level, not leave it as an optional optimization.

### Locked Design Decisions Realized

- The hard-case-first rule from `00_overview.md` is realized here: the first differentiable particle-filter gate is SV2D plus LEDH plus a 29-step Jacobian path, not a toy one-step flow.
- Family-generic registries are preserved. The universal interface is "particle filter" or "flow particle filter"; `bootstrap_particle` and `ledh_ot` are concrete registration data.
- Conditional resampling is forbidden on gradient paths. The differentiable path consumes `always`, `smooth`, or `never` from the Phase 04 policy object.
- The Phase 04 OT backward path is reused rather than bypassed with a direct dense inverse.
- The Phase 02 capability split is realized: base `StateSpaceModel` stays minimal, while state-dependent observation covariance is opt-in and exercised by the hard gate.
- Global `float64` is preserved. There is no per-experiment dtype branch inside the compiled flow path.
- Explicit parameter trees and pure model calls are preserved. The flow filter never writes learned values onto the model instance.

### JIT Boundary Decisions

`bootstrap_particle_filter_apply` should run under `@tf.function(jit_compile=True, reduce_retracing=True)` with time recursion expressed through `tf.while_loop` or `tf.scan`. The resampling method and policy mode are static closure data; observations, particles, weights, and seeds are traced tensors.

`ledh_loglik_apply` runs under `@tf.function(reduce_retracing=True)` with two nested compiled loops: one over time, one over lambda steps. The LEDH forward flow body should also be `jit_compile=True` compatible so the implementation retains the option to XLA-compile the forward path even when backward-through-transport remains plain `tf.function`.

If checkpointing is enabled, it wraps the per-substep flow body only. Pydantic validation, registry lookup, scenario loading, result saving, and artifact writing stay outside `tf.function`. No file I/O, `.numpy()`, or progress reporting is allowed inside the compiled particle or flow kernels.

## 5. Dependencies

This phase depends on:

- Phase 01 for package scaffold, scenario fixtures, and the JSON result sink.
- Phase 02 for `ParameterSchema`, explicit parameter trees, `StateSpaceModel`, autodiff Jacobian fallback, and `HasStateDependentObsCov`.
- Phase 03 for the exact-filter numerical discipline: Cholesky solves, no matrix inversion, and explicit differentiable likelihood entry points.
- Phase 04 for `ResamplingPolicy`, `GradientResamplingPolicy`, ESS logic, and the regularized OT backward path.

Later dependencies:

- Phase 06 consumes both outputs of this phase: a finite-N particle-filter likelihood for HMC and the LEDH differentiable likelihood interface for sampler integration.
- Phase 07 diagnostics use the particle histories, per-step logdet traces, and seed discipline established here.
- Phase 08 migration maps the existing LEDH/BPF config families into the new registry-backed scenario format and ports the corresponding tests onto the uniform result sink.
