# Phase 04: Resampling Policy and Core Stochastic Resamplers

## 1. Goal

Build the first-class resampling policy and the systematic, stratified, multinomial, and entropy-regularized OT resampling kernels in TensorFlow that later non-gradient and gradient-facing particle filters use without hardcoded ESS branches.

## 2. What Gets Built

Files and modules owned by this phase:

- `jpml_tf/resamplers/__init__.py` - imports concrete resampler modules so their registry decorators run.
- `jpml_tf/core/registries.py` - extends with `RESAMPLER_REGISTRY` and `register_resampler` if not already created.
- `jpml_tf/resamplers/policies.py` - `ResamplingPolicy`, `GradientResamplingPolicy`, ESS calculation, policy decision helpers, dispatch, and result containers.
- `jpml_tf/resamplers/systematic.py` - systematic resampling kernel.
- `jpml_tf/resamplers/stratified.py` - stratified resampling kernel.
- `jpml_tf/resamplers/multinomial.py` - multinomial resampling kernel.
- `jpml_tf/resamplers/ot_entropy.py` - entropy-regularized OT (Sinkhorn) resampler with regularized backward, the registered differentiable transport path required by gradient-facing filters.
- `tests/test_04_resampling.py` - the single gate test file for this phase.

Policy classes and result types:

```python
# jpml_tf/core/registries.py
RESAMPLER_REGISTRY: dict[str, ResamplerBuilder] = {}

def register_resampler(name: str, *, schema: type[BaseModel]) -> Callable[[ResamplerBuilder], ResamplerBuilder]: ...

# jpml_tf/resamplers/policies.py
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
    def resample(self, particles: PyTree, log_weights: tf.Tensor, seed: tf.Tensor, params: dict[str, Any]) -> ResampleOutput: ...

class ResampleDecision(NamedTuple):
    should_resample: tf.Tensor      # shape (), bool
    ess: tf.Tensor                  # shape ()

class ResampleOutput(NamedTuple):
    particles: PyTree               # each leaf shape (N, ...)
    log_weights: tf.Tensor          # shape (N,)
    ancestors: tf.Tensor            # shape (N,), int32
    did_resample: tf.Tensor         # shape (), bool
    ess: tf.Tensor                  # shape ()
```

Concrete resamplers register as instances:

```python
@register_resampler("systematic", schema=SystematicResamplerParams)
def build_systematic_resampler(params: SystematicResamplerParams) -> Resampler: ...

@register_resampler("stratified", schema=StratifiedResamplerParams)
def build_stratified_resampler(params: StratifiedResamplerParams) -> Resampler: ...

@register_resampler("multinomial", schema=MultinomialResamplerParams)
def build_multinomial_resampler(params: MultinomialResamplerParams) -> Resampler: ...

@register_resampler("ot_entropy", schema=OTEntropyResamplerParams)
def build_ot_entropy_resampler(params: OTEntropyResamplerParams) -> Resampler: ...
```

Functions:

```python
def effective_sample_size_from_log_weights(log_weights: tf.Tensor) -> tf.Tensor: ...

def decide_resampling(
    policy: ResamplingPolicy,
    log_weights: tf.Tensor,         # shape (N,)
    t: tf.Tensor,
) -> ResampleDecision: ...

def systematic_ancestors(log_weights: tf.Tensor, seed: tf.Tensor) -> tf.Tensor: ...   # shape (N,)
def stratified_ancestors(log_weights: tf.Tensor, seed: tf.Tensor) -> tf.Tensor: ...
def multinomial_ancestors(log_weights: tf.Tensor, seed: tf.Tensor) -> tf.Tensor: ...

def systematic_resample(particles: PyTree, log_weights: tf.Tensor, seed: tf.Tensor) -> ResampleOutput: ...
def stratified_resample(particles: PyTree, log_weights: tf.Tensor, seed: tf.Tensor) -> ResampleOutput: ...
def multinomial_resample(particles: PyTree, log_weights: tf.Tensor, seed: tf.Tensor) -> ResampleOutput: ...

def ot_entropy_resample(
    particles: PyTree,
    log_weights: tf.Tensor,
    seed: tf.Tensor,                # ignored for deterministic transport but accepted for API uniformity
    params: OTEntropyResamplerParams,
) -> ResampleOutput: ...

def resample(
    policy: ResamplingPolicy,
    particles: PyTree,
    log_weights: tf.Tensor,
    seed: tf.Tensor,
    t: tf.Tensor,
) -> ResampleOutput: ...
```

`seed` is an explicit `tf.int32` tensor of shape `[2]`, suitable for `tf.random.stateless_*` calls. Resamplers must never read process-level RNG state (no `tf.random.uniform` without `seed` arg, no `np.random`, no `.numpy()` on the seed inside compiled code — recall C2 from `hmc_pipeline_issues.md`).

OT-entropy resampler particulars:

```python
class OTEntropyResamplerParams(BaseModel):
    epsilon: float                # Sinkhorn entropy regularization
    n_sinkhorn_iters: int         # bounded; passed as maximum_iterations to tf.while_loop
    cost: Literal["sqeuclidean"] = "sqeuclidean"
    backward_jitter: float = 1e-6 # regularization on the implicit VJP solve (addresses A4)
    stop_gradient: bool = False
```

The OT backward path must regularize the dense `(2N-1)x(2N-1)` solve. The implementation can use either:

(a) Tikhonov regularization: solve `(M^T M + jitter * I) v = M^T u` instead of `M v = u`.
(b) An iterative Sinkhorn-derivative approximation that does not invert.

Direct dense inversion of the unregularized system, as in current `ot_entropy.py:436,448`, is rejected (it is the documented A4 source of exploding gradients).

Expected pytree shapes for the gate:

```text
N:                    8 or 16
particles:            PyTree leaves with leading shape (N, ...), including at least one tensor leaf (N, state_dim)
log_weights:          (N,), normalized or unnormalized accepted if function normalizes internally
ancestors:            (N,), int32
resampled_particles:  same pytree structure and leaf shapes as particles
reset_log_weights:    (N,), all equal to -log(N) after resampling
should_resample:      (), bool
ess:                  ()
```

This phase builds the non-gradient fast-filter resampling policy, registers the three core stochastic resamplers, and registers the entropy-regularized OT differentiable transport resampler. The OT gate validates gradient plumbing here; full differentiable-likelihood gates in Phase 05 exercise it inside the multi-step Jacobian accumulation path.

## 3. What Gets Tested and Acceptance Criteria

Gate test file: `tests/test_04_resampling.py`

- `test_effective_sample_size_uniform_and_degenerate(result_sink)` asserts ESS is `N` for uniform weights and approximately `1` for one dominant weight. Recommended absolute tolerances: `1e-12` (float64) for uniform; `1e-10` for the degenerate finite approximation.
- `test_conditional_policy_decision_for_all_methods(result_sink)` builds `ResamplingPolicy(mode="conditional", method=method, ess_threshold=0.5)` for `method in ["systematic", "stratified", "multinomial"]` and asserts `should_resample` is false when `ESS / N >= 0.5` and true when `ESS / N < 0.5`.
- `test_always_and_never_policy_decision_for_all_methods(result_sink)` asserts `always` and `never` modes ignore ESS for all methods.
- `test_ancestor_contract_for_all_methods(result_sink)` calls `systematic_ancestors`, `stratified_ancestors`, and `multinomial_ancestors` twice with the same seed and log weights and asserts identical ancestor indices, shape `(N,)`, int32 dtype, and values in `[0, N)`.
- `test_resample_contract_for_all_methods(result_sink)` runs the three discrete-ancestor `*_resample` functions on the same small particle pytree and asserts the pytree structure is preserved, every leaf shape is preserved, log weights are uniform `-log(N)`, ancestors are valid, and dtypes are unchanged.
- `test_resample_dispatch_tffunction_matches_eager_for_all_methods(result_sink)` wraps `resample` in `@tf.function(reduce_retracing=True)` for an always-resample policy for each of the three discrete methods plus `ot_entropy`, then asserts eager and traced outputs match for a fixed seed.
- `test_conditional_resample_preserves_inputs_when_not_resampling(result_sink)` uses high-ESS weights and a conditional policy for all three discrete methods, asserts particles and log weights are unchanged and `did_resample` is false. The implementation must use `tf.cond` over the boolean ESS decision; Python `if` on a traced tensor is not acceptable.
- `test_ot_entropy_resampler_preserves_pytree_and_resets_weights(result_sink)` runs OT resampling on a small fixture and asserts pytree structure, leaf shapes, uniform reset weights, and finite values.
- `test_ot_entropy_resampler_grad_is_finite_nonzero(result_sink)` records the OT resampler under `tf.GradientTape`, computes a scalar loss on transported particles (e.g. mean of squared transported coordinate), and asserts `tape.gradient` with respect to non-uniform log weights is finite and non-zero.
- `test_ot_entropy_backward_regularization_prevents_blowup(result_sink)` constructs a near-degenerate weight set (one weight dominates), invokes OT resampling under `tf.GradientTape`, and asserts the gradient norm stays bounded (e.g. `< 1e6`). With `backward_jitter=0` the same test must fail or produce dramatically larger norms; this directly gates the A4 fix.
- `test_resampling_jit_compile(result_sink)` wraps each registered resampler in `@tf.function(jit_compile=True)` (forward only) and asserts execution succeeds. Implementations must use `tf.while_loop(..., maximum_iterations=...)` for any iterative inner loop (Sinkhorn iterations, branchless cumulative sum) — XLA will not lower an unbounded `tf.while_loop`.

Gate-pass condition:

- `pytest tests/test_04_resampling.py` passes.
- Every test saves JSON results through the Phase 01 fixture.
- Conditional, always, and never policies have explicit tested behavior for systematic, stratified, and multinomial methods.
- All three discrete-ancestor kernels satisfy the same shape, dtype, valid-index, deterministic-for-fixed-seed, and `tf.function` compatibility contract.
- All three discrete-ancestor kernels preserve pytree structure, reset weights after resampling, and leave inputs unchanged when conditional policy decides not to resample.
- The OT entropy resampler satisfies the same pytree and reset contract, plus a finite, non-zero gradient under non-uniform log weights, plus bounded gradient norm under near-degenerate weights.
- Forward-only `jit_compile=True` succeeds for each registered resampler.

No systematic-only minimal alternative is accepted for this phase. The resampling abstraction must be gated against all four kernels it is expected to support so later particle-filter code does not grow method-specific design patterns. The OT regularization gate is non-negotiable — it is a primary fix for the documented A4 instability.

## 4. What the Reader Needs to Understand

### Key Concepts

Resampling policy is a first-class object, not an inline branch. `design.md` section 3 `Resampling Policy` requires conditional ESS resampling as the default for non-gradient runs and smooth/always policies for gradient-facing filters. This phase implements the non-gradient policy surface, all three discrete-ancestor stochastic kernels, and the regularized differentiable OT kernel before any particle filter can hardcode its own ESS logic.

ESS decisions are valid under `tf.function` only when expressed with TF control flow. `diagnosis.md` `Control Flow and Mutation` flags tensor-dependent branch points and Python `if ess < ...` logic as non-graph-safe or non-smooth. This phase makes the decision function return a tensor boolean and requires `resample` to use `tf.cond` for conditional dispatch.

The discrete-ancestor methods are stochastic but deterministic for a fixed `tf.random.stateless_*` seed. This follows the explicit RNG discipline introduced in Phase 02 and avoids hidden RNG state. Later PF steps replay exact ancestor choices by replaying seeds — this is critical for gradient-replay diagnostics in Phase 07 (leapfrog replay).

The gate covers the hardest expected shape contract for this abstraction, not only the easiest method. A single dispatch and result type must work for all four resamplers so bootstrap PF, LEDH PF, and any future filter variant do not introduce separate resampling interfaces.

Gradient-facing resampling is represented through the same registry and must include a regularized differentiable implementation in this phase. This addresses A4 and the more general A5 concern (conditional resampling creates non-smooth likelihood); the OT path is what HMC will use in Phase 06 to keep the likelihood surface smooth.

### Invariants Established

- `ResamplingPolicy` explicitly carries mode, method, ESS threshold, and method params.
- Conditional resampling compares `ESS / N` to `ess_threshold` for non-gradient filters.
- `always` and `never` modes bypass ESS decisions.
- All four resampler outputs satisfy the same `ResampleOutput` contract.
- Resampling resets log weights to uniform `-log(N)` when resampling occurs.
- Conditional no-resample paths preserve particles and log weights unchanged.
- All resamplers consume explicit `[2]`-int32 stateless seeds; no process-level RNG.
- The resampling path is `tf.function`-compatible for static policy choices.
- The OT resampler advertises differentiability in its schema, supports finite, non-zero gradients through non-uniform weights, and stays bounded under near-degenerate weights through `backward_jitter`.
- Every iterative inner loop sets `maximum_iterations` so the path is XLA-lowerable.

### Tricky Bits and Rationale

Weights should be handled in log space at the public boundary. Particle filters naturally produce log weights, and log-space normalization avoids underflow. Each kernel may convert to normalized probabilities internally with `tf.nn.softmax`, but callers should not be required to pre-normalize in probability space.

The policy object should be static at the `tf.function` boundary. Mode and method determine compiled control flow and should not be traced strings. Tensors such as log weights, particles, and seeds are traced values. Pass policy as a Python-side argument captured in the closure of the wrapper, not as a `tf.constant`-of-string.

`tf.cond` is acceptable for non-gradient fast filters. It is **not** the default for differentiable likelihood filters because the branch is discontinuous in parameters (this is the documented A5 failure mode). For Phase 05 LEDH/HMC paths, the policy will be `always` + `ot_entropy` so the likelihood is smooth in θ.

Resampling for arbitrary pytrees should apply ancestor indexing over the leading particle axis of every leaf. The gate must include a pytree input, not just a single tensor, because LEDH carries particles plus per-particle covariances and diagnostics (this is also why B1 — full OT differentiation through transported covariances — surfaced in current code).

OT entropy backward must be regularized. The current `ot_entropy.py:436,448` solves an unregularized dense `(2N-1)x(2N-1)` system that becomes ill-conditioned when the transport plan is sharp. `backward_jitter` and bounded Sinkhorn iterations together gate this.

Multinomial resampling has higher variance than systematic or stratified. Supporting it in the same phase prevents later method-specific exceptions from leaking into PF code.

OT cost matrix is computed exactly once. Current code recomputes it inside `compute_transport_matrix_from_potentials` (C3 in `hmc_pipeline_issues.md`); the new implementation passes it through.

### Alternatives Considered

Gating only systematic resampling is rejected. It would validate the easiest kernel and could still allow method-specific assumptions that break when stratified, multinomial, or OT is added later.

Bundling bootstrap PF into this phase is rejected because it would combine policy mechanics, stochastic propagation, likelihood weighting, scan structure, and diagnostics into one gate. The full resampling contract can and should be retired before PF.

Registering an unregularized OT path is rejected because A4 is one of the documented gradient-explosion sources for HMC. A regularized OT path is the only acceptable differentiable transport here.

Using Python `if` for ESS decisions is rejected because `diagnosis.md` `Control Flow and Mutation` identifies tensor-dependent branch points as a `tf.function` risk.

### Locked Design Decisions Realized

- Decision 2: conditional ESS-threshold resampling is the default policy for non-gradient runs and is configurable, not hardcoded.
- Decision 1: non-gradient and gradient-facing filters keep separate resampling policies rather than one forced interface.
- Decision 4: tests save JSON results through the Phase 01 helper.
- `design.md` section 3 `Resampling Policy`: implements the first-class policy surface and result shape for all core resamplers.
- `diagnosis.md` `Control Flow and Mutation`: replaces Python tensor-dependent branches with TF-safe `tf.cond`.
- `hmc_pipeline_issues.md` A4: regularizes OT backward.
- `hmc_pipeline_issues.md` A5: provides `always` mode and a smooth differentiable transport so HMC can avoid conditional-resampling discontinuities.
- `hmc_pipeline_issues.md` C3: cost matrix computed once.
- `hmc_pipeline_issues.md` Wall 2 prep: `maximum_iterations` set on the Sinkhorn loop so the resampler is XLA-lowerable in forward mode.

### TF Function / JIT Boundary Decisions

`effective_sample_size_from_log_weights`, `decide_resampling`, the three ancestor functions, the four resample functions, and `resample` must be compatible with `@tf.function(reduce_retracing=True)` when the policy is static and particles, log weights, seeds, and time index are traced values.

The branch for conditional resampling must be expressed with `tf.cond`. Dispatch across `method` is static. Config parsing, result saving, and JSON writing remain outside `tf.function`. No filter time scan is introduced in this phase; that belongs to Phase 05.

Forward-only `jit_compile=True` is required to pass for each registered resampler. End-to-end XLA through gradients is not gated here (recall Walls 2-3 in `hmc_pipeline_issues.md` §E for the LEDH+HMC outer path); this phase's resamplers must at least be XLA-lowerable in forward mode so the Phase 05 LEDH path retains its 1.3x forward speedup.

## 5. Dependencies

This phase depends on:

- Phase 01 for package scaffold, scenario schema, RNG discipline, and the JSON gate-test result sink.
- Phase 02 for shared tensor/seed/pytree type conventions.

This phase does not depend on Phase 03; the Kalman filter and resampling policy can be implemented in either order after Phase 02. The recommended sequence is Phase 03 before Phase 04 because exact filters retire reference-model risk before stochastic machinery.

Later dependencies:

- Bootstrap and LEDH particle filters in Phase 05 will use `ResamplingPolicy`, ESS calculation, and the systematic/stratified/multinomial kernels.
- The Phase 05 hard-case gate (LEDH + 29-step Jacobian + SV2D) requires the regularized OT differentiable transport registered here.
- HMC in Phase 06 sets `always_resample=True` plus `ot_entropy` to keep the likelihood smooth in θ.
- Scenario config typing in Phase 08 will validate resampling sections against `ResamplingPolicy` / `GradientResamplingPolicy`.
