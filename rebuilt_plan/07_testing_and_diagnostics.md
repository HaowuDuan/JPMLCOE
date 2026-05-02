# Phase 07: Host-Side Diagnostics and Numerical Test Discipline

## 1. Goal

Build the diagnostics layer as host-side post-processing over outputs from Phases 05 and 06: chain convergence summaries, leapfrog energy replay, posterior Hessian at the MAP peak, particle-cloud anisotropy, and seed-variability sweeps, all saved through the uniform JSON result sink and never executed inside `tf.function` kernels.

## 2. What Gets Built

Files and modules owned by this phase:

- `jpml_tf/diagnostics/__init__.py` - exports the host-side diagnostic surface.
- `jpml_tf/diagnostics/common.py` - host-only guards, artifact helpers, and small shared containers.
- `jpml_tf/diagnostics/chain_stats.py` - R-hat and ESS extraction from multi-chain sample arrays or parameter trees.
- `jpml_tf/diagnostics/energy_replay.py` - leapfrog trajectory replay and energy-drift summarization for HMC/NUTS traces.
- `jpml_tf/diagnostics/hessian.py` - posterior Hessian-at-peak utilities using nested `tf.GradientTape` over the exact MAP objective.
- `jpml_tf/diagnostics/particles.py` - particle-cloud anisotropy and weighted covariance summaries from particle-filter outputs.
- `jpml_tf/diagnostics/seed_sweep.py` - repeated-run sweep helpers that vary stateless seeds, summarize variability, and write large traces as external artifacts.
- `tests/test_07_testing_and_diagnostics.py` - the single gate test file for this phase.

Host-only utilities and result containers:

```python
# jpml_tf/diagnostics/common.py
def ensure_host_side_only() -> None: ...

class DiagnosticArtifact(NamedTuple):
    metrics: Mapping[str, float]
    artifacts: Mapping[str, str]

# jpml_tf/diagnostics/chain_stats.py
class ChainDiagnostics(NamedTuple):
    rhat: tf.Tensor                         # shape (event_size,) or tree leaves
    ess_bulk: tf.Tensor                     # shape (event_size,) or tree leaves
    ess_tail: tf.Tensor | None
    mean_accept_prob: tf.Tensor             # shape ()

def summarize_chains(samples: Params | tf.Tensor) -> ChainDiagnostics: ...

# jpml_tf/diagnostics/energy_replay.py
class LeapfrogTrace(NamedTuple):
    positions: tf.Tensor                    # shape (L + 1, event_size)
    momenta: tf.Tensor                      # shape (L + 1, event_size)
    step_size: tf.Tensor                    # shape ()
    mass_diag: tf.Tensor                    # shape (event_size,)
    logdensity: tf.Tensor                   # shape (L + 1,)

class EnergyReplayResult(NamedTuple):
    delta_h: tf.Tensor                      # shape (L,)
    max_abs_delta_h: tf.Tensor              # shape ()
    mean_abs_delta_h: tf.Tensor             # shape ()

def replay_leapfrog_energy(
    trace: LeapfrogTrace,
    objective_fn: Callable[[tf.Tensor], LogDensityResult],
) -> EnergyReplayResult: ...

# jpml_tf/diagnostics/hessian.py
class HessianResult(NamedTuple):
    hessian: tf.Tensor                      # shape (event_size, event_size)
    eigenvalues: tf.Tensor                  # shape (event_size,)
    condition_number: tf.Tensor             # shape ()

def posterior_hessian_at_peak(
    objective_fn: Callable[[tf.Tensor], LogDensityResult],
    peak_position: tf.Tensor,
) -> HessianResult: ...

# jpml_tf/diagnostics/particles.py
class ParticleAnisotropyResult(NamedTuple):
    weighted_cov: tf.Tensor                 # shape (state_dim, state_dim)
    singular_values: tf.Tensor              # shape (state_dim,)
    anisotropy_ratio: tf.Tensor             # shape ()

def particle_cloud_anisotropy(
    particles: tf.Tensor,                   # shape (N, state_dim)
    log_weights: tf.Tensor | None = None,
) -> ParticleAnisotropyResult: ...

# jpml_tf/diagnostics/seed_sweep.py
class SeedSweepResult(NamedTuple):
    per_seed_metrics: Mapping[str, tf.Tensor]
    mean_metrics: Mapping[str, tf.Tensor]
    std_metrics: Mapping[str, tf.Tensor]
    max_spread: Mapping[str, tf.Tensor]

def run_seed_sweep(
    run_once: Callable[[tf.Tensor], Mapping[str, tf.Tensor]],
    seeds: tf.Tensor,                       # shape (K, 2), int32
) -> SeedSweepResult: ...
```

The key architectural rule is enforced by `ensure_host_side_only()`: diagnostics are called after compiled kernels return arrays to the host. They do not sit inside particle-filter scans, sampler step functions, or `tf.while_loop` bodies.

Expected diagnostic input shapes:

```text
hmc_samples:                  (4, num_kept, event_size) or tree-equivalent
accept_prob_trace:            (4, num_kept)
leapfrog_positions:           (L + 1, event_size)
leapfrog_momenta:             (L + 1, event_size)
map_peak_position:            (event_size,)
particle_cloud:               (N, state_dim)
particle_log_weights:         (N,)
seed_matrix:                  (K, 2), int32
```

Large traces, Hessians, or particle histories are not embedded inline in JSON. They are written as `.npy` or `.npz` artifacts under the Phase 01 artifact root and referenced by relative path in saved results.

## 3. What Gets Tested and Acceptance Criteria

Gate test file: `tests/test_07_testing_and_diagnostics.py`

- `test_rhat_and_ess_extractor_matches_reference_values(result_sink)` feeds the 4-chain Linear Gaussian HMC output from Phase 06 into `summarize_chains`, compares R-hat and ESS against either TFP reference functions or a known analytical fixture, and asserts matching within tight test-code tolerance. The test saves scalar summaries through the uniform sink.
- `test_leapfrog_energy_replay_recovers_energy_drift(result_sink)` records a short HMC trace with fixed seed, replays it with `replay_leapfrog_energy`, and asserts the recomputed `delta_h` agrees with the stored energy changes within test-code tolerance on the healthy trace. A perturbed synthetic trace should show materially larger drift, proving the diagnostic is sensitive rather than decorative.
- `test_posterior_hessian_at_peak_matches_linear_gaussian_reference(result_sink)` uses the MAP peak from Phase 06 on the exact Kalman objective, computes `posterior_hessian_at_peak`, and asserts symmetry, finite eigenvalues, and agreement with an analytical Linear Gaussian Hessian or finite-difference Hessian within test-code tolerance.
- `test_particle_cloud_anisotropy_identifies_isotropic_and_collapsed_clouds(result_sink)` evaluates `particle_cloud_anisotropy` on an isotropic synthetic cloud and on a deliberately collapsed or strongly elongated cloud, then asserts the isotropic ratio is near `1` while the collapsed ratio exceeds a test-code threshold such as `10`.
- `test_seed_variability_sweep_summarizes_particle_or_sampler_runs(result_sink)` runs a small repeated experiment across a fixed set of seeds, asserts `SeedSweepResult` returns per-seed metrics plus mean, standard deviation, and spread summaries, and asserts any large per-seed traces are saved externally rather than inline in JSON.
- `test_diagnostics_reject_tf_function_context(result_sink)` wraps one representative diagnostic call in `@tf.function` and asserts `ensure_host_side_only()` raises a clear host-side-only error. This is the explicit gate that keeps `.numpy()`, file I/O, and progress reporting out of compiled kernels.

Gate-pass condition:

- `pytest tests/test_07_testing_and_diagnostics.py` passes.
- Every test saves a JSON result through the Phase 01 `result_sink` fixture.
- R-hat and ESS extraction agrees with a trusted reference implementation.
- Leapfrog replay can both verify a healthy trace and flag an unhealthy one.
- Hessian-at-peak is finite, symmetric, and numerically consistent with the exact Linear Gaussian reference.
- Particle anisotropy correctly distinguishes balanced and collapsed clouds.
- Seed sweep summaries are host-side, reproducible, and artifact-aware.
- Diagnostics reject execution inside `tf.function` contexts.

No "plots only" alternative is accepted for this phase. Diagnostics must be machine-readable, testable, and saved through the same result infrastructure as the rest of the rebuild.

## 4. What the Reader Needs to Understand

### Key Concepts

Diagnostics are not part of the numerical kernels. They are host-side readers over outputs already computed by filters, likelihoods, and samplers. This separation is what fixes the current codebase's mixture of computation, logging, plotting, JSON writing, and `.numpy()` conversions inside hot paths.

The phase covers two diagnostic families. Chain diagnostics and leapfrog replay help decide whether sampler outputs from Phase 06 are trustworthy. Hessian and particle-cloud diagnostics help explain why objectives or particle systems look the way they do. Seed sweeps quantify how much stochastic variability remains after the explicit-seed discipline from earlier phases.

The exact Linear Gaussian objective is still the right reference for some diagnostics even though particle and sampler gates are already harder. Hessian-at-peak is easiest to validate where a reference answer exists, so this phase deliberately uses the Phase 03 Kalman objective as its first hard reference while still consuming sampler and particle outputs from later phases.

### Invariants Established

- Diagnostics are host-side functions guarded by `ensure_host_side_only()`.
- No diagnostic writes files or calls `.numpy()` from inside compiled filter or sampler kernels.
- Every diagnostic returns scalar summaries and optional artifact references through the Phase 01 result sink contract.
- Chain diagnostics operate on sample arrays or parameter trees, not sampler-internal mutable state.
- Leapfrog replay consumes an explicit saved trace with positions, momenta, step size, and mass matrix.
- Hessian-at-peak operates on explicit objective callables and explicit peak positions.
- Seed-sweep comparisons use explicit stateless seed matrices rather than hidden RNG state.

### Tricky Bits and Rationale

R-hat and ESS must be extracted on a stable parameterization. For low-dimensional Linear Gaussian gates, evaluating them on the unconstrained sampled coordinates is acceptable and easy to audit. If later models need constrained-space summaries as well, those should be added explicitly instead of silently transforming some diagnostics and not others.

Leapfrog replay only works if the objective function is deterministic under replay. That is why Phase 06 fixed the particle objective seed policy. If replay changes resampling randomness or particle noise between evaluations, energy drift becomes impossible to interpret.

Hessian-at-peak needs symmetry enforcement. Nested `tf.GradientTape` can produce small asymmetries from finite numerical noise, especially when the objective internally uses Cholesky solves. The diagnostic should symmetrize with `(H + H^T) / 2` before computing eigenvalues and should report the conditioning explicitly rather than hiding it.

Particle anisotropy is most informative in weighted space, not raw cloud space. The implementation therefore uses weighted covariance when log weights are available. A tiny ridge is acceptable before SVD so the diagnostic does not fail on exactly collapsed clouds; the ridge is a diagnostic regularizer, not a model change.

Seed sweeps can produce many arrays. The scalar summaries go into JSON, but per-seed traces belong in external `.npz` artifacts. This phase extends the Phase 01 result-sink discipline to diagnostic workloads without weakening the compact-JSON rule.

### Alternatives Considered

Embedding R-hat, ESS, or energy-drift computation inside `tf.function` kernels is rejected because it recreates the side-effect and host-conversion problems already diagnosed.

Keeping diagnostics as ad hoc notebooks or plots is rejected because the rebuild needs auditable numerical gates, not only visual inspection.

Approximating Hessian-at-peak from optimizer moments is rejected because it conflates optimizer internals with curvature and cannot be compared cleanly to a reference target.

Skipping the host-side-only guard is rejected because the repo has already shown how easily diagnostic code leaks into compiled paths once the boundary is informal.

### Locked Design Decisions Realized

- The diagnostic-control rule from `design.md` is realized in TensorFlow form: kernels return arrays, hosts write sinks.
- The uniform JSON result sink from Phase 01 is now used by every diagnostic gate.
- The hard sampler gate from Phase 06 is extended with chain-quality diagnostics rather than being treated as self-certifying.
- The exact Kalman objective from Phase 03 remains the curvature reference for Hessian validation.
- Seed determinism is preserved end to end: diagnostics do not introduce new RNG side paths.

### JIT Boundary Decisions

Diagnostics stay outside `tf.function`. They may consume eager tensors, NumPy arrays converted after kernel return, or externally saved artifacts, but they do not sit inside compiled loops and they do not get their own `jit_compile=True` wrappers.

Compiled kernels from earlier phases are allowed to return richer trace bundles when a diagnostic mode is enabled. That is still not the same thing as running the diagnostic in the kernel. The numerical kernel returns tensors; the host-side diagnostic module interprets them after execution.

Result saving, artifact writing, plotting, and summary JSON construction all remain host-side in this phase. If a later optimization proves a pure metric computation worth compiling, it must still remain outside the model/filter/sampler hot path and keep the same host-side contract.

## 5. Dependencies

This phase depends on:

- Phase 01 for the JSON result sink and external artifact reference conventions.
- Phase 03 for the exact Kalman objective used as the Hessian reference target.
- Phase 05 for particle histories, particle-cloud outputs, flow diagnostics, and explicit seed discipline on particle objectives.
- Phase 06 for HMC/NUTS chain outputs, leapfrog traces, MAP peaks, and convergence summaries.

Later dependencies:

- Phase 08 ports the legacy diagnostic tests and artifacts onto these host-side modules and verifies they all write through the same result-sink contract.
