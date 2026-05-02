# Rebuilt Plan — Overview

**Goal**: rebuild the JPMLCOE codebase from scratch within **TensorFlow 2.x + TFP**, keeping the same research capability (differentiable particle filters, HMC/MAP on nonlinear state-space models) while fixing the structural debt accumulated in the current code.

**Why not JAX**: the existing `step*.md` files plan a JAX rebuild. This plan keeps TensorFlow because (a) the gradient, optimizer, and HMC infrastructure in the current repo is TF-based and validated on RTX 3090, (b) Office's install chain (CUDA libs, XLA flags, TF_USE_LEGACY_KERAS) is TF-specific and working, (c) the team's mental model is TF. Switching frameworks doubles the rebuild scope.

**What stays from the JAX plan** (design principles are framework-agnostic):
- Family-generic registries instead of algorithm-named enums
- Capability protocols instead of monolithic base classes
- Explicit parameter trees instead of mutable model attributes
- Hard-case-first gate tests (SV2D + LEDH + 29-step Jacobian from the start)
- Uniform JSON result sink
- Pydantic-based scenario config

**What changes for TF**:
- `tf.function(jit_compile=True)` / `tf.while_loop` instead of `jax.jit` / `jax.lax.scan`
- `tf.random.stateless_*` + explicit `[seed, step]` keys instead of JAX PRNGKey splits
- `tf.nest.map_structure` for parameter trees (equivalent to JAX pytrees)
- `tf.Module` + `tf.TensorSpec` for signature-stable filter interfaces
- `tfp.mcmc` kernels for HMC/NUTS

## Non-goals (out of scope for this plan)
- Neural operator workstream (keep existing under `neural_operator/`)
- Report (`report/main_reorganized.tex`) — rebuilds the code, not the writing
- Migration of stochastic_volatility_2d HMC freeze (separate debugging)
- JAX port (explicitly rejected)

## Top-level phase structure

| Phase | Content | Gate |
|---|---|---|
| 01 | Foundation, config, results | Pydantic scenario parses hard fixture; result sink writes deterministic JSON |
| 02 | Parameters & models | Linear Gaussian model with learnable param; autodiff Jacobians default |
| 03 | Kalman filter | Learnable-param gradient through Kalman log-likelihood is finite non-zero |
| 04 | Resampling registry | ResamplingPolicy object with conditional/always/never + 3 methods + OT with differentiable gradient |
| 05 | Particle filter + flow | LEDH with 29-step Jacobian accumulation on SV2D; autodiff = finite-diff within tolerance |
| 06 | HMC + MAP + mass matrix | TFP kernel + preconditioned HMC + dual averaging; 4-chain R-hat on LG passes |
| 07 | Diagnostics | R-hat/ESS extractor, leapfrog replay, Hessian-at-peak, anisotropy, seed variability |
| 08 | Migration & cleanup | Old `code/` moved to `legacy/`; new tree under `code/`; tests green |

## Key anti-patterns to avoid (from existing `diagnosis.md`, `hmc_pipeline_issues.md`, `design_pressure_audit_v2.md`)

1. **Mutable model attributes** for learnable parameters (`model.sigma2 = new_value` inside gradient tape). Breaks `tf.function` re-tracing and thread-safety.
2. **YAML-coupled parameter trainability** — determining trainable by config keys, not explicit schema. Implicit coupling.
3. **Python loops over dynamic time** — use `tf.while_loop` or `tf.scan`.
4. **Conditional resampling (`tf.cond(ess < thresh,...)`) in gradient paths** — discontinuity kills HMC. Use `always_resample=True` or smooth resampling.
5. **Algorithm names in interface types** — use family + registry `algorithm: str`.
6. **Inline matrix inversions** without regularization — every backward solve needs ridge or SVD truncation.
7. **Hidden RNG state** — every stochastic op takes an explicit seed.
8. **Large artifacts in JSON** — save arrays as `.npy`/`.npz`, reference paths.

## Critical lessons encoded as requirements

From `hmc_pipeline_issues.md` A1–A6 and `particle_filter_gradient_bias.md`:
- LEDH must support state-dependent observation covariance (`R_i = f(x_i)`), not constant R
- OT backward pass must be regularized (avoid direct `(2N−1)×(2N−1)` inversion)
- Jacobian accumulation through 29 flow steps must be autodiff-compatible AND memory-bounded (checkpointing if needed)
- Always-on resampling must be selectable per-call (default True for gradient paths)
- Conditional resampling is allowed only for forward-only filtering (no gradients)

From `particle_filter_gradient_bias.md`:
- Accept finite-N PF likelihood bias as **structural**, not a bug. Report honestly.
- Bias magnitude ~1% for linear, ~13% for RB, ~20% for SV2D — measured, not speculation.
- Mitigations are research-level (multi-seed logmeanexp target, correlated PM, unbiased couplings); not part of the base rebuild.

From `design_pressure_audit_v2.md`:
- Each abstraction gate uses the HARDEST current fixture, not the easiest.
- Registry-based dispatch — adding a new filter or sampler does not touch framework code.
- Capability protocols — base `StateSpaceModel` is `sample_initial / sample_transition / log_obs_prob` only. Gaussian moments, analytic Jacobians, closed-form priors are opt-in.

## File structure of this plan

```
rebuilt_plan/
  00_overview.md                      (this file)
  01_foundation_and_config.md         (package, scenario config, startup, results sink)
  02_parameters_and_models.md         (ParameterSchema, model protocol, LG model)
  03_kalman_filter.md                 (first differentiable filter; gate on learnable param)
  04_resampling_registry.md           (policy + registry; systematic/stratified/OT)
  05_particle_filter_and_flow.md      (BPF then LEDH+OT; SV2D hard-case gate)
  06_hmc_and_map.md                   (TFP HMC/NUTS kernels; preconditioned HMC; MAP runner)
  07_testing_and_diagnostics.md       (gate tests, R-hat/ESS, leapfrog replay, Hessian)
  08_migration.md                     (moving from old code/; legacy dir; config migration)
```

Each phase file has: objective, deliverable files, gate tests, pass criteria, risks, estimated effort.
