# HMC Gradient Optimization — Action Plan

Based on analysis in [HMC_performance.md](HMC_performance.md).

## Goal

Reduce per-gradient-eval time from **75s → ~2-5s** (15-40x speedup) without major architectural changes. This brings a 15-step HMC test run from ~3.1 hours to ~7-12 minutes.

---

## Implementation: `LEDHParticleFlowFilterHMC` subclass

**Approach**: Instead of modifying `ledh_invertible.py`, create a new subclass in [ledh_invertible_hmc.py](src/filters/particle/ledh_invertible_hmc.py) that inherits from `LEDHParticleFlowFilter`. The parent class is untouched. The `filter()` method works identically (inherited). Only `log_marginal_likelihood_tf()` is overridden.

**Status**: IMPLEMENTED

---

## Optimization 1: Compiled flow loop

**What**: The 29-step flow integration is wrapped in `@tf.function`. GradientTape records it as a single `PartitionedCall` op; the backward pass runs via graph-level differentiation in C++ instead of ~32,000 Python-dispatched per-op backward calls.

**How**: `_build_compiled_flow()` creates a closure that captures `n_lambda_steps` and returns a `@tf.function` that unrolls the entire flow loop into one compiled graph. Called from the overridden `log_marginal_likelihood_tf()`.

**Assumption**: `model.observation_jacobian_batch` and `model.observation_function_batch` are parameter-independent (true for Kitagawa: x/10 and x²/20). The trace is valid regardless of sigma_V/sigma_W values.

**Risk**: First call triggers tracing (one-time cost, ~10-30s). If tracing fails due to some Python-side-effect we missed, we'll see an error on the first HMC step.

**Expected impact**: ~32,000 tape entries → ~3,000. Backward pass: 75s → ~5-15s.

---

## Optimization 2: Optional `stop_gradient` on resampling

**What**: When `stop_gradient_resampling=True` (default), `tf.stop_gradient` is applied to particles, weights, and covariances after resampling. This prevents GradientTape from backpropagating through the resampling operation.

**Configurable resampling method**: When `stop_gradient` is on, the HMC path defaults to `systematic` resampling (cheap, non-differentiable). When `stop_gradient` is off, it uses the parent's method (e.g., `ot_entropy`). The user can override via:
- `hmc_resampling_method`: 'systematic', 'soft', or 'ot_entropy'
- `hmc_resampling_config`: kwargs for the chosen method

**Rationale**: In PMCMC (Andrieu et al. 2010), the likelihood estimator p̂(y|θ) = ∏_t (1/N) ∑_i w_t^i doesn't depend on the resampling transport plan. The gradient of log p̂(y|θ) w.r.t. θ flows through particle dynamics and weights, not through which particles survive resampling.

**Trade-off**: The OT transport plan does provide a secondary (noisy) gradient signal through resampling. Cutting it loses this signal but saves the expensive Sinkhorn backward pass (~40 calls × ~100 iterations on 200×200 matrix per gradient eval).

**Testing strategy**: Compare HMC acceptance rates with `stop_gradient_resampling=true` vs `false`. If acceptance rates are similar, the resampling gradient wasn't helping.

**Expected impact**: Eliminates ~40 Sinkhorn backward passes per gradient eval. Combined with Opt 1: 75s → ~3-8s.

---

## Optimization 3: Stripped diagnostics

**What**: The HMC path in `log_marginal_likelihood_tf()` inlines a clean predict/update cycle with no diagnostic tracking:
- No `weights_history.append(...)` — was keeping 100 tensors alive on tape
- No `ess_history.append(...)` — unnecessary for likelihood computation
- No `log_likelihoods.append(...)` — accumulates `total_log_lik` directly
- No `resampled_at.append(...)` — diagnostic only
- No `particles.numpy()` — was forcing GPU→CPU sync inside tape
- No `np.unique(...)` — was running numpy inside tape

**Also**: R and R_inv are pre-computed once before the timestep loop (constant for a given parameter proposal). Also fixed a latent bug: parent's `R_inv_cache` was never reset between HMC proposals, so stale R_inv could be used when sigma_W changes. The subclass resets it in `initialize()`.

**Expected impact**: Reduced memory pressure, fewer tape entries. Combined with Opts 1+2: 75s → ~2-5s.

---

## Benchmark and validation

**What**: Run the same HMC test with the new config and compare:

1. **Timing**: per-step time (should drop from 750s to ~30-50s)
2. **Gradient correctness**: compare HMC acceptance rate
3. **Result correctness**: posterior samples should be similar (within MCMC noise)
4. **Memory**: check peak memory

**How**: Run:
```bash
# Optimized (stop_gradient=true, systematic resampling)
python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc

# Optimized (stop_gradient=false, OT resampling — tests compiled flow only)
python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc \
  filter.stop_gradient_resampling=false

# Original (baseline)
python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh
```

Compare `summary.json` timing and diagnostics across runs.

---

## Future: Full `@tf.function` compilation (Step 5)

Not part of this round. Documented here for reference.

**What**: Refactor `log_marginal_likelihood_tf` to be purely functional — all state (particles, covs, weights) passed as tensors, no `tf.Variable.assign`. Wrap the entire function in `@tf.function`.

**Impact**: Tape entries drop to 1. Backward pass drops to ~0.1-1s. A 750-step HMC run becomes ~12-125 minutes.

**Why not now**: Requires significant refactoring of predict/update to eliminate all mutable state. Better to first validate that the current optimizations give sufficient speedup.

---

## Files

| File | Status | Purpose |
|------|--------|---------|
| [ledh_invertible_hmc.py](src/filters/particle/ledh_invertible_hmc.py) | NEW | HMC-optimized subclass |
| [ledh_invertible.py](src/filters/particle/ledh_invertible.py) | UNCHANGED | Parent class |
| [particle/__init__.py](src/filters/particle/__init__.py) | MODIFIED | Registers new class |
| [kitagawa_ledh_hmc.yaml](configs/dpf/kitagawa_ledh_hmc.yaml) | NEW | Config for HMC-optimized LEDH |
| [hmc_runner.py](src/DF/hmc_runner.py) | UNCHANGED | Timing data already added |

## Summary

| Optimization | Risk | Speedup | Configurable |
|-------------|------|---------|-------------|
| Compiled flow loop | Low (tracing may fail) | ~5-15x | Always on |
| stop_gradient on resample | Low (PMCMC theory) | +2-5x | `stop_gradient_resampling` flag |
| Strip diagnostics | Minimal | +1.5-2x | Always on in HMC path |
| **Combined** | | **15-40x** | |
