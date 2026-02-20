# DPF Component Test Plan

## Context

Testing DPF components incrementally before tackling LEDH invertible on Kitagawa (hard). Two test files isolate different pieces:

1. **Bootstrap PF + OT resampling on range-bearing** — tests OT resampling in isolation (no flow complexity)
2. **Full LEDH invertible pipeline on cubic sensor** — tests the complete flow machinery on an easy 1D model

## File 1: `code/tests/dpf/test_bootstrap_ot_range_bearing.py`

**Setup**: `RangeBearingModel(sigma_range=0.1, sigma_bearing=0.1)`, T=25, N=150, seed=42. Session-scoped fixtures cache `FilterResult` to avoid re-running.

| # | Test | What it checks | Threshold |
|---|------|---------------|-----------|
| 1 | `test_bpf_ot_runs_without_error` | OT bootstrap PF completes | no error |
| 2 | `test_output_shapes` | means (T,2), covs (T,2,2), ess (T,), weights (T,N) | exact shape |
| 3 | `test_ess_does_not_collapse` | min ESS across timesteps | > 5% of N (7.5) |
| 4 | `test_rmse_reasonable` | RMSE of filter means vs true states | < 1.0 |
| 5 | `test_ot_vs_systematic_both_work` | Both resampling methods track target, finite log-lik | RMSE < 1.0 each |
| 6 | `test_ot_transport_matrix_doubly_stochastic` | Direct call to `ot_entropy_resample` — row sums ~1/N, non-negative, uniform output weights | rtol=0.1 |
| 7 | `test_log_likelihood_finite` | Total and per-timestep log-lik finite | all finite |

**Key classes**: `ParticleFilterTF` (bootstrap_pf_tf.py), `ot_entropy_resample` (resampling/ot_entropy.py), `RangeBearingModel`

## File 2: `code/tests/dpf/test_ledh_cubic_sensor.py`

**Setup**: `CubicSensorModel(a=0.9, c=0.05, sigma_V=1.0, sigma_W=1.0)`, T=20, N=100, n_lambda_steps=15, seed=42. Session-scoped fixtures cache results.

| # | Test | What it checks | Threshold |
|---|------|---------------|-----------|
| 1 | `test_ledh_runs_without_error` | LEDH filter completes on 1D cubic sensor | no error |
| 2 | `test_output_shapes_1d` | means (T,1), covs (T,1,1), ess (T,), log_liks (T,) | exact shape |
| 3 | `test_ess_stays_healthy` | min ESS across timesteps | > 5% of N (5.0) |
| 4 | `test_rmse_reasonable_1d` | RMSE vs true states | < 3.0 (generous, stationary var ~5.26) |
| 5 | `test_log_likelihood_finite` | Total and per-timestep log-lik finite | all finite |
| 6 | `test_gradient_connected_and_finite` | `d/d_sigma_W` of `log_marginal_likelihood_tf` using OT resampling | not None, finite, > 1e-10 |
| 7 | `test_weight_clipping` | Filter runs with clip=30 and without; both finite | both finite, clipped ESS > 2% of N |
| 8 | `test_both_resampling_methods` | Systematic and OT both produce valid LEDH results | RMSE < 3.0 each, ESS > 5% |

**Key classes**: `LEDHParticleFlowFilter` (ledh_invertible.py), `CubicSensorModel` (cubic_sensor.py)

## Design Decisions

- **Session-scoped fixtures** for filter results — PF runs are expensive, multiple tests inspect the same result
- **Self-contained files** — no modifications to existing `conftest.py`
- **Small params for speed** — N=100-150, T=20-25, n_lambda_steps=15
- **Gradient test uses OT** — systematic is non-differentiable; OT is required for DPF parameter inference
- **Transport matrix test is isolated** — calls `ot_entropy_resample` directly, not through the full PF

## Verification

```bash
# Run each file
pytest code/tests/dpf/test_bootstrap_ot_range_bearing.py -v -s
pytest code/tests/dpf/test_ledh_cubic_sensor.py -v -s

# Run all DPF tests together
pytest code/tests/dpf/ -v -s
```
