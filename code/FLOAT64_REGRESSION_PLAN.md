# Float64 Regression Investigation Plan

## Problem

LEDH invertible filter on acoustic tracking gives significantly worse results with float64 vs float32:

| Run | dtype | RMSE | Log-lik | Mean ESS |
|-----|-------|------|---------|----------|
| Old (committed, pre-fix code) | float32 | 14.16 | 1019.74 | — |
| New code | float32 | 16.68 | 1048.10 | smaller |
| New code | float64 | 21.17 | 104.53 | larger |

Float64 should improve accuracy, not regress it. The 10x drop in log-likelihood (1048 → 104) is too large for just a different random draw.

## Root Cause Hypothesis

`tf.random.stateless_normal` produces **different values** for different dtypes even with the same seed. This means:
- Ground truth trajectories differ between float32 and float64 runs
- Initial particles differ
- The entire experiment is a different random realization

This confounds the comparison. A controlled experiment is needed.

## Investigation Plan

### Step 1: Controlled comparison (same data, different filter dtype)

Modify `run_experiment.py` (or write a standalone script) to:
1. Generate data in float64 (highest precision for ground truth)
2. Save ground truth states + observations to disk
3. Run LEDH filter in float32 on the saved data
4. Run LEDH filter in float64 on the same saved data
5. Compare RMSE and log-likelihood on identical trajectories

This isolates the dtype effect on the **filter** from the dtype effect on **data generation**.

### Step 2: If float64 filter is genuinely worse on same data

Trace the per-particle covariances through the pipeline:
1. After `batched_ekf_predict` — are covariances reasonable?
2. After `compute_flow_params_batch` — are A(λ) and b(λ) well-conditioned?
3. After flow integration — are particles in valid regions?
4. After `batched_ekf_update` (Joseph form) — are updated covariances reasonable?

Check for:
- **Eigenvalue spectrum of per-particle covariances**: float64 may preserve tiny eigenvalues that float32 rounds to zero, leading to ill-conditioned flow
- **safe_cholesky jitter**: `1e-10 * avg_diag` — in float32 this is near machine epsilon, in float64 it's 6 orders above. The jitter may behave differently
- **Weight computation**: `compute_flow_weights` in `distributions.py` — check if log-probabilities overflow differently in float64

### Step 3: If float64 filter is equal or better on same data

The regression was entirely due to different random realizations. Run multiple seeds (e.g., 10 seeds) to get statistical comparison:
```
for seed in 42 43 44 45 46 47 48 49 50 51; do
  python -m code.src.experiments.run_experiment \
    experiment=acoustic_tracking/acoustic_tracking_ledh_invertible \
    seed=$seed dtype=float32
  python -m code.src.experiments.run_experiment \
    experiment=acoustic_tracking/acoustic_tracking_ledh_invertible \
    seed=$seed dtype=float64
done
```

### Step 4: Address P vs S regularization consistency

From earlier analysis: regularizing P (prior covariance) is philosophically consistent — the regularized P should be used everywhere in the flow equations for consistency. Currently:
- P is regularized in `compute_flow_params_batch` via explicit `regularization` parameter (default 1e-8)
- S (innovation covariance) gets implicit regularization via `safe_cholesky` jitter (1e-10 × avg_diag)

If only S is regularized, there is a philosophical inconsistency: S depends on P via S = λHPH^T + R, so regularizing S without regularizing P means the flow equations use two different versions of P.

Current approach (regularize P) is correct. No change needed unless Step 2 reveals otherwise.

## Files Involved

- `code/src/filters/particle/ledh_invertible.py` — LEDH filter (now inherits dtype from model)
- `code/src/filters/kalman/batched_ekf.py` — Joseph form + Cholesky solve
- `code/src/utils/flow_params.py` — A(λ), b(λ) computation
- `code/src/utils/distributions.py` — weight computation
- `code/src/utils/linalg.py` — safe_cholesky, safe_solve
- `code/src/models/acoustic_tracking_full.py` — model with configurable dtype
- `code/src/experiments/run_experiment.py` — experiment runner with dtype propagation

## Step 5: MATLAB-to-Python Parameter Re-tuning

Parameters from the MATLAB implementation were calibrated for MATLAB's numerical environment (float64 default, Intel MKL, different RNG). These may not be optimal for TensorFlow (float32, different BLAS, stateless RNG on MPS/CPU). Candidates for re-tuning:

### Flow integration
- **`n_lambda_steps = 29`**: Number of pseudo-time integration steps. More steps = finer Euler discretization but slower. MATLAB's float64 may tolerate fewer steps; TF float32 may need more (or fewer) for best accuracy/speed tradeoff.
- **`q = 1.2`** (exponential growth factor in `_generate_lambda_steps`): Controls how step sizes grow from small (early) to large (late). This ratio was chosen for MATLAB's precision. A different `q` may better suit float32 numerics.
- **`epsilon_1`** (initial step size, derived from `q` and `n_lambda_steps`): Implicitly tuned via `q` and `n_lambda_steps`. With float32, very small initial steps may underflow or be numerically zero.

### Regularization
- **`regularization = 1e-8`** (flow_params P regularization): Scales with `trace(P)/state_dim`. In float32, machine epsilon is ~1e-7, so 1e-8 regularization may be below numerical noise. In float64, it's well above machine epsilon (~1e-16). May need dtype-dependent tuning.
- **`safe_cholesky jitter = 1e-10`**: Same concern — near float32 machine epsilon, well above float64 machine epsilon.

### Resampling
- **`resample_threshold = 0.5`**: Resample when ESS/N < 0.5. Currently resampling at every timestep (rate = 1.0), suggesting weights degenerate heavily. Threshold may need adjustment.

### Observation model
- **`1e-10` in `tf.sqrt(dx**2 + dy**2 + 1e-10)`** (acoustic_tracking_full.py): Numerical floor for distance computation. Appropriate for float64 but may compete with float32 precision.

### Approach
For each parameter, test sensitivity by sweeping values while holding others fixed on a reference dataset. Start with `n_lambda_steps` and `q` as they directly control flow integration accuracy.

## Current Status

- Experiment config set to `dtype: float32` for now (working)
- LEDH filter dtype inheritance fixed (uses `getattr(model, 'dtype', tf.float64)`)
- Investigation deferred to later optimization pass
