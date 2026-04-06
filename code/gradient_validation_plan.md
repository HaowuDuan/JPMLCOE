# Gradient Validation: LEDH+OT Autodiff vs Numerical Gradient (sigma2)

## Goal

Find the gradient bug in LEDH+OT for the 2D SV model. Single parameter (sigma2) is sufficient. Test at multiple evaluation points across the parameter space.

## Scope

- **Filter**: LEDH+OT only
- **Parameter**: sigma2 only (scalar)
- **Evaluation points**: {0.5, 1.0, 1.5, 2.0}
- **File**: `code/tests/hmc/test_gradient_vs_numerical.py`

## Algorithm: Paired Central Differences

For each evaluation point theta_0:

```
1. Choose M radii: r_k = k * h / M, for k = 1..M
2. For each radius r_k:
   - Evaluate f_plus  = ll(theta_0 + r_k) with fixed PF seed
   - Evaluate f_minus = ll(theta_0 - r_k) with fixed PF seed
   - Compute slope_k = (f_plus - f_minus) / (2 * r_k)
3. Check slope consistency across radii (should be stable if linear)
4. Numerical gradient = median of slope_k values
5. Autodiff gradient:
   - Set sigma2 = tf.constant(theta_0)
   - tape.watch, run filter, tape.gradient
6. Compare: sign agreement + relative/absolute error
```

### Why paired central differences over OLS
- Each radius gives an independent gradient estimate
- Can check consistency across radii (if slopes vary wildly, surface is too rough)
- Symmetric pairs cancel even-order curvature bias
- No regression weighting issues (large perturbations don't dominate)

## Handling Resampling Discontinuities

Use `always_resample=True` on the filter. This eliminates the ESS-triggered `tf.cond` branch that can flip between perturbations, causing the likelihood to jump discontinuously for small parameter changes.

## Test Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M (radii) | 10 | 20 total evaluations per point, enough to check consistency |
| h (max perturbation) | 0.05 | Small enough for linearity |
| PF seed | Fixed (42) | Eliminates stochastic noise |
| dtype | float64 | Reduces numerical precision issues |
| N particles | 200 | Small for speed |
| T | 20 | Short series, fast per evaluation |
| n_lambda_steps | 15 | Reduced from 29 for speed |
| eager_mode | True | Simpler, avoids DPFRunner wiring, tests same math |
| always_resample | True | Eliminates resampling branch discontinuities |

## Acceptance Criteria

For each evaluation point:
1. Autodiff gradient is not None and is finite
2. Sign agreement between autodiff and numerical gradient
3. Slopes across radii are consistent (std/|mean| < 0.5)
4. Relative error < 0.30 OR absolute error < 0.5

## Expected Runtime

- Each LEDH evaluation: ~0.5-1s (200 particles × T=20 × 15 lambda steps, eager)
- Per evaluation point: 20 evaluations + 1 autodiff = ~20s
- 4 evaluation points: ~80s
- Total: ~2 minutes

## What Failure Means

- **Autodiff grad None**: Gradient path severed
- **Sign mismatch at all points**: Fundamental gradient bug
- **Sign correct but magnitude off**: Flow Jacobian accumulation or OT transport distorting gradient scale
- **Slopes inconsistent across radii**: Likelihood surface too rough at this scale, or resampling discontinuity not eliminated
- **Correct at theta=1.0 but wrong at theta=2.0**: Basin-of-attraction issue, not gradient bug per se
