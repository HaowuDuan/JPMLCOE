# Pre-Run Likelihood Scan Methodology

## Motivation

MAP and HMC runs on differentiable particle filters are expensive. A
single RB HMC chain at N=500, T=50 takes a few hours. A single SV2D
chain at N=500, T=200 takes tens of hours. Committing multi-hour
compute to find out whether a setting change (particle count,
OT epsilon, number of flow steps, random seed, etc.) affects the
target is wasteful if the answer can be obtained in minutes.

The particle-filter log-likelihood `log p_hat(y | theta; seed)` is
what MAP minimises (negated, plus prior) and what HMC samples from.
Both sit on top of the same underlying function. The shape of that
function fully determines where MAP converges and where HMC
concentrates. If a lever does not move the shape, it cannot change
the MAP or the HMC posterior in a meaningful way.

Evaluating the shape does not require optimisation or sampling.
It requires only forward passes of the filter at a handful of
parameter values. Those forward passes are the cheapest thing the
pipeline can do.

## The Approach

Run the filter forward a few dozen times at a small parameter grid
near the suspected target region, for each candidate setting of a
lever you are considering. Record `log p_hat` only; no gradients,
no autodiff tape, no optimisation. Read off the argmax and the
curvature.

### Steps

1. Pick a parameter grid. For a 1-parameter problem, five points
   straddling the expected peak is enough. For example, truth plus
   two bracket points either side: `[0.08, 0.10, 0.113, 0.13, 0.15]`.
2. Pick the levers to probe and enumerate settings:
   - filter architecture (LEDH vs BPF)
   - particle count N
   - OT epsilon
   - flow steps (n_lambda)
   - observation or dynamics model variant
   - PF seed (stochastic variability per run)
3. For each setting and each grid point, construct a fresh filter
   with those settings, set the model parameter, and call
   `log_marginal_likelihood_tf(obs, seed)`. Record the scalar.
4. Compute the argmax per setting. That is the MAP without running
   any optimiser.
5. Compare argmax locations across settings.

Total cost: `n_points * n_settings` forward filter evaluations.
Each evaluation is one filter pass, typically 10-60 seconds on CPU.
A four-lever, five-point sweep finishes in 10-15 minutes. A
ten-seed variability sweep finishes in 15-25 minutes.

## What the Scan Diagnoses

- **Is a lever a real dial?** Compare argmax across lever settings.
  If argmax does not move when particle count goes from 500 to 2000,
  the current MAP is not finite-N limited.
- **Is PF seed variability a concern?** Run the grid for ten
  different PF seeds. If argmax locations cluster at one value,
  the target surface is seed-robust and a single-seed HMC run is
  representative. If they spread, the single-seed HMC samples one
  of many possible target surfaces.
- **Is truth inside the likely credible interval?** A 1-nat drop
  from peak defines roughly one standard deviation. If truth sits
  within a 1-nat drop of the grid peak, the posterior will cover
  truth without any intervention.
- **Does a candidate model variant fix a bias?** Compute the scan
  at the new model vs the baseline. If the new model's argmax is
  closer to truth than the baseline, the new model is a real fix.
  If not, rebuilding the model is not the bottleneck.

## When to Use It

- Before running any MAP or HMC chain on a new filter setting.
- Before switching between filter architectures.
- When the current single-chain HMC result looks off and you want
  to decide whether the target is broken or just the chain is
  under-sampled.
- Before deciding which configurations to run multi-chain.

## When Not to Use It

- To validate HMC mixing. Chain convergence is orthogonal to target
  shape; the scan tells you nothing about R-hat or ESS.
- To validate posterior calibration when the prior is non-trivial.
  The scan computes only `log p_hat`, not `log p_hat + log p(theta)`.
  Add the prior by hand if the prior is informative.
- When the gradient of `log p_hat` matters (e.g., diagnosing
  gradient spikes). Forward-only evaluation hides gradient
  pathologies.

## Existing Examples

### Lever sweep

`code/tests/filters/test_rb_bias_lever_sweep.py`

Diagnoses which of (particle count, OT epsilon, flow steps) moves
the MAP peak for the RB model. Five grid points times four lever
settings equals twenty forward evaluations. Result on this dataset:
all four settings argmax at `sigma_b = 0.113`. No lever moves the
peak, so all three candidate "fixes" for the RB bearing bias can
be ruled out cheaply.

### PF seed variability

`code/tests/filters/test_rb_seed_variability.py`

Probes whether the RB target is PF-seed-dependent. Fifty forward
evaluations (ten seeds times five grid points). Distribution of
argmax locations across seeds distinguishes "one target surface"
from "many random surfaces".

### Particle-cloud anisotropy

`code/tests/filters/test_rb_particle_cloud_anisotropy.py`

Not a likelihood scan but the same spirit: a single forward filter
run at truth parameters, no gradients, no optimisation. Measures
particle cloud covariance projected onto radial and tangential
axes to quantify geometric anisotropy. Runtime: minutes.

## General Template

```python
@pytest.mark.parametrize("setting_name, setting_kwargs", SETTINGS)
def test_scan(setting_name, setting_kwargs):
    obs_tf = tf.constant(generate_data(...))
    results = {}
    for theta_val in PARAM_GRID:
        filt = make_filter(**setting_kwargs, theta=theta_val)
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=FIXED_SEED)
        results[theta_val] = float(ll.numpy())
    argmax_theta = max(results, key=results.get)
    save_results(setting_name, results, argmax_theta)
```

Fixed seed gives a deterministic scan. Looping over seeds gives
variability. Looping over settings with a common seed gives a
lever comparison. All three reuse the same forward-only evaluation
loop; the outer loop is what changes.

## Guidelines

- Always use the same observation data across scans. The target
  surface depends on the data; comparing scans on different data
  is meaningless.
- Use the same PF seed across settings when comparing levers.
  Different seeds within a comparison confound seed variability
  with lever effects.
- Record the raw `log p_hat` values, not just argmax. Curvature
  (second difference of log-likelihood on the grid) estimates the
  posterior standard deviation at that point, which calibrates how
  close truth should be to the peak.
- Keep the grid small. Five to nine points is enough for a
  one-dimensional scan. If the peak is near the boundary of the
  grid, extend the grid rather than densify it.
- Save results to JSON so they can be re-read without rerunning.

## Relation to the MAP/HMC Pipeline

The scan is a cheaper, stripped-down version of MAP. MAP uses
gradients to locate the argmax; the scan uses a coarse
enumeration. For high-dimensional problems MAP is unavoidable,
but for the one- to three-parameter models common in this
codebase, the scan is faster and unambiguous.

The scan does not replace HMC. HMC samples the full posterior
and produces credible intervals and R-hat. The scan only locates
the peak. Use the scan to decide whether HMC is worth running
and with what settings, then run HMC for the actual inference
deliverable.
