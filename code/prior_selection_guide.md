# Prior Selection Guide for HMC with Differentiable Particle Filters

## Context

In this project, HMC samples from the posterior `p(theta | y) ~ p(y | theta) * p(theta)` where `p(y | theta)` is estimated by a differentiable particle filter (BPF or LEDH). The particle filter likelihood is biased and noisy, which makes prior choice more important than in exact-likelihood settings.

The prior represents your belief about the parameter — it is the **same** regardless of whether you use MAP or HMC. But it affects each method differently:
- **MAP** finds the posterior mode → the prior's **mode** directly pulls the MAP point estimate
- **HMC** samples the full posterior → the prior's **median/mean** affect where the chain concentrates

Choose one prior per parameter. Use it for both MAP and HMC.

## Benchmarking context

In this project we know the true parameter values (simulated data). The goal is to validate the inference pipeline. For benchmarking:

**Set the prior mode = true value.** This ensures any bias in the MAP estimate comes from the pipeline (particle filter, gradient, optimizer), not from prior misalignment.

The workflow:
1. Run MAP with prior mode at the known truth → validates gradients and convergence
2. Run HMC with the same prior → validates posterior sampling
3. If you don't know the truth for HMC, run a series of MAP runs first to locate the optimum, then center the prior there

---

## Choosing a Prior Family

### Positive parameters (noise std, variance, scale)

**Preferred: LogNormal(mu, sigma)**
- Use when uncertainty is multiplicative ("within a factor of 2 or 3")
- Smooth, transform-friendly, well-behaved for HMC
- Set `mu = log(median)` so the median matches your best guess

**When to use HalfNormal(scale)**
- When zero is a plausible shrinkage point (e.g., effect sizes)
- Mode is at zero — bad for MAP on observation noise where the true value is nonzero

**Avoid: InverseGamma(a, b) with a, b ~ 0.001**
- Classic "noninformative" IG creates bad tail geometry for HMC
- Only use if you have a domain reason to put a prior directly on variance

### Bounded parameters (e.g., persistence alpha in [0, 1])

**Preferred: Beta(a, b) on the normalized interval**
- Use `a, b > 1` to avoid boundary-seeking behavior
- `Beta(10, 2)`: weak belief in high persistence (~0.83 mode)
- `Beta(20, 3)`: moderate belief near 0.87

**Avoid: Uniform(0, 1)**
- Maximally uninformative — gives no regularization
- HMC spends time in boundary regions where the particle filter degenerates
- Only appropriate for genuine prior ignorance or sensitivity analysis

---

## Setting LogNormal Parameters

For `theta ~ LogNormal(mu, s)`:

```
median = exp(mu)
mode   = exp(mu - s^2)
mean   = exp(mu + s^2 / 2)
```

### To center the median at a target value `m`:
```
mu = log(m)
```

### To center the MAP mode at a target value `m`:
```
mu = log(m) + s^2
```

### Choosing scale `s`:

| s | Rough 95% range (factor from median) | Use when |
|---|---|---|
| 0.3 | x0.55 to x1.8 | Strong prior knowledge |
| 0.5 | x0.37 to x2.7 | Weakly informative (recommended default) |
| 0.7 | x0.25 to x4.0 | Moderate uncertainty |
| 1.0 | x0.14 to x7.1 | Order-of-magnitude uncertainty |

**Recommendation**: Use `s = 0.5` as the default for weakly informative priors. Only use `s = 1.0` when order-of-magnitude variation is genuinely plausible.

---

## Current HMC Configs — Assessment and Recommendations

### Linear Gaussian: obs_noise_std (true = 1.0)

**Current**: `LogNormal(loc=0.0, scale=1.0)`
- median = 1.0 (good), but mode = 0.368 (far from true for MAP)
- 95% range: [0.14, 7.1] — too vague; HMC visits degenerate noise scales

**Recommended**: `LogNormal(loc=0.25, scale=0.5)`
- mode = 1.0, median = 1.284, mean = 1.455
- 95% range: [0.47, 3.5] — concentrated enough to avoid particle degeneracy
- Mode at true value means MAP won't be pulled away; still weakly informative

### Stochastic Volatility 1D: alpha (true = 0.91, bounded [0.001, 0.999])

**Current**: `Uniform(0.0, 1.0)`
- Maximally uninformative; mean = 0.5 vs true = 0.91
- HMC explores low-persistence regions where the SV model degenerates

**Recommended**: `Beta(10, 2)` mapped to [0.001, 0.999]
- Mode ~ 0.89, mean ~ 0.83, concentrated in the high-persistence region
- Still allows the data to pull the posterior if alpha is genuinely lower
- Implementation: use `tfp.distributions.Beta(10, 2)` and map via the Sigmoid bijector

### Stochastic Volatility 2D: sigma2 (true = 1.0, positive)

**Current**: `LogNormal(loc=0.0, scale=0.5)`
- median = 1.0, mode = 0.779, mean = 1.133
- Well-aligned with true value

**Assessment**: Keep as-is. This is a good weakly informative prior.

### Range-Bearing: sigma_range, sigma_bearing (true = 0.1, positive)

**Current**: `LogNormal(loc=-2.3, scale=0.5)`
- median ~ 0.1, mode ~ 0.078, mean ~ 0.128
- Well-centered near the true value

**Assessment**: Keep as-is. For MAP, if you want mode exactly at 0.1, use `loc = log(0.1) + 0.25 = -2.053` instead.

---

## Prior Predictive Checks

Before running HMC, verify the prior doesn't send the particle filter into degenerate regions:

1. **Sample 100 parameter values from the prior**
2. **For each, run the particle filter forward** (just the likelihood, no gradient)
3. **Check**:
   - Does ESS stay above 10% of N? (If not, prior allows particle collapse)
   - Is the log-likelihood finite? (If not, prior allows impossible parameter values)
   - Are observations from the prior-predictive plausible? (If not, prior is misaligned)

If >20% of prior draws produce collapsed or infinite-likelihood runs, the prior is too vague for this particle filter. Tighten the scale.

---

## Common Mistakes

1. **Using the "textbook noninformative" prior** (Uniform, IG(0.001, 0.001)): These are designed for exact likelihoods. With noisy PF likelihoods, they expose HMC to regions with degenerate gradient estimates.

2. **Centering LogNormal loc at 0 without thinking**: `LogNormal(0, s)` has median 1.0, which is only right if the parameter's natural scale is ~1. For small parameters (like sigma_range=0.1), use `loc = log(0.1) = -2.3`.

3. **Confusing mode and median**: For MAP, the mode matters (it's the peak of the posterior). For HMC, the median/mean matter more (they control where the chain spends time). LogNormal's mode ≠ median ≠ mean — check all three.

4. **Using scale=1.0 "to be safe"**: This gives a 50x range (0.14 to 7.1) which is almost never appropriate for a noise parameter. Use 0.5 as the default.

5. **Flat prior on bounded parameters**: `Uniform(0, 1)` for a persistence parameter wastes HMC samples in low-persistence regions where the model is physically unrealistic.

---

## Summary of Recommended Changes

| Config | Parameter | Current Prior | Recommended Prior |
|---|---|---|---|
| LG bpf_ot | obs_noise_std | LogNormal(0.0, 1.0) | **LogNormal(0.25, 0.5)** |
| LG ledh_ot | obs_noise_std | LogNormal(0.0, 1.0) | **LogNormal(0.25, 0.5)** |
| SV1D ledh_ot | alpha | Uniform(0, 1) | **Beta(10, 2)** |
| SV2D ledh_ot_sigma2 | sigma2 | LogNormal(0.0, 0.5) | Keep |
| RB ledh_ot | sigma_range | LogNormal(-2.3, 0.5) | Keep |
| RB ledh_ot | sigma_bearing | LogNormal(-2.3, 0.5) | Keep |
