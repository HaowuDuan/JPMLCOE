# Prior Selection Guide for MAP and HMC

## LogNormal Primer

All trainable parameters in this project are positive (noise std devs, scale params),
so we use `LogNormal(loc=μ, scale=σ)` priors. Key properties:

| Property | Formula           | Depends on |
|----------|-------------------|------------|
| Mode     | exp(μ - σ²)       | Both μ and σ |
| Median   | exp(μ)            | Only μ |
| Mean     | exp(μ + σ²/2)     | Both μ and σ |

The LogNormal is right-skewed: **mode < median < mean**.

**Critical insight**: The mode shifts left as σ increases. With σ=1, the mode is at
exp(μ-1), which is 2.7x smaller than the median exp(μ). With σ=0.5, the mode is
at exp(μ-0.25), much closer to the median.

---

## Why MAP and HMC Need Different Prior Considerations

### MAP (gradient diagnostic)
MAP finds the **posterior mode**: argmax [log p(y|θ) + log p(θ)].

The prior's mode directly shifts the MAP estimate. If the prior mode ≠ true value,
the MAP converges to a biased point even with perfect gradients. For MAP as a
gradient diagnostic, the prior mode should equal the true value so that
**gradient = 0 at θ_true**.

**Rule: For MAP testing, set μ = log(θ_true) + σ²** so that mode = θ_true.

### HMC (full posterior sampling)
HMC samples the **full posterior distribution**, not just the mode.

The posterior mean/median are likelihood-dominated with sufficient data (T >> 1).
The prior's mode location matters less — what matters is:
1. The prior has reasonable support around the true value (high density, not in the tail)
2. The prior isn't so tight that it distorts the posterior
3. The prior isn't so diffuse that it causes slow mixing in low-data regimes

**Rule: For HMC, set μ = log(θ_true)** (median at true value). Use σ=0.5–1.0
for weakly informative priors. The mode offset is fine because HMC doesn't
target the mode.

---

## Current Prior Audit

### Linear Gaussian — `obs_noise_std` (true = 1.0)

| Config | Prior | Mode | Median | Status |
|--------|-------|------|--------|--------|
| HMC | LogNormal(0, 1) | 0.37 | 1.0 | OK for HMC (median=true) |
| MAP | LogNormal(0, 1) | 0.37 | 1.0 | BAD for MAP (mode ≠ true) |

**MAP fix**: LogNormal(loc=1.0, scale=1.0) → mode = exp(1-1) = 1.0
Or tighter: LogNormal(loc=0.25, scale=0.5) → mode = exp(0.25-0.25) = 1.0

### Range-Bearing — `sigma_range`, `sigma_bearing` (true = 0.1)

| Config | Prior | Mode | Median | Status |
|--------|-------|------|--------|--------|
| HMC | LogNormal(-2.3, 0.5) | 0.078 | 0.10 | OK for HMC |
| MAP | LogNormal(-2.3, 0.5) | 0.078 | 0.10 | Slightly off for MAP |

log(0.1) = -2.3026. With σ=0.5: mode = exp(-2.3 - 0.25) = 0.078.
**MAP fix**: LogNormal(loc=-2.05, scale=0.5) → mode = exp(-2.05-0.25) = exp(-2.3) = 0.10

Note: the current prior is already close (0.078 vs 0.10) because σ=0.5 is small.
This is why range-bearing MAP worked — the mode offset was only 22%, vs 63% for
linear_gaussian (0.37 vs 1.0).

### Kitagawa — `sigma_V` (true ≈ 3.162), `sigma_W` (true = 1.0)

| Param | Prior | Mode | Median | Status |
|-------|-------|------|--------|--------|
| sigma_V | LogNormal(1.15, 0.5) | 2.46 | 3.16 | OK for HMC, slightly off for MAP |
| sigma_W | LogNormal(0, 1) | 0.37 | 1.0 | OK for HMC, BAD for MAP |

**MAP fixes**:
- sigma_V: LogNormal(loc=1.40, scale=0.5) → mode = exp(1.40-0.25) = exp(1.15) ≈ 3.16
- sigma_W: LogNormal(loc=1.0, scale=1.0) → mode = exp(1-1) = 1.0
  Or: LogNormal(loc=0.25, scale=0.5) → mode = 1.0

### Cubic Sensor — `sigma_V`, `sigma_W` (both true = 1.0)

| Param | Prior | Mode | Median | Status |
|-------|-------|------|--------|--------|
| sigma_V | LogNormal(0, 1) | 0.37 | 1.0 | OK for HMC, BAD for MAP |
| sigma_W | LogNormal(0, 1) | 0.37 | 1.0 | OK for HMC, BAD for MAP |

**MAP fix**: Same as linear_gaussian — LogNormal(loc=1.0, scale=1.0) or LogNormal(loc=0.25, scale=0.5).

### Stochastic Volatility — `alpha` (0.91), `sigma` (1.0), `beta` (0.5)

| Param | Prior | Mode | Median | Status |
|-------|-------|------|--------|--------|
| alpha | LogNormal(-0.094, 0.5) | 0.71 | 0.91 | OK for HMC, off for MAP |
| sigma | LogNormal(0, 0.5) | 0.78 | 1.0 | OK for HMC, slightly off for MAP |
| beta | LogNormal(-0.693, 0.5) | 0.39 | 0.50 | OK for HMC, slightly off for MAP |

**MAP fixes**:
- alpha: LogNormal(loc=0.156, scale=0.5) → mode = exp(0.156-0.25) = exp(-0.094) ≈ 0.91
- sigma: LogNormal(loc=0.25, scale=0.5) → mode = exp(0.25-0.25) = 1.0
- beta: LogNormal(loc=-0.443, scale=0.5) → mode = exp(-0.443-0.25) = exp(-0.693) = 0.5

---

## Summary Table: Recommended Priors

### For MAP (mode = true value)

| Model | Param | True | Recommended Prior | Mode |
|-------|-------|------|-------------------|------|
| Linear Gaussian | obs_noise_std | 1.0 | LogNormal(0.25, 0.5) | 1.0 |
| Range-Bearing | sigma_range | 0.1 | LogNormal(-2.05, 0.5) | 0.1 |
| Range-Bearing | sigma_bearing | 0.1 | LogNormal(-2.05, 0.5) | 0.1 |
| Kitagawa | sigma_V | 3.162 | LogNormal(1.40, 0.5) | 3.16 |
| Kitagawa | sigma_W | 1.0 | LogNormal(0.25, 0.5) | 1.0 |
| Cubic Sensor | sigma_V | 1.0 | LogNormal(0.25, 0.5) | 1.0 |
| Cubic Sensor | sigma_W | 1.0 | LogNormal(0.25, 0.5) | 1.0 |
| Stoch. Vol. | alpha | 0.91 | LogNormal(0.156, 0.5) | 0.91 |
| Stoch. Vol. | sigma | 1.0 | LogNormal(0.25, 0.5) | 1.0 |
| Stoch. Vol. | beta | 0.5 | LogNormal(-0.443, 0.5) | 0.5 |

Formula: `loc = log(θ_true) + σ²`. Using σ=0.5 (so σ²=0.25) for all MAP priors
keeps them moderately informative and consistent.

### For HMC (median = true value, weakly informative)

Current HMC priors are already correct — they use `loc = log(θ_true)` which puts
the median at the true value. No changes needed for HMC configs.

---

## Quick Reference Formula

To construct a LogNormal prior with mode at a target value `v`:

```
loc = log(v) + scale²
```

Examples:
- Mode at 1.0, scale=0.5: loc = log(1) + 0.25 = 0.25
- Mode at 1.0, scale=1.0: loc = log(1) + 1.0 = 1.0
- Mode at 0.1, scale=0.5: loc = log(0.1) + 0.25 = -2.05
- Mode at 3.16, scale=0.5: loc = log(3.16) + 0.25 = 1.40
