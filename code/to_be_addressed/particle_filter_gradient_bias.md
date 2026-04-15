# Particle Filter Gradient Bias in MAP/HMC Inference

## The Problem

When using a differentiable particle filter for MAP optimization or HMC sampling, the gradient of the estimated log-likelihood is **biased** relative to the gradient of the true log-likelihood. This causes:
- MAP estimates that don't converge to the true parameter value
- Persistent gradient that doesn't go to zero at the truth
- Overshoot past the optimum under stochastic gradient optimization

## Source of the Bias

### Unbiased likelihood, biased log-likelihood

A particle filter with N particles estimates the likelihood as:

```
p̂(y_{1:T} | θ) = Π_t [ (1/N) Σ_i w_t^i ]
```

This estimate is **unbiased**: `E[p̂(y|θ)] = p(y|θ)`.

The log-likelihood estimate is:

```
log p̂(y|θ) = Σ_t log( (1/N) Σ_i w_t^i )
```

By **Jensen's inequality** (log is concave):

```
E[log p̂(y|θ)] ≤ log E[p̂(y|θ)] = log p(y|θ)
```

So the log-likelihood estimate is **biased downward**. The bias is approximately:

```
bias(θ) ≈ -Var[p̂(y|θ)] / (2 * p(y|θ)^2)
```

### The bias depends on θ

The variance of the likelihood estimate depends on the parameter θ:
- At some θ values, the particle filter has low weight variance → small bias
- At other θ values, weight degeneracy is worse → large bias

Since the bias is a function of θ, **the gradient of the bias is nonzero**:

```
∇_θ log p̂(y|θ) ≈ ∇_θ log p(y|θ) + ∇_θ bias(θ)
```

The spurious term `∇_θ bias(θ)` pushes the MAP estimate toward regions where the particle filter has lower variance, not where the true likelihood peaks. These are generally not the same location.

### Severity depends on the model

| Model | Bias severity | Why |
|---|---|---|
| Linear Gaussian | Small | Particle filter is nearly exact; weights are uniform |
| Stochastic Volatility 1D | Moderate | Nonlinear observation; weight variance depends on volatility |
| Stochastic Volatility 2D | Large | State-dependent observation noise `exp(x_2)` creates high weight variance when particles have diverse volatilities |
| Range-Bearing | Moderate-Large | Nonlinear observation function; bearing wrapping adds discontinuities |

## What does NOT cause this bias

- **Entropy-regularized OT resampling**: The bias exists for any particle filter regardless of resampling method (systematic, multinomial, soft, OT, or none). OT affects gradient flow through the resampling step, but the Jensen's inequality bias is from the particle approximation itself.
- **LEDH flow model mismatch**: The flow mismatch (e.g., constant R assumption for SV2D) causes gradient **spikes** and **noise**, but not the persistent downward bias. Fixing the flow (per-particle R) reduces noise but doesn't eliminate the Jensen bias.
- **Learning rate schedule**: Cosine decay reduces overshoot from noisy gradients but doesn't correct the bias — the MAP estimate will still converge to a biased location.

## Mitigation strategies

### Reduce the bias (reduce variance of p̂)

| Strategy | Effect | Cost |
|---|---|---|
| **More particles (N)** | Var ∝ 1/N → bias ∝ 1/N | Linear in N |
| **Better proposal (LEDH flow)** | Lower weight variance → smaller bias | Per-step flow cost |
| **Fewer timesteps (T)** | Less bias accumulation | Reduces data used |

### Work around the bias

| Strategy | Effect | Tradeoff |
|---|---|---|
| **Fixed seed** (`random_seed: false`) | Deterministic surface; gradient goes to zero at the (biased) optimum | MAP point is biased toward that seed's particle configuration |
| **Gradient averaging** (K seeds per step) | Reduces gradient noise by √K; bias unchanged | K× slower |
| **Lower learning rate** | Reduces overshoot; final estimate still biased | Slower convergence |
| **HMC instead of MAP** | Metropolis correction accounts for biased gradients; posterior is asymptotically correct as N→∞ | Much more expensive |

### Correct the bias (research-level)

| Strategy | Reference |
|---|---|
| **Pseudo-marginal MCMC** | Andrieu & Roberts (2009) — uses the unbiased p̂ directly, not log p̂ |
| **Correlated pseudo-marginal** | Deligiannidis et al. (2018) — correlates successive likelihood estimates |
| **Bias correction terms** | Poyiadjis et al. (2011) — O(N) bias-corrected score estimator |

## Practical implications for this project

1. **MAP estimates will be biased** by ~1-10% depending on the model and N. This is expected, not a bug.
2. **Increasing N is the most reliable way to reduce bias.** For benchmarking, use N ≥ 1000.
3. **For SV2D with N=1000, expect ~5-10% bias** in the MAP estimate. This is consistent with what we observe (sigma2 converges to ~0.90 instead of 1.0).
4. **HMC is more robust to the bias** than MAP because the Metropolis accept/reject step corrects for it in the stationary distribution. The bias only affects the proposal quality (acceptance rate), not the asymptotic correctness.
5. **Comparing filters**: When benchmarking BPF vs LEDH on the same model, the filter with lower weight variance will have smaller bias → MAP estimate closer to truth. This is a valid comparison metric.
