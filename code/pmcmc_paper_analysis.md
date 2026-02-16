# PMCMC Paper Analysis — Andrieu, Doucet & Holenstein (2010)

*"Particle Markov chain Monte Carlo methods", J. R. Statist. Soc. B, 72, Part 3, pp. 269–342*

## Part 1: PMMH as a Benchmark

**What it is**: Particle Marginal Metropolis-Hastings. A random-walk Metropolis sampler where the intractable likelihood p(y|θ) is replaced by the particle filter's unbiased estimate p̂(y|θ).

**Algorithm** (for symmetric random-walk proposal):

```
Initialize: set θ(0), run particle filter → p̂_θ(0)(y)
For i = 1, 2, ...:
  1. Propose θ* = θ(i-1) + ε,  ε ~ N(0, Σ_proposal)
  2. Run particle filter with θ* → p̂_θ*(y)   [FRESH random seed]
  3. Accept with probability:
       min(1,  p̂_θ*(y) · p(θ*)  /  p̂_θ(i-1)(y) · p(θ(i-1)) )
  4. If accepted: θ(i) = θ*, p̂(i) = p̂_θ*(y)
     Otherwise:  θ(i) = θ(i-1), p̂(i) = p̂(i-1)   [keep OLD estimate]
```

**No gradients needed.** Only the particle filter likelihood estimate (which we already compute).

### Key theoretical results

- **Theorem 2**: PMMH is a standard IMH update on an extended space. It leaves the correct posterior invariant for **any N ≥ 1** particles.
- **Theorem 3**: The PMMH sampler is ergodic — it converges to the correct posterior.
- **Theorem 1, eq. 28**: Likelihood estimate variance scales as Var(Ẑ^N / Z) ≤ C·P / N (linear in time steps, inverse in particles).

### Their exact settings for the Kitagawa model

Section 3.1 (p. 280–283) uses **exactly our model**:

$$X_n = \frac{X_{n-1}}{2} + 25\frac{X_{n-1}}{1 + X_{n-1}^2} + 8\cos(1.2n) + V_n, \quad V_n \sim \mathcal{N}(0, \sigma_V^2)$$
$$Y_n = \frac{X_n^2}{20} + W_n, \quad W_n \sim \mathcal{N}(0, \sigma_W^2)$$

| Setting | Value |
|---------|-------|
| True params | σ_V² = 10, σ_W² = 1 |
| Parameterization | σ_V² and σ_W² (**variances**, not std devs) |
| Priors | Inverse Gamma: σ_V² ~ IG(0.01, 0.01), σ_W² ~ IG(0.01, 0.01) |
| T (observations) | 500 |
| N (particles) | 5000 for main results, 200–2000 in acceptance rate study |
| Resampling | Multinomial (simplest possible) |
| Proposal | Normal random walk, diagonal covariance |
| Proposal std dev | 0.15 for σ_V, 0.08 for σ_W |
| MCMC iterations | 50,000 |
| Burn-in | 10,000 |
| Initial values | σ_V(0)² = σ_W(0)² = 10 |

### Acceptance rate vs. N (from Fig. 3, for T = 100)

| N particles | Acceptance rate |
|------------|----------------|
| 100 | ~10% |
| 200 | ~27% |
| 500 | ~50% |
| 1000 | ~65% |
| 2000 | ~80% |

### Why PMMH sidesteps our gradient cliff problem

The acceptance ratio uses only the **ratio** of two likelihood estimates. Even though each estimate is noisy, the Metropolis correction handles this correctly. No gradient computation at all — the landscape's cliffs and discontinuities are irrelevant.

### Trade-off vs. HMC

Each PMMH step is cheap (one particle filter run, no gradient), but mixing is slow (random walk). HMC steps are expensive (multiple gradient evaluations through the filter) but can make large moves. The paper needs ~50,000 PMMH iterations with N=5000, but they are using a basic bootstrap particle filter which is much cheaper per step than our LEDH flow filter.

### Critical implementation detail: seed handling

- **HMC**: Uses FIXED seed (`seed=[42,0]`) — makes the likelihood surface deterministic for gradient computation.
- **PMMH**: Uses FRESH seed each evaluation — stochasticity is part of the algorithm, not a bug.
- On rejection, keep the OLD likelihood estimate (don't re-evaluate with new randomness).

---

## Part 2: Useful Tricks for Our Approach

### Trick 1 — Fresh randomness each MCMC step

The PMMH algorithm runs a **new, independent** particle filter at each iteration with fresh random numbers. The stochasticity is built into the theory.

Our HMC uses a **fixed seed** to make the likelihood deterministic for gradients. This is correct but means we optimize over one specific realization. Different seeds give different likelihood surfaces — this partly explains the randomness observed ("wrong for arbitrary 50 out of 100 every time I run the filter").

*Possible use*: After HMC converges with a fixed seed, do a final Metropolis-with-fresh-seeds stage to "average over" the particle filter randomness.

### Trick 2 — Parameterization matters

The paper uses **variances** (σ²) with **Inverse Gamma** priors, not standard deviations (σ) with LogNormal priors. IG(a,b) is the conjugate prior for Gaussian variance — the posterior is better-behaved.

Our setup: σ_V and σ_W with Exp bijector (log-space HMC). The gradient of log p(y|σ) w.r.t. log(σ) involves an extra chain rule factor of σ, which amplifies gradient magnitudes for large σ. Switching to σ² parameterization might give a smoother landscape.

### Trick 3 — The "one at a time" MH trap (p. 283)

The paper explicitly warns that standard block-wise MCMC (updating latent states x_{1:T} one block at a time, then updating θ) "tends to become trapped in a local mode of the multimodal posterior...and results in an overestimation of the true value of σ_V." This happened on most runs.

PMMH and Particle Gibbs did NOT have this trapping problem. Our difficulty might be partly a multi-modality issue, not just a gradient cliff issue.

### Trick 4 — Proposal scale calibration

For σ_V ≈ 3.16 (√10) and σ_W = 1.0, they use proposal std of 0.15 and 0.08 respectively. That's roughly **5% of the parameter value** for σ_V and **8% for σ_W**. Very conservative.

For our custom HMC with gradient clipping, this suggests the effective step size should produce moves of similar magnitude — a few percent of the parameter value per step.

### Trick 5 — Likelihood variance scales linearly with T (Theorem 1)

$$\text{Var}(\hat{Z}^N / Z) \leq \frac{C \cdot T}{N}$$

Doubling T requires doubling N to maintain the same acceptance rate. For T=100, N=1000 should work. For T=500 (their setting), they needed N=5000.

### Trick 6 — Adaptive proposal covariance (p. 283)

They use adaptive MCMC (Andrieu & Thoms, 2008) to learn the proposal covariance from a preliminary run. For our custom HMC, this translates to: run a short preliminary chain, estimate the posterior covariance, and use it to precondition the mass matrix.

### Trick 7 — Reuse rejected particles (Section 4.6, Theorem 6)

Even when a PMMH proposal is rejected, the N particles generated for that proposal are valid samples from p_θ*(x|y). These can be recycled for computing posterior expectations, effectively getting free samples.

---

## Summary: What to Try Next

| Priority | Action | Effort | Expected impact |
|----------|--------|--------|----------------|
| 1 | Test custom HMC with grad clipping on CUDA | Low — already written | Should get >0% acceptance |
| 2 | Implement PMMH as gradient-free benchmark | Medium | Avoids gradient cliff entirely |
| 3 | Try σ² parameterization with IG prior | Medium | Smoother gradient landscape |
| 4 | Reduce T from 100 to 50 to test | Low — config change | Reduces likelihood variance |
| 5 | Adaptive mass matrix from short pilot run | Medium | Better-conditioned HMC |
