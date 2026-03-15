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

### Model (Section 3.1, p. 280, eq. 14–15)

$$X_n = \frac{X_{n-1}}{2} + 25\frac{X_{n-1}}{1 + X_{n-1}^2} + 8\cos(1.2n) + V_n, \quad V_n \sim \mathcal{N}(0, \sigma_V^2)$$
$$Y_n = \frac{X_n^2}{20} + W_n, \quad W_n \sim \mathcal{N}(0, \sigma_W^2)$$

where X₁ ~ N(0, 5). True parameters: σ_V² = 10, σ_W² = 1. Parameterized as θ = (σ_V, σ_W).

### Three algorithms compared (Section 3.1, p. 283)

All three use the same data (T=500), same priors (σ_V² ~ IG(0.01, 0.01), σ_W² ~ IG(0.01, 0.01)), same initial values (σ_V²(0) = σ_W²(0) = 10), and 50,000 MCMC iterations with 10,000 burn-in.

#### (a) Standard MH one-at-a-time (baseline)
- Updates x_{1:T} in blocks using MH with proposal density f_θ(x_n | x_{n-1})
- Updates N state variables at each iteration, then updates θ
- **Result**: Gets trapped in a local mode on most runs. Overestimates σ_V.

#### (b) Particle Gibbs (PG) sampler (Section 2.4.3, p. 278)

| Setting | Value |
|---------|-------|
| N (particles) | 5000 |
| SMC proposal | Prior: q(x_n \| y_n, x_{n-1}) = f(x_n \| x_{n-1}) |
| Resampling | Stratified |
| θ-step | Sample from full conditional p(θ \| x_{1:T}, y_{1:T}) |
| Initial values | σ_V²(0) = σ_W²(0) = 10 |

The PG θ-step (p. 278, step 2a) says: "sample θ(i) ~ p{·\|y_{1:T}, X_{1:T}(i-1)}". With IG priors on σ² and Gaussian likelihoods, this is a **conjugate update** — direct sample from the posterior IG, no MH needed:
- σ_V² \| x, y ~ IG(a + T/2, b + 0.5 · Σ(x_t - f(x_{t-1}))²)
- σ_W² \| x, y ~ IG(a + T/2, b + 0.5 · Σ(y_t - x_t²/20)²)

The x-step uses conditional SMC (CSMC) with a reference trajectory pinned (Section 2.4.3).

**Result**: Never trapped. Both σ_V and σ_W converge correctly (Fig. 4b).

#### (c) PMMH sampler (Section 2.4.2, p. 277)

| Setting | Value |
|---------|-------|
| N (particles) | 5000 |
| SMC proposal | Prior |
| Resampling | Stratified |
| θ-step | Random-walk MH: proposal std 0.15 (σ_V), 0.08 (σ_W) |
| Initial values | σ_V²(0) = σ_W²(0) = 10 |

Each iteration runs a full particle filter with a **fresh random seed**. Accepts/rejects based on the ratio of likelihood estimates (eq. 13). On rejection, keeps the old estimate.

**Result**: Never trapped. Both σ_V and σ_W converge correctly (Fig. 4c).

### Key result (p. 283)

"the MH one at a time update appears to mix well...However, this algorithm tends to become trapped in a local mode of the multimodal posterior...and results in an overestimation of the true value of σ_V. This occurred on most runs when using initializations from the prior for X_{1:T}. **Using the same initial values, the PMMH and the PG samplers never became trapped in this local mode.**"

### ACF comparison (Fig. 5, p. 282)

PG ACF decays slower than PMMH ACF. Need N ≥ 2000 particles for PG to have reasonable ACF. With N=5000, both PG and PMMH ACFs are similar and close to the idealized MMH.

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

The paper uses **variances** (σ²) with **Inverse Gamma** priors. IG is the conjugate prior for Gaussian variance, so the PG θ-step is a direct sample (no MCMC needed for θ). Any valid MCMC kernel (MH, HMC) also works for the θ-step — convergence is guaranteed regardless — but conjugate sampling is more efficient.

Our setup: σ_V and σ_W (std devs) with LogNormal priors and Softplus bijector. Not conjugate, so we use MH or HMC for the θ-step. This is valid but slower to mix.

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

## Our setup vs. paper

| | Paper (PG) | Our PGibbs |
|---|---|---|
| T | 500 | 100 |
| N particles | 5000 | 500 (LEDH) / 1000 (BPF) |
| MCMC iterations | 50,000 | 3,000 |
| Burn-in | 10,000 | 1,000 |
| θ-step | Conjugate IG (exact) | MH or HMC (approximate) |
| Priors | IG(0.01, 0.01) on σ² | LogNormal on σ |
| x-step | Bootstrap CSMC | LEDH CSMC or Bootstrap CSMC |
| Resampling | Stratified | Systematic |

The paper uses 5x more data, 10x more particles, 17x more iterations, and exact conjugate θ-updates. Our LEDH CSMC x-step should produce better proposals than their bootstrap CSMC, partially compensating for fewer particles.

---

## What "Conjugate IG" Actually Means

The paper's PG sampler updates σ_V² and σ_W² by **sampling directly** from the posterior — no MH, no HMC, no accept/reject. Here's why that works.

### The trick: IG prior × Gaussian likelihood = IG posterior

The Inverse-Gamma (IG) distribution is the "matching" prior for the variance of a Gaussian. When you multiply them, the result is another IG — just with updated numbers. That's all "conjugate" means: **the math simplifies so the answer is a known distribution you can sample from directly**.

### Step-by-step for σ_V²

**Prior:** We believe σ_V² comes from IG(α₀, β₀) with α₀ = β₀ = 0.01 (vague — we barely constrain it).

**Data:** Given a trajectory x₀, x₁, ..., x_T, we can compute the transition residuals — how far each state is from what the dynamics predicted:

    e_t = x_t − f(x_{t−1})    for t = 1, ..., T

These residuals are Gaussian with variance σ_V² (by definition of the model).

**Posterior:** The IG prior "absorbs" the Gaussian likelihood and stays IG:

    σ_V² | trajectory  ~  IG(α₀ + T/2,  β₀ + ½ Σ e_t²)

That's it. The posterior shape parameter is α₀ + T/2 (prior shape + half the data points), and the posterior rate is β₀ + ½ Σ e_t² (prior rate + half the sum of squared residuals).

To sample: just call `invgamma.rvs(a = α₀ + T/2, scale = β₀ + ½ Σ e_t²)`. One line. Done.

### Same for σ_W²

Observation residuals:

    r_t = y_t − x_t² / 20    for t = 1, ..., T

Posterior:

    σ_W² | trajectory, observations  ~  IG(α₀ + T/2,  β₀ + ½ Σ r_t²)

### Why we can't do this

We parameterize as σ_V and σ_W (std devs, not variances) with LogNormal priors. LogNormal is NOT conjugate to Gaussian — the math doesn't simplify to a known distribution. So we have to use MH or HMC to approximately sample from the theta-posterior, which is slower and requires tuning (step size, proposal std, etc.).

### Could we switch to conjugate?

Yes. If we reparameterized to σ² with IG priors, the theta-step would be instant and exact. The trade-off: IG priors on variance are less intuitive than LogNormal priors on std dev, and we'd lose the ability to use HMC for theta (though we wouldn't need it). For the Kitagawa model specifically, this would be a strict improvement.

### Hidden advantage in the paper's results

The paper buries the conjugate IG update as an implementation detail — step 2a just says "sample θ from conditional" without elaboration. But this is a massive hidden advantage:

- **100% acceptance** on every theta-step (exact draw, not a proposal)
- **Zero tuning** (no step size, no proposal std, no leapfrog steps to pick)
- **Instant computation** (one sum of squares + one random draw)

The impressive PG results in Fig. 4b (never trapped, correct convergence) benefit heavily from this. If they had used random-walk MH for the theta-step — which is what you'd need for any model without conjugate structure — the theta chain would mix much slower, require proposal tuning, and only achieve ~30-50% acceptance.

The paper never shows PG with a non-conjugate theta-step. So the reader can't tell how much of PG's performance comes from the algorithm itself vs. the conjugate shortcut. For our setup (LogNormal priors, MH/HMC theta-step), every theta-step is approximate and costly. Our PG is solving a strictly harder problem than what the paper demonstrates.
