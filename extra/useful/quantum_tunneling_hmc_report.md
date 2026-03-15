# Breaking Artificial Potential Barriers in HMC via Quantum Tunneling Ideas

## The Problem

In Hamiltonian Monte Carlo (HMC), the leapfrog integrator creates **artificial energy barriers** from discretization error. Even without discretization issues, HMC fundamentally follows classical trajectories that **cannot cross regions where U(q) >> E** (the total energy). The particle simply bounces back.

This is particularly problematic for:
- Multimodal distributions separated by low-probability regions
- Particle filters where HMC kernels are used to diversify/rejuvenate particles
- High-dimensional settings where barriers become increasingly difficult to cross

## The Quantum Tunneling Analogy

In quantum mechanics, a particle's wavefunction decays exponentially inside a barrier but doesn't vanish — it has nonzero amplitude on the other side. The tunneling probability scales as:

$$P_{\text{tunnel}} \propto \exp\!\Bigl(-\frac{2}{\hbar}\int_a^b \sqrt{2m\bigl(V(x)-E\bigr)}\,dx\Bigr)$$

The key insight: **quantum particles explore classically forbidden regions**. Several MCMC methods borrow this idea.

---

## Approach 1: Wavefunction Ansatz / Quantum Potential (Bohm-inspired)

Add a **quantum potential** to the Hamiltonian:

$$Q(q) = -\frac{\hbar^2}{2m}\frac{\nabla^2 \sqrt{\rho(q)}}{\sqrt{\rho(q)}}$$

This is the de Broglie-Bohm quantum potential. It effectively **repels particles from high-density regions and attracts them to low-density regions**, counteracting barrier trapping.

### Challenges
- Requires knowledge of ρ(q), which is the quantity we're trying to estimate
- In practice, need a density estimate (e.g., kernel density from the particle ensemble)
- Gradient of the quantum potential can be noisy

### References
- Bohm, D. (1952). A Suggested Interpretation of the Quantum Theory in Terms of "Hidden" Variables.
- Related to score-matching approaches in modern generative modeling

---

## Approach 2: Path Integral Monte Carlo (Ring Polymer)

From Feynman's path integral formulation: replace each sample with a **ring polymer** of P replicas connected by harmonic springs:

$$U_{\text{eff}} = \sum_{k=1}^{P}\Bigl[\frac{P}{2\beta^2\hbar^2}|q_k - q_{k+1}|^2 + \frac{1}{P}U(q_k)\Bigr]$$

The replicas can **collectively stretch across a barrier** even when no single replica has enough energy to cross it. Anneal ℏ → 0 to recover the classical target distribution.

### How it works
1. Each particle is replaced by a "necklace" of P beads
2. Adjacent beads are connected by harmonic springs (strength ∝ P/ℏ²)
3. The ring polymer can span across barriers — some beads are on one side, some on the other
4. As ℏ → 0, the beads collapse to a single point (classical limit)

### Trade-offs
- High effectiveness for deep, narrow barriers
- Computational cost scales with P (number of beads)
- Annealing schedule for ℏ needs tuning

---

## Approach 3: Non-Gaussian (Heavy-Tailed) Momentum

Replace the Gaussian momentum distribution p ~ N(0, M) with a **heavy-tailed distribution** like Student-t:

$$p \sim t_\nu(0, M)$$

This occasionally generates very large momenta that give the particle enough kinetic energy to **punch through** barriers. It is the cheapest approximation to tunneling — rare large fluctuations mimic the exponential tail of a wavefunction inside a barrier.

### Implementation
```python
# Standard HMC momentum
p = tf.random.normal(shape, dtype=dtype) @ tf.linalg.cholesky(M)

# Heavy-tailed momentum (Student-t with nu degrees of freedom)
nu = 4.0  # degrees of freedom; lower = heavier tails
z = tf.random.normal(shape, dtype=dtype)
chi2 = tf.random.gamma(shape=[1], alpha=nu/2, beta=nu/2)
p = (z / tf.sqrt(chi2)) @ tf.linalg.cholesky(M)
```

### Parameter guidance
- ν = 3–5: moderate tunneling, good for mildly separated modes
- ν = 1–2: aggressive tunneling (Cauchy-like), may reduce acceptance rate
- ν → ∞: recovers standard Gaussian HMC

### Trade-offs
- Extremely easy to implement (one-line change)
- No additional hyperparameters beyond ν
- May reduce acceptance rate due to energy conservation violations
- Metropolis correction still ensures correctness

---

## Approach 4: Parallel Tempering / Replica Exchange

Run HMC at multiple **inverse temperatures** β₁ < β₂ < ... < βK:

$$U_\beta(q) = \beta \cdot U(q)$$

At low β (high temperature), barriers shrink — analogous to increasing ℏ in quantum mechanics. Propose **swaps** between adjacent temperature levels.

### Algorithm
1. Maintain K chains at temperatures T₁ > T₂ > ... > TK (where TK = 1 is the target)
2. Run HMC independently on each chain for some steps
3. Propose swap between adjacent chains (i, i+1) with acceptance probability:
   $$\alpha = \min\bigl(1,\, \exp\bigl((\beta_i - \beta_{i+1})(U(q_i) - U(q_{i+1}))\bigr)\bigr)$$
4. Collect samples only from the coldest chain (β = 1)

### Connection to quantum tunneling
- High temperature = large ℏ → enhanced tunneling
- Swap mechanism = measurement / wavefunction collapse to target distribution
- Temperature ladder = adiabatic path from quantum to classical regime

### Trade-offs
- Very effective for well-separated modes
- Requires K times the computational cost
- Temperature ladder spacing is critical — too sparse and swaps never accept

---

## Approach 5: Stochastic Tunneling

Transform the potential before running dynamics:

$$\tilde{U}(q) = 1 - e^{-\gamma(U(q) - U_{\min})}$$

This **compresses** high barriers while preserving minima locations. Run dynamics on Ũ, then correct with a Metropolis step against the original U.

### How it works
- For U(q) near U_min: Ũ ≈ γ(U(q) - U_min) (linear, preserves local structure)
- For U(q) >> U_min: Ũ → 1 (saturates, barrier is capped at height 1)
- The parameter γ controls how aggressively barriers are flattened

### Implementation
```python
def stochastic_tunneling_potential(U, U_min, gamma):
    """Transform potential to flatten barriers."""
    return 1.0 - tf.exp(-gamma * (U - U_min))

def stochastic_tunneling_gradient(U, grad_U, U_min, gamma):
    """Gradient of transformed potential."""
    return gamma * tf.exp(-gamma * (U - U_min)) * grad_U
```

### Trade-offs
- Easy to implement
- Need to estimate U_min (can use running minimum)
- γ too large → reverts to original potential; γ too small → loses local structure
- Metropolis correction against original U ensures correctness but may reduce acceptance

---

## Comparison Summary

| Approach | Ease of Implementation | Effectiveness | Computational Overhead | Best For |
|----------|----------------------|---------------|----------------------|----------|
| Heavy-tailed momentum | Very Easy | Moderate | Negligible | Mildly separated modes |
| Stochastic tunneling | Easy | Moderate | Negligible | Known barrier structure |
| Parallel tempering | Medium | High | K× cost (K = num temps) | Well-separated modes |
| Quantum potential (Bohm) | Hard | High | Density estimation cost | Ensemble methods / particle filters |
| Ring polymer (PIMC) | Hard | Very High | P× cost (P = num beads) | Deep narrow barriers |

---

## Recommendations for Particle Filter HMC Kernels

In the context of differentiable particle filters with HMC-based rejuvenation kernels:

1. **Start with heavy-tailed momentum** (Student-t, ν ≈ 3–5). This is a drop-in replacement requiring minimal code changes and no additional hyperparameters beyond ν.

2. **If insufficient, try stochastic tunneling**. Transform the log-weights before computing gradients, and correct with Metropolis. This is particularly natural in particle filters where you already track log-weights.

3. **Consider the quantum potential approach** if you have a particle ensemble. The ensemble itself provides a natural density estimate via kernel density estimation, making the Bohm potential more tractable than in standard MCMC.

4. **Parallel tempering is expensive** but may be warranted if modes are truly well-separated and other approaches fail.

---

## Further Reading

- Neal, R. M. (2011). MCMC using Hamiltonian dynamics. *Handbook of Markov Chain Monte Carlo*.
- Zhang, Y. et al. (2016). Quantum-inspired Hamiltonian Monte Carlo.
- Swendsen, R. H. & Wang, J.-S. (1986). Replica Monte Carlo simulation of spin-glasses. *Physical Review Letters*.
- Wenzel, W. & Hamacher, K. (1999). Stochastic tunneling approach for global minimization of complex potential energy landscapes. *Physical Review Letters*.
- Craig, I. R. & Manolopoulos, D. E. (2004). Quantum statistics and classical mechanics: Real time correlation functions from ring polymer molecular dynamics. *Journal of Chemical Physics*.
