# Particle Gibbs + HMC Implementation Plan

## 1. Background and Motivation

### The Problem

We want to infer static parameters $\theta = (\sigma_V, \sigma_W)$ from observations $y_{1:T}$ in state-space models like Kitagawa. Our current approach runs HMC on the marginal posterior $p(\theta | y)$ by differentiating through the particle filter's log-likelihood estimate $\hat{p}(y|\theta)$.

This fails for three reasons:
1. **Gradient cliffs**: The particle filter likelihood has discontinuities from resampling, producing NaN or extreme gradients that crash HMC's leapfrog integrator.
2. **Weight collapse on CUDA**: Numerical precision differences between MPS and CUDA cause accumulated Jacobian products (29 flow steps) to collapse to zero on the RTX 3090.
3. **Bimodality trapping**: The Kitagawa model's $y = x^2/20$ observation creates a bimodal posterior over latent trajectories. Block-wise MCMC (sample $x$, then $\theta$) gets trapped in one mode, biasing the $\theta$ estimate (Andrieu et al. 2010, Section 3.1).

### Two Orthogonal Tricks

PGibbs + CSMC solves these problems via **two independent tricks**:

| Trick | Mechanism | Solves |
|-------|-----------|--------|
| **1. Gibbs decomposition** | Separate $p(\theta, x \mid y)$ into $\theta$-step + $x$-step | **Efficiency**: $\theta$-step is HMC on smooth closed-form surface (no PF, no gradient cliffs, no weight collapse) |
| **2. Reference pinning (CSMC)** | Tag one particle to follow previous trajectory; always survives resampling | **Bimodality**: N-1 free particles explore other modes; reference anchors ensure ergodicity |

These are **orthogonal** — each can be used independently:
- Trick 1 alone: Gibbs with a regular (non-conditional) PF for x-step. Fixes gradient cliffs but not bimodality.
- Trick 2 alone: CSMC inside current HMC-through-PF. Fixes bimodality but not gradient cliffs.
- Both together: Full PGibbs + CSMC. Solves both.

### Applying to LEDH

The x-step PF can be **any filter** — bootstrap or LEDH:
- **Phase 1**: Bootstrap CSMC (`conditional_smc.py`) — simplest, validates the framework
- **Phase 2**: LEDH CSMC (`ledh_invertible_csmc.py`) — better proposals for free particles, needed for high-dim models

For LEDH CSMC: the flow applies only to N-1 free particles; the reference stays pinned (no flow). See `hmc_dpf_notes.md` for full math.

### The $\theta$-step Target (closed-form)

Given a fixed trajectory $x_{1:T}$, the log-posterior of $\theta$ is a simple sum of Gaussian log-densities — no PF needed:

$$\log p(\sigma_V, \sigma_W \mid x_{1:T}, y_{1:T}) \propto \sum_{t=1}^{T} \log \mathcal{N}(x_t; f(x_{t-1}, t), \sigma_V^2) + \sum_{t=1}^{T} \log \mathcal{N}(y_t; x_t^2/20, \sigma_W^2) + \log \text{prior}(\sigma_V, \sigma_W)$$

This is smooth, cheap, and differentiable. Gradients are closed-form sums.

---

## 2. Conditional SMC (CSMC) — Detailed Algorithm

CSMC is a modified particle filter that takes a **reference trajectory** $x^*_{1:T}$ (from the previous MCMC iteration) and produces a new trajectory sample from $p(x_{1:T} | \theta, y_{1:T})$.

### Algorithm

```
Input: observations y_{1:T}, parameters theta, reference trajectory x*_{1:T}, N particles

t = 0 (initialization):
  - Sample particles x^i_0 ~ p(x_0) for i = 1, ..., N-1
  - Set x^N_0 = x*_0                    <-- reference is "pinned" as particle N
  - Set weights w^i_0 = 1/N for all i

For t = 1, ..., T:
  1. PROPAGATE:
     - For i = 1, ..., N-1: sample x^i_t ~ p(x_t | x^{a_i}_{t-1}, theta)
     - Set x^N_t = x*_t                  <-- reference follows its prescribed path

  2. WEIGHT:
     - For all i: w^i_t ∝ p(y_t | x^i_t, theta)
     - Normalize weights: w^i_t = w^i_t / sum_j w^j_t

  3. RESAMPLE (with reference protection):
     - Resample ancestor indices {a_1, ..., a_{N-1}} from weights
     - Set a_N = N                        <-- reference always survives
     - Remap particles: x^i_t = x^{a_i}_t for i = 1, ..., N-1

At end (t = T):
  - Sample one trajectory index k ~ Categorical(w^1_T, ..., w^N_T)
  - Trace back the ancestry of particle k to get the full trajectory x^k_{1:T}
  - This trajectory becomes x*_{1:T} for the next MCMC iteration
```

### Why CSMC handles bimodality

The N-1 free particles explore both modes of the posterior. The reference trajectory anchors the sampler (guaranteeing ergodicity), while the free particles can discover the alternative mode. When sampling the output trajectory at $t = T$, the sampler might jump to a particle that took a completely different path.

### Ancestor sampling extension (Lindsten et al. 2014)

Without ancestor sampling, CSMC suffers from **path degeneracy**: the reference trajectory tends to be re-selected because all other particles share a common ancestor early in time. This causes slow mixing.

**Ancestor sampling** adds one extra step at each $t$: resample the ancestor of the reference particle itself. At time $t$:
- For each candidate ancestor $j \in \{1, \ldots, N\}$, compute:
  $$w^j_{t|T} \propto w^j_{t-1} \cdot p(x^*_t | x^j_{t-1}, \theta)$$
- Sample $a_N \sim \text{Categorical}(w^1_{t|T}, \ldots, w^N_{t|T})$

This allows the reference trajectory's history to "detach" and graft onto any particle's past, dramatically improving mixing. The cost is N extra transition density evaluations per timestep.

---

## 3. Full PGibbs + HMC Algorithm

```
Initialize:
  theta^(0) = (sigma_V_init, sigma_W_init)
  x^(0)_{1:T} = run bootstrap PF with theta^(0), pick one trajectory

For iteration m = 1, 2, ..., M:

  1. THETA-STEP (HMC on smooth surface):
     - Target: log p(theta | x^(m-1)_{1:T}, y_{1:T})
     - This is a closed-form sum of Gaussian log-densities (see Section 2)
     - Run L leapfrog steps with step size eps
     - Accept/reject via standard Metropolis
     -> theta^(m)

  2. X-STEP (Conditional SMC):
     - Run CSMC with theta^(m) and reference x^(m-1)_{1:T}
     - Uses ancestor sampling for better mixing
     - Sample output trajectory
     -> x^(m)_{1:T}
```

---

## 4. Implementation Architecture

### 4.1. New file: `code/src/DF/pgibbs_runner.py`

This is the main orchestrator. Follows the same pattern as `PMMHRunner` and `DPFRunner`.

```python
class PGibbsRunner:
    """
    Particle Gibbs + HMC for joint inference of parameters and latent states.

    Alternates:
      1. HMC step for theta | x_{1:T}, y_{1:T}  (smooth closed-form target)
      2. CSMC step for x_{1:T} | theta, y_{1:T}  (particle filter, no gradients)
    """

    def __init__(
        self,
        base_model,           # StateSpaceModel instance
        filter_class,         # PF class for CSMC (e.g., BootstrapPF or LEDH)
        filter_kwargs,        # kwargs for filter construction
        param_specs,          # Dict[str, ParameterSpec]
        csmc_n_particles=500, # N for the CSMC step
        ancestor_sampling=True,
        hmc_num_leapfrog=10,
        hmc_step_size=0.01,
    ):
        ...

    def _theta_log_posterior(self, theta_unconstrained, trajectory, observations):
        """
        Closed-form log p(theta | x_{1:T}, y_{1:T}).
        No particle filter. Just sum of Gaussian log-densities.
        """
        ...

    def _theta_step_hmc(self, trajectory, observations, current_theta):
        """
        Run HMC for theta given fixed trajectory.
        Uses tf.GradientTape on _theta_log_posterior.
        """
        ...

    def _x_step_csmc(self, observations, theta, reference_trajectory):
        """
        Run Conditional SMC to sample new trajectory.
        """
        ...

    def run_inference(
        self,
        observations,
        num_samples=1000,
        num_burnin=500,
        hmc_steps_per_iter=1,   # HMC steps per Gibbs iteration
        seed=42,
    ) -> DPFResult:
        ...
```

### 4.2. CREATED: `code/src/filters/particle/conditional_smc.py`

Bootstrap CSMC (Phase 1). Class: `BootstrapConditionalSMC`.

Key features:
- Bootstrap PF proposals (transition density) — weight = likelihood only
- Reference particle (index N-1) pinned to previous trajectory
- Ancestor sampling for better mixing
- Full genealogy storage for trajectory reconstruction

### 4.2b. CREATED: `code/src/filters/particle/ledh_invertible_csmc.py`

LEDH CSMC (Phase 2). Class: `LEDHConditionalSMC`.

Key features:
- LEDH flow applied to N-1 free particles only (reference skipped)
- Heterogeneous weights: free particles use full LEDH weight (Jacobian + transition ratio), reference uses likelihood only
- Per-particle EKF covariance tracking (same as ledh_invertible.py)
- Ancestor sampling + reference protection

Both classes share the same interface:

```python
def run(self, observations, reference_trajectory, seed):
    """
    Args:
        observations: (T, obs_dim)
        reference_trajectory: (T+1, state_dim) — x*_{0:T}
        seed: int

    Returns:
        new_trajectory: (T+1, state_dim) sampled from CSMC output
    """
        ...

    def _ancestor_sampling_weights(self, particles, ref_state_next, theta, t):
        """Compute ancestor sampling weights for reference particle."""
        ...
```

### 4.3. The theta-step log-posterior (closed-form)

This is the mathematical heart of the approach. For a generic state-space model:

```python
def _theta_log_posterior(self, unconstrained_params, trajectory, observations):
    """
    log p(theta | x_{1:T}, y_{1:T}) ∝
        sum_t log p(x_t | x_{t-1}, theta)   [transition terms]
      + sum_t log p(y_t | x_t, theta)       [observation terms]
      + log prior(theta)

    All terms are Gaussian log-densities with known mean/variance,
    parameterized by theta. Fully differentiable, no particle filter.
    """
    # 1. Transform to constrained space
    constrained = self.param_handler.constrain(unconstrained_params)
    self.diff_model.update_parameters(constrained)

    T = observations.shape[0]
    log_prob = tf.constant(0.0, dtype=self.dtype)

    # 2. Initial state term: log p(x_0)
    x_0 = trajectory[0]
    mu_0 = self.diff_model.mu_0
    Sigma_0 = self.diff_model.Sigma_0
    log_prob += _log_gaussian(x_0, mu_0, Sigma_0)

    # 3. Transition terms: sum_t log p(x_t | x_{t-1}, theta)
    for t in range(1, T + 1):
        x_prev = trajectory[t - 1]
        x_curr = trajectory[t]  # trajectory is (T+1, sd) including x_0
        self.diff_model.t = t
        mean = self.diff_model.state_transition_mean(x_prev, t=t)
        Q = self.diff_model.process_noise_cov   # depends on sigma_V
        log_prob += _log_gaussian(x_curr, mean, Q)

    # 4. Observation terms: sum_t log p(y_t | x_t, theta)
    for t in range(T):
        x_t = trajectory[t + 1]
        y_t = observations[t]
        log_prob += self.diff_model.log_observation_prob(y_t, x_t)

    # 5. Prior + Jacobian adjustment
    log_prob += self.param_handler.log_prior(constrained)

    return log_prob
```

Note: The `for` loops here are fine because they're tracing through simple Gaussian log-densities (no PF), and T is typically 100. For T > 1000, these should be vectorized (see Section 7).

### 4.4. Integration with existing infrastructure

| Existing component | How PGibbs uses it | Changes needed |
|---|---|---|
| `ParameterHandler` | Bijectors for theta-step HMC | None |
| `DifferentiableModel` | Wraps model for theta-step gradients | None |
| `DPFResult` | Return type | None |
| `run_dpf_experiment.py` | Entry point | Add `sampler: pgibbs` branch |
| `DF/__init__.py` | Exports | Add `PGibbsRunner` |
| Model classes (Kitagawa etc.) | Used by both CSMC and theta-step | None — existing `state_transition_mean`, `log_observation_prob`, `process_noise_cov` etc. are sufficient |
| Resampling (systematic) | Used by CSMC | None |

### 4.5. Config file: `code/configs/dpf/kitagawa_pgibbs_hmc.yaml`

```yaml
# @package _global_
# PGibbs + HMC: Particle Gibbs with HMC theta-step

model:
  _target_: src.models.kitagawa.KitagawaModel
  sigma_V: 5.0       # Initial guess (true: 10.0)
  sigma_W: 2.0       # Initial guess (true: 1.0)
  initial_var: 5.0

filter:
  # Filter used for CSMC x-step (bootstrap PF is sufficient)
  _target_: src.filters.particle.bootstrap_pf_tf.ParticleFilterTF
  n_particles: 1000
  resampling_method: systematic
  resample_threshold: 0.5

dpf:
  sampler: pgibbs

  trainable_params:
    sigma_V:
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 2.3
        scale: 0.5
    sigma_W:
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 0.0
        scale: 1.0

  pgibbs:
    num_samples: 2000
    num_burnin: 1000
    csmc_n_particles: 1000
    ancestor_sampling: true
    hmc_steps_per_iter: 5    # multiple HMC steps per Gibbs sweep
    hmc_step_size: 0.05      # can be larger — smooth surface
    hmc_num_leapfrog: 10
    target_accept_prob: 0.8
    seed: 42

data:
  T: 100
  seed: 42
  true_params:
    sigma_V: 10.0
    sigma_W: 1.0
```

---

## 5. Relation to the Look-Ahead Sign Correction Plan

The `kitagawa_ledh_improvement.md` plan addresses the same bimodality problem from a **different angle**. Here is how the two plans relate:

### Different levels of the problem

| Aspect | Look-Ahead Sign Correction | Particle Gibbs + HMC |
|--------|---------------------------|---------------------|
| **Goal** | Better filtering (tracking $x_{1:T}$) | Better parameter inference (estimating $\theta$) |
| **Problem addressed** | LEDH flow collapses to one mode because Daum-Huang drift is unimodal | HMC through the particle filter has gradient cliffs and weight collapse |
| **Bimodality handling** | Uses future observations $y_{t+1:t+K}$ to discriminate $+x$ vs $-x$ | CSMC explores both modes via free particles + reference anchoring |
| **Where it operates** | Inside the filter's `update()` step | Around the filter (MCMC wrapper) |
| **Needs gradients** | No (heuristic flip) | Only for the theta-step (which is PF-free) |

### Are they complementary or redundant?

**They are complementary, and can be combined.** The CSMC x-step in PGibbs uses a particle filter to sample trajectories. Any particle filter can be used here — bootstrap PF, LEDH, or LEDH-with-look-ahead.

- **CSMC with bootstrap PF**: Simplest. Works for the Kitagawa model with enough particles (N >= 1000). The bootstrap PF naturally maintains multiple modes because it doesn't impose a unimodal flow. Recommended as the starting point.

- **CSMC with LEDH**: Could be beneficial for higher-dimensional models where the bootstrap PF struggles, but the unimodal LEDH flow would still collapse modes within the CSMC step. The reference trajectory protection keeps one mode alive, but the free particles might all collapse to the flow's preferred mode.

- **CSMC with LEDH + look-ahead**: The best of both worlds for bimodal models. The look-ahead correction helps the free particles in CSMC explore both modes, while the reference trajectory guarantees ergodicity. This combination would give the best trajectory mixing.

### Recommendation

1. **Phase 1**: Implement PGibbs with bootstrap PF (this plan). This solves the parameter inference problem (gradient cliffs, weight collapse) without needing the look-ahead.
2. **Phase 2**: If CSMC mixing is slow (reference trajectory dominates, slow mode switching), swap in the LEDH + look-ahead filter as the CSMC proposal. This would improve trajectory exploration.

The look-ahead plan is **not needed** for PGibbs to work. It's an optimization for the CSMC step that would help when:
- The model is higher-dimensional (bootstrap PF inefficient)
- T is large and path degeneracy is severe
- The bimodality is strong and symmetric (both modes equally likely)

---

## 6. Implementation Order

### Step 1: Conditional SMC (`conditional_smc.py`)

New file. The CSMC filter is the core algorithmic component.

Key implementation details:
- **Trajectory storage**: Need to store full particle genealogy (ancestry) to trace back trajectories at the end. Use a `(T, N, state_dim)` tensor for particles and `(T, N)` integer tensor for ancestor indices.
- **Reference pinning**: At each resampling step, force the reference particle's ancestor to be itself (or the ancestor-sampled index).
- **Ancestor sampling**: Compute transition log-probabilities from all particles to the reference's next state. This requires `model.log_transition_prob(x_next, x_prev)` — which we can compute from `state_transition_mean` and `process_noise_cov`:
  ```
  log p(x_next | x_prev) = log N(x_next; f(x_prev, t), Q)
  ```
  This method doesn't exist on the model base class but can be computed from existing methods. Add it as a utility function in the CSMC class rather than modifying the model.

### Step 2: Closed-form theta log-posterior

Add to `PGibbsRunner`. This is straightforward — iterate over the trajectory, sum Gaussian log-densities. Key detail: must handle `model.t` correctly for Kitagawa's time-dependent transition.

Vectorized version for efficiency:
```python
# Transition terms (vectorized over T)
means = model.state_transition_mean_batch(trajectory[:-1], t=t_indices)  # (T, sd)
diffs = trajectory[1:] - means                                           # (T, sd)
Q = model.process_noise_cov                                              # (sd, sd)
# log p = -0.5 * (T * log|Q| + sum_t (x_t - f(x_{t-1}))^T Q^{-1} (x_t - f(x_{t-1})))
```

### Step 3: HMC theta-step

Reuse the leapfrog + dual averaging logic from `DPFRunner._run_custom_hmc`. The target is `_theta_log_posterior` instead of `_negative_log_posterior`. No gradient clipping needed (surface is smooth).

### Step 4: PGibbs runner main loop

Wire steps 1-3 together. The main loop alternates CSMC and HMC.

### Step 5: Experiment runner integration

Add `sampler: pgibbs` branch in `run_dpf_experiment.py`, following the pattern of the existing `pmmh` branch.

### Step 6: Config file

Create `kitagawa_pgibbs_hmc.yaml`.

---

## 7. Optimization Notes

### Vectorizing the theta-step

The naive for-loop in `_theta_log_posterior` calls `state_transition_mean` T times. For T=100 this is fine (~1ms). For T=500+, vectorize using `state_transition_mean_batch`:

```python
# Stack all x_{t-1} into a batch
x_prev_batch = trajectory[:-1]              # (T, sd)
x_curr_batch = trajectory[1:]               # (T, sd)
# Need time indices for Kitagawa
t_indices = tf.range(1, T + 1)
means = model.state_transition_mean_batch(x_prev_batch, t=t_indices)  # (T, sd)
```

This requires `state_transition_mean_batch` to accept a batch of time indices. Currently Kitagawa's implementation accepts a scalar `t`. A small modification is needed:
- Either loop in the runner (acceptable for T=100)
- Or add `t` broadcasting to `state_transition_mean_batch` (better for T=500+)

### CSMC filter choice

For the CSMC step, a **bootstrap PF is strongly preferred** over LEDH:
1. Bootstrap PF is cheap (one transition sample + one weight evaluation per particle per timestep)
2. LEDH has 29 flow steps per timestep — 29x more expensive
3. CSMC doesn't need gradients, so the flow's differentiability advantage is irrelevant
4. Bootstrap PF naturally maintains particle diversity (no unimodal flow collapse)
5. The reference trajectory + ancestor sampling handle mode exploration

### Multiple HMC steps per Gibbs iteration

Each Gibbs iteration should run **multiple HMC steps** (e.g., 5-10) for the theta-step. The theta-step is cheap (~1ms for T=100), while the CSMC step is expensive (~100ms for N=1000, T=100). Running multiple HMC steps amortizes the CSMC cost.

### Thinning

Store every k-th sample (e.g., k=5) to reduce autocorrelation. PGibbs chains are more autocorrelated than independent PMMH chains because adjacent samples share trajectory structure.

---

## 8. Expected Performance

### Comparison with current approaches

| Method | Gradient through PF? | Handles bimodality? | Per-step cost (T=100, N=1000) | Expected acceptance |
|--------|---------------------|--------------------|-----------------------------|-------------------|
| HMC (current) | Yes — gradient cliffs | No | ~5s (LEDH + backward) | 0-10% |
| PMMH | No | Yes (marginalizes x) | ~0.1s (BPF forward) | 15-30% |
| **PGibbs + HMC** | **No** | **Yes (CSMC)** | **~0.1s CSMC + ~0.01s HMC** | **60-80% (HMC on smooth surface)** |

### Why PGibbs + HMC should outperform PMMH

- PMMH uses random-walk proposals for $\theta$ — slow mixing, ~5% of parameter value per step
- PGibbs uses HMC for $\theta$ — can make large, directed moves along the smooth posterior surface
- HMC acceptance on a smooth Gaussian-sum surface should be 65-80% with proper tuning
- Expected ESS/iteration much higher for PGibbs

### Potential issues

1. **Path degeneracy**: Even with ancestor sampling, CSMC can have slow mixing for long time series (T >> 100). Monitor the fraction of CSMC output trajectories that differ from the reference.
2. **Gibbs coupling**: If $\theta$ and $x_{1:T}$ are strongly correlated, the Gibbs sampler mixes slowly (each conditional update makes small moves). HMC helps here because it can take large theta-moves given a fixed trajectory.
3. **Warm-up**: The first trajectory (from a bootstrap PF with wrong $\theta$) may be poor. Allow for a generous burn-in (1000+ iterations).

---

## 9. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `code/src/filters/particle/conditional_smc.py` | **DONE** | Bootstrap CSMC (`BootstrapConditionalSMC`) — Phase 1 |
| `code/src/filters/particle/ledh_invertible_csmc.py` | **DONE** | LEDH CSMC (`LEDHConditionalSMC`) — Phase 2 |
| `code/src/filters/particle/__init__.py` | **DONE** | Export both new classes |
| `code/hmc_dpf_notes.md` | **DONE** | Comprehensive math notes for PGibbs + CSMC + LEDH |
| `code/src/DF/pgibbs_runner.py` | **DONE** | PGibbs + HMC orchestrator (`PGibbsRunner`) |
| `code/src/DF/__init__.py` | **DONE** | Export `PGibbsRunner` |
| `code/src/experiments/run_dpf_experiment.py` | **DONE** | Add `sampler: pgibbs` branch |
| `code/configs/dpf/pgibbs/kitagawa_bpf_mh.yaml` | **DONE** | Exp 1: PGibbs + MH + Bootstrap CSMC (reproduce PMCMC paper) |
| `code/configs/dpf/pgibbs/kitagawa_ledh_csmc_hmc.yaml` | **DONE** | Exp 4: PGibbs + HMC + LEDH CSMC (full framework) |
| `code/configs/filter/ledh_csmc.yaml` | **DONE** | Exp 2: LEDH CSMC standalone filtering config |
| `code/src/DF/pgibbs_runner.py` | **DONE** | Added `_theta_step_mh()` + `theta_sampler` option |
| `code/src/filters/particle/ledh_invertible_csmc.py` | **DONE** | Added `filter()` method for standalone filtering |

No changes needed to model classes, parameter handler, resampling, or existing LEDH filters.

---

## 10. Experiment Plan

Four experiments to validate the PGibbs framework and compare tricks:

### Experiment 1: Reproduce PMCMC paper (Andrieu et al. 2010)

**Goal**: Validate PGibbs framework by reproducing the original paper's setup.

**Setup**:
- **θ-step**: Random-walk Metropolis-Hastings on closed-form $\log p(\theta | x, y)$
- **x-step**: Bootstrap Conditional SMC (`conditional_smc.py`) with ancestor sampling
- **Model**: Kitagawa, T=100, true $\sigma_V=10$, $\sigma_W=1$

**Why MH, not HMC**: The paper uses random-walk MH for the θ-step. Reproducing with MH first validates the Gibbs framework independently of HMC. If MH works, it confirms the θ-step target and CSMC are correct; then replacing MH with HMC should only improve efficiency.

**Config**: `configs/dpf/pgibbs/kitagawa_bpf_mh.yaml`

**Run**:
```bash
python -m src.experiments.run_dpf_experiment dpf=pgibbs/kitagawa_bpf_mh
```

**Expected outcome**: Posterior concentrates around true values. Acceptance ~20-40% (typical for random-walk MH). Slow but correct.

### Experiment 2: LEDH CSMC filtering (Trick 2 — reference pinning)

**Goal**: Evaluate CSMC reference pinning as a standalone filtering improvement for bimodal Kitagawa.

**Setup**:
- `LEDHConditionalSMC` with true parameters (no parameter inference)
- `filter()` method generates a prior reference trajectory, runs one CSMC sweep
- N=500 particles, 15 lambda steps, ancestor sampling enabled

**What it tests**: Does pinning one particle to a reference trajectory help LEDH handle the bimodal $y = x^2/20$ observation? The N-1 free particles use LEDH flow; the reference stays pinned. Ancestor sampling allows the reference's history to graft onto better paths.

**Config**: `configs/filter/ledh_csmc.yaml` + `model=kitagawa`

**Run**:
```bash
python -m src.experiments.run_experiment model=kitagawa filter=ledh_csmc
```

**Compare against**: Experiment 3 (same model, different trick).

### Experiment 3: LEDH bimodal filtering (Look-ahead sign correction)

**Goal**: Evaluate look-ahead sign correction as a standalone filtering improvement for bimodal Kitagawa.

**Setup**:
- `LEDHInvertibleBimodal` with true parameters
- K=2 lookahead steps, N=500 particles, 29 lambda steps

**What it tests**: Does scoring $+\eta_1$ vs $-\eta_1$ against future observations help LEDH pick the correct mode? This is a purely local trick (no CSMC machinery, no reference trajectory).

**Config**: `configs/filter/ledh_invertible_bimodal.yaml` + `model=kitagawa`

**Run**:
```bash
python -m src.experiments.run_experiment model=kitagawa filter=ledh_invertible_bimodal
```

### Experiments 2+3 comparison

**Metrics to compare**:
- RMSE against true states
- ESS over time (higher = better particle diversity)
- Log-likelihood (higher = better fit)
- Mode-tracking: does the filter follow sign changes in $x_t$?

**Interpretation**:
- If Exp 2 (CSMC pinning) wins → reference pinning is more effective for bimodality
- If Exp 3 (look-ahead) wins → local sign correction is sufficient
- If both help in different ways → combine them (LEDH CSMC + look-ahead) for future work

### Experiment 4: HMC + PGibbs + LEDH CSMC (full framework)

**Goal**: Test the full PGibbs + HMC framework with LEDH CSMC for the x-step.

**Setup**:
- **θ-step**: HMC on closed-form $\log p(\theta | x, y)$ — 5 leapfrog steps, dual averaging
- **x-step**: LEDH Conditional SMC with 500 particles, 15 lambda steps
- **Model**: Kitagawa, T=100, initial guesses $\sigma_V=5$, $\sigma_W=2$

**Why this matters**: This is the end goal — HMC for efficient θ-exploration on a smooth surface, plus LEDH CSMC for high-quality trajectory sampling. If it works, it solves all three original problems (gradient cliffs, weight collapse, bimodality).

**Config**: `configs/dpf/pgibbs/kitagawa_ledh_csmc_hmc.yaml`

**Run**:
```bash
python -m src.experiments.run_dpf_experiment dpf=pgibbs/kitagawa_ledh_csmc_hmc
```

**Expected outcome**: Posterior should concentrate around true values. HMC acceptance ~60-80% (smooth target). Compare against Experiment 1 (MH) — HMC should have higher ESS per iteration.

### Experiment dependency graph

```
Exp 1 (MH + BPF CSMC)     → Validates PGibbs framework
Exp 2 (LEDH CSMC filter)  ─┐
                            ├→ Compare filtering tricks → informs Exp 4
Exp 3 (LEDH bimodal filter)┘
Exp 4 (HMC + LEDH CSMC)   → Full framework (the goal)
```
