# Report Restructuring Plan

## Summary of Feedback

The report must be transformed from an academic-paper-style document into a **client-facing technical report** accessible to readers without ML backgrounds. Key requirements:

1. **Clear overarching narrative**: What are we doing? How? What algorithm at each step?
2. **Comprehensive mathematical detail** with formal **Algorithm blocks**
3. **Complete results** for at least one simple model (e.g., Linear Gaussian)
4. **HMC specifics**: settings, adaptation, mass matrix, convergence diagnostics (trace plots)
5. **Initial distribution discussion**: is standard normal optimal?
6. **Accessible to non-ML clients**: no assumptions about prior knowledge

---

## Current Structure (Problems Identified)

```
1. Literature Review                          <-- mixes background + method derivations
   1.1 State-Space Modeling                   <-- good intro but informal tone
   1.2 Kalman Family                          <-- OK but sparse
   1.3 Particle based methods                 <-- too many methods crammed together
   1.4 HMC                                   <-- placed inside "literature review"
2. Problems and experiments                   <-- vague section name
   2.1 State Space models                     <-- 8 models listed, unclear which are used
   2.2 Code organization                      <-- useful but doesn't belong in main report
   2.3 Part I / 2.5 Part II                   <-- meaningless labels
   2.4 Filter comparison                      <-- 4-line section
3. Differentiable particle filter             <-- most important section, needs expansion
   3.5 Workflow and current status             <-- reads like a lab notebook
   3.6 Benchmarking                           <-- incomplete results
Appendices A-C                                <-- good, keep mostly as-is
```

**Key issues:**
- No executive summary or problem statement
- "Part I" and "Part II" are meaningless labels for a client
- Literature review contains method derivations that should be in a Methods section
- 8 models listed but only 2-3 used seriously; confusing
- Code organization section is internal documentation, not client-facing
- HMC settings (step size, mass matrix, adaptation, burn-in) are never clearly stated
- No formal Algorithm blocks anywhere
- No trace plots for convergence assessment
- Tone is too informal in places ("I spent sometime...", "It turned out...")
- Figures are globally disabled (`\excludecomment{figure}`)
- No discussion of initial distribution choice (feedback item #1)

---

## Proposed New Structure

```
1. Introduction and Problem Statement          [NEW - ~2 pages]
2. Background: State-Space Models              [REWRITE from 1.1]
3. Methods                                     [RESTRUCTURE from 1.2-1.4 + appendices]
   3.1 Kalman Filter Family
   3.2 Particle Filters (Bootstrap)
   3.3 Particle Flow Filters (EDH/LEDH)
   3.4 Resampling Strategies
   3.5 Hamiltonian Monte Carlo
   3.6 Differentiable Particle Filters
4. Experimental Setup                          [REWRITE from 2.1 + parts of 2.2]
   4.1 Models Used
   4.2 Data Generation
   4.3 Prior and Initialization Choices
   4.4 HMC Configuration
5. Results                                     [RESTRUCTURE from 2.3-2.5 + 3.x]
   5.1 Filtering Performance (known parameters)
   5.2 Parameter Estimation: HMC + Kalman
   5.3 Parameter Estimation: HMC + DPF
   5.4 Convergence Diagnostics
6. Discussion and Next Steps                   [NEW]
Appendices (keep A-C, add D)
```

---

## Detailed Section-by-Section Plan

### Section 1: Introduction and Problem Statement [NEW - write from scratch]

**Purpose**: Give a non-ML reader the big picture in plain language before any math.

**Content**:
- What is the problem? (Estimating hidden states and model parameters from noisy observations)
- Why does it matter? (Financial applications: stochastic volatility, regime detection, risk)
- What is our approach? (Particle filters for state estimation + HMC for parameter estimation)
- What is a "differentiable particle filter" and why do we need it? (1-2 paragraphs, no equations)
- Roadmap of the report: "Section 2 covers... Section 3 describes..."

**Action**: Write ~2 pages. Pull the financial motivation from current Section 1.1 (lines 54-69) and rephrase for accessibility.

---

### Section 2: Background — State-Space Models [REWRITE from current 1.1]

**Purpose**: Define the mathematical framework clearly. This is the foundation everything else builds on.

**Content**:
- State-space model definition (Eq 1 from current report — keep)
- Concrete 1D example with numbers (not just symbols)
- The two tasks: filtering (estimate $x_k$) and parameter estimation (estimate $\theta$)
- Prediction-update recursion (keep current Eq from lines 118-123)
- Likelihood and its role in parameter estimation (keep lines 127-137)
- **NEW: Initial distribution discussion** — address feedback item #1:
  - What is the initial distribution $p(x_0)$?
  - When is $\mathcal{N}(0,1)$ appropriate? When is it not?
  - Impact on filter convergence (e.g., for anisotropic priors like bearing-only)
  - Stationary distribution as an alternative (e.g., for stochastic volatility: $x_0 \sim \mathcal{N}(0, \sigma^2/(1-\alpha^2))$)

**Action**: Rewrite lines 52-137. Add 1-2 pages on initialization. Remove all commented-out text.

---

### Section 3: Methods [MAJOR RESTRUCTURE]

This is where the bulk of rewriting happens. Each method gets:
1. Intuitive explanation (what it does, when to use it)
2. Mathematical derivation (move detailed derivations to appendix if long)
3. **Formal Algorithm block** (pseudocode)
4. Limitations and when it fails

#### 3.1 Kalman Filter Family [REWRITE from current 1.2 + Appendix A]

**Current state**: Section 1.2 (lines 189-204) is a brief overview; Appendix A has the full derivation.

**Action**:
- Keep Appendix A as the detailed derivation
- In the main text, present:
  - KF assumptions (linear, Gaussian)
  - **Algorithm 1: Kalman Filter** (pseudocode block with predict + update)
  - **Algorithm 2: Extended Kalman Filter** (show the Jacobian linearization)
  - **Algorithm 3: Unscented Kalman Filter** (sigma point generation + propagation)
  - Joseph form explanation (keep current lines 517-559, move here from "experiments")
- Remove informal language ("This works fine if...", "But here I do want to mention...")

#### 3.2 Bootstrap Particle Filter [REWRITE from current 1.3, lines 206-246]

**Current state**: The math is already there but not in Algorithm block format.

**Action**:
- Add intuitive explanation: "represent distribution with weighted samples"
- **Algorithm 4: Bootstrap Particle Filter** — formal pseudocode:
  ```
  Input: observations z_{1:T}, model (f, h), N particles
  Initialize: x_0^{(i)} ~ p(x_0), w_0^{(i)} = 1/N
  For t = 1 to T:
    Predict: x_t^{(i)} ~ p(x_t | x_{t-1}^{(i)})
    Update: w_t^{(i)} ∝ p(z_t | x_t^{(i)})
    If N_eff < N/2: Resample
  ```
- **Algorithm 5: Systematic Resampling** (lines 237-245 already have the math)
- Particle degeneracy explanation (keep, but consolidate — currently split between lines 233 and 631)

#### 3.3 Particle Flow Filters [REWRITE from current 1.3.1 + Appendix B]

**Current state**: Lines 249-264 give an overview; Appendix B has the derivation. Good content but scattered.

**Action**:
- Explain the core idea: move particles deterministically instead of using weights
- **Algorithm 6: EDH Flow Filter** — pseudocode with explicit $A(\lambda)$, $b(\lambda)$
- **Algorithm 7: LEDH Invertible Filter** — pseudocode showing per-particle linearization + Jacobian accumulation
- Keep detailed derivation in Appendix B
- The stochastic flow section (current 2.5.1, lines 810-850) should move here as a subsection

#### 3.4 Resampling Strategies [REWRITE from current 1.3.3 + 2.5.2]

**Current state**: OT resampling (lines 281-298) in lit review; soft/OT comparison (lines 868-894) in experiments.

**Action**:
- Consolidate into one section with three subsections:
  - Systematic resampling (already described)
  - Soft resampling: definition, $\alpha$ parameter, differentiability
  - OT Sinkhorn resampling: entropy regularization, Sinkhorn iterations
- **Algorithm 8: Soft Resampling**
- **Algorithm 9: OT Sinkhorn Resampling**
- Include the differentiability comparison table (current Table at line 1045)

#### 3.5 Hamiltonian Monte Carlo [REWRITE from current 1.4]

**Current state**: Lines 303-338 have good content but lack specifics.

**Action**:
- Keep the physics analogy and leapfrog explanation
- **Algorithm 10: HMC with Dual Averaging** — full pseudocode:
  ```
  Input: initial θ, target log-density U(θ), gradient ∇U
  Hyperparameters: L (leapfrog steps), ε (step size), M (mass matrix)
  Adaptation: dual averaging for ε, warmup for M
  For i = 1 to N_samples:
    p ~ N(0, M)
    (θ*, p*) = Leapfrog(θ, p, ε, L)
    α = min(1, exp(-ΔH))
    Accept/reject
  ```
- **NEW**: Explicitly state our HMC settings:
  - Step size adaptation: dual averaging (reference Appendix C.2)
  - Mass matrix: diagonal (identity) — discuss why, what full mass matrix would buy
  - Leapfrog steps: how many, how chosen
  - Burn-in: how many steps, what fraction is adaptation
  - Target acceptance rate: 0.65-0.90 and why
- Move SGD comparison (current 1.4.1) here briefly

#### 3.6 Differentiable Particle Filters for Parameter Estimation [REWRITE from current Section 3]

**Current state**: Section 3 (lines 898-1194) has excellent content but reads like a research diary.

**Action**:
- Restructure around the three inference strategies:
  1. **SGD/Adam for MAP estimation** (point estimate, no uncertainty)
  2. **HMC on differentiable likelihood** (full posterior, requires smooth gradients)
  3. **PMCMC (gradient-free)** (uses PF as black box, no differentiability needed)
- The differentiability-bias trade-off (current 3.4, lines 1034-1082) is excellent — keep but polish
- Remove lab-notebook style text ("I initially thought...", "Too slow, too many things to debug...")
- **Algorithm 11: DPF-HMC Pipeline** — end-to-end pseudocode:
  ```
  Input: observations z_{1:T}, model class, trainable parameters θ
  Step 1 (optional): SGD warm-start to find θ_MAP
  Step 2: Fix PF seed
  Step 3: Run HMC with U(θ) = -log p(z_{1:T}|θ) - log p(θ)
           where log p(z_{1:T}|θ) comes from the differentiable PF
  Step 4: Convergence diagnostics (trace plots, R-hat, ESS)
  ```

---

### Section 4: Experimental Setup [REWRITE from current 2.1-2.2]

**Purpose**: A client should be able to reproduce every experiment from this section.

#### 4.1 Models Used [REWRITE from current 2.1]

**Current state**: 8 models listed (lines 344-485). Most are never used in the results.

**Action**:
- Focus on the 3-4 models actually used in experiments:
  1. **Linear Gaussian** — the ground-truth benchmark
  2. **Cubic Sensor** — mild nonlinearity, good for debugging
  3. **Stochastic Volatility** — strong nonlinearity, the primary test case
  4. **Kitagawa** — bimodal posterior, structural difficulty
- For each model, clearly state:
  - Equations
  - All parameter values
  - Dimensionality
  - Why this model was chosen (what it tests)
- Move unused models (Acoustic Tracking, Range-Bearing, Lorenz 96, Two-Sensor Bearing) to an appendix or remove

#### 4.2 Data Generation [NEW]

**Action**:
- How is synthetic data generated? (simulate from the model with known θ)
- Time series length T
- Random seed policy
- How many independent datasets?

#### 4.3 Prior and Initialization Choices [NEW — addresses feedback item #1]

**Action**:
- For each model: what is $p(x_0)$? What is $p(\theta)$?
- Justify each choice (e.g., stationary distribution for SV, diffuse prior for Kitagawa)
- Sensitivity analysis: what happens with a bad initial distribution?
- Discuss the standard normal choice and when it is/isn't appropriate

#### 4.4 HMC Configuration [NEW — addresses feedback items on HMC]

**Action**:
- Number of burn-in steps and samples
- Target acceptance rate
- Step size adaptation schedule
- Mass matrix: identity vs. diagonal vs. full
- Leapfrog trajectory length
- Gradient clipping threshold
- Fixed PF seed value and number of particles
- How convergence is assessed (trace plots, R-hat, ESS)

---

### Section 5: Results [RESTRUCTURE from current 2.3-3.6]

#### 5.1 Filtering Performance (Known Parameters) [consolidate from 2.3-2.5]

**Content** (keep these, restructure):
- Joseph form vs standard update (current 2.3.1) — keep table
- EKF/UKF failure on stochastic volatility (current 2.3.2) — keep
- BPF results and particle degeneracy (current 2.3.3) — keep with figures
- EDH/LEDH comparison (current 2.3.4) — keep tables
- Resampling comparison (current 2.5.2) — keep table
- Filter comparison summary (current 2.4) — expand from 4 lines to a proper synthesis

**Action**:
- Re-enable figures (`\excludecomment{figure}` must be removed!)
- One subsection per model, not per filter
- Remove "Part I" / "Part II" labels
- Add summary table comparing all filters across all models

#### 5.2 Parameter Estimation: HMC + Kalman Family [from current 3.6.1]

**Action**:
- Present EKF/UKF + HMC results as the baseline
- Include posterior histograms (current figures)
- **NEW: Add trace plots** — show parameter chains over HMC iterations
- **NEW: Add convergence diagnostics table** (ESS, R-hat per parameter)

#### 5.3 Parameter Estimation: HMC + DPF [from current 3.6.2 + 3.4 + 3.5]

**Action**:
- Organize by model:
  - Linear Gaussian: systematic vs soft vs OT (complete results)
  - Cubic Sensor: gradient issues, what works and what doesn't
  - Stochastic Volatility: the primary result
- Include for each:
  - Posterior histograms
  - **Trace plots** (parameter value vs HMC iteration)
  - Step size evolution plot (keep current Figure for step size collapse)
  - ESS and R-hat
  - Wall time
- The differentiability-bias discussion (current 3.4) should introduce this subsection
- Remove narrative/diary style text

#### 5.4 Convergence Diagnostics [NEW]

**Action**:
- Dedicated subsection showing:
  - Trace plots for all reported experiments
  - Autocorrelation plots
  - R-hat and ESS tables
  - Discussion of what constitutes "converged" for these experiments

---

### Section 6: Discussion and Next Steps [NEW]

**Content**:
- Summary of what works and what doesn't
- Practical recommendations (which filter + resampling for which model class)
- Known limitations (initialization sensitivity, gradient cliffs, scale)
- Future work:
  - MAP warm-start for HMC
  - NUTS implementation
  - Look-ahead trick for bimodal models
  - Scaling to higher dimensions
  - Neural network augmentations (briefly)

---

### Appendices [keep + polish]

- **Appendix A: Kalman Family** — keep as-is (detailed derivation)
- **Appendix B: Particle Flow Derivation** — keep as-is
- **Appendix C: MCMC Diagnostics** — keep as-is
- **Appendix D: Additional Models** [NEW] — move unused models here (Acoustic Tracking, Range-Bearing, Lorenz 96, Two-Sensor Bearing)
- **Appendix E: Code Organization** [move from current 2.2] — useful reference but not main text

---

## Writing Style Changes

| Current | Target |
|---------|--------|
| "I spent some time..." | Remove first-person narrative |
| "It turned out that..." | State facts directly |
| "I initially thought..." | Remove; just state what works |
| "This is a well-structured test and I learned a lot" | Remove entirely |
| Commented-out text blocks | Delete all commented-out text |
| `\excludecomment{figure}` | Remove — figures must be shown |
| Informal subsection names ("Part I", "Part II") | Descriptive names |
| Mixed tense | Consistent present tense for methods, past tense for experiments |
| "we" and "I" alternating | Consistent "we" throughout |

---

## Specific Technical Items to Address

### 1. Initial Distribution (Feedback #1)
- Add discussion in Section 4.3
- For Linear Gaussian: $x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$ — standard normal is fine as it's conjugate
- For Stochastic Volatility: $x_0 \sim \mathcal{N}(0, \sigma^2/(1-\alpha^2))$ — the stationary distribution is better than $\mathcal{N}(0,1)$
- For Kitagawa: $x_0 \sim \mathcal{N}(0, 5)$ — already specified, discuss sensitivity
- General principle: when the stationary distribution is known, use it; otherwise use a diffuse prior and allow burn-in

### 2. Algorithm Blocks (Feedback #2)
Need to add at minimum:
- Algorithm 1: Kalman Filter
- Algorithm 2: EKF
- Algorithm 3: UKF
- Algorithm 4: Bootstrap Particle Filter (with systematic resampling)
- Algorithm 5: EDH/LEDH Flow Filter
- Algorithm 6: LEDH Invertible Filter (with Jacobian)
- Algorithm 7: OT Sinkhorn Resampling
- Algorithm 8: HMC with Dual Averaging
- Algorithm 9: DPF-HMC Pipeline (end-to-end)

Use `\usepackage{algorithm2e}` or `\usepackage{algorithmic}` for formatting.

### 3. HMC Specifics (Feedback #2)
State explicitly in a table or itemized list:
- Number of burn-in steps (currently 100)
- Number of post-burn-in samples (currently 200)
- Target acceptance rate (currently 0.9)
- Mass matrix type (currently identity)
- Leapfrog steps per proposal
- Step size initialization
- Gradient clipping threshold
- Number of particles in the PF
- PF random seed

### 4. Trace Plots (Feedback #2)
- Add trace plots for every HMC experiment
- Show: parameter value vs iteration, acceptance rate vs iteration, step size vs iteration
- These exist in the code output but are not in the report

### 5. Complete Results for One Model
- Choose **Linear Gaussian** as the complete example
- Show every step: data generation → filtering → parameter estimation → diagnostics
- All algorithms, all resampling methods, all diagnostics
- This serves as the "reference" that validates the pipeline

---

## Priority Order for Implementation

### Phase 1: Structure (Week 1)
1. Create new Section 1 (Introduction)
2. Move code organization to Appendix E
3. Rename/restructure sections to match proposed outline
4. Delete all commented-out text
5. Remove `\excludecomment{figure}`
6. Add `\usepackage{algorithm2e}` or similar

### Phase 2: Methods Polish (Week 1-2)
7. Write Algorithm blocks for all methods (KF, EKF, UKF, BPF, EDH, LEDH, resampling, HMC)
8. Rewrite Section 3 (Methods) — consolidate scattered content
9. Polish mathematical notation for consistency

### Phase 3: Experimental Setup (Week 2)
10. Write Section 4.2 (Data Generation)
11. Write Section 4.3 (Prior and Initialization — addresses feedback #1)
12. Write Section 4.4 (HMC Configuration)
13. Trim model list to 3-4 primary models; move others to appendix

### Phase 4: Results (Week 2-3)
14. Run experiments to generate trace plots and convergence diagnostics
15. Restructure results by model (not by filter)
16. Complete Linear Gaussian as the reference case
17. Add convergence diagnostics subsection (5.4)
18. Re-enable and include all figures

### Phase 5: Polish (Week 3-4)
19. Write Section 6 (Discussion)
20. Fix writing style (remove first-person narrative, lab-notebook tone)
21. Ensure consistent notation throughout
22. Proofread for typos and grammar
23. Check all cross-references and citations

---

## Files to Generate / Experiments to Run

- [ ] Trace plots for all HMC experiments (parameter chains, acceptance rate, step size)
- [ ] Autocorrelation plots for HMC chains
- [ ] Complete Linear Gaussian results: all filters × all resampling methods × HMC
- [ ] Initialization sensitivity study: vary $p(x_0)$ and show effect on convergence
- [ ] Summary comparison table: all models × all methods
