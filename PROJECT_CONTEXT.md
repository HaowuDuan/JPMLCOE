# JPMLCOE Project Context

> **Purpose**: This document is the authoritative context file for AI assistants managing this project. It covers the research goal, codebase architecture, experimental status, open issues, and report standards. Read this before making any planning or implementation decisions.

---

## 1. Project Big Picture

### What This Project Is

This is a **research and engineering project** on **Bayesian parameter inference in state-space models**, with a primary focus on two methodological contributions:

1. **Differentiable Particle Filters (DPF)**: Particle filters whose resampling steps are made differentiable (via OT-entropy or soft resampling), allowing gradients of the log-marginal-likelihood to flow through the entire filter. This enables gradient-based parameter inference.

2. **Particle Flow Filters (EDH/LEDH)**: Instead of the standard predict–weight–resample cycle, particles are continuously *flowed* toward the posterior by solving an ODE (the Daum-Huang flow). The Linearized EDH (LEDH) approximates this flow per-particle via a local linearization.

These two components are combined with **Hamiltonian Monte Carlo (HMC)** to produce full posterior distributions over unknown model parameters.

### Research Motivation

Financial systems are driven by hidden latent processes (stochastic volatility, regime shifts, limit order book dynamics) that are not directly observable. By modeling these systems as **continuous state-space models (SSMs)**:

```
x_k = f(x_{k-1}, w_k)     (latent state evolution)
z_k = h(x_k, u_k)         (observation model)
```

we need two things:
- **Filtering**: estimate the hidden state `x_k` given observations `z_{1:k}` — the *filtering distribution* `p(x_k | z_{1:k})`
- **Bayesian inference**: estimate model parameters `θ` from data — the *posterior* `p(θ | z_{1:T})`

### How DPF + HMC Works

The key pipeline:
1. Given parameter `θ`, run a differentiable particle filter to compute `log p(z_{1:T} | θ)` (log marginal likelihood).
2. HMC uses `∇_θ [-log p(z_{1:T}|θ) - log p(θ)]` to propose moves in parameter space.
3. Because the particle filter is differentiable (OT or soft resampling), `GradientTape` can backpropagate through all T timesteps.
4. The result is a Markov chain that samples from the true posterior `p(θ | z_{1:T})`.

### Project Scope

This is a **controlled experiment study**: synthetic data is generated from known models with known parameters. The inference pipeline is benchmarked against exact methods (Kalman filter) where available. Success means the posterior median is close to the true parameter, with calibrated uncertainty.

**This is NOT yet a live financial system.** It is a validated research framework that forms the foundation for real-world applications.

---

## 2. Implemented Algorithms

### 2.1 Filtering Algorithms

| Algorithm | File | Key property |
|-----------|------|-------------|
| Kalman Filter (KF) | `src/filters/kalman/kalman.py` | Exact for linear-Gaussian; gold standard |
| Extended KF (EKF) | `src/filters/kalman/extended_kalman.py` | Linearizes nonlinear models via Jacobian |
| Unscented KF (UKF) | `src/filters/kalman/unscented_kalman.py` | Sigma-point propagation; captures higher-order moments |
| Bootstrap PF (BPF) | `src/filters/particle/bootstrap_pf_hmc.py` | General-purpose; differentiable via resampling |
| EDH Flow | `src/filters/particle/edh_flow.py` | Exact Daum-Huang particle flow |
| LEDH Invertible | `src/filters/particle/ledh_invertible.py` | Per-particle linearized flow; tracks Jacobians |
| LEDH (HMC variant) | `src/filters/particle/ledh_invertible_hmc.py` | `@tf.function` + `tf.while_loop` for fast gradients |

**KF Family details**:
- Joseph form covariance update for numerical stability when observation noise is small
- Batched UKF: generates 2n+1 sigma points per particle
- EKF uses per-step Jacobians `F_k = ∂f/∂x`, `H_k = ∂h/∂x`

**LEDH details**:
- 29 Euler integration steps (λ: 0 → 1) per observation assimilation
- Per-step flow parameters: `A(λ)` (drift matrix) and `b(λ)` (bias) computed per-particle
- Jacobian accumulation: `log|det(I + dλ·A)|` summed over 29 steps → numerically sensitive
- Key utility: `graph_safe_log_abs_det_fast` in `src/utils/linalg.py` (custom gradient)

### 2.2 Resampling Strategies

| Strategy | Config key | Gradient | Notes |
|----------|-----------|----------|-------|
| Systematic | `systematic` | `stop_gradient` | Non-differentiable; used for baseline |
| Soft | `soft` | Differentiable | α ∈ [0,1]; α→0 = hard, α→1 = no-op |
| OT-Entropy (Sinkhorn) | `ot_entropy` | Differentiable | Entropy-regularized OT; ε=0.5 default |

**OT-entropy details** (`src/resampling/ot_entropy.py`):
- Sinkhorn algorithm solves regularized OT: `T_ij` = transport probability from particle i to position j
- Normalization (center + scale particles) applied before solve; `stop_gradient` on normalization is configurable
- ε (epsilon) controls regularization: small = sharper transport, large = near-trivial

### 2.3 Parameter Inference Methods

| Method | Type | Notes |
|--------|------|-------|
| HMC | Full posterior | Dual averaging step-size adaptation; fixed leapfrog steps |
| NUTS | Full posterior | Auto trajectory length; infrastructure exists, not fully tested |
| MAP (Adam) | Point estimate | Warm-start for HMC; `random_seed: true` by default |
| Particle MCMC | Full posterior | Fixed seed → deterministic likelihood surface |

**HMC parameter handling**:
- `ParameterHandler`: manages bijector transforms (unconstrained ↔ constrained)
- `DifferentiableModel`: wraps base model; tracks trainable params via `setattr`
- `ParameterSpec`: stores `init_value`, `constraint` (e.g., `positive`), and prior

---

## 3. State-Space Models

| Model | State dim | Obs dim | Key challenge | Status |
|-------|-----------|---------|---------------|--------|
| **1D Linear-Gaussian** | 1 | 1 | Baseline; KF exact | ✅ Primary benchmark |
| **Kitagawa (UNGM)** | 1 | 1 | Bimodal transition | Configs done; results partial |
| **Cubic Sensor** | 1 | 1 | Nonlinear obs (x³/20) | Difficult; used for testing but may not appear in final report |
| **Range-Bearing** | 2 | 2 | Nonlinear obs (range + atan2) | ⚠️ PF fails; EKF/UKF work; secondary benchmark |
| **Stochastic Volatility** | 1 | 1 | Multiplicative noise; obs Jacobian = 0 | **Primary nonlinear benchmark**; LEDH flow currently collapses; future UKF improvement planned |
| **5D Linear-Gaussian** | 5 | 5 | Higher dimension; partial obs variants | Configs + runners done; results pending |
| **Acoustic Tracking** | 3 | 2 (TDOA) | Realistic multi-sensor | Configs done; less active |
| **Lorenz-96** | 40 | partial | Chaotic, high-dim | Config only; not prioritized |

**Models used in paper**: Focus on Linear-Gaussian (primary benchmark) and Stochastic Volatility (primary nonlinear benchmark). Cubic Sensor is used for internal testing but its inclusion in the final report is not confirmed — it is a difficult model, not a broken one. Range-Bearing remains a secondary benchmark. Future work includes extending the SV model family and improving UKF to work on SV.

---

## 4. Codebase Architecture

```
code/
├── src/
│   ├── filters/
│   │   ├── kalman/          # KF, EKF, UKF
│   │   └── particle/        # BPF, EDH, LEDH variants
│   ├── models/              # State-space model definitions
│   ├── resampling/          # ot_entropy.py, systematic, soft
│   ├── experiments/
│   │   ├── run_dpf_experiment.py    # Main Hydra entry point (HMC inference)
│   │   ├── run_experiment.py        # Filtering-only entry point
│   │   └── visualization_hmc.py    # Trace + posterior plots
│   ├── DF/
│   │   ├── hmc_runner.py            # Core HMC/MAP runner
│   │   ├── pgibbs_runner.py         # Particle Gibbs (PGAS)
│   │   └── pgibbs_runner.py         # Particle MH
│   ├── core/                # Base classes
│   └── utils/
│       ├── linalg.py        # safe_cholesky, graph_safe_log_abs_det_fast
│       ├── flow_params.py   # compute_flow_params_batch (A, b matrices)
│       ├── distributions.py # compute_flow_weights, log-likelihood utils
│       └── device.py        # GPU/CPU device handling
├── configs/
│   ├── dpf/hmc/{model}/     # HMC inference configs per model
│   ├── dpf/map/{model}/     # MAP configs per model
│   ├── filter/              # Filter-type configs (ledh_invertible_ot.yaml, etc.)
│   ├── model/               # Model parameter configs
│   └── experiment/          # High-level experiment specs (1D, 5D, etc.)
├── tests/
│   ├── unit/
│   ├── filters/
│   ├── hmc/
│   │   ├── test_ledh_gradient_diagnosis.py   # 9-part LEDH diagnosis suite
│   │   └── test_rb_hmc_diagnosis.py
│   └── dpf/
├── outputs/                 # All experiment results (auto-generated by Hydra)
│   └── dpf/hmc/{model}/{method}/
│       ├── summary.json     # ESS, R-hat, acceptance, posterior stats
│       ├── trace.csv        # Per-step parameter values
│       └── samples_*.npy   # Raw posterior samples
└── analysis/
    └── hmc.ipynb            # Visualization notebook
```

### Gradient Flow Path

```
run_dpf_experiment.py (Hydra main)
  → instantiate model, filter, resampler
  → DPFRunner.run_inference()  [hmc_runner.py]
    → HMC (TFP sampler)
      → GradientTape
        → _negative_log_posterior(θ)
          → filter.log_marginal_likelihood_tf(observations, seed)
            → for t in [1..T]:
                predict / flow (LEDH) / resample / update
            → return Σ_t logsumexp(log w_t)
          → + log_prior(θ)
```

### Configuration System

The project uses **Hydra** for configuration management. Configs are composed from multiple YAML files:

```bash
# Example run
python src/experiments/run_dpf_experiment.py \
  +experiment=dpf/hmc/linear_gaussian/bpf_ot
```

**Typical HMC config fields**:
```yaml
hmc:
  num_samples: 200-400
  num_burnin: 100-200
  num_leapfrog_steps: 5
  step_size: 0.001
  target_accept_prob: 0.65  # 0.9 for deterministic filters
filter:
  n_particles: 500-2000
  resampling_method: ot_entropy  # or soft, systematic
  stop_gradient_resampling: false
data:
  T: 50                # time steps
  seed: 42
trainable_params:
  obs_noise_std:
    init_value: 2.0
    constraint: positive
    prior: LogNormal(0.0, 1.0)
```

---

## 5. Experimental Status

### 5.1 Completed and Working

#### Experiment A: 1D Linear-Gaussian, 1 Parameter (`obs_noise_std`)

**Setup**: True `obs_noise_std=1.0`, init=2.0, T=50, LogNormal(0,1) prior, 400 post-burnin samples.

| Method | Posterior Mean | ESS | R-hat | Status |
|--------|---------------|-----|-------|--------|
| Kalman | 1.122 ± 0.188 | 200 | 0.826 | ✅ Gold standard |
| EKF | 1.122 ± 0.188 | 200 | 0.826 | ✅ |
| UKF | 1.120 ± 0.188 | 200 | 0.849 | ✅ |
| BPF-sys | 1.175 ± 0.145 | 6.0 | 0.966 | ✅ (low ESS expected) |
| BPF-OT | 1.164 ± 0.161 | 400 | 0.993 | ✅ Best PF |
| BPF-soft | 1.137 ± 0.163 | 47.4 | 0.916 | ✅ |
| LEDH-sys | 1.115 ± 0.172 | 16.8 | 0.885 | ✅ |
| LEDH-OT | 1.155 ± 0.177 | 9.3 | 1.067 | ✅ |
| LEDH-soft | 1.156 ± 0.191 | 18.6 | 0.946 | ✅ |

> Note: upward bias (~1.1–1.2 vs truth 1.0) is consistent across ALL methods including Kalman → data realization effect, not a filter bug.

#### Experiment B: 1D Linear-Gaussian, 2 Parameters (`process_noise_std` + `obs_noise_std`)

Both true = 1.0. Results show process_noise underestimated (~0.8), obs_noise overestimated (~1.2) — a known identifiability tradeoff in linear-Gaussian models. All methods still valid.

| Method | ESS (proc) | ESS (obs) | Best? |
|--------|-----------|-----------|-------|
| EKF/UKF | 200 / 200 | 200 / 200 | Exact |
| BPF-soft | 400 / 400 | Best PF — most robust |
| BPF-OT | 400 / 40.4 | Asymmetric (obs harder) |
| BPF-sys | 57 / 158 | Moderate |
| LEDH-sys/OT/soft | 14–200 / 12–200 | Variable |

#### Experiment C: Range-Bearing, 2 Parameters

True `sigma_range=sigma_bearing=0.1`. **EKF/UKF work; all particle filters struggle.**

| Method | Mean range | Mean bearing | ESS | Notes |
|--------|-----------|-------------|-----|-------|
| EKF | 0.095 ± 0.061 | 0.104 ± 0.054 | 20 | ✅ Best |
| UKF | 0.072 ± 0.037 | 0.101 ± 0.034 | 11 | ✅ Close |
| BPF-OT | 0.137 ± 0.011 | 0.113 ± 0.015 | ~5 | ⚠️ Biased, low ESS |
| BPF-soft | 0.139 ± ~0 | 0.028 ± ~0 | ~4 | ❌ Stuck |
| BPF-sys | 0.093 ± ~0 | 0.128 ± ~0 | ~5 | ❌ Stuck |

> LEDH range-bearing results are outdated (10 samples, old pipeline) and should not be cited.

### 5.2 In Progress / Pending

| Task | Status | Priority |
|------|--------|----------|
| LEDH gradient bug diagnosis (9-test suite) | Tests written; not yet run to conclusion | **CRITICAL** |
| Range-bearing with T=200 (more observations) | T=50 stuck at 0.208, T=200 reaches 0.160 | HIGH |
| LEDH re-run on range-bearing (updated pipeline) | Pending LEDH bug fix | HIGH |
| Stochastic Volatility HMC (BPF only; LEDH blocked) | Pending | MEDIUM |
| 5D Linear-Gaussian experiments | Configs done; not run at scale | MEDIUM |
| Kitagawa HMC inference | Config exists; not benchmarked | MEDIUM |
| MAP warm-start → HMC pipeline | Design in `MAP_implementation.md` | MEDIUM |

### 5.3 Outputs Location

All results live in `code/outputs/dpf/hmc/{model}/{method}/`:
- `summary.json` — ESS, R-hat, acceptance rate, posterior mean/std
- `trace.csv` — parameter values + acceptance at each HMC step
- `samples_{param}.npy` — raw posterior samples
- `posterior_histograms.png` — posterior distribution plots
- `trace_plot.png` — parameter chain plot

---

## 6. Open Problems and Known Bugs

### 6.1 CRITICAL: LEDH Gradient Bug

**Symptom**: LEDH produces systematically wrong posteriors on 1D Linear-Gaussian:
- True `obs_noise_std=1.0`
- LEDH-OT posterior mean: **0.16** (should be ~1.12)
- LEDH-soft: **0.07**
- LEDH-sys: **0.47**
- All BPF variants converge correctly (~1.16) with the same HMC runner

**Conclusion**: Bug is definitively in the LEDH flow computation, not in resampling, prior, or HMC runner.

**Diagnostic plan**: 9 tests in `tests/hmc/test_ledh_gradient_diagnosis.py`
- Test A: Is the LEDH *likelihood surface* itself biased? (evaluate at grid of θ)
- Test B: Does autodiff gradient match finite-difference?
- Test C: Is bias from filter or prior?
- Test D: Component-wise gradient decomposition (obs likelihood / transition prior / Jacobian)
- Test E: Does `stop_gradient(theta)` on Jacobian fix it?
- Test F: Single timestep (T=1) — does error exist per-step?
- Test G: Eager vs compiled (`tf.function`) gradient comparison
- Test H: Sensitivity to number of Euler steps (3, 5, 10, 15, 29)
- Test I: float64 vs float32 precision

**Key source files for diagnosis**:
- `src/filters/particle/ledh_invertible_hmc.py:198–215` — flow loop (29 Euler steps)
- `src/utils/linalg.py:208–258` — `graph_safe_log_abs_det_fast` (custom gradient)
- `src/utils/flow_params.py:167–270` — `compute_flow_params_batch` (A, b matrices)
- `src/utils/distributions.py:118–227` — `compute_flow_weights` (weight formula)

### 6.2 HIGH: LEDH Weight Collapse on CUDA

**Symptom**: Same code + same config → weight collapse on RTX 3090 (CUDA), works fine on Apple Silicon (MPS). All particles get weight ~0, no filtering.

**Root cause**: Likely float32 precision differences. LEDH accumulates 29 `log|det(I + dλ·A)|` terms; CUDA float32 rounding diverges from MPS.

**Proposed fix**: Test float64. Trade-off: ~2× slower.

**Note**: This issue may be connected to the gradient bug (Test I in the diagnostic suite).

### 6.3 HIGH: Range-Bearing Particle Filter Failure

**Symptom**: All PF variants (BPF-sys, BPF-soft, BPF-OT) fail on range-bearing:
- BPF-sys and BPF-soft get stuck (std ~0, biased mean)
- BPF-OT partially converges but biased (0.18 vs truth 0.1)
- EKF and UKF work correctly

**Root causes (hypotheses)**:
1. T=50 too short; likelihood surface too flat to distinguish 0.2 from 0.1
   - *Evidence*: T=200 reaches 0.160 vs T=50 stuck at 0.208
2. Finite-particle bias from fixed seed accumulates worse on nonlinear models
3. OT normalization stop-gradient creates systematic gradient bias on nonlinear obs

**Open experiments needed**:
- Run with T=200 to completion
- Test random seed vs fixed seed in MAP (to understand MAP/HMC discrepancy)
- Run LEDH on range-bearing once gradient bug is fixed

### 6.4 MEDIUM: Stochastic Volatility (LEDH Incompatible; UKF Improvement Planned)

**Symptom**: LEDH flow uses observation Jacobian `H = ∂h/∂x`. For SV model where `y|x ~ N(0, σ²exp(x))`, `H = 0` everywhere → flow parameters A(λ)=0, b(λ)=0 → particles don't move.

**Proposed fix for LEDH**: Score-based generalization (use `∇_x log p(y|x)` instead of H). Design documented in `score_based_flow_report.md` (41 KB). Not yet implemented.

**Planned work**: Improve UKF implementation so that the Kalman family can handle SV better. SV is the primary nonlinear benchmark going forward, with future model variations planned.

**Workaround**: BPF works for SV now. Use BPF for initial SV experiments.

### 6.5 LOW: Cubic Sensor Model

Cubic sensor is a genuinely difficult model used for internal testing. Its inclusion in the report is TBD. Do not exclude it from the codebase; it serves a role in testing edge cases of nonlinear filtering.

---

## 7. Report Standards and Requirements

### 7.1 Document Type

The report (`report/main_polished.tex`) is intended as a **client-facing technical report** — not a purely academic paper. It must be:
- Accessible to readers without ML backgrounds
- Comprehensive enough to reproduce every experiment
- Professional in tone (no informal narrative, no lab-notebook style)

### 7.2 Target Structure (from RESTRUCTURE_PLAN.md)

```
1. Introduction and Problem Statement          (~2 pages, new)
2. Background: State-Space Models              (rewrite from current 1.1)
3. Methods                                     (major restructure)
   3.1 Kalman Filter Family
   3.2 Bootstrap Particle Filter
   3.3 Particle Flow Filters (EDH/LEDH)
   3.4 Resampling Strategies
   3.5 Hamiltonian Monte Carlo
   3.6 Differentiable Particle Filters (end-to-end pipeline)
4. Experimental Setup
   4.1 Models Used (focus on 3-4 primary models)
   4.2 Data Generation
   4.3 Prior and Initialization Choices
   4.4 HMC Configuration
5. Results
   5.1 Filtering Performance (known parameters)
   5.2 Parameter Estimation: Kalman family + HMC
   5.3 Parameter Estimation: DPF + HMC
   5.4 Convergence Diagnostics
6. Discussion and Next Steps
Appendices A–E
```

### 7.3 Writing Standards

| Current problem | Required fix |
|----------------|-------------|
| "I spent some time..." | Remove all first-person narrative |
| "It turned out that..." | State facts directly |
| "Part I" / "Part II" as section labels | Use descriptive section names |
| `\excludecomment{figure}` globally disabling figures | **Remove — figures must appear** |
| No formal Algorithm blocks | Add pseudocode for each method (use `algorithm2e` package) |
| HMC settings never clearly stated | Add explicit table: num_samples, burnin, step_size, leapfrog, mass matrix, acceptance target |
| Tone alternates between "we" and "I" | Consistent "we" throughout |
| Commented-out text blocks | Delete completely |
| Mixed tense | Present tense for methods, past tense for experiments |
| 8 models listed, only 2–3 used | Focus on primary models; move others to Appendix D |

### 7.4 Required Algorithm Blocks (pseudocode)

All nine algorithms must have formal `\begin{algorithm}` blocks:
1. Kalman Filter (predict + update)
2. Extended Kalman Filter
3. Unscented Kalman Filter
4. Bootstrap Particle Filter
5. EDH/LEDH Particle Flow Filter (with A(λ), b(λ))
6. LEDH Invertible Filter (with Jacobian accumulation)
7. OT Sinkhorn Resampling
8. HMC with Dual Averaging (full pseudocode)
9. DPF-HMC End-to-End Pipeline

### 7.5 Required Figures / Tables

- **Trace plots**: parameter value vs HMC iteration for every reported experiment
- **Posterior histograms**: already generated; need to be included in LaTeX
- **Convergence diagnostics table**: ESS, R-hat per parameter per method
- **Resampling comparison table** (differentiability vs. bias trade-off)
- **Summary comparison table**: all methods × all models (filtering + parameter estimation)
- **Step size evolution plot**: shows dual averaging adaptation

### 7.6 Required New Sections

**Section 4.3: Prior and Initialization Choices** (addresses feedback item #1)
- For each model: what is `p(x_0)`? When is N(0,1) appropriate?
- When to use stationary distribution (e.g., SV: `x_0 ~ N(0, σ²/(1-α²))`)
- Sensitivity analysis: effect of bad initial distribution on filter convergence

**Section 4.4: HMC Configuration** (addresses feedback item #2)
- Explicit table: num_burnin, num_samples, target_accept, step_size init, leapfrog steps, mass matrix type, gradient clipping threshold, PF seed, num_particles
- Rationale for each setting choice

**Section 5.4: Convergence Diagnostics** (new)
- R-hat and ESS tables for all experiments
- Autocorrelation plots
- What constitutes "converged" in this context

### 7.7 Primary Result: Linear-Gaussian as Reference Case

The complete Linear-Gaussian case must be shown end-to-end as the reference/validation:
- Data generation → filtering (all methods) → parameter estimation (all methods) → convergence diagnostics
- This validates the entire pipeline before showing harder models

### 7.8 Models in Paper vs. Appendix

**Main results** (Sections 4–5):
1. 1D Linear-Gaussian (complete; both 1-param and 2-param variants) — the validation benchmark
2. Stochastic Volatility — the primary nonlinear benchmark; future model variants planned; UKF improvement for SV in scope
3. Range-Bearing — secondary nonlinear benchmark; shows EKF/UKF success and PF limitations
4. Kitagawa — if results completed

**Cubic Sensor**: Used for testing (it is a genuinely difficult model). Whether it appears in the final report is TBD; do not assume it is excluded, but do not plan on it as a primary result.

**Move to Appendix D**:
- Acoustic Tracking, Lorenz-96, Two-Sensor Bearing, 5D Linear-Gaussian (reference configs)

---

## 8. Infrastructure

### 8.1 Local Machine (development)
- macOS Apple Silicon
- Python 3.14, TF 2.16 + tensorflow-metal (MPS backend)
- Venv at repo root: `.venv/`
- **No CUDA** — cannot run CUDA-specific tests locally

### 8.2 Remote Machine "office" (GPU training)
- Host: `office` (100.81.105.64), user: `haowu`
- GPU: NVIDIA RTX 3090, 24 GB VRAM
- Python 3.12.3, TF 2.20, TFP 0.25, tf-keras 2.20.1
- Venv: `~/JPML/code/.venv`
- Remote dir: `~/JPML/code` (NOT `~/JPMLCOE/code`)

**Required env vars on office**:
```bash
source ~/JPML/code/.venv/bin/activate
export LD_LIBRARY_PATH=...  # nvidia site-packages
export XLA_FLAGS=...        # xla_gpu_cuda_data_dir
export TF_USE_LEGACY_KERAS=1
```

**Known issue on office**: LEDH weight collapse on CUDA (float32 precision). Float64 test pending.

### 8.3 Sync Workflow (Makefile)
```bash
make push         # Sync code to office (excludes: .venv, outputs, __pycache__)
make push-dry     # Dry-run
make pull-results # Download outputs from office
make remote-setup # Initialize venv on office
```

### 8.4 Running Experiments

```bash
cd code
# Local (MPS)
python src/experiments/run_dpf_experiment.py +experiment=dpf/hmc/linear_gaussian/bpf_ot

# Remote (after make push)
ssh office "cd ~/JPML/code && [env vars] python src/experiments/run_dpf_experiment.py ..."
```

Shell scripts for batch runs: `code/run_hmc_linear_gaussian.sh`, `code/run_range_bearing_filters.sh`, etc.

---

## 9. Key Design Decisions and Constraints

### 9.1 TensorFlow Graph Mode
All core filter loops are in `@tf.function` with `tf.while_loop`. This is required for:
- Gradient computation through T timesteps without Python overhead
- Compatibility with TFP's HMC sampler (needs compiled graph)
- Do NOT use `tf.py_function` — it breaks the gradient tape

### 9.2 Fixed PF Seed in HMC
HMC uses `seed=[42, 0]` fixed for every likelihood evaluation. This creates a deterministic energy surface (good for HMC trajectory) but introduces systematic bias from one particle configuration. This is the Particle MCMC approach and is intentional.

### 9.3 No setup.py
The project does NOT use `setup.py`. Import paths are managed via `sys.path.insert()` in experiment scripts.

### 9.4 Float32 vs Float64
- Production code: float32 (TF default; faster on GPU)
- LEDH precision issue: 29-step Jacobian accumulation may need float64
- Key diagnostic: Test I in `test_ledh_gradient_diagnosis.py`

### 9.5 Stop-Gradient in OT Normalization
The OT resampling normalizes particles (center + scale) before solving Sinkhorn. The gradient through this normalization is optional (`stop_gradient_normalization` config flag). This is a configurable trade-off between gradient quality and bias.

---

## 10. Priority Task List (Current)

### CRITICAL (blockers)
1. **Run LEDH gradient diagnostic suite** (`tests/hmc/test_ledh_gradient_diagnosis.py`)
   - Start with Test A (likelihood surface) and Test B (autodiff vs FD)
   - Follow the decision tree in `LEDH_gradient_diagnosis.md`
   - The bug must be found and fixed before LEDH results can appear in the paper

### HIGH
2. **Range-bearing with T=200**: Complete the HMC run (T=200 gives enough signal)
3. **LEDH on range-bearing**: Re-run with updated pipeline once gradient bug is fixed
4. **Generate all trace plots**: Required for the report; already computed in `trace.csv` files

### MEDIUM
5. **Report restructure**: Follow `RESTRUCTURE_PLAN.md` phase plan (5 phases)
   - Phase 1: Structure (section renaming, enable figures, add algorithm package)
   - Phase 2: Algorithm blocks for all 9 methods
   - Phase 3: Experimental setup sections (4.2, 4.3, 4.4)
   - Phase 4: Results restructure + trace plots
   - Phase 5: Polish (tone, notation, cross-references)
6. **Stochastic Volatility HMC** (BPF only first; improve UKF for SV; LEDH needs score-based fix)
7. **Kitagawa HMC** (configs exist; benchmark needed)
8. **UKF improvement for SV** (planned future work; improves Kalman-family baseline on SV)

### LOW
8. **Score-based LEDH for SV**: Design in `score_based_flow_report.md`; substantial implementation work
9. **NUTS tuning**: Infrastructure exists; needs benchmarking
10. **Mass matrix auto-estimation**: Design in `MAP_implementation.md`

---

## 11. Key Reference Files

| File | Purpose |
|------|---------|
| `code/RESTRUCTURE_PLAN.md` | Detailed report restructuring plan with section-by-section instructions |
| `code/LEDH_gradient_diagnosis.md` | 9-test diagnostic plan for LEDH gradient bug |
| `code/hmc_benchmark_report.md` | Complete experimental results with interpretation |
| `code/rb_report.md` | Range-bearing specific diagnostics and open questions |
| `code/score_based_flow_report.md` | Design for fixing LEDH on SV (score-based generalization) |
| `code/MAP_implementation.md` | Auto mass matrix and MAP warm-start design |
| `code/OT_EPSILON_SCALING_PLAN.md` | OT epsilon sensitivity experiment design |
| `code/instruction.md` | TF/TFP API reference (every function used in the project) |
| `code/prior_selection_guide.md` | LogNormal prior tuning guide for different models |
| `report/main_polished.tex` | LaTeX report source (current: ~400+ lines, partially restructured) |

---

## 12. Frequently Confused / Important Distinctions

- **BPF-OT vs LEDH-OT**: BPF uses standard predict-weight-resample; LEDH replaces the weight step with a continuous particle flow. Both can use OT resampling, but they are fundamentally different algorithms.
- **stop_gradient vs differentiable**: `BPF-sys` uses `stop_gradient` through resampling (no gradient from resampling). `BPF-OT` and `BPF-soft` provide gradients through the resampling step. This is the core differentiable PF contribution.
- **ESS = num_samples for deterministic filters**: When ESS = 200 for Kalman/EKF/UKF with 200 samples, this means all samples are effectively independent — it is the *best possible* result, not a coincidence or bug.
- **Acceptance rate near 1.0 for Kalman**: Expected for smooth deterministic likelihood surface. Not suspicious.
- **R-hat < 1.0**: Possible when computing from a single split chain; values slightly below 1.0 (e.g., 0.85) just mean the two halves are very similar. Values < 1.1 indicate convergence.
- **Cubic sensor**: User explicitly said this is a bad model. Do not use in paper.
- **Remote dir**: `~/JPML/code` not `~/JPMLCOE/code`.
