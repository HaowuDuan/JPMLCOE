# Range-Bearing Model: HMC Inference Report

## Model Setup
- **Model**: Range-Bearing (2D), infers `sigma_range` and `sigma_bearing`
- **True parameters**: sigma_range=0.1, sigma_bearing=0.1
- **Initial values**: sigma_range=0.3, sigma_bearing=0.3
- **Prior**: LogNormal(loc=-2.3, scale=0.5) for both — median at 0.1, 95% interval [0.038, 0.27]
- **Data**: T=50 time steps, seed=42
- **Filter**: Bootstrap PF, 500 particles, OT entropy resampling (epsilon=0.5)

## HMC Runs

### Identity mass [1, 1] — bpf_ot
- Parameters converge fast: sigma_range=0.117, sigma_bearing=0.164 by step 21
- sigma_range close to truth (0.1), sigma_bearing lagging (0.164)
- Acceptance rate: dropped to 33% early, recovering to 52% by step 21
- Step size crushed to ~0.002-0.006 after early overshoot to 0.52
- ~55s per step, ~9 hours total for 600 steps

### Preconditioned mass [1, 5] — bpf_ot_precond
- sigma_range=0.089, sigma_bearing=0.162 by step 3, then barely moved
- Mass=5 on bearing dimension made it move even slower — wrong direction
- Gradient magnitudes similar in both dims (~32 vs ~27), so [1,5] is mismatched
- Acceptance rate dropped faster than [1,1]

### Conclusion on mass matrix
- [1, 1] outperforms [1, 5] — bearing already converges slower, extra mass makes it worse
- If preconditioning needed, mass on sigma_range (steeper) not sigma_bearing

## MAP Diagnostic

### Both parameters free (bpf_ot, T=50)
- Does NOT converge to (0.1, 0.1)

### sigma_range only (freeze sigma_bearing=0.1, T=50)
- Converges to sigma_range ~ 0.208, NOT 0.1
- Gradient goes to zero at the wrong spot (|grad|=0.14 and shrinking at step 138)
- Loss plateaus at -14.37, ll=15.77
- **This is a fundamental problem**: the likelihood surface optimum is in the wrong place

### sigma_range only (freeze sigma_bearing=0.1, T=200)
- At step 131: sigma_range=0.160, |grad|=82.54 — still moving (not stuck)
- Compare T=50: stuck at 0.208 with |grad|->0 by step 131
- **Confirms T=50 had too few observations** — likelihood surface too flat to distinguish 0.2 from 0.1
- T=200 gives enough signal for gradient to keep pushing toward truth
- 16s/step (~4x the T=50 cost, as expected)

### sigma_bearing only (freeze sigma_range=0.1, T=200)
- Not yet run

## Current Config Changes
- Leapfrog steps: 10 -> 5 for all range-bearing BPF configs (halves wall-clock time)
- Burn-in/samples: 200/400 -> 100/200 for bpf_ot and bpf_soft (faster iteration)
- Target acceptance: 0.65 across the board

## Saved HMC Results (from output/.hydra/config.yaml)

### Feb 25 results (T=50, old configs with target_accept=0.9, leapfrog=5 for EKF/UKF/sys)

| Filter       | sigma_range | sigma_bearing | Accept | Step Size | Notes         |
|--------------|-------------|---------------|--------|-----------|---------------|
| EKF          | 0.116       | 0.113         | 97%    | 0.42      | Works well    |
| UKF          | 0.115       | 0.112         | 98%    | 0.39      | Works well    |
| BPF sys      | 0.287       | 0.290         | 93%    | 1.7e-5    | FAILED        |
| BPF soft     | 0.298       | 0.300         | 92%    | 9.4e-6    | FAILED        |

### Mar 2 bpf_ot result (T=50, leapfrog=10, target_accept=0.65, burnin=200, samples=400)
- sigma_range: mean=0.181, std=0.019
- sigma_bearing: mean=0.086, std=0.010
- Accept=75.5%, step_size=0.019
- ESS: 21.4 (range), 19.8 (bearing) — very low
- R-hat: 1.018 / 1.004

### Key observation
- EKF/UKF converge correctly (~0.11) — the data and model are fine
- All PF variants struggle — systematic bias in PF likelihood gradients for range-bearing
- bpf_ot (Mar 2) is best PF result but still biased (0.18 vs 0.1 for range)

## MAP seed issue
- MAP uses `random_seed: true` -> different PF seed each step (stochastic gradient)
- HMC uses fixed seed `[42, 0]` -> deterministic likelihood surface
- Mismatch: MAP optimizes a stochastic average, HMC samples a fixed surface
- Should test `random_seed: false` in MAP to match what HMC sees

## Open Questions
1. **PF gradient bias**: all PF variants (sys, soft, OT) fail on range-bearing. Not OT-specific.
2. **T sensitivity**: T=50 gets stuck at 0.208, T=200 reaches 0.160 but also stalls.
3. **MAP seed**: random vs fixed seed may explain MAP/HMC discrepancy.
4. **Is range-bearing too hard for PF-based HMC?** EKF/UKF work, PFs don't.

## Experiments In Progress
- Re-running `run_range_bearing.sh` with updated configs

## Fallback Models (if range-bearing doesn't work)

Requirements: nonlinear (PF needed), smooth likelihood, 1-2 parameters, low-dim state.
Cubic sensor is excluded (bad model per user).

### 1. Stochastic Volatility (already implemented)
- x_t = phi*x_{t-1} + sigma_v*v_t (log-volatility)
- y_t = beta*exp(x_t/2)*e_t (returns)
- 1D state, infer phi and/or sigma_v
- Classic PF benchmark, smooth nonlinearity (exp)
- Already have configs — check if HMC works there first

### 2. Nonlinear Growth Model (UNGM)
- x_t = 0.5*x_{t-1} + 25*x_{t-1}/(1+x_{t-1}^2) + 8*cos(1.2*t) + v_t
- y_t = x_t^2/20 + w_t
- 1D state, infer process noise sigma_v and/or obs noise sigma_w
- THE standard PF benchmark — used in nearly every DPF paper
- Bimodal transition makes it hard for Kalman filters (justifies PF)
- Smooth observation (x^2) gives decent gradients

### 3. Nonlinear Pendulum
- State: [angle, angular_velocity]
- Dynamics: ddtheta = -g/L*sin(theta) + noise
- Observation: sin(theta) + noise (bearing-like but simpler)
- 2D state, infer damping or noise params
- Trigonometric but simpler geometry than range-bearing

### 4. Kitagawa (already implemented)
- Check existing results — may already work
