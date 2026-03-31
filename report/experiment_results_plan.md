# Plan: Adding Experiment Results to main_reorganized.tex

## Overview

All tables in the current tex are commented out. The plan is to uncomment, update with latest data, and add discussion text matching the author's writing style (concise, first-person where appropriate, technically precise, no fluff). Each subsection below specifies exactly what goes where, which output files to pull from, and whether re-runs are needed.

**All experiments now standardized to T=100.**

---

## Cross-cutting: Initial Conditions (add to Section 2.4 or as a new subsection in Section 2)

The report already discusses initial conditions for the 1D linear case (lines 160-171): stationary variance $P_0 = Q/(1-a^2)$ when $|a|<1$, diffuse prior when $|a|\geq 1$. This principle must be applied consistently across all models. Add a paragraph or table summarizing the initial condition choice for each model and **why** it is correct. This is important because a wrong initial condition can look like a filter failure.

### Initial condition choices per model

| Model | $\mu_0$ | $\Sigma_0$ | Rationale |
|-------|---------|------------|-----------|
| **1D Linear** ($a=0.95, B=0.5$) | $0$ | $Q/(1-a^2) = 0.25/0.0975 \approx 2.56$ | Stationary variance of AR(1). Process is stable ($|a|<1$), so the stationary distribution exists and is the natural starting point. Using $I$ would be overconfident. Solved via discrete Lyapunov equation $P = F P F^T + Q$ in code (`scipy.linalg.solve_discrete_lyapunov`). |
| **5D Linear** (all variants) | $\mathbf{0}$ | Lyapunov solution of $P = F P F^T + Q$ | Same principle as 1D but multivariate. The discrete Lyapunov equation generalizes the scalar formula. When $\Sigma_0$ is not explicitly provided in the config, the code computes it; when provided, the config value is used. |
| **Stochastic Volatility 1D** ($\alpha=0.91, \sigma=1.0$) | $0$ | $\sigma^2/(1-\alpha^2) \approx 5.26$ | Exact stationary variance of the AR(1) log-volatility process. Already documented in the report (line 354). |
| **Stochastic Volatility 2D** | $\mathbf{0}$ | Lyapunov solution | Two coupled AR(1) processes. The 2D model code solves the discrete Lyapunov equation, same as linear Gaussian. |
| **Range-Bearing** | $(5.0, 5.0)^T$ | $I_2$ | No stationary distribution — the state is a physical position, not a mean-reverting process. The prior must be chosen to be consistent with the sensor geometry. We use $\mu_0 = (5,5)$ (range $\approx 7.07$ from sensor at origin) rather than $(1,1)$ (range $\approx 1.41$) because at short range the bearing Jacobian $\partial \text{atan2}/\partial x$ has a singularity, causing EKF/UKF instability. The covariance $I_2$ expresses moderate uncertainty around a physically reasonable starting point. |
| **Acoustic Tracking** (4 targets) | $(20,20,0,0)^T$ | $\text{diag}(33.33, 33.33, 1, 1)$ | Position centered in the sensor grid $[0,40]^2$; velocity centered at zero. $\Sigma_0$ is a Gaussian prior consistent with $\mu_0$. Particles are sampled from this Gaussian via Cholesky decomposition — NOT from a uniform distribution, which would be inconsistent with $\mu_0/\Sigma_0$ and confuse filters that use these moments directly. |
| **Kitagawa** | $0$ | $5.0$ | Canonical choice from Andrieu et al. (2010). Close to the true stationary variance ($\approx 5.26$). |

### Discussion points to add:
- **Why this matters:** A filter initialized at the wrong distribution will spend the first several timesteps "catching up" to the truth. For short time series (T=100), this transient dominates the RMSE. For linearization-based filters (EKF, UKF, flow), a bad initial condition can cause permanent divergence because the linearization point is wrong from the start.
- **Stationary vs non-stationary models:** For models with stable state dynamics ($|a|<1$ or spectral radius of $F < 1$), the stationary distribution is the only principled choice. For tracking models (range-bearing, acoustic tracking), there is no stationary distribution — the initial condition is a modeling choice that encodes prior knowledge about where the target starts.
- **Lyapunov equation:** For multivariate linear Gaussian models, the stationary covariance solves $P = F P F^T + Q$. This is the discrete Lyapunov equation, solved in our code via `scipy.linalg.solve_discrete_lyapunov`. Using identity $I$ when the true stationary covariance is $\approx 2.56$ (1D case) or has off-diagonal structure (5D case) introduces systematic bias in the first few filter steps.
- **Range-bearing $\mu_0$ choice:** Moving the default from $(1,1)$ to $(5,5)$ avoids the bearing Jacobian singularity near the sensor. At range $r$, the bearing Jacobian scales as $1/r^2$, so at $r=1.41$ the Jacobian is $\sim 25\times$ larger than at $r=7.07$, causing EKF/UKF covariance updates to overshoot. This is not a filter bug — it is a consequence of the geometry.
- **Acoustic tracking sampling consistency:** The model defines $\mu_0$ and $\Sigma_0$ as Gaussian parameters. Sampling initial particles from a different distribution (e.g., uniform over the grid) creates a mismatch: the EKF/UKF guide filter uses $\mu_0/\Sigma_0$ but the particles are drawn from something else. Our code uses Cholesky-based Gaussian sampling to keep these consistent.

### Where to place in the report:
Option A: Expand the existing initial condition discussion (lines 160-171) into a general principle, then reference it in each model's definition section.
Option B: Add a brief paragraph in each model's experiment subsection explaining the initial condition choice.
Recommendation: Option A — state the principle once, then each model definition just states its $\mu_0, \Sigma_0$ with a one-line justification.

---

## Section 2: Kalman Filter (line ~344)

### 2.4 Numerical experiments and failure on nonlinear models

**What to add:** Two result summaries showing KF fails on both SV and range-bearing.

| Filter | Model | Output Path | RMSE | Log-Lik | Status |
|--------|-------|-------------|------|---------|--------|
| KF | Stochastic Volatility | `stochastic_volatility/stochastic_volatility_kf/` | Check summary.json | Check | **Exists** |
| KF | Range-Bearing | `range_bearing/range_bearing_kf/` | 1.42 | -3026.1 | **Exists** |

**Figures:** `plot.png` from each directory. Currently commented out (lines 405-414). Uncomment and point to correct paths.

**Discussion points:**
- KF on SV: h(x) = 0, so K_k = 0 always. Filter just propagates prior. RMSE should be terrible.
- KF on range-bearing: linearization at prior mean is poor because h is nonlinear (atan2, sqrt). Filter diverges.
- This motivates EKF/UKF.

**Re-run needed:** No.

---

## Section 3: EKF and UKF (line ~556)

### 3.3 EKF/UKF on non-linear models

**What to add:** Comparison table for EKF and UKF on both models.

| Filter | Model | Output Path | RMSE | Log-Lik | Status |
|--------|-------|-------------|------|---------|--------|
| EKF | SV (raw) | `stochastic_volatility/stochastic_volatility_ekf/` | 2.54 | -8731.1 | **Exists** |
| UKF | SV (raw) | `stochastic_volatility/stochastic_volatility_ukf/` | 2.54 | -8731.1 | **Exists** |
| EKF | SV (log-transform) | `stochastic_volatility/stochastic_volatility_ekf_log/` | Check | Check | **Exists** |
| UKF | SV (log-transform) | `stochastic_volatility/stochastic_volatility_ukf_log/` | Check | Check | **Exists** |
| EKF | Range-Bearing | `range_bearing/range_bearing_ekf/` | 0.42 | -760.0 | **Exists** |
| UKF | Range-Bearing | `range_bearing/range_bearing_ukf/` | 0.64 | -573.8 | **Exists** |

**Discussion points:**
- EKF/UKF on SV (raw): both fail identically because h'(x) near zero at prior mean gives negligible info gain. RMSE same as KF.
- EKF/UKF on SV (log-transform): the log-squared trick linearizes the observation. Should improve substantially.
- EKF/UKF on range-bearing: EKF outperforms UKF here because range-bearing Jacobian is well-behaved near truth. Both work, unlike KF.
- Include runtime and memory from performance metadata.

**Figures:** Uncomment the figures at lines 405-414. Plot.png from each run.

**Re-run needed:** No.

---

## Section 4: Particle-Based Methods

### 4.2 Bootstrap Particle Filter (line ~635)

#### 4.2.2 Particle Degeneracy and Resampling

**What to add:** BPF results on SV and range-bearing showing it works where EKF/UKF struggled.

| Filter | Model | Output Path | RMSE | Log-Lik | Particles | Resample Rate | Status |
|--------|-------|-------------|------|---------|-----------|---------------|--------|
| BPF | SV | `stochastic_volatility/stochastic_volatility_pf/` | 1.13 | -546.5 | 1000 | 0.364 | **Exists** |
| BPF | Range-Bearing | `range_bearing/range_bearing_pf/` | 0.087 | 326.6 | 1000 | ? | **Exists** |
| BPF | Acoustic Tracking | `acoustic_tracking/acoustic_tracking_pf/` | 22.2 | -5708.5 | 1000 | 0.92 | **Exists** |

**Discussion points:**
- BPF on SV: RMSE 1.13 vs EKF/UKF's 2.54. No linearization needed.
- Weight collapse discussion: resampling rate 0.364 means 36% of steps triggered resampling. This is manageable for 1D SV.
- Acoustic tracking: high resampling rate (0.92) signals severe particle degeneracy in high dimensions (16D state, 25 sensors). BPF struggles.
- Discuss ESS trajectory and weight histograms from the plots.
- Runtime comparison: BPF is fast (3s for 500 steps on SV) but needs many particles.

**Figures:** Uncomment lines 697-723. Point to `plot.png` from each run.

**Re-run needed:** No.

---

### 4.3 EDH/LEDH Flow Filter (line ~888)

#### 4.3.5 Experiments

**What to add:** Uncomment and update Tables at lines 896-951 (range-bearing comparison, acoustic tracking comparison).

##### Range-Bearing Comparison Table

| Filter | Output Path | RMSE | Particles | Status |
|--------|-------------|------|-----------|--------|
| PF | `range_bearing/range_bearing_pf/` | 0.087 | 1000 | **Exists** |
| EDH Flow | `range_bearing/range_bearing_edh_flow/` | 0.075 | 500 | **Exists** |
| EDH Invertible | `range_bearing/range_bearing_edh_invertible/` | 0.096 | 500 | **Exists** |
| LEDH Flow | `range_bearing/range_bearing_ledh_flow/` | 0.093 | 500 | **Exists** |
| LEDH Invertible | `range_bearing/range_bearing_ledh_invertible/` | 0.088 | 500 | **Exists** |
| EDH Flow (global) | `range_bearing/range_bearing_edh_flow_global/` | 1.31 | 500 | **Exists** |

**Discussion:** EDH flow with intermediate re-linearization works well on range-bearing. Global linearization fails (RMSE 1.31). LEDH variants competitive with BPF but use fewer particles. Include wall time and memory columns.

##### Acoustic Tracking Comparison Table

| Filter | Output Path | RMSE | Status |
|--------|-------------|------|--------|
| PF | `acoustic_tracking/acoustic_tracking_pf/` | 22.2 | **Exists** |
| LEDH Flow | `acoustic_tracking/acoustic_tracking_ledh_flow/` | ? | **Exists** |
| LEDH Invertible | `acoustic_tracking/acoustic_tracking_ledh_invertible/` | ? | **Exists** |
| Stochastic EDH | `acoustic_tracking/acoustic_tracking_stochastic_edh/` | ? | **Exists** |

**Discussion:** In higher dimensions, global EDH completely fails. LEDH variants dramatically outperform because per-particle linearization handles the nonlinearity better. Include geometric lambda schedule results from the `ledh_invertible_q*` directories.

##### Stochastic Volatility Comparison Table (line ~980)

| Filter | Output Path | RMSE | Status |
|--------|-------------|------|--------|
| PF | `stochastic_volatility/stochastic_volatility_pf/` | 1.13 | **Exists** |
| EKF | `stochastic_volatility/stochastic_volatility_ekf/` | 2.54 | **Exists** |
| UKF | `stochastic_volatility/stochastic_volatility_ukf/` | 2.54 | **Exists** |
| EDH Flow | `stochastic_volatility/stochastic_volatility_edh_flow/` | 2.58 | **Exists** |
| Stochastic EDH | `stochastic_volatility/stochastic_volatility_stochastic_edh/` | 2.60 | **Exists** |
| LEDH Flow (log) | `stochastic_volatility/stochastic_volatility_ledh_flow_log/` | ? | **Exists** |
| LEDH Invertible (log) | `stochastic_volatility/stochastic_volatility_ledh_invertible_log/` | 1.28 | **Exists** |
| Kernel (scalar) | `stochastic_volatility/stochastic_volatility_kernel_scalar/` | 2.68 | **Exists** |
| Kernel (matrix) | `stochastic_volatility/stochastic_volatility_kernel_matrix/` | 2.68 | **Exists** |

**Discussion:** ALL flow filters fail on raw SV (RMSE ~2.5-2.6) because the observation function has near-zero Jacobian at the prior mean. The log-transform helps flow filters: LEDH invertible (log) achieves 1.28 vs PF's 1.13. BPF wins because it avoids linearization entirely. This is the key narrative: flow methods require informative Jacobians.

**Re-run needed:** No. All data exists.

**Question for author:** The `_log_ukf` variants (e.g., `stochastic_volatility_edh_flow_log_ukf`) — should these be included in the SV table? They use UKF as the guide filter instead of EKF.

---

### 4.4 Invertible Flow Filter (line ~1022)

**What to add:** No separate experiment subsection currently exists. The results are folded into the EDH experiments table. The invertible variants (`edh_invertible`, `ledh_invertible`) are already covered above.

**Discussion to add after Algorithm 6 (line ~1115):**
- Weight behavior: compare resampling rates between flow-only (EDH) and invertible (EDH+Jacobian) variants.
- On range-bearing: EDH invertible slightly worse than EDH flow (0.096 vs 0.075) — Jacobian accumulation over many steps introduces noise.
- On acoustic tracking: LEDH invertible best overall.

**Re-run needed:** No.

---

### 4.5 Stochastic Flow Filter (line ~1117)

**What to add:** Results showing stochastic EDH on range-bearing with and without local correction.

| Filter | Output Path | RMSE | Status |
|--------|-------------|------|--------|
| Stochastic EDH (no correction) | `range_bearing/range_bearing_stochastic_edh/` | 1.31 | **Exists** |
| Stochastic EDH (100 steps) | `range_bearing/range_bearing_stochastic_edh_100steps/` | 1.31 | **Exists** |
| SDE Local Correction | `range_bearing/range_bearing_sde_local_correction/` | 0.25 | **Exists** |
| SDE Local Correction (100 steps) | `range_bearing/range_bearing_sde_local_correction_100steps/` | 0.25 | **Exists** |
| SDE Local Correction (optimal) | `range_bearing/range_bearing_sde_local_correction_optimal/` | 0.25 | **Exists** |

**Discussion:** Without local correction, stochastic EDH completely diverges (RMSE 1.31, same as KF). With local correction, it works (RMSE 0.25), though worse than deterministic flow variants. The "optimal" stiffness mitigation schedule did not improve over fine uniform steps. Uncomment figures at lines 1159-1171.

**Re-run needed:** No.

---

### 4.6 Kernel Flow Filter (line ~1175)

**What to add:** Results on SV and range-bearing.

| Filter | Model | Output Path | RMSE | Particles | Wall Time | Status |
|--------|-------|-------------|------|-----------|-----------|--------|
| Kernel (scalar) | SV | `stochastic_volatility/stochastic_volatility_kernel_scalar/` | 2.68 | 20 | 304s | **Exists** |
| Kernel (matrix) | SV | `stochastic_volatility/stochastic_volatility_kernel_matrix/` | 2.68 | 20 | ? | **Exists** |
| Kernel (scalar) | SV (log) | `stochastic_volatility/stochastic_volatility_kernel_scalar_log/` | ? | 20 | ? | **Exists** |
| Kernel (matrix) | SV (log) | `stochastic_volatility/stochastic_volatility_kernel_matrix_log/` | ? | 20 | ? | **Exists** |
| Kernel (scalar) | RB | `range_bearing/range_bearing_kernel_scalar/` | 0.082 | 20 | 51s | **Exists** |
| Kernel (matrix) | RB | `range_bearing/range_bearing_kernel_matrix/` | 0.080 | 20 | 44s | **Exists** |
| Kernel (scalar) | Lorenz96 | `lorenz96/lorenz96_kernel_scalar/` | ? | ? | ? | **Exists** |
| Kernel (matrix) | Lorenz96 | `lorenz96/lorenz96_kernel_matrix/` | ? | ? | ? | **Exists** |

**Discussion:**
- Kernel methods competitive on range-bearing with only 20 particles (vs BPF's 1000).
- On SV: fails like all linearization-based methods (RMSE 2.68).
- Matrix kernel vs scalar kernel: marginal difference in accuracy but matrix kernel prevents collapse in higher dimensions.
- Very slow: 304s for 500 steps of 1D SV with only 20 particles. The iterative ODE solve dominates cost.
- Discuss memory: 307 MB peak for scalar kernel on SV (vs 6 MB for BPF). The kernel matrix is N x N.

**Re-run needed:** No.

---

### 4.7 Filter Comparison Summary (line ~1373)

**What to add:** A comprehensive summary table across all filters and models. Pull from all the above.

**Format:** One table per model (SV, Range-Bearing, Acoustic Tracking) with columns: Filter, RMSE, Log-Lik, Wall Time, Time/Step, Peak Mem, Particles.

**Discussion:** The current text (line 1374) is good but brief. Expand with:
- Runtime/memory comparison: flow methods are 10-100x slower than BPF. Kernel methods are the slowest.
- Memory scaling: kernel methods O(N^2), flow methods O(N*d), BPF O(N).
- The "no free lunch" message is correct. Add concrete numbers to support it.

**Re-run needed:** No.

---

## Section 5: Resampling and Optimal Transport (line ~1381)

### 5.4 Soft Resampling and OT Resampling Comparison (line ~1499)

**What to add:** Uncomment table at lines 1501-1524.

| Method | Param | Output Path | RMSE | Log-Lik | Wall Time | Status |
|--------|-------|-------------|------|---------|-----------|--------|
| PF (Systematic) | --- | `stochastic_volatility/stochastic_volatility_pf/` | 1.13 | -546.5 | 3.1s | **Exists** |
| OT eps=0.1 | eps=0.1 | `stochastic_volatility/stochastic_volatility_pf_ot_eps0.1/` | 1.14 | -548.7 | 203s | **Exists** |
| OT eps=0.3 | eps=0.3 | `stochastic_volatility/stochastic_volatility_pf_ot_eps0.3/` | ? | ? | ? | **Exists** |
| OT eps=0.5 | eps=0.5 | `stochastic_volatility/stochastic_volatility_pf_ot_eps0.5/` | ? | ? | ? | **Exists** |
| OT eps=1.0 | eps=1.0 | `stochastic_volatility/stochastic_volatility_pf_ot_eps1.0/` | ? | ? | ? | **Exists** |
| Soft alpha=0.5 | alpha=0.5 | `stochastic_volatility/stochastic_volatility_pf_soft_alpha0.5/` | 1.13 | -545.9 | 4.9s | **Exists** |
| Soft alpha=0.7 | alpha=0.7 | `stochastic_volatility/stochastic_volatility_pf_soft_alpha0.7/` | ? | ? | ? | **Exists** |
| Soft alpha=0.9 | alpha=0.9 | `stochastic_volatility/stochastic_volatility_pf_soft_alpha0.9/` | ? | ? | ? | **Exists** |

Also include 5D linear results for resampling comparison:

| Method | Output Path | Status |
|--------|-------------|--------|
| PF (sys) | `5d_linear_partial_strong/5d_partial_strong_pf/` | **Exists** |
| PF (OT) | `5d_linear_partial_strong/5d_partial_strong_pf_ot/` | **Exists** |
| PF (soft) | `5d_linear_partial_strong/5d_partial_strong_pf_soft/` | **Exists** |
| LEDH inv (OT) | `5d_linear_partial_strong/5d_partial_strong_ledh_invertible_ot/` | **Exists** |
| LEDH inv (soft) | `5d_linear_partial_strong/5d_partial_strong_ledh_invertible_soft/` | **Exists** |

**Discussion:**
- For pure filtering on SV: systematic resampling is already excellent. OT resampling is 60-100x slower with similar or worse RMSE.
- OT's real value is not filtering accuracy but **differentiability** for parameter estimation (Section 6).
- Soft resampling is cheap and competitive.
- Runtime/memory discussion: OT's cost is dominated by Sinkhorn iterations. Memory scales O(N^2) for cost matrix.
- Show that on 5D linear (partial observation), the picture may differ — more dimensions may benefit from OT's diversity preservation.

**Re-run needed:** No. All data exists.

---

## Section 6: Parameter Estimation with Differentiable Particle Filters (line ~1532)

### 6.4 Benchmark: Linear Gaussian Model (line ~1660)

#### 6.4.2 1-D Model, 1 Parameter (line ~1700)

**What to add:** Fill in "(Numerical results to be filled.)" at line 1713.

| Method | Filter | Resampling | Output Path | Status |
|--------|--------|------------|-------------|--------|
| MAP | LEDH (sys) | systematic | `dpf/map/linear_gaussian/ledh_sys/` | **Exists** |
| MAP | LEDH (soft) | soft | `dpf/map/linear_gaussian/ledh_soft/` | **Exists** |
| MAP | LEDH | default | `dpf/map/linear_gaussian/ledh/` | **Exists** |
| HMC | Kalman | --- | `dpf/hmc/linear_gaussian/kalman/` | **Exists** |
| HMC | EKF | --- | `dpf/hmc/linear_gaussian/ekf/` | **Exists** |
| HMC | UKF | --- | `dpf/hmc/linear_gaussian/ukf/` | **Exists** |
| HMC | BPF (sys) | systematic | `dpf/hmc/linear_gaussian/bpf_sys/` | **Exists** |
| HMC | BPF (soft) | soft | `dpf/hmc/linear_gaussian/bpf_soft/` | **Exists** |
| HMC | BPF (OT) | OT | `dpf/hmc/linear_gaussian/bpf_ot/` | **Exists** |
| HMC | LEDH (sys) | systematic | `dpf/hmc/linear_gaussian/ledh_sys/` | **Exists** |
| HMC | LEDH (soft) | soft | `dpf/hmc/linear_gaussian/ledh_soft/` | **Exists** |
| HMC | LEDH (OT) | OT | `dpf/hmc/linear_gaussian/ledh_ot/` | **Exists** |

**Figures:** `posterior_histograms.png` and `trace_plot.png` from each run.

**Discussion:**
- MAP convergence plot: show loss curves converging to truth.
- HMC diagnostics: acceptance rate, ESS, R-hat from summary.json.
- Compare Kalman (ground truth) vs particle-based posteriors.
- Show that all three resampling methods recover the correct parameter on this simple model.
- Runtime comparison for HMC: OT much slower per iteration but gradient quality may differ.

**Re-run needed:** No.

#### 6.4.3 Multi-Dimensional Model, 1 Parameter (line ~1715)

**What to add:** Fill in "(Results to be filled.)" at line 1717.

| Method | Filter | Output Path | Status |
|--------|--------|-------------|--------|
| HMC | EKF | `dpf/hmc/linear_gaussian_full/ekf/` | **Exists** |
| HMC | UKF | `dpf/hmc/linear_gaussian_full/ukf/` | **Exists** |
| HMC | BPF (sys/soft/OT) | `dpf/hmc/linear_gaussian_full/bpf_*/` | **Exists** |
| HMC | LEDH (sys/soft/OT) | `dpf/hmc/linear_gaussian_full/ledh_*/` | **Exists** |

**Discussion:** Same structure as 1D but tests scaling. Report posterior mean/std, ESS, R-hat.

**Re-run needed:** No.

---

### 6.5 (Commented out) Benchmark: Stochastic Volatility (line ~1721)

**What to add:** Uncomment and fill. Data exists but may be incomplete.

| Method | Filter | Output Path | Status |
|--------|--------|-------------|--------|
| HMC | BPF (sys) | `dpf/hmc/stochastic_volatility/bpf_sys/` | **Exists** |
| HMC | BPF (soft) | `dpf/hmc/stochastic_volatility/bpf_soft/` | **Exists** |
| HMC | BPF (OT) | `dpf/hmc/stochastic_volatility/bpf_ot/` | **Exists** |
| HMC | BPF (sys, small) | `dpf/hmc/stochastic_volatility/bpf_sys_small/` | **Exists** |
| HMC | BPF (soft, small) | `dpf/hmc/stochastic_volatility/bpf_soft_small/` | **Exists** |
| HMC | BPF (OT, small) | `dpf/hmc/stochastic_volatility/bpf_ot_small/` | **Check** |

**Question:** The `bpf_sys_long` directory has only a log — may have failed. Check if enough SV HMC results exist for a meaningful table.

**Re-run needed:** Possibly. Check if `bpf_ot_small` completed. If SV DPF results are thin, may need re-runs for a complete story.

---

### Additional DPF results: Cubic Sensor, Kitagawa, Range-Bearing

These models have extensive HMC results in `dpf/hmc/`:

| Model | Filters Available | Output Path | Status |
|-------|-------------------|-------------|--------|
| Cubic Sensor | BPF (sys/soft/OT), LEDH (sys/soft/OT), EKF, UKF | `dpf/hmc/cubic_sensor/` | **Exists** |
| Kitagawa | BPF (sys/soft/OT), LEDH (sys/soft/OT) | `dpf/hmc/kitagawa/` | **Exists** |
| Range-Bearing | BPF (sys/soft/OT), LEDH (sys/soft/OT), EKF, UKF | `dpf/hmc/range_bearing/` | **Exists** |

There are also `dpf_smoke/` variants (with `_lc` = log-correction?) for cubic sensor and range-bearing.

**Question for author:** Should cubic sensor and Kitagawa models be included in the report? They are not currently mentioned in the filtering sections. The models are defined in the codebase but not in the tex. If yes, need to add model definitions first. The Kitagawa model is particularly interesting for bimodality (see `kitagawa_ledh_invertible_bimodal_k*` directories).

**Re-run needed:** No for existing data. Yes if cubic sensor/Kitagawa models are to be included — need filtering result tables too.

---

## Additional Data Available But Not Currently in Report

### Models with filtering results not in the report:

1. **5D Linear (full observation):** `outputs/5d_linear/` — all filters. Good for showing flow methods scale to moderate dimensions.
2. **5D Linear (partial, weak/strong):** `outputs/5d_linear_partial_weak/`, `outputs/5d_linear_partial_strong/` — tests partial observation.
3. **2D Stochastic Volatility:** `outputs/stochastic_volatility_2d/` — all flow variants with and without UKF guide. Important for showing multi-dimensional SV.
4. **Kitagawa:** `outputs/kitagawa/` — all flow variants including bimodal experiments.
5. **Cubic Sensor:** `outputs/cubic_sensor/` — EKF, UKF, PF (sys/soft/OT).
6. **Lorenz96:** `outputs/lorenz96/` — kernel methods only.
7. **Two-Sensor Bearing:** `outputs/two_sensor_bearing/` — Monte Carlo comparison.

**Question for author:** Which of these should be included? The 5D linear and 2D SV seem most relevant to the narrative. Kitagawa is good for bimodality story. Cubic sensor is good for DPF story since HMC results exist.

---

## Runtime and Memory Discussion Strategy

Every `summary.json` contains `performance.wall_time_seconds`, `performance.time_per_timestep_ms`, `performance.peak_memory_mb`, and `performance.memory_increase_mb`. The approach:

1. **Per-table:** Include Wall Time, Time/Step, and Peak Mem columns in every comparison table.
2. **Cross-cutting discussion (in Section 4.7 Filter Comparison Summary):** A paragraph comparing the runtime/memory profiles:
   - BPF: fast, low memory, but needs many particles in high dimensions.
   - EDH/LEDH Flow: 10-50x slower than BPF, moderate memory. LEDH is N times more expensive than EDH because of per-particle linearization.
   - Invertible variants: similar cost to flow + small overhead for Jacobian accumulation.
   - Kernel: slowest (100-600ms/step), highest memory (O(N^2) kernel matrix). Only 20 particles but each step is iterative.
   - OT resampling: 60-100x overhead vs systematic. Memory O(N^2) for cost matrix.
3. **In DPF section:** OT resampling cost is amortized over the MCMC chain — pay once per gradient evaluation. Compare HMC wall time across resampling methods.

---

## Open Questions for Author

1. **SV with UKF guide variants (`_log_ukf`):** Include in tables? These pair log-transformed SV model with UKF as the guide filter for flow methods.
2. **Cubic Sensor and Kitagawa in report?** The DPF section has HMC results for both but the models are not defined in the filtering sections.
3. **5D linear and 2D SV:** Include as separate subsections or just mention in comparison summary?
4. **Two-sensor bearing Monte Carlo:** What narrative does this serve? Currently only 2 runs (mc_linear, mc_optimal).
5. **Lorenz96 kernel results:** Include as a high-dimensional showcase for kernel methods?
6. **DPF smoke tests vs full runs:** The `dpf_smoke/` directory has many results with `_lc` variants. Are these the canonical results or just smoke tests to exclude?
7. **Particle Gibbs (pgibbs):** Results exist for Kitagawa. Include in DPF section?
8. **SV DPF completeness:** Some SV HMC runs may have failed (bpf_sys_long has no summary.json). Need to verify and possibly re-run.

---

## Codex Review Findings — Status

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | T mismatch across runs | Critical | **RESOLVED** — all configs set to T=100, re-runs pending |
| 2 | Missing summary.json for some runs | High | **RESOLVED** — identified 3 missing (acoustic edh_inv, acoustic ledh_inv, sv kf_log); re-runs in `run_all_filters.sh` |
| 3 | SV UKF parameterization (alpha) | High | Needs verification after re-run |
| 4 | Raw-vs-log SV conflation | High | **ACKNOWLEDGED** — plan now separates raw and log tables |
| 5 | MAP convergence claims unsupported | High | Deferred (DPF section not in this iteration) |
| 6 | SV DPF artifacts suspicious | High | Deferred (DPF section not in this iteration) |
| 7 | Figure slot collision (Sec 2 vs 3) | Medium | Add dedicated KF figures in Section 2 |
| 8 | Causal claims too strong | Medium | Phrase as observations, not conclusions |
| 9 | Tables too wide (null log-lik) | Medium | Use focused accuracy table + one runtime/memory table |
| 10 | Execution order suboptimal | Medium | Reordered below |
| 11 | Too many DPF figures | Medium | Deferred (DPF section not in this iteration) |

---

## Execution Order

1. ~~Read all remaining `summary.json` files to fill in the `?` entries above.~~ → Re-run all experiments at T=100 first.
2. Resolve open questions with author.
3. After re-runs complete, extract metrics from new summary.json files.
4. Write the initial condition discussion (expand lines 160-171).
5. Uncomment existing tables, update numbers from new data.
6. Add discussion paragraphs after each table.
7. Add figures: uncomment existing `\includegraphics` blocks, update paths.
8. Write the runtime/memory discussion paragraphs.
9. DPF section deferred to next iteration.
