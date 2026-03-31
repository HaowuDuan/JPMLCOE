# Plan: Visualization Notebooks for Report Figures

## Overview

Create one notebook per report section, each producing publication-quality figures. All notebooks live in `code/analysis/` and save output to `report/figures/`. They read `.npy` files from `code/outputs/` (produced by `run_all_filters.sh` at T=100).

**Existing notebooks to keep/update:**
- `kalman.ipynb` — EKF vs UKF comparison (Sections 2-3)
- `particle.ipynb` — BPF weight collapse and ESS (Section 4.2)
- `hmc.ipynb` — HMC posterior diagnostics (Section 6, deferred)
- `figure3_kernel_comparison.ipynb` — Kernel filter reproduction (Section 4.6)

**Existing notebooks to delete (superseded):**
- `resampling.ipynb` — parses old log files, replaced by new notebook
- `resmpling.ipynb` — typo duplicate, empty or broken

**New notebooks to create:** 5

---

## Notebook 1: `kf_failure.ipynb` (Section 2.4)

**Purpose:** Show that KF fails on nonlinear models.

**Plots:**

1. **KF on Stochastic Volatility** — true state vs KF estimate. The KF estimate should be flat (K=0, no updates). 2-sigma band should be meaningless.
   - Data: `stochastic_volatility/stochastic_volatility_kf/{states,means,covs,observations}.npy`

2. **KF on Range-Bearing** — true state (2D) vs KF estimate. KF diverges because linearization at prior mean is wrong.
   - Data: `range_bearing/range_bearing_kf/{states,means,covs,observations}.npy`

3. **Side-by-side: KF vs EKF vs UKF on Range-Bearing** — show EKF/UKF track, KF doesn't.
   - Data: `range_bearing/range_bearing_{kf,ekf,ukf}/`

**Output files:** `report/figures/kf_failure_sv.pdf`, `report/figures/kf_failure_rb.pdf`, `report/figures/kf_ekf_ukf_rb.pdf`

---

## Notebook 2: `ekf_ukf_comparison.ipynb` (Section 3.3)

**Purpose:** EKF/UKF on both models, including log-transform trick for SV.

**Plots:**

1. **EKF vs UKF on SV (raw)** — both fail identically. Show flat estimates overlaid on true state.
   - Data: `stochastic_volatility/stochastic_volatility_{ekf,ukf}/`

2. **EKF vs UKF on SV (log-transform)** — both work after log-squared trick. Compare with ground truth.
   - Data: `stochastic_volatility/stochastic_volatility_{ekf_log,ukf_log}/`

3. **EKF vs UKF on Range-Bearing** — both work. Show 2D tracking with 2-sigma ellipses.
   - Data: `range_bearing/range_bearing_{ekf,ukf}/`

4. **Summary bar chart:** RMSE across all filter×model combinations (KF, EKF, UKF × SV raw, SV log, RB).
   - Data: `summary.json` from each run.

**Output files:** `report/figures/ekf_ukf_sv_raw.pdf`, `report/figures/ekf_ukf_sv_log.pdf`, `report/figures/ekf_ukf_rb.pdf`, `report/figures/kalman_family_rmse.pdf`

**Note:** Subsumes the current `kalman.ipynb`. After this notebook is created, `kalman.ipynb` can be archived.

---

## Notebook 3: `flow_filter_comparison.ipynb` (Section 4.3-4.5)

**Purpose:** Compare all flow filters (EDH, LEDH, invertible, stochastic) on range-bearing, SV, and acoustic tracking.

**Plots:**

1. **Range-Bearing: PF vs EDH vs LEDH vs EDH-Inv vs LEDH-Inv** — overlay filtered means on true state. One subplot per filter, shared axes.
   - Data: `range_bearing/range_bearing_{pf,edh_flow,edh_invertible,ledh_flow,ledh_invertible}/`

2. **Range-Bearing: Global vs intermediate re-linearization** — EDH global (fails, RMSE 1.31) vs EDH re-linearized (works, RMSE 0.075). Direct visual proof that re-linearization matters.
   - Data: `range_bearing/range_bearing_{edh_flow_global,edh_flow}/`

3. **SV: PF vs flow filters (raw space)** — PF tracks, all flow filters fail. Show why: flat estimates from flow filters.
   - Data: `stochastic_volatility/stochastic_volatility_{pf,edh_flow,stochastic_edh}/`

4. **SV: PF vs flow filters (log space)** — After log-transform, flow filters recover. Compare PF-log vs LEDH-invertible-log.
   - Data: `stochastic_volatility/stochastic_volatility_{pf_log,ledh_invertible_log,ledh_flow_log}/`

5. **Stochastic EDH: with vs without local correction on Range-Bearing** — diverged vs recovered.
   - Data: `range_bearing/range_bearing_{stochastic_edh,sde_local_correction}/`

6. **ESS comparison across flow filters on Range-Bearing** — overlay ESS traces. Weighted filters (invertible) should show ESS drops; unweighted (EDH/LEDH flow) have no ESS concept.
   - Data: `ess.npy` from weighted variants, N/A for unweighted.

7. **RMSE bar chart:** All filters × {RB, SV raw, SV log}. Pull from `summary.json`.

**Output files:** `report/figures/flow_rb_comparison.pdf`, `report/figures/flow_global_vs_relinear.pdf`, `report/figures/flow_sv_raw.pdf`, `report/figures/flow_sv_log.pdf`, `report/figures/sde_local_correction.pdf`, `report/figures/flow_ess_rb.pdf`, `report/figures/flow_rmse_bar.pdf`

---

## Notebook 4: `resampling_comparison.ipynb` (Section 5.4)

**Purpose:** Compare systematic vs soft vs OT resampling on SV and 5D linear.

**Plots:**

1. **SV: RMSE vs epsilon (OT) and alpha (soft)** — line plot. Systematic baseline as horizontal line.
   - Data: `summary.json` from `stochastic_volatility_pf`, `stochastic_volatility_pf_ot_eps{0.1,0.3,0.5,1.0}`, `stochastic_volatility_pf_soft_alpha{0.5,0.7,0.9}`

2. **SV: Runtime vs epsilon/alpha** — same structure. Shows OT is 60-100x slower.
   - Data: `performance.wall_time_seconds` from each `summary.json`.

3. **SV: ESS traces — systematic vs soft vs OT** — overlay ESS for systematic, soft (alpha=0.7), OT (eps=0.3).
   - Data: `ess.npy` from `stochastic_volatility_pf`, `stochastic_volatility_pf_soft_alpha0.7`, `stochastic_volatility_pf_ot_eps0.3`

4. **5D linear partial strong: RMSE bar chart** — PF (sys/soft/OT) and LEDH-inv (sys/soft/OT). Shows whether OT helps in higher dimensions.
   - Data: `summary.json` from `5d_linear_partial_strong/5d_partial_strong_{pf,pf_ot,pf_soft,ledh_invertible,ledh_invertible_ot,ledh_invertible_soft}`

**Output files:** `report/figures/resampling_rmse_sweep.pdf`, `report/figures/resampling_runtime_sweep.pdf`, `report/figures/resampling_ess.pdf`, `report/figures/resampling_5d.pdf`

---

## Notebook 5: `acoustic_tracking.ipynb` (Section 4.3.5 acoustic tracking table)

**Purpose:** Visualize acoustic tracking results — this is the high-dimensional showcase.

**Plots:**

1. **2D tracking plot** — true target trajectories (4 targets) overlaid with filter estimates. Sensor positions as markers. One panel per filter (PF, LEDH Flow, LEDH Invertible).
   - Data: `acoustic_tracking/acoustic_tracking_{pf,ledh_flow,ledh_invertible}/{states,means}.npy`

2. **RMSE over time** — per-timestep RMSE for each filter. Shows where/when filters diverge.
   - Data: compute from `states.npy` and `means.npy`

3. **ESS over time** — PF vs LEDH invertible. PF should show near-constant low ESS (0.92 resampling rate).
   - Data: `ess.npy` from PF and LEDH invertible.

**Output files:** `report/figures/acoustic_tracking_2d.pdf`, `report/figures/acoustic_rmse_time.pdf`, `report/figures/acoustic_ess.pdf`

---

## Notebook 6: `filter_summary.ipynb` (Section 4.7)

**Purpose:** Cross-model summary tables and comparison charts.

**Plots:**

1. **RMSE heatmap** — rows = filters, columns = models. Color-coded. Shows at a glance which filters work where.
   - Data: all `summary.json` files.

2. **Runtime vs RMSE scatter** — each point is a filter×model combination. Reveals the accuracy-cost tradeoff.
   - Data: `rmse` and `wall_time_seconds` from all `summary.json`.

3. **Memory bar chart** — peak memory by filter family, grouped by model.
   - Data: `peak_memory_mb` from all `summary.json`.

**Output files:** `report/figures/rmse_heatmap.pdf`, `report/figures/runtime_vs_rmse.pdf`, `report/figures/memory_comparison.pdf`

---

## Shared Utilities

Create `analysis/plot_utils.py` with reusable functions extracted from existing notebooks:

```python
def load_filter_results(output_dir):
    """Load states, means, covs, observations, and optional ess/weights from .npy files."""

def plot_filter_tracking(states, means, covs, observations, filter_name, ax, color):
    """Standard 1D tracking plot: true state, estimate, 2-sigma band."""

def plot_filter_tracking_2d(states, means, filter_name, ax, color):
    """2D tracking plot for range-bearing / acoustic tracking."""

def load_summary(output_dir):
    """Load summary.json, return dict with rmse, log_likelihood, performance."""

def collect_summaries(model_dir, filter_names):
    """Collect summary metrics from multiple filters into a DataFrame."""

def rmse_bar_chart(df, ax, title):
    """Grouped bar chart of RMSE from a DataFrame."""
```

---

## Execution Order

1. Create `analysis/plot_utils.py` with shared utilities.
2. Create Notebook 1 (`kf_failure.ipynb`) — simplest, validates the plotting pattern.
3. Create Notebook 2 (`ekf_ukf_comparison.ipynb`) — subsumes `kalman.ipynb`.
4. Wait for `run_all_filters.sh` to complete (re-runs at T=100).
5. Create Notebooks 3-6 after new data is available.
6. Archive old notebooks: move `kalman.ipynb`, `resampling.ipynb`, `resmpling.ipynb` to `analysis/archive/`.
7. Keep `particle.ipynb` (still useful for BPF-specific plots), `hmc.ipynb` (deferred), `figure3_kernel_comparison.ipynb` (kernel section).

---

## Mapping: Notebook → Report Section → Figures

| Report Section | Notebook | Key Figures |
|----------------|----------|-------------|
| 2.4 KF failure | `kf_failure.ipynb` | kf_failure_sv, kf_failure_rb, kf_ekf_ukf_rb |
| 3.3 EKF/UKF comparison | `ekf_ukf_comparison.ipynb` | ekf_ukf_sv_raw, ekf_ukf_sv_log, ekf_ukf_rb, kalman_family_rmse |
| 4.2 BPF degeneracy | `particle.ipynb` (existing) | particle tracking, ESS, weight dist, diversity |
| 4.3 Flow filters | `flow_filter_comparison.ipynb` | flow_rb, global_vs_relinear, flow_sv_raw, flow_sv_log, sde_local_correction |
| 4.5 Stochastic flow | `flow_filter_comparison.ipynb` | sde_local_correction |
| 4.6 Kernel flow | `figure3_kernel_comparison.ipynb` (existing) | kernel scatter |
| 4.7 Filter summary | `filter_summary.ipynb` | rmse_heatmap, runtime_vs_rmse, memory_comparison |
| 5.4 Resampling comparison | `resampling_comparison.ipynb` | resampling_rmse_sweep, runtime_sweep, ess, 5d |
| Acoustic tracking | `acoustic_tracking.ipynb` | acoustic_2d, acoustic_rmse_time, acoustic_ess |
