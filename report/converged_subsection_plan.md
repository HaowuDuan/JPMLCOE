# Plan — Add "Converged 4-Chain Results" subsection to main_reorganized.tex

Status: plan only, no edits applied. Awaiting user review → codex review → user approval.

## Where to insert

After `\subsection{HMC Benchmarks}` (line 2264), as a new sibling subsection — before `\subsection{Gradient Diagnostics on SV2D}` (line 2408). The current HMC benchmarks subsection covers single-chain comparisons across filters; the new subsection extends to 4-chain convergence diagnostics for the cases where R-hat satisfies the modern threshold.

Proposed subsection title: `\subsection{Multi-Chain Convergence Diagnostics}` (label: `sec:hmc-converged`)

Or following the user's wording: `\subsection{Converged 4-Chain Runs}`.

## What to include — 4 models, all R-hat ≤ 1.01

| Model | Filter | Config | Samples/chain | R-hat (split) |
|---|---|---|---|---|
| 1D LG | BPF+OT (long) | `linear_gaussian/bpf_ot_long_c{1..4}` | 2000 | 1.0012 |
| 1D LG | LEDH+OT | `linear_gaussian/ledh_ot_c{1..4}` | 400 | 1.0007 |
| 1D SV | LEDH+OT | `stochastic_volatility/ledh_ot_c{1..4}` | 400 | 1.0056 |
| Range-Bearing | LEDH+OT axisstep, L=10 | `range_bearing/ledh_ot_axisstep_l10_c{1..4}` | 400 | 1.005 (σ_bearing), 0.998 (σ_range) |

Excluded (not converged at modern 1.01 threshold):
- LG / BPF+OT (short, 400 samples): R-hat 1.087
- RB / LEDH+OT default (no axisstep): R-hat 1.094 for σ_bearing
- All SV2D runs: chains crashed
- RB mass_vector attempt: chain crashed during burnin

## Subsection structure

```
\subsection{Multi-Chain Convergence Diagnostics}
\label{sec:hmc-converged}

[1 paragraph intro: what this section adds beyond HMC Benchmarks; what the
modern R-hat ≤ 1.01 threshold means; what 4 chains with varied inits add over
single-chain runs.]

\subsubsection{1D LG, BPF+OT (Long)}
[1 paragraph: setup — 4 chains, 2000 samples/chain, varied inits, num_burnin.]
[Stats table.]
[Trace plot (all 4 chains).]
[Posterior histogram (all 4 chains overlaid + truth marker).]

\subsubsection{1D LG, LEDH+OT}
[Same structure, 400 samples/chain.]

\subsubsection{1D SV, LEDH+OT}
[Same structure.]

\subsubsection{Range-Bearing, LEDH+OT (Per-Axis Step)}
[Same structure. Plus a brief paragraph on why per-axis step + 10 leapfrog
steps was needed — bearing-axis mixing rate. Reference back to HMC Benchmarks
where the default config showed R-hat 1.094 for σ_bearing.]

\paragraph{Cross-comparison.}
[Wrap-up paragraph + summary table comparing across all 4 cases:
posterior bias (mean − truth in σ units), R-hat, ESS, samples/chain.]
```

## Stats table per model (suggested format)

For each parameter:

| Parameter | Truth | Posterior mean | Posterior std | 90% CI | Bias (σ) | R-hat (split) | ESS |
|---|---|---|---|---|---|---|---|
| ... | | | | [q5, q95] | (mean−truth)/std | | |

Per-chain agreement column optional (chain means as a tuple).

## Plots needed (do not exist yet)

For each of the 4 models, generate two plots from existing `samples_*.npy`:

**Plot type 1: Multi-chain trace.** Post-burn-in iterations on x-axis, parameter value on y-axis, 4 chains overlaid in different colors. Truth as a horizontal dashed line.

**Plot type 2: Multi-chain posterior histogram.** All 4 chains pooled into one histogram. Truth as vertical dashed line. Optionally show per-chain histogram outlines in lighter shade beneath the pooled histogram.

For multi-parameter models (range-bearing, 2 parameters), make 2 panels per plot: one per parameter.

## Plot generation

Add a script `code/analysis/plot_converged_chains.py` that:
- Loops over the 4 model/filter pairs above
- Reads `samples_*.npy` from each chain directory
- Pools post-burn-in samples
- Saves 2 PNG files per model into `report/figures/` (or wherever existing report figures live)

Filename convention (matching existing report figures):

```
hmc_4chain_lg_bpf_long_trace.png
hmc_4chain_lg_bpf_long_histogram.png
hmc_4chain_lg_ledh_ot_trace.png
hmc_4chain_lg_ledh_ot_histogram.png
hmc_4chain_sv1d_ledh_trace.png
hmc_4chain_sv1d_ledh_histogram.png
hmc_4chain_rb_ledh_trace.png       (2-panel)
hmc_4chain_rb_ledh_histogram.png   (2-panel)
```

## What needs writing

1. The plot script (~100 lines, single file).
2. The LaTeX subsection (~200 lines including 8 plots and 4 stats tables).
3. Possibly: a per-model JSON of stats so the LaTeX table values are reproducible (rather than hardcoded).

Order: build script → run it → confirm plots → write LaTeX referencing the new figures.

## Open questions for user / codex

1. Should the existing HMC benchmarks section stay single-chain, with this new section being purely 4-chain? Or do we cross-reference?
2. For LG, should we show BPF+OT_long and LEDH+OT side-by-side in one combined plot, or separate per-model subsubsections? Combined would make filter comparison sharper; separate keeps consistency with the other models.
3. R-hat number to report: classic vs split. I'd default to **split** (modern Stan convention, more conservative). Confirm.
4. ESS: I'll report the rough lag-1 estimate. If you want bulk ESS / tail ESS (Vehtari 2021), that's ~30 lines of additional code in the plot script. Confirm scope.
5. For RB axisstep run, do we describe the per-axis-step trick in this subsection or refer to a previous part of the report?
6. Should we include a brief failure-case note ("the SV2D run did not converge under any tested setting; see §X") or stay silent?

## Estimated effort

- Plot script: 1 hour to write, maybe 30 min iterating on aesthetics.
- LaTeX subsection: 2-3 hours to write the prose, tables, figure blocks.
- Codex review pass: 30 min round-trip after a draft is ready.

Total: ~half a day's work once the plan is approved.
