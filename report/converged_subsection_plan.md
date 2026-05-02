# Plan — Add "Multi-Chain Convergence Diagnostics" subsection to main_reorganized.tex

Status: revised after codex review (2026-05-02). Awaiting user approval.

## Codex's main pushback (incorporated below)

1. This is a **narrative cleanup**, not a clean append. The existing report at lines 2332 and 2459 makes claims based on single-chain split-$\hat R$ as if it were meaningful, and says "not a true multi-chain diagnostic" as a stated limitation. Adding multi-chain results without revising those passages reads as contradicting itself. The plan now includes editing those existing passages.
2. Title "Converged 4-Chain Runs" sounds cherry-picked. Renamed to **"Multi-Chain Convergence Diagnostics"**. Section now opens with a **screening table including failures**, not just successes.
3. For LG, fold BPF+OT-long and LEDH+OT into ONE subsubsection with side-by-side panels.
4. Use **Vehtari-style rank-normalized bulk and tail ESS**, not lag-1 autocorr. Lag-1 is too weak for a methods-paper diagnostic.
5. **3 main-text figures, not 8**. Combine trace + histogram per case into one composite figure. Per-chain breakdowns go to an appendix if at all.
6. **Bias-in-σ column risks misreading** — renamed to "truth offset / posterior sd" with explicit footnote that this is realization-specific centering, NOT a chain failure indicator.
7. Missing items added: chain initialization strategy, warmup / adaptation policy, step size and mass matrix per case, whether adaptation was shared or per-chain, count of numerical pathologies (rejected proposals, divergences, ΔH blowups).

## Writing style — match the existing report

The existing prose in §2264–2406 has identifiable conventions to preserve:

- **First-person "I" voice.** "I report HMC runs...", "I include this run as an honest diagnostic", "I record the run here rather than omit it".
- **British spelling.** "linearisation", "analysed", "characterise". Match this in any new text.
- **Em dashes for asides** rather than parentheses where it reads naturally.
- **Direct red-flag calls** without hedging: "is the red flag in this section", "effectively no independent information is extracted", "this is a known gap".
- **Honest about limitations.** "I record the run here rather than omit it --- the diagnostic is what tells us the chain is not ready."
- **Para structure:** setup → numbers → interpretation → caveat. Not bullet-list dumps.
- **No filler.** "Perhaps", "maybe", "it is possible that" — avoid.
- **Short emphasis sentences** after a technical paragraph: "This run is a known gap." "The chain is converged by the standard threshold."
- **Cite specific numbers**, not just qualitative summaries: "ESS is $4.0$ over $500$ samples", "split-$\hat R = 1.15$".

The new subsection's prose must read as continuous with §HMC Benchmarks, not as a different author's voice.

## Where to insert

After `\subsection{HMC Benchmarks}` (line 2264), as a new sibling subsection — before `\subsection{Gradient Diagnostics on SV2D}` (line 2408).

Title: `\subsection{Multi-Chain Convergence Diagnostics}`
Label: `sec:hmc-multichain`

## Edits to existing text (REQUIRED — not optional)

1. **Line 2332** (`\paragraph{Diagnostic readout.}` of the LG single-chain section). Currently reports `split-$\hat R = 0.999$` and `split-$\hat R = 1.15$` from single chains as if meaningful. Edit to add a parenthetical or short forward reference: "(single-chain split $\hat R$; the multi-chain $\hat R$ is reported in §\ref{sec:hmc-multichain})". Avoid contradiction.

2. **Line 2459** (`\subsection{Limitations}` item ii). Currently states `(ii)~$\hat{R}$ in this report is a single-chain split $\hat R$, not a true multi-chain diagnostic.` This needs to be revised to: `(ii)~the single-chain split-$\hat R$ values reported in §\ref{sec:linear-gaussian-benchmark} are exploratory; multi-chain $\hat R$ for the converged retuned configurations is reported in §\ref{sec:hmc-multichain}.` Or similar.

3. **Line 2378** (SV1D single-chain split-$\hat R = 0.66$ "trapped chain" interpretation). The new SV1D multi-chain run gives a clean R-hat (1.0056). This contradicts the trapped-chain claim. Edit needed: clarify that the trapped-chain reading was based on a specific single-chain run with a step-size and num-leapfrog that did not mix; the multi-chain run with retuned settings converges. Reference forward to the new section.

4. **Line 2406** (cross-comparison sentence: "LG BPF+OT and SV1D LEDH+OT are the two red flags"). The SV1D red-flag claim needs softening — it was true for the specific config used in §HMC Benchmarks but the multi-chain SV1D run with re-tuning is fine. Update to: "LG BPF+OT and the original SV1D LEDH+OT runs are red flags at the configurations used here; both are revisited with retuned settings in §\ref{sec:hmc-multichain}."

5. **Line 2386** (caption of Table~\ref{tab:hmc_diagnostics}). Currently describes split-$\hat R$ as the standard convergence metric. Add to caption: "Single-chain split-$\hat R$ is reported here; multi-chain $\hat R$ for retuned configurations is in Table~\ref{tab:hmc_multichain_diagnostics}."

## New subsection structure

```
\subsection{Multi-Chain Convergence Diagnostics}
\label{sec:hmc-multichain}

[Intro paragraph: §HMC Benchmarks reported single-chain results as
exploratory diagnostics. This subsection reports proper multi-chain
convergence checks: 4 chains with varied initial conditions, fixed PF seed,
and split rank-normalized $\hat R$ as the modern threshold.
For each model where the retuned configuration converges to $\hat R \le 1.01$,
I report the run; for the configurations where multi-chain still fails, I
report the failure pattern. Then short paragraph explaining what changed in
each retuned config: BPF+OT-long extends $N_\text{samples}$ to 2000;
LG-LEDH+OT and SV1D-LEDH+OT keep the original config; range-bearing requires
per-axis step ($\varepsilon_{\text{range}} = 0.001$, $\varepsilon_{\text{bearing}} = 0.004$, ratio preserved by DA) plus $L = 10$ leapfrog steps.]

\paragraph{Attempted multi-chain configurations.}
[Screening table — Table~\ref{tab:hmc_multichain_attempts} — listing all 4-chain
attempts including failures. Columns: Model / filter / config / outcome / split-$\hat R$.
Successes: LG-BPF+OT-long, LG-LEDH+OT, SV1D-LEDH+OT, RB-LEDH+OT-axisstep-l10.
Failures: LG-BPF+OT-short ($\hat R = 1.087$), RB-default ($\hat R = 1.094$ on $\sigma_\text{bearing}$),
RB-mass-vector (DA runaway, MatrixSolve crash), SV2D-LEDH+OT (OT backward singular).
This is the screening: I do not cherry-pick the converged runs.]

\subsubsection{1D Linear Gaussian (BPF+OT and LEDH+OT)}
[1 paragraph: 4 chains, varied $\sigma_\text{obs}$ inits, true value $1.0$.
BPF+OT-long uses 2000 samples/chain because the original 400-sample run
($\hat R = 1.087$) was under-mixed; LEDH+OT is fine at 400.]
[Stats table — both filters in one table.]
[Composite figure: 2-panel trace + 2-panel histogram, BPF+OT-long left,
LEDH+OT right.]
[1 paragraph interpretation: chain means agree, posterior offsets
from truth match each other (realization bias, not chain failure).]

\subsubsection{1D Stochastic Volatility (LEDH+OT)}
[Brief setup. 1 paragraph contrasting with the original §2378 single-chain
that read as "trapped" — the multi-chain rerun with the same config
converges, and the prior trap interpretation was an artefact of running
one chain.]
[Stats table.]
[Composite figure.]
[1 paragraph interpretation including bias.]

\subsubsection{Range-Bearing (LEDH+OT, Per-Axis Step + $L = 10$)}
[1 paragraph: setup. The original config in §HMC Benchmarks had
$\hat R_{\sigma_\text{bearing}} = 1.094$. Per-axis step ($\varepsilon = [0.001, 0.004]$, $4\times$ on bearing axis) drops it to $1.025$;
adding $L = 10$ drops further to $1.005$. Brief why: bearing has narrower
posterior std and the gradient through atan2 is noisier, so its mixing
rate per leapfrog was the bottleneck.]
[Stats table.]
[Composite figure (4-panel: 2-axis trace + 2-axis histogram).]
[1 paragraph interpretation: $\sigma_\text{range}$ matches truth within 1 sd; $\sigma_\text{bearing}$ posterior centred at $0.130$ with the truth $0.10$ at $1.87\sigma$ — realization-specific bias, not a chain problem.]

\paragraph{Cross-comparison.}
[Wrap-up paragraph + summary table: all converged runs side-by-side. Columns:
truth, posterior mean ± std, 90\% CI, truth offset / posterior sd, split-$\hat R$, bulk ESS, tail ESS.
Plus a brief comment on per-config tuning costs (RB needed per-axis step
and longer $L$; LG-BPF needed $5\times$ more samples).]

[Brief footnote / remark on the realization-bias column: it measures where
the empirical posterior centres relative to truth, not chain convergence.
The convergence diagnostic is $\hat R$ and ESS; truth offset is a separate
question about how informative this particular dataset is.]
```

## Stats table per model (revised)

For each parameter:

| Parameter | Truth | Posterior mean | Posterior sd | 90% CI | Truth offset / posterior sd | Split $\hat R$ | Bulk ESS | Tail ESS |
|---|---|---|---|---|---|---|---|---|

Notes:
- "Truth offset / posterior sd" replaces "bias in σ". Footnote: realization-specific posterior centering, not a chain failure metric.
- Bulk and tail ESS via Vehtari 2021 rank-normalized formula. Tail ESS particularly relevant for the 5%/95% quantiles which the 90% CI relies on.
- All 4 chain means listed in a footnote to give per-chain agreement evidence.

## Plots — 3 composite figures

Each composite is a single `\includegraphics` PNG with subpanels. The plot script generates each as a single PNG to keep the figure block clean.

**Figure 1: 1D LG.**
- 4 panels: (top-left) BPF+OT-long trace 4 chains; (top-right) BPF+OT-long histogram pooled + truth; (bottom-left) LEDH+OT trace; (bottom-right) LEDH+OT histogram.

**Figure 2: 1D SV.**
- 2 panels: trace, histogram. Single filter (LEDH+OT).

**Figure 3: Range-Bearing.**
- 4 panels: 2 traces (σ_range top, σ_bearing bottom on left) and 2 histograms (right). Or 2×2 grid: traces top row, histograms bottom row. Decide while building.

Filename convention (codex's suggestion, matching existing report style):

```
hmc_lg_4chain_combined.png       (Figure 1, BPF+OT-long + LEDH+OT panels)
hmc_sv1d_4chain_combined.png     (Figure 2)
hmc_rb_4chain_combined.png       (Figure 3)
```

(If user prefers per-filter separation for LG, the plan can swap to:
`hmc_lg_4chain_bpf_ot_long.png` + `hmc_lg_4chain_ledh_ot.png` — but codex
recommended combined to keep figure count down.)

## Plot generation

Add `code/analysis/plot_converged_chains.py`. Reads existing `samples_*.npy`,
generates 3 PNGs into `report/figures/`. Computes the same bulk + tail ESS
that goes into the stats tables, saves a JSON of stats so the LaTeX is
reproducible from the script's output rather than hand-edited values.

Pseudocode for ESS (Vehtari 2021):
```python
def ess_bulk_tail(chains):
    # rank-normalize across all chains
    # compute bulk: ess on rank-normalized samples
    # compute tail: min(ess on lower 5%, ess on upper 95%)
    # return (bulk_ess, tail_ess)
```

(Vehtari et al. 2021 has the exact recipe; ArviZ implements it. We can
either depend on ArviZ or hand-code the rank-normalization and Geyer's
initial-positive estimator.)

## Items now added (per codex point 7)

The new prose must include for each model:
- Chain initialization values (the 4 σ inits per chain).
- Fixed PF seed `[42, 0]` per HMC step.
- Burn-in length (200) and adaptation policy (DA with shrinkage_target = init step, target_accept = 0.75).
- Step size: scalar value or per-axis vector. Specify: LG 0.001, SV1D 0.001, RB [0.001, 0.004].
- Mass matrix: identity for all converged runs (per-axis step replaced mass for RB; the mass-vector attempt failed and is recorded in the screening table).
- Whether adaptation was shared across chains: each chain runs independent DA. State this once in the intro.
- Numerical pathology counts: number of rejected proposals over total, divergent transitions if any (HMC has no NUTS-style termination so "divergence" maps to gradient-NaN events — log them from existing chain output if recorded).

## Order of work

1. Apply the edits to existing text (lines 2332, 2378, 2386, 2406, 2459) so the report no longer contradicts itself.
2. Build `plot_converged_chains.py`. Run it. Confirm 3 PNGs + 1 stats JSON.
3. Write the new subsection prose, in the user's voice, using the stats JSON for table values.
4. Send the LaTeX draft + plots to codex for one final review pass before user sign-off.

## Open questions (re-asked after codex review)

1. **Edit existing report passages: confirm in scope.** Codex said this is not optional; the new section without these edits reads contradictory. Confirm.
2. **Bulk / tail ESS via ArviZ vs hand-coded.** ArviZ adds a dep but is well-tested; hand-coded keeps the project dep-free. User preference?
3. **Failure screening table (Table~\ref{tab:hmc_multichain_attempts})** — codex insists on it. Confirm including the SV2D crash, RB-mass-vector crash, BPF-short under-mixing as rows alongside the successes.
4. **Composite figures vs separate.** Codex wants 3 composite figures; user originally implied 8. Composite reduces float count by 5 and groups related panels. Confirm.
5. **Truth offset rename.** "Truth offset / posterior sd" replacing "bias in σ" — confirm the renaming.
6. **Voice / style match** — anything specific in the existing prose to mimic that I haven't called out (specific phrasings, transitional phrases, etc.)?
