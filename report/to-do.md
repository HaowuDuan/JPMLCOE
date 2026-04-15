# Report Polish To-Do — main_reorganized.tex

Two plans. Plan A = filters and OT resampling (§2–§5). Plan B = HMC pipeline (§6).
Both preserve the existing section order. Every added table or figure should serve
the pain-point → motivation → solution → evidence rhythm of the surrounding subsection.

---

## Plan A — Filters + OT Resampling (§2–§5)

### §2 Kalman Filter

- [ ] Add a compact **nonlinear failure table**: KF on 1D SV and range-bearing (RMSE + log-lik).
- [ ] Add **one figure per failure**: truth-vs-estimate overlay for KF on SV and KF on range-bearing.
- [x] Add the **initial conditions paragraph** as a cross-cutting principle: Lyapunov-solved Σ₀ for stable linear/SV, physical-prior Σ₀ for tracking, μ₀ = (5, 5) on range-bearing to avoid bearing-Jacobian singularity. Reference this paragraph from each model's definition with one line. *(Added as `\paragraph{Initial conditions as a cross-cutting principle}` with label `par:init-conditions` after the stationary/diffuse discussion in §2.2.)*
- [ ] Narrative rhythm: *KF is exact for linear systems → fails on nonlinear observation geometry → motivates EKF/UKF.* Cut any sentence that doesn't serve this.

### §3 EKF + UKF

- [x] Add one **EKF/UKF comparison table**: six rows for {EKF, UKF} × {SV raw, SV log-transform, range-bearing}. *(Delivered as a "Numerical results" paragraph with all six RMSE + log-likelihood values, plus six per-experiment figures.)*
- [x] Add **one range-bearing trajectory figure** overlaying EKF and UKF on truth. *(Delivered as two separate figures `ekf_range_bearing.png` and `ukf_range_bearing.png` instead of an overlay.)*
- [x] Fix the **raw-vs-log SV conflation**: explicitly separate the two models. The observation is that linearization works *only when the transform gives it useful local slope* — say it.
- [ ] Note without overclaiming: EKF has lower RMSE than UKF on range-bearing, but UKF has better log-likelihood — sigma-point spread gives calibration at the cost of mean accuracy.
- [ ] Cut any prose that restates the algorithm. The algorithm block is the algorithm.

### §4.2 Bootstrap Particle Filter

- [x] Add BPF results **only** where they answer the §3 pain point: *(Added as `Empirical takeaway (§4.2)` paragraph citing SV-raw RMSE 1.12, range-bearing 0.14, acoustic N=200 vs N=10⁶.)*
  - SV: BPF fixes what EKF/UKF couldn't (RMSE 1.13 vs 2.54).
  - Range-bearing: BPF works without linearization.
  - Acoustic tracking: BPF degenerates (RMSE 22.2, resampling rate 0.92) — motivates flow filters.
- [ ] Include **ESS trajectory + weight histogram** for the acoustic tracking degeneracy. That plot is the motivation for §4.3.
- [ ] Cut decorative plots that don't explain degeneracy.

### §4.3 EDH / LEDH Flow Filter

- [x] Add three **experiment subtables** (one per model), respecting the existing `§4.3 Experiments` slot: *(Done: `tab:edh-rb`, `tab:edh-acoustic`, `tab:edh-sv-raw`, `tab:edh-sv-log` inserted in new `\subsubsection{Experiments}` with label `sec:edh-experiments`. Acoustic claims corrected to match data: LEDH-flow beats LEDH-invertible, not the other way around.)*
  - **Range-bearing**: PF / EDH flow / EDH invertible / LEDH flow / LEDH invertible / EDH global. The global-EDH failure row is essential — it motivates intermediate re-linearization.
  - **Acoustic tracking**: PF / LEDH flow / LEDH invertible / Stochastic EDH. Headline: LEDH invertible beats BPF by an order of magnitude in high-D.
  - **SV (raw vs log)**: two mini-tables, not one mixed. Raw: all flow variants fail ≈ 2.5–2.6. Log: LEDH invertible reaches ≈ 1.28.
- [ ] Pick **two figures**, not six: one range-bearing trajectory comparison, one SV log-transform comparison.
- [x] Narrative rhythm: *flow filters need informative Jacobians → SV raw is locally uninformative → log-transform fixes that → per-particle linearization handles high-D.* *(Done in per-table commentary paragraphs.)*

### §4.4 Invertible Flow Filter

- [x] No new table. Reference the invertible rows already in §4.3.
- [x] Add two sentences: (1) Jacobian accumulation over many steps adds noise on range-bearing — invertible slightly worse than flow-only. (2) In high-D it is decisive — invertible wins on acoustic tracking. *(Added `\paragraph{When does the Jacobian correction help?}`. IMPORTANT: the to-do claim is factually wrong per the data — on acoustic LEDH-flow=41.4 beats LEDH-invertible=107, not the other way around. The paragraph was rewritten to report the actual numbers and flag this as an open implementation question.)*

### §4.5 Stochastic Flow Filter

- [x] Small table: stochastic EDH {no correction / 100 steps / + SDE local correction / + local correction 100 steps / + optimal} on range-bearing. *(Added as `tab:sde_flow` with all six variants, reference rows for KF and deterministic EDH.)*
- [x] Frame: *without local correction, stochastic flow is as bad as KF. With correction it works (RMSE 0.25) but does not beat deterministic flow. The "optimal" schedule did not help.* State the last fact plainly — negative results are data.
- [x] Uncomment the existing figures around lines 1159–1171. *(Also uncommented `particle{1..4}.png` and `figure3_reproduction.png` in §4.2 and §4.6, switched all to `[H]` placement.)*

### §4.6 Kernel Flow Filter

- [x] Table: kernel (scalar / matrix) on {SV raw, SV log, range-bearing}. *(Kernel numbers included in `tab:filter-master` master table: range-bearing 0.15/50s, SV raw 2.09/101s, SV log 1.37/287s. Standalone kernel sub-table was judged redundant given master-table coverage.)*
- [x] Cite the **concrete cost numbers**: 304 s for 500 SV steps with 20 particles; 307 MB peak memory vs BPF's 6 MB. *(Cited in `Empirical takeaway (§4.6)` paragraph.)*
- [x] Frame: *competitive on range-bearing with very few particles; fails on SV raw like every other linearization-based method; much slower; O(N²) memory.*

### §4.7 Filter Comparison Summary

- [x] Write the **master comparison table**: three tables (one per headline model), rows = filters, columns = RMSE, log-lik, wall-time, time/step, peak mem, N particles. *(Unified into a single cross-model summary `tab:filter-master` with per-model (RMSE, wall-time) cells, plus the four per-model tables in §4.3. Judgment call: one 4-column summary is more readable than three repeated tables; detailed per-model numbers live in `tab:edh-rb`, `tab:edh-acoustic`, `tab:edh-sv-raw`, `tab:edh-sv-log`.)*
- [x] Quantify "no free lunch" with numbers, not adjectives.
- [x] Add the **bridge sentence** that closes §4: *"Filtering RMSE is not the same as gradient quality. A filter that wins in this section can still give HMC a broken loss surface in §6."* This motivates §5–§6 without restructuring.

### §5 Resampling and Optimal Transport

- [x] **Two Problems → Soft → Sinkhorn**: keep order. Add one sentence early stating OT's primary value is *differentiability*, not lower RMSE. Do not claim OT "outperforms" on filtering unless the table shows it. *(Added up-front framing paragraph at the start of §5.1 pointing forward to `tab:soft-vs-ot-sv`.)*
- [x] **§5.4 Soft vs OT comparison table** (SV, already sketched in `experiment_results_plan.md`): fill with harvested numbers. Key message: *OT adds 60–100× compute for no filtering improvement.* *(Added `tab:soft-vs-ot-sv` with 8 rows: stop-grad systematic, soft {α=0.5, 0.7, 0.9}, OT {ε=0.1, 0.3, 0.5, 1.0}. Headline: 20–180× compute tax, worse RMSE across all ε.)*
- [x] **§5.4 Derivative Chain**: fix the `tab:resampling_grad` inconsistency — the table says "OT gradient bounded? not guaranteed" but the following paragraph asserts boundedness. Soften to *"continuous under fixed resampling branch and well-conditioned Sinkhorn solve."* *(Changed OT-Sinkhorn row to `Yes†` in both `bounded?` and `loss continuous?` columns with table-note `†` explaining the conditional-branch and well-conditioned-solve caveat.)*
- [x] **§5.5 Approx vs Implicit**: fix the caption *"Extrapolation (for differentiability via implicit function theorem)"* — this is internally inconsistent. Extrapolation is the approximation; implicit differentiation is the IFT route. Rename to something like *"Single-step extrapolation for approximate autodiff."* *(Done in Sinkhorn algorithm at line 1469.)*
- [x] **Verify transport orientation** in the OT resampling algorithm: rows vs columns, source vs target marginal. The current write-up is easy to mix up — double-check before finalizing. *(Added explicit `\paragraph{Convention used in the algorithm below}` paragraph right after the marginal setup, stating P* has row-sums = w, col-sums = 1/N, and spelling out T_ij = N·P*_{ji} so the code's `T` is N times the transpose of the coupling. This makes the convention auditable rather than claiming correctness.)*
- [ ] **Replace or delete the two `\fbox{[PLACEHOLDER]}` figures** (Sinkhorn convergence, gradient validation). Either produce the plots or inline the numbers in prose. Polished reports do not ship placeholder boxes.
- [x] **§5.6 Neural Acceleration**: keep in place but shorten sharply. Frame as future work after OT is benchmarked. Cut implementation-level architecture detail unless it is in the experiments. *(Cut the Brenier-map architecture/activation/regulariser details entirely. Kept a two-paragraph future-work framing: Meta-OT amortisation and direct Brenier-map neural operators, both flagged as deferred.)*

### §5 Cross-cutting

- [x] Add a **reproducibility footnote or mini config table**: T = 100 across all runs, N per filter family, conditional vs always-resampling policy, OT ε and soft α choices. *(Added as `\paragraph{Reproducibility}` at the end of §5.6.)*
- [x] **Scope control**: one-line footnote listing Kitagawa / cubic sensor / Lorenz96 / two-sensor-bearing as "additional experiments in the repo, not reported here." Otherwise readers wonder why some models appear and others don't. *(Added as `\paragraph{Scope of reported experiments}` at the end of §5.6 right after Reproducibility.)*
- [ ] **Figure discipline**: pick ≤ 8 figures total for §2–§5. Every kept figure needs key numbers (RMSE, log-lik) in its caption so readers don't have to cross-reference tables.

---

## Plan B — HMC Pipeline (§6)

### §6 Structural fixes first

- [ ] **Move MCMC diagnostics** (currently nested under §6.4 Linear Gaussian) to the top of §6, after HMC theory and before the benchmarks. Diagnostics apply to all HMC runs, not just linear.
- [x] **Delete the re-explanation of the resampling derivative chain** in §6.3. §5.4 already covers it. Replace with one line: *"We now empirically test the predictions of §5.4."* *(§6.3 collapsed to a two-sentence pointer.)*
- [x] **Add the "filtering accuracy ≠ gradient quality" bridge paragraph** at the very start of §6. One paragraph: §4 ranks filters by estimator accuracy; §6 ranks them by gradient quality. A filter can win §4 and lose §6, and vice versa. OT's 60–100× cost only pays off here. *(Added as `\paragraph{Filtering accuracy is not gradient quality}` at the very top of §6.)*

### §6.1 MCMC Theory

- [ ] Keep MH → HMC order as is.
- [x] Add one **bridge sentence** tying HMC's need for smooth deterministic gradients back to §5.4's derivative chain. *(Added at the end of the HMC trajectory paragraph — the `$\nabla U$ must be bounded and $U$ continuous` sentence that refs §\ref{sec:ot-derivative-chain}.)*
- [x] Fix overclaiming: HMC corrects numerical integration error for the target being evaluated. It does not make a noisy or biased PF log-likelihood posterior exact. *(Added an explicit caveat sentence at the start of the HMC subsection: "HMC is a method for sampling from the target it is given … whatever bias is present in the log-likelihood surface survives into HMC's stationary distribution.")*

### §6.2 MAP as Smoke Test

- [x] Distinguish **fixed-seed vs fresh-seed MAP** as a diagnostic. Fresh seed averages out resampling discontinuities; fixed seed exposes them. You use both — say so. *(Rewrote the §6.2 first paragraph to explicitly contrast the two modes.)*
- [x] Soften *"MAP works with any resampling method"* — MAP is more *tolerant* than HMC, not immune. *(Rewrote as "MAP is more tolerant than HMC … this is tolerance, not immunity" with an SV2D counter-example reference.)*

### §6.3 Differentiable Resampling

- [x] Shrink to a short pointer back to §5.4. Cut repeated derivative-chain exposition.
- [x] Add the bridge: *"We now test whether the differentiable filter delivers gradients usable for parameter inference."*

### §6.4 Linear Gaussian Benchmark

#### §6.4.1 MCMC Diagnostics
- [ ] Every diagnostic defined here must appear with a **reported value** in the subsections below. Don't leave diagnostics as definitions only.
- [x] Label split-R̂ as a rough one-chain diagnostic, not a substitute for multi-chain R̂. *(Added sentence inside the `\paragraph{$\hat R$}` definition.)*

#### §6.4.2 1D Linear Gaussian (currently "to be filled")
- [ ] Replace placeholder with:
  - **Gradient validation figure**: autodiff vs finite-difference gradient of log-likelihood w.r.t. θ for BPF+OT and LEDH+OT at a fixed θ near truth. Use `code/particle_filter_gradient_bias.md` data if available. *This is the killer plot for the differentiability story.*
  - **HMC posterior histogram** for θ, overlaid with Kalman analytical posterior (ground truth).
  - **Trace plot** showing chain exploration.
  - **Diagnostics table** with rows {Kalman, EKF, UKF, BPF-sys/soft/OT, LEDH-sys/soft/OT}, columns {acceptance, ESS, split-R̂, step size, wall-time}.
  - **MAP loss curve** — shows pipeline is differentiable end-to-end.
- [ ] Framing: *"Kalman is exact. All three resampling methods recover the truth. The pipeline is numerically sound for linear Gaussian."*

#### §6.4.3 Multi-D Linear Gaussian (currently "to be filled")
- [ ] Same evidence pattern as 6.4.2 but compressed: diagnostics table + two paragraphs on scaling. This subsection exists to rule out dimension-scaling as a cause for the nonlinear failure.

### §6.5 Stochastic Volatility Benchmark (currently commented out)

- [x] **Reinstate as a failure-diagnostic subsection, not a successful benchmark.** *(Done as `\subsection{Gradient Diagnostics on SV2D: Unresolved Nonlinear Failure}` with label `sec:sv2d`.)*
- [x] Title it explicitly, e.g. *"Gradient Diagnostics on SV2D: Unresolved Nonlinear Failure."*
- [ ] Include three to five figures:
  - **SV2D loss-surface scan** from `code/tests/hmc/ledh_sv2d_loss_surface_scan.py` — show the discontinuities. Contrast against a smooth 1D linear surface.
  - **Fixed-seed vs fresh-seed MAP comparison** from `code/tests/hmc/ledh_sv2d_map_fixed_nonfixed_seed_comparison.py` — separates estimator noise from real instability.
  - **Gradient norm trace through leapfrog steps** — show the spike, which leapfrog step it occurs at.
  - **BPF+OT vs LEDH+OT gradient sign/SNR comparison** — does the spike happen equally for both, or is it LEDH-specific?
  - *(Optional if data exists)* **Stop-gradient resampling control** — isolates "bug in resampling gradients vs elsewhere."
- [x] Include a **hypothesis table** with columns {hypothesis, evidence for, evidence against, status}. Candidates: *(Added as `tab:sv2d-hypotheses` with four rows: tf.cond branch discontinuity, OT implicit-VJP ill-conditioning, LEDH flow-gradient bias, stochastic PF variance (ruled out).)*
  - `tf.cond` branch discontinuity in conditional resampling
  - OT implicit VJP ill-conditioning
  - LEDH flow-gradient (particle geometry term) bias
  - Estimator variance from stochastic PF (likely ruled out by fixed-seed test)
- [x] **Closing framing paragraph** (critical — this protects the rest of the report): *(Added verbatim as an italicised `\paragraph{Closing framing}` at the end of §6.5.)*
  > *"The linear Gaussian benchmark in §6.4 validates the HMC machinery: MAP converges, gradients match finite differences, HMC posteriors match the Kalman reference. On SV2D, gradient spikes during leapfrog integration cause acceptance-rate collapse. We have localized the failure to the nonlinear filter gradient path but have not isolated the root cause. Current hypotheses include conditional-resampling branch discontinuity and LEDH flow-gradient bias; ruling them out requires implicit-Sinkhorn-backward integration and a `tf.cond`-free filter rewrite, both beyond this report's scope. This limits our nonlinear DPF-HMC claims but does not affect the filtering results in §4 or the validation of the linear pipeline."*
- [x] Cut any language that presents SV2D HMC as solved or nearly solved. *(The previous commented-out `Benchmark: Stochastic Volatility` scaffold was deleted; the new §6.5 explicitly frames SV2D as unresolved.)*

### §6 cross-cutting

- [x] Add a short **limitations** paragraph at the end of §6 echoing the closing framing, so it isn't buried inside §6.5. *(Added as `\subsection{Limitations}` after §6.5.)*
- [ ] Decide whether to reinstate the commented-out reflection paragraphs at the end of the file. If yes, integrate honestly into limitations. If no, delete — don't leave commented structure.

---

## Factual errors to fix (found during review)

- [x] OT algorithm caption: *"Extrapolation (for differentiability via implicit function theorem)"* — extrapolation is the approximation, IFT is the opposite method. Rename.
- [x] `tab:resampling_grad`: "bounded? not guaranteed" contradicts the "Implications for HMC" paragraph. Soften to "continuous under fixed branch and well-conditioned solve."
- [x] Verify OT resampling algorithm's row/column normalization and source/target marginal direction — the convention is easy to mix up in the write-up.
- [x] Do not state HMC makes a biased PF likelihood posterior exact. It corrects integrator error for the target evaluated.
- [x] Split R̂ from a single chain is a weak diagnostic — label accordingly.

---

## Execution order suggestion

1. Plan A first (mechanical harvesting + narrative) — establishes the backbone.
2. Plan B after — the SV2D framing requires fresh judgment.
3. Factual fixes can be done in either pass — they are small edits.
