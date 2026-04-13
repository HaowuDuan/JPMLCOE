# Step 11 Quality Test — Open Issue Report

**Status:** unresolved as of 2026-04-09 evening
**Branch:** `jit-compile`
**Failing test:** `code/neural_operator/tests/test_resampler_quality.py`
**Passing test:** `code/neural_operator/tests/test_resampler.py` (infrastructure smoke)

This document covers Step 11 (map-based resampler) only. It captures the
quality test that is currently failing, everything that has been tried, the
two diagnostic interpretations on the table, and what is known vs unknown.

---

## 0. Two tests, do not confuse them

There are **two** Step-11 tests for the neural operator resampler. They check
very different things and only one of them is failing.

### `test_resampler.py` — infrastructure smoke test — **PASSES**

Trains a tiny model for 200 steps, then asks: does the
`NeuralOperatorResampler` *interface* work end-to-end? It checks shapes,
uniform weights, that local Jacobians are populated and PSD, that
`transport_matrix` and `ancestor_indices` are `None`, and that the resampler
is deterministic on repeated calls. It does **NOT** check whether the
trained map is any good.

Most recent run on office (2026-04-09 20:45):

```
--- Train 200 steps ---
step     0/200 | loss=3.3498e+01 | ... | min_eig=1.825 | residual_abs=3.8802
step    50/200 | loss=3.1176e-01 | ... | min_eig=1.042 | residual_abs=0.2309
step   100/200 | loss=6.5298e-03 | ... | min_eig=1.011 | residual_abs=0.0713
step   150/200 | loss=3.9147e-03 | ... | min_eig=1.010 | residual_abs=0.0544
step   199/200 | loss=1.4241e-02 | ... | min_eig=1.014 | residual_abs=0.1022

Shapes OK: particles=(200, 1), J=(200, 1, 1)
Weights uniform OK (sum=1.000000)
J PSD OK: min_eig=1.0033
Deterministic OK

=== Resampler smoke test passed ===
```

This is good news: the wiring of the resampler into the `code/src/resampling`
package is sound, the `ResampleResult.local_jacobians` field is populated
correctly, the model encodes-then-transports without crashing, and Keras
gives back deterministic outputs at inference time.

### `test_resampler_quality.py` — model quality test — **FAILS**

Trains a small model on synthetic data (currently 500 steps, was about to
go up to 2000), then asks: does the trained map move particles toward the
heavy-weight side better than literal identity, on four hand-crafted
low-ESS clouds? This is the question that is unresolved and is the entire
subject of this document.

---

## 1. What this step is supposed to deliver

Step 11 of `code/neural_operator_plan.md` is "Implement map-based resampler
in `code/src/resampling/`." The deliverable has two parts:

1. **Infrastructure** — a `NeuralOperatorResampler` class wrapping a trained
   neural operator and exposing the standard `ResampleResult` interface
   (mapped particles, uniform weights, per-particle Jacobians).
2. **Quality** — empirical evidence that the trained map actually does
   something useful with the weights, not just behave like identity.

Part 1 is **done and tested**. Part 2 is **not done**.

This document is about part 2.

---

## 2. What the quality test actually checks

`test_resampler_quality.py` constructs four hand-crafted 1D clouds with
**very lopsided weights**, trains a small neural operator on synthetic
data, then asks: does the trained map move particles toward the heavy-weight
side better than literally doing nothing (identity)?

### The four clouds

| Cloud | Particle layout | Weight pattern | ESS / N |
|---|---|---|---|
| 0 | bimodal at ±3 | 95% on left mode, 5% on right | 0.55 |
| 1 | bimodal at ±2 | 10% on left, 90% on right | 0.61 |
| 2 | three modes at -4 / 0 / +4 | 5% / 90% / 5% | 0.41 |
| 3 | uniform on [-2, 2] | exponential tilt right: w ∝ exp(2x) | 0.25 |

These are deliberately easy to reason about. For each cloud the *weighted*
mean is far from the *unweighted* mean, so identity (do nothing) gives a
large mean error.

### The metric

For each cloud:

```
mean_err = || mean(T(x_i)) − Σ_i w_i x_i || / || Σ_i w_i x_i ||
```

- `mean_err = 0` → perfect: output mean lands on the weighted target
- `mean_err = 1` → output mean is at the unweighted center, i.e. the output
  completely ignored the weights
- `mean_err > 1` → the output moved in the wrong direction

`cov_err` is the analogous Frobenius-norm relative error on the covariance.

### The assertions

```python
assert avg_T_mean < avg_id_mean   # T should beat identity on the mean
assert avg_T_cov  < avg_id_cov    # T should beat identity on the covariance
```

Averaged over the four clouds.

---

## 3. Latest concrete result (2026-04-09)

This is the result *after* fix 1 (gamma + multimodality) but *before* fix 2
(position-correlated weights). Fix 2 has not been retrained-and-tested yet
because the retry crashed on a separate matvec shape bug (see §5.7).

```
cloud 0: identity mean_err=0.998 cov_err=3.86  |  T mean_err=1.018 cov_err=3.88
cloud 1: identity mean_err=1.014 cov_err=1.45  |  T mean_err=1.002 cov_err=1.47
cloud 2: identity mean_err=0.406 cov_err=5.03  |  T mean_err=0.953 cov_err=5.08
cloud 3: identity mean_err=1.000 cov_err=4.50  |  T mean_err=0.997 cov_err=4.59

Avg: identity 0.854 / 3.71  |  T 0.993 / 3.76
```

Plain reading:

- The trained T has mean error ~1.0 on every cloud, which is the value you
  get when you ignore the weights entirely.
- On Cloud 2, where identity was somewhat OK (0.41), the trained T made it
  much worse (0.95).
- The model is producing essentially identity behavior on these structured
  low-ESS clouds, despite training.

Training itself was healthy: MA loss converged from ~120 down to ~5e-3 in
500 steps, no NaN, gradients flowing, J PSD with `min_eig ≈ 1.005`.

---

## 4. What is working

This is important so we don't lose ground when we resume.

| Component | Status |
|---|---|
| `NeuralOperatorResampler` interface (shapes, uniform weights, PSD J, determinism) | passes `test_resampler.py` |
| Forward pass through encoder + trunk | passes `test_models.py`, `test_conditional_map.py`, `test_encoder.py` |
| Analytic Jacobian matches autodiff | passes `test_conditional_map.py` |
| Training pipeline runs end-to-end | passes `test_train_smoke.py` |
| Bandwidth annealing schedule | works, visible in logs as h_scale 2.0 → 1.0 |
| Eigenvalue barrier on J | passes `test_eigenvalue_barrier.py` |
| Evaluation metrics module | passes `test_evaluate.py` (MA residual 588 → 0.019 on training-distribution clouds) |
| Implicit Sinkhorn gradient backward (separate work) | gradient ratio 0.999 |

The Step 11 *infrastructure* (resampler class, `ResampleResult.local_jacobians`
field, package wiring, smoke test) is fully functional. What is failing is
specifically the *learned model quality* on the held-out structured clouds.

---

## 5. Things that have been tried for the quality issue

### 5.1 Initial attempt
- Built `NeuralOperatorResampler` and the original combined test with a
  single held-out random cloud and a moment-improvement assertion.
- That assertion failed because the random cloud happened to have nearly
  uniform weights — identity was almost optimal, trained T was slightly
  worse from harmless perturbations.

### 5.2 Average over many random clouds
- Switched the assertion to average over 10 random clouds.
- Still failed: identity 0.025, trained T 0.029.
- This led to consulting Codex.

### 5.3 First Codex consultation
Findings:
- **Bug in `data.py`:** the `alpha` parameter was sampled but never actually
  used in the gamma generator. Every training cloud had Dirichlet(1) weights
  regardless of the supposedly tunable `alpha`. The seed was also reused
  between alpha sampling and gamma sampling.
- The data generator only ever produced single-Gaussian clouds despite the
  docstring claiming "1-3 mixture components".
- The "untrained baseline" comparison was misleading because
  `alpha_init_bias=-3` is not a literal identity init.
- MA-only training is *theoretically* moment-matching at convergence (by
  test function `f(y)=y` against the pushforward equality), but the finite
  sample MA-residual MSE plus the free output bias allows real moment drift
  in practice.

### 5.4 Fix 1 — data generator gamma + multimodality
Made `sample_random_cloud`:
- Use `tf.random.stateless_gamma` so `alpha` actually controls skew.
- Use independent seeds for every sub-draw.
- Sample 1-3 mixture components with random means and stds.
- Sample `alpha` from log-uniform `[0.1, 5]` to actually exercise low ESS.

Then dropped the moment-improvement assertion from the smoke test, and
added this dedicated quality test on fixed low-ESS clouds.

After fix 1, on the four hand-crafted clouds, the trained map was still
essentially identity (the result reproduced in §3 above).

### 5.5 Second Codex consultation
Sharper diagnosis:
> The current failure is not strong evidence that moment loss is required;
> it is strong evidence that the model was trained on the wrong family of
> weight-position relationships.

The point: in the post-fix-1 generator, weights were still **independent of
particle position**. They were Dirichlet noise, just with controllable
skew. But in real PF resampling — and in our four hand-crafted test clouds
— the weights are *correlated with position*: heavy modes have more mass,
the likelihood tilts in some direction, and so on. The training distribution
never taught the model how weights interact with positions, so on
position-structured clouds it falls back to near-identity.

### 5.6 Fix 2 — position-correlated weights in `data.py`
Updated `sample_random_cloud` to build log-weights as:

```
log_w_total = log_w_pos + log_w_comp + log_w_noise
```

where:
- `log_w_pos = beta^T (x − mu_global)` is a random linear tilt over particle
  positions, mimicking a likelihood
- `log_w_comp` is a random per-mixture-component bias, so some modes carry
  more posterior mass than others
- `log_w_noise = log Gamma(alpha)` keeps the Dirichlet noise so the operator
  still sees a wide range of skew levels

Then increased training to 2000 steps in `test_resampler_quality.py`.

### 5.7 Crash on retry
This run crashed on a stupid `(N, d) @ (d,)` matmul shape error in the new
position tilt — TensorFlow does not allow that broadcast. Fixed with
`tf.linalg.matvec(particles - mu_global[None, :], beta)`.

**This is where we are right now.** The matvec fix is committed locally but
the test has *not been re-run* with the position-correlated training data
yet. We do not yet know whether fix 2 actually changes the outcome.

### 5.8 JIT acceleration discussion (deferred)
Separately, the user pointed out that the neural operator training does not
use JIT compilation — `train_step` is fully eager. 2000 steps takes ~10
minutes on a 3090. Codex provided a clean plan for adding XLA-compiled
training:

- Module-level `_make_compiled_train_step(...)` factory closing over Python
  branch params.
- Module-level `_warmup_model_and_optimizer(...)` helper.
- Try `jit_compile=True` first; if XLA fails (likely culprit:
  `tf.linalg.eigvalsh`), fall back to `jit_compile=False`; if even that
  fails, fall back to eager.
- Keep the existing `train_step` function unchanged as the eager reference.
- Don't move data sampling inside the compiled region (`n_components` is
  dynamic).
- Don't decorate `NeuralOperatorResampler.__call__` yet; defer compiled
  inference to step 12.

This plan was approved but **not implemented** before the break. It is the
clean next mechanical task to pick up.

---

## 6. Files touched during Step 11 (current state)

| File | What changed |
|---|---|
| `code/src/resampling/types.py` | Added optional `local_jacobians` field to `ResampleResult` |
| `code/src/resampling/neural_operator_resample.py` | New: `NeuralOperatorResampler` class with inlined Silverman bandwidth, returns mapped particles + uniform weights + local Jacobians |
| `code/src/resampling/__init__.py` | Exported `NeuralOperatorResampler` |
| `code/neural_operator/src/data.py` | Two rounds of fixes: (1) gamma sampling + multimodality, (2) position-correlated weights via linear tilt + per-component bias + Dirichlet noise. Last fix also corrected a matvec shape bug. |
| `code/neural_operator/tests/test_resampler.py` | New: smoke test of resampler interface (shapes, weights, PSD J, determinism). Quality assertion intentionally NOT here. |
| `code/neural_operator/tests/test_resampler_quality.py` | New: quality test on 4 hand-crafted low-ESS clouds. Trains 2000 steps. Currently the failing test. |
| `code/neural_operator/issues_to_be_addressed/2_quality_test.md` | This document |

`train.py`, `losses.py`, `encoder.py`, `conditional_map.py`, `kde.py`,
`models.py` — **untouched in Step 11** (touched only in earlier steps).

---

## 7. Two interpretations on the table

There are two reads of the quality-test failure. They lead to different
next moves. Both come from the same Codex session and both could be right.

### 7a. Distribution shift interpretation (Codex's current main hypothesis)

**Story:** The failure is *training data*, not loss design.

- Until fix 2, the training distribution had weights drawn independently of
  particle position. The four quality-test clouds have weights *correlated*
  with position (by mode, by linear tilt). The model never saw the relevant
  pattern.
- A near-identity fallback on out-of-distribution clouds is exactly what
  you would expect from a model that was trained on a different family.
- Codex's exact phrase: *"strong evidence that the model was trained on the
  wrong family of weight-position relationships."*

**Prediction:** After training on the position-correlated data (fix 2),
plus enough steps to actually move the trunk away from its identity basin,
the trained T should beat identity on the four clouds.

**Status:** untested. Fix 2 has not been retrained-and-tested yet.

**Next move under this interpretation:** add JIT acceleration (so 2000
steps is fast), retrain, re-run the quality test, look at the new numbers.

### 7b. MA-loss-is-too-weak interpretation (my earlier hypothesis)

**Story:** Even with the right training data, MA residual gives a weak
gradient signal for moments. The trunk has a free output bias that is only
loosely controlled by the PDE constraint, so the optimizer prefers to leave
it near identity. The output bias is the natural place for moment drift.

**Prediction:** Even after fix 2 + 2000 steps, T will continue to be
near-identity on the structured clouds.

**Next move under this interpretation:** add a normalized auxiliary moment
loss on `T(x_i)` against the weighted empirical moments (Codex's "approach
B"). Concretely:

```
L_mean = || mean(T(x_i)) − Σ_i w_i x_i ||² / (|| Σ_i w_i x_i ||² + ε)
L_cov  = || cov(T(x_i)) − Σ_i w_i (x_i − μ_w)(x_i − μ_w)^T ||²_F /
         (|| Σ_i w_i (x_i − μ_w)(x_i − μ_w)^T ||²_F + ε)
loss   = L_ma + 0.1 * L_mean + 0.1 * L_cov
```

Codex's view: this is sound, but **don't add it until we know fix 2 alone
isn't enough**. Otherwise we will not know which thing fixed it.

### Which is correct?

We don't know. The cleanest experiment is:

1. Run fix 2 with enough training steps (2000+).
2. Look at the result.
3. If T now beats identity on the four clouds — interpretation 7a was
   right, we are done with Step 11 and can move on to Step 12.
4. If T still hugs identity — interpretation 7b becomes likely, add the
   normalized moment loss as the next intervention.

The matvec crash blocked us from getting to step 1. The JIT acceleration
work was started to make step 1 cheap to iterate on.

### 7c. Test design problem: KDE bandwidth vs mode gap

Realized after the fact: three of the four hand-crafted clouds in
`test_resampler_quality.py` are **structurally unsolvable** by a KDE-based
operator, regardless of how the model is trained.

The clouds in question (0, 1, 2) all have within-mode standard deviation
$\sigma \approx 0.4{-}0.6$ but mode separation of $4{-}8$. Silverman's
rule gives bandwidth

$$
h \;\approx\; \sigma \cdot N^{-1/(d+4)} \;\approx\; 0.4 \cdot 200^{-1/5} \;\approx\; 0.14,
$$

so the gap between modes is $\sim 30 h$. The KDEs $p_h$ and $q_h$ are
essentially zero in the gap. Collocation points sampled from $p_h$ during
training **never land between modes**, so the operator gets no gradient
signal for moving particles across the gap. The trained map cannot transport
mass from one mode to another because it never saw the gap during training,
and on the test cloud it falls back to near-identity.

Cloud 3 (uniform support on $[-2, 2]$ with exponential weight tilt) is the
only cloud in the regime the network is designed for: the support is a
single connected region, the KDE is smooth everywhere, and the operator's
collocation points cover the support. **Cloud 3 also failed** in the latest
run (`mean_err = 0.997` vs identity $1.000$), so the model issue is real,
but only one of the four data points was actually informative.

**Implications:**

1. The previous failure is *less damning* than the headline numbers
   suggested. Three of the four clouds are unsolvable test inputs, not
   evidence of model failure. The actual signal is one cloud (cloud 3)
   showing T behaving as identity on a smooth distribution.
2. The four hand-crafted clouds need to be redesigned into the
   smooth-cloud regime before the next quality run.
3. The new clouds should have within-cloud spread comparable to or larger
   than the mode separation, so the KDE bandwidth bridges any structure
   the operator is asked to handle.

**Proposed redesign for the next iteration:**

| Cloud | Particles | Weights | Why this is in the right regime |
|---|---|---|---|
| A | $x_i \sim \mathcal{N}(0, 2^2)$, $N=200$ | $w_i \propto e^{0.5\,x_i}$ | Single broad mode, smooth tilt |
| B | half from $\mathcal{N}(-1, 1)$, half from $\mathcal{N}(+1, 1)$ | $w_i \propto e^{0.7\,x_i}$ | Weakly bimodal: gap $\approx \sigma$, KDE bridges |
| C | $x_i \sim \mathcal{N}(0, 3^2)$ | $w_i \propto e^{-(x_i - 1.5)^2 / 4}$ | Wide mode, Gaussian likelihood centered off-axis |
| D | $x_i \in [-2, 2]$ uniform spaced, $N=200$ | $w_i \propto e^{2 x_i}$ | Original cloud 3 — already smooth |

For all four, ESS in the 0.3–0.6 range and the weighted-vs-unweighted mean
gap should be at least one within-cloud standard deviation, so identity is
a meaningful baseline error.

This redesign should be done **before** the next quality-test run, not
after. Otherwise we will spend more compute confirming that unsolvable
inputs are unsolvable.

### 7d. Graded ladder result + training instability is the dominant problem

After the test was redesigned into a smooth-cloud graded ladder
(`test_resampler_quality.py` now has tiers 1–4 at increasing weight skew
on the same broad Gaussian particle distribution), we re-ran with the
existing model and **no other changes**.

**Cloud ladder:**

| Tier | β (exp tilt) | ESS/N | weighted mean |
|---|---|---|---|
| 1 (easy)        | 0.10 | 0.96 | +0.45 |
| 2 (easy-medium) | 0.30 | 0.73 | +1.18 |
| 3 (medium)      | 0.40 | 0.57 | +1.53 |
| 4 (hard)        | 0.55 | 0.36 | +2.05 |

**Per-tier results (run 1, no other changes):**

| Tier | identity_mean | T_mean | verdict |
|---|---|---|---|
| 1 (easy) | 0.81 | **1.80** | fail (T much worse than identity) |
| 2 | 0.93 | 1.22 | fail |
| 3 | 0.94 | 1.11 | fail |
| 4 (hard) | 0.96 | 1.01 | fail (T converges to identity) |

The model fails on **every tier**, including the easiest (`ESS/N = 0.96`,
the cloud is barely skewed). On easy tiers T is much worse than identity;
on hard tiers it regresses to identity. The pattern is consistent with a
weakly-context-sensitive map applying nearly the same correction to
every cloud — that correction overshoots on easy clouds and is masked by
the worse identity baseline on hard ones.

**Training is wildly unstable.** Loss varies by 100×–1000× across steps,
gradient norms hit several thousand at peak, and the bandwidth `h`
swings 24× across steps:

```
step  200: loss=0.24    |grad|=2.3      h=3.13
step  400: loss=19.6    |grad|=18       h=0.81
step  600: loss=0.05    |grad|=0.4      h=1.84
step  800: loss=3.6     |grad|=35       h=0.54
step 1000: loss=42.0    |grad|=2896     h=0.13   <- biggest spike
step 1400: loss=11.3    |grad|=252      h=0.45
```

The spikes correlate cleanly with small `h` (smoking gun: step 1000 has
the smallest `h` and the biggest gradient norm). To confirm this with a
full per-step record we instrumented `train_step` and `train` to write a
CSV trace of `(step, loss, grad_norm, h, h_scale, n_eff,
weighted_std, unweighted_std, residual_abs_mean, …)` to
`code/neural_operator/tests/traces/baseline.csv`. Pure logging, no
behavior change.

**Diagnosis (Codex confirmed).** The mechanism is the **weighted std**
inside `silverman_bandwidth_scalar` collapsing onto the heavy-weight
region of skewed clouds. Each training cloud has different weights, so
each cloud gets a different `h`. Sharp clouds give a tiny `h`, which
makes the KDE peaked, the loss stiff, and the gradient huge. Calm clouds
give a normal `h`. The optimizer takes a giant step on a sharp cloud,
relaxes on a calm one, then spikes again — never settles.

(Important correction: my earlier guess that `N_eff^{-1/(d+4)}` was the
collapse driver was wrong. That factor actually *grows* as ESS drops,
which would *increase* `h`. The actual collapse driver is the weighted
covariance shrinking onto the heavy region.)

**Codex's ranked single-fix list (apply one at a time):**

1. **Source-based bandwidth** in `kde.py`: replace weighted std + `N_eff`
   with **unweighted** std + raw `N`. The bandwidth then describes the
   particle cloud's geometry, not its weight skew, so all training
   clouds at the same particle spread get the same `h`. Most likely
   root cause.
2. **`h` floor** at `0.15 × std_unweighted` as a safety net.
3. **Lower learning rate** from `1e-3` to `3e-4`.
4. **Tighten gradient clipping** from `10` to `5`.
5. **Turn on small local-linearity penalty** (`linearity_weight = 1e-2`),
   currently zero.

(The earlier proposal to add an auxiliary moment loss is deferred until
the bandwidth fix has been tested. Sobolev / Gaussian-to-Gaussian
pretraining is filed as a contingency in
`3_gradient_supervision.md` — use it only if the bandwidth fix and the
moment loss both fail to recover the easy tier.)

The graded test now lazily caches the trained model to
`code/neural_operator/tests/checkpoints/operator_v1`. To force a retrain
after a single fix, delete that directory before running.

---

## 8. What is *not* known (open questions)

1. **Does fix 2 (position-correlated training data) actually change the
   trained model's behavior on the four held-out clouds?** Not measured yet.
2. If fix 2 helps but does not fully close the gap, **how big does the
   training budget need to be** before we declare it insufficient and add
   the moment loss?
3. **Is `tf.linalg.eigvalsh` actually a problem under XLA?** Codex flagged
   it as the most likely XLA blocker. We will find out as soon as JIT is
   enabled.
4. **Should the neural operator return `local_jacobians` in float64 for the
   downstream LEDH covariance transport (Step 12), or is float32 enough?**
   Not yet relevant but worth keeping in mind.
5. **Will a model trained on synthetic clouds generalize to real PF
   snapshots?** This is the Step 15 question. We are assuming yes for now.

---

## 9. Concrete next-session pickup list

Ordered. Each step is small, reversible, and **applies exactly one
change before re-running**. No bundles. The discipline is: change one
variable, observe, decide. See `~/.claude/.../feedback_one_fix_at_a_time.md`.

**State already in place (do not redo):**
- Graded ladder test (`test_resampler_quality.py`, 4 tiers, ESS/N from
  0.96 to 0.36).
- Lazy checkpoint at `code/neural_operator/tests/checkpoints/operator_v1`.
- Per-step trace dumped to
  `code/neural_operator/tests/traces/baseline.csv` whenever training runs.
- Baseline trace from the graded ladder run is committed to office.

**Next steps:**

1. **Apply Codex's fix #1: source-based bandwidth in `kde.py`.**
   In `silverman_bandwidth_scalar`, replace
   `weighted_covariance(particles, weights)` with the unweighted
   covariance, and replace `n_eff = 1 / Σ wᵢ²` with the raw particle
   count `N`. Two-line change. The bandwidth then depends on the
   particle cloud's geometry only, not on the weight skew.
2. Delete the cached checkpoint to force a retrain:
   `rm -rf code/neural_operator/tests/checkpoints/operator_v1`.
3. Run `python neural_operator/tests/test_resampler_quality.py`. The
   trace will be written to `traces/baseline.csv`, overwriting the
   previous baseline.
4. **Decision point** (compare against the previous baseline):
   - **If `h` is now stable across steps and the gradient spikes are
     gone**, the diagnosis is confirmed. Look at the per-tier results.
     - **Tier 1 PASSES**: the bandwidth fix was sufficient. Move on
       to evaluating the higher tiers, then to Step 12 of the main
       neural operator plan.
     - **Tier 1 still fails but training is now stable**: training is
       fine, the loss is just not strong enough. Apply Codex's next
       single change (h floor → lower LR → tighter clip → local
       linearity penalty), one at a time.
     - **Tier 1 still fails AND training is still wild**: bandwidth
       was not the (only) instability source. Look at the new trace
       to find what changed and what didn't.
   - **If `h` is now stable but the per-tier numbers are unchanged**,
     the bandwidth was a real bug but not the root cause. Move down
     Codex's list.
5. Whatever happens in step 4, **write a follow-up entry in §10 below**
   with the new per-tier numbers, the new training-trace summary, and
   the next single change to try.

**Deferred work** (do not start until tier 1 passes):
- JIT acceleration of `train_step` (Codex-approved plan: module-level
  `_make_compiled_train_step` factory, warmup, XLA-first fallback).
  Speeds the iteration loop from ~10 min per training run to ~30 sec.
  Touch only after the model is actually learning.
- Step 12 onwards: LEDH local-Jacobian covariance transport, end-to-end
  gradient test through the compiled filter, benchmark vs implicit
  Sinkhorn, scale to d=2.

## 10. Run history

Each entry is one training run with one change applied. Add to the
bottom; do not edit prior entries.

| Date | Change applied | Trace file | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Notes |
|---|---|---|---|---|---|---|---|
| 2026-04-10 | None (graded ladder + instrumentation only) | `traces/baseline.csv` | fail (id 0.81 → T 1.80) | fail (0.93 → 1.22) | fail (0.94 → 1.11) | fail (0.96 → 1.01) | Training wildly unstable. Loss jumps 100×, grad spikes to 475. h varies 24× across steps. Spikes correlate with small h. |
| 2026-04-11 | Same — second run for instrumentation reproducibility | `traces/baseline.csv` (overwritten) | fail (0.81 → 1.34) | fail (0.93 → 1.00) | fail (0.94 → 0.95) | PASS by luck (0.96 → 0.88) | Same instability. Spike at step 1000 hit `\|grad\|=2896`, 6× the previous run. Random seed dictates which tier passes. |

---

## 10. Codex session id

The Codex session for this step is resumable via:
```
codex exec --skip-git-repo-check resume --last
```
identifying as Claude (claude-opus-4-6) in the next message. Two
consultations have already happened in that session: (a) initial diagnosis
of the data generator bug and the smoke-test framing issue; (b) the
distribution-shift hypothesis and the JIT acceleration plan.
