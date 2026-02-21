# Particle Filter Test Plan — Group 2 & 3

## Scope

**Group 2a** — Bootstrap Particle Filter (`bootstrap_pf_tf.py`)
**Group 2b** — Deterministic flow filters:
- Kernel flow / EDH flow (`kernel_flow.py`)
- LEDH flow with global covariance (`ledh_flow.py`)
- LEDH invertible with per-particle covariance (`ledh_invertible_bimodal.py`)

**Group 3** — Stochastic EDH (separate plan, later)

**Not in scope:** HMC wrappers (`bootstrap_pf_hmc.py`, `ledh_invertible_hmc.py`).
Those are already covered by `test_hmc_bpf.py` and `test_hmc_ledh.py`.
This plan tests the **filters themselves** — correctness, weight health, numerical stability.

---

## What Already Exists

| File | Covers |
|------|--------|
| `test_hmc_bpf.py` | BPF log-likelihood vs KF, gradient direction, autodiff vs FD |
| `test_hmc_ledh.py` | LEDH log-likelihood vs KF, gradient, Jacobian uniformity, weight uniformity |
| `test_kalman_family.py` | EKF/UKF reduce to KF on linear-Gaussian |
| `test_utils_*.py` | linalg, distributions, flow_params, ODE solvers |

**Gap:** No standalone filter correctness tests (posterior mean/cov vs Kalman),
no systematic numerical health monitoring, no stress tests, no resampling tests,
no EDH flow or LEDH flow (non-invertible) tests at all.

---

## Test Organization

```
code/tests/
  plans/
    PARTICLE_FILTER_TEST_PLAN.md    <-- this file
  test_resampling.py                <-- new: shared resampling component
  test_bootstrap_pf.py              <-- new: Group 2a
  test_flow_filters.py              <-- new: Group 2b
  conftest_filters.py               <-- new: shared fixtures and diagnostic harness
```

---

## Shared Infrastructure: `conftest_filters.py`

### Fixtures

**`lg_model_and_kf_result`** — Linear-Gaussian model + Kalman filter ground truth.
- 2D state, 1D observation, T=50 timesteps, fixed seed.
- Returns: model, observations, KF means, KF covs, KF log-likelihood.
- This is the gold standard: every filter must converge to these values.

**`nl_model_and_data`** — Nonlinear model (Kitagawa or cubic sensor) + observations.
- For tests where linear-Gaussian is too easy.
- No exact ground truth, but we can compare filters against each other.

### Diagnostic Collector

A helper class that wraps a filter run and records per-timestep health indicators.
NOT a base class or monkey-patch — just reads from the filter's existing diagnostics
(ESS history, weights history) and computes derived metrics.

Records:
- `ess_ratio[t]` = ESS(t) / N
- `log_weight_range[t]` = max(log_w) - min(log_w) before normalization
- `max_weight[t]` = max(normalized_w)
- `has_nan[t]` = any NaN in weights or particles
- `particle_spread[t]` = tr(sample covariance of particles)
- `n_unique_after_resample[t]` = number of unique particles after resampling

For flow filters additionally:
- `log_theta_range[t]` = max(log_theta) - min(log_theta) across particles
- `log_theta_mean[t]` = mean accumulated log-Jacobian
- `flow_drift_max[t]` = max |drift| across particles at each lambda step
- `A_max_eigenvalue[t]` = max real eigenvalue of A matrix (should be <= 0)

---

## File 1: `test_resampling.py`

Tests the three resampling methods in isolation before they're used inside filters.
All tests use synthetic weight vectors — no filter needed.

### R1 — Systematic resampling is unbiased

Feed weights = [0.5, 0.3, 0.2] (N=3 particles in 1D).
Run 10000 times with different seeds.
Empirical frequency of selecting particle i should match w_i within Monte Carlo error.

**Assert:** |freq_i - w_i| < 3/sqrt(10000) for each i.

### R2 — Systematic resampling: uniform weights produce uniform selection

weights = [1/N] * N. After resampling, every particle should be selected exactly once
(systematic resampling is deterministic given uniform weights and a single random offset).

**Assert:** resampled indices are a permutation of [0, ..., N-1].

### R3 — Systematic resampling: degenerate weights select one particle

weights = [0, 0, ..., 1, ..., 0]. All resampled particles should be copies of the
single non-zero particle.

**Assert:** all resampled particles identical to particle k.

### R4 — Soft resampling: alpha=1.0 equals hard resampling

With alpha=1.0, soft resampling reduces to systematic resampling.
Same weights, same seed → same resampled particles.

**Assert:** particles and output weights match systematic exactly.

### R5 — Soft resampling: alpha=0.0 is pure importance sampling

With alpha=0.0, q(k) = 1/N (uniform). No actual resampling.
Output weights = w_k / (1/N) = N * w_k, then renormalized.

**Assert:** resampled particles are just the originals (no shuffling),
output weights proportional to input weights.

### R6 — Soft resampling: output weights sum to 1

For any alpha in {0.1, 0.5, 0.9}, any input weights → output weights sum to 1,
all non-negative.

### R7 — OT resampling: output weights sum to 1, all finite

For any epsilon in {0.01, 0.1, 1.0} → output weights sum to 1, no NaN/Inf.

### R8 — OT resampling: uniform weights → particles unchanged

If input weights are uniform, optimal transport has no work to do.
Resampled particles should be (nearly) identical to input.

### R9 — All methods: extreme weight ratio doesn't crash

weights = softmax([100, 0, 0, ..., 0]). One particle has nearly all weight.
All three methods should return finite particles and weights without exception.

---

## File 2: `test_bootstrap_pf.py`

### Tier 1 — Correctness vs Kalman

#### B1 — Posterior mean converges to Kalman mean

Linear-Gaussian model. Run BPF with N=5000 particles, fixed seed.
Compare filter posterior mean at each timestep to Kalman mean.

**Assert:** max |BPF_mean(t) - KF_mean(t)| < threshold (e.g., 0.5 for 2D state).

Why 5000: bootstrap PF on a 2D linear-Gaussian with well-separated observations
needs ~1000+ particles for the mean to be accurate. 5000 gives margin.

#### B2 — Posterior mean is unbiased (Monte Carlo over seeds)

Run BPF with N=500 particles across 20 different seeds.
Average the 20 posterior mean trajectories.
The averaged mean should be closer to Kalman than any single run.

**Assert:** |avg_BPF_mean(t) - KF_mean(t)| < |single_BPF_mean(t) - KF_mean(t)|
for at least 80% of timesteps.

Why: A single PF run has variance. If the mean over seeds doesn't converge,
something is systematically biased (wrong weight formula, biased resampling).

#### B3 — Log-likelihood converges to Kalman log-likelihood

Same setup as B1. Sum of log p(y_t | y_{1:t-1}) should approach KF value.

**Assert:** relative error < 10%.

This is already partially tested in `test_hmc_bpf.py` but through the HMC wrapper.
Here we test the base `ParticleFilterTF` directly (no HMC machinery).

#### B4 — Resampling method doesn't change the posterior mean (much)

Run BPF with same seed, same particles, but different resampling methods
(systematic, soft, OT). All should produce similar posterior means.

**Assert:** max difference between any two methods < 2x the KF error.

Why: Resampling affects variance, not bias. If one method gives a wildly
different mean, it has a bug.

### Tier 2 — Numerical Health

#### B5 — No NaN/Inf at any timestep

Run BPF on both linear-Gaussian and nonlinear models.
Check weights, particles, and log-likelihood at every timestep.

**Assert:** no NaN or Inf anywhere in diagnostics.

#### B6 — ESS never drops below 2

On linear-Gaussian at true parameters, the bootstrap PF should maintain
reasonable ESS (observations are not too informative).

**Assert:** min ESS > 2 across all timesteps.

Why 2: ESS=1 means complete weight collapse. ESS=2 is the minimum for
the filter to have any diversity. On a well-tuned linear problem, ESS
should be much higher, but 2 is the "something is catastrophically wrong" line.

#### B7 — Weight range stays bounded

Log-weight range = max(log_w) - min(log_w) before normalization.
On linear-Gaussian this should stay moderate.

**Assert:** max log-weight range < 50 across all timesteps.

Why 50: exp(50) ~ 5e21. Beyond this, softmax numerics start to degrade
even with the max-subtraction trick. On a well-tuned linear problem
the range should be < 10.

#### B8 — Particle diversity after resampling

After systematic resampling, count unique particles.
On linear-Gaussian, should retain decent diversity.

**Assert:** n_unique / N > 0.1 at every timestep where resampling occurs.

### Tier 3 — Edge Cases

#### B9 — Very few particles (N=10) doesn't crash

Filter should complete without exception. Results will be poor.

**Assert:** no exception, all outputs finite.

#### B10 — Very informative observation (R = 0.001 * I) doesn't crash

Likelihood is a spike. ESS will collapse. But the filter should not produce NaN.

**Assert:** no NaN in particles or weights. ESS will be low — that's expected.

---

## File 3: `test_flow_filters.py`

Parametrized over filter classes. Each filter gets the same suite of tests.

**Filter matrix:**

| Short name | Class | Has weights? | Has Jacobian? | Covariance type |
|---|---|---|---|---|
| `kernel_flow` | `KernelMappingPF` | No (equal weights) | No | N/A |
| `ledh_flow` | `LocalExactDaumHuangFlow` | Yes | Yes | Global (single EKF) |
| `ledh_inv` | `LEDHParticleFlowFilter` | Yes | Yes | Per-particle (batched EKF) |
| `ledh_inv_bimodal` | `LEDHInvertibleBimodal` | Yes | Yes | Per-particle + lookahead |

### Tier 1 — Correctness vs Kalman

#### F1 — Posterior mean converges to Kalman mean

Same as B1 but for each flow filter. N=500 particles (flow filters
are more efficient than bootstrap, so fewer particles needed).

**Assert:** max |flow_mean(t) - KF_mean(t)| < threshold.

Expected: flow filters should be MORE accurate than bootstrap PF
with the same N, because the flow transports particles to high-likelihood
regions rather than relying on random importance sampling.

#### F2 — Posterior covariance converges to Kalman covariance

For flow filters (especially LEDH invertible which tracks per-particle cov),
the sample covariance should be meaningful.

**Assert:** |tr(flow_cov(t)) - tr(KF_cov(t))| / tr(KF_cov(t)) < 0.5.

Why 0.5: particle sample covariance is noisy. With N=500 and d=2,
50% relative error on trace is a reasonable pass criterion.

#### F3 — Flow filters outperform bootstrap PF at same N

Run BPF and each flow filter with N=200 on the same linear-Gaussian data.
Compare MSE of posterior mean vs Kalman.

**Assert:** flow filter MSE < BPF MSE (or at most equal).

Why: this validates that the flow is actually doing useful work. If a flow
filter is worse than bootstrap, the flow is doing more harm than good.

#### F4 — Kernel flow produces equal weights

Kernel flow is a deterministic transport — no importance weights.
After every update, all particle weights should be exactly 1/N.

**Assert:** all weights == 1/N at every timestep.

### Tier 2 — Weight Health (LEDH flow and LEDH invertible only)

#### F5 — No NaN/Inf in weights or particles at any timestep

**Assert:** no NaN/Inf in weights, particles, log_theta, or log-likelihood.

#### F6 — Jacobian accumulation is bounded

log_theta = sum of log|det(I + dlambda * A)| over lambda steps.
On linear-Gaussian (constant H), all particles have identical A, so
log_theta should be identical across particles.

**Assert (linear-Gaussian):** std(log_theta) / |mean(log_theta)| < 0.01
(coefficient of variation < 1%).

This is the same check as test 2.6 in `test_hmc_ledh.py`, but here we
run it on the non-HMC filter directly.

#### F7 — Jacobian spread on nonlinear model stays bounded

On a nonlinear model, particles DO have different Jacobians, so log_theta
will vary. But the spread should not explode.

**Assert:** max(log_theta) - min(log_theta) < 30 at every timestep.

Why 30: exp(30) ~ 1e13. If the Jacobian correction varies by more than
13 orders of magnitude across particles, it's destabilizing the weights
rather than helping.

#### F8 — ESS stays above collapse threshold

On linear-Gaussian at true parameters, LEDH should maintain high ESS
because the flow transports particles to the posterior (weights should
be nearly uniform after a good flow).

**Assert:** average ESS/N > 0.5 across timesteps.

Same check as test 2.7 in `test_hmc_ledh.py`, but for the non-HMC filter.

#### F9 — det(I + dlambda * A) stays positive at each lambda step

The matrix I + dlambda * A should be positive definite at each flow step
(the flow is a diffeomorphism). A negative determinant means the mapping
has folded, which invalidates the change-of-variables formula.

Requires instrumented flow loop (subclass or manual stepping).

**Assert:** det(I + dlambda * A) > 0 for all particles, all lambda steps.

### Tier 2 — Flow Health (all flow filters)

#### F10 — Particles don't explode during flow

Track max |particle| at each lambda step. Should stay bounded.

**Assert:** max particle norm < 1000 * max(initial particle norm).

#### F11 — Flow drift magnitude is bounded

At each lambda step, drift = A @ eta + b. The drift magnitude should
not spike by orders of magnitude, which would indicate stiffness.

**Assert:** max |drift| doesn't increase by more than 100x from first to last
lambda step.

#### F12 — A matrix eigenvalues are non-positive

The Daum-Huang flow matrix A = -0.5 * P @ H^T @ (lambda*HPH + R)^-1 @ H
should have non-positive eigenvalues (stable, contracting flow).

**Assert:** max real eigenvalue of A <= epsilon (small positive tolerance).

Already tested in `test_utils_flow_params.py` (T4.4, T4.8) for the
compute functions. Here we verify it holds DURING an actual filter run
where P and H change at each timestep.

### Tier 3 — Edge Cases

#### F13 — Very few particles (N=10) doesn't crash

**Assert:** no exception, all outputs finite.

#### F14 — Single lambda step (n_lambda_steps=1) doesn't crash

Minimal flow. The filter should still produce a valid result, just less accurate.

**Assert:** no exception, outputs finite, posterior mean not wildly wrong.

#### F15 — Very informative observation doesn't crash

R = 0.001 * I. Flow filter should handle the tight likelihood better than
bootstrap PF (this is what flow filters are designed for).

**Assert:** no NaN. ESS should be higher than BPF under the same conditions.

#### F16 — LEDH invertible bimodal: lookahead with cubic sensor

On cubic sensor (y = x^3/20 + noise), the bimodal variant should maintain
particles in both modes, while standard LEDH collapses to one.

**Assert:** after update, particle sample has bimodal structure
(two clusters with separation > some threshold).

This is a qualitative test — hard to set a precise threshold.
Implementation approach: compute the sample kurtosis or check that
particles span both positive and negative values when truth is positive.

---

## Diagnostic Infrastructure: How to Instrument the Flow Loop

The filters' `update()` methods run the lambda loop internally. For Tier 2 flow
health tests (F9, F10, F11, F12), we need per-lambda-step data.

**Approach: Test subclass with instrumented update()**

Create a thin subclass in the test file that overrides `update()`:
1. Copies the parent's update logic
2. Records per-step diagnostics into a list
3. Calls the same math — no behavior change

This is the same pattern used in `test_hmc_ledh.py` test 2.6 (manual flow loop).
Generalize it into a reusable helper in `conftest_filters.py`.

**Alternative (simpler, less thorough):** Only check post-update diagnostics.
Use the filter's built-in `metadata` dict (ESS, weights, log_theta).
This covers F5-F8 without any instrumentation. Only F9-F12 need the
per-lambda-step access.

**Recommendation:** Start with the post-update checks (F5-F8). Add the
per-lambda-step instrumentation only if post-update checks pass but
you suspect internal instability. This avoids premature complexity.

---

## Priority Order for Implementation

| Priority | Tests | Rationale |
|---|---|---|
| 1 | R1-R9 | Resampling is a shared foundation. If it's broken, all filter tests are meaningless. |
| 2 | B1, B3 | BPF Kalman match. Simplest filter, validates weight-resample-estimate pipeline. |
| 3 | B5-B8 | BPF diagnostics. Establishes the health check pattern. |
| 4 | F1, F2, F4 | Flow filter Kalman match. Validates flow + Jacobian end-to-end. |
| 5 | F5-F8 | Flow filter weight health. Post-update checks, no instrumentation needed. |
| 6 | F3 | Flow vs BPF comparison. Validates the flow actually helps. |
| 7 | F9-F12 | Flow internals. Requires instrumented loop. Only if F5-F8 pass. |
| 8 | B9-B10, F13-F16 | Edge cases. Only after correctness is confirmed. |

---

## Relationship to Existing HMC Tests

The HMC tests (`test_hmc_bpf.py`, `test_hmc_ledh.py`) test through the HMC wrapper:
- Log-likelihood as a function of parameters
- Gradient direction and magnitude
- Autodiff vs finite difference

The tests in this plan test the **filter directly**:
- Posterior mean/cov accuracy
- Per-timestep numerical health
- Internal flow stability

These are complementary. If a filter test fails here, the corresponding HMC test
will likely also fail (bad filter → bad likelihood → bad gradient). But HMC tests
can pass while filter tests fail (e.g., log-likelihood is correct on average but
individual timesteps have weight collapse that cancels out).

---

## Models for Testing

| Model | Dimension | Why |
|---|---|---|
| Linear-Gaussian | 2D state, 1D obs | Gold standard. Kalman gives exact answer. |
| Kitagawa | 1D state, 1D obs | Classic nonlinear benchmark. Highly nonlinear transition. |
| Cubic sensor | 1D state, 1D obs | Bimodal posterior. Tests LEDH bimodal variant. |

Avoid higher dimensions for now — tests should be fast (< 30s each).
Higher-dimensional stress tests can be added in a follow-up.

---

## How to Run

```bash
cd /Users/haowuduan/Documents/githubrepos/JPMLCOE/code

# All particle filter tests:
.venv/bin/python -m pytest tests/test_resampling.py tests/test_bootstrap_pf.py tests/test_flow_filters.py -v -s

# Just resampling:
.venv/bin/python -m pytest tests/test_resampling.py -v -s

# Just bootstrap PF:
.venv/bin/python -m pytest tests/test_bootstrap_pf.py -v -s

# Just flow filters:
.venv/bin/python -m pytest tests/test_flow_filters.py -v -s

# Save full output:
.venv/bin/python -m pytest tests/test_resampling.py tests/test_bootstrap_pf.py tests/test_flow_filters.py -v -s 2>&1 | tee tests/pf_test_results.txt
```
