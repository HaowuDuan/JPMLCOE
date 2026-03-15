# LEDH Gradient Diagnosis Plan

## Problem Statement

The LEDH particle flow filter produces systematically wrong parameter inference
on the 1D linear Gaussian model (`obs_noise_std` true=1.0).

### Evidence

| Filter   | Resampling | HMC Posterior Mean | Correct? |
|----------|------------|-------------------|----------|
| Kalman   | —          | 1.12              | YES      |
| EKF      | —          | 1.12              | YES      |
| UKF      | —          | 1.12              | YES      |
| BPF      | OT         | 1.16              | YES      |
| BPF      | Soft       | 1.15              | YES      |
| BPF      | Systematic | 1.18              | YES      |
| **LEDH** | **OT**     | **0.16**          | **NO**   |
| **LEDH** | **Soft**   | **0.07**          | **NO**   |
| **LEDH** | **Systematic** | **0.47**      | **NO**   |

MAP with LEDH also fails: params drift to ~0.6 with |grad|≈32.

### Conclusion from Evidence

The bug is **definitively in the LEDH particle flow**, not in resampling
(all resampling methods fail equally), not in the prior, not in the HMC/MAP
runner (BPF works with the same runner).

## LEDH Gradient Chain

The LEDH log-marginal-likelihood gradient flows through these components:

```
obs_noise_std → R = σ² · D·Dᵀ
                ↓
    ┌──────────┼──────────────────────────┐
    ↓          ↓                          ↓
  R_inv    flow_params(R, P, H, ...)   log p(y|η₁, R)
    ↓          ↓           ↓
              A(λ), b(λ)
    ↓          ↓           ↓
           η₁ positions   log|det(I + dλ·A)|
           (29 Euler       (29 steps accumulated
            steps)          into log_theta)
    ↓          ↓           ↓
         log p(η₁|x_{k-1})  log θ
    ↓          ↓           ↓
    └──────────┴───────────┘
               ↓
        log w = log p(y|η₁) + log p(η₁|x) + log θ - log p(η₀|x) + log w_prev
               ↓
        log marginal likelihood = Σ_t logsumexp(log w_t)
```

BPF has none of the flow components (no A/b, no η₁ flow, no Jacobian θ).
The LEDH-specific components are the suspects.

## Diagnostic Tests

All tests use the 1D linear Gaussian model (simplest case, KF gives exact reference).
Tests are in `code/tests/hmc/test_ledh_gradient_diagnosis.py` (pytest).

### Test A: Likelihood Surface Bias

**Question**: Is the LEDH log-likelihood surface itself biased, or just the gradient?

**Method**: Evaluate LEDH, BPF, and KF log-likelihood at a grid of obs_noise_std
values. If LEDH likelihood peaks far from 1.0, the forward computation is biased.

**Interpretation**:
- LEDH peak ≈ KF peak → likelihood is correct, gradient is broken
- LEDH peak ≠ KF peak → likelihood is biased (more fundamental issue)

### Test B: Autodiff vs Finite Difference

**Question**: Does TF's autodiff gradient match the true gradient of the LEDH likelihood?

**Method**: Central finite difference at multiple obs_noise_std values with fixed seed.

**Interpretation**:
- Autodiff ≈ FD → gradient chain is correct; the likelihood surface is just wrong
- Autodiff ≠ FD → backward pass is broken (custom gradient, tf.function, etc.)

### Test C: Likelihood-Only Gradient (No Prior)

**Question**: Is the gradient bias from the filter or the prior?

**Method**: Compute gradient of `-log_likelihood` only (no prior term).

**Interpretation**:
- LEDH lik-grad matches BPF lik-grad → filter gradient is fine, prior is the issue
- LEDH lik-grad ≠ BPF lik-grad → filter backward pass is the culprit

### Test D: Component-wise Gradient Decomposition

**Question**: Which term in the weight formula has the wrong gradient?

**Method**: In eager mode, compute each component under separate GradientTapes:
1. `∂/∂σ log p(y|η₁)` — observation likelihood
2. `∂/∂σ log p(η₁|x_{k-1})` — transition prior (through flowed positions)
3. `∂/∂σ log θ` — Jacobian determinant
4. `∂/∂σ log p(η₀|x_{k-1})` — proposal (should be ~0)

**Interpretation**: The component with anomalous gradient reveals the culprit.

### Test E: Stop-Gradient Jacobian

**Question**: Is the Jacobian gradient (`graph_safe_log_abs_det_fast` custom gradient) the source of bias?

**Method**: Apply `tf.stop_gradient(theta)` before weight computation.

**Interpretation**:
- Gradient correct with stopped θ → Jacobian backward is broken
- Gradient still wrong → η₁ positions carry the bias (flow dynamics issue)

### Test F: Single Timestep (T=1)

**Question**: Does the gradient error occur per-timestep or accumulate over time?

**Method**: Run gradient check with T=1 (no resampling involved).

**Interpretation**:
- Correct at T=1, wrong at T=50 → error accumulates (resampling interaction)
- Wrong at T=1 → per-timestep flow gradient is fundamentally broken

### Test G: Eager vs Compiled

**Question**: Does `tf.function` / `tf.while_loop` compilation introduce gradient errors?

**Method**: Compare gradient in `eager_mode=True` vs `eager_mode=False`.

**Interpretation**:
- Different → compilation bug (tf.function tracing issue)
- Same → the math is wrong in both modes

### Test H: Flow Steps Sensitivity

**Question**: Does gradient bias grow with the number of Euler integration steps?

**Method**: Vary n_lambda_steps = {3, 5, 10, 15, 29}, check gradient at true parameter.

**Interpretation**:
- Bias grows → Euler discretization error accumulates
- Bias constant → per-step computation is wrong, not accumulation

### Test I: Float64 vs Float32

**Question**: Is float32 precision the root cause?

**Method**: Compare LEDH gradient in float32 vs float64.

**Interpretation**:
- Float64 correct, float32 wrong → precision issue in 29-step accumulation
- Both wrong → math bug, not precision

## Running the Tests

```bash
cd code
pytest tests/hmc/test_ledh_gradient_diagnosis.py -v -s 2>&1 | tee ledh_gradient_diagnosis.txt
```

## Decision Tree

```
Test A: Is likelihood surface biased?
├── YES → The LEDH forward computation is wrong.
│         Test D will reveal which component.
│         Test H: does it worsen with more steps?
│         Test I: does float64 fix it?
└── NO  → Likelihood is correct, gradient is broken.
          Test B: autodiff vs FD
          ├── Match → impossible (surface correct + gradient correct = should work)
          └── Mismatch → backward pass bug
              Test E: stop Jacobian gradient
              ├── Fixes it → graph_safe_log_abs_det_fast custom grad is wrong
              └── Still broken → gradient through η₁ positions is wrong
                  Test G: eager vs compiled
                  ├── Different → tf.function compilation bug
                  └── Same → flow dynamics gradient math is wrong
```

## Key Source Files

- `src/filters/particle/ledh_invertible_hmc.py:198-215` — flow loop (29 steps)
- `src/filters/particle/ledh_invertible_hmc.py:364-467` — eager mode flow loop
- `src/utils/linalg.py:208-258` — `graph_safe_log_abs_det_fast` (custom gradient)
- `src/utils/flow_params.py:167-270` — `compute_flow_params_batch` (A, b)
- `src/utils/distributions.py:118-227` — `compute_flow_weights` (weight formula)
- `src/models/linear_gaussian.py:145-151` — R property (R = σ² · D·Dᵀ)
