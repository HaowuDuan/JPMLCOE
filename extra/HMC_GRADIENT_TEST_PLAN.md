# Diagnostic Pytest Plan: HMC Log-Likelihood & Gradient

## Context

The LEDH particle flow filter's HMC parameter estimation drifts to wrong `obs_noise_std` values on the linear Gaussian model (true=1.0, estimates 0.07–0.33). We fixed the log-likelihood estimator (equally-weighted → importance-weighted), which helped systematic resampling (0.07→0.33) but soft resampling still collapses. Before guessing further, we need **empirical diagnostics** to pinpoint the exact failure.

**Strategy**: Two-stage testing on the **linear Gaussian model only** (where KF gives exact ground truth).
- **Stage 1**: BPF + sys/soft/ot — validates the baseline. If BPF fails, the bug is in resampling or the HMC runner.
- **Stage 2**: LEDH + sys/soft/ot — compares against BPF. Any difference must come from the flow, Jacobian, or importance weights.

## File

`code/tests/test_hmc_gradient_diagnosis.py`

## Shared Fixtures

```python
@pytest.fixture(scope="session")
def lg_model_and_data():
    """1D Linear Gaussian: F=0.9, B=1, H=1, D=1, true obs_noise_std=1.0, T=50."""
    model = LinearGaussianModel(F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
                                 obs_noise_std=1.0, dtype=tf.float64)
    rng = np.random.default_rng(42)
    states, _, observations = generate_data(model, T=50, rng=rng)
    return model, observations, states
```

Small params for speed: **N=200 particles, n_lambda_steps=15, eager_mode=True, float64, seed=42**.

## Helpers

```python
def compute_gradient(model, observations_tf, filter_class, filter_kwargs, param_specs):
    """One gradient evaluation. Returns (nlp, grad, constrained_params, log_lik)."""
    runner = DPFRunner(base_model=model, filter_class=filter_class,
                       filter_kwargs=filter_kwargs, param_specs=param_specs)
    runner._observations_tf = observations_tf
    q = runner.param_handler.unconstrained_init
    with tf.GradientTape() as tape:
        tape.watch(q)
        nlp = runner._negative_log_posterior(q)
    grad = tape.gradient(nlp, q)
    constrained = runner.param_handler.constrain(q)
    log_lik = -nlp - runner.param_handler.log_prior(constrained)
    return nlp, grad, constrained, log_lik

def kf_log_likelihood(model, observations):
    """Exact KF log-likelihood at current model params."""
    kf = KalmanFilter(model)
    result = kf.filter(observations)
    return result.log_likelihood
```

## Imports

```python
from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data
from src.filters.kalman.kalman import KalmanFilter
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.hmc_runner import DPFRunner
from src.DF.types import ParameterSpec
```

---

## Stage 1: BPF Baseline (6 tests)

### Test 1.1: `test_bpf_log_likelihood_vs_kf`
**Purpose**: Does BPF log-likelihood match the exact KF answer at the true parameter?

- Set `obs_noise_std=1.0` on model
- Run KF → get exact `kf_ll`
- Run BPF (sys, stop_grad=True, N=1000) → get `bpf_ll`
- **Print**: `kf_ll`, `bpf_ll`, relative error
- **Assert**: relative error < 10%

### Test 1.2: `test_bpf_log_likelihood_surface`
**Purpose**: Does the BPF log-likelihood surface peak at the same place as KF?

- For `obs_noise_std` in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
  - Set model param, compute KF ll and BPF ll (sys, N=1000)
- **Print**: table of (sigma, kf_ll, bpf_ll, difference)
- **Assert**: argmax of BPF ll matches argmax of KF ll

### Test 1.3: `test_bpf_gradient_direction`
**Purpose**: Does the BPF gradient point toward the true value from both sides?

- At `obs_noise_std=2.0` (above true): NLL gradient should be positive (push down)
- At `obs_noise_std=0.5` (below true): NLL gradient should be negative (push up)
- BPF + systematic + stop_gradient
- **Print**: sigma, grad, direction
- **Assert**: gradient signs are correct at both points

### Test 1.4: `test_bpf_autodiff_vs_finite_difference`
**Purpose**: Is the BPF autodiff gradient correct for each resampling method?

- At `obs_noise_std=2.0`, for each resampling method:
  - sys (stop_grad=True), soft (stop_grad=False), ot (stop_grad=False)
- Compute autodiff gradient and finite-difference gradient (ε=1e-4)
- **Print**: method, autodiff_grad, fd_grad, relative_error
- **Assert**: relative error < 20% for each

### Test 1.5: `test_bpf_gradient_by_resampling_method`
**Purpose**: Compare BPF gradient magnitude and direction across resampling methods.

- At `obs_noise_std=2.0`, compute gradient for sys, soft, ot
- **Print**: method, |grad|, grad_value, log_lik
- **Assert**: all gradients point same direction (positive NLL grad at sigma=2.0)
- **Assert**: |grad| within 100x of each other (flag if one explodes)

### Test 1.6: `test_bpf_gradient_vs_timesteps`
**Purpose**: Does BPF gradient grow linearly or exponentially with T?

- For T in [5, 10, 20, 50], compute gradient using first T observations
- Do for sys (stop_grad=True) and soft (stop_grad=False)
- **Print**: T, method, |grad|, |grad|/T
- **Assert**: |grad|/T stays within 10x from T=5 to T=50 (linear growth)

---

## Stage 2: LEDH vs BPF (7 tests)

### Test 2.1: `test_ledh_log_likelihood_vs_kf_and_bpf`
**Purpose**: Does LEDH log-likelihood match KF and BPF at the true parameter?

- Set `obs_noise_std=1.0`
- Compute: KF ll, BPF ll (sys), LEDH ll (sys, stop_grad=True)
- **Print**: kf_ll, bpf_ll, ledh_ll, all pairwise differences
- **Assert**: LEDH within 20% of KF

### Test 2.2: `test_ledh_log_likelihood_surface`
**Purpose**: Does the LEDH log-likelihood surface peak at the correct place?

- For `obs_noise_std` in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
  - Compute KF ll, BPF ll (sys), LEDH ll (sys, stop_grad=True)
- **Print**: table of (sigma, kf_ll, bpf_ll, ledh_ll)
- **Assert**: LEDH argmax matches KF argmax
- **Key diagnostic**: if LEDH peaks at a DIFFERENT sigma → importance weights are biased

### Test 2.3: `test_ledh_gradient_direction`
**Purpose**: Does the LEDH gradient point toward truth?

- At `obs_noise_std=2.0` and `obs_noise_std=0.5`
- Compute LEDH gradient (sys, stop_grad=True) and BPF gradient for reference
- **Print**: sigma, ledh_grad, bpf_grad
- **Assert**: LEDH gradient direction matches BPF at both points

### Test 2.4: `test_ledh_autodiff_vs_finite_difference`
**Purpose**: Is the LEDH autodiff gradient correct for each resampling method?

- At `obs_noise_std=2.0`, for sys, soft, ot:
  - Compute autodiff gradient and FD gradient (ε=1e-4)
- **Print**: method, autodiff_grad, fd_grad, relative_error
- **Assert**: relative error < 20%
- **Key diagnostic**: if FD and autodiff disagree for soft/ot but not sys → gradient-through-resampling bug

### Test 2.5: `test_ledh_gradient_by_resampling_method`
**Purpose**: Compare LEDH gradient across resampling methods.

- At `obs_noise_std=2.0`, compute gradient for sys, soft, ot
- Also compute BPF sys gradient for reference
- **Print**: filter, method, |grad|, grad_value
- **Assert**: all point same direction
- **Key diagnostic**: if LEDH soft/ot gradient has wrong sign but sys is correct → cross-timestep gradient issue

### Test 2.6: `test_ledh_jacobian_uniformity`
**Purpose**: For linear Gaussian (constant H), all particles should have identical Jacobians.

- Run LEDH filter for 1 timestep with `obs_noise_std=1.0`
- Extract per-particle log_det_J values
- **Print**: mean, std, min, max of log_det_J across particles
- **Assert**: coefficient of variation < 1% (Jacobians nearly identical)
- **Key diagnostic**: if Jacobians vary → flow linearization issue even on linear model

### Test 2.7: `test_ledh_importance_weights_uniformity`
**Purpose**: For a well-matched flow, importance weights should be nearly uniform at the true parameter.

- Run LEDH filter at `obs_noise_std=1.0` for all T=50 timesteps
- After each timestep, record ESS of importance weights
- **Print**: per-timestep ESS, ESS/N ratio, average ESS/N
- **Assert**: average ESS/N > 0.5
- **Key diagnostic**: if ESS/N << 1 even at true parameter → flow produces poor proposals, importance correction has high variance → biased log-likelihood via Jensen's inequality

---

## Run Commands

```bash
# Stage 1 only (BPF baseline):
pytest code/tests/test_hmc_gradient_diagnosis.py -v -s -k "bpf" 2>&1 | tee bpf_diagnosis.txt

# Stage 2 only (LEDH):
pytest code/tests/test_hmc_gradient_diagnosis.py -v -s -k "ledh" 2>&1 | tee ledh_diagnosis.txt

# All:
pytest code/tests/test_hmc_gradient_diagnosis.py -v -s 2>&1 | tee gradient_diagnosis.txt
```

The `-s` flag is critical — all diagnostic prints go to stdout.

## Decision Tree

| Test | If passes | If fails | Conclusion |
|------|-----------|----------|------------|
| 1.1 | BPF lik correct | BPF lik wrong | Bug in BPF log-lik formula or KF setup |
| 1.2 | BPF surface correct | BPF surface peaks wrong | BPF log-lik has parameter-dependent bias |
| 1.3 | BPF grad correct | BPF grad wrong direction | Gradient computation bug in HMC runner |
| 1.4 | Autodiff = FD | Autodiff ≠ FD | Backward pass bug in specific resampling method |
| 1.5 | All methods agree | One method explodes | Resampling-specific gradient amplification |
| 1.6 | Linear growth | Exponential growth | Cross-timestep gradient explosion (RNN-style) |
| 2.1 | LEDH ≈ KF | LEDH ≠ KF | Importance weight bias in LEDH |
| 2.2 | LEDH peaks at 1.0 | Peaks elsewhere | **Biased likelihood surface — root cause** |
| 2.3 | LEDH grad = BPF grad | Different direction | LEDH gradient distorted by flow terms |
| 2.4 | Autodiff = FD | Disagree for soft/ot | Gradient-through-resampling breaks LEDH |
| 2.5 | All methods agree | Soft/OT wrong direction | Cross-timestep gradient issue specific to LEDH |
| 2.6 | Jacobians uniform | Jacobians vary | Flow linearization bug (shouldn't happen on linear model) |
| 2.7 | ESS/N > 0.5 | ESS/N << 1 | High-variance importance weights → Jensen's bias |
