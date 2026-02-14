# HMC Bayesian Inference Readiness Report

## Context

This report audits the codebase to determine if all components needed for Hamiltonian Monte Carlo (HMC) Bayesian inference with an arbitrary differentiable filter are in place. We assume the differentiability of the filter itself is handled separately.

---

## 1. WHAT EXISTS AND IS READY

### 1.1 HMC/MCMC Infrastructure (fully built)

The `code/src/DF/` directory contains a **complete HMC framework** on TensorFlow Probability:

| File | What it does |
|------|-------------|
| `code/src/DF/hmc_runner.py` | `DPFRunner` class: wraps model, manages param transforms via `ParameterHandler`, defines `_negative_log_posterior()` = `-log p(y|theta) - log p(theta)`, runs `tfp.mcmc.HamiltonianMonteCarlo` with `SimpleStepSizeAdaptation`. Computes ESS, R-hat, acceptance rate, posterior summary. |
| `code/src/DF/parameter_handler.py` | `ParameterHandler`: TFP bijectors (`Exp`, `Sigmoid`, scaled `Sigmoid`, `Identity`) for constraints, constrained<->unconstrained transforms, log-prior with Jacobian adjustment. |
| `code/src/DF/differentiable_model.py` | `DifferentiableModel` wrapper: intercepts attribute access, dynamically updates trainable params on the base model. |
| `code/src/DF/types.py` | `ParameterSpec` dataclass (name, init_value, constraint, prior as `tfp.distributions.Distribution`), `DPFResult` dataclass. |
| `code/src/DF/example_usage.py` | Working example: infers `alpha` and `sigma` of `StochasticVolatilityModel` using `ExactDaumHuangFlow` as inner filter. |

### 1.2 Prior Distribution Infrastructure (ready)

`ParameterSpec` accepts any `tfp.distributions.Distribution` as a prior. Constraint types: `'positive'`, `'unit'`, `(a, b)` intervals, unconstrained (`None`).

### 1.3 Hydra DPF Configs (ready)

- `code/configs/dpf/stochastic_volatility.yaml` -- complete config with model, filter, trainable params, priors, HMC settings.
- `code/configs/dpf/linear_gaussian.yaml` -- template (partially configured).

### 1.4 Filter Infrastructure (extensive, 13+ filters)

**Kalman filters** (all TF, all compute log-likelihoods):
- `KalmanFilter` -- linear, returns `log_likelihood` in `FilterResult`
- `ExtendedKalmanFilter` -- `@tf.function` predict/update, per-step log-lik via innovation covariance
- `UnscentedKalmanFilter` -- `@tf.function` predict/update, per-step log-lik

**Particle filters** (all TF):
- `ParticleFilterTF` (bootstrap) -- `@tf.function` main loop, log-lik via log-sum-exp
- `EDHParticleFlowFilter` (invertible) -- computes `log_likelihood`
- `LEDHParticleFlowFilter` (local EDH) -- computes `log_likelihood`
- `ExactDaumHuangFlow` -- **returns `log_likelihood=None`**
- `StochasticEDHFlow` -- inherits from above, **`log_likelihood=None`**

### 1.5 Log-Likelihood via `FilterResult`

`FilterResult` (in `code/src/core/types.py`) has:
- `log_likelihood: float` -- total log p(y_{1:T})
- `log_likelihoods: Optional[np.ndarray]` -- per-timestep (T,)

### 1.6 Differentiable Resampling (ready)

- **Soft resampling** (`code/src/resampling/soft.py`) -- maintains gradients via importance-weighted mixture
- **OT entropy resampling** (`code/src/resampling/ot_entropy.py`) -- Sinkhorn + `@tf.custom_gradient` for implicit differentiation

### 1.7 Models (8 total, mixed TF readiness)

| Model | Fully TF? | Log-lik compatible? |
|-------|-----------|-------------------|
| `LinearGaussianModel` | Yes | Yes |
| `RangeBearingModel` | Yes | Yes |
| `TwoSensorBearingOnlyModel` | Yes | Yes |
| `StochasticVolatilityModel` | Partial (numpy batch methods) | Needs TF batch methods |
| `KitagawaModel` | No (all numpy) | No |
| `AcousticTrackingModel` | No (numpy) | No |
| `Lorenz96Model` | No (entirely numpy) | No |

### 1.8 Dependencies (ready)

From `code/requirements.txt`:
- `tensorflow>=2.13.0,<2.17.0`
- `tensorflow-probability>=0.21.0` (HMC, bijectors, distributions)
- `hydra-core>=1.3.0`, `omegaconf>=2.3.0`
- `numpy`, `scipy`, `matplotlib`

---

## 2. CRITICAL ISSUES (must fix before HMC runs)

### 2.1 `tf.py_function` in HMC Runner Breaks Gradients

**File:** `code/src/DF/hmc_runner.py`, lines 77-94

```python
@tf.function(reduce_retracing=True)
def _negative_log_posterior(self, unconstrained_params):
    ...
    log_likelihood = tf.py_function(update_and_run_filter, [], tf.float32)
```

`tf.py_function` is **not differentiable** -- it exits the TF computation graph. HMC needs gradients of the log-posterior. The filter forward pass must run entirely inside TF's graph/tape.

### 2.2 `DifferentiableModel` Converts Tensors to Python Floats

**File:** `code/src/DF/differentiable_model.py`, lines 53-60

```python
def update_parameters(self, param_dict):
    value = float(param_value.numpy())   # <-- breaks gradient tape
    setattr(self._base_model, param_name, value)
```

Converting to `float` via `.numpy()` severs the computation graph. Parameters must stay as `tf.Tensor` or `tf.Variable` for gradients to flow.

### 2.3 Flow Filters Return `log_likelihood=None`

**File:** `code/src/filters/particle/flow_base.py`, line 146

`ExactDaumHuangFlow` and `StochasticEDHFlow` both inherit `FlowFilterBase` which hardcodes `log_likelihood=None`. The example usage in `example_usage.py` uses `ExactDaumHuangFlow`, so `result.log_likelihood` will be `None`, causing a crash in the HMC runner.

### 2.4 Filters Convert Log-Likelihood to Python Float at Return

Even filters that compute log-likelihood convert to numpy at return, e.g. EKF:
```python
total_log_likelihood = float(tf.reduce_sum(log_liks_tf).numpy())
```
This returns a Python `float`, not a TF tensor -- incompatible with gradient tape.

### 2.5 No End-to-End Differentiable Filter Function Signature

No filter currently provides a single TF-traceable function:
```
(params_tensor, observations) -> log_likelihood_tensor
```
This is the key missing abstraction for plugging any filter into HMC.

---

## 3. MINOR GAPS

| Gap | Impact | Effort |
|-----|--------|--------|
| No NUTS sampler (only basic HMC) | Suboptimal tuning of leapfrog steps | Small -- swap to `tfp.mcmc.NoUTurnSampler` |
| No DPF experiment runner script | Can't launch from Hydra CLI | Medium -- create `run_dpf_experiment.py` |
| Mixed numpy/TF in some models | Those models can't be used with HMC | Per-model effort to convert |

---

## 4. SUMMARY SCORECARD

| Component | Status |
|-----------|--------|
| HMC sampler code | EXISTS but BROKEN (gradient issue) |
| Parameter handler (bijectors, priors) | READY |
| Prior distributions | READY |
| Parameter constraints | READY |
| DifferentiableModel wrapper | EXISTS but BROKEN (.numpy() kills gradients) |
| EKF/UKF with log-lik | READY (needs thin wrapper) |
| Bootstrap PF with log-lik | READY (needs thin wrapper) |
| Flow filters with log-lik | MISSING (returns None) |
| Invertible EDH/LEDH with log-lik | READY |
| Fully TF models | 3 of 7 ready |
| Differentiable resampling | READY |
| Hydra DPF configs | READY |
| DPF experiment runner | MISSING |
| NUTS sampler | MISSING (easy add) |
| End-to-end `(params, obs) -> log_lik` | MISSING (key abstraction) |
| TF + TFP dependencies | READY |

---

## 5. RECOMMENDED FIX PLAN (priority order)

### Step 1: Create a differentiable filter interface

Define an abstract method / protocol that every filter must implement:

```python
def log_marginal_likelihood_tf(self, observations: tf.Tensor) -> tf.Tensor:
    """Run full filter, return scalar log p(y_{1:T}) as a tf.Tensor (not float)."""
```

This is the single contract that HMC needs. Each filter implements it by keeping all computation in TF.

### Step 2: Fix `DifferentiableModel` to keep TF tensors

Replace `.numpy()` conversion with `tf.Variable.assign()` or direct tensor assignment so parameters stay on the TF graph.

### Step 3: Rewrite `_negative_log_posterior` without `tf.py_function`

Have the filter's forward pass run directly inside `tf.GradientTape`, using the new `log_marginal_likelihood_tf` method.

### Step 4: Add log-likelihood to flow filters

Override `filter()` in `FlowFilterBase` subclasses to compute:
```
log_lik_t = logsumexp(log_obs_probs) - log(N)
```

### Step 5: Add NUTS support

One-line swap to `tfp.mcmc.NoUTurnSampler` with a config flag.

### Step 6: Create DPF experiment runner

`run_dpf_experiment.py` that reads Hydra config, generates data, runs HMC, saves results.

### Quickest path to a working demo

Use **EKF + LinearGaussianModel** or **EKF + RangeBearingModel** -- both are fully TF and return log-likelihoods. Fix issues 2.1-2.2 (the `tf.py_function` and `.numpy()` problems), add a thin wrapper, and HMC should run.
