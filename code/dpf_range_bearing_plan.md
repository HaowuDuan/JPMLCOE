# Plan: Range-Bearing DPF Experiments + Pytest Diagnostic Framework

## Context

Two goals:
1. Add DPF parameter inference experiments for the range-bearing model (mirror of the Kitagawa setup)
2. Remove diagnostic print code from the production pipeline and replace with a pytest-based diagnostic/testing framework using callback injection

---

## Part 1: Range-Bearing DPF Config

### Model parameters to infer
Range-bearing has two observation noise params: `sigma_range` and `sigma_bearing` (both 0.1 true). These are the analogs of `sigma_W` in Kitagawa. Process noise `Q` (0.01·I) is fixed.

### Config: `code/configs/dpf/range_bearing_ledh_hmc.yaml`

```yaml
# @package _global_
model:
  _target_: src.models.range_bearing.RangeBearingModel
  sigma_range: 0.3      # initial guess (true: 0.1)
  sigma_bearing: 0.3    # initial guess (true: 0.1)

filter:
  _target_: src.filters.particle.ledh_invertible_hmc.LEDHParticleFlowFilterHMC
  n_particles: 500
  n_lambda_steps: 29
  resampling_method: ot_entropy
  resampling_config:
    epsilon: 0.5
  weight_clip_range: 50.0
  stop_gradient_resampling: true
  eager_mode: false

dpf:
  sampler: hmc    # TFP tfp.mcmc.HamiltonianMonteCarlo

  trainable_params:
    sigma_range:
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: -2.3     # centered near 0.1
        scale: 0.5
    sigma_bearing:
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: -2.3
        scale: 0.5

  hmc:
    num_samples: 500
    num_burnin: 250
    step_size: 0.01
    num_leapfrog_steps: 10
    max_tree_depth: 10
    adaptation_rate: 0.8
    target_accept_prob: 0.7
    seed: 42

data:
  T: 200
  seed: 42
  true_params:
    sigma_range: 0.1
    sigma_bearing: 0.1
```

**Note**: `RangeBearingModel.observation_noise_cov` returns `self.R`, a `tf.constant` computed at `__init__` time.
When `DifferentiableModel.update_parameters` calls `setattr(model, 'sigma_range', new_val)`, `self.R` becomes stale.
Fix: make `R` a property computed on the fly from `sigma_range`/`sigma_bearing`.

---

## Part 2: Remove Debug Code from Production Pipeline

### What to remove

1. **`DPFRunner.debug_gradients`** flag and `if self.debug_gradients: print(...)` block in `_value_and_grad`
   — [code/src/DF/hmc_runner.py](code/src/DF/hmc_runner.py)
2. **`LEDHParticleFlowFilterHMC.debug_gradients`** attribute and per-timestep print block in `_run_eager`
   — [code/src/filters/particle/ledh_invertible_hmc.py](code/src/filters/particle/ledh_invertible_hmc.py)
3. Remove `debug_gradients: true` from all dpf yaml configs

### What replaces it

Both classes get an optional **callback** parameter. Callbacks are `None` by default (zero overhead in production), and tests inject callables that collect data.

**`DPFRunner.__init__`**: add `on_grad: Optional[Callable] = None`
```python
# signature: on_grad(step: int, nlp: float, grad: tf.Tensor) -> None
```
Called inside `_value_and_grad` only if not None.

**`LEDHParticleFlowFilterHMC.__init__`**: add `on_timestep: Optional[Callable] = None`
```python
# signature: on_timestep(t: int, log_lik_t: float, ess: float, max_log_theta: float) -> None
```
Called inside `_run_eager` only if not None. Compiled path is unaffected (can't inject into `tf.while_loop`).

---

## Part 3: Pytest Testing Framework

### File structure

```
code/tests/
├── conftest.py                          # shared fixtures (existing)
├── filters/particle/
│   └── test_hmc_gradient.py            # existing gradient tests (keep)
└── dpf/
    ├── __init__.py
    ├── conftest.py                      # dpf-specific fixtures
    ├── test_range_bearing_unit.py       # unit tests: gradient finiteness, filter forward pass
    └── test_range_bearing_integration.py # integration tests: short HMC run, posterior sanity
```

### `code/tests/dpf/conftest.py` — shared fixtures

```python
@pytest.fixture
def range_bearing_model():
    """Model at true params (sigma_range=0.1, sigma_bearing=0.1)."""

@pytest.fixture
def range_bearing_model_wrong():
    """Model at initial guess params (sigma_range=0.3, sigma_bearing=0.3)."""

@pytest.fixture
def short_observations(range_bearing_model):
    """T=20 synthetic observations using true model, seed=42."""

@pytest.fixture
def ledh_filter_hmc(range_bearing_model_wrong):
    """LEDHParticleFlowFilterHMC with n_particles=100, n_lambda_steps=10, eager_mode=True."""

@pytest.fixture
def grad_collector():
    """Dict accumulating on_grad callback calls: {'steps', 'nlps', 'norms'}."""

@pytest.fixture
def timestep_collector():
    """Dict accumulating on_timestep callback calls: {'ts', 'log_liks', 'ess_vals', 'max_log_thetas'}."""
```

### Unit tests: `test_range_bearing_unit.py`

| Test | What it checks |
|------|----------------|
| `test_gradient_finite_at_true_params` | Gradient at true params is finite and nonzero |
| `test_gradient_finite_at_wrong_params` | Gradient at 3× wrong params is finite |
| `test_gradient_magnitude_reasonable` | Initial \|grad\| < 1e6 via `on_grad` callback |
| `test_ess_does_not_collapse` | ESS > 10% of n_particles at every timestep via `on_timestep` |
| `test_per_timestep_log_lik_finite` | Per-timestep log-likelihood is finite everywhere |
| `test_filter_log_lik_positive_at_true_params` | log p(y\|θ_true) > log p(y\|θ_wrong) |

### Integration tests: `test_range_bearing_integration.py`

All marked `@pytest.mark.slow`:

| Test | What it checks |
|------|----------------|
| `test_short_hmc_run_produces_samples` | 10 burnin + 10 samples → DPFResult with correct shape |
| `test_posterior_mean_closer_to_truth` | Posterior mean closer to 0.1 than initial guess 0.3 |
| `test_acceptance_rate_nonzero` | HMC acceptance rate > 0% |
| `test_gradient_norm_during_hmc` | Max \|grad\| during 10-step burn-in < 1e7 |

### Pytest marks

```bash
pytest code/tests/dpf/                        # unit only (fast)
pytest code/tests/dpf/ -m slow               # integration only
pytest code/tests/dpf/ -m "not slow"         # unit only (explicit)
```

---

## Part 4: Step-by-Step Debug Plan (Gradient Explosion Root Cause)

### Step 1 — Baseline at true params
**Test**: `test_gradient_finite_at_true_params`
**Goal**: If gradient is already huge at true params, problem is structural, not parameter-dependent.

### Step 2 — ESS collapse at wrong params
**Test**: `test_ess_does_not_collapse` with `range_bearing_model_wrong`
**Goal**: At σ=0.3 (3× off), does ESS → 1? Confirms weight collapse as root cause.

### Step 3 — Gradient vs ESS correlation
**Test**: `test_gradient_norm_during_hmc` + `timestep_collector` injected
**Goal**: Does \|grad\| spike exactly when ESS is minimum? Confirms ESS → gradient cascade.

### Step 4 — Float32 vs float64
**Test**: Parameterize `ledh_filter_hmc` fixture with `dtype` in [float32, float64]
**Goal**: If float64 reduces \|grad\| by orders of magnitude → CUDA float32 precision is root cause.

### Step 5 — Weight clip sensitivity
**Test**: Parameterize `weight_clip_range` in [5, 10, 20, 50]
**Goal**: Find tightest clip that keeps ESS > 10% with reasonable gradient magnitude.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `code/configs/dpf/range_bearing_ledh_hmc.yaml` | **Create** |
| `code/src/models/range_bearing.py` | **Modify**: make `R` a property computed from `sigma_range`/`sigma_bearing` |
| `code/src/DF/hmc_runner.py` | **Modify**: remove `debug_gradients`, add `on_grad` callback |
| `code/src/filters/particle/ledh_invertible_hmc.py` | **Modify**: remove `debug_gradients`, add `on_timestep` callback |
| `code/configs/dpf/kitagawa_ledh_hmc_sys.yaml` + all variants | **Modify**: remove `debug_gradients: true` |
| `code/tests/dpf/__init__.py` | **Create** |
| `code/tests/dpf/conftest.py` | **Create** |
| `code/tests/dpf/test_range_bearing_unit.py` | **Create** |
| `code/tests/dpf/test_range_bearing_integration.py` | **Create** |

---

## Verification

```bash
# Run unit tests only (fast, no HMC)
cd code && python -m pytest tests/dpf/test_range_bearing_unit.py -v

# Run integration tests (slow, real HMC)
cd code && python -m pytest tests/dpf/test_range_bearing_integration.py -m slow -v

# Debug step-by-step: isolate gradient explosion
cd code && python -m pytest tests/dpf/test_range_bearing_unit.py::test_ess_does_not_collapse -v -s

# Run actual range-bearing DPF experiment
cd code && python -m src.experiments.run_dpf_experiment dpf=range_bearing_ledh_hmc
```
