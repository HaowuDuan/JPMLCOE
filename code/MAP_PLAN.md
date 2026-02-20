# Plan: Add `run_map()` + Reorganize DPF Configs

## Context

HMC-based DPF works well with BPF+systematic but LEDH causes leapfrog divergence. Before investing in HMC tuning, we want a fast MAP estimation method (Adam/SGD) to verify that gradients point toward the true parameters. Also, the 24 flat config files in `configs/dpf/` need reorganization into subdirectories by sampler method and model name.

---

## Part 1: Add `run_map()` to DPFRunner

**File: `code/src/DF/hmc_runner.py`**

Add a `run_map()` method that runs Adam (or SGD) optimization to find the MAP point. Reuses existing infrastructure: `param_handler`, `diff_model`, `filter_obj`.

```python
def run_map(
    self,
    observations: np.ndarray,
    num_steps: int = 200,
    learning_rate: float = 0.01,
    optimizer: str = 'adam',       # 'adam' or 'sgd'
    random_seed: bool = False,     # True = different PF seed each step (stochastic)
    seed: int = 42,
    print_every: int = 10,
) -> DPFResult:
```

**Core loop logic:**
1. Init `q = tf.Variable(self.param_handler.unconstrained_init)`
2. Create `tf.keras.optimizers.Adam(learning_rate)` or `SGD`
3. Each step:
   - `GradientTape` over: constrain -> update model -> `log_marginal_likelihood_tf` -> log_prior -> loss = -(ll + prior)
   - Seed: `[42, step]` if `random_seed=True`, else `[42, 0]` (fixed, deterministic surface)
   - `optimizer.apply_gradients([(grad, q)])`
   - Track loss history, best q, print progress every `print_every` steps
4. Return `DPFResult`:
   - `samples`: dict with MAP point as 1-element arrays (the best q found)
   - `summary`: mean/std from optimization trace (last N steps)
   - `diagnostics`: `{'final_loss': ..., 'loss_history': [...], 'converged': bool}`
   - `metadata`: optimizer type, num_steps, learning_rate, etc.

**Why not reuse `_negative_log_posterior`**: It has a hardcoded seed and `tf.print` diagnostic. Better to inline the loss computation in `run_map()` directly for clarity and to support per-step random seeds.

---

## Part 2: Wire MAP into Experiment Runner

**File: `code/src/experiments/run_dpf_experiment.py`**

Add `sampler == 'map'` branch (alongside existing `'pmmh'` and HMC branches):

```python
elif sampler == 'map':
    from src.DF import DPFRunner
    runner = DPFRunner(...)
    map_cfg = cfg.dpf.map
    result = runner.run_map(
        observations=observations,
        num_steps=map_cfg.num_steps,
        learning_rate=map_cfg.learning_rate,
        optimizer=map_cfg.get('optimizer', 'adam'),
        random_seed=map_cfg.get('random_seed', False),
        seed=map_cfg.get('seed', 42),
    )
```

Results printing section already handles `DPFResult` -- works as-is since MAP returns the same type.

---

## Part 3: Reorganize Config Directory

**Current structure** (24 files, flat):
```
configs/dpf/
  kitagawa_ledh_hmc_sys.yaml
  kitagawa_bpf_hmc_ot.yaml
  ...
```

**New structure** (sampler -> model -> filter_resampling):
```
configs/dpf/
  hmc/
    linear_gaussian/
      ledh_sys.yaml
      ledh_soft.yaml
      ledh_ot.yaml
      bpf_sys.yaml
      bpf_soft.yaml
      bpf_ot.yaml
    cubic_sensor/
      ...
    kitagawa/
      ...
    range_bearing/
      ...
  map/
    linear_gaussian/
      ledh.yaml
      bpf.yaml
    cubic_sensor/
      ...
    kitagawa/
      ...
    range_bearing/
      ...
```

**Hydra usage changes:**
- Old: `python -m src.experiments.run_dpf_experiment dpf=kitagawa_ledh_hmc_sys`
- New: `python -m src.experiments.run_dpf_experiment dpf=hmc/kitagawa/ledh_sys`

**Output dir** (`${hydra:runtime.choices.dpf}`) naturally becomes `outputs/dpf/hmc/kitagawa/ledh_sys/` -- no change needed in `config_dpf.yaml`.

**MAP config template** (e.g., `configs/dpf/map/kitagawa/ledh.yaml`):
```yaml
# @package _global_
model:
  _target_: src.models.kitagawa.KitagawaModel
  sigma_V: 5.0
  sigma_W: 2.0
  initial_var: 5.0

filter:
  _target_: src.filters.particle.ledh_invertible_hmc.LEDHParticleFlowFilterHMC
  n_particles: 200
  n_lambda_steps: 29
  resampling_method: ot_entropy
  resampling_config:
    epsilon: 0.5
  weight_clip_range: 50.0
  stop_gradient_resampling: false
  eager_mode: false

dpf:
  sampler: map
  trainable_params:
    sigma_V:
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 2.3
        scale: 0.5
    sigma_W:
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 0.0
        scale: 1.0
  map:
    num_steps: 200
    learning_rate: 0.01
    optimizer: adam
    random_seed: false
    seed: 42

data:
  T: 100
  seed: 42
  true_params:
    sigma_V: 10.0
    sigma_W: 1.0
```

---

## Execution Order

1. Add `run_map()` method to `DPFRunner` in `code/src/DF/hmc_runner.py`
2. Add `sampler == 'map'` branch in `code/src/experiments/run_dpf_experiment.py`
3. Create new directory structure and move existing HMC configs
4. Create MAP configs (8 files: 4 models x 2 filters)

## Verification

```bash
# Test MAP on easiest model first
cd code && python -m src.experiments.run_dpf_experiment dpf=map/linear_gaussian/bpf

# Test HMC still works with new paths
python -m src.experiments.run_dpf_experiment dpf=hmc/linear_gaussian/bpf_sys
```
