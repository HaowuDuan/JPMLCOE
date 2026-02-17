# Plan: Reproduce Paper Section 4 (Two-Sensor Bearing MC Experiment)

## Goal
Reproduce Table 1 from "Stiffness Mitigation in Stochastic Particle Flow Filters" Section 4:
20 Monte Carlo runs comparing linear schedule (β=λ) vs optimal schedule (β*(λ)).
Uses the same `config.yaml` + `+experiment=` pattern as `run_experiment.py`.

## What stays the same (no changes needed)
- Model: `TwoSensorBearingOnlyModel` — already correct
- Filter: `StochasticEDHFlow` — already implements both schedules
- `configs/config.yaml` — reused as-is
- `FlowFilterBase.filter()` — re-seeds particles each call, handles the MC loop naturally

---

## Change 1: Fix `configs/filter/stochastic_edh_optimal.yaml`

Wrong param names that don't exist in `StochasticEDHFlow.__init__`.

**Replace entire file with:**
```yaml
_target_: src.filters.particle.stochastic_edh.StochasticEDHFlow
n_particles: 50
n_lambda_steps: 100
diffusion_scale: 0.0
schedule_mu: 0.2
```

Leave `configs/filter/stochastic_edh.yaml` unchanged — changing it would affect all other
experiments. Instead, override n_particles in the experiment config (see Change 2).

---

## Change 2: New experiment config `configs/experiment/two_sensor_bearing/two_sensor_bearing_mc.yaml`

Uses `filter: stochastic_edh` (linear) as the base — the runner creates the optimal
filter by overriding `schedule_mu=0.2` at instantiation time (Hydra supports this).
`n_particles` is overridden here to match the paper, leaving the filter default untouched.

```yaml
# @package _global_

defaults:
  - override /model: two_sensor_bearing
  - override /filter: stochastic_edh

output_dir: outputs/two_sensor_bearing/mc

# Paper Section 4 fixed values
truth: [4.0, 4.0]
z_fixed: [0.4754, 1.1868]
n_mc_runs: 20

# Override filter params for this experiment only
filter:
  n_particles: 50       # paper uses 50; stochastic_edh.yaml default left unchanged
```

---

## Change 3: New runner `src/experiments/run_mc_experiment.py`

Same `config_name="config"` as `run_experiment.py`. Instantiates both filters from
the same cfg.filter — the optimal one just overrides `schedule_mu`.

```python
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    model = hydra.utils.instantiate(cfg.model, dtype=dtype_tf)

    truth   = np.array(cfg.truth)
    z_fixed = np.array(cfg.z_fixed).reshape(1, -1)  # shape (1, obs_dim) for filter()

    filter_linear  = hydra.utils.instantiate(cfg.filter, model=model)
    filter_optimal = hydra.utils.instantiate(cfg.filter, model=model, schedule_mu=0.2)

    for run in range(cfg.n_mc_runs):
        rng = np.random.default_rng(run)          # CRN: same seed for both
        r_lin = filter_linear.filter(z_fixed,  random_state=rng)
        rng   = np.random.default_rng(run)
        r_opt = filter_optimal.filter(z_fixed, random_state=rng)
        # collect MSE = ||mean - truth||^2, tr_P = trace(covs[0])

    # print table + save bar plot to output_dir
```

Run with:
```bash
python -m src.experiments.run_mc_experiment +experiment=two_sensor_bearing/two_sensor_bearing_mc
```

---

## Summary of files to touch

| File | Action |
|------|--------|
| `configs/filter/stochastic_edh_optimal.yaml` | Fix wrong param names |
| `configs/filter/stochastic_edh.yaml` | **No change** |
| `configs/experiment/two_sensor_bearing/two_sensor_bearing_mc.yaml` | **New** — adds truth, z_fixed, n_mc_runs |
| `src/experiments/run_mc_experiment.py` | **New** — same pattern as run_experiment.py, adds MC loop |

No changes to model, filter, `run_experiment.py`, or `config.yaml`.
