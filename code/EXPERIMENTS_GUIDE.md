# Experiments Guide

## Overview

The framework includes a Hydra-based experiment runner that makes it easy to run and compare different filters with configurable parameters.

## Quick Start

### Running Pre-configured Experiments

```bash
# Linear Gaussian with Kalman Filter
python -m src.experiments.run_experiment experiment=linear_test

# Stochastic Volatility with EKF
python -m src.experiments.run_experiment experiment=sv_comparison filter=ekf

# Stochastic Volatility with UKF
python -m src.experiments.run_experiment experiment=sv_comparison filter=ukf

# Stochastic Volatility with Particle Filter
python -m src.experiments.run_experiment experiment=sv_comparison filter=particle

# Range-Bearing tracking with UKF
python -m src.experiments.run_experiment experiment=range_bearing_test

# Flow filter comparison
python -m src.experiments.run_experiment experiment=flow_comparison filter=edh_flow
python -m src.experiments.run_experiment experiment=flow_comparison filter=ledh_flow
python -m src.experiments.run_experiment experiment=flow_comparison filter=edh_invertible
python -m src.experiments.run_experiment experiment=flow_comparison filter=ledh_invertible
python -m src.experiments.run_experiment experiment=flow_comparison filter=kernel_flow
```

### Customizing Experiments via Command Line

Override any parameter from the command line:

```bash
# Change number of particles
python -m src.experiments.run_experiment filter=particle filter.n_particles=5000

# Change random seed and timesteps
python -m src.experiments.run_experiment seed=123 T=200

# Change model parameters
python -m src.experiments.run_experiment model=stochastic_volatility model.alpha=0.95 model.beta=0.7

# Enable plotting
python -m src.experiments.run_experiment plot=true

# Change output directory
python -m src.experiments.run_experiment output_dir=my_results
```

## Experiment Configuration Files

Pre-configured experiments are in `configs/experiment/`:

- **linear_test.yaml**: Kalman Filter on Linear Gaussian model
- **sv_comparison.yaml**: Compare filters on Stochastic Volatility
- **range_bearing_test.yaml**: UKF on Range-Bearing tracking
- **flow_comparison.yaml**: Compare flow filters

## Creating Custom Experiments

### Option 1: Command Line Override

Simply override parameters from the command line:

```bash
python -m src.experiments.run_experiment \
  model=stochastic_volatility \
  filter=particle \
  filter.n_particles=2000 \
  T=150 \
  seed=42 \
  plot=true
```

### Option 2: Create Custom Config File

Create a new file in `configs/experiment/`:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_

defaults:
  - override /model: stochastic_volatility
  - override /filter: particle

# Experiment settings
T: 200
seed: 99
plot: true
output_dir: outputs/my_experiment

# Model parameters
model:
  alpha: 0.95
  sigma: 1.2
  beta: 0.6

# Filter parameters
filter:
  n_particles: 2000
  resample_threshold: 0.3
```

Then run:

```bash
python -m src.experiments.run_experiment experiment=my_experiment
```

## Output Structure

Results are saved to the output directory (default: `outputs/`):

```
outputs/
└── run_name/
    ├── means.npy              # Filtered state estimates
    ├── covs.npy               # Covariance matrices
    ├── states.npy             # True states
    ├── observations.npy       # Observations
    ├── summary.json           # Metrics and metadata
    ├── plot.png              # Visualization (if plot=true)
    └── .hydra/               # Hydra config tracking
        └── config.yaml
```

## Available Models

Configure via `model=<name>` or in experiment YAML:

- **linear_gaussian**: Linear-Gaussian systems (optimal for KF)
- **stochastic_volatility**: 1D nonlinear stochastic volatility
- **range_bearing**: 2D tracking with range-bearing observations

## Available Filters

Configure via `filter=<name>` or in experiment YAML:

**Kalman Family:**
- **kalman**: Standard Kalman Filter
- **ekf**: Extended Kalman Filter
- **ukf**: Unscented Kalman Filter

**Particle Methods:**
- **particle**: Bootstrap Particle Filter (TensorFlow + GPU)

**Flow Filters:**
- **edh_flow**: Exact Daum-Huang flow (no resampling)
- **ledh_flow**: Local EDH flow (kernel-based)
- **edh_invertible**: EDH with global linearization
- **ledh_invertible**: LEDH with per-particle linearization
- **kernel_flow**: Kernel mapping PF (scalar or matrix kernels)

## Comparing Multiple Filters

To compare multiple filters, run them separately and use the visualization module:

```python
from src.experiments.visualization import plot_comparison
import numpy as np

# Load results from multiple runs
results = {
    'EKF': {
        'means': np.load('outputs/run1/means.npy'),
        'covs': np.load('outputs/run1/covs.npy')
    },
    'UKF': {
        'means': np.load('outputs/run2/means.npy'),
        'covs': np.load('outputs/run2/covs.npy')
    },
    'PF': {
        'means': np.load('outputs/run3/means.npy'),
        'covs': np.load('outputs/run3/covs.npy')
    }
}

states = np.load('outputs/run1/states.npy')
observations = np.load('outputs/run1/observations.npy')

plot_comparison(states, observations, results, save_path='comparison.png')
```

## Hydra Features

### Multi-run for Parameter Sweeps

Run experiments with different parameters:

```bash
# Sweep over particle counts
python -m src.experiments.run_experiment -m \
  filter=particle \
  filter.n_particles=100,500,1000,5000

# Sweep over model parameters
python -m src.experiments.run_experiment -m \
  model=stochastic_volatility \
  model.alpha=0.8,0.9,0.95

# Sweep over seeds
python -m src.experiments.run_experiment -m \
  seed=1,2,3,4,5
```

### Accessing Config in Code

```python
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="configs", config_name="config")
def my_experiment(cfg: DictConfig):
    print(cfg.model)
    print(cfg.filter)
    print(cfg.T)
```

## Tips

1. **Start small**: Use fewer timesteps (T=20-50) when testing flow filters as they're computationally intensive
2. **GPU for particles**: Bootstrap PF automatically uses GPU if available
3. **Threading for flows**: Flow filters use CPU threading - adjust `n_threads` parameter
4. **Reproducibility**: Set `seed` for reproducible experiments
5. **Output organization**: Use meaningful `output_dir` names for easy result tracking

## Example Workflow

Complete workflow for comparing filters:

```bash
# 1. Run experiments
python -m src.experiments.run_experiment experiment=sv_comparison filter=ekf
python -m src.experiments.run_experiment experiment=sv_comparison filter=ukf
python -m src.experiments.run_experiment experiment=sv_comparison filter=particle

# 2. Results are saved with timestamped names in outputs/
# 3. Load and compare results using visualization module
# 4. Or simply enable plot=true to get individual plots

# Quick comparison with plots
python -m src.experiments.run_experiment experiment=sv_comparison filter=ekf plot=true
python -m src.experiments.run_experiment experiment=sv_comparison filter=ukf plot=true
python -m src.experiments.run_experiment experiment=sv_comparison filter=particle plot=true
```

## Advanced: Custom Experiment Script

For complex comparisons, create a custom script:

```python
from src.experiments.run_experiment import run_filter_experiment
from src.experiments.visualization import plot_comparison, plot_log_likelihood_comparison
from hydra import compose, initialize
import numpy as np

# Initialize Hydra
initialize(config_path="configs", version_base=None)

# Run multiple filters
results = {}
for filter_name in ['ekf', 'ukf', 'particle']:
    cfg = compose(config_name="config", overrides=[
        f"filter={filter_name}",
        "model=stochastic_volatility",
        "T=100",
        "seed=42"
    ])
    results[filter_name.upper()] = run_filter_experiment(cfg)

# Compare
states = results['EKF']['states']
observations = results['EKF']['observations']

comparison_data = {
    name: {'means': r['means'], 'covs': r['covs']}
    for name, r in results.items()
}

plot_comparison(states, observations, comparison_data, save_path='filter_comparison.png')

# Compare log-likelihoods
log_liks = {name: r['log_likelihood'] for name, r in results.items()}
plot_log_likelihood_comparison(log_liks, save_path='loglik_comparison.png')
```
