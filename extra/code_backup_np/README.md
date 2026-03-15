# Particle Filtering Framework - Part 1 Implementation

A modular, production-ready implementation of state-space models and filtering algorithms for Part 1 experiments.

## Architecture

This implementation follows a clean separation of concerns:

- **Pure NumPy for Kalman filters** (KF, EKF, UKF) - lightweight and efficient
- **TensorFlow for particle filters** - GPU-accelerated with `@tf.function` and `tf.scan`
- **Hydra configuration** - reproducible experiments with YAML configs
- **Numerical stability** - built-in safeguards for matrix operations

## Features

### Models
- **LinearGaussianModel**: Linear-Gaussian systems (optimal for KF)
- **StochasticVolatilityModel**: 1D nonlinear model (financial applications)
- **RangeBearingModel**: 2D tracking with nonlinear observations

All models provide both NumPy (for Kalman) and TensorFlow (for particles) interfaces.

### Filters

**Kalman Family (Pure NumPy):**
- **KalmanFilter**: Optimal for linear-Gaussian systems
- **ExtendedKalmanFilter**: Linearization via Jacobians
- **UnscentedKalmanFilter**: Sigma-point method (no Jacobians needed)

**Particle Methods (TensorFlow + GPU):**
- **ParticleFilter**: Bootstrap filter with systematic resampling

**Flow Filters (Pure NumPy + Threading):**
- **ExactDaumHuangFlow**: Global deterministic flow (no resampling)
- **LocalExactDaumHuangFlow**: Local kernel-based flow (better for multimodal)
- **EDHParticleFlowFilter**: Invertible flow with global linearization
- **LEDHParticleFlowFilter**: Invertible flow with per-particle linearization
- **KernelMappingPF**: Kernel-embedded gradient flow (scalar/matrix kernels)

### Resampling (TensorFlow)
- **Systematic resampling**: Low-variance, deterministic
- **Stratified resampling**: Independent per-stratum
- **ESS tracking**: Effective Sample Size monitoring

## Installation

```bash
cd code
pip install -r requirements.txt
```

Requirements:
- numpy>=1.20.0
- scipy>=1.7.0
- matplotlib>=3.3.0
- pytest>=7.0.0
- tensorflow>=2.13.0
- tensorflow-probability>=0.21.0
- hydra-core>=1.3.0
- omegaconf>=2.3.0

## Quick Start

### Run Example

```bash
python example.py
```

This will:
1. Run Kalman Filter on a 2D constant-velocity model
2. Compare EKF, UKF, and PF on Stochastic Volatility model
3. Generate plots showing filter performance

### Run Experiments with Hydra

```bash
# Run pre-configured experiments
python -m src.experiments.run_experiment experiment=linear_test
python -m src.experiments.run_experiment experiment=sv_comparison filter=ekf
python -m src.experiments.run_experiment experiment=flow_comparison filter=edh_flow

# Custom experiments via command line
python -m src.experiments.run_experiment model=stochastic_volatility filter=particle filter.n_particles=5000 T=200 plot=true
```

See [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md) for detailed usage, parameter sweeps, and comparison workflows.

### Run Tests

```bash
pytest tests/ -v
```

### Using Models

```python
from src.models.stochastic_volatility import StochasticVolatilityModel
from src.models.utils import generate_data

# Create model
model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)

# Generate data
import numpy as np
rng = np.random.default_rng(42)
states, observations = generate_data(model, T=100, rng=rng)
```

### Using Kalman Filters

```python
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter

# Create filter
ekf = ExtendedKalmanFilter(model)

# Initialize and run
ekf.initialize(mean=np.zeros(1), cov=np.eye(1))
result = ekf.filter(observations)

print(f"Log-likelihood: {result.log_likelihood}")
print(f"Filtered means shape: {result.means.shape}")
```

### Using Particle Filters

```python
from src.filters.particle.bootstrap_pf import ParticleFilter

# Create filter
pf = ParticleFilter(model, n_particles=1000, resample_threshold=0.5)

# Run (automatically uses GPU if available)
result = pf.filter(observations)

print(f"Mean ESS: {result.metadata['mean_ess']}")
```

### Using Flow Filters

```python
from src.filters.particle.edh_flow import ExactDaumHuangFlow
from src.filters.particle.ledh_invertible import LEDHParticleFlowFilter

# EDH Flow (no resampling, equal weights)
edh = ExactDaumHuangFlow(
    model,
    n_particles=1000,
    n_lambda_steps=100,
    integration_method='euler',
    n_threads=4
)
means, covs = edh.filter(observations)

# LEDH Invertible (with resampling, EKF/UKF prediction)
ledh = LEDHParticleFlowFilter(
    model,
    n_particles=1000,
    n_lambda_steps=100,
    filter_type='ekf',
    resample_threshold=0.5
)
means, covs = ledh.filter(observations)
print(f"Mean ESS: {np.mean(ledh.ess_history)}")
```

## Project Structure

```
code/
├── src/
│   ├── core/              # Core abstractions
│   │   ├── types.py       # FilterResult, FilterState
│   │   ├── model_base.py  # StateSpaceModel ABC
│   │   └── filter_base.py # Filter ABC
│   ├── models/            # State-space models
│   │   ├── linear_gaussian.py
│   │   ├── stochastic_volatility.py
│   │   ├── range_bearing.py
│   │   └── utils.py       # generate_data
│   ├── filters/
│   │   ├── kalman/        # NumPy implementations
│   │   │   ├── kalman.py
│   │   │   ├── extended_kalman.py
│   │   │   └── unscented_kalman.py
│   │   └── particle/      # Particle filter implementations
│   │       ├── bootstrap_pf.py       # TensorFlow bootstrap PF
│   │       ├── flow_base.py          # Base class for flow filters
│   │       ├── edh_flow.py           # Exact Daum-Huang flow
│   │       ├── ledh_flow.py          # Local EDH flow
│   │       ├── edh_invertible.py     # EDH invertible flow
│   │       ├── ledh_invertible.py    # LEDH invertible flow
│   │       └── kernel_flow.py        # Kernel mapping PF
│   ├── resampling/        # TensorFlow resampling
│   │   ├── systematic.py
│   │   ├── stratified.py
│   │   └── utils.py
│   ├── utils/             # Numerical stability
│   │   ├── linalg.py      # safe_cholesky, log_det
│   │   ├── distributions.py # log_gaussian_prob
│   │   └── ode_solvers.py # euler_step, rk4_step
│   └── experiments/       # Hydra experiment runner
│       ├── run_experiment.py   # Main experiment script
│       └── visualization.py    # Plotting utilities
├── configs/               # Hydra configurations
│   ├── config.yaml        # Main config
│   ├── model/             # Model configs
│   ├── filter/            # Filter configs (9 filters)
│   └── experiment/        # Pre-configured experiments
├── tests/                 # Smoke tests
├── example.py             # Quick usage examples
├── README.md              # This file
└── EXPERIMENTS_GUIDE.md   # Detailed experiment guide
```

## Design Principles

1. **Numerical Stability**: All matrix operations use safe wrappers (safe_cholesky, log-sum-exp)
2. **Modularity**: Models and filters are decoupled via abstract interfaces
3. **Performance**: TensorFlow particle filters use tf.scan for efficient GPU execution
4. **Reproducibility**: Stateless random number generation, Hydra config tracking
5. **Type Safety**: FilterResult dataclasses ensure consistent return types

## GPU Acceleration

Particle filters automatically use GPU if available:

```python
import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Enable memory growth (recommended)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

Expected speedups with GPU:
- 1000 particles: ~5-10x faster
- 5000 particles: ~15-20x faster
- 10000 particles: ~20-30x faster

## Notes

- This implementation covers **all Part 1** requirements including flow filters
- `part1/` directory remains completely unchanged and intact
- All components are tested with smoke tests in `tests/`
- Flow filters use pure NumPy with optional threading for parallelization
- Bootstrap PF uses TensorFlow for GPU acceleration

## References

**Kalman Filters:**
- Kalman Filter: Optimal for linear-Gaussian systems
- EKF: First-order linearization via Jacobians
- UKF: Unscented transform with sigma points

**Particle Filters:**
- Bootstrap PF: Sequential importance resampling

**Flow Filters:**
- EDH Flow: Daum & Huang (2008) - Exact particle flow without resampling
- LEDH Flow: Local EDH using kernel density estimation
- EDH/LEDH Invertible: Flow with EKF/UKF linearization and Jacobian tracking
- Kernel Flow: Pulido & van Leeuwen (2019) - Kernel mapping particle filter

## License

This is an academic implementation for coursework.
