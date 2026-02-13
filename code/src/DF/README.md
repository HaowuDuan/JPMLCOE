# Differentiable Filter Framework (DPF)

Parameter inference for state-space models using HMC and any filter.

## Quick Start

```python
from src.DF import DPFRunner, ParameterSpec
from src.models.stochastic_volatility import StochasticVolatilityModel
from src.filters.particle.edh_flow import ExactDaumHuangFlow
import tensorflow_probability as tfp

# 1. Create model
model = StochasticVolatilityModel(alpha=0.85, sigma=0.8, beta=0.5)

# 2. Specify parameters to infer
param_specs = {
    'alpha': ParameterSpec(
        name='alpha',
        init_value=0.85,
        constraint=(0, 1),
        prior=tfp.distributions.Beta(9.0, 1.0)
    ),
    'sigma': ParameterSpec(
        name='sigma',
        init_value=0.8,
        constraint='positive',
        prior=tfp.distributions.LogNormal(0.0, 0.5)
    )
}

# 3. Create runner
runner = DPFRunner(
    base_model=model,
    filter_class=ExactDaumHuangFlow,
    filter_kwargs={'n_particles': 100, 'n_lambda_steps': 50},
    param_specs=param_specs
)

# 4. Run HMC
result = runner.run_hmc(observations, num_samples=1000, num_burnin=500)

# 5. Access results
print(result.summary)  # Posterior means, stds, quantiles
print(result.diagnostics)  # ESS, R-hat, acceptance rate
```

## Supported Constraints

- `'positive'`: x > 0 (Exp bijector)
- `'unit'`: x ∈ (0, 1) (Sigmoid bijector)
- `(a, b)`: x ∈ (a, b) (Scaled sigmoid bijector)
- `None`: unconstrained (Identity bijector)

## Run Example

```bash
cd code
python -m src.DF.example_usage
```

## Use with Hydra

See `code/configs/dpf/stochastic_volatility.yaml` for configuration template.

