"""Example usage of the Differentiable Filter Framework."""

import numpy as np
import tensorflow_probability as tfp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.DF import DPFRunner, ParameterSpec
from src.models.stochastic_volatility import StochasticVolatilityModel
from src.filters.particle.edh_flow import ExactDaumHuangFlow
from src.models.utils import generate_data


def example_stochastic_volatility():
    """
    Example: Infer parameters of Stochastic Volatility model.
    
    1. Generate synthetic data from true model
    2. Run HMC to infer parameters
    3. Compare inferred parameters with true values
    """
    print("="*60)
    print("Stochastic Volatility Parameter Inference Example")
    print("="*60)
    
    # Step 1: Generate synthetic data from true model
    print("\n1. Generating synthetic data...")
    true_model = StochasticVolatilityModel(
        alpha=0.91,  # True parameter
        sigma=1.0,   # True parameter
        beta=0.5     # True parameter (fixed, not inferred)
    )
    
    rng = np.random.default_rng(42)
    T = 100
    states, observations = generate_data(true_model, T=T, rng=rng, observe_initial=True)
    
    print(f"   Generated {T} observations")
    print(f"   True parameters: alpha=0.91, sigma=1.0, beta=0.5")
    
    # Step 2: Set up inference model with initial guesses
    print("\n2. Setting up inference model...")
    inference_model = StochasticVolatilityModel(
        alpha=0.85,  # Initial guess (wrong!)
        sigma=0.8,   # Initial guess (wrong!)
        beta=0.5     # Fixed (not inferred)
    )
    
    # Step 3: Define parameter specifications
    param_specs = {
        'alpha': ParameterSpec(
            name='alpha',
            init_value=0.85,
            constraint=(0, 1),  # alpha must be in (0, 1)
            prior=tfp.distributions.Beta(9.0, 1.0)  # Peaked near 0.9
        ),
        'sigma': ParameterSpec(
            name='sigma',
            init_value=0.8,
            constraint='positive',  # sigma > 0
            prior=tfp.distributions.LogNormal(0.0, 0.5)
        )
    }
    
    print("   Parameters to infer: alpha, sigma")
    print("   Initial guesses: alpha=0.85, sigma=0.8")
    
    # Step 4: Create DPF runner
    print("\n3. Creating DPF runner with EDH Flow filter...")
    runner = DPFRunner(
        base_model=inference_model,
        filter_class=ExactDaumHuangFlow,
        filter_kwargs={
            'n_particles': 50,      # Reduced for faster example
            'n_lambda_steps': 20,   # Reduced for faster example
            'integration_method': 'euler',
            'filter_type': 'ekf'
        },
        param_specs=param_specs
    )
    
    # Step 5: Run HMC inference
    print("\n4. Running HMC inference...")
    print("   (This may take a few minutes...)")
    
    result = runner.run_hmc(
        observations=observations,
        num_samples=200,      # Reduced for faster example
        num_burnin=100,       # Reduced for faster example
        step_size=0.01,
        num_leapfrog_steps=5,
        seed=42
    )
    
    # Step 6: Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nPosterior Summary:")
    print("-" * 60)
    for param_name, stats in result.summary.items():
        true_value = 0.91 if param_name == 'alpha' else 1.0
        print(f"\n{param_name}:")
        print(f"  True value:      {true_value:.3f}")
        print(f"  Posterior mean:  {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Posterior median: {stats['median']:.3f}")
        print(f"  95% CI:          [{stats['q5']:.3f}, {stats['q95']:.3f}]")
        
        # Check if true value is in credible interval
        in_ci = stats['q5'] <= true_value <= stats['q95']
        print(f"  True value in 90% CI: {'✓' if in_ci else '✗'}")
    
    print("\nDiagnostics:")
    print("-" * 60)
    print(f"Acceptance rate:  {result.diagnostics['acceptance_rate']:.1%}")
    print(f"Final step size:  {result.diagnostics['final_step_size']:.4f}")
    
    print("\nEffective Sample Size (ESS):")
    for param_name, ess in result.diagnostics['ess'].items():
        print(f"  {param_name}: {ess:.1f}")
    
    print("\nR-hat (convergence diagnostic, should be < 1.1):")
    for param_name, rhat in result.diagnostics['rhat'].items():
        status = "✓" if rhat < 1.1 else "⚠"
        print(f"  {param_name}: {rhat:.3f} {status}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    
    return result


if __name__ == '__main__':
    # Run example
    result = example_stochastic_volatility()
    
    # Optional: Save results
    # import pickle
    # with open('dpf_result.pkl', 'wb') as f:
    #     pickle.dump(result, f)
    # print("\nResults saved to dpf_result.pkl")


