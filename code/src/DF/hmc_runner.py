"""Main HMC runner for parameter inference with differentiable filters."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Any, Dict, Optional, Type
import warnings

from .types import ParameterSpec, DPFResult
from .parameter_handler import ParameterHandler
from .differentiable_model import DifferentiableModel
from ..core.filter_base import Filter


class DPFRunner:
    """
    Differentiable Particle Filter runner for parameter inference via HMC.
    
    Main orchestrator that:
    1. Wraps a model with DifferentiableModel
    2. Manages parameter transformations via ParameterHandler
    3. Defines target log posterior using any filter
    4. Runs HMC sampling with adaptive step size
    5. Computes diagnostics and summary statistics
    """
    
    def __init__(
        self,
        base_model: Any,
        filter_class: Type[Filter],
        filter_kwargs: Dict[str, Any],
        param_specs: Dict[str, ParameterSpec]
    ):
        """
        Initialize DPF runner.
        
        Args:
            base_model: State-space model instance
            filter_class: Filter class (e.g., ExactDaumHuangFlow, ExtendedKalmanFilter)
            filter_kwargs: Keyword arguments for filter initialization
            param_specs: Dictionary of parameter specifications
        """
        self.base_model = base_model
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        
        # Create differentiable model wrapper
        trainable_param_names = list(param_specs.keys())
        self.diff_model = DifferentiableModel(base_model, trainable_param_names)
        
        # Create parameter handler
        self.param_handler = ParameterHandler(param_specs)
        
        # Storage for observations (set during run_hmc)
        self.observations = None
    
    @tf.function(reduce_retracing=True)
    def _negative_log_posterior(self, unconstrained_params: tf.Tensor) -> tf.Tensor:
        """
        Compute negative log posterior: -log p(θ|y).
        
        This is the objective function for HMC:
        -log p(θ|y) = -log p(y|θ) - log p(θ) + const
        
        Args:
            unconstrained_params: Parameters in unconstrained space, shape (num_params,)
        
        Returns:
            Negative log posterior (scalar)
        """
        # 1. Transform to constrained space
        constrained_params = self.param_handler.constrain(unconstrained_params)
        
        # 2. Update model with new parameters (in Python space)
        # Note: This breaks out of tf.function but is necessary for model updates
        # We use py_function to handle this
        def update_and_run_filter():
            # Update model parameters
            self.diff_model.update_parameters(constrained_params)
            
            # Create filter with updated model
            filter_obj = self.filter_class(self.diff_model, **self.filter_kwargs)
            
            # Run filter on observations
            result = filter_obj.filter(self.observations)
            
            return result.log_likelihood
        
        # Execute filter in eager mode (required for model parameter updates)
        log_likelihood = tf.py_function(
            update_and_run_filter,
            [],
            tf.float32
        )
        log_likelihood.set_shape([])  # Set scalar shape
        
        # 3. Compute log prior with Jacobian adjustment
        log_prior = self.param_handler.log_prior(constrained_params)
        
        # 4. Return negative log posterior
        return -(log_likelihood + log_prior)
    
    def run_hmc(
        self,
        observations: np.ndarray,
        num_samples: int = 1000,
        num_burnin: int = 500,
        step_size: float = 0.01,
        num_leapfrog_steps: int = 10,
        adaptation_rate: float = 0.8,
        target_accept_prob: float = 0.75,
        seed: Optional[int] = None
    ) -> DPFResult:
        """
        Run HMC to sample from posterior p(θ|y).
        
        Args:
            observations: Observation sequence, shape (T, obs_dim)
            num_samples: Number of posterior samples to collect
            num_burnin: Number of burn-in iterations
            step_size: Initial HMC step size
            num_leapfrog_steps: Number of leapfrog steps per HMC iteration
            adaptation_rate: Fraction of burn-in for step size adaptation
            target_accept_prob: Target acceptance probability for adaptation
            seed: Random seed for reproducibility
        
        Returns:
            DPFResult with posterior samples, summary statistics, and diagnostics
        """
        # Store observations for use in negative_log_posterior
        self.observations = observations
        
        # Define target log probability (positive log posterior)
        @tf.function
        def target_log_prob_fn(unconstrained_params):
            return -self._negative_log_posterior(unconstrained_params)
        
        # Initialize HMC kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps
        )
        
        # Add adaptive step size tuning
        num_adaptation_steps = int(adaptation_rate * num_burnin)
        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc_kernel,
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob
        )
        
        # Set random seed if provided
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Define trace function to capture diagnostics
        def trace_fn(_, pkr):
            return {
                'is_accepted': pkr.inner_results.is_accepted,
                'step_size': pkr.new_step_size
            }
        
        # Run MCMC sampling
        print(f"Running HMC with {num_samples} samples and {num_burnin} burn-in iterations...")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            
            samples_unconstrained, trace_results = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin,
                current_state=self.param_handler.unconstrained_init,
                kernel=adaptive_kernel,
                trace_fn=trace_fn
            )
        
        # Transform samples to constrained space
        print("Transforming samples to constrained space...")
        samples_constrained = self._transform_samples(samples_unconstrained)
        
        # Compute diagnostics
        print("Computing diagnostics...")
        diagnostics = self._compute_diagnostics(
            samples_constrained,
            trace_results['is_accepted'],
            trace_results['step_size']
        )
        
        # Compute summary statistics
        print("Computing summary statistics...")
        summary = self._compute_summary(samples_constrained)
        
        # Restore original model parameters
        self.diff_model.restore_parameters()
        
        print("HMC sampling complete!")
        
        return DPFResult(
            samples=samples_constrained,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                'model_type': self.base_model.__class__.__name__,
                'filter_type': self.filter_class.__name__,
                'num_samples': num_samples,
                'num_burnin': num_burnin,
                'num_observations': len(observations)
            }
        )
    
    def _transform_samples(self, samples_unconstrained: tf.Tensor) -> Dict[str, np.ndarray]:
        """
        Transform unconstrained samples to constrained space.
        
        Args:
            samples_unconstrained: Tensor of shape (num_samples, num_params)
        
        Returns:
            Dictionary mapping parameter names to arrays of shape (num_samples,)
        """
        num_samples = samples_unconstrained.shape[0]
        samples_constrained = {name: [] for name in self.param_handler.param_names}
        
        # Transform each sample
        for i in range(num_samples):
            constrained = self.param_handler.constrain(samples_unconstrained[i])
            for name, value in constrained.items():
                samples_constrained[name].append(float(value.numpy()))
        
        # Convert to numpy arrays
        return {name: np.array(values) for name, values in samples_constrained.items()}
    
    def _compute_diagnostics(
        self,
        samples: Dict[str, np.ndarray],
        is_accepted: tf.Tensor,
        step_sizes: tf.Tensor
    ) -> Dict[str, Any]:
        """
        Compute MCMC diagnostics.
        
        Args:
            samples: Dictionary of posterior samples
            is_accepted: Boolean tensor of acceptance indicators
            step_sizes: Tensor of step sizes over iterations
        
        Returns:
            Dictionary with diagnostics (ESS, R_hat, acceptance_rate, etc.)
        """
        diagnostics = {}
        
        # Acceptance rate
        acceptance_rate = float(tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy())
        diagnostics['acceptance_rate'] = acceptance_rate
        
        # Final step size
        final_step_size = float(step_sizes[-1].numpy())
        diagnostics['final_step_size'] = final_step_size
        
        # Effective sample size (ESS) per parameter
        ess_dict = {}
        for name, param_samples in samples.items():
            # Reshape for tfp: (num_chains=1, num_samples)
            param_samples_tf = tf.constant(param_samples[np.newaxis, :], dtype=tf.float32)
            ess = tfp.mcmc.effective_sample_size(param_samples_tf)
            ess_dict[name] = float(ess.numpy()[0])
        diagnostics['ess'] = ess_dict
        
        # R-hat (potential scale reduction) - requires multiple chains
        # For single chain, we split into two halves
        rhat_dict = {}
        for name, param_samples in samples.items():
            if len(param_samples) >= 4:  # Need at least 2 samples per chain
                mid = len(param_samples) // 2
                chain1 = param_samples[:mid]
                chain2 = param_samples[mid:2*mid]
                # Stack as (2 chains, n samples)
                chains = tf.constant([chain1, chain2], dtype=tf.float32)
                rhat = tfp.mcmc.potential_scale_reduction(chains)
                rhat_dict[name] = float(rhat.numpy())
            else:
                rhat_dict[name] = np.nan
        diagnostics['rhat'] = rhat_dict
        
        return diagnostics
    
    def _compute_summary(self, samples: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for posterior samples.
        
        Args:
            samples: Dictionary of posterior samples
        
        Returns:
            Dictionary mapping parameter names to summary statistics
        """
        summary = {}
        
        for name, param_samples in samples.items():
            summary[name] = {
                'mean': float(np.mean(param_samples)),
                'std': float(np.std(param_samples)),
                'median': float(np.median(param_samples)),
                'q5': float(np.percentile(param_samples, 5)),
                'q25': float(np.percentile(param_samples, 25)),
                'q75': float(np.percentile(param_samples, 75)),
                'q95': float(np.percentile(param_samples, 95)),
                'min': float(np.min(param_samples)),
                'max': float(np.max(param_samples))
            }
        
        return summary


