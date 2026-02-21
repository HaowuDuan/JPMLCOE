"""Parameter transformation and bijector management."""

import tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict, List
import numpy as np

from .types import ParameterSpec


class ParameterHandler:
    """
    Manages parameter transformations between constrained and unconstrained spaces.
    
    Uses bijectors to transform parameters for HMC:
    - HMC operates in unconstrained space (ℝ^n)
    - Bijectors map to constrained space (e.g., (0,1), (0,∞))
    - Handles Jacobian adjustments for log probability
    """
    
    def __init__(self, param_specs: Dict[str, ParameterSpec], dtype=None):
        """
        Initialize parameter handler.

        Args:
            param_specs: Dictionary mapping parameter names to specifications
            dtype: TensorFlow dtype for parameter tensors (default: tf.float32)
        """
        self.dtype = dtype if dtype is not None else tf.float32
        self.param_specs = param_specs
        self.param_names = list(param_specs.keys())
        self.num_params = len(self.param_names)
        
        # Create bijectors for each parameter
        self.bijectors = self._create_bijectors()
        
        # Initialize unconstrained parameters
        self.unconstrained_init = self._initialize_unconstrained()
    
    def _create_bijectors(self) -> Dict[str, tfp.bijectors.Bijector]:
        """
        Create appropriate bijector for each parameter based on constraint.
        
        Returns:
            Dictionary mapping parameter names to bijectors
        """
        bijectors = {}
        
        for name, spec in self.param_specs.items():
            constraint = spec.constraint
            
            if constraint is None:
                # Unconstrained: identity bijector
                bijectors[name] = tfp.bijectors.Identity()
            
            elif constraint == 'positive':
                # x > 0: softplus bijector (Exp overflows to Inf for large
                # unconstrained values during HMC leapfrog; Softplus grows
                # linearly instead. Jacobian correction keeps it exact.)
                bijectors[name] = tfp.bijectors.Softplus()
            
            elif constraint == 'unit':
                # x ∈ (0, 1): sigmoid bijector
                bijectors[name] = tfp.bijectors.Sigmoid()
            
            elif isinstance(constraint, tuple):
                # x ∈ (a, b): scaled sigmoid bijector
                a, b = constraint
                bijectors[name] = tfp.bijectors.Sigmoid(low=a, high=b)
            
            else:
                raise ValueError(f"Unknown constraint for {name}: {constraint}")
        
        return bijectors
    
    def _initialize_unconstrained(self) -> tf.Tensor:
        """
        Initialize unconstrained parameters from constrained initial values.
        
        Returns:
            Tensor of shape (num_params,) with unconstrained initial values
        """
        unconstrained_values = []
        
        for name in self.param_names:
            spec = self.param_specs[name]
            bijector = self.bijectors[name]
            
            # Transform constrained → unconstrained via inverse bijector
            constrained = tf.constant(spec.init_value, dtype=self.dtype)
            unconstrained = bijector.inverse(constrained)
            unconstrained_values.append(unconstrained)

        return tf.stack(unconstrained_values)
    
    def constrain(self, unconstrained_params: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Transform unconstrained parameters to constrained space.
        
        Args:
            unconstrained_params: Tensor of shape (num_params,) in unconstrained space
        
        Returns:
            Dictionary mapping parameter names to constrained values
        """
        constrained = {}
        
        for i, name in enumerate(self.param_names):
            bijector = self.bijectors[name]
            unconstrained_value = unconstrained_params[i]
            constrained[name] = bijector.forward(unconstrained_value)
        
        return constrained
    
    def unconstrain(self, constrained_params: Dict[str, float]) -> tf.Tensor:
        """
        Transform constrained parameters to unconstrained space.
        
        Args:
            constrained_params: Dictionary mapping parameter names to constrained values
        
        Returns:
            Tensor of shape (num_params,) in unconstrained space
        """
        unconstrained_values = []
        
        for name in self.param_names:
            bijector = self.bijectors[name]
            constrained = tf.constant(constrained_params[name], dtype=self.dtype)
            unconstrained = bijector.inverse(constrained)
            unconstrained_values.append(unconstrained)
        
        return tf.stack(unconstrained_values)
    
    def log_prior(self, constrained_params: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Compute log prior with Jacobian adjustment for change of variables.
        
        log p(θ_unconstrained) = log p(θ_constrained) + log |det J|
        
        where J is the Jacobian of the transformation.
        
        Args:
            constrained_params: Dictionary of parameters in constrained space
        
        Returns:
            Log prior probability (scalar)
        """
        log_prob = tf.constant(0.0, dtype=self.dtype)
        
        for name in self.param_names:
            spec = self.param_specs[name]
            bijector = self.bijectors[name]
            constrained_value = constrained_params[name]
            
            # Log probability in constrained space
            # Cast to prior's dtype if needed (prior may be float32 from Hydra config)
            prior_dtype = spec.prior.dtype
            value_for_prior = tf.cast(constrained_value, prior_dtype)
            log_prob_constrained = tf.cast(spec.prior.log_prob(value_for_prior), self.dtype)
            
            # Jacobian adjustment (forward log determinant)
            # This accounts for change of variables from unconstrained to constrained
            log_det_jacobian = bijector.forward_log_det_jacobian(
                bijector.inverse(constrained_value),
                event_ndims=0
            )
            
            # Total log probability in unconstrained space
            log_prob += log_prob_constrained + log_det_jacobian
        
        return log_prob
    
    def get_initial_constrained(self) -> Dict[str, float]:
        """
        Get initial parameter values in constrained space.
        
        Returns:
            Dictionary mapping parameter names to initial constrained values
        """
        return {name: spec.init_value for name, spec in self.param_specs.items()}


