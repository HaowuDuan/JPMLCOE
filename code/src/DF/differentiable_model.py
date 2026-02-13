"""Wrapper for models to support parameter updates during inference."""

import tensorflow as tf
from typing import List, Dict, Any
import copy


class DifferentiableModel:
    """
    Wrapper that enables parameter updates on existing models.
    
    Intercepts attribute access and allows dynamic parameter updates
    during HMC sampling while preserving all other model functionality.
    
    Key features:
    - Transparent wrapping: behaves exactly like base model
    - Parameter updates: dynamically modifies trainable parameters
    - State preservation: can restore original parameter values
    - No model modification: works with any StateSpaceModel
    """
    
    def __init__(self, base_model: Any, trainable_params: List[str]):
        """
        Initialize differentiable model wrapper.
        
        Args:
            base_model: The underlying model to wrap
            trainable_params: List of parameter names that will be updated
        """
        # Store base model and trainable parameter names
        object.__setattr__(self, '_base_model', base_model)
        object.__setattr__(self, '_trainable_params', trainable_params)
        object.__setattr__(self, '_original_values', {})
        
        # Save original parameter values for restoration
        for param_name in trainable_params:
            if not hasattr(base_model, param_name):
                raise ValueError(f"Model does not have parameter: {param_name}")
            original_value = getattr(base_model, param_name)
            self._original_values[param_name] = copy.deepcopy(original_value)
    
    def update_parameters(self, param_dict: Dict[str, tf.Tensor]) -> None:
        """
        Update trainable parameters in the wrapped model.
        
        Args:
            param_dict: Dictionary mapping parameter names to new values (TF tensors)
        """
        for param_name, param_value in param_dict.items():
            if param_name not in self._trainable_params:
                raise ValueError(f"Parameter {param_name} is not trainable")
            
            # Convert TF tensor to appropriate type for model
            # Most models store parameters as Python floats or numpy arrays
            if hasattr(param_value, 'numpy'):
                value = float(param_value.numpy())
            else:
                value = float(param_value)
            
            setattr(self._base_model, param_name, value)
    
    def restore_parameters(self) -> None:
        """Restore original parameter values."""
        for param_name, original_value in self._original_values.items():
            setattr(self._base_model, param_name, copy.deepcopy(original_value))
    
    def get_current_parameters(self) -> Dict[str, float]:
        """
        Get current values of trainable parameters.
        
        Returns:
            Dictionary mapping parameter names to current values
        """
        return {
            param_name: getattr(self._base_model, param_name)
            for param_name in self._trainable_params
        }
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate all other attribute access to base model.
        
        This makes the wrapper transparent - it behaves exactly like the base model.
        """
        # Avoid infinite recursion by accessing __dict__ directly
        if name in ['_base_model', '_trainable_params', '_original_values']:
            return object.__getattribute__(self, name)
        
        base_model = object.__getattribute__(self, '_base_model')
        return getattr(base_model, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Delegate attribute setting to base model (except internal attributes).
        """
        if name in ['_base_model', '_trainable_params', '_original_values']:
            object.__setattr__(self, name, value)
        else:
            base_model = object.__getattribute__(self, '_base_model')
            setattr(base_model, name, value)
    
    def __repr__(self) -> str:
        """String representation showing wrapper and base model."""
        base_model = object.__getattribute__(self, '_base_model')
        trainable_params = object.__getattribute__(self, '_trainable_params')
        return (f"DifferentiableModel(\n"
                f"  base_model={base_model.__class__.__name__},\n"
                f"  trainable_params={trainable_params}\n"
                f")")


