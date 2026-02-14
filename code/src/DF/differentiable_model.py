"""Wrapper for models to support differentiable parameter updates during HMC."""

import tensorflow as tf
from typing import List, Dict, Any
import copy


class DifferentiableModel:
    """
    Wrapper that enables differentiable parameter updates on existing models.

    During HMC, parameter values are TF tensors flowing through the computation
    graph. This wrapper uses setattr to set them directly on the base model,
    preserving the gradient chain for tf.GradientTape.

    Key design decision: NO tf.Variable.assign() — that would sever the gradient
    chain between the proposed parameters and the log-likelihood. Instead, we set
    model attributes directly to the constrained tensor values via setattr.
    The model's property-based methods (e.g. Q property) then use these tensors
    in their computation, maintaining full differentiability.
    """

    def __init__(self, base_model: Any, trainable_params: List[str]):
        """
        Initialize differentiable model wrapper.

        Args:
            base_model: The underlying model to wrap
            trainable_params: List of parameter names that will be updated
        """
        object.__setattr__(self, '_base_model', base_model)
        object.__setattr__(self, '_trainable_params', trainable_params)
        object.__setattr__(self, '_original_values', {})

        for param_name in trainable_params:
            if not hasattr(base_model, param_name):
                raise ValueError(f"Model does not have parameter: {param_name}")
            original_value = getattr(base_model, param_name)
            self._original_values[param_name] = copy.deepcopy(original_value)

    def update_parameters(self, param_dict: Dict[str, tf.Tensor]) -> None:
        """
        Update trainable parameters via setattr — preserves gradient chain.

        The constrained tensor is set directly as a model attribute. When model
        methods (e.g. state_transition_cov) read this attribute, the computation
        stays connected to the original unconstrained_params through the bijector.

        Args:
            param_dict: Dictionary mapping parameter names to TF tensor values
        """
        for param_name, param_value in param_dict.items():
            if param_name not in self._trainable_params:
                raise ValueError(f"Parameter {param_name} is not trainable")
            setattr(self._base_model, param_name, param_value)

    def restore_parameters(self) -> None:
        """Restore original Python values (after HMC completes)."""
        for param_name, original_value in self._original_values.items():
            setattr(self._base_model, param_name, copy.deepcopy(original_value))

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current values of trainable parameters."""
        return {
            name: getattr(self._base_model, name)
            for name in self._trainable_params
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base model (transparent wrapper)."""
        if name in ('_base_model', '_trainable_params', '_original_values'):
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, '_base_model'), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to base model (except internal attrs)."""
        if name in ('_base_model', '_trainable_params', '_original_values'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_base_model'), name, value)

    def __repr__(self) -> str:
        base = object.__getattribute__(self, '_base_model')
        params = object.__getattribute__(self, '_trainable_params')
        return f"DifferentiableModel(base={base.__class__.__name__}, params={params})"
