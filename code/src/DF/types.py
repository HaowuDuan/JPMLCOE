"""Data types for differentiable filter framework."""

from dataclasses import dataclass, field
from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import tensorflow_probability as tfp


@dataclass
class ParameterSpec:
    """
    Specification for a trainable parameter.
    
    Attributes:
        name: Parameter name (must match model attribute)
        init_value: Initial value for parameter
        constraint: Constraint specification:
            - 'positive': x > 0
            - 'unit': x ∈ (0, 1)
            - (a, b): x ∈ (a, b)
            - None: unconstrained
        prior: Prior distribution (on constrained space)
    """
    name: str
    init_value: float
    constraint: Union[str, Tuple[float, float], None]
    prior: tfp.distributions.Distribution
    
    def __post_init__(self):
        """Validate parameter specification."""
        if self.constraint is not None:
            if isinstance(self.constraint, str):
                if self.constraint not in ['positive', 'unit']:
                    raise ValueError(f"Unknown constraint: {self.constraint}")
            elif isinstance(self.constraint, tuple):
                if len(self.constraint) != 2:
                    raise ValueError(f"Constraint tuple must have 2 elements: {self.constraint}")
                if self.constraint[0] >= self.constraint[1]:
                    raise ValueError(f"Invalid bounds: {self.constraint}")


@dataclass
class DPFResult:
    """
    Result from differentiable filter parameter inference.
    
    Attributes:
        samples: Posterior samples for each parameter, or a single-element MAP
            point estimate for MAP runs, shape (num_samples,).
        summary: Summary statistics (mean, std, quantiles) per parameter
        diagnostics: MCMC diagnostics (ESS, R_hat, acceptance_rate, etc.)
        metadata: Additional information (filter type, model type, etc.)
    """
    samples: Dict[str, np.ndarray]
    summary: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
