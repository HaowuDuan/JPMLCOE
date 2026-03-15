"""Data types for filter results."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class FilterState:
    """State of a filter at a single timestep."""
    mean: np.ndarray  # (state_dim,)
    cov: np.ndarray   # (state_dim, state_dim)
    particles: Optional[np.ndarray] = None  # For particle filters: (N, state_dim)
    weights: Optional[np.ndarray] = None    # For particle filters: (N,)


@dataclass
class FilterResult:
    """Complete filtering result over a sequence."""
    means: np.ndarray           # (T, state_dim)
    covs: np.ndarray            # (T, state_dim, state_dim)
    log_likelihood: float       # Total log p(y_{1:T})
    metadata: Dict[str, Any] = field(default_factory=dict)  # Filter-specific metadata (scalars, not arrays)
    
    # Optional particle filter diagnostics (saved as separate .npy files, not in metadata)
    log_likelihoods: Optional[np.ndarray] = None  # (T,) per-timestep log-likelihoods
    ess: Optional[np.ndarray] = None              # (T,) effective sample sizes
    weights_history: Optional[np.ndarray] = None  # (T, N) particle weights over time
    resampled_at: Optional[list] = None           # List of timesteps where resampling occurred
    n_unique: Optional[np.ndarray] = None         # Number of unique particles after each resampling
