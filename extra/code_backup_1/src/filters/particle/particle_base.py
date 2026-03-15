"""Base class for particle filters with diagnostic tracking."""

import numpy as np
from abc import abstractmethod
from typing import Dict, Any

from ...core.filter_base import Filter


class ParticleFilterBase(Filter):
    """
    Abstract base class for particle filters with diagnostic tracking.

    Provides common functionality for all particle filter implementations:
    - Diagnostic tracking (ESS, weights, unique particles, resampling events)
    - Common interface for particle filter operations

    Subclasses must implement:
    - filter() method
    - _effective_sample_size() method (or use default implementation)
    """

    def __init__(self, model, n_particles: int = 1000, resample_threshold: float = 0.5):
        """
        Initialize base particle filter.

        Args:
            model: State-space model instance
            n_particles: Number of particles
            resample_threshold: Resample when ESS/N < threshold
        """
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold

        # Particle state
        self.particles = None
        self.weights = None

        # Filtering results
        self.means = []
        self.covs = []
        self.log_likelihoods = []

        # Diagnostic tracking
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

    def reset(self):
        """Reset filter state and clear history."""
        self.particles = None
        self.weights = None
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

    def _effective_sample_size(self, weights: np.ndarray = None) -> float:
        """
        Compute effective sample size (ESS).

        ESS = 1 / Σ(w_i²) where w_i are normalized weights.

        Args:
            weights: Normalized weights. If None, uses self.weights.

        Returns:
            Effective sample size (between 1 and N)
        """
        if weights is None:
            weights = self.weights

        if weights is None:
            return self.n_particles

        return 1.0 / np.sum(weights ** 2)

    def predict(self):
        """Predict step (not used in batch particle filters)."""
        raise NotImplementedError("Use filter() method instead")

    def update(self, observation):
        """Update step (not used in batch particle filters)."""
        raise NotImplementedError("Use filter() method instead")

    @abstractmethod
    def filter(self, observations: np.ndarray):
        """
        Run particle filter on observation sequence.

        Must be implemented by subclasses.
        Subclasses should manually track diagnostics by appending to:
        - self.ess_history
        - self.weights_history (before resampling to capture actual weights)
        - self.resampled_at
        - self.n_unique_particles
        """
        pass
