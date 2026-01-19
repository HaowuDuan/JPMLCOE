"""Abstract base class for filters."""

from abc import ABC, abstractmethod
import numpy as np
from .types import FilterResult


class Filter(ABC):
    """Abstract base class for all filters (Kalman and Particle)."""

    @abstractmethod
    def reset(self):
        """
        Reset filter to initial state and clear history.

        Initial state should be defined in __init__.
        """
        pass

    @abstractmethod
    def predict(self):
        """Prediction step: propagate state forward through dynamics."""
        pass

    @abstractmethod
    def update(self, observation: np.ndarray):
        """
        Update step: incorporate measurement.

        Args:
            observation: Observation of shape (obs_dim,)
        """
        pass

    @abstractmethod
    def filter(self, observations: np.ndarray) -> FilterResult:
        """
        Run filter on observation sequence.

        Args:
            observations: Observation sequence of shape (T, obs_dim)

        Returns:
            FilterResult with means, covariances, log-likelihood, and metadata
        """
        pass
