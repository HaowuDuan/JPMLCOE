"""Abstract base class for state-space models."""

from abc import ABC, abstractmethod
import numpy as np


class StateSpaceModel(ABC):
    """
    Abstract base class for state-space models.

    All models must implement:
    1. Properties: state_dim, obs_dim
    2. Sampling methods (for data generation & particle filters)
    3. Deterministic methods (for Kalman/EKF/UKF)
    4. Log-probability methods (for particle filters)
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state space."""
        pass

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of observation space."""
        pass

    # Sampling methods (for data generation & particle filters)

    @abstractmethod
    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from initial state distribution p(x_0).

        Args:
            rng: Random number generator

        Returns:
            Initial state of shape (state_dim,)
        """
        pass

    @abstractmethod
    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample next state from transition distribution p(x_{t+1} | x_t).

        Args:
            x: Current state of shape (state_dim,)
            rng: Random number generator

        Returns:
            Next state of shape (state_dim,)
        """
        pass

    @abstractmethod
    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample observation from observation distribution p(y_t | x_t).

        Args:
            x: Current state of shape (state_dim,)
            rng: Random number generator

        Returns:
            Observation of shape (obs_dim,)
        """
        pass

    # Deterministic methods (for Kalman/EKF/UKF)

    @abstractmethod
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Mean of state transition: E[x_{t+1} | x_t].

        Args:
            x: Current state of shape (state_dim,)

        Returns:
            Mean of shape (state_dim,)
        """
        pass

    @abstractmethod
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """
        Covariance of state transition: Cov[x_{t+1} | x_t].

        Args:
            x: Current state of shape (state_dim,)

        Returns:
            Covariance of shape (state_dim, state_dim)
        """
        pass

    @abstractmethod
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of state transition function: ∂f/∂x.

        Args:
            x: Current state of shape (state_dim,)

        Returns:
            Jacobian of shape (state_dim, state_dim)
        """
        pass

    @abstractmethod
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Mean of observation: E[y_t | x_t].

        Args:
            x: Current state of shape (state_dim,)

        Returns:
            Mean of shape (obs_dim,)
        """
        pass

    @abstractmethod
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """
        Covariance of observation: Cov[y_t | x_t].

        Args:
            x: Current state of shape (state_dim,)

        Returns:
            Covariance of shape (obs_dim, obs_dim)
        """
        pass

    @abstractmethod
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function: ∂h/∂x.

        Args:
            x: Current state of shape (state_dim,)

        Returns:
            Jacobian of shape (obs_dim, state_dim)
        """
        pass

    # For particle filters

    @abstractmethod
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y_t | x_t).

        Args:
            y: Observation of shape (obs_dim,)
            x: State of shape (state_dim,)

        Returns:
            Log probability (scalar)
        """
        pass
