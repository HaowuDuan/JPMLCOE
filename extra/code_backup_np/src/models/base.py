"""Base classes for state-space models."""
from abc import ABC, abstractmethod
import numpy as np


class StateSpaceModel(ABC):
    """
    These are common methods needed for all filters:
    1, Get the dimension of the ground truth states
    2, Get the dimention of the observed states
    3, Sample from the initial condition 
    4, Sample from the evolution model to get the next state
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
    
    @abstractmethod
    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from initial state distribution.
        
        Args:
            rng: Random number generator
            
        Returns:
            Initial state of shape (state_dim,)
        """
        pass
    
    @abstractmethod
    def sample_dynamics(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample next state given current state.
        
        Args:
            x: Current state of shape (state_dim,)
            rng: Random number generator for the noise
            
        Returns:
            Next state of shape (state_dim,)
        """
        pass


    @abstractmethod
    def sample_observation(self, x, rng) -> np.ndarray:
        """Needed for data generation in ALL experiments. Run in parallel with the evolution model"""
        pass

    # Batch operations for optimized particle filtering

    def state_transition_mean_batch(self, particles: np.ndarray) -> np.ndarray:
        """
        Compute transition means for batch of particles.

        Args:
            particles: Particles of shape (N, state_dim)

        Returns:
            Means of shape (N, state_dim)

        Default: Loop over particles (override for efficiency)
        """
        return np.array([self.state_transition_mean(x) for x in particles])

    def state_transition_cov_batch(self, particles: np.ndarray) -> np.ndarray:
        """
        Compute transition covariances for batch of particles.

        Args:
            particles: Particles of shape (N, state_dim)

        Returns:
            - If Q is state-independent: returns single cov of shape (state_dim, state_dim)
            - If Q is state-dependent: returns (N, state_dim, state_dim)

        Default: Returns single Q (assumes state-independent)
        """
        # Default assumes Q is constant - just return for first particle
        return self.state_transition_cov(particles[0])

    def log_observation_prob_batch(self, observation: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """
        Compute log observation probabilities for batch of particles.

        Args:
            observation: Single observation of shape (obs_dim,)
            particles: Particles of shape (N, state_dim)

        Returns:
            Log probabilities of shape (N,)

        Default: Loop over particles (override for efficiency)
        """
        return np.array([self.log_observation_prob(observation, x) for x in particles])

    @property
    def has_state_dependent_noise(self) -> bool:
        """
        Whether process noise covariance Q depends on state.

        Returns:
            False by default (constant Q). Override if Q(x) varies with x.
        """
        return False



