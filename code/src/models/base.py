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

class nonlinear_supporting_methods(ABC):
       
  