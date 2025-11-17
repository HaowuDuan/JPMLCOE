"""
State-space models for filtering applications.

This module provides:
1. StateSpaceModel - Abstract base class for state-space models
2. LinearGaussianModel - Linear-Gaussian state-space model
3. StochasticVolatilityModel - 1D stochastic volatility model
4. generate_data - Data generation utility
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class StateSpaceModel(ABC):
    """
    Abstract base class for state-space models.
    
    A state-space model defines:
    - State transition: x_t = f(x_{t-1}) + w_t
    - Observation: y_t = h(x_t) + v_t
    
    where w_t and v_t are noise processes.
    """
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """State dimension."""
        pass
    
    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Observation dimension."""
        pass
    
    @abstractmethod
    def sample_initial_state(self, random_state: np.random.Generator) -> np.ndarray:
        """Sample from initial state distribution."""
        pass
    
    @abstractmethod
    def sample_state_transition(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' ~ p(x' | x)."""
        pass
    
    @abstractmethod
    def sample_observation(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample observation: y ~ p(y | x)."""
        pass
    
    # Methods for EKF/UKF
    @abstractmethod
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x]."""
        pass
    
    @abstractmethod
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Cov[x' | x]."""
        pass
    
    @abstractmethod
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂f/∂x."""
        pass
    
    @abstractmethod
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[y | x]."""
        pass
    
    @abstractmethod
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Cov[y | x]."""
        pass
    
    @abstractmethod
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of observation: ∂h/∂x."""
        pass
    
    # Method for Particle Filter
    @abstractmethod
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """Log probability of observation: log p(y | x)."""
        pass


class LinearGaussianModel(StateSpaceModel):
    """
    Linear-Gaussian state-space model.
    
    Model:
        X_n = F·X_{n-1} + B·V_n,  V_n ~ N(0, I)
        Y_n = H·X_n + D·W_n,  W_n ~ N(0, I)
    
    where:
        - F: State transition matrix (nx, nx)
        - B: Process noise matrix (nx, nv)
        - H: Observation matrix (ny, nx)
        - D: Observation noise matrix (ny, nw)
        - Q = B·B^T: Process noise covariance
        - R = D·D^T: Observation noise covariance
        - mu_0: Initial state mean (nx,)
        - Sigma_0: Initial state covariance (nx, nx)
    
    This is the standard linear-Gaussian model for which the Kalman Filter
    is optimal. Can also be used with EKF/UKF/PF (though KF is optimal).
    """
    
    def __init__(
        self,
        F: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        D: np.ndarray,
        mu_0: Optional[np.ndarray] = None,
        Sigma_0: Optional[np.ndarray] = None,
        random_state: Optional[np.random.Generator] = None
    ):
        """
        Initialize Linear-Gaussian Model.
        
        Args:
            F: State transition matrix (nx, nx)
            B: Process noise matrix (nx, nv)
            H: Observation matrix (ny, nx)
            D: Observation noise matrix (ny, nw)
            mu_0: Initial state mean (nx,). If None, uses zero vector.
            Sigma_0: Initial state covariance (nx, nx). If None, uses identity.
            random_state: Optional numpy random generator
        """
        # Validate dimensions
        self.nx = F.shape[0]
        self.nv = B.shape[1]
        self.ny = H.shape[0]
        self.nw = D.shape[1]
        
        if F.shape != (self.nx, self.nx):
            raise ValueError(f"F must be ({self.nx}, {self.nx}), got {F.shape}")
        if B.shape != (self.nx, self.nv):
            raise ValueError(f"B must be ({self.nx}, {self.nv}), got {B.shape}")
        if H.shape != (self.ny, self.nx):
            raise ValueError(f"H must be ({self.ny}, {self.nx}), got {H.shape}")
        if D.shape != (self.ny, self.nw):
            raise ValueError(f"D must be ({self.ny}, {self.nw}), got {D.shape}")
        
        self.F = F
        self.B = B
        self.H = H
        self.D = D
        
        # Compute noise covariances
        self.Q = B @ B.T  # Process noise covariance
        self.R = D @ D.T  # Observation noise covariance
        
        # Initial state distribution
        self.mu_0 = mu_0 if mu_0 is not None else np.zeros(self.nx)
        self.Sigma_0 = Sigma_0 if Sigma_0 is not None else np.eye(self.nx)
        
        if self.mu_0.shape != (self.nx,):
            raise ValueError(f"mu_0 must be ({self.nx},), got {self.mu_0.shape}")
        if self.Sigma_0.shape != (self.nx, self.nx):
            raise ValueError(f"Sigma_0 must be ({self.nx}, {self.nx}), got {self.Sigma_0.shape}")
        
        self.random_state = random_state if random_state is not None else np.random.default_rng()
    
    @property
    def state_dim(self) -> int:
        return self.nx
    
    @property
    def obs_dim(self) -> int:
        return self.ny
    
    def sample_initial_state(self, random_state: np.random.Generator) -> np.ndarray:
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0)."""
        return random_state.multivariate_normal(self.mu_0, self.Sigma_0)
    
    def sample_state_transition(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample from state transition: X' = F·X + B·V, V ~ N(0, I)."""
        v = random_state.multivariate_normal(np.zeros(self.nv), np.eye(self.nv))
        return self.F @ x + self.B @ v
    
    def sample_observation(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample observation: Y = H·X + D·W, W ~ N(0, I)."""
        w = random_state.multivariate_normal(np.zeros(self.nw), np.eye(self.nw))
        return self.H @ x + self.D @ w
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[X' | X] = F·X."""
        return self.F @ x
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Cov[X' | X] = Q."""
        return self.Q
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂(F·x)/∂x = F."""
        return self.F
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[Y | X] = H·X."""
        return self.H @ x
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Cov[Y | X] = R."""
        return self.R
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of observation: ∂(H·x)/∂x = H."""
        return self.H
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).
        
        p(y | x) = N(y | H·x, R)
        """
        mean = self.H @ x
        diff = y - mean
        return -0.5 * (diff.T @ np.linalg.solve(self.R, diff) + 
                       np.log(np.linalg.det(2 * np.pi * self.R)))
    
    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters: returns H·x."""
        return self.H @ x
    
    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters."""
        return self.R


class StochasticVolatilityModel(StateSpaceModel):
    """
    1D Stochastic Volatility Model.
    
    State evolution (linear):
        x_t = α·x_{t-1} + σ·w_t,  w_t ~ N(0, 1)
    
    Observation (nonlinear, non-Gaussian):
        y_t = β·exp(x_t/2)·v_t,  v_t ~ N(0, 1)
    
    Parameters:
        α: persistence parameter (0 < α < 1)
        σ: volatility of volatility
        β: scale parameter
    
    Key features:
    - Linear state evolution
    - Nonlinear observation (exponential)
    - Non-Gaussian observation likelihood
    - Stationary variance: σ²/(1 - α²)
    """
    
    def __init__(self, alpha: float = 0.91, sigma: float = 1.0, beta: float = 0.5,
                 random_state: Optional[np.random.Generator] = None):
        """
        Initialize Stochastic Volatility Model.
        
        Args:
            alpha: Persistence parameter (0 < alpha < 1)
            sigma: Volatility of volatility
            beta: Scale parameter
            random_state: Optional numpy random generator
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.random_state = random_state if random_state is not None else np.random.default_rng()
        
        # Stationary variance
        self.stationary_var = (sigma ** 2) / (1 - alpha ** 2)
    
    @property
    def state_dim(self) -> int:
        return 1
    
    @property
    def obs_dim(self) -> int:
        return 1
    
    def sample_initial_state(self, random_state: np.random.Generator) -> np.ndarray:
        """Sample from stationary distribution: N(0, σ²/(1-α²))."""
        return np.array([random_state.normal(0, np.sqrt(self.stationary_var))])
        
    def sample_state_transition(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = α·x + σ·w."""
        w = random_state.normal(0, 1)
        return np.array([self.alpha * x[0] + self.sigma * w])
    
    def sample_observation(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample observation: y = β·exp(x/2)·v."""
        v = random_state.normal(0, 1)
        return np.array([self.beta * np.exp(x[0] / 2) * v])
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = α·x."""
        return np.array([self.alpha * x[0]])
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Var[x' | x] = σ²."""
        return np.array([[self.sigma ** 2]])
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂f/∂x = α."""
        return np.array([[self.alpha]])
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[y | x] = 0."""
        return np.array([0.0])
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Var[y | x] = β²·exp(x)."""
        return np.array([[self.beta ** 2 * np.exp(x[0])]])
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation mean.
        
        Note: Since E[y | x] = 0, the Jacobian is 0.
        For EKF, this means the filter cannot update based on the observation mean.
        Alternative approaches use the observation variance or squared observations.
        """
        return np.array([[0.0]])
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).
        
        p(y | x) = N(y | 0, β²·exp(x))
        """
        var = self.beta ** 2 * np.exp(x[0])
        return -0.5 * (np.log(2 * np.pi * var) + (y[0] ** 2) / var)
    
    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters: returns observation mean."""
        return self.observation_mean(x)
    
    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters.
        
        For stochastic volatility, use observation covariance at stationary mean (x=0).
        """
        return np.array([[self.beta ** 2]])


def generate_data(model: StateSpaceModel, T: int = 500,
                  random_state: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a state-space model.
    
    Args:
        model: State-space model instance
        T: Number of time steps
        random_state: Optional numpy random generator for reproducibility
        
    Returns:
        Tuple of (true_states, observations) where:
        - true_states: Array of shape (T, state_dim)
        - observations: Array of shape (T, obs_dim)
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    true_states = np.zeros((T, model.state_dim))
    observations = np.zeros((T, model.obs_dim))
    
    # Sample initial state
    true_states[0] = model.sample_initial_state(random_state)
    observations[0] = model.sample_observation(true_states[0], random_state)
    
    # Generate remaining states and observations
    for t in range(1, T):
        true_states[t] = model.sample_state_transition(true_states[t-1], random_state)
        observations[t] = model.sample_observation(true_states[t], random_state)
    
    return true_states, observations

