"""
State-space models for filtering applications.

This module provides:
1. StateSpaceModel - Abstract base class for state-space models
2. LinearGaussianModel - Linear-Gaussian state-space model
3. StochasticVolatilityModel - 1D stochastic volatility model
4. RangeBearingModel - Range-bearing observation model (2D position tracking)
5. generate_data - Data generation utility
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal, poisson


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
    
    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return self.Q

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
    
    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return np.array([[self.sigma ** 2]])

class RangeBearingModel(StateSpaceModel):
    """
    Range-Bearing Observation Model.
    
    State: 2D position [x, y]
    Observation: [range, bearing]
    
    State evolution (linear with process noise):
        x_t = F·x_{t-1} + w_t,  w_t ~ N(0, Q)
    
    Observation (nonlinear):
        range = sqrt((x - x_sensor)² + (y - y_sensor)²) + v_range
        bearing = atan2(y - y_sensor, x - x_sensor) + v_bearing
        
    where:
        - (x_sensor, y_sensor): Sensor position
        - v_range ~ N(0, σ_range²): Range measurement noise
        - v_bearing ~ N(0, σ_bearing²): Bearing measurement noise
    
    Parameters:
        F: State transition matrix (2, 2) - typically identity for constant position
        Q: Process noise covariance (2, 2)
        sensor_pos: Sensor position [x_sensor, y_sensor]
        sigma_range: Standard deviation of range measurement noise
        sigma_bearing: Standard deviation of bearing measurement noise (in radians)
        mu_0: Initial state mean [x_0, y_0]
        Sigma_0: Initial state covariance (2, 2)
    """
    
    def __init__(
        self,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        sensor_pos: np.ndarray = np.array([0.0, 0.0]),
        sigma_range: float = 0.1,
        sigma_bearing: float = 0.01,
        mu_0: Optional[np.ndarray] = None,
        Sigma_0: Optional[np.ndarray] = None,
        random_state: Optional[np.random.Generator] = None
    ):
        """
        Initialize Range-Bearing Model.
        
        Args:
            F: State transition matrix (2, 2). If None, uses identity.
            Q: Process noise covariance (2, 2). If None, uses 0.01 * I.
            sensor_pos: Sensor position [x_sensor, y_sensor]
            sigma_range: Standard deviation of range measurement noise
            sigma_bearing: Standard deviation of bearing measurement noise (radians)
            mu_0: Initial state mean [x_0, y_0]. If None, uses [1.0, 1.0].
            Sigma_0: Initial state covariance (2, 2). If None, uses I.
            random_state: Optional numpy random generator
        """
        if sigma_range <= 0:
            raise ValueError(f"sigma_range must be positive, got {sigma_range}")
        if sigma_bearing <= 0:
            raise ValueError(f"sigma_bearing must be positive, got {sigma_bearing}")
        
        # Default state transition (identity for constant position)
        if F is None:
            F = np.eye(2)
        if F.shape != (2, 2):
            raise ValueError(f"F must be (2, 2), got {F.shape}")
        self.F = F
        
        # Default process noise
        if Q is None:
            Q = 0.01 * np.eye(2)
        if Q.shape != (2, 2):
            raise ValueError(f"Q must be (2, 2), got {Q.shape}")
        self.Q = Q
        
        # Sensor position
        sensor_pos = np.asarray(sensor_pos)
        if sensor_pos.shape != (2,):
            raise ValueError(f"sensor_pos must be (2,), got {sensor_pos.shape}")
        self.sensor_pos = sensor_pos
        
        # Observation noise parameters
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing
        self.R = np.diag([sigma_range ** 2, sigma_bearing ** 2])
        
        # Initial state distribution
        self.mu_0 = mu_0 if mu_0 is not None else np.array([1.0, 1.0])
        self.Sigma_0 = Sigma_0 if Sigma_0 is not None else np.eye(2)
        
        if self.mu_0.shape != (2,):
            raise ValueError(f"mu_0 must be (2,), got {self.mu_0.shape}")
        if self.Sigma_0.shape != (2, 2):
            raise ValueError(f"Sigma_0 must be (2, 2), got {self.Sigma_0.shape}")
        
        self.random_state = random_state if random_state is not None else np.random.default_rng()
    
    @property
    def state_dim(self) -> int:
        return 2
    
    @property
    def obs_dim(self) -> int:
        return 2
    
    def sample_initial_state(self, random_state: np.random.Generator) -> np.ndarray:
        """Sample from initial state distribution: X_0 ~ N(mu_0, Sigma_0)."""
        return random_state.multivariate_normal(self.mu_0, self.Sigma_0)
    
    def sample_state_transition(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = F·x + w, w ~ N(0, Q)."""
        w = random_state.multivariate_normal(np.zeros(2), self.Q)
        return self.F @ x + w
    
    def sample_observation(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """
        Sample observation: [range, bearing] with additive noise.
        
        range = sqrt((x - x_sensor)² + (y - y_sensor)²) + v_range
        bearing = atan2(y - y_sensor, x - x_sensor) + v_bearing
        """
        # Relative position
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        
        # True range and bearing
        range_true = np.sqrt(dx ** 2 + dy ** 2)
        bearing_true = np.arctan2(dy, dx)
        
        # Add noise
        v_range = random_state.normal(0, self.sigma_range)
        v_bearing = random_state.normal(0, self.sigma_bearing)
        
        return np.array([range_true + v_range, bearing_true + v_bearing])
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = F·x."""
        return self.F @ x
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition: Cov[x' | x] = Q."""
        return self.Q
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition: ∂(F·x)/∂x = F."""
        return self.F
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Mean of observation: E[y | x] = [range, bearing].
        
        range = sqrt((x - x_sensor)² + (y - y_sensor)²)
        bearing = atan2(y - y_sensor, x - x_sensor)
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        range_val = np.sqrt(dx ** 2 + dy ** 2)
        bearing_val = np.arctan2(dy, dx)
        return np.array([range_val, bearing_val])
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation: Cov[y | x] = R (constant)."""
        return self.R
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation: ∂h/∂x.
        
        h(x) = [sqrt((x-xs)² + (y-ys)²), atan2(y-ys, x-xs)]
        
        ∂range/∂x = (x - xs) / range
        ∂range/∂y = (y - ys) / range
        ∂bearing/∂x = -(y - ys) / range²
        ∂bearing/∂y = (x - xs) / range²
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        range_val = np.sqrt(dx ** 2 + dy ** 2)
        
        # Avoid division by zero
        if range_val < 1e-10:
            # At sensor position, use unit vector direction
            return np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # Jacobian matrix
        H = np.array([
            [dx / range_val, dy / range_val],
            [-dy / (range_val ** 2), dx / (range_val ** 2)]
        ])
        return H
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).
        
        p(y | x) = N(y | h(x), R)
        where h(x) = [range, bearing]
        """
        mean = self.observation_mean(x)
        diff = y - mean
        return -0.5 * (diff.T @ np.linalg.solve(self.R, diff) + 
                       np.log(np.linalg.det(2 * np.pi * self.R)))
    
    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters: returns [range, bearing]."""
        return self.observation_mean(x)
    
    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters."""
        return self.R
    
    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return self.Q

class MultiTargetAcousticModel(StateSpaceModel):
    def __init__(self):
        self.n_targets = 4
        self.n_sensors = 25
        self.tracking_area = 40.0  # 40m × 40m
        self.Psi = 10.0  # Sound amplitude
        self.d0 = 0.1
        self.sigma_w = 0.01  # Observation noise variance
        
        # Sensors at grid intersections (5×5 grid)
        grid = np.linspace(0, 40, 5)
        xx, yy = np.meshgrid(grid, grid)
        self.sensor_positions = np.column_stack([xx.ravel(), yy.ravel()])
        
        # State transition matrix F (constant velocity)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process noise (true dynamics)
        self.V_true = (1/20) * np.array([
            [1/3, 0,   0.5, 0  ],
            [0,   1/3, 0,   0.5],
            [0.5, 0,   1,   0  ],
            [0,   0.5, 0,   1  ]
        ])
        
        # Process noise (for filters - larger uncertainty)
        self.V_filter = np.array([
            [3,   0,   0.1,  0   ],
            [0,   3,   0,    0.1 ],
            [0.1, 0,   0.03, 0   ],
            [0,   0.1, 0,    0.03]
        ])
        
        # Initial states
        self.initial_states = np.array([
            [12, 6,  0.001,  0.001],
            [32, 32, -0.001, -0.005],
            [20, 13, -0.1,   0.01],
            [15, 35, 0.002,  0.002]
        ])
    
    @property
    def state_dim(self) -> int:
        return 4 * self.n_targets  # 16
    
    @property
    def obs_dim(self) -> int:
        return self.n_sensors  # 25
    
    def sample_initial_state(self, random_state: np.random.Generator) -> np.ndarray:
        """Sample initial state (can add noise to initial_states)."""
        x0 = self.initial_states.ravel()  # Shape (16,)
        return x0
    
    def sample_state_transition(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """x' = F·x + v, where v ~ N(0, V_true)."""
        x_reshaped = x.reshape(self.n_targets, 4)
        x_next = np.zeros_like(x_reshaped)
        
        for c in range(self.n_targets):
            x_next[c] = self.F @ x_reshaped[c] + random_state.multivariate_normal(
                np.zeros(4), self.V_true
            )
        
        return x_next.ravel()
    
    def sample_observation(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """z^s = sum_c Ψ/(||pos_c - R_s||^2 + d0) + w^s, w^s ~ N(0, σ²_w)."""
        z_mean = self.observation_mean(x)
        z = z_mean + random_state.normal(0, np.sqrt(self.sigma_w), self.n_sensors)
        return z
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """E[x' | x] = F·x."""
        x_reshaped = x.reshape(self.n_targets, 4)
        x_next = np.array([self.F @ x_reshaped[c] for c in range(self.n_targets)])
        return x_next.ravel()
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Block diagonal covariance (targets independent)."""
        return np.kron(np.eye(self.n_targets), self.V_filter)
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """∂f/∂x = block_diag(F, F, F, F)."""
        return np.kron(np.eye(self.n_targets), self.F)
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """h(x) = [z̄¹(x), ..., z̄^25(x)]."""
        x_reshaped = x.reshape(self.n_targets, 4)
        z = np.zeros(self.n_sensors)
        
        for s in range(self.n_sensors):
            for c in range(self.n_targets):
                pos_c = x_reshaped[c, :2]  # [x, y]
                dist_sq = np.sum((pos_c - self.sensor_positions[s])**2)
                z[s] += self.Psi / (dist_sq + self.d0)
        
        return z
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Cov[z | x] = σ²_w · I."""
        return self.sigma_w * np.eye(self.n_sensors)
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """∂h/∂x. Shape (25, 16)."""
        x_reshaped = x.reshape(self.n_targets, 4)
        H = np.zeros((self.n_sensors, self.state_dim))
        
        for s in range(self.n_sensors):
            for c in range(self.n_targets):
                pos_c = x_reshaped[c, :2]
                diff = pos_c - self.sensor_positions[s]
                dist_sq = np.sum(diff**2)
                
                # ∂z^s/∂x^c = -2Ψ·(x^c - R_s) / (||x^c - R_s||² + d0)²
                denom = (dist_sq + self.d0)**2
                H[s, c*4:c*4+2] = -2 * self.Psi * diff / denom
        
        return H
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """log p(y | x) for particle filter."""
        z_mean = self.observation_mean(x)
        return multivariate_normal.logpdf(y, mean=z_mean, cov=self.sigma_w * np.eye(self.n_sensors))

class LinearGaussianSensorNetwork(StateSpaceModel):
    def __init__(self, d=64, sigma_z=1.0):
        self.d = d
        self.alpha = 0.9
        self.alpha0 = 3.0
        self.alpha1 = 0.01
        self.beta = 20.0
        self.sigma_z = sigma_z
        
        # Sensor positions on sqrt(d) × sqrt(d) grid
        grid_size = int(np.sqrt(d))
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i, j])
        self.positions = np.array(positions)
        
        # Precompute Σ (process noise covariance)
        self.Sigma = self._compute_Sigma()
    
    def _compute_Sigma(self) -> np.ndarray:
        """Σ_{i,j} = α₀·exp(-||R_i - R_j||²/(2β)) + α₁·δ_{i,j}."""
        Sigma = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                dist_sq = np.sum((self.positions[i] - self.positions[j])**2)
                Sigma[i, j] = self.alpha0 * np.exp(-dist_sq / (2 * self.beta))
                if i == j:
                    Sigma[i, j] += self.alpha1
        return Sigma
    
    @property
    def state_dim(self) -> int:
        return self.d
    
    @property
    def obs_dim(self) -> int:
        return self.d
    
    def sample_initial_state(self, random_state: np.random.Generator) -> np.ndarray:
        """x_0 = 0."""
        return np.zeros(self.d)
    
    def sample_state_transition(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """x' = α·x + v, v ~ N(0, Σ)."""
        v = random_state.multivariate_normal(np.zeros(self.d), self.Sigma)
        return self.alpha * x + v
    
    def sample_observation(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """z = x + w, w ~ N(0, σ²_z·I)."""
        w = random_state.normal(0, self.sigma_z, self.d)
        return x + w
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * x
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        return self.Sigma
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * np.eye(self.d)
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        return self.sigma_z**2 * np.eye(self.d)
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.eye(self.d)
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        return multivariate_normal.logpdf(y, mean=x, cov=self.sigma_z**2 * np.eye(self.d))

class SkewedTPoissonModel(StateSpaceModel):
    def __init__(self, d=144, nu=5.0, m1=1.0, m2=1/3):
        self.d = d
        self.nu = nu  # Shape parameter
        self.alpha = 0.9  # State transition coefficient
        self.alpha0 = 3.0
        self.alpha1 = 0.01
        self.beta = 20.0
        self.m1 = m1
        self.m2 = m2
        
        # Sensor positions
        grid_size = int(np.sqrt(d))
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i, j])
        self.positions = np.array(positions)
        
        # Σ matrix (same structure as model 2)
        self.Sigma = self._compute_Sigma()
        
        # Skewness parameter γ (you need to specify this)
        self.gamma = np.zeros(d)  # Symmetric case; set non-zero for skewness
    
    def _compute_Sigma(self) -> np.ndarray:
        Sigma = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                dist_sq = np.sum((self.positions[i] - self.positions[j])**2)
                Sigma[i, j] = self.alpha0 * np.exp(-dist_sq / (2 * self.beta))
                if i == j:
                    Sigma[i, j] += self.alpha1
        return Sigma
    
    @property
    def state_dim(self) -> int:
        return self.d
    
    @property
    def obs_dim(self) -> int:
        return self.d
    
    def sample_initial_state(self, random_state: np.random.Generator) -> np.ndarray:
        return np.zeros(self.d)
    
    def sample_state_transition(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Sample from GH skewed-t distribution (approximation via mixture)."""
        # This is complex; simplified: sample from multivariate t-distribution
        # Exact sampling requires mixture representation
        mu_k = self.alpha * x
        # Approximate with Gaussian (for proper implementation, use t-distribution sampling)
        return random_state.multivariate_normal(mu_k, self.Sigma)
    
    def sample_observation(self, x: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """z^c ~ Poisson(m₁·exp(m₂·x^c))."""
        rates = self.m1 * np.exp(self.m2 * x)
        z = np.array([random_state.poisson(rate) for rate in rates])
        return z
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * x
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        # Approximate covariance for GH skewed-t
        nu_tilde = self.nu
        cov_scale = nu_tilde / (nu_tilde - 2) if nu_tilde > 2 else 1.0
        return cov_scale * self.Sigma
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * np.eye(self.d)
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """E[z | x] = m₁·exp(m₂·x)."""
        return self.m1 * np.exp(self.m2 * x)
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Var[Poisson(λ)] = λ, so diagonal matrix."""
        rates = self.m1 * np.exp(self.m2 * x)
        return np.diag(rates)
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """∂h/∂x where h(x) = m₁·exp(m₂·x)."""
        # ∂(m₁·exp(m₂·x^c))/∂x^c = m₁·m₂·exp(m₂·x^c)
        return np.diag(self.m1 * self.m2 * np.exp(self.m2 * x))
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """log p(y | x) = Σ_c log Poisson(y^c; m₁·exp(m₂·x^c))."""
        rates = self.m1 * np.exp(self.m2 * x)
        return np.sum(poisson.logpmf(y.astype(int), rates))

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

