"""Minimal Single-Sensor Acoustic Tracking Test Model for Flow Filter Debugging.

This model is designed for systematic debugging with configurable difficulty levels.
"""

import numpy as np
from typing import Optional
from ..core.model_base import StateSpaceModel


class AcousticTrackingTestModel(StateSpaceModel):
    """
    Minimal single-sensor acoustic tracking model for debugging flow filters.
    
    State: [x, y, vx, vy]
        - (x, y): source position in 2D plane
        - (vx, vy): source velocity
    
    State evolution (constant velocity):
        x_{t+1} = F @ x_t + w_t, w_t ~ N(0, Q)
    
    Observation (amplitude decay at single sensor):
        z = Ψ / (r^2 + d_0) + noise
        where r = ||[x,y] - sensor_pos||
    
    Difficulty Levels:
        - easy: σ_w=0.5, d_0=5.0 (less informative, more regularized)
        - medium: σ_w=0.3, d_0=1.0
        - hard: σ_w=0.1, d_0=0.1 (paper settings)
    """
    
    def __init__(
        self,
        sensor_position: Optional[list] = None,
        source_intensity: float = 10.0,
        regularization: float = 5.0,
        measurement_noise_std: float = 0.5,
        process_noise_cov: Optional[np.ndarray] = None,
        dt: float = 1.0,
        difficulty: str = 'easy'
    ):
        """
        Initialize minimal test model.
        
        Args:
            sensor_position: [x, y] sensor location (default: [20, 20])
            source_intensity: Ψ (default: 10.0)
            regularization: d_0 (default: 5.0 for easy)
            measurement_noise_std: σ_w (default: 0.5 for easy)
            process_noise_cov: Q matrix (default: from paper)
            dt: time step (default: 1.0)
            difficulty: 'easy', 'medium', or 'hard' (overrides params if set)
        """
        self._state_dim = 4
        self._obs_dim = 1  # Single sensor
        self.dt = dt
        
        # Apply difficulty preset
        if difficulty == 'easy':
            regularization = 5.0
            measurement_noise_std = 0.5
        elif difficulty == 'medium':
            regularization = 1.0
            measurement_noise_std = 0.3
        elif difficulty == 'hard':
            regularization = 0.1
            measurement_noise_std = 0.1
        
        # Sensor at center of tracking region
        if sensor_position is None:
            sensor_position = [20.0, 20.0]
        self.sensor_position = np.array(sensor_position)
        
        # Measurement parameters
        self.source_intensity = source_intensity
        self.regularization = regularization
        self.measurement_noise_std = measurement_noise_std
        
        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process noise (from paper's filter covariance)
        if process_noise_cov is None:
            Q_single = (1/20) * np.array([
                [1/3, 0, 0.5, 0],
                [0, 1/3, 0, 0.5],
                [0.5, 0, 1, 0],
                [0, 0.5, 0, 1]
            ])
            self.Q = Q_single
        else:
            self.Q = np.array(process_noise_cov)
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise_std**2]])
    
    @property
    def state_dim(self) -> int:
        return self._state_dim
    
    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def process_noise_cov(self) -> np.ndarray:
        return self.Q
    
    @property
    def observation_noise_cov(self) -> np.ndarray:
        return self.R
    
    @property
    def initial_state_mean(self) -> np.ndarray:
        """Initial state mean: start at (10, 10) with velocity (1, 0.5)."""
        return np.array([10.0, 10.0, 1.0, 0.5])
    
    @property
    def initial_state_cov(self) -> np.ndarray:
        """Initial state covariance: moderate uncertainty."""
        return np.diag([4.0, 4.0, 1.0, 1.0])
    
    # NumPy sampling methods
    
    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample initial state from prior."""
        return rng.multivariate_normal(self.initial_state_mean, self.initial_state_cov)
    
    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample from state transition: x' = F @ x + w, w ~ N(0, Q)."""
        mean = self.F @ x
        noise = rng.multivariate_normal(np.zeros(4), self.Q)
        return mean + noise
    
    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample observation: amplitude at single sensor with Gaussian noise."""
        # Distance from source to sensor
        dx = x[0] - self.sensor_position[0]
        dy = x[1] - self.sensor_position[1]
        r_squared = dx**2 + dy**2
        
        # Amplitude decay model: z = Ψ / (r^2 + d_0)
        true_amplitude = self.source_intensity / (r_squared + self.regularization)
        
        # Add Gaussian noise
        noise = rng.normal(0, self.measurement_noise_std)
        return np.array([true_amplitude + noise])
    
    # Deterministic methods (for Kalman/EKF/UKF/Flow filters)
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of state transition: E[x' | x] = F @ x."""
        return self.F @ x
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of state transition (constant for linear model)."""
        return self.Q
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition (constant F matrix)."""
        return self.F
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of observation: E[y | x] = h(x)."""
        dx = x[0] - self.sensor_position[0]
        dy = x[1] - self.sensor_position[1]
        r_squared = dx**2 + dy**2
        amplitude = self.source_intensity / (r_squared + self.regularization)
        return np.array([amplitude])
    
    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for flow filters."""
        return self.observation_mean(x)
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Covariance of observation (constant)."""
        return self.R
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function: ∂h/∂x.
        
        h(x) = Ψ / (r^2 + d_0)
        where r^2 = (x - s_x)^2 + (y - s_y)^2
        
        ∂h/∂x = -2 * Ψ * (x - s_x) / (r^2 + d_0)^2
        ∂h/∂y = -2 * Ψ * (y - s_y) / (r^2 + d_0)^2
        ∂h/∂vx = 0
        ∂h/∂vy = 0
        
        Returns:
            H: (1, 4) Jacobian matrix
        """
        dx = x[0] - self.sensor_position[0]
        dy = x[1] - self.sensor_position[1]
        r_squared = dx**2 + dy**2
        denom = (r_squared + self.regularization)**2
        
        H = np.zeros((1, 4))
        H[0, 0] = -2 * self.source_intensity * dx / denom  # ∂h/∂x
        H[0, 1] = -2 * self.source_intensity * dy / denom  # ∂h/∂y
        H[0, 2] = 0  # ∂h/∂vx
        H[0, 3] = 0  # ∂h/∂vy
        
        return H
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """Log probability of observation given state: log p(y|x)."""
        h_x = self.observation_function(x)
        residual = y - h_x
        log_prob = -0.5 * (
            np.log(2 * np.pi * self.R[0, 0]) +
            (residual.T @ np.linalg.inv(self.R) @ residual).item()
        )
        return log_prob
    
    def observe(self, x: np.ndarray) -> np.ndarray:
        """Deterministic observation (no noise)."""
        return self.observation_mean(x)

