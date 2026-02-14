"""Lorenz 96 model for high-dimensional filtering experiments."""

import numpy as np
from typing import Optional

from ..core.model_base import StateSpaceModel
from ..utils.ode_solvers import rk4_step


class Lorenz96Model(StateSpaceModel):
    """
    Lorenz 96 model for testing high-dimensional filtering algorithms.
    
    State evolution (Eq 24 from Hu & van Leeuwen 2021):
        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        
    with cyclic boundary conditions: x_0 = x_n, x_{-1} = x_{n-1}, x_{n+1} = x_1.
    
    Time discretization uses 4th-order Runge-Kutta with step dt.
    
    Observations:
        y = H @ x + epsilon, epsilon ~ N(0, R)
        
    where H observes every `obs_spacing`-th variable (e.g., every 4th for 25% coverage).
    
    Reference:
        Hu, C.-C., & van Leeuwen, P. J. (2021). A particle flow filter for
        high-dimensional system applications. QJRMS 147(738), 2352-2374.
    """
    
    def __init__(
        self,
        state_dim: int = 1000,
        forcing: float = 8.0,
        dt: float = 0.05,
        obs_interval: int = 20,
        obs_spacing: int = 4,
        obs_noise_std: float = 1.0,
        process_noise_std: float = 0.0,
        initial_spread: float = 1.0,
        spinup_steps: int = 1000
    ):
        """
        Initialize Lorenz 96 model.
        
        Args:
            state_dim: Dimension of state space (default 1000)
            forcing: Forcing constant F (default 8.0 for chaotic regime)
            dt: Time step for RK4 integration (default 0.05)
            obs_interval: Number of RK4 steps between observations (default 20)
                         Δt_obs = obs_interval * dt = 1.0 in paper
            obs_spacing: Observe every obs_spacing-th variable (default 4 = 25%)
            obs_noise_std: Observation noise standard deviation (default 1.0)
            process_noise_std: Process noise std (default 0.0, deterministic dynamics)
            initial_spread: Std of initial perturbation (default 1.0)
            spinup_steps: Steps to reach attractor (default 1000)
        """
        self._state_dim = state_dim
        self.forcing = forcing
        self.dt = dt
        self.obs_interval = obs_interval
        self.obs_spacing = obs_spacing
        self.obs_noise_std = obs_noise_std
        self.process_noise_std = process_noise_std
        self.initial_spread = initial_spread
        self.spinup_steps = spinup_steps
        
        # Spinup state (computed lazily)
        self._spinup_state = None
        
        # Build observation matrix H: observe every obs_spacing-th variable
        # Observed indices: 3, 7, 11, ... (0-indexed: every 4th starting from index 3)
        # This gives x^(4), x^(8), ... in 1-indexed notation
        self.obs_indices = np.arange(obs_spacing - 1, state_dim, obs_spacing)
        self._obs_dim = len(self.obs_indices)
        
        # Observation matrix H (obs_dim x state_dim)
        self.H = np.zeros((self._obs_dim, state_dim))
        for i, idx in enumerate(self.obs_indices):
            self.H[i, idx] = 1.0
            
        # Observation noise covariance R = obs_noise_std^2 * I
        self.R = np.eye(self._obs_dim) * (obs_noise_std ** 2)
        
        # Process noise covariance Q (optional)
        self.Q = np.eye(state_dim) * (process_noise_std ** 2)
        
        # Climatological mean (equilibrium for F=8 is approximately F)
        self.climatological_mean = np.ones(state_dim) * forcing
        
    @property
    def state_dim(self) -> int:
        return self._state_dim
    
    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    def _lorenz96_tendency(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Lorenz 96 tendencies: dx/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        
        Vectorized with cyclic boundary conditions via np.roll.
        """
        # roll(-1) shifts left: x[i+1], roll(1) shifts right: x[i-1], roll(2): x[i-2]
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + self.forcing
    
    def _get_spinup_state(self, rng: np.random.Generator) -> np.ndarray:
        """Get a state on the attractor via spinup."""
        if self._spinup_state is None:
            # Start from perturbed climatological mean
            x = self.climatological_mean + rng.normal(0, 0.01, self._state_dim)
            # Integrate to reach attractor
            for _ in range(self.spinup_steps):
                x = rk4_step(x, self._lorenz96_tendency, self.dt)
            self._spinup_state = x
        return self._spinup_state.copy()
    
    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from initial state distribution.
        
        Returns a state on the attractor with small perturbation.
        """
        base_state = self._get_spinup_state(rng)
        return base_state + rng.normal(0, self.initial_spread, self._state_dim)
    
    def sample_state_transition(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample next state: obs_interval RK4 steps + optional process noise.
        
        Each transition covers the time between observations (Δt_obs = obs_interval * dt).
        """
        x_next = x.copy()
        for _ in range(self.obs_interval):
            x_next = rk4_step(x_next, self._lorenz96_tendency, self.dt)
            if self.process_noise_std > 0:
                x_next += rng.normal(0, self.process_noise_std, self._state_dim)
            
        return x_next
    
    def sample_observation(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample observation: y = H @ x + noise.
        """
        mean = self.H @ x
        noise = rng.normal(0, self.obs_noise_std, self._obs_dim)
        return mean + noise
    
    # Methods for Kalman-type and flow filters
    
    def state_transition_mean(self, x: np.ndarray) -> np.ndarray:
        """Deterministic state transition: E[x' | x] = obs_interval RK4 steps."""
        x_next = x.copy()
        for _ in range(self.obs_interval):
            x_next = rk4_step(x_next, self._lorenz96_tendency, self.dt)
        return x_next
    
    def state_transition_cov(self, x: np.ndarray) -> np.ndarray:
        """Process noise covariance."""
        return self.Q
    
    def state_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of state transition: ∂f/∂x (numerical approximation).
        
        Uses finite differences since Lorenz 96 + RK4 is complex analytically.
        """
        eps = 1e-7
        n = self._state_dim
        f0 = rk4_step(x, self._lorenz96_tendency, self.dt)
        J = np.zeros((n, n))
        
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            f_pert = rk4_step(x_pert, self._lorenz96_tendency, self.dt)
            J[:, j] = (f_pert - f0) / eps
            
        return J
    
    def observation_mean(self, x: np.ndarray) -> np.ndarray:
        """Observation function: h(x) = H @ x."""
        return self.H @ x
    
    def observation_cov(self, x: np.ndarray) -> np.ndarray:
        """Observation noise covariance."""
        return self.R
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function: dh/dx = H.
        
        For linear observations, this is just H (constant).
        """
        return self.H
    
    def observe(self, x: np.ndarray) -> np.ndarray:
        """Observation function h(x) for kernel flow filter."""
        return self.H @ x
    
    def log_observation_prob(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Log probability of observation: log p(y | x).
        
        p(y | x) = N(y | H @ x, R)
        """
        mean = self.H @ x
        diff = y - mean
        
        # log p(y|x) = -0.5 * [log|2πR| + (y-μ)^T R^{-1} (y-μ)]
        # For diagonal R: log|R| = sum(log(diag(R)))
        log_det = np.sum(np.log(2 * np.pi * np.diag(self.R)))
        mahalanobis = np.sum(diff**2 / np.diag(self.R))
        
        return -0.5 * (log_det + mahalanobis)
    
    @property
    def observation_noise_cov(self) -> np.ndarray:
        """Observation noise covariance R for flow filters."""
        return self.R
    
    @property
    def process_noise_cov(self) -> np.ndarray:
        """Process noise covariance Q for flow filters."""
        return self.Q

    # Batch methods for optimized particle filtering

    def state_transition_mean_batch(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized state transition mean using RK4 integration (obs_interval steps)."""
        result = particles.copy()
        for _ in range(self.obs_interval):
            result = np.array([rk4_step(x, self._lorenz96_tendency, self.dt) for x in result])
        return result

    def state_transition_cov_batch(self, particles: np.ndarray) -> np.ndarray:
        """Q is constant - return single matrix."""
        return self.Q

    def log_observation_prob_batch(self, observation: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Vectorized observation log-prob for all particles."""
        # Observed states: particles[:, self.observed_dims]
        observed_states = particles[:, self.obs_indices]  # (N, obs_dim)

        # diff: (N, obs_dim)
        diff = observation - observed_states

        # R is diagonal: sigma^2 * I
        variance = self.observation_noise_std ** 2

        # Mahalanobis: sum((diff/sigma)^2) → (N,)
        mahalanobis = np.sum(diff**2, axis=1) / variance

        # Log determinant for diagonal R
        logdet = self.obs_dim * np.log(2 * np.pi * variance)

        return -0.5 * (logdet + mahalanobis)

    def integrate_steps(self, x: np.ndarray, n_steps: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Integrate multiple RK4 steps.
        
        Useful for sparse observations (e.g., observe every 20 steps).
        
        Args:
            x: Current state
            n_steps: Number of integration steps
            rng: Random generator (optional, for process noise)
            
        Returns:
            State after n_steps integrations
        """
        for _ in range(n_steps):
            x = rk4_step(x, self._lorenz96_tendency, self.dt)
            if self.process_noise_std > 0 and rng is not None:
                x += rng.normal(0, self.process_noise_std, self._state_dim)
        return x
    
    def spinup(self, n_steps: int = 1000, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Spin up the model to reach the attractor.
        
        Args:
            n_steps: Number of spinup steps
            rng: Random generator for initial perturbation
            
        Returns:
            State on the attractor
        """
        if rng is None:
            rng = np.random.default_rng()
            
        # Start from climatological mean with perturbation
        x = self.sample_initial_state(rng)
        
        # Integrate to reach attractor
        for _ in range(n_steps):
            x = rk4_step(x, self._lorenz96_tendency, self.dt)
            
        return x

