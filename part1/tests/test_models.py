"""Tests for state-space models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from models import (
    LinearGaussianModel,
    StochasticVolatilityModel,
    RangeBearingModel,
    MultiTargetAcousticModel,
    LinearGaussianSensorNetwork,
    SkewedTPoissonModel,
    generate_data
)


class TestLinearGaussianModel:
    """Tests for LinearGaussianModel."""
    
    def test_initialization(self, rng):
        """Test model initialization and dimension validation."""
        F = np.array([[0.9, 0.1], [0.0, 0.8]])
        B = np.array([[0.5], [0.3]])
        H = np.array([[1.0, 0.0]])
        D = np.array([[0.2]])
        
        model = LinearGaussianModel(F, B, H, D, random_state=rng)
        
        assert model.state_dim == 2
        assert model.obs_dim == 1
        assert model.Q.shape == (2, 2)
        assert model.R.shape == (1, 1)
        assert np.allclose(model.Q, B @ B.T)
        assert np.allclose(model.R, D @ D.T)
    
    def test_sampling_consistency(self, linear_model, rng):
        """Test that sampling matches initial distribution."""
        n_samples = 1000
        samples = np.array([
            linear_model.sample_initial_state(rng) for _ in range(n_samples)
        ])
        
        sample_mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples.T)
        
        # Check mean is close to mu_0
        assert np.allclose(sample_mean, linear_model.mu_0, atol=0.1)
        # Check covariance is close to Sigma_0
        assert np.allclose(sample_cov, linear_model.Sigma_0, atol=0.1)
    
    def test_deterministic_methods(self, linear_model):
        """Test deterministic mean and Jacobian methods."""
        x = np.array([1.0, 2.0])
        
        # State transition mean
        x_pred = linear_model.state_transition_mean(x)
        assert np.allclose(x_pred, linear_model.F @ x)
        
        # Observation mean
        y_pred = linear_model.observation_mean(x)
        assert np.allclose(y_pred, linear_model.H @ x)
        
        # Jacobians
        assert np.allclose(linear_model.state_jacobian(x), linear_model.F)
        assert np.allclose(linear_model.observation_jacobian(x), linear_model.H)
    
    def test_log_observation_prob(self, linear_model, rng):
        """Test log probability computation."""
        x = np.array([1.0, 2.0])
        y = np.array([1.5])
        
        log_prob = linear_model.log_observation_prob(y, x)
        
        # Manual computation
        mean = linear_model.H @ x
        diff = y - mean
        expected = -0.5 * (diff.T @ np.linalg.solve(linear_model.R, diff) + 
                           np.log(np.linalg.det(2 * np.pi * linear_model.R)))
        
        assert np.allclose(log_prob, expected)
    
    def test_state_transition_sampling(self, linear_model, rng):
        """Test state transition sampling."""
        x = np.array([1.0, 0.5])
        x_next = linear_model.sample_state_transition(x, rng)
        
        assert x_next.shape == (2,)
        # Mean should be F @ x
        # We can't test exact value due to randomness, but can test shape


class TestStochasticVolatilityModel:
    """Tests for StochasticVolatilityModel."""
    
    def test_initialization(self, rng):
        """Test model initialization."""
        model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5, random_state=rng)
        
        assert model.state_dim == 1
        assert model.obs_dim == 1
        assert model.stationary_var > 0
    
    def test_state_evolution(self, sv_model, rng):
        """Test state transition follows AR(1) with correct variance."""
        x = np.array([0.5])
        n_samples = 1000
        
        samples = np.array([
            sv_model.sample_state_transition(x, rng) for _ in range(n_samples)
        ])
        
        # Mean should be alpha * x
        sample_mean = np.mean(samples)
        expected_mean = sv_model.alpha * x[0]
        assert np.abs(sample_mean - expected_mean) < 0.1
        
        # Variance should be sigma^2
        sample_var = np.var(samples)
        assert np.abs(sample_var - sv_model.sigma ** 2) < 0.1
    
    def test_observation_scaling(self, sv_model, rng):
        """Test observation uses exp(x/2) scaling."""
        x = np.array([1.0])
        y = sv_model.sample_observation(x, rng)
        
        assert y.shape == (1,)
        # Can't test exact value due to randomness
    
    def test_observation_jacobian_zero(self, sv_model):
        """Test observation Jacobian is zero (since E[y|x]=0)."""
        x = np.array([0.5])
        H = sv_model.observation_jacobian(x)
        assert np.allclose(H, np.array([[0.0]]))
    
    def test_stationary_variance(self, sv_model):
        """Test stationary variance calculation."""
        expected_var = (sv_model.sigma ** 2) / (1 - sv_model.alpha ** 2)
        assert np.allclose(sv_model.stationary_var, expected_var)


class TestRangeBearingModel:
    """Tests for RangeBearingModel."""
    
    def test_observation_mean(self, rb_model):
        """Test range/bearing computation."""
        x = np.array([3.0, 4.0])  # 5 units from origin
        y = rb_model.observation_mean(x)
        
        expected_range = np.sqrt(3.0**2 + 4.0**2)
        expected_bearing = np.arctan2(4.0, 3.0)
        
        assert np.allclose(y[0], expected_range)
        assert np.allclose(y[1], expected_bearing)
    
    def test_observation_jacobian(self, rb_model):
        """Test observation Jacobian computation."""
        x = np.array([3.0, 4.0])
        H = rb_model.observation_jacobian(x)
        
        assert H.shape == (2, 2)
        
        # Check range derivatives
        range_val = np.sqrt(3.0**2 + 4.0**2)
        assert np.allclose(H[0, 0], 3.0 / range_val)
        assert np.allclose(H[0, 1], 4.0 / range_val)
    
    def test_jacobian_at_sensor_position(self, rb_model):
        """Test Jacobian handles sensor position (division by zero)."""
        # Place target at sensor position
        x = rb_model.sensor_pos.copy()
        H = rb_model.observation_jacobian(x)
        
        # Should return valid matrix (not NaN/Inf)
        assert H.shape == (2, 2)
        assert np.all(np.isfinite(H))
    
    def test_sampling(self, rb_model, rng):
        """Test state and observation sampling."""
        x = np.array([1.0, 1.0])
        x_next = rb_model.sample_state_transition(x, rng)
        y = rb_model.sample_observation(x, rng)
        
        assert x_next.shape == (2,)
        assert y.shape == (2,)


class TestHighDimensionalModels:
    """Smoke tests for high-dimensional models."""
    
    def test_multi_target_acoustic_init(self, rng):
        """Test MultiTargetAcousticModel initialization."""
        model = MultiTargetAcousticModel()
        
        assert model.state_dim == 16  # 4 targets × 4 states
        assert model.obs_dim == 25  # 5×5 sensor grid
    
    def test_multi_target_acoustic_sampling(self, rng):
        """Test MultiTargetAcousticModel sampling."""
        model = MultiTargetAcousticModel()
        x0 = model.sample_initial_state(rng)
        x_next = model.sample_state_transition(x0, rng)
        y = model.sample_observation(x0, rng)
        
        assert x0.shape == (model.state_dim,)
        assert x_next.shape == (model.state_dim,)
        assert y.shape == (model.obs_dim,)
    
    def test_linear_gaussian_sensor_network_init(self, rng):
        """Test LinearGaussianSensorNetwork initialization."""
        model = LinearGaussianSensorNetwork(d=64, sigma_z=1.0)
        
        assert model.state_dim == 64
        assert model.obs_dim == 64
    
    def test_linear_gaussian_sensor_network_sampling(self, rng):
        """Test LinearGaussianSensorNetwork sampling."""
        model = LinearGaussianSensorNetwork(d=64, sigma_z=1.0)
        x0 = model.sample_initial_state(rng)
        x_next = model.sample_state_transition(x0, rng)
        y = model.sample_observation(x0, rng)
        
        assert x0.shape == (model.state_dim,)
        assert x_next.shape == (model.state_dim,)
        assert y.shape == (model.obs_dim,)
    
    def test_skewed_t_poisson_init(self, rng):
        """Test SkewedTPoissonModel initialization."""
        model = SkewedTPoissonModel(d=144, nu=5.0)
        
        assert model.state_dim == 144
        assert model.obs_dim == 144
    
    def test_skewed_t_poisson_sampling(self, rng):
        """Test SkewedTPoissonModel sampling."""
        model = SkewedTPoissonModel(d=144, nu=5.0)
        x0 = model.sample_initial_state(rng)
        x_next = model.sample_state_transition(x0, rng)
        y = model.sample_observation(x0, rng)
        
        assert x0.shape == (model.state_dim,)
        assert x_next.shape == (model.state_dim,)
        assert y.shape == (model.obs_dim,)


class TestDataGeneration:
    """Tests for data generation utility."""
    
    def test_generate_data_shapes(self, linear_model, rng):
        """Test generate_data returns correct shapes."""
        T = 50
        true_states, observations = generate_data(linear_model, T=T, random_state=rng)
        
        assert true_states.shape == (T, linear_model.state_dim)
        assert observations.shape == (T, linear_model.obs_dim)
    
    def test_generate_data_consistency(self, linear_model, rng):
        """Test state evolution is consistent with model dynamics."""
        T = 10
        true_states, observations = generate_data(linear_model, T=T, random_state=rng)
        
        # Check that states evolve according to dynamics (approximately)
        # Can't test exactly due to noise, but can check shapes and ranges
        assert np.all(np.isfinite(true_states))
        assert np.all(np.isfinite(observations))
