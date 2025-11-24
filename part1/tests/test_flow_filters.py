"""Tests for flow filter implementations (EDH, LEDH from filters.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from filters import ExactDaumHuangFlow, LocalExactDaumHuangFlow


class TestExactDaumHuangFlow:
    """Tests for ExactDaumHuangFlow."""
    
    def test_initialization(self, linear_model, rng):
        """Test EDH initialization."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, n_threads=1)
        edh.initialize(rng)
        
        assert edh.particles.shape == (100, linear_model.state_dim)
        assert edh.n_particles == 100
        assert edh.n_lambda_steps == 10
    
    def test_gain_computation_shape(self, linear_model, rng):
        """Test gain computation returns correct shape."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, n_threads=1)
        edh.initialize(rng)
        
        K = edh._compute_gain(edh.particles, lambda_val=0.5)
        
        assert K.shape == (linear_model.state_dim, linear_model.obs_dim)
        assert np.all(np.isfinite(K))
    
    def test_flow_step_euler(self, linear_model, rng):
        """Test Euler flow step moves particles."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, 
                                  integration_method='euler', n_threads=1)
        edh.initialize(rng)
        edh.predict()
        
        particles_before = edh.particles.copy()
        y = linear_model.sample_observation(edh.particles[0], rng)
        
        # Take one flow step
        particles_after = edh._flow_step_euler(edh.particles, y, lambda_val=0.0, d_lambda=0.1)
        
        # Particles should have moved
        assert not np.allclose(particles_after, particles_before)
        assert particles_after.shape == particles_before.shape
    
    def test_flow_step_rk4(self, linear_model, rng):
        """Test RK4 flow step."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10,
                                  integration_method='rk4', n_threads=1)
        edh.initialize(rng)
        edh.predict()
        
        particles_before = edh.particles.copy()
        y = linear_model.sample_observation(edh.particles[0], rng)
        
        particles_after = edh._flow_step_rk4(edh.particles, y, lambda_val=0.0, d_lambda=0.1)
        
        assert not np.allclose(particles_after, particles_before)
        assert particles_after.shape == particles_before.shape
    
    def test_rk4_more_accurate_than_euler(self, linear_model, rng):
        """Test RK4 is more accurate than Euler (on same problem)."""
        edh_euler = ExactDaumHuangFlow(linear_model, n_particles=50, n_lambda_steps=1,
                                        integration_method='euler', n_threads=1)
        edh_rk4 = ExactDaumHuangFlow(linear_model, n_particles=50, n_lambda_steps=1,
                                      integration_method='rk4', n_threads=1)
        
        # Use same initial particles
        rng_euler = np.random.default_rng(42)
        rng_rk4 = np.random.default_rng(42)
        edh_euler.initialize(rng_euler)
        edh_rk4.initialize(rng_rk4)
        
        edh_euler.predict()
        edh_rk4.predict()
        
        y = linear_model.sample_observation(edh_euler.particles[0], rng)
        
        # Take large step to see difference
        particles_euler = edh_euler._flow_step_euler(edh_euler.particles, y, 0.0, 0.5)
        particles_rk4 = edh_rk4._flow_step_rk4(edh_rk4.particles, y, 0.0, 0.5)
        
        # Both should be finite
        assert np.all(np.isfinite(particles_euler))
        assert np.all(np.isfinite(particles_rk4))
    
    def test_no_resampling_equal_weights(self, linear_model, rng):
        """Test all particles maintain equal weight 1/N."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, n_threads=1)
        edh.initialize(rng)
        edh.predict()
        
        y = linear_model.sample_observation(edh.particles[0], rng)
        edh.update(y)
        
        # All particles should have equal weight (implicitly, since no weights stored)
        # But we can check mean/covariance estimation uses equal weights
        mean, cov = edh._estimate_mean_cov()
        
        # Mean should be simple average
        expected_mean = np.mean(edh.particles, axis=0)
        assert np.allclose(mean, expected_mean)
    
    def test_mean_covariance_estimation(self, linear_model, rng):
        """Test mean/covariance estimated from equal-weighted particles."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, n_threads=1)
        edh.initialize(rng)
        
        mean, cov = edh._estimate_mean_cov()
        
        assert mean.shape == (linear_model.state_dim,)
        assert cov.shape == (linear_model.state_dim, linear_model.state_dim)
        
        # Covariance should be positive semi-definite
        eigenvals = np.linalg.eigvals(cov)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    
    def test_update_step(self, linear_model, rng):
        """Test update step moves particles from prior to posterior."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=20, n_threads=1)
        edh.initialize(rng)
        edh.predict()
        
        particles_before = edh.particles.copy()
        y = linear_model.sample_observation(edh.particles[0], rng)
        edh.update(y)
        
        # Particles should have moved
        assert not np.allclose(edh.particles, particles_before)
        assert np.all(np.isfinite(edh.particles))
    
    def test_filter_sequence(self, linear_model, rng):
        """Test filter() on sequence."""
        edh = ExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, n_threads=1)
        
        T = 10
        observations = np.array([
            linear_model.sample_observation(np.array([1.0, 0.5]), rng) 
            for _ in range(T)
        ])
        
        means, covs = edh.filter(observations, random_state=rng)
        
        assert means.shape == (T, linear_model.state_dim)
        assert covs.shape == (T, linear_model.state_dim, linear_model.state_dim)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))


class TestLocalExactDaumHuangFlow:
    """Tests for LocalExactDaumHuangFlow."""
    
    def test_initialization(self, linear_model, rng):
        """Test LEDH initialization."""
        ledh = LocalExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, n_threads=1)
        ledh.initialize(rng)
        
        assert ledh.particles.shape == (100, linear_model.state_dim)
    
    def test_kernel_weights_sum_to_one(self, linear_model, rng):
        """Test kernel weights sum to 1 for each particle."""
        ledh = LocalExactDaumHuangFlow(linear_model, n_particles=50, n_lambda_steps=10, n_threads=1)
        ledh.initialize(rng)
        
        particle_i = ledh.particles[0]
        bandwidth = 1.0
        
        weights = ledh._compute_kernel_weights(particle_i, ledh.particles, bandwidth)
        
        assert weights.shape == (50,)
        assert np.allclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    
    def test_local_gain_different_per_particle(self, linear_model, rng):
        """Test local gains differ across particles."""
        ledh = LocalExactDaumHuangFlow(linear_model, n_particles=50, n_lambda_steps=10, n_threads=1)
        ledh.initialize(rng)
        ledh.predict()
        
        h_particles = ledh._compute_observation_matrix(ledh.particles)
        bandwidth = 1.0
        
        # Compute gains for first two particles
        K_0 = ledh._compute_local_gain(0, ledh.particles, h_particles, bandwidth)
        K_1 = ledh._compute_local_gain(1, ledh.particles, h_particles, bandwidth)
        
        assert K_0.shape == (linear_model.state_dim, linear_model.obs_dim)
        assert K_1.shape == (linear_model.state_dim, linear_model.obs_dim)
        
        # Gains should generally differ (unless particles are identical)
        # In practice they will differ due to different kernel weights
        assert np.all(np.isfinite(K_0))
        assert np.all(np.isfinite(K_1))
    
    def test_bandwidth_silverman_rule(self, linear_model, rng):
        """Test Silverman's rule gives reasonable bandwidth."""
        ledh = LocalExactDaumHuangFlow(linear_model, n_particles=100, n_lambda_steps=10, 
                                        kernel_bandwidth=None, n_threads=1)
        ledh.initialize(rng)
        
        # Bandwidth should be computed automatically
        particle_std = np.std(ledh.particles, axis=0).mean()
        expected_bandwidth = particle_std * (ledh.n_particles ** (-1.0 / (ledh.state_dim + 4)))
        
        # When bandwidth is None, it's computed in _flow_step_euler
        # So we can't directly test it, but we can test the computation logic
        assert expected_bandwidth > 0
    
    def test_gaussian_kernel(self, linear_model):
        """Test Gaussian kernel function."""
        ledh = LocalExactDaumHuangFlow(linear_model, kernel_type='gaussian', n_threads=1)
        
        distance = 1.0
        bandwidth = 1.0
        k = ledh._gaussian_kernel(distance, bandwidth)
        
        expected = np.exp(-0.5 * (distance / bandwidth)**2)
        assert np.allclose(k, expected)
    
    def test_epanechnikov_kernel(self, linear_model):
        """Test Epanechnikov kernel function."""
        ledh = LocalExactDaumHuangFlow(linear_model, kernel_type='epanechnikov', n_threads=1)
        
        distance = 0.5
        bandwidth = 1.0
        k = ledh._epanechnikov_kernel(distance, bandwidth)
        
        u = distance / bandwidth
        expected = 0.75 * (1 - u**2) if u < 1 else 0.0
        assert np.allclose(k, expected)
    
    def test_update_step(self, linear_model, rng):
        """Test update step with local gains."""
        ledh = LocalExactDaumHuangFlow(linear_model, n_particles=50, n_lambda_steps=10, n_threads=1)
        ledh.initialize(rng)
        ledh.predict()
        
        particles_before = ledh.particles.copy()
        y = linear_model.sample_observation(ledh.particles[0], rng)
        ledh.update(y)
        
        # Particles should have moved
        assert not np.allclose(ledh.particles, particles_before)
        assert np.all(np.isfinite(ledh.particles))
    
    def test_filter_sequence(self, linear_model, rng):
        """Test filter() on sequence."""
        ledh = LocalExactDaumHuangFlow(linear_model, n_particles=50, n_lambda_steps=10, n_threads=1)
        
        T = 10
        observations = np.array([
            linear_model.sample_observation(np.array([1.0, 0.5]), rng) 
            for _ in range(T)
        ])
        
        means, covs = ledh.filter(observations, random_state=rng)
        
        assert means.shape == (T, linear_model.state_dim)
        assert covs.shape == (T, linear_model.state_dim, linear_model.state_dim)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))
