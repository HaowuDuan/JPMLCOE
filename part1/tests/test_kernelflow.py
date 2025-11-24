"""Tests for Kernel Mapping Particle Filter."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from kernelflow import KernelMappingPF


class TestKernelMappingPF:
    """Tests for KernelMappingPF."""
    
    def test_initialization(self, linear_model, rng):
        """Test KernelMappingPF initialization."""
        kmpf = KernelMappingPF(linear_model, n_particles=100, kernel_type='scalar', n_threads=1)
        
        assert kmpf.n_particles == 100
        assert kmpf.kernel_type == 'scalar'
        assert kmpf.model == linear_model
    
    def test_gradient_computation_shape(self, linear_model, rng):
        """Test gradient computation returns correct shape."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='scalar', n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        h_particles = kmpf._compute_observation_matrix(kmpf.particles)
        
        grad = kmpf._compute_grad_log_posterior(0, y, h_particles)
        
        assert grad.shape == (linear_model.state_dim,)
        assert np.all(np.isfinite(grad))
    
    def test_scalar_kernel_update(self, linear_model, rng):
        """Test scalar kernel update moves particles."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='scalar', 
                                max_iter=5, epsilon=0.01, n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        particles_before = kmpf.particles.copy()
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        kmpf.update(y)
        
        # Particles should have moved
        assert not np.allclose(kmpf.particles, particles_before)
        assert np.all(np.isfinite(kmpf.particles))
    
    def test_matrix_kernel_update(self, linear_model, rng):
        """Test matrix kernel update moves particles."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='matrix',
                                max_iter=5, epsilon=0.01, n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        particles_before = kmpf.particles.copy()
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        kmpf.update(y)
        
        # Particles should have moved
        assert not np.allclose(kmpf.particles, particles_before)
        assert np.all(np.isfinite(kmpf.particles))
    
    def test_bandwidth_selection_scalar(self, linear_model, rng):
        """Test auto-bandwidth selection for scalar kernel."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='scalar',
                                alpha=None, n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        # alpha should be set to state_dim when None
        # This is tested implicitly in _update_scalar
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        kmpf.update(y)  # Should not crash
        
        assert np.all(np.isfinite(kmpf.particles))
    
    def test_bandwidth_selection_matrix(self, linear_model, rng):
        """Test auto-bandwidth selection for matrix kernel."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='matrix',
                                alpha=None, n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        # alpha should be set to 1/N when None
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        kmpf.update(y)  # Should not crash
        
        assert np.all(np.isfinite(kmpf.particles))
    
    def test_custom_bandwidth(self, linear_model, rng):
        """Test custom bandwidth parameter."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='scalar',
                                alpha=2.0, n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        assert kmpf.alpha == 2.0
        
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        kmpf.update(y)
        
        assert np.all(np.isfinite(kmpf.particles))
    
    def test_invalid_kernel_type(self, linear_model):
        """Test invalid kernel type raises error."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='invalid', n_threads=1)
        kmpf.initialize(np.random.default_rng(42))
        kmpf.predict()
        
        y = np.array([1.0])
        with pytest.raises(ValueError, match="Unknown kernel type"):
            kmpf.update(y)
    
    def test_gradient_components(self, linear_model, rng):
        """Test gradient has both likelihood and prior components."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='scalar', n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        h_particles = kmpf._compute_observation_matrix(kmpf.particles)
        
        grad = kmpf._compute_grad_log_posterior(0, y, h_particles)
        
        # Gradient should be non-zero (generally)
        # It combines likelihood gradient and prior gradient
        assert np.any(np.abs(grad) > 1e-10)
    
    def test_filter_sequence(self, linear_model, rng):
        """Test filter() on sequence."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='scalar',
                                max_iter=5, n_threads=1)
        
        T = 10
        observations = np.array([
            linear_model.sample_observation(np.array([1.0, 0.5]), rng) 
            for _ in range(T)
        ])
        
        means, covs = kmpf.filter(observations, random_state=rng)
        
        assert means.shape == (T, linear_model.state_dim)
        assert covs.shape == (T, linear_model.state_dim, linear_model.state_dim)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))
    
    def test_equal_weights_maintained(self, linear_model, rng):
        """Test particles maintain equal weights (no resampling)."""
        kmpf = KernelMappingPF(linear_model, n_particles=50, kernel_type='scalar',
                                max_iter=5, n_threads=1)
        kmpf.initialize(rng)
        kmpf.predict()
        
        y = linear_model.sample_observation(kmpf.particles[0], rng)
        kmpf.update(y)
        
        # All particles should have equal weight (implicitly)
        # Check mean/covariance uses equal weights
        mean, cov = kmpf._estimate_mean_cov()
        expected_mean = np.mean(kmpf.particles, axis=0)
        assert np.allclose(mean, expected_mean)
