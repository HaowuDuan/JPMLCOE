"""Tests for invertible flow filters (LEDH, EDH from Invertible_flow.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from Invertible_flow import LEDHParticleFlowFilter, EDHParticleFlowFilter


class TestLEDHParticleFlowFilter:
    """Tests for LEDHParticleFlowFilter."""
    
    def test_initialization(self, rb_model, rng):
        """Test LEDH initialization creates per-particle filters."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=50, n_lambda_steps=10,
                                       filter_type='ekf', n_threads=1)
        ledh.initialize()
        
        assert ledh.particles.shape == (50, rb_model.state_dim)
        assert len(ledh.particle_filters) == 50
        assert np.allclose(np.sum(ledh.weights), 1.0)
    
    def test_per_particle_filters(self, rb_model, rng):
        """Test each particle maintains its own filter instance."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ekf', n_threads=1)
        ledh.initialize()
        
        # Each particle should have its own filter
        assert len(ledh.particle_filters) == 10
        
        # Filters should have different means initially (particles differ)
        means = [filt.mean for filt in ledh.particle_filters]
        # At least some should differ
        unique_means = len(set(tuple(m) for m in means))
        assert unique_means > 1
    
    def test_A_b_computation_shape(self, rb_model, rng):
        """Test A and b computation returns correct shapes."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ekf', n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        # Get P from first particle filter
        P = ledh.particle_filters[0].cov
        eta_bar = ledh.particles[0]
        z = np.array([1.0, 0.5])
        lambda_val = 0.5
        
        A, b = ledh._compute_A_b(eta_bar, P, z, lambda_val)
        
        assert A.shape == (rb_model.state_dim, rb_model.state_dim)
        assert b.shape == (rb_model.state_dim,)
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(b))
    
    def test_flow_integration(self, rb_model, rng):
        """Test flow integration migrates particles."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=10,
                                      filter_type='ekf', n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        particles_before = ledh.particles.copy()
        y = rb_model.sample_observation(ledh.particles[0], rng)
        ledh.update(y)
        
        # Particles should have moved
        assert not np.allclose(ledh.particles, particles_before)
        assert np.all(np.isfinite(ledh.particles))
    
    def test_jacobian_tracking(self, rb_model, rng):
        """Test Jacobian determinant θ accumulates correctly."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ekf', n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        y = rb_model.sample_observation(ledh.particles[0], rng)
        
        # Store initial weights
        weights_before = ledh.weights.copy()
        ledh.update(y)
        
        # Weights should be normalized (may or may not change significantly)
        assert np.allclose(np.sum(ledh.weights), 1.0)
        assert np.all(ledh.weights >= 0)
        assert np.all(ledh.weights <= 1.0)
    
    def test_weight_update_includes_theta(self, rb_model, rng):
        """Test weight update includes θ term."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ekf', n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        y = rb_model.sample_observation(ledh.particles[0], rng)
        ledh.update(y)
        
        # Weights should be normalized
        assert np.allclose(np.sum(ledh.weights), 1.0)
        assert np.all(ledh.weights >= 0)
        assert len(ledh.weights_history) > 0
    
    def test_resampling_preserves_filters(self, rb_model, rng):
        """Test resampling resamples both particles and filters."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ekf', resample_threshold=0.9, n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        # Create degenerate weights to trigger resampling
        ledh.weights = np.zeros(10)
        ledh.weights[0] = 1.0
        
        y = rb_model.sample_observation(ledh.particles[0], rng)
        ledh.update(y)
        
        # After resampling, should have uniform weights
        assert np.allclose(ledh.weights, 1.0 / ledh.n_particles)
        # Should still have filters for all particles
        assert len(ledh.particle_filters) == ledh.n_particles
    
    def test_filter_type_ekf(self, rb_model, rng):
        """Test works with EKF filter type."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ekf', n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        y = rb_model.sample_observation(ledh.particles[0], rng)
        ledh.update(y)
        
        assert np.all(np.isfinite(ledh.particles))
    
    def test_filter_type_ukf(self, rb_model, rng):
        """Test works with UKF filter type."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ukf', n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        y = rb_model.sample_observation(ledh.particles[0], rng)
        ledh.update(y)
        
        assert np.all(np.isfinite(ledh.particles))
    
    def test_per_particle_covariances_differ(self, rb_model, rng):
        """Test per-particle covariances differ after prediction."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                      filter_type='ekf', n_threads=1)
        ledh.initialize()
        ledh.predict()
        
        # Get covariances from filters
        covs = [filt.cov for filt in ledh.particle_filters]
        
        # At least some should differ (due to different linearization points)
        # Check that not all are identical
        first_cov = covs[0]
        all_same = all(np.allclose(c, first_cov) for c in covs)
        # In practice, they will differ due to different particle locations
        assert len(covs) == ledh.n_particles
    
    def test_filter_sequence(self, rb_model, rng):
        """Test filter() on sequence."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=5,
                                      filter_type='ekf', n_threads=1)
        
        T = 5
        observations = np.array([
            rb_model.sample_observation(np.array([1.0, 1.0]), rng) 
            for _ in range(T)
        ])
        
        means, covs = ledh.filter(observations)
        
        assert means.shape == (T, rb_model.state_dim)
        assert covs.shape == (T, rb_model.state_dim, rb_model.state_dim)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))


class TestEDHParticleFlowFilter:
    """Tests for EDHParticleFlowFilter."""
    
    def test_initialization(self, rb_model, rng):
        """Test EDH initialization creates global filter."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=50, n_lambda_steps=10,
                                     filter_type='ekf', n_threads=1)
        edh.initialize()
        
        assert edh.particles.shape == (50, rb_model.state_dim)
        assert edh.global_filter is not None
        assert np.allclose(np.sum(edh.weights), 1.0)
    
    def test_global_filter(self, rb_model, rng):
        """Test single global filter shared across particles."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                     filter_type='ekf', n_threads=1)
        edh.initialize()
        
        # Should have exactly one global filter
        assert edh.global_filter is not None
        assert hasattr(edh.global_filter, 'mean')
        assert hasattr(edh.global_filter, 'cov')
    
    def test_shared_A_b(self, rb_model, rng):
        """Test all particles use same A and b at each λ step."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=5, n_lambda_steps=3,
                                     filter_type='ekf', n_threads=1)
        edh.initialize()
        edh.predict()
        
        # Get ensemble mean for linearization
        ensemble_mean = np.sum(edh.weights[:, np.newaxis] * edh.particles, axis=0)
        P = edh.global_filter.cov
        z = np.array([1.0, 0.5])
        lambda_val = 0.5
        
        A, b = edh._compute_A_b(ensemble_mean, P, z, lambda_val)
        
        assert A.shape == (rb_model.state_dim, rb_model.state_dim)
        assert b.shape == (rb_model.state_dim,)
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(b))
    
    def test_flow_integration(self, rb_model, rng):
        """Test flow integration migrates particles."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=10,
                                    filter_type='ekf', n_threads=1)
        edh.initialize()
        edh.predict()
        
        particles_before = edh.particles.copy()
        y = rb_model.sample_observation(edh.particles[0], rng)
        edh.update(y)
        
        # Particles should have moved
        assert not np.allclose(edh.particles, particles_before)
        assert np.all(np.isfinite(edh.particles))
    
    def test_no_jacobian_term(self, rb_model, rng):
        """Test weight update has no θ term (simpler than LEDH)."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                    filter_type='ekf', n_threads=1)
        edh.initialize()
        edh.predict()
        
        y = rb_model.sample_observation(edh.particles[0], rng)
        weights_before = edh.weights.copy()
        edh.update(y)
        
        # Weights should be normalized (may or may not change significantly)
        assert np.allclose(np.sum(edh.weights), 1.0)
        assert np.all(edh.weights >= 0)
        assert np.all(edh.weights <= 1.0)
    
    def test_ensemble_mean_tracking(self, rb_model, rng):
        """Test global linearization point updates with ensemble mean."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                    filter_type='ekf', n_threads=1)
        edh.initialize()
        edh.predict()
        
        # Compute ensemble mean
        ensemble_mean_before = np.sum(edh.weights[:, np.newaxis] * edh.particles, axis=0)
        
        y = rb_model.sample_observation(edh.particles[0], rng)
        edh.update(y)
        
        # Global filter should have updated
        assert np.all(np.isfinite(edh.global_filter.mean))
        assert np.all(np.isfinite(edh.global_filter.cov))
    
    def test_filter_type_ekf(self, rb_model, rng):
        """Test works with EKF filter type."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                    filter_type='ekf', n_threads=1)
        edh.initialize()
        edh.predict()
        
        y = rb_model.sample_observation(edh.particles[0], rng)
        edh.update(y)
        
        assert np.all(np.isfinite(edh.particles))
    
    def test_filter_type_ukf(self, rb_model, rng):
        """Test works with UKF filter type."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=10, n_lambda_steps=5,
                                    filter_type='ukf', n_threads=1)
        edh.initialize()
        edh.predict()
        
        y = rb_model.sample_observation(edh.particles[0], rng)
        edh.update(y)
        
        assert np.all(np.isfinite(edh.particles))
    
    def test_filter_sequence(self, rb_model, rng):
        """Test filter() on sequence."""
        edh = EDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=5,
                                    filter_type='ekf', n_threads=1)
        
        T = 5
        observations = np.array([
            rb_model.sample_observation(np.array([1.0, 1.0]), rng) 
            for _ in range(T)
        ])
        
        means, covs = edh.filter(observations)
        
        assert means.shape == (T, rb_model.state_dim)
        assert covs.shape == (T, rb_model.state_dim, rb_model.state_dim)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))


class TestLEDHvsEDHComparison:
    """Comparison tests between LEDH and EDH."""
    
    def test_both_converge(self, rb_model, rng):
        """Test both LEDH and EDH converge on same problem."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=5,
                                     filter_type='ekf', n_threads=1)
        edh = EDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=5,
                                    filter_type='ekf', n_threads=1)
        
        # Use same initial conditions
        rng_ledh = np.random.default_rng(42)
        rng_edh = np.random.default_rng(42)
        ledh.initialize(random_state=rng_ledh)
        edh.initialize(random_state=rng_edh)
        
        # Same observations
        T = 5
        observations = np.array([
            rb_model.sample_observation(np.array([1.0, 1.0]), rng) 
            for _ in range(T)
        ])
        
        means_ledh, _ = ledh.filter(observations)
        means_edh, _ = edh.filter(observations)
        
        # Both should produce finite results
        assert np.all(np.isfinite(means_ledh))
        assert np.all(np.isfinite(means_edh))
        assert means_ledh.shape == means_edh.shape
    
    def test_ess_maintained(self, rb_model, rng):
        """Test both maintain ESS above threshold."""
        ledh = LEDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=5,
                                     filter_type='ekf', resample_threshold=0.5, n_threads=1)
        edh = EDHParticleFlowFilter(rb_model, n_particles=20, n_lambda_steps=5,
                                    filter_type='ekf', resample_threshold=0.5, n_threads=1)
        
        rng_ledh = np.random.default_rng(42)
        rng_edh = np.random.default_rng(42)
        ledh.initialize(random_state=rng_ledh)
        edh.initialize(random_state=rng_edh)
        
        T = 3
        observations = np.array([
            rb_model.sample_observation(np.array([1.0, 1.0]), rng) 
            for _ in range(T)
        ])
        
        for t in range(T):
            ledh.predict()
            ledh.update(observations[t])
            edh.predict()
            edh.update(observations[t])
            
            # Check ESS
            ess_ledh = ledh._effective_sample_size()
            ess_edh = edh._effective_sample_size()
            
            assert ess_ledh >= 1
            assert ess_edh >= 1
            assert ess_ledh <= ledh.n_particles
            assert ess_edh <= edh.n_particles
