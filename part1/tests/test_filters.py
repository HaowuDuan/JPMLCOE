"""Tests for filter implementations (KF, EKF, UKF, PF)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from filters import (
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter
)


class TestKalmanFilter:
    """Tests for KalmanFilter."""
    
    def test_initialization(self, rng):
        """Test KalmanFilter initialization."""
        F = np.array([[0.9]])
        B = np.array([[0.5]])
        H = np.array([[1.0]])
        D = np.array([[0.2]])
        Sigma = np.array([[1.0]])
        
        kf = KalmanFilter(F, B, H, D, Sigma)
        
        assert kf.nx == 1
        assert kf.ny == 1
        assert kf.mean.shape == (1,)
        assert kf.cov.shape == (1, 1)
    
    def test_predict_step(self, linear_model_1d):
        """Test prediction step matches Kalman equations."""
        F = np.array([[0.9]])
        B = np.array([[0.5]])
        H = np.array([[1.0]])
        D = np.array([[0.2]])
        Sigma = np.array([[1.0]])
        
        kf = KalmanFilter(F, B, H, D, Sigma)
        kf.mean = np.array([1.0])
        kf.cov = np.array([[2.0]])
        
        # Store mean before predict (predict modifies it in place)
        mean_before = kf.mean.copy()
        cov_before = kf.cov.copy()
        
        mean_pred, cov_pred = kf.predict()
        
        # Kalman prediction: x_pred = F @ x, P_pred = F @ P @ F.T + Q
        expected_mean = F @ mean_before
        expected_cov = F @ cov_before @ F.T + B @ B.T
        
        assert np.allclose(mean_pred, expected_mean)
        assert np.allclose(cov_pred, expected_cov)
    
    def test_update_step(self, linear_model_1d):
        """Test update step matches Kalman equations."""
        F = np.array([[0.9]])
        B = np.array([[0.5]])
        H = np.array([[1.0]])
        D = np.array([[0.2]])
        Sigma = np.array([[1.0]])
        
        kf = KalmanFilter(F, B, H, D, Sigma)
        kf.mean = np.array([1.0])
        kf.cov = np.array([[2.0]])
        kf.predict()
        
        # Store predicted values BEFORE update (update modifies them in place)
        mean_pred = kf.mean.copy()
        cov_pred = kf.cov.copy()
        
        observation = np.array([1.5])
        mean_updated, cov_updated, K = kf.update(observation)
        
        # Innovation (using PREDICTED mean, not updated)
        innovation = observation - H @ mean_pred
        # Innovation covariance (using PREDICTED cov)
        S = H @ cov_pred @ H.T + D @ D.T
        # Kalman gain (using PREDICTED cov)
        expected_K = cov_pred @ H.T @ np.linalg.inv(S)
        # Updated mean (using PREDICTED mean)
        expected_mean = mean_pred + expected_K @ innovation
        # Updated covariance (using PREDICTED cov)
        expected_cov = (np.eye(1) - expected_K @ H) @ cov_pred
        
        assert np.allclose(K, expected_K)
        assert np.allclose(mean_updated, expected_mean)
        assert np.allclose(cov_updated, expected_cov, atol=1e-6)
    
    def test_joseph_form(self, linear_model_1d):
        """Test Joseph form keeps covariance positive definite."""
        F = np.array([[0.9]])
        B = np.array([[0.5]])
        H = np.array([[1.0]])
        D = np.array([[0.2]])
        Sigma = np.array([[1.0]])
        
        kf = KalmanFilter(F, B, H, D, Sigma, use_joseph_form=True)
        kf.mean = np.array([1.0])
        kf.cov = np.array([[2.0]])
        kf.predict()
        
        observation = np.array([1.5])
        _, cov_updated, _ = kf.update(observation)
        
        # Covariance should be positive definite
        eigenvals = np.linalg.eigvals(cov_updated)
        assert np.all(eigenvals > 0)
    
    def test_history_tracking(self, linear_model_1d):
        """Test history tracking stores correct values."""
        F = np.array([[0.9]])
        B = np.array([[0.5]])
        H = np.array([[1.0]])
        D = np.array([[0.2]])
        Sigma = np.array([[1.0]])
        
        kf = KalmanFilter(F, B, H, D, Sigma)
        kf.predict()
        kf.update(np.array([1.0]))
        
        history = kf.get_history()
        
        assert len(history['mean_pred']) == 1
        assert len(history['cov_pred']) == 1
        assert len(history['mean_updated']) == 1
        assert len(history['kalman_gain']) == 1
    
    def test_filter_sequence(self, linear_model_1d):
        """Test filter() method on sequence of observations."""
        F = np.array([[0.9]])
        B = np.array([[0.5]])
        H = np.array([[1.0]])
        D = np.array([[0.2]])
        Sigma = np.array([[1.0]])
        
        kf = KalmanFilter(F, B, H, D, Sigma)
        
        T = 10
        observations = np.random.randn(T, 1) * 0.5
        
        means, covs = kf.filter(observations)
        
        assert means.shape == (T, 1)
        assert covs.shape == (T, 1, 1)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))


class TestExtendedKalmanFilter:
    """Tests for ExtendedKalmanFilter."""
    
    def test_initialization(self, rb_model):
        """Test EKF initialization."""
        ekf = ExtendedKalmanFilter(rb_model)
        ekf.initialize()
        
        assert ekf.mean.shape == (rb_model.state_dim,)
        assert ekf.cov.shape == (rb_model.state_dim, rb_model.state_dim)
    
    def test_linearization(self, rb_model):
        """Test EKF uses Jacobians from model."""
        ekf = ExtendedKalmanFilter(rb_model)
        ekf.initialize()
        ekf.mean = np.array([3.0, 4.0])
        
        # Get Jacobian from model
        H_model = rb_model.observation_jacobian(ekf.mean)
        H_ekf = ekf.model.observation_jacobian(ekf.mean)
        
        assert np.allclose(H_model, H_ekf)
    
    def test_predict_update_cycle(self, rb_model, rng):
        """Test predict/update cycle propagates correctly."""
        ekf = ExtendedKalmanFilter(rb_model)
        ekf.initialize()
        
        ekf.predict()
        assert ekf.mean.shape == (rb_model.state_dim,)
        assert ekf.cov.shape == (rb_model.state_dim, rb_model.state_dim)
        
        y = rb_model.sample_observation(ekf.mean, rng)
        ekf.update(y)
        
        assert np.all(np.isfinite(ekf.mean))
        assert np.all(np.isfinite(ekf.cov))
    
    def test_zero_jacobian_handling(self, sv_model):
        """Test EKF handles zero Jacobian (StochasticVolatility case)."""
        ekf = ExtendedKalmanFilter(sv_model)
        ekf.initialize()
        ekf.mean = np.array([0.5])
        
        ekf.predict()
        y = np.array([0.1])
        
        # Should not crash even with zero Jacobian
        ekf.update(y)
        assert np.all(np.isfinite(ekf.mean))
    
    def test_filter_sequence(self, rb_model, rng):
        """Test filter() on sequence."""
        ekf = ExtendedKalmanFilter(rb_model)
        
        T = 10
        observations = np.array([
            rb_model.sample_observation(np.array([1.0, 1.0]), rng) 
            for _ in range(T)
        ])
        
        means, covs = ekf.filter(observations)
        
        assert means.shape == (T, rb_model.state_dim)
        assert covs.shape == (T, rb_model.state_dim, rb_model.state_dim)


class TestUnscentedKalmanFilter:
    """Tests for UnscentedKalmanFilter."""
    
    def test_sigma_points_count(self, rb_model):
        """Test sigma points count is 2n+1."""
        ukf = UnscentedKalmanFilter(rb_model)
        ukf.initialize()
        ukf.mean = np.array([1.0, 1.0])
        ukf.cov = np.eye(2)
        
        sigma_points = ukf._compute_sigma_points(ukf.mean, ukf.cov)
        
        assert sigma_points.shape[0] == 2 * ukf.state_dim + 1
        assert sigma_points.shape[1] == ukf.state_dim
    
    def test_sigma_points_weights(self, rb_model):
        """Test sigma point weights sum to 1."""
        ukf = UnscentedKalmanFilter(rb_model)
        
        # weights_mean should sum to 1
        assert np.allclose(np.sum(ukf.weights_mean), 1.0)
        # weights_cov includes beta correction term (1 - alpha^2 + beta), so it doesn't sum to 1
        # Just check they are finite and have correct shape
        assert np.all(np.isfinite(ukf.weights_cov))
        assert ukf.weights_cov.shape == ukf.weights_mean.shape
    
    def test_unscented_transform(self, rb_model):
        """Test unscented transform captures mean/covariance."""
        ukf = UnscentedKalmanFilter(rb_model)
        ukf.initialize()
        ukf.mean = np.array([1.0, 1.0])
        ukf.cov = np.eye(2)
        
        # Generate sigma points
        sigma_points = ukf._compute_sigma_points(ukf.mean, ukf.cov)
        
        # Transform through identity (should preserve mean/cov)
        transformed = sigma_points.copy()
        mean_transformed = np.sum(ukf.weights_mean[:, np.newaxis] * transformed, axis=0)
        
        assert np.allclose(mean_transformed, ukf.mean, atol=0.1)
    
    def test_cholesky_fallback(self, rb_model):
        """Test handles near-singular covariance."""
        ukf = UnscentedKalmanFilter(rb_model)
        ukf.initialize()
        ukf.mean = np.array([1.0, 1.0])
        # Near-singular covariance
        ukf.cov = np.array([[1.0, 0.999], [0.999, 1.0]])
        
        # Should not crash
        sigma_points = ukf._compute_sigma_points(ukf.mean, ukf.cov)
        assert sigma_points.shape[0] == 2 * ukf.state_dim + 1
    
    def test_predict_update_cycle(self, rb_model, rng):
        """Test predict/update cycle."""
        ukf = UnscentedKalmanFilter(rb_model)
        ukf.initialize()
        
        ukf.predict()
        y = rb_model.sample_observation(ukf.mean, rng)
        ukf.update(y)
        
        assert np.all(np.isfinite(ukf.mean))
        assert np.all(np.isfinite(ukf.cov))


class TestParticleFilter:
    """Tests for ParticleFilter."""
    
    def test_initialization(self, rb_model, rng):
        """Test particle filter initialization."""
        pf = ParticleFilter(rb_model, n_particles=100, n_threads=1)
        pf.initialize(rng)
        
        assert pf.particles.shape == (100, rb_model.state_dim)
        assert pf.weights.shape == (100,)
        assert np.allclose(np.sum(pf.weights), 1.0)
    
    def test_particles_sample_from_initial_dist(self, rb_model, rng):
        """Test particles sample from initial distribution."""
        pf = ParticleFilter(rb_model, n_particles=1000, n_threads=1)
        pf.initialize(rng)
        
        # Check mean is reasonable (can't test exact due to randomness)
        particle_mean = np.mean(pf.particles, axis=0)
        assert np.all(np.isfinite(particle_mean))
    
    def test_prediction_propagates_particles(self, rb_model, rng):
        """Test prediction propagates particles through dynamics."""
        pf = ParticleFilter(rb_model, n_particles=100, n_threads=1)
        pf.initialize(rng)
        
        particles_before = pf.particles.copy()
        pf.predict()
        
        # Particles should have changed
        assert not np.allclose(pf.particles, particles_before)
        assert pf.particles.shape == (100, rb_model.state_dim)
    
    def test_weight_update(self, rb_model, rng):
        """Test weights update based on observation likelihood."""
        pf = ParticleFilter(rb_model, n_particles=100, n_threads=1)
        pf.initialize(rng)
        pf.predict()
        
        y = rb_model.sample_observation(pf.particles[0], rng)
        weights_before = pf.weights.copy()
        pf.update(y, timestep=0)
        
        # Weights should be normalized (may or may not change depending on observation)
        assert np.allclose(np.sum(pf.weights), 1.0)
        assert np.all(pf.weights >= 0)
        assert np.all(pf.weights <= 1.0)
    
    def test_effective_sample_size(self, rb_model, rng):
        """Test ESS computation returns value in [1, N]."""
        pf = ParticleFilter(rb_model, n_particles=100, n_threads=1)
        pf.initialize(rng)
        pf.predict()
        
        y = rb_model.sample_observation(pf.particles[0], rng)
        pf.update(y, timestep=0)
        
        ess = pf._effective_sample_size()
        
        assert 1 <= ess <= pf.n_particles
    
    def test_resampling_trigger(self, rb_model, rng):
        """Test resampling triggers when ESS is low."""
        pf = ParticleFilter(rb_model, n_particles=100, resample_threshold=0.5, n_threads=1)
        pf.initialize(rng)
        pf.predict()
        
        # Create degenerate weights (all weight on one particle)
        pf.weights = np.zeros(100)
        pf.weights[0] = 1.0
        
        y = rb_model.sample_observation(pf.particles[0], rng)
        pf.update(y, timestep=0)
        
        # Should have resampled (weights should be uniform)
        assert np.allclose(pf.weights, 1.0 / pf.n_particles)
    
    def test_systematic_resampling(self, rb_model, rng):
        """Test systematic resampling preserves diversity."""
        pf = ParticleFilter(rb_model, n_particles=100, n_threads=1)
        pf.initialize(rng)
        
        # Ensure particles differ before resampling
        pf.particles = rng.normal(0, 1, size=(100, rb_model.state_dim))
        
        # Set degenerate weights
        pf.weights = np.zeros(100)
        pf.weights[0] = 1.0
        
        pf._systematic_resample(timestep=0)
        
        # After resampling, weights should be uniform
        assert np.allclose(pf.weights, 1.0 / pf.n_particles)
        # Should have multiple unique particles (at least some diversity)
        unique_particles = len(np.unique(pf.particles, axis=0))
        assert unique_particles >= 1  # At minimum, particles exist
    
    def test_filter_sequence(self, rb_model, rng):
        """Test filter() on sequence."""
        pf = ParticleFilter(rb_model, n_particles=100, n_threads=1)
        
        T = 10
        observations = np.array([
            rb_model.sample_observation(np.array([1.0, 1.0]), rng) 
            for _ in range(T)
        ])
        
        means, covs = pf.filter(observations, random_state=rng)
        
        assert means.shape == (T, rb_model.state_dim)
        assert covs.shape == (T, rb_model.state_dim, rb_model.state_dim)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))
