"""Tests for ExtendedKalmanFilter implementation."""

import pytest
import numpy as np
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
from src.models.stochastic_volatility import StochasticVolatilityModel
from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data


# ============================================================================
# FIXTURES - Reusable test setup
# ============================================================================

@pytest.fixture
def stochastic_volatility_model():
    """Create a stochastic volatility model for testing."""
    return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)


@pytest.fixture
def range_bearing_model():
    """Create a range-bearing model for testing."""
    return RangeBearingModel(
        F=np.eye(2),
        Q=0.01 * np.eye(2),
        sensor_pos=np.array([0.0, 0.0]),
        sigma_range=0.1,
        sigma_bearing=0.01
    )


# ============================================================================
# TEST 1: INITIALIZATION
# ============================================================================

class TestInitialization:
    """Test ExtendedKalmanFilter initialization."""
    
    def test_basic_init(self, stochastic_volatility_model):
        """Test basic initialization works."""
        model = stochastic_volatility_model
        ekf = ExtendedKalmanFilter(model)
        
        assert ekf.state_dim == 1
        assert ekf.mean is not None
        assert ekf.cov is not None
    
    def test_init_with_custom_state(self, range_bearing_model):
        """Test initialization with custom initial state."""
        model = range_bearing_model
        mean_0 = np.array([2.0, 3.0])
        Sigma_0 = 2.0 * np.eye(2)
        
        ekf = ExtendedKalmanFilter(model, mean_0=mean_0, Sigma_0=Sigma_0)
        
        assert np.allclose(ekf.mean, mean_0)
        assert np.allclose(ekf.cov, Sigma_0)
    
    def test_init_uses_model_defaults(self, stochastic_volatility_model):
        """Test initialization uses model's stationary variance."""
        model = stochastic_volatility_model
        ekf = ExtendedKalmanFilter(model)
        
        # Should use stationary variance
        expected_var = model.stationary_var
        assert np.allclose(ekf.Sigma_0[0, 0], expected_var)
    
    def test_init_log_likelihoods_empty(self, range_bearing_model):
        """Test log-likelihoods start empty."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        assert len(ekf.log_likelihoods) == 0


# ============================================================================
# TEST 2: RESET
# ============================================================================

class TestReset:
    """Test reset functionality."""
    
    def test_reset_returns_to_initial(self, range_bearing_model):
        """Test reset returns filter to initial state."""
        mean_0 = np.array([1.0, 2.0])
        ekf = ExtendedKalmanFilter(range_bearing_model, mean_0=mean_0)
        
        # Run some steps
        ekf.predict()
        ekf.update(np.array([1.5, 0.5]))
        
        # Reset
        ekf.reset()
        
        assert np.allclose(ekf.mean, mean_0)
    
    def test_reset_clears_log_likelihoods(self, stochastic_volatility_model):
        """Test reset clears log-likelihoods."""
        ekf = ExtendedKalmanFilter(stochastic_volatility_model)
        
        ekf.predict()
        ekf.update(np.array([1.0]))
        assert len(ekf.log_likelihoods) > 0
        
        ekf.reset()
        assert len(ekf.log_likelihoods) == 0


# ============================================================================
# TEST 3: PREDICT STEP
# ============================================================================

class TestPredict:
    """Test prediction step."""
    
    def test_predict_propagates_mean(self, stochastic_volatility_model):
        """Test mean propagation through nonlinear dynamics."""
        model = stochastic_volatility_model
        mean_0 = np.array([1.0])
        ekf = ExtendedKalmanFilter(model, mean_0=mean_0)
        
        ekf.predict()
        
        # For SV model: x' = alpha * x
        expected_mean = model.alpha * mean_0
        assert np.allclose(ekf.mean, expected_mean)
    
    def test_predict_updates_covariance(self, range_bearing_model):
        """Test covariance is updated during prediction."""
        ekf = ExtendedKalmanFilter(range_bearing_model, Sigma_0=np.eye(2))
        cov_initial = ekf.cov.copy()
        
        ekf.predict()
        
        # Covariance should change (increase due to process noise)
        assert not np.allclose(ekf.cov, cov_initial)
    
    def test_predict_maintains_symmetry(self, range_bearing_model):
        """Test predicted covariance stays symmetric."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        ekf.predict()
        
        assert np.allclose(ekf.cov, ekf.cov.T)
    
    def test_predict_uses_jacobian(self, stochastic_volatility_model):
        """Test prediction uses state Jacobian correctly."""
        model = stochastic_volatility_model
        mean_0 = np.array([0.5])
        Sigma_0 = np.array([[2.0]])
        ekf = ExtendedKalmanFilter(model, mean_0=mean_0, Sigma_0=Sigma_0)
        
        ekf.predict()
        
        # For SV model: F = alpha, Q = sigma^2
        # P' = F * P * F^T + Q
        F = model.alpha
        Q = model.sigma ** 2
        expected_cov = F * Sigma_0 * F + Q
        
        assert np.allclose(ekf.cov, expected_cov, rtol=1e-5)


# ============================================================================
# TEST 4: UPDATE STEP
# ============================================================================

class TestUpdate:
    """Test update step."""
    
    def test_update_with_observation(self, range_bearing_model):
        """Test update processes observation."""
        ekf = ExtendedKalmanFilter(range_bearing_model, mean_0=np.array([1.0, 1.0]))
        
        ekf.predict()
        cov_before = ekf.cov.copy()
        
        # Observation: [range, bearing]
        obs = np.array([1.5, 0.7])
        ekf.update(obs)
        
        # Update should reduce covariance
        assert np.trace(ekf.cov) < np.trace(cov_before)
    
    def test_update_computes_log_likelihood(self, stochastic_volatility_model):
        """Test log-likelihood is computed."""
        ekf = ExtendedKalmanFilter(stochastic_volatility_model)
        
        ekf.predict()
        ekf.update(np.array([1.0]))
        
        assert len(ekf.log_likelihoods) == 1
        assert np.isfinite(ekf.log_likelihoods[0])
    
    def test_update_maintains_symmetry(self, range_bearing_model):
        """Test updated covariance stays symmetric."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        ekf.predict()
        ekf.update(np.array([1.5, 0.5]))
        
        assert np.allclose(ekf.cov, ekf.cov.T)
    
    def test_update_uses_observation_jacobian(self, range_bearing_model):
        """Test update uses observation Jacobian."""
        mean_0 = np.array([1.0, 1.0])
        ekf = ExtendedKalmanFilter(range_bearing_model, mean_0=mean_0)
        
        ekf.predict()
        mean_before = ekf.mean.copy()
        
        ekf.update(np.array([1.5, 0.7]))
        
        # Mean should have changed
        assert not np.allclose(ekf.mean, mean_before)
    
    def test_update_with_zero_jacobian(self, stochastic_volatility_model):
        """Test update handles zero observation Jacobian (SV model)."""
        # SV model has E[y|x] = 0, so observation Jacobian is zero
        ekf = ExtendedKalmanFilter(stochastic_volatility_model)
        
        ekf.predict()
        mean_before = ekf.mean.copy()
        
        ekf.update(np.array([1.0]))
        
        # State should not change when Jacobian is zero
        assert np.allclose(ekf.mean, mean_before)


# ============================================================================
# TEST 5: FILTER METHOD (FULL SEQUENCE)
# ============================================================================

class TestFilter:
    """Test the full filter method."""
    
    def test_filter_runs_on_sequence(self, range_bearing_model):
        """Test filter processes observation sequence."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        T = 20
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ekf.filter(observations)
        
        assert result.means.shape == (T, 2)
        assert result.covs.shape == (T, 2, 2)
        assert np.isfinite(result.log_likelihood)
    
    def test_filter_output_dimensions(self, stochastic_volatility_model):
        """Test filter outputs have correct dimensions."""
        ekf = ExtendedKalmanFilter(stochastic_volatility_model)
        
        T = 15
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 1))
        result = ekf.filter(observations)
        
        assert result.means.shape == (T, 1)
        assert result.covs.shape == (T, 1, 1)
    
    def test_filter_log_likelihood_sum(self, range_bearing_model):
        """Test total log-likelihood is sum of per-step values."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        T = 10
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ekf.filter(observations)
        
        expected = np.sum(result.metadata['log_likelihoods'])
        assert np.allclose(result.log_likelihood, expected)
    
    def test_filter_metadata_complete(self, stochastic_volatility_model):
        """Test filter result has all metadata."""
        ekf = ExtendedKalmanFilter(stochastic_volatility_model)
        
        T = 5
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 1))
        result = ekf.filter(observations)
        
        assert result.metadata['filter_type'] == 'ExtendedKalmanFilter'
        assert 'log_likelihoods' in result.metadata
        assert len(result.metadata['log_likelihoods']) == T
    
    def test_filter_multiple_calls_identical(self, range_bearing_model):
        """Test multiple filter calls give same results."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((10, 2))
        
        result1 = ekf.filter(observations)
        result2 = ekf.filter(observations)
        
        assert np.allclose(result1.means, result2.means)
        assert np.allclose(result1.covs, result2.covs)
    
    def test_filter_all_finite(self, range_bearing_model):
        """Test all outputs are finite."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        T = 20
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ekf.filter(observations)
        
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)


# ============================================================================
# TEST 6: NUMERICAL STABILITY
# ============================================================================

class TestNumericalStability:
    """Test numerical stability."""
    
    def test_long_sequence_stays_stable(self, stochastic_volatility_model):
        """Test filter stays stable over long sequence."""
        ekf = ExtendedKalmanFilter(stochastic_volatility_model)
        
        T = 500
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 1))
        result = ekf.filter(observations)
        
        # Should stay finite
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
    
    def test_covariance_stays_positive_definite(self, range_bearing_model):
        """Test covariance stays positive definite."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        T = 50
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ekf.filter(observations)
        
        # All covariances should be positive definite
        for t in range(T):
            eigvals = np.linalg.eigvalsh(result.covs[t])
            assert np.all(eigvals > 0)


# ============================================================================
# TEST 7: CORRECTNESS
# ============================================================================

class TestCorrectness:
    """Test correctness of filtering."""
    
    def test_with_generated_data(self, stochastic_volatility_model):
        """Test with data generated from model."""
        model = stochastic_volatility_model
        rng = np.random.default_rng(42)
        states, observations = generate_data(model, T=30, rng=rng)
        
        ekf = ExtendedKalmanFilter(model)
        result = ekf.filter(observations)
        
        # Should produce reasonable estimates
        assert result.means.shape == states.shape
        assert np.all(np.isfinite(result.means))
    
    def test_linearization_around_current_state(self, range_bearing_model):
        """Test EKF linearizes around current state estimate."""
        ekf = ExtendedKalmanFilter(range_bearing_model, mean_0=np.array([1.0, 1.0]))
        
        # Run one predict-update cycle
        ekf.predict()
        mean_pred = ekf.mean.copy()
        
        # The Jacobian should be evaluated at mean_pred
        ekf.update(np.array([1.5, 0.7]))
        
        # Check that update was performed (mean changed)
        assert not np.allclose(ekf.mean, mean_pred)
    
    def test_update_reduces_uncertainty(self, range_bearing_model):
        """Test that updates reduce uncertainty."""
        ekf = ExtendedKalmanFilter(range_bearing_model)
        
        ekf.predict()
        cov_pred = ekf.cov.copy()
        
        ekf.update(np.array([1.5, 0.7]))
        cov_updated = ekf.cov.copy()
        
        # Update should reduce uncertainty
        assert np.trace(cov_updated) < np.trace(cov_pred)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

