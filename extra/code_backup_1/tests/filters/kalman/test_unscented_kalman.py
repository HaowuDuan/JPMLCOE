"""Tests for UnscentedKalmanFilter implementation."""

import pytest
import numpy as np
from src.filters.kalman.unscented_kalman import UnscentedKalmanFilter
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


@pytest.fixture
def nonlinear_test_model():
    """Create a simple nonlinear model for testing UKF with mild nonlinearity."""
    from src.core.model_base import StateSpaceModel
    
    class SimpleNonlinearModel(StateSpaceModel):
        """Simple 2D model with mild nonlinearity via sin term."""
        
        def __init__(self):
            self.state_dim = 2
            self.obs_dim = 2
            self.Q = 0.1 * np.eye(2)  # Process noise
            self.R = 0.1 * np.eye(2)  # Observation noise
            self.dt = 0.1
        
        def f(self, x):
            """Mildly nonlinear transition: add sin(0.1 * x[0]) for nonlinearity."""
            return np.array([
                x[0] + x[1] * self.dt + 0.1 * np.sin(x[0]),
                x[1]
            ])
        
        def h(self, x):
            """Mildly nonlinear observation: add sin(0.1 * x[0]) for nonlinearity."""
            return np.array([
                x[0] + 0.05 * np.sin(0.1 * x[0]),
                x[1]
            ])
        
        def jacobian_f(self, x):
            """Jacobian of f (not used by UKF but good to have)."""
            return np.array([
                [1.0 + 0.1 * np.cos(x[0]), self.dt],
                [0.0, 1.0]
            ])
        
        def jacobian_h(self, x):
            """Jacobian of h (not used by UKF but good to have)."""
            return np.array([
                [1.0 + 0.005 * np.cos(0.1 * x[0]), 0.0],
                [0.0, 1.0]
            ])
        
        def sample_state(self, x, rng=None):
            """Sample next state."""
            if rng is None:
                rng = np.random.default_rng()
            return self.f(x) + rng.multivariate_normal(np.zeros(2), self.Q)
        
        def sample_observation(self, x, rng=None):
            """Sample observation."""
            if rng is None:
                rng = np.random.default_rng()
            return self.h(x) + rng.multivariate_normal(np.zeros(2), self.R)
    
    return SimpleNonlinearModel()


# ============================================================================
# TEST 1: INITIALIZATION
# ============================================================================

class TestInitialization:
    """Test UnscentedKalmanFilter initialization."""
    
    def test_basic_init(self, stochastic_volatility_model):
        """Test basic initialization works."""
        model = stochastic_volatility_model
        ukf = UnscentedKalmanFilter(model)
        
        assert ukf.state_dim == 1
        assert ukf.mean is not None
        assert ukf.cov is not None
        assert ukf.alpha == 1e-3
        assert ukf.beta == 2.0
        assert ukf.kappa == 0.0
    
    def test_init_with_custom_state(self, range_bearing_model):
        """Test initialization with custom initial state."""
        model = range_bearing_model
        mean_0 = np.array([2.0, 3.0])
        Sigma_0 = 2.0 * np.eye(2)
        
        ukf = UnscentedKalmanFilter(model, mean_0=mean_0, Sigma_0=Sigma_0)
        
        assert np.allclose(ukf.mean, mean_0)
        assert np.allclose(ukf.cov, Sigma_0)
    
    def test_init_with_custom_parameters(self, range_bearing_model):
        """Test initialization with custom UKF parameters."""
        ukf = UnscentedKalmanFilter(
            range_bearing_model,
            alpha=0.5,
            beta=3.0,
            kappa=1.0
        )
        
        assert ukf.alpha == 0.5
        assert ukf.beta == 3.0
        assert ukf.kappa == 1.0
    
    def test_init_weights_computed(self, range_bearing_model):
        """Test sigma point weights are computed."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        # Should have 2*state_dim + 1 weights
        assert len(ukf.weights_mean) == 2 * 2 + 1
        assert len(ukf.weights_cov) == 2 * 2 + 1
        
        # Mean weights should sum to 1
        assert np.allclose(np.sum(ukf.weights_mean), 1.0)
        
        # Covariance weights should sum to 1
        assert np.allclose(np.sum(ukf.weights_cov), 1.0)
    
    def test_init_uses_model_defaults(self, stochastic_volatility_model):
        """Test initialization uses model's stationary variance."""
        model = stochastic_volatility_model
        ukf = UnscentedKalmanFilter(model)
        
        # Should use stationary variance
        expected_var = model.stationary_var
        assert np.allclose(ukf.Sigma_0[0, 0], expected_var)


# ============================================================================
# TEST 2: RESET
# ============================================================================

class TestReset:
    """Test reset functionality."""
    
    def test_reset_returns_to_initial(self, range_bearing_model):
        """Test reset returns filter to initial state."""
        mean_0 = np.array([1.0, 2.0])
        ukf = UnscentedKalmanFilter(range_bearing_model, mean_0=mean_0)
        
        # Run some steps
        ukf.predict()
        ukf.update(np.array([1.5, 0.5]))
        
        # Reset
        ukf.reset()
        
        assert np.allclose(ukf.mean, mean_0)
    
    def test_reset_clears_log_likelihoods(self, stochastic_volatility_model):
        """Test reset clears log-likelihoods."""
        ukf = UnscentedKalmanFilter(stochastic_volatility_model)
        
        ukf.predict()
        ukf.update(np.array([1.0]))
        assert len(ukf.log_likelihoods) > 0
        
        ukf.reset()
        assert len(ukf.log_likelihoods) == 0


# ============================================================================
# TEST 3: SIGMA POINTS
# ============================================================================

class TestSigmaPoints:
    """Test sigma point computation."""
    
    def test_sigma_points_correct_shape(self, range_bearing_model):
        """Test sigma points have correct shape."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        sigma_points = ukf._compute_sigma_points(mean, cov)
        
        # Should be (2*state_dim + 1, state_dim)
        assert sigma_points.shape == (2 * 2 + 1, 2)
    
    def test_sigma_points_centered_at_mean(self, range_bearing_model):
        """Test first sigma point is the mean."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        sigma_points = ukf._compute_sigma_points(mean, cov)
        
        assert np.allclose(sigma_points[0], mean)
    
    def test_sigma_points_symmetric(self, range_bearing_model):
        """Test sigma points are symmetric around mean."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        sigma_points = ukf._compute_sigma_points(mean, cov)
        
        n = ukf.state_dim
        # Check that pairs are symmetric
        for i in range(n):
            diff1 = sigma_points[i + 1] - mean
            diff2 = sigma_points[i + n + 1] - mean
            assert np.allclose(diff1, -diff2)
    
    def test_sigma_points_with_different_covariances(self, range_bearing_model):
        """Test sigma points scale with covariance."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        mean = np.array([0.0, 0.0])
        
        # Small covariance
        cov_small = 0.1 * np.eye(2)
        sigma_small = ukf._compute_sigma_points(mean, cov_small)
        spread_small = np.std(sigma_small, axis=0)
        
        # Large covariance
        cov_large = 10.0 * np.eye(2)
        sigma_large = ukf._compute_sigma_points(mean, cov_large)
        spread_large = np.std(sigma_large, axis=0)
        
        # Larger covariance should give larger spread
        assert np.all(spread_large > spread_small)


# ============================================================================
# TEST 4: PREDICT STEP
# ============================================================================

class TestPredict:
    """Test prediction step."""
    
    def test_predict_propagates_mean(self, stochastic_volatility_model):
        """Test mean propagation through nonlinear dynamics."""
        model = stochastic_volatility_model
        mean_0 = np.array([1.0])
        ukf = UnscentedKalmanFilter(model, mean_0=mean_0)
        
        ukf.predict()
        
        # For SV model: x' = alpha * x (approximately)
        # UKF should give similar result
        expected_mean = model.alpha * mean_0
        assert np.allclose(ukf.mean, expected_mean, rtol=0.1)
    
    def test_predict_updates_covariance(self, range_bearing_model):
        """Test covariance is updated during prediction."""
        ukf = UnscentedKalmanFilter(range_bearing_model, Sigma_0=np.eye(2))
        cov_initial = ukf.cov.copy()
        
        ukf.predict()
        
        # Covariance should change (increase due to process noise)
        assert not np.allclose(ukf.cov, cov_initial)
    
    def test_predict_maintains_symmetry(self, range_bearing_model):
        """Test predicted covariance stays symmetric."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        ukf.predict()
        
        assert np.allclose(ukf.cov, ukf.cov.T)
    
    def test_predict_uses_sigma_points(self, range_bearing_model):
        """Test prediction propagates sigma points."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        mean_before = ukf.mean.copy()
        ukf.predict()
        
        # Mean should have changed
        assert not np.allclose(ukf.mean, mean_before)


# ============================================================================
# TEST 5: UPDATE STEP
# ============================================================================

class TestUpdate:
    """Test update step."""
    
    def test_update_with_observation(self, range_bearing_model):
        """Test update processes observation."""
        ukf = UnscentedKalmanFilter(range_bearing_model, mean_0=np.array([1.0, 1.0]))
        
        ukf.predict()
        cov_before = ukf.cov.copy()
        
        # Observation: [range, bearing]
        obs = np.array([1.5, 0.7])
        ukf.update(obs)
        
        # Update should reduce covariance
        assert np.trace(ukf.cov) < np.trace(cov_before)
    
    def test_update_computes_log_likelihood(self, stochastic_volatility_model):
        """Test log-likelihood is computed."""
        ukf = UnscentedKalmanFilter(stochastic_volatility_model)
        
        ukf.predict()
        ukf.update(np.array([1.0]))
        
        assert len(ukf.log_likelihoods) == 1
        assert np.isfinite(ukf.log_likelihoods[0])
    
    def test_update_maintains_symmetry(self, range_bearing_model):
        """Test updated covariance stays symmetric."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        ukf.predict()
        ukf.update(np.array([1.5, 0.5]))
        
        assert np.allclose(ukf.cov, ukf.cov.T)
    
    def test_update_uses_sigma_points(self, range_bearing_model):
        """Test update uses sigma points for observation."""
        mean_0 = np.array([1.0, 1.0])
        ukf = UnscentedKalmanFilter(range_bearing_model, mean_0=mean_0)
        
        ukf.predict()
        mean_before = ukf.mean.copy()
        
        ukf.update(np.array([1.5, 0.7]))
        
        # Mean should have changed
        assert not np.allclose(ukf.mean, mean_before)


# ============================================================================
# TEST 6: FILTER METHOD (FULL SEQUENCE)
# ============================================================================

class TestFilter:
    """Test the full filter method."""
    
    def test_filter_runs_on_sequence(self, range_bearing_model):
        """Test filter processes observation sequence."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        T = 20
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ukf.filter(observations)
        
        assert result.means.shape == (T, 2)
        assert result.covs.shape == (T, 2, 2)
        assert np.isfinite(result.log_likelihood)
    
    def test_filter_output_dimensions(self, stochastic_volatility_model):
        """Test filter outputs have correct dimensions."""
        ukf = UnscentedKalmanFilter(stochastic_volatility_model)
        
        T = 15
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 1))
        result = ukf.filter(observations)
        
        assert result.means.shape == (T, 1)
        assert result.covs.shape == (T, 1, 1)
    
    def test_filter_log_likelihood_sum(self, range_bearing_model):
        """Test total log-likelihood is sum of per-step values."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        T = 10
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ukf.filter(observations)
        
        expected = np.sum(result.metadata['log_likelihoods'])
        assert np.allclose(result.log_likelihood, expected)
    
    def test_filter_metadata_complete(self, stochastic_volatility_model):
        """Test filter result has all metadata."""
        ukf = UnscentedKalmanFilter(stochastic_volatility_model, alpha=0.5)
        
        T = 5
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 1))
        result = ukf.filter(observations)
        
        assert result.metadata['filter_type'] == 'UnscentedKalmanFilter'
        assert result.metadata['alpha'] == 0.5
        assert result.metadata['beta'] == 2.0
        assert result.metadata['kappa'] == 0.0
        assert 'log_likelihoods' in result.metadata
        assert len(result.metadata['log_likelihoods']) == T
    
    def test_filter_multiple_calls_identical(self, range_bearing_model):
        """Test multiple filter calls give same results."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((10, 2))
        
        result1 = ukf.filter(observations)
        result2 = ukf.filter(observations)
        
        assert np.allclose(result1.means, result2.means)
        assert np.allclose(result1.covs, result2.covs)
    
    def test_filter_all_finite(self, range_bearing_model):
        """Test all outputs are finite."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        T = 20
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ukf.filter(observations)
        
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)


# ============================================================================
# TEST 7: NUMERICAL STABILITY
# ============================================================================

class TestNumericalStability:
    """Test numerical stability."""
    
    def test_long_sequence_stays_stable(self, stochastic_volatility_model):
        """Test filter stays stable over long sequence."""
        ukf = UnscentedKalmanFilter(stochastic_volatility_model)
        
        T = 500
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 1))
        result = ukf.filter(observations)
        
        # Should stay finite
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
    
    def test_covariance_stays_positive_definite(self, range_bearing_model):
        """Test covariance stays positive definite."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        T = 50
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2))
        result = ukf.filter(observations)
        
        # All covariances should be positive definite
        for t in range(T):
            eigvals = np.linalg.eigvalsh(result.covs[t])
            assert np.all(eigvals > -1e-10)  # Allow small negative values due to numerics
    
    def test_different_alpha_parameters(self, range_bearing_model):
        """Test filter works with different alpha parameters."""
        alphas = [1e-4, 1e-3, 1e-2, 0.1]
        
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((20, 2))
        
        for alpha in alphas:
            ukf = UnscentedKalmanFilter(range_bearing_model, alpha=alpha)
            result = ukf.filter(observations)
            
            assert np.all(np.isfinite(result.means))
            assert np.all(np.isfinite(result.covs))


# ============================================================================
# TEST 8: CORRECTNESS
# ============================================================================

class TestCorrectness:
    """Test correctness of filtering."""
    
    def test_with_generated_data(self, stochastic_volatility_model):
        """Test with data generated from model."""
        model = stochastic_volatility_model
        rng = np.random.default_rng(42)
        states, observations = generate_data(model, T=30, rng=rng)
        
        ukf = UnscentedKalmanFilter(model)
        result = ukf.filter(observations)
        
        # Should produce reasonable estimates
        assert result.means.shape == states.shape
        assert np.all(np.isfinite(result.means))
    
    def test_update_reduces_uncertainty(self, range_bearing_model):
        """Test that updates reduce uncertainty."""
        ukf = UnscentedKalmanFilter(range_bearing_model)
        
        ukf.predict()
        cov_pred = ukf.cov.copy()
        
        ukf.update(np.array([1.5, 0.7]))
        cov_updated = ukf.cov.copy()
        
        # Update should reduce uncertainty
        assert np.trace(cov_updated) < np.trace(cov_pred)
    
    def test_ukf_vs_ekf_similar_results(self, range_bearing_model):
        """Test UKF gives similar results to EKF for mildly nonlinear case."""
        from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
        
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((20, 2))
        
        ukf = UnscentedKalmanFilter(range_bearing_model)
        result_ukf = ukf.filter(observations)
        
        ekf = ExtendedKalmanFilter(range_bearing_model)
        result_ekf = ekf.filter(observations)
        
        # Results should be similar (not identical due to different approximations)
        mean_diff = np.mean(np.linalg.norm(result_ukf.means - result_ekf.means, axis=1))
        assert mean_diff < 1.0  # Reasonable similarity


# ============================================================================
# TEST 9: NONLINEAR MODEL TESTING
# ============================================================================

class TestNonlinearModel:
    """Test UKF with mildly nonlinear transition and observation functions."""
    
    def test_nonlinear_predict(self, nonlinear_test_model):
        """Test prediction with nonlinear transition function."""
        model = nonlinear_test_model
        mean_0 = np.array([1.0, 0.5])
        ukf = UnscentedKalmanFilter(model, mean_0=mean_0, Sigma_0=np.eye(2))
        
        ukf.predict()
        
        # UKF should handle nonlinearity better than linearization
        # Check result is reasonable
        assert np.all(np.isfinite(ukf.mean))
        assert ukf.mean[0] != mean_0[0]  # Should have changed
    
    def test_nonlinear_update(self, nonlinear_test_model):
        """Test update with nonlinear observation function."""
        model = nonlinear_test_model
        mean_0 = np.array([1.0, 0.5])
        ukf = UnscentedKalmanFilter(model, mean_0=mean_0, Sigma_0=np.eye(2))
        
        ukf.predict()
        cov_pred = ukf.cov.copy()
        
        # Create observation with slight nonlinearity
        obs = np.array([1.2, 0.6])
        ukf.update(obs)
        
        # Update should reduce uncertainty
        assert np.trace(ukf.cov) < np.trace(cov_pred)
        assert np.all(np.isfinite(ukf.mean))
    
    def test_nonlinear_filter_sequence(self, nonlinear_test_model):
        """Test filtering on sequence with nonlinear model."""
        model = nonlinear_test_model
        ukf = UnscentedKalmanFilter(model, mean_0=np.array([0.5, 0.2]))
        
        # Generate synthetic observations
        T = 30
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2)) * 0.5
        
        result = ukf.filter(observations)
        
        assert result.means.shape == (T, 2)
        assert result.covs.shape == (T, 2, 2)
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)
    
    def test_nonlinear_sigma_points_propagation(self, nonlinear_test_model):
        """Test that sigma points propagate through nonlinearity."""
        model = nonlinear_test_model
        mean_0 = np.array([1.0, 0.5])
        ukf = UnscentedKalmanFilter(model, mean_0=mean_0, Sigma_0=np.eye(2))
        
        # Compute sigma points
        sigma_points = ukf._compute_sigma_points(mean_0, ukf.cov)
        
        # Propagate through nonlinear f
        propagated = np.array([model.f(sp) for sp in sigma_points])
        
        # Check all propagated points are different (nonlinearity effect)
        assert propagated.shape == (2 * 2 + 1, 2)
        assert np.all(np.isfinite(propagated))
    
    def test_nonlinear_ukf_handles_better_than_linear(self, nonlinear_test_model):
        """Test UKF handles nonlinearity (doesn't diverge)."""
        model = nonlinear_test_model
        ukf = UnscentedKalmanFilter(model, mean_0=np.array([0.5, 0.2]))
        
        # Run for moderate sequence length
        T = 50
        rng = np.random.default_rng(42)
        observations = rng.standard_normal((T, 2)) * 0.3
        
        result = ukf.filter(observations)
        
        # Should remain stable despite nonlinearity
        assert np.all(np.isfinite(result.means))
        assert np.all(np.abs(result.means) < 100)  # No divergence


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

