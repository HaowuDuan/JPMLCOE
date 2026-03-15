"""Tests for KalmanFilter implementation."""

import pytest
import numpy as np
from src.filters.kalman.kalman import KalmanFilter
from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data


# ============================================================================
# FIXTURES - Reusable test setup
# ============================================================================

@pytest.fixture
def basic_2d_system():
    """Create a basic 2D Kalman filter system."""
    F = np.eye(2)
    B = 0.1 * np.eye(2)
    H = np.array([[1.0, 0.0]])
    D = 0.1 * np.eye(1)
    return F, B, H, D


# ============================================================================
# TEST 1: INITIALIZATION
# ============================================================================

class TestInitialization:
    """Test KalmanFilter initialization."""
    
    def test_basic_init(self, basic_2d_system):
        """Test basic initialization works."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        assert kf.nx == 2
        assert kf.ny == 1
        assert np.allclose(kf.mean, np.zeros(2))
        assert np.allclose(kf.cov, np.eye(2))
    
    def test_init_with_custom_state(self):
        """Test initialization with custom initial state."""
        F = np.eye(2)
        B = 0.1 * np.eye(2)
        H = np.eye(2)
        D = 0.1 * np.eye(2)
        mean_0 = np.array([1.0, 2.0])
        Sigma_0 = 2.0 * np.eye(2)
        
        kf = KalmanFilter(F, B, H, D, mean_0=mean_0, Sigma_0=Sigma_0)
        
        assert np.allclose(kf.mean, mean_0)
        assert np.allclose(kf.cov, Sigma_0)
    
    def test_init_from_lists(self):
        """Test initialization from Python lists (for Hydra configs)."""
        F = [[1.0, 0.0], [0.0, 1.0]]
        B = [[0.1, 0.0], [0.0, 0.1]]
        H = [[1.0, 0.0]]
        D = [[0.1]]
        
        kf = KalmanFilter(F, B, H, D)
        
        assert isinstance(kf.F, np.ndarray)
        assert kf.F.dtype == np.float64
    
    def test_covariance_matrices_computed(self):
        """Test Q and R are computed correctly."""
        B = np.array([[0.1, 0.0], [0.0, 0.2]])
        D = np.array([[0.3, 0.0], [0.0, 0.4]])
        
        kf = KalmanFilter(F=np.eye(2), B=B, H=np.eye(2), D=D)
        
        assert np.allclose(kf.Q, B @ B.T)
        assert np.allclose(kf.R, D @ D.T)


# ============================================================================
# TEST 2: RESET
# ============================================================================

class TestReset:
    """Test reset functionality."""
    
    def test_reset_returns_to_initial(self, basic_2d_system):
        """Test reset returns filter to initial state."""
        F, B, H, D = basic_2d_system
        mean_0 = np.array([1.0, 2.0])
        kf = KalmanFilter(F, B, H, D, mean_0=mean_0)
        
        # Run some steps
        kf.predict()
        kf.update(np.array([1.5]))
        
        # Reset
        kf.reset()
        
        assert np.allclose(kf.mean, mean_0)
    
    def test_reset_clears_history(self, basic_2d_system):
        """Test reset clears history."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        kf.predict()
        kf.update(np.array([1.0]))
        assert len(kf.history['mean_pred']) > 0
        
        kf.reset()
        assert len(kf.history['mean_pred']) == 0


# ============================================================================
# TEST 3: PREDICT STEP
# ============================================================================

class TestPredict:
    """Test prediction step."""
    
    def test_predict_propagates_mean(self):
        """Test mean propagation: x = F @ x."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = 0.1 * np.eye(2)
        mean_0 = np.array([1.0, 2.0])
        
        kf = KalmanFilter(F, B, np.eye(2), 0.1*np.eye(2), mean_0=mean_0)
        mean_pred, _ = kf.predict()
        
        assert np.allclose(mean_pred, F @ mean_0)
    
    def test_predict_propagates_covariance(self):
        """Test covariance propagation: P = F P F^T + Q."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.1, 0.0], [0.0, 0.2]])
        Sigma_0 = 2.0 * np.eye(2)
        
        kf = KalmanFilter(F, B, np.eye(2), 0.1*np.eye(2), Sigma_0=Sigma_0)
        _, cov_pred = kf.predict()
        
        Q = B @ B.T
        expected = F @ Sigma_0 @ F.T + Q
        assert np.allclose(cov_pred, expected)
    
    def test_predict_maintains_symmetry(self):
        """Test predicted covariance stays symmetric."""
        F = np.array([[1.0, 0.5], [0.0, 1.0]])
        kf = KalmanFilter(F, 0.1*np.eye(2), np.eye(2), 0.1*np.eye(2))
        
        _, cov = kf.predict()
        
        assert np.allclose(cov, cov.T)
    
    def test_predict_stores_history(self, basic_2d_system):
        """Test prediction stores history."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        kf.predict()
        
        assert len(kf.history['mean_pred']) == 1
        assert len(kf.history['cov_pred']) == 1


# ============================================================================
# TEST 4: UPDATE STEP
# ============================================================================

class TestUpdate:
    """Test update step."""
    
    def test_update_with_clean_observation(self):
        """Test update pulls estimate toward clean observation."""
        kf = KalmanFilter(
            F=np.eye(2),
            B=0.1 * np.eye(2),
            H=np.eye(2),
            D=0.01 * np.eye(2)  # Very clean
        )
        
        kf.predict()
        obs = np.array([1.0, 2.0])
        mean_updated, _, _ = kf.update(obs)
        
        # Should be close to observation
        assert np.allclose(mean_updated, obs, atol=0.1)
    
    def test_update_kalman_gain(self):
        """Test Kalman gain behaves correctly."""
        # Clean observation should give high gain
        kf1 = KalmanFilter(F=np.eye(1), B=0.1*np.eye(1), H=np.eye(1), D=0.01*np.eye(1))
        kf1.predict()
        _, _, K1 = kf1.update(np.array([5.0]))
        
        # Noisy observation should give low gain
        kf2 = KalmanFilter(F=np.eye(1), B=0.1*np.eye(1), H=np.eye(1), D=10.0*np.eye(1))
        kf2.predict()
        _, _, K2 = kf2.update(np.array([5.0]))
        
        assert K1[0, 0] > K2[0, 0]
    
    def test_update_joseph_vs_standard(self):
        """Test Joseph form matches standard form."""
        F = np.eye(2)
        B = 0.1 * np.eye(2)
        H = np.array([[1.0, 0.0]])
        D = 0.1 * np.eye(1)
        obs = np.array([1.5])
        
        kf_joseph = KalmanFilter(F, B, H, D, use_joseph_form=True)
        kf_joseph.predict()
        mean_j, cov_j, _ = kf_joseph.update(obs)
        
        kf_standard = KalmanFilter(F, B, H, D, use_joseph_form=False)
        kf_standard.predict()
        mean_s, cov_s, _ = kf_standard.update(obs)
        
        assert np.allclose(mean_j, mean_s)
        assert np.allclose(cov_j, cov_s, rtol=1e-6)
    
    def test_update_maintains_symmetry(self):
        """Test updated covariance stays symmetric."""
        kf = KalmanFilter(
            F=np.eye(2),
            B=0.1 * np.eye(2),
            H=np.array([[1.0, 0.5]]),
            D=0.1 * np.eye(1)
        )
        
        kf.predict()
        _, cov, _ = kf.update(np.array([1.0]))
        
        assert np.allclose(cov, cov.T)
    
    def test_update_computes_log_likelihood(self, basic_2d_system):
        """Test log-likelihood is computed."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        kf.predict()
        kf.update(np.array([1.0]))
        
        log_lik = kf.history['log_likelihoods'][0]
        assert np.isfinite(log_lik)
        assert log_lik < 0  # Log probability should be negative
    
    def test_update_stores_history(self, basic_2d_system):
        """Test update stores all history."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        kf.predict()
        kf.update(np.array([1.0]))
        
        assert len(kf.history['mean_updated']) == 1
        assert len(kf.history['kalman_gain']) == 1
        assert len(kf.history['innovations']) == 1
        assert len(kf.history['log_likelihoods']) == 1


# ============================================================================
# TEST 5: FILTER METHOD (FULL SEQUENCE)
# ============================================================================

class TestFilter:
    """Test the full filter method."""
    
    def test_filter_runs_on_sequence(self, basic_2d_system):
        """Test filter processes observation sequence."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        T = 20
        observations = np.random.randn(T, 1)
        result = kf.filter(observations)
        
        assert result.means.shape == (T, 2)
        assert result.covs.shape == (T, 2, 2)
        assert np.isfinite(result.log_likelihood)
    
    def test_filter_output_dimensions(self):
        """Test filter outputs have correct dimensions."""
        nx, ny, T = 3, 2, 15
        F = np.eye(nx)
        B = 0.1 * np.eye(nx)
        H = np.random.randn(ny, nx)
        D = 0.1 * np.eye(ny)
        
        observations = np.random.randn(T, ny)
        kf = KalmanFilter(F, B, H, D)
        result = kf.filter(observations)
        
        assert result.means.shape == (T, nx)
        assert result.covs.shape == (T, nx, nx)
    
    def test_filter_log_likelihood_sum(self, basic_2d_system):
        """Test total log-likelihood is sum of per-step values."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        observations = np.random.randn(10, 1)
        result = kf.filter(observations)
        
        expected = np.sum(result.metadata['log_likelihoods'])
        assert np.allclose(result.log_likelihood, expected)
    
    def test_filter_metadata_complete(self, basic_2d_system):
        """Test filter result has all metadata."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D, use_joseph_form=True)
        
        T = 5
        observations = np.random.randn(T, 1)
        result = kf.filter(observations)
        
        assert result.metadata['filter_type'] == 'KalmanFilter'
        assert result.metadata['use_joseph_form'] is True
        assert 'kalman_gains' in result.metadata
        assert 'innovations' in result.metadata
        assert 'condition_numbers' in result.metadata
    
    def test_filter_multiple_calls_identical(self, basic_2d_system):
        """Test multiple filter calls give same results."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        observations = np.random.randn(10, 1)
        result1 = kf.filter(observations)
        result2 = kf.filter(observations)
        
        assert np.allclose(result1.means, result2.means)
        assert np.allclose(result1.covs, result2.covs)
    
    def test_filter_all_finite(self, basic_2d_system):
        """Test all outputs are finite."""
        F, B, H, D = basic_2d_system
        kf = KalmanFilter(F, B, H, D)
        
        observations = np.random.randn(20, 1)
        result = kf.filter(observations)
        
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)


# ============================================================================
# TEST 6: NUMERICAL STABILITY
# ============================================================================

class TestNumericalStability:
    """Test numerical stability."""
    
    def test_joseph_form_long_sequence(self):
        """Test Joseph form stays stable over long sequence."""
        kf = KalmanFilter(
            F=np.array([[1.0, 0.1], [0.0, 1.0]]),
            B=0.01 * np.eye(2),
            H=np.array([[1.0, 0.0]]),
            D=0.1 * np.eye(1),
            use_joseph_form=True
        )
        
        T = 500
        observations = np.random.randn(T, 1)
        result = kf.filter(observations)
        
        # Should stay finite
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
    
    def test_covariance_stays_positive_definite(self):
        """Test covariance stays positive definite."""
        kf = KalmanFilter(
            F=np.eye(2),
            B=0.1 * np.eye(2),
            H=np.array([[1.0, 0.0]]),
            D=0.1 * np.eye(1)
        )
        
        observations = np.random.randn(50, 1)
        result = kf.filter(observations)
        
        # All covariances should be positive definite
        for t in range(50):
            eigvals = np.linalg.eigvalsh(result.covs[t])
            assert np.all(eigvals > 0)


# ============================================================================
# TEST 7: CORRECTNESS
# ============================================================================

class TestCorrectness:
    """Test correctness of filtering."""
    
    def test_update_reduces_uncertainty(self):
        """Test that updates reduce uncertainty."""
        kf = KalmanFilter(
            F=np.eye(2),
            B=0.5 * np.eye(2),
            H=np.array([[1.0, 0.0]]),
            D=0.1 * np.eye(1)
        )
        
        kf.predict()
        cov_pred = kf.cov.copy()
        
        kf.update(np.array([1.0]))
        cov_updated = kf.cov.copy()
        
        # Update should reduce uncertainty
        assert np.trace(cov_updated) < np.trace(cov_pred)
    
    def test_predict_increases_uncertainty(self):
        """Test that prediction increases uncertainty."""
        kf = KalmanFilter(
            F=np.eye(2),
            B=0.5 * np.eye(2),
            H=np.eye(2),
            D=0.1 * np.eye(2)
        )
        
        cov_initial = kf.cov.copy()
        kf.predict()
        cov_pred = kf.cov.copy()
        
        # Prediction should increase uncertainty
        assert np.trace(cov_pred) > np.trace(cov_initial)
    
    def test_with_generated_data(self):
        """Test with data generated from LinearGaussianModel."""
        F = np.array([[0.9, 0.1], [0.0, 0.9]])
        B = 0.1 * np.eye(2)
        H = np.array([[1.0, 0.0]])
        D = 0.2 * np.eye(1)
        
        model = LinearGaussianModel(F, B, H, D)
        rng = np.random.default_rng(42)
        states, observations = generate_data(model, T=30, rng=rng)
        
        kf = KalmanFilter(F, B, H, D)
        result = kf.filter(observations)
        
        # Should track reasonably well
        errors = np.linalg.norm(result.means - states, axis=1)
        mean_error = np.mean(errors)
        assert mean_error < 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

