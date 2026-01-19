"""Smoke tests for Kalman filters."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.linear_gaussian import LinearGaussianModel
from src.models.stochastic_volatility import StochasticVolatilityModel
from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.kalman.kalman import KalmanFilter
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
from src.filters.kalman.unscented_kalman import UnscentedKalmanFilter


def test_kalman_filter():
    """Test KalmanFilter runs without error."""
    # Create model
    F = np.eye(2)
    B = 0.1 * np.eye(2)
    H = np.array([[1.0, 0.0]])
    D = 0.1 * np.eye(1)
    model = LinearGaussianModel(F, B, H, D)

    # Generate data
    rng = np.random.default_rng(42)
    states, observations = generate_data(model, T=20, rng=rng)

    # Create and run filter
    kf = KalmanFilter(F, B, H, D, Sigma=np.eye(2))
    result = kf.filter(observations)

    assert result.means.shape == (20, 2)
    assert result.covs.shape == (20, 2, 2)
    assert np.isfinite(result.log_likelihood)
    assert np.all(np.isfinite(result.means))
    assert np.all(np.isfinite(result.covs))


def test_extended_kalman_filter():
    """Test ExtendedKalmanFilter runs without error."""
    # Use StochasticVolatilityModel (nonlinear)
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)

    # Generate data
    rng = np.random.default_rng(42)
    states, observations = generate_data(model, T=20, rng=rng)

    # Create and run filter
    ekf = ExtendedKalmanFilter(model)
    ekf.initialize(mean=np.zeros(1), cov=np.eye(1))
    result = ekf.filter(observations)

    assert result.means.shape == (20, 1)
    assert result.covs.shape == (20, 1, 1)
    assert np.isfinite(result.log_likelihood)
    assert np.all(np.isfinite(result.means))


def test_unscented_kalman_filter():
    """Test UnscentedKalmanFilter runs without error."""
    # Use RangeBearingModel (nonlinear)
    model = RangeBearingModel()

    # Generate data
    rng = np.random.default_rng(42)
    states, observations = generate_data(model, T=20, rng=rng)

    # Create and run filter
    ukf = UnscentedKalmanFilter(model, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf.initialize(mean=np.array([1.0, 1.0]), cov=np.eye(2))
    result = ukf.filter(observations)

    assert result.means.shape == (20, 2)
    assert result.covs.shape == (20, 2, 2)
    assert np.isfinite(result.log_likelihood)
    assert np.all(np.isfinite(result.means))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
