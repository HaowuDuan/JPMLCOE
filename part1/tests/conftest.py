"""Shared fixtures for pytest tests."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from models import (
    LinearGaussianModel,
    StochasticVolatilityModel,
    RangeBearingModel,
    generate_data
)


@pytest.fixture
def rng():
    """Fixed random seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def linear_model(rng):
    """Small LinearGaussianModel (2D state, 1D obs)."""
    nx, ny = 2, 1
    F = np.array([[0.9, 0.1], [0.0, 0.8]])
    B = np.array([[0.5], [0.3]])
    H = np.array([[1.0, 0.0]])
    D = np.array([[0.2]])
    return LinearGaussianModel(F, B, H, D, random_state=rng)


@pytest.fixture
def linear_model_1d(rng):
    """1D LinearGaussianModel for hand-calculated verification."""
    F = np.array([[0.9]])
    B = np.array([[0.5]])
    H = np.array([[1.0]])
    D = np.array([[0.2]])
    return LinearGaussianModel(F, B, H, D, random_state=rng)


@pytest.fixture
def sv_model(rng):
    """StochasticVolatilityModel (1D)."""
    return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5, random_state=rng)


@pytest.fixture
def rb_model(rng):
    """RangeBearingModel (2D position, 2D obs)."""
    return RangeBearingModel(
        sensor_pos=np.array([0.0, 0.0]),
        sigma_range=0.1,
        sigma_bearing=0.01,
        random_state=rng
    )


@pytest.fixture
def synthetic_data(linear_model, rng):
    """Generated trajectories (T=50 timesteps)."""
    return generate_data(linear_model, T=50, random_state=rng)


@pytest.fixture
def simple_observations(linear_model):
    """Minimal observation sequence for quick tests."""
    T = 10
    observations = np.zeros((T, linear_model.obs_dim))
    for t in range(T):
        observations[t] = np.array([t * 0.1])
    return observations
