"""
Shared fixtures and helpers for particle filter correctness tests.

Used by:
  test_resampling.py    — resampling methods in isolation
  test_bootstrap_pf.py  — bootstrap PF correctness and diagnostics
  test_flow_filters.py  — flow filter correctness and diagnostics

Provides:
  - Linear-Gaussian model + Kalman ground truth
  - Diagnostic assertion helpers
"""

import pytest
import numpy as np
import tensorflow as tf

from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data
from src.filters.kalman.kalman import KalmanFilter


# ============================================================================
# CONSTANTS
# ============================================================================

DTYPE = tf.float64
NP_DTYPE = np.float64

# 2D linear-Gaussian model for gold-standard Kalman comparison
# x_t = F @ x_{t-1} + B @ v_t,  v_t ~ N(0, I)
# y_t = H @ x_t + D @ w_t,      w_t ~ N(0, I)
LG_F = [[0.9, 0.1], [0.0, 0.95]]   # stable 2D dynamics
LG_B = [[1.0, 0.0], [0.0, 0.5]]    # process noise
LG_H = [[1.0, 0.0]]                 # observe first state only
LG_D = [[1.0]]                      # observation noise

T = 50
SEED = 42


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def lg_model():
    """2D linear-Gaussian model."""
    return LinearGaussianModel(
        F=LG_F, B=LG_B, H=LG_H, D=LG_D, dtype=DTYPE,
    )


@pytest.fixture(scope="session")
def lg_data(lg_model):
    """Generate data from the 2D linear-Gaussian model."""
    rng = np.random.default_rng(SEED)
    initial_state, true_states, observations = generate_data(lg_model, T=T, rng=rng)
    return initial_state, true_states, observations


@pytest.fixture(scope="session")
def kf_result(lg_model, lg_data):
    """Kalman filter ground truth on the 2D linear-Gaussian data."""
    _, _, observations = lg_data
    kf = KalmanFilter(
        F=LG_F, B=LG_B, H=LG_H, D=LG_D, dtype=DTYPE,
    )
    return kf.filter(observations)


@pytest.fixture(scope="session")
def lg_observations_tf(lg_data):
    """Observations as float64 TF tensor."""
    _, _, observations = lg_data
    return tf.constant(observations, dtype=DTYPE)


# ============================================================================
# DIAGNOSTIC HELPERS
# ============================================================================

def assert_no_nan_inf(arr, name="array"):
    """Assert that an array contains no NaN or Inf values."""
    arr = np.asarray(arr)
    assert np.all(np.isfinite(arr)), (
        f"{name} contains non-finite values: "
        f"NaN count={np.sum(np.isnan(arr))}, Inf count={np.sum(np.isinf(arr))}"
    )


def assert_weights_valid(weights, name="weights"):
    """Assert that weights are non-negative and sum to ~1."""
    weights = np.asarray(weights)
    assert_no_nan_inf(weights, name)
    assert np.all(weights >= -1e-10), f"{name} has negative values: min={weights.min()}"
    total = weights.sum()
    assert abs(total - 1.0) < 1e-4, f"{name} sum={total}, expected 1.0"


def compute_ess(weights):
    """Compute effective sample size from normalized weights."""
    weights = np.asarray(weights)
    return 1.0 / np.sum(weights ** 2)
