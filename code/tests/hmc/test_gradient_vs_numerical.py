"""Gradient validation: LEDH+OT autodiff vs paired central-difference numerical gradient.

Tests that tf.GradientTape gradient through the LEDH+OT filter likelihood
matches numerical gradient estimated via paired central differences at
multiple radii. Single parameter (sigma2) tested at multiple evaluation
points across the parameter space.

Uses the compiled path (eager_mode=False) with DifferentiableModel, matching
the production HMC pipeline.

Run: python -m pytest tests/hmc/test_gradient_vs_numerical.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel


# ============================================================================
# Constants
# ============================================================================

DTYPE = tf.float64
N_PARTICLES = 200
T = 20
N_LAMBDA_STEPS = 15
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42

M_RADII = 10           # number of paired radii
H_MAX = 0.05           # max perturbation
REL_TOL = 0.30          # 30% relative error
ABS_TOL = 0.5           # absolute error fallback
SLOPE_CV_TOL = 0.5      # coefficient of variation for slope consistency

SIGMA2_EVAL_POINTS = [0.5, 1.0, 1.5, 2.0]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def observations():
    """Generate observations once with true parameters."""
    model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


# ============================================================================
# Helpers
# ============================================================================

def _make_fresh_model_and_filter(sigma2_val):
    """Create a fresh model wrapped in DifferentiableModel + compiled filter.

    This matches the production HMC pipeline: DifferentiableModel provides
    trainable_param_names, and the filter uses eager_mode=False (compiled).
    """
    base_model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=sigma2_val, b=1.0, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma2'])
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


def _eval_ll(obs_tf, sigma2_val, base_sigma2=None):
    """Evaluate log-likelihood at a given sigma2 value with fixed seed.

    Creates model at base_sigma2 (for consistent initialization), then
    updates sigma2 to the target value via DifferentiableModel, matching
    the autodiff path exactly.
    """
    if base_sigma2 is None:
        base_sigma2 = sigma2_val
    diff_model, filt = _make_fresh_model_and_filter(base_sigma2)
    diff_model.update_parameters({'sigma2': tf.constant(sigma2_val, dtype=DTYPE)})
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    diff_model.restore_parameters()
    return float(ll.numpy())


def _autodiff_gradient(obs_tf, sigma2_val):
    """Compute gradient of log-likelihood w.r.t. sigma2 via GradientTape."""
    diff_model, filt = _make_fresh_model_and_filter(sigma2_val)

    var = tf.constant(sigma2_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'sigma2': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)

    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    return grad


def _numerical_gradient(obs_tf, sigma2_val):
    """Compute numerical gradient via paired central differences at M radii.

    All evaluations use the same base model (initialized at sigma2_val),
    then update sigma2 via DifferentiableModel — matching the autodiff path.

    Returns (median_slope, slopes_array).
    """
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)

    for i, r in enumerate(radii):
        theta_plus = sigma2_val + r
        theta_minus = sigma2_val - r

        # Guard: sigma2 must be positive
        if theta_minus <= 0.01:
            slopes[i] = np.nan
            continue

        f_plus = _eval_ll(obs_tf, theta_plus, base_sigma2=sigma2_val)
        f_minus = _eval_ll(obs_tf, theta_minus, base_sigma2=sigma2_val)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)

    # Drop NaN entries
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


# ============================================================================
# Tests
# ============================================================================

class TestLEDHOTGradientSigma2:
    """Compare LEDH+OT autodiff gradient to numerical gradient for sigma2.

    Uses compiled path (eager_mode=False) with DifferentiableModel,
    matching the production HMC pipeline.
    """

    @pytest.mark.parametrize("sigma2_val", SIGMA2_EVAL_POINTS)
    def test_gradient_at_point(self, observations, sigma2_val):
        print(f"\n--- sigma2 = {sigma2_val} ---")

        # Autodiff gradient
        ad_grad = _autodiff_gradient(observations, sigma2_val)
        assert ad_grad is not None, f"Autodiff gradient is None at sigma2={sigma2_val}"
        ad_val = ad_grad.numpy()
        assert np.isfinite(ad_val), f"Autodiff gradient not finite: {ad_val}"
        print(f"  Autodiff gradient:  {ad_val:.6f}")

        # Numerical gradient
        num_median, slopes = _numerical_gradient(observations, sigma2_val)
        assert len(slopes) >= 3, f"Too few valid slopes: {len(slopes)}"
        print(f"  Numerical gradient: {num_median:.6f}")
        print(f"  Slopes by radius:   {np.array2string(slopes, precision=4)}")

        # Check slope consistency across radii
        slope_mean = np.mean(slopes)
        slope_std = np.std(slopes)
        if abs(slope_mean) > 1e-6:
            cv = slope_std / abs(slope_mean)
            print(f"  Slope CV:           {cv:.4f} (threshold: {SLOPE_CV_TOL})")
            if cv > SLOPE_CV_TOL:
                print(f"  WARNING: slopes inconsistent (CV={cv:.4f})")

        # Sign agreement
        if abs(num_median) > 0.1:  # only check sign if gradient is non-trivial
            signs_agree = np.sign(ad_val) == np.sign(num_median)
            print(f"  Signs agree:        {signs_agree}")
            assert signs_agree, (
                f"Sign mismatch at sigma2={sigma2_val}: "
                f"ad={ad_val:.4f}, num={num_median:.4f}"
            )

        # Magnitude check
        abs_err = abs(ad_val - num_median)
        rel_err = abs_err / max(abs(num_median), 1e-10)
        print(f"  Abs error:          {abs_err:.6f}")
        print(f"  Rel error:          {rel_err:.4f}")

        passes = abs_err < ABS_TOL or rel_err < REL_TOL
        assert passes, (
            f"Gradient mismatch at sigma2={sigma2_val}: "
            f"ad={ad_val:.4f}, num={num_median:.4f}, "
            f"abs_err={abs_err:.4f}, rel_err={rel_err:.4f}"
        )
        print(f"  PASS")
