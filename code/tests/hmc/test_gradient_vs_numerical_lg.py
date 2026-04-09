"""Gradient validation: LEDH+OT autodiff vs numerical on Linear Gaussian model.

Same test structure as test_gradient_vs_numerical.py (SV2D), but for the
LG model with obs_noise_std as the trainable parameter. If this gives
ratio ~1.0 but SV2D gives 2.5x, the bug is model-specific.

Run: python -m pytest tests/hmc/test_gradient_vs_numerical_lg.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel


# ============================================================================
# Constants
# ============================================================================

DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
TRUE_OBS_NOISE_STD = 1.0

M_RADII = 5
H_MAX = 0.05


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def obs_lg():
    """T=20 observations from LG model."""
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=TRUE_OBS_NOISE_STD, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=20, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


# ============================================================================
# Helpers
# ============================================================================

def _make_filter(ons_val, n_lambda_steps=N_LAMBDA_STEPS):
    """Create LG model + DifferentiableModel + LEDH+OT compiled filter."""
    base_model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ons_val, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['obs_noise_std'])
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=n_lambda_steps,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


def _eval_ll(obs_tf, ons_val, **kwargs):
    """Evaluate log-likelihood. Fresh model+filter per call."""
    diff_model, filt = _make_filter(ons_val, **kwargs)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _autodiff_grad(obs_tf, ons_val, **kwargs):
    """Autodiff gradient + forward value."""
    diff_model, filt = _make_filter(ons_val, **kwargs)
    var = tf.constant(ons_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'obs_noise_std': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    ad_val = grad.numpy() if grad is not None else None
    return float(ll.numpy()), ad_val


def _numerical_grad(obs_tf, ons_val, **kwargs):
    """Numerical gradient via central differences."""
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)
    for i, r in enumerate(radii):
        if ons_val - r <= 0.01:
            slopes[i] = np.nan
            continue
        f_plus = _eval_ll(obs_tf, ons_val + r, **kwargs)
        f_minus = _eval_ll(obs_tf, ons_val - r, **kwargs)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


def _report(label, obs_tf, ons_val, **kwargs):
    ll_val, ad = _autodiff_grad(obs_tf, ons_val, **kwargs)
    num_med, slopes = _numerical_grad(obs_tf, ons_val, **kwargs)
    print(f"\n  [{label}]")
    print(f"    Forward ll:       {ll_val:.4f}")
    print(f"    Autodiff grad:    {ad}")
    print(f"    Numerical grad:   {num_med:.4f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if ad is not None and abs(num_med) > 1e-6:
        ratio = ad / num_med
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    return ad, num_med


# ============================================================================
# Tests — same structure as SV2D diagnostics
# ============================================================================

class TestLGForwardValues:
    def test_forward_value_match(self, obs_lg):
        ll_ad, _ = _autodiff_grad(obs_lg, TRUE_OBS_NOISE_STD)
        ll_num = _eval_ll(obs_lg, TRUE_OBS_NOISE_STD)
        print(f"\n  Forward ll (autodiff path): {ll_ad:.6f}")
        print(f"  Forward ll (numerical path): {ll_num:.6f}")
        print(f"  Difference: {abs(ll_ad - ll_num):.6f}")
        assert abs(ll_ad - ll_num) < 1e-4


class TestLGTimesteps:
    def test_ledh_ot_T1(self, obs_lg):
        ad, num = _report("LG LEDH+OT T=1", obs_lg[:1], TRUE_OBS_NOISE_STD)
        assert ad is not None

    def test_ledh_ot_T3(self, obs_lg):
        ad, num = _report("LG LEDH+OT T=3", obs_lg[:3], TRUE_OBS_NOISE_STD)
        assert ad is not None

    def test_ledh_ot_T20(self, obs_lg):
        ad, num = _report("LG LEDH+OT T=20", obs_lg, TRUE_OBS_NOISE_STD)
        assert ad is not None


class TestLGFlowSteps:
    def test_ledh_ot_1step(self, obs_lg):
        ad, num = _report("LG LEDH+OT n_lambda=1", obs_lg, TRUE_OBS_NOISE_STD,
                          n_lambda_steps=1)
        assert ad is not None

    def test_ledh_ot_3steps(self, obs_lg):
        ad, num = _report("LG LEDH+OT n_lambda=3", obs_lg, TRUE_OBS_NOISE_STD,
                          n_lambda_steps=3)
        assert ad is not None


class TestLGFullComparison:
    def test_ledh_ot_full(self, obs_lg):
        ad, num = _report("LG LEDH+OT full (T=20, n_lambda=15)", obs_lg,
                          TRUE_OBS_NOISE_STD)
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")

    @pytest.mark.parametrize("ons_val", [0.5, 1.0, 1.5, 2.0])
    def test_ledh_ot_at_points(self, obs_lg, ons_val):
        ad, num = _report(f"LG LEDH+OT ons={ons_val}", obs_lg, ons_val)
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")
