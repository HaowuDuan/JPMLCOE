"""Gradient validation: LEDH+OT autodiff vs numerical on Range Bearing model.

Trainable parameter: sigma_range (observation noise — enters through R,
like LG's obs_noise_std, but the observation model is nonlinear).

Same test structure as SV2D/LG tests.

Run: python -m pytest tests/hmc/test_gradient_vs_numerical_rb.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42

M_RADII = 5
H_MAX = 0.02

TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.01


@pytest.fixture(scope="module")
def obs_rb():
    model = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=TRUE_SIGMA_BEARING,
        dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=20, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def _make_filter(sr_val, n_lambda_steps=N_LAMBDA_STEPS):
    base_model = RangeBearingModel(
        sigma_range=sr_val,
        sigma_bearing=TRUE_SIGMA_BEARING,
        dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma_range'])
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


def _eval_ll(obs_tf, sr_val, **kwargs):
    diff_model, filt = _make_filter(sr_val, **kwargs)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _autodiff_grad(obs_tf, sr_val, **kwargs):
    diff_model, filt = _make_filter(sr_val, **kwargs)
    var = tf.constant(sr_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'sigma_range': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    ad_val = grad.numpy() if grad is not None else None
    return float(ll.numpy()), ad_val


def _numerical_grad(obs_tf, sr_val, **kwargs):
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)
    for i, r in enumerate(radii):
        if sr_val - r <= 0.001:
            slopes[i] = np.nan
            continue
        f_plus = _eval_ll(obs_tf, sr_val + r, **kwargs)
        f_minus = _eval_ll(obs_tf, sr_val - r, **kwargs)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


def _report(label, obs_tf, sr_val, **kwargs):
    ll_val, ad = _autodiff_grad(obs_tf, sr_val, **kwargs)
    num_med, slopes = _numerical_grad(obs_tf, sr_val, **kwargs)
    print(f"\n  [{label}]")
    print(f"    Forward ll:       {ll_val:.4f}")
    print(f"    Autodiff grad:    {ad}")
    print(f"    Numerical grad:   {num_med:.4f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if ad is not None and abs(num_med) > 1e-6:
        ratio = ad / num_med
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    return ad, num_med


class TestRBTimesteps:
    def test_T1(self, obs_rb):
        ad, num = _report("RB LEDH+OT T=1", obs_rb[:1], TRUE_SIGMA_RANGE)
        assert ad is not None

    def test_T3(self, obs_rb):
        ad, num = _report("RB LEDH+OT T=3", obs_rb[:3], TRUE_SIGMA_RANGE)
        assert ad is not None

    def test_T20(self, obs_rb):
        ad, num = _report("RB LEDH+OT T=20", obs_rb, TRUE_SIGMA_RANGE)
        assert ad is not None


class TestRBFullComparison:
    def test_full(self, obs_rb):
        ad, num = _report("RB LEDH+OT full", obs_rb, TRUE_SIGMA_RANGE)
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")

    @pytest.mark.parametrize("sr_val", [0.05, 0.1, 0.2, 0.5])
    def test_at_points(self, obs_rb, sr_val):
        ad, num = _report(f"RB sigma_range={sr_val}", obs_rb, sr_val)
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")
