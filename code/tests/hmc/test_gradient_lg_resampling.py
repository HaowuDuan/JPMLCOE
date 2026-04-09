"""Gradient validation: LEDH on LG with different resampling methods.

Isolates whether the gradient bias comes from OT resampling specifically
or from any cross-timestep gradient path.

Methods tested:
1. OT entropy (stop_gradient=False) — differentiable
2. Soft (stop_gradient=False) — differentiable
3. Systematic (stop_gradient=True) — non-differentiable baseline
4. No resampling (threshold set so resampling never triggers)

Run: python -m pytest tests/hmc/test_gradient_lg_resampling.py -v -s
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


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
TRUE_OBS_NOISE_STD = 1.0

M_RADII = 5
H_MAX = 0.05


@pytest.fixture(scope="module")
def obs_lg():
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=TRUE_OBS_NOISE_STD, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=20, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def _make_filter(ons_val, resampling_method, stop_gradient_resampling,
                 resampling_config=None, always_resample=True):
    base_model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ons_val, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['obs_noise_std'])

    kwargs = dict(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method=resampling_method,
        weight_clip_range=50.0,
        stop_gradient_resampling=stop_gradient_resampling,
        eager_mode=False,
        always_resample=always_resample,
    )
    if resampling_config is not None:
        kwargs['resampling_config'] = resampling_config
    if resampling_method == 'ot_entropy':
        kwargs['hmc_resampling_method'] = 'ot_entropy'
        kwargs['hmc_resampling_config'] = resampling_config or {'epsilon': 0.5}
    elif resampling_method == 'soft':
        kwargs['hmc_resampling_method'] = 'soft'
        kwargs['hmc_resampling_config'] = resampling_config or {'alpha': 0.5}

    filt = LEDHParticleFlowFilterHMC(**kwargs)
    return diff_model, filt


def _eval_ll(obs_tf, ons_val, **fkwargs):
    diff_model, filt = _make_filter(ons_val, **fkwargs)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _autodiff_grad(obs_tf, ons_val, **fkwargs):
    diff_model, filt = _make_filter(ons_val, **fkwargs)
    var = tf.constant(ons_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'obs_noise_std': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    ad_val = grad.numpy() if grad is not None else None
    return float(ll.numpy()), ad_val


def _numerical_grad(obs_tf, ons_val, **fkwargs):
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)
    for i, r in enumerate(radii):
        if ons_val - r <= 0.01:
            slopes[i] = np.nan
            continue
        f_plus = _eval_ll(obs_tf, ons_val + r, **fkwargs)
        f_minus = _eval_ll(obs_tf, ons_val - r, **fkwargs)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


def _report(label, obs_tf, ons_val, **fkwargs):
    ll_val, ad = _autodiff_grad(obs_tf, ons_val, **fkwargs)
    num_med, slopes = _numerical_grad(obs_tf, ons_val, **fkwargs)
    print(f"\n  [{label}]")
    print(f"    Forward ll:       {ll_val:.4f}")
    print(f"    Autodiff grad:    {ad}")
    print(f"    Numerical grad:   {num_med:.4f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if ad is not None and abs(num_med) > 1e-6:
        ratio = ad / num_med
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    return ad, num_med


class TestOTEntropy:
    def test_ot_entropy(self, obs_lg):
        ad, num = _report("OT entropy, stop_grad=False", obs_lg, TRUE_OBS_NOISE_STD,
                          resampling_method='ot_entropy',
                          stop_gradient_resampling=False,
                          resampling_config={'epsilon': 0.5})
        assert ad is not None
        if abs(num) > 0.1:
            print(f"    RATIO: {ad / num:.4f}")


class TestSoft:
    def test_soft(self, obs_lg):
        ad, num = _report("Soft, stop_grad=False", obs_lg, TRUE_OBS_NOISE_STD,
                          resampling_method='soft',
                          stop_gradient_resampling=False,
                          resampling_config={'alpha': 0.5})
        assert ad is not None
        if abs(num) > 0.1:
            print(f"    RATIO: {ad / num:.4f}")


class TestSystematic:
    def test_systematic_stopgrad(self, obs_lg):
        ad, num = _report("Systematic, stop_grad=True", obs_lg, TRUE_OBS_NOISE_STD,
                          resampling_method='systematic',
                          stop_gradient_resampling=True)
        assert ad is not None
        if abs(num) > 0.1:
            print(f"    RATIO: {ad / num:.4f}")


class TestNoResampling:
    def test_no_resampling(self, obs_lg):
        """Set always_resample=False and high threshold so resampling never triggers."""
        ad, num = _report("No resampling", obs_lg, TRUE_OBS_NOISE_STD,
                          resampling_method='systematic',
                          stop_gradient_resampling=True,
                          always_resample=False)
        assert ad is not None
        if abs(num) > 0.1:
            print(f"    RATIO: {ad / num:.4f}")
