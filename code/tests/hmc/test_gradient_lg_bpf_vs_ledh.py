"""Gradient validation: BPF+OT vs LEDH+OT on Linear Gaussian.

If BPF+OT ratio ~1.0 but LEDH+OT ratio ~1.38, the bias comes from
the LEDH flow loop interacting with OT, not OT alone.

Run: python -m pytest tests/hmc/test_gradient_lg_bpf_vs_ledh.py -v -s
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
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
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


def _make_bpf(ons_val):
    base_model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ons_val, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['obs_noise_std'])
    filt = BootstrapPFHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


def _make_ledh(ons_val):
    base_model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ons_val, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['obs_noise_std'])
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


def _eval_ll(obs_tf, ons_val, make_fn):
    diff_model, filt = make_fn(ons_val)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _autodiff_grad(obs_tf, ons_val, make_fn):
    diff_model, filt = make_fn(ons_val)
    var = tf.constant(ons_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'obs_noise_std': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    ad_val = grad.numpy() if grad is not None else None
    return float(ll.numpy()), ad_val


def _numerical_grad(obs_tf, ons_val, make_fn):
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)
    for i, r in enumerate(radii):
        if ons_val - r <= 0.01:
            slopes[i] = np.nan
            continue
        f_plus = _eval_ll(obs_tf, ons_val + r, make_fn)
        f_minus = _eval_ll(obs_tf, ons_val - r, make_fn)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


def _report(label, obs_tf, ons_val, make_fn):
    ll_val, ad = _autodiff_grad(obs_tf, ons_val, make_fn)
    num_med, slopes = _numerical_grad(obs_tf, ons_val, make_fn)
    print(f"\n  [{label}]")
    print(f"    Forward ll:       {ll_val:.4f}")
    print(f"    Autodiff grad:    {ad}")
    print(f"    Numerical grad:   {num_med:.4f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if ad is not None and abs(num_med) > 1e-6:
        ratio = ad / num_med
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    return ad, num_med


class TestBPFvsLEDH:

    def test_bpf_ot(self, obs_lg):
        ad, num = _report("BPF+OT T=20", obs_lg, TRUE_OBS_NOISE_STD, _make_bpf)
        assert ad is not None
        if abs(num) > 0.1:
            print(f"    RATIO: {ad / num:.4f}")

    def test_ledh_ot(self, obs_lg):
        ad, num = _report("LEDH+OT T=20", obs_lg, TRUE_OBS_NOISE_STD, _make_ledh)
        assert ad is not None
        if abs(num) > 0.1:
            print(f"    RATIO: {ad / num:.4f}")
