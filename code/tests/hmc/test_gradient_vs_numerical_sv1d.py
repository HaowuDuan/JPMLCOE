"""Gradient validation: LEDH+OT autodiff vs numerical on 1D SV model.

Trainable parameter: alpha (persistence, enters through transitions).
Same test structure as SV2D test.

Run: python -m pytest tests/hmc/test_gradient_vs_numerical_sv1d.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.models.stochastic_volatility import StochasticVolatilityModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel
from src.utils.linalg import graph_safe_inv


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42

M_RADII = 5
H_MAX = 0.02  # smaller for alpha since it's in (0,1)

# SV1D uses log_space=True so EKF works (H_x=1 instead of 0)
TRUE_ALPHA = 0.91
TRUE_SIGMA = 1.0
TRUE_BETA = 0.5


@pytest.fixture(scope="module")
def obs_sv1d():
    model = StochasticVolatilityModel(
        alpha=TRUE_ALPHA, sigma=TRUE_SIGMA, beta=TRUE_BETA,
        log_space=True, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs_raw = generate_data(model, T=20, rng=rng)
    obs = model.transform_observations(obs_raw)
    return tf.constant(obs, dtype=DTYPE)


def _make_filter(alpha_val, n_lambda_steps=N_LAMBDA_STEPS):
    base_model = StochasticVolatilityModel(
        alpha=alpha_val, sigma=TRUE_SIGMA, beta=TRUE_BETA,
        log_space=True, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['alpha'])
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


def _eval_ll(obs_tf, alpha_val, **kwargs):
    diff_model, filt = _make_filter(alpha_val, **kwargs)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _compiled_ll_with_current_params(obs_tf, filt):
    """Run compiled filter without reinitializing particles/covs."""
    R = filt.model.observation_noise_cov
    R_inv = graph_safe_inv(R)
    regularization_tf = tf.constant(filt.regularization, dtype=filt.dtype)

    param_values = []
    for name in filt.model.trainable_param_names:
        val = getattr(filt.model, name)
        val = (
            tf.identity(val)
            if isinstance(val, tf.Tensor)
            else tf.constant(float(val), dtype=filt.dtype)
        )
        param_values.append(val)
    param_values = tf.stack(param_values)

    return filt._compiled_filter(
        obs_tf,
        filt.particles.value(),
        filt.weights.value(),
        filt.particle_covs.value(),
        R,
        R_inv,
        regularization_tf,
        filt.rng_key[0],
        param_values,
    )


def _eval_ll_detached_init(obs_tf, base_alpha, eval_alpha, **kwargs):
    """Evaluate LL with initial conditions baked in at base_alpha.

    log_marginal_likelihood_tf() always calls initialize(), so using it after
    update_parameters() would make Sigma_0 depend on eval_alpha. Initialize
    here at base_alpha, then call the compiled loop directly so only the model
    dynamics/weights see eval_alpha.
    """
    diff_model, filt = _make_filter(base_alpha, **kwargs)
    filt.initialize(random_seed=int(PF_SEED[0].numpy()))
    try:
        diff_model.update_parameters({'alpha': tf.constant(eval_alpha, dtype=DTYPE)})
        ll = _compiled_ll_with_current_params(obs_tf, filt)
    finally:
        diff_model.restore_parameters()
    return float(ll.numpy())


def _autodiff_grad(obs_tf, alpha_val, **kwargs):
    diff_model, filt = _make_filter(alpha_val, **kwargs)
    var = tf.constant(alpha_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'alpha': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    ad_val = grad.numpy() if grad is not None else None
    return float(ll.numpy()), ad_val


def _numerical_grad(obs_tf, alpha_val, **kwargs):
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)
    for i, r in enumerate(radii):
        if alpha_val - r <= 0.01 or alpha_val + r >= 0.999:
            slopes[i] = np.nan
            continue
        f_plus = _eval_ll_detached_init(obs_tf, alpha_val, alpha_val + r, **kwargs)
        f_minus = _eval_ll_detached_init(obs_tf, alpha_val, alpha_val - r, **kwargs)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


def _report(label, obs_tf, alpha_val, **kwargs):
    ll_val, ad = _autodiff_grad(obs_tf, alpha_val, **kwargs)
    num_med, slopes = _numerical_grad(obs_tf, alpha_val, **kwargs)
    print(f"\n  [{label}]")
    print(f"    Forward ll:       {ll_val:.4f}")
    print(f"    Autodiff grad:    {ad}")
    print(f"    Numerical grad:   {num_med:.4f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if ad is not None and abs(num_med) > 1e-6:
        ratio = ad / num_med
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    return ad, num_med


class TestSV1DTimesteps:
    def test_T1(self, obs_sv1d):
        ad, num = _report("SV1D LEDH+OT T=1", obs_sv1d[:1], TRUE_ALPHA)
        assert ad is not None

    def test_T3(self, obs_sv1d):
        ad, num = _report("SV1D LEDH+OT T=3", obs_sv1d[:3], TRUE_ALPHA)
        assert ad is not None

    def test_T20(self, obs_sv1d):
        ad, num = _report("SV1D LEDH+OT T=20", obs_sv1d, TRUE_ALPHA)
        assert ad is not None


class TestSV1DFullComparison:
    def test_full(self, obs_sv1d):
        ad, num = _report("SV1D LEDH+OT full", obs_sv1d, TRUE_ALPHA)
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")

    @pytest.mark.parametrize("alpha_val", [0.5, 0.7, 0.91, 0.95])
    def test_at_points(self, obs_sv1d, alpha_val):
        ad, num = _report(f"SV1D alpha={alpha_val}", obs_sv1d, alpha_val)
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")
