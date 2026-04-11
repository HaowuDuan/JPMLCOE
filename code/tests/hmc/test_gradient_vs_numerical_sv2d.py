"""Gradient validation: LEDH+OT autodiff vs numerical FD on 2D Stochastic Volatility.

Trainable parameter: sigma2 (enters transition).
Tolerance: TOL_SV2D (0.15 relative error) — loosest because of multiplicative
state-dependent observation noise.

Each test case prints + saves to tests/hmc/results/test_gradient_vs_numerical_sv2d.json.

Run: python -m pytest tests/hmc/test_gradient_vs_numerical_sv2d.py -v -s
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

from _gradient_test_utils import (
    TOL_SV2D, gradient_case, reset_results,
)


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
T = 20
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
TRUE_SIGMA2 = 1.0
FD_H = 1e-4

MODEL = 'stochastic_volatility_2d'
FILTER = 'ledh_ot'


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


@pytest.fixture(scope="module")
def obs_sv2d():
    model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=TRUE_SIGMA2, b=1.0, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def _make_filter(sigma2_val, n_lambda_steps=N_LAMBDA_STEPS):
    base_model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=sigma2_val, b=1.0, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma2'])
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


def _eval_ll(obs_tf, sigma2_val, n_lambda_steps=N_LAMBDA_STEPS):
    diff_model, filt = _make_filter(sigma2_val, n_lambda_steps=n_lambda_steps)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _run_case(case_name, obs_tf, sigma2_val, n_lambda_steps=N_LAMBDA_STEPS, T_used=T):
    diff_model, filt = _make_filter(sigma2_val, n_lambda_steps=n_lambda_steps)
    eval_fn = lambda x: _eval_ll(obs_tf, x, n_lambda_steps=n_lambda_steps)
    return gradient_case(
        test_file=__file__,
        case_name=case_name,
        model_name=MODEL,
        filter_name=FILTER,
        param_name='sigma2',
        param_val=sigma2_val,
        eval_fn=eval_fn,
        diff_model=diff_model,
        filt=filt,
        obs_tf=obs_tf,
        dtype=DTYPE,
        seed=PF_SEED,
        tol=TOL_SV2D,
        n_particles=N_PARTICLES,
        T=T_used,
        n_lambda_steps=n_lambda_steps,
        fd_h=FD_H,
    )


class TestSV2DTimesteps:
    def test_T1(self, obs_sv2d):
        _run_case("SV2D LEDH+OT T=1", obs_sv2d[:1], TRUE_SIGMA2, T_used=1)

    def test_T5(self, obs_sv2d):
        _run_case("SV2D LEDH+OT T=5", obs_sv2d[:5], TRUE_SIGMA2, T_used=5)

    def test_T20(self, obs_sv2d):
        _run_case("SV2D LEDH+OT T=20", obs_sv2d, TRUE_SIGMA2, T_used=T)


class TestSV2DAtPoints:
    @pytest.mark.parametrize("sigma2_val", [0.5, 1.0, 1.5, 2.0])
    def test_param_grid(self, obs_sv2d, sigma2_val):
        _run_case(f"SV2D LEDH+OT sigma2={sigma2_val}", obs_sv2d, sigma2_val)
