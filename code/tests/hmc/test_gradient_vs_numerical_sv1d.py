"""Gradient validation: LEDH+OT autodiff vs numerical FD on 1D Stochastic Volatility.

Trainable parameter: alpha (persistence — enters transition).
Tolerance: TOL_SV1D (0.10 relative error).

Each test case prints + saves to tests/hmc/results/test_gradient_vs_numerical_sv1d.json.

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

from _gradient_test_utils import (
    TOL_SV1D, gradient_case, reset_results,
)


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
T = 20
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42

TRUE_ALPHA = 0.91
TRUE_SIGMA = 1.0
TRUE_BETA = 0.5
FD_H = 1e-4

MODEL = 'stochastic_volatility_1d'
FILTER = 'ledh_ot'


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


@pytest.fixture(scope="module")
def obs_sv1d():
    model = StochasticVolatilityModel(
        alpha=TRUE_ALPHA, sigma=TRUE_SIGMA, beta=TRUE_BETA,
        log_space=True, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs_raw = generate_data(model, T=T, rng=rng)
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


def _eval_ll(obs_tf, alpha_val, n_lambda_steps=N_LAMBDA_STEPS):
    diff_model, filt = _make_filter(alpha_val, n_lambda_steps=n_lambda_steps)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _run_case(case_name, obs_tf, alpha_val, n_lambda_steps=N_LAMBDA_STEPS, T_used=T):
    diff_model, filt = _make_filter(alpha_val, n_lambda_steps=n_lambda_steps)
    eval_fn = lambda x: _eval_ll(obs_tf, x, n_lambda_steps=n_lambda_steps)
    return gradient_case(
        test_file=__file__,
        case_name=case_name,
        model_name=MODEL,
        filter_name=FILTER,
        param_name='alpha',
        param_val=alpha_val,
        eval_fn=eval_fn,
        diff_model=diff_model,
        filt=filt,
        obs_tf=obs_tf,
        dtype=DTYPE,
        seed=PF_SEED,
        tol=TOL_SV1D,
        n_particles=N_PARTICLES,
        T=T_used,
        n_lambda_steps=n_lambda_steps,
        fd_h=FD_H,
    )


class TestSV1DTimesteps:
    def test_T1(self, obs_sv1d):
        _run_case("SV1D LEDH+OT T=1", obs_sv1d[:1], TRUE_ALPHA, T_used=1)

    def test_T5(self, obs_sv1d):
        _run_case("SV1D LEDH+OT T=5", obs_sv1d[:5], TRUE_ALPHA, T_used=5)

    def test_T20(self, obs_sv1d):
        _run_case("SV1D LEDH+OT T=20", obs_sv1d, TRUE_ALPHA, T_used=T)


class TestSV1DAtPoints:
    @pytest.mark.parametrize("alpha_val", [0.5, 0.7, 0.91, 0.95])
    def test_param_grid(self, obs_sv1d, alpha_val):
        _run_case(f"SV1D LEDH+OT alpha={alpha_val}", obs_sv1d, alpha_val)
