"""Gradient validation: LEDH+OT autodiff vs numerical FD on Range-Bearing model.

Trainable parameter: sigma_range.
Tolerance: TOL_RB (0.08 relative error).

Each test case prints + saves to tests/hmc/results/test_gradient_vs_numerical_rb.json.

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

from _gradient_test_utils import (
    TOL_RB, gradient_case, reset_results,
)


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
T = 20
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.01
FD_H = 1e-4

MODEL = 'range_bearing'
FILTER = 'ledh_ot'


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


@pytest.fixture(scope="module")
def obs_rb():
    model = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=TRUE_SIGMA_BEARING,
        dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
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


def _eval_ll(obs_tf, sr_val, n_lambda_steps=N_LAMBDA_STEPS):
    diff_model, filt = _make_filter(sr_val, n_lambda_steps=n_lambda_steps)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _run_case(case_name, obs_tf, sr_val, n_lambda_steps=N_LAMBDA_STEPS, T_used=T):
    diff_model, filt = _make_filter(sr_val, n_lambda_steps=n_lambda_steps)
    eval_fn = lambda x: _eval_ll(obs_tf, x, n_lambda_steps=n_lambda_steps)
    return gradient_case(
        test_file=__file__,
        case_name=case_name,
        model_name=MODEL,
        filter_name=FILTER,
        param_name='sigma_range',
        param_val=sr_val,
        eval_fn=eval_fn,
        diff_model=diff_model,
        filt=filt,
        obs_tf=obs_tf,
        dtype=DTYPE,
        seed=PF_SEED,
        tol=TOL_RB,
        n_particles=N_PARTICLES,
        T=T_used,
        n_lambda_steps=n_lambda_steps,
        fd_h=FD_H,
    )


class TestRBTimesteps:
    def test_T1(self, obs_rb):
        _run_case("RB LEDH+OT T=1", obs_rb[:1], TRUE_SIGMA_RANGE, T_used=1)

    def test_T5(self, obs_rb):
        _run_case("RB LEDH+OT T=5", obs_rb[:5], TRUE_SIGMA_RANGE, T_used=5)

    def test_T20(self, obs_rb):
        _run_case("RB LEDH+OT T=20", obs_rb, TRUE_SIGMA_RANGE, T_used=T)


class TestRBAtPoints:
    @pytest.mark.parametrize("sr_val", [0.05, 0.1, 0.2, 0.5])
    def test_param_grid(self, obs_rb, sr_val):
        _run_case(f"RB LEDH+OT sigma_range={sr_val}", obs_rb, sr_val)
