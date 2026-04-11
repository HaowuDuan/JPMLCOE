"""Gradient validation: LEDH+OT autodiff vs numerical FD on Linear Gaussian.

Trainable parameter: obs_noise_std.
Tolerance: TOL_LG (0.03 relative error) — LG should be very tight.

Each test case prints + saves to tests/hmc/results/test_gradient_vs_numerical_lg.json.

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

from _gradient_test_utils import (
    TOL_LG, gradient_case, reset_results,
)


# ============================================================================
# Constants
# ============================================================================

DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
T = 20
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
TRUE_OBS_NOISE_STD = 1.0
FD_H = 1e-4

MODEL = 'linear_gaussian'
FILTER = 'ledh_ot'


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    """Wipe the results JSON once at the start of the module run."""
    reset_results(__file__)
    yield


@pytest.fixture(scope="module")
def obs_lg():
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=TRUE_OBS_NOISE_STD, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


# ============================================================================
# Helpers
# ============================================================================

def _make_filter(ons_val, n_lambda_steps=N_LAMBDA_STEPS):
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


def _eval_ll(obs_tf, ons_val, n_lambda_steps=N_LAMBDA_STEPS):
    diff_model, filt = _make_filter(ons_val, n_lambda_steps=n_lambda_steps)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _run_case(case_name, obs_tf, ons_val, n_lambda_steps=N_LAMBDA_STEPS, T_used=T):
    diff_model, filt = _make_filter(ons_val, n_lambda_steps=n_lambda_steps)
    eval_fn = lambda x: _eval_ll(obs_tf, x, n_lambda_steps=n_lambda_steps)
    return gradient_case(
        test_file=__file__,
        case_name=case_name,
        model_name=MODEL,
        filter_name=FILTER,
        param_name='obs_noise_std',
        param_val=ons_val,
        eval_fn=eval_fn,
        diff_model=diff_model,
        filt=filt,
        obs_tf=obs_tf,
        dtype=DTYPE,
        seed=PF_SEED,
        tol=TOL_LG,
        n_particles=N_PARTICLES,
        T=T_used,
        n_lambda_steps=n_lambda_steps,
        fd_h=FD_H,
    )


# ============================================================================
# Tests
# ============================================================================

class TestLGForwardValues:
    def test_forward_value_match(self, obs_lg):
        """Forward log-lik should match between autodiff and numerical paths."""
        diff_model, filt = _make_filter(TRUE_OBS_NOISE_STD)
        var = tf.constant(TRUE_OBS_NOISE_STD, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(var)
            diff_model.update_parameters({'obs_noise_std': var})
            ll_ad = float(filt.log_marginal_likelihood_tf(obs_lg, seed=PF_SEED).numpy())
        diff_model.restore_parameters()
        ll_num = _eval_ll(obs_lg, TRUE_OBS_NOISE_STD)
        diff = abs(ll_ad - ll_num)
        print(f"\n  ll_ad={ll_ad:.6f}  ll_num={ll_num:.6f}  diff={diff:.2e}")
        assert diff < 1e-4, f"Forward value mismatch: {diff:.2e}"


class TestLGTimesteps:
    def test_T1(self, obs_lg):
        _run_case("LG LEDH+OT T=1", obs_lg[:1], TRUE_OBS_NOISE_STD, T_used=1)

    def test_T5(self, obs_lg):
        _run_case("LG LEDH+OT T=5", obs_lg[:5], TRUE_OBS_NOISE_STD, T_used=5)

    def test_T20(self, obs_lg):
        _run_case("LG LEDH+OT T=20", obs_lg, TRUE_OBS_NOISE_STD, T_used=T)


class TestLGAtPoints:
    @pytest.mark.parametrize("ons_val", [0.5, 1.0, 1.5, 2.0])
    def test_param_grid(self, obs_lg, ons_val):
        _run_case(f"LG LEDH+OT ons={ons_val}", obs_lg, ons_val)
