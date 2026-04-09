"""Gradient validation using the exact DPFRunner path that HMC uses.

Compares autodiff gradient from DPFRunner._negative_log_posterior against
numerical gradient using fresh runners per evaluation (same pattern as
conftest_hmc_diag.py:finite_difference_gradient).

Tests both LG (obs_noise_std, known to pass in old tests) and SV2D (sigma2).
If LG passes here but failed in test_gradient_vs_numerical_lg.py, then our
DifferentiableModel-only test harness was the problem, not the filter.

Run: python -m pytest tests/hmc/test_gradient_vs_numerical_dpfrunner.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.linear_gaussian import LinearGaussianModel
from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.hmc_runner import DPFRunner
from src.DF.parameter_handler import ParameterHandler
from src.DF.types import ParameterSpec


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
SEED = 42
T = 20
M_RADII = 5
H_MAX = 0.05


# ============================================================================
# LG model helpers
# ============================================================================

def _lg_param_specs(init_ons):
    return {
        'obs_noise_std': ParameterSpec(
            name='obs_noise_std',
            init_value=init_ons,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                loc=tf.constant(0.0, dtype=DTYPE),
                scale=tf.constant(1.0, dtype=DTYPE),
            ),
        )
    }


def _lg_filter_kwargs():
    return dict(
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )


def _make_lg_runner(init_ons, observations_tf):
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=init_ons, dtype=DTYPE,
    )
    runner = DPFRunner(
        base_model=model,
        filter_class=LEDHParticleFlowFilterHMC,
        filter_kwargs=_lg_filter_kwargs(),
        param_specs=_lg_param_specs(init_ons),
        sampler='hmc',
    )
    runner._observations_tf = observations_tf
    return runner


# ============================================================================
# SV2D model helpers
# ============================================================================

def _sv2d_param_specs(init_sigma2):
    return {
        'sigma2': ParameterSpec(
            name='sigma2',
            init_value=init_sigma2,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                loc=tf.constant(0.0, dtype=DTYPE),
                scale=tf.constant(0.5, dtype=DTYPE),
            ),
        )
    }


def _sv2d_filter_kwargs():
    return dict(
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )


def _make_sv2d_runner(init_sigma2, observations_tf):
    model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=init_sigma2, b=1.0, dtype=DTYPE,
    )
    runner = DPFRunner(
        base_model=model,
        filter_class=LEDHParticleFlowFilterHMC,
        filter_kwargs=_sv2d_filter_kwargs(),
        param_specs=_sv2d_param_specs(init_sigma2),
        sampler='hmc',
    )
    runner._observations_tf = observations_tf
    return runner


# ============================================================================
# Generic gradient helpers (through DPFRunner)
# ============================================================================

def _autodiff_grad_dpfrunner(make_runner_fn, init_val, observations_tf):
    """Autodiff gradient through DPFRunner._negative_log_posterior."""
    runner = make_runner_fn(init_val, observations_tf)
    q = runner.param_handler.unconstrained_init
    with tf.GradientTape() as tape:
        tape.watch(q)
        nlp = runner._negative_log_posterior(q)
    grad = tape.gradient(nlp, q)
    return float(nlp.numpy()), grad.numpy().copy()


def _numerical_grad_dpfrunner(make_runner_fn, init_val, observations_tf):
    """Numerical gradient using fresh runner per evaluation (same as conftest)."""
    handler = ParameterHandler(
        make_runner_fn(init_val, observations_tf).param_handler.param_specs,
        dtype=DTYPE,
    )
    q = handler.unconstrained_init

    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)

    for i, r in enumerate(radii):
        # Fresh runner for each evaluation
        runner_plus = make_runner_fn(init_val, observations_tf)
        nlp_plus = float(runner_plus._negative_log_posterior(q + r).numpy())

        runner_minus = make_runner_fn(init_val, observations_tf)
        nlp_minus = float(runner_minus._negative_log_posterior(q - r).numpy())

        slopes[i] = (nlp_plus - nlp_minus) / (2.0 * r)

    return np.median(slopes), slopes


def _report(label, make_runner_fn, init_val, observations_tf):
    nlp, ad_grad = _autodiff_grad_dpfrunner(make_runner_fn, init_val, observations_tf)
    num_med, slopes = _numerical_grad_dpfrunner(make_runner_fn, init_val, observations_tf)
    ad_val = ad_grad[0]
    print(f"\n  [{label}]")
    print(f"    NLP:              {nlp:.4f}")
    print(f"    Autodiff grad:    {ad_val:.6f}")
    print(f"    Numerical grad:   {num_med:.6f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if abs(num_med) > 1e-6:
        ratio = ad_val / num_med
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    return ad_val, num_med


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def obs_lg():
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=1.0, dtype=DTYPE,
    )
    rng = np.random.default_rng(SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


@pytest.fixture(scope="module")
def obs_sv2d():
    model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0, dtype=DTYPE,
    )
    rng = np.random.default_rng(SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


# ============================================================================
# Tests
# ============================================================================

class TestLGViaDPFRunner:
    """LG model through DPFRunner — this path passed in old tests."""

    def test_at_true(self, obs_lg):
        ad, num = _report("LG DPFRunner ons=1.0", _make_lg_runner, 1.0, obs_lg)
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")

    def test_at_init(self, obs_lg):
        ad, num = _report("LG DPFRunner ons=2.0", _make_lg_runner, 2.0, obs_lg)
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")


class TestSV2DViaDPFRunner:
    """SV2D model through DPFRunner — the production path."""

    def test_at_true(self, obs_sv2d):
        ad, num = _report("SV2D DPFRunner sigma2=1.0", _make_sv2d_runner, 1.0, obs_sv2d)
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")

    def test_at_init(self, obs_sv2d):
        ad, num = _report("SV2D DPFRunner sigma2=2.0", _make_sv2d_runner, 2.0, obs_sv2d)
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")
