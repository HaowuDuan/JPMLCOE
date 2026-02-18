"""Shared fixtures for DPF range-bearing tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np
import tensorflow as tf

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC


@pytest.fixture(scope='session')
def range_bearing_model():
    """RangeBearingModel at true params (sigma_range=0.1, sigma_bearing=0.1)."""
    return RangeBearingModel(sigma_range=0.1, sigma_bearing=0.1)


@pytest.fixture(scope='session')
def range_bearing_model_wrong():
    """RangeBearingModel at initial guess (sigma_range=0.3, sigma_bearing=0.3)."""
    return RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)


@pytest.fixture(scope='session')
def short_observations(range_bearing_model):
    """T=20 synthetic observations from the true model, seed=42."""
    rng = np.random.default_rng(42)
    _, _, obs = generate_data(range_bearing_model, T=20, rng=rng)
    return tf.constant(obs, dtype=tf.float32)


@pytest.fixture
def ledh_filter_hmc(range_bearing_model_wrong):
    """Small LEDH filter in eager mode — fast enough for unit tests."""
    return LEDHParticleFlowFilterHMC(
        range_bearing_model_wrong,
        n_particles=100,
        n_lambda_steps=10,
        weight_clip_range=50.0,
        stop_gradient_resampling=True,
        eager_mode=True,
    )


@pytest.fixture
def grad_collector():
    """Accumulates on_grad callback calls.

    Returns dict with keys: 'steps', 'nlps', 'norms'.
    Pass as on_grad= to DPFRunner.
    """
    data = {'steps': [], 'nlps': [], 'norms': []}

    def collect(step, nlp, grad):
        data['steps'].append(step)
        data['nlps'].append(nlp)
        data['norms'].append(float(tf.norm(grad).numpy()))

    data['callback'] = collect
    return data


@pytest.fixture
def timestep_collector():
    """Accumulates on_timestep callback calls.

    Returns dict with keys: 'ts', 'log_liks', 'ess_vals', 'max_log_thetas'.
    Pass as on_timestep= to LEDHParticleFlowFilterHMC.
    """
    data = {'ts': [], 'log_liks': [], 'ess_vals': [], 'max_log_thetas': []}

    def collect(t, log_lik_t, ess, max_log_theta):
        data['ts'].append(t)
        data['log_liks'].append(log_lik_t)
        data['ess_vals'].append(ess)
        data['max_log_thetas'].append(max_log_theta)

    data['callback'] = collect
    return data
