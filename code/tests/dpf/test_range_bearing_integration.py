"""Integration tests for range-bearing DPF parameter inference.

These tests run short HMC chains with the TFP sampler and check that the
posterior makes sense. Marked @pytest.mark.slow — run explicitly with:

    pytest tests/dpf/test_range_bearing_integration.py -m slow -v -s

Note: gradient monitoring during TFP HMC would require wrapping internals
(on_grad only fires in custom_hmc path). Those tests are intentionally
omitted — use unit tests for gradient diagnostics instead.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np
import tensorflow as tf

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.hmc_runner import DPFRunner
from src.DF.types import ParameterSpec
import tensorflow_probability as tfp


pytestmark = pytest.mark.slow


def _make_param_specs():
    return {
        'sigma_range': ParameterSpec(
            name='sigma_range',
            init_value=0.3,
            constraint='positive',
            prior=tfp.distributions.LogNormal(loc=-2.3, scale=0.5),
        ),
        'sigma_bearing': ParameterSpec(
            name='sigma_bearing',
            init_value=0.3,
            constraint='positive',
            prior=tfp.distributions.LogNormal(loc=-2.3, scale=0.5),
        ),
    }


def _run_short_hmc(observations, num_burnin=10, num_samples=10):
    model = RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)
    filter_kwargs = dict(
        n_particles=100,
        n_lambda_steps=10,
        weight_clip_range=50.0,
        stop_gradient_resampling=True,
        eager_mode=True,
    )
    runner = DPFRunner(
        base_model=model,
        filter_class=LEDHParticleFlowFilterHMC,
        filter_kwargs=filter_kwargs,
        param_specs=_make_param_specs(),
        sampler='hmc',   # TFP tfp.mcmc.HamiltonianMonteCarlo
    )
    result = runner.run_inference(
        observations=observations.numpy(),
        num_samples=num_samples,
        num_burnin=num_burnin,
        step_size=0.01,
        num_leapfrog_steps=5,
        target_accept_prob=0.7,
        seed=42,
    )
    return result


@pytest.fixture(scope='module')
def short_observations_module():
    """T=20 observations from true model, shared across this module."""
    true_model = RangeBearingModel(sigma_range=0.1, sigma_bearing=0.1)
    rng = np.random.default_rng(42)
    _, _, obs = generate_data(true_model, T=20, rng=rng)
    return tf.constant(obs, dtype=tf.float32)


def test_short_hmc_run_produces_samples(short_observations_module):
    """10 burnin + 10 samples completes; DPFResult has the right shape."""
    result = _run_short_hmc(short_observations_module, num_burnin=10, num_samples=10)

    assert 'sigma_range' in result.samples
    assert 'sigma_bearing' in result.samples
    assert len(result.samples['sigma_range']) == 10
    assert len(result.samples['sigma_bearing']) == 10

    print(f"\n  sigma_range samples:  {np.round(result.samples['sigma_range'], 4)}")
    print(f"  sigma_bearing samples: {np.round(result.samples['sigma_bearing'], 4)}")


def test_acceptance_rate_nonzero(short_observations_module):
    """HMC acceptance rate > 0% — not stuck at a gradient cliff.

    0% acceptance is the end-to-end symptom of gradient explosion:
    proposals always land in a region with -inf log-posterior.
    """
    result = _run_short_hmc(short_observations_module, num_burnin=10, num_samples=10)
    accept_rate = result.diagnostics['acceptance_rate']

    print(f"\n  acceptance_rate = {accept_rate:.1%}")
    assert accept_rate > 0.0, (
        f"0% acceptance — HMC is stuck (likely gradient explosion or NaN nlp)"
    )


def test_posterior_mean_closer_to_truth(short_observations_module):
    """After 50 burnin + 50 samples, posterior means move toward truth (0.1).

    True value: 0.1, initial guess: 0.3. Even a noisy short chain should
    shift the mean in the right direction if gradients are correct.
    """
    result = _run_short_hmc(short_observations_module, num_burnin=50, num_samples=50)

    truth = 0.1
    initial = 0.3

    for name in ('sigma_range', 'sigma_bearing'):
        posterior_mean = result.summary[name]['mean']
        err_posterior = abs(posterior_mean - truth)
        err_initial = abs(initial - truth)

        print(f"\n  {name}:")
        print(f"    initial={initial}  truth={truth}  "
              f"posterior_mean={posterior_mean:.4f}")
        print(f"    |posterior_mean - truth|={err_posterior:.4f}  "
              f"|initial - truth|={err_initial:.4f}")

        assert err_posterior < err_initial, (
            f"{name}: posterior mean {posterior_mean:.4f} farther from truth {truth} "
            f"than initial {initial} — HMC is moving in the wrong direction"
        )
