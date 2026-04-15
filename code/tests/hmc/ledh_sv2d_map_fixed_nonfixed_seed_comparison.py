"""Paired MAP comparison: fixed PF seed vs fresh-per-step PF seed.

Single-purpose diagnostic. Runs the same 20-step MAP inference twice on
identical observations, initial sigma2, Adam hyperparameters, and Hydra/map
seed — only differing in whether the PF seed used inside
log_marginal_likelihood_tf is fixed or regenerated each step:

  - test_fixed_seed:  random_seed=False  → PF seed is [map_seed, 0] every step
  - test_random_seed: random_seed=True   → PF seed is [map_seed, step] every step

Records per-step grad_norm, loss, log_likelihood, sigma2 trajectory.
No assertion. Output is a paired trace for eyeballing.

Purpose: determine whether the gradient spikes seen in the production
failing MAP run (random_seed=True) are caused by seed-to-seed variance
of the stochastic log-likelihood estimator. If fixed-seed MAP is smooth
and random-seed MAP spikes, the spikes are pure seed variance — per-seed
gradients are mathematically correct (confirmed by
`test_gradient_vs_numerical_sv2d.py`) but the estimator's gradient
distribution across seeds is heavy-tailed.

Config mirrors `code/configs/dpf/map/stochastic_volatility_2d/ledh_ot_sigma2.yaml`
verbatim except num_steps is shortened from 300 to 20.

Run:
  python -m pytest tests/hmc/ledh_sv2d_map_fixed_nonfixed_seed_comparison.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF import DPFRunner
from src.DF.types import ParameterSpec

from _gradient_test_utils import save_result, reset_results


DTYPE = tf.float32          # match production failing config
N_PARTICLES = 1000
N_LAMBDA_STEPS = 29
T = 200
DATA_SEED = 42
TRUE_SIGMA2 = 1.0
INIT_SIGMA2 = 2.0

NUM_STEPS = 20
LEARNING_RATE = 0.01
MAP_SEED = 42

MODEL = 'stochastic_volatility_2d'
FILTER = 'ledh_ot'


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


@pytest.fixture(scope="module")
def observations():
    model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=TRUE_SIGMA2, b=1.0, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return obs


def _make_runner():
    """Fresh DPFRunner per test (each needs its own filter + optimizer state)."""
    base_model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=INIT_SIGMA2, b=1.0, dtype=DTYPE,
    )
    param_specs = {
        'sigma2': ParameterSpec(
            name='sigma2',
            init_value=INIT_SIGMA2,
            constraint='positive',
            prior=tfp.distributions.LogNormal(loc=0.0, scale=0.5),
        ),
    }
    filter_kwargs = dict(
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
    )
    return DPFRunner(
        base_model=base_model,
        filter_class=LEDHParticleFlowFilterHMC,
        filter_kwargs=filter_kwargs,
        param_specs=param_specs,
        sampler='map',
    )


def _run_and_record(obs, random_seed, case_name):
    runner = _make_runner()
    result = runner.run_map(
        observations=obs,
        num_steps=NUM_STEPS,
        learning_rate=LEARNING_RATE,
        optimizer='adam',
        random_seed=random_seed,
        seed=MAP_SEED,
        print_every=1,
    )

    grad_hist = list(result.diagnostics.get('grad_norm_history', []))
    loss_hist = list(result.diagnostics.get('loss_history', []))
    ll_hist = list(result.diagnostics.get('log_likelihood_history', []))
    sigma2_hist = list(result.metadata.get('param_history', {}).get('sigma2', []))

    grad_arr = np.array(grad_hist)
    loss_arr = np.array(loss_hist)

    print(f"\n  [{case_name}]")
    print(f"    N={N_PARTICLES}  T={T}  n_lambda={N_LAMBDA_STEPS}  num_steps={NUM_STEPS}")
    print(f"    random_seed={random_seed}  map_seed={MAP_SEED}  "
          f"init_sigma2={INIT_SIGMA2}  true_sigma2={TRUE_SIGMA2}")
    print(f"    |grad| min/median/max: "
          f"{grad_arr.min():.4f} / {float(np.median(grad_arr)):.4f} / {grad_arr.max():.4f}")
    print(f"    loss  min/median/max:  "
          f"{loss_arr.min():.4f} / {float(np.median(loss_arr)):.4f} / {loss_arr.max():.4f}")
    for step in range(len(grad_hist)):
        print(f"      step {step:3d}  loss={loss_hist[step]:10.4f}  "
              f"|grad|={grad_hist[step]:8.4f}  sigma2={sigma2_hist[step]:.4f}")

    save_result(__file__, {
        'case_name': case_name,
        'model_name': MODEL,
        'filter_name': FILTER,
        'n_particles': N_PARTICLES,
        'n_lambda_steps': N_LAMBDA_STEPS,
        'T': T,
        'num_steps': NUM_STEPS,
        'learning_rate': LEARNING_RATE,
        'optimizer': 'adam',
        'random_seed': random_seed,
        'map_seed': MAP_SEED,
        'init_sigma2': INIT_SIGMA2,
        'true_sigma2': TRUE_SIGMA2,
        'grad_norm_history': [float(x) for x in grad_hist],
        'loss_history': [float(x) for x in loss_hist],
        'log_likelihood_history': [float(x) for x in ll_hist],
        'sigma2_history': [float(x) for x in sigma2_hist],
        'grad_max': float(grad_arr.max()),
        'grad_median': float(np.median(grad_arr)),
        'loss_std': float(np.std(loss_arr)),
    })


class TestLEDHSV2DMAPFixedNonfixedSeedComparison:
    def test_fixed_seed(self, observations):
        _run_and_record(
            observations,
            random_seed=False,
            case_name='LEDH+OT SV2D MAP fixed_seed 20 steps',
        )

    def test_nonfixed_seed(self, observations):
        _run_and_record(
            observations,
            random_seed=True,
            case_name='LEDH+OT SV2D MAP nonfixed_seed 20 steps',
        )
