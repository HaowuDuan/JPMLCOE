"""SV2D LEDH+OT loss-surface scan parametrized over particle count N.

Mirrors tests/hmc/ledh_sv2d_loss_surface_scan.py but runs at two N values
(1000 and 2000) to diagnose whether more particles improves the gradient
stability / approach to zero at the MLE.

Each value of N runs a sigma2 sweep with fixed PF seed [42, 0], T=200,
n_lambda=29, float32, always_resample=True. Output is log-likelihood and
autodiff gradient at each grid point.

CPU-forced (user requirement). At N=2000 T=200, each forward+gradient call
is ~2-3 minutes CPU. 15 grid points * 2 N values = ~75-90 min total.

Run:
  cd code
  .venv/bin/python -m pytest tests/hmc/ledh_sv2d_loss_surface_n_sweep.py -v -s
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
    autodiff_grad,
    save_result,
    reset_results,
)


DTYPE = tf.float32
N_LAMBDA_STEPS = 29
T = 200
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
TRUE_SIGMA2 = 1.0

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


def _make_filter(sigma2_val, n_particles):
    base_model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=sigma2_val, b=1.0, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma2'])
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=n_particles,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


class TestLEDHSV2DLossSurfaceNSweep:
    @pytest.mark.parametrize("n_particles", [1000, 2000])
    def test_sigma2_sweep(self, obs_sv2d, n_particles):
        # Wide grid: covers truth (1.0), intermediate, and init (2.0).
        # Expect |grad| to cross zero near the MLE.
        sigma2_grid = np.array([0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0])

        diff_model, filt = _make_filter(float(sigma2_grid[0]), n_particles)

        print(
            f"\n  [LEDH+OT SV2D loss-surface scan  "
            f"N={n_particles} T={T} n_lambda={N_LAMBDA_STEPS} seed=[42,0]]",
            flush=True,
        )
        print(
            f"    sweeping {len(sigma2_grid)} grid points "
            f"[{sigma2_grid[0]:.2f}..{sigma2_grid[-1]:.2f}] step=0.01",
            flush=True,
        )

        lls, grads = [], []
        for i, val in enumerate(sigma2_grid):
            ll, g = autodiff_grad(
                diff_model, filt, obs_sv2d,
                'sigma2', float(val), DTYPE, PF_SEED,
            )
            lls.append(ll)
            grads.append(g)
            print(
                f"      [{i+1:2d}/{len(sigma2_grid)}]  sigma2={val:.2f}  "
                f"ll={ll:10.4f}  grad={g:10.4f}",
                flush=True,
            )

        grads_arr = np.array(grads)
        # grid is non-uniform: use per-pair step
        per_pair_steps = np.diff(sigma2_grid)
        jumps_per_unit = np.abs(np.diff(grads_arr)) / per_pair_steps
        max_jump = float(jumps_per_unit.max())
        median_jump = float(np.median(jumps_per_unit))

        print(
            f"\n  [LEDH+OT SV2D N={n_particles} summary]"
        )
        print(f"    grad range:             [{grads_arr.min():.4f}, "
              f"{grads_arr.max():.4f}]")
        print(f"    |grad| range:           "
              f"[{np.abs(grads_arr).min():.4f}, {np.abs(grads_arr).max():.4f}]")
        print(f"    max  |Δgrad/Δsigma2|:   {max_jump:.4f}")
        print(f"    median |Δgrad/Δsigma2|: {median_jump:.4f}")
        print(f"    max/median ratio:       "
              f"{max_jump / max(median_jump, 1e-12):.2f}")

        save_result(__file__, {
            'case_name': f'LEDH+OT SV2D loss-surface scan N={n_particles}',
            'model_name': MODEL,
            'filter_name': FILTER,
            'n_particles': n_particles,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'T': T,
            'seed': [42, 0],
            'sigma2_grid': sigma2_grid.tolist(),
            'log_lik': lls,
            'autodiff_grad': grads,
            'jumps_per_unit_sigma2': jumps_per_unit.tolist(),
            'max_jump_per_unit_sigma2': max_jump,
            'median_jump_per_unit_sigma2': median_jump,
            'max_over_median_ratio': float(
                max_jump / max(median_jump, 1e-12)
            ),
            'grad_abs_min': float(np.abs(grads_arr).min()),
            'grad_abs_max': float(np.abs(grads_arr).max()),
            'grad_abs_mean': float(np.abs(grads_arr).mean()),
        })
