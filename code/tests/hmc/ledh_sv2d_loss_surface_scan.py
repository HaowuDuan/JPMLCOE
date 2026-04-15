"""Loss-surface smoothness scan for LEDH+OT on 2D Stochastic Volatility.

Single-purpose diagnostic. Sweeps sigma2 on a fine grid at FIXED PF seed
and records the autodiff gradient ∂(log L)/∂sigma2 at each grid point.

Config matches the production failing MAP config exactly:
  N=1000, T=200, float32, n_lambda=29, seed=[42,0], weight_clip_range=50.0,
  stop_gradient_resampling=False, ot_entropy resampling, epsilon=0.5.

Why this test exists:
  The MAP fixed-seed rerun at production config shows |grad| roughly flat
  at ~30 as sigma2 moves from 2.0 toward 1.86 over 20 Adam steps, rather
  than decreasing as one would expect if sigma2 were approaching the MLE.
  An earlier loss-surface scan at T=20 float64 showed a smooth monotone
  curve, but that scan measures a different likelihood function (over the
  first 20 observations only) and does not directly predict the T=200
  float32 behavior. This scan is the T=200 float32 version, so the curve
  is directly comparable to the MAP's fixed-seed trajectory at the
  production scale.

No assertion. Prints + saves the curve to JSON for inspection.

Run:
  python -m pytest tests/hmc/ledh_sv2d_loss_surface_scan.py -v -s
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


DTYPE = tf.float32           # match production MAP
N_PARTICLES = 1000           # match production MAP
N_LAMBDA_STEPS = 29          # match production MAP
T = 200                      # match production MAP
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


def _make_filter(sigma2_val):
    base_model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=sigma2_val, b=1.0, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma2'])
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


class TestLEDHSV2DLossSurfaceScan:
    def test_sigma2_sweep(self, obs_sv2d):
        # Narrow range: cover the sigma2 values the MAP fixed-seed rerun
        # actually traversed (2.00 -> 1.86 over 20 Adam steps). Step 0.01
        # gives 15 points; each autodiff call at T=200 is roughly the cost
        # of one MAP step (~18s), so total runtime is ~5 minutes post-compile.
        sigma2_grid = np.round(np.arange(1.86, 2.005, 0.01), 4)

        # Build the filter ONCE. Rebuilding per grid point would re-trace
        # the compiled forward each time. autodiff_grad uses
        # diff_model.update_parameters(...) to swap the scalar param tensor
        # without touching the filter's compiled graph.
        diff_model, filt = _make_filter(float(sigma2_grid[0]))

        print(
            f"\n  [LEDH+OT SV2D loss-surface scan  "
            f"N={N_PARTICLES} T={T} n_lambda={N_LAMBDA_STEPS} seed=[42,0]]",
            flush=True,
        )
        print(
            f"    sweeping {len(sigma2_grid)} grid points "
            f"[{sigma2_grid[0]:.2f}..{sigma2_grid[-1]:.2f}] step=0.01",
            flush=True,
        )

        lls = []
        grads = []
        for i, val in enumerate(sigma2_grid):
            ll, g = autodiff_grad(
                diff_model, filt, obs_sv2d,
                'sigma2', float(val), DTYPE, PF_SEED,
            )
            lls.append(ll)
            grads.append(g)
            # Per-iteration progress so it is visibly not stuck.
            print(
                f"      [{i+1:2d}/{len(sigma2_grid)}]  sigma2={val:.2f}  "
                f"ll={ll:10.4f}  grad={g:10.4f}",
                flush=True,
            )

        grads_arr = np.array(grads)
        step = float(sigma2_grid[1] - sigma2_grid[0])
        jumps_per_unit = np.abs(np.diff(grads_arr)) / step
        max_jump = float(jumps_per_unit.max())
        median_jump = float(np.median(jumps_per_unit))

        print(
            f"\n  [LEDH+OT SV2D loss-surface scan  "
            f"N={N_PARTICLES} T={T} n_lambda={N_LAMBDA_STEPS} seed=[42,0]]"
        )
        print(
            f"    sigma2 grid: {sigma2_grid[0]:.2f}..{sigma2_grid[-1]:.2f} step={step}"
        )
        print(f"    grad range:  [{grads_arr.min():.4f}, {grads_arr.max():.4f}]")
        print(f"    max  |Δgrad/Δsigma2|: {max_jump:.4f}")
        print(f"    median |Δgrad/Δsigma2|: {median_jump:.4f}")
        print(f"    max/median ratio:       {max_jump / max(median_jump, 1e-12):.2f}")
        for s, ll, g in zip(sigma2_grid, lls, grads):
            print(f"      sigma2={s:.2f}  ll={ll:.4f}  grad={g:.4f}")

        save_result(__file__, {
            'case_name': f'LEDH+OT SV2D loss-surface scan N={N_PARTICLES} T={T} n_lambda={N_LAMBDA_STEPS}',
            'model_name': MODEL,
            'filter_name': FILTER,
            'n_particles': N_PARTICLES,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'T': T,
            'seed': [42, 0],
            'sigma2_grid': sigma2_grid.tolist(),
            'log_lik': lls,
            'autodiff_grad': grads,
            'jumps_per_unit_sigma2': jumps_per_unit.tolist(),
            'max_jump_per_unit_sigma2': max_jump,
            'median_jump_per_unit_sigma2': median_jump,
            'max_over_median_ratio': float(max_jump / max(median_jump, 1e-12)),
        })
