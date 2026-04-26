"""Log-likelihood surface scan through the HMC freeze point for LEDH+OT SV2D.

Purpose
-------
The production HMC config for SV2D (configs/dpf/hmc/stochastic_volatility_2d/
ledh_ot_sigma2.yaml) uses the filter with `always_resample=False` (the filter
class default), i.e. resampling is controlled by `tf.cond(ess < thresh, ...)`.
LG and RB use the same code path and do not freeze. SV2D HMC freezes at
sigma2 ~ 1.75 after 1 accepted step.

This scan discriminates between three candidate explanations that ARE
SV2D-specific (per-particle R_i = exp(x_{2,i}) gives weights with exponential
dynamic range that LG/RB do not have):

  (a) Discontinuity:  log p_hat(sigma2) has visible step jumps where the
      ESS-vs-time trajectory crosses the resample threshold for some t.
      Autodiff gradient through `tf.cond` then returns the one-sided slope,
      inconsistent with finite differences across the jump.

  (b) Curvature:      log p_hat(sigma2) is smooth but has a ridge near 1.75
      so sharp that leapfrog with step_size=0.02 overshoots and produces a
      large energy error per trajectory.

  (c) Numerical:      log p_hat(sigma2) is ragged / noisy at fine scale due
      to 29-step LEDH flow Jacobian accumulation under per-particle R.

Design
------
- always_resample=False (matches production HMC config, NOT the existing
  ledh_sv2d_loss_surface_scan.py which sets always_resample=True and therefore
  cannot observe (a)).
- N=500, T=200, float32, n_lambda=29, seed=[42, 0]: matches
  configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2.yaml exactly.
- sigma2 grid [1.0, 2.0] step 0.01 (101 points). Covers truth (1.0), config
  initial guess (1.6), HMC freeze point (~1.75), and beyond.
- Per-point records log_lik and autodiff gradient; post-hoc compares
  autodiff grad to centered-difference ll slope to catch (a).

Interpretation
--------------
- (a) signature:   max_jump / median_jump >> 10, and
                   |autodiff_grad - fd_slope| / |fd_slope| > 1 near jumps.
- (b) signature:   surface smooth, max_jump / median_jump ~ 1, but
                   |fd_slope| peaks sharply (> 50) at some sigma2.
- (c) signature:   log_lik ragged at all scales, max/median ratio large but
                   autodiff and fd agree (both spiky).

No assertion. Prints + saves curve + diagnostic stats to JSON.

Runtime ~ 30 minutes at N=500 T=200 on RTX 3090 post-compile.

Run:
  python -m pytest tests/hmc/ledh_sv2d_hmc_freeze_surface_scan.py -v -s
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
N_PARTICLES = 500            # matches production HMC config
N_LAMBDA_STEPS = 29
T = 200
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
TRUE_SIGMA2 = 1.0

SIGMA2_LO = 1.0
SIGMA2_HI = 2.0
SIGMA2_STEP = 0.01           # 101 points; HMC freeze point ~1.75 is covered

MODEL = 'stochastic_volatility_2d'
FILTER = 'ledh_ot_no_always_resample'


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
        always_resample=False,   # production HMC default; the branch LG/RB also use
    )
    return diff_model, filt


class TestLEDHSV2DHMCFreezeSurfaceScan:
    def test_sigma2_sweep(self, obs_sv2d):
        sigma2_grid = np.round(
            np.arange(SIGMA2_LO, SIGMA2_HI + 0.5 * SIGMA2_STEP, SIGMA2_STEP),
            4,
        )

        diff_model, filt = _make_filter(float(sigma2_grid[0]))

        print(
            f"\n  [LEDH+OT SV2D HMC-freeze surface scan  "
            f"N={N_PARTICLES} T={T} n_lambda={N_LAMBDA_STEPS} "
            f"always_resample=False seed=[42,0]]",
            flush=True,
        )
        print(
            f"    sweeping {len(sigma2_grid)} grid points "
            f"[{sigma2_grid[0]:.2f}..{sigma2_grid[-1]:.2f}] step={SIGMA2_STEP}",
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
            print(
                f"      [{i+1:3d}/{len(sigma2_grid)}]  sigma2={val:.3f}  "
                f"ll={ll:12.4f}  autodiff_grad={g:12.4f}",
                flush=True,
            )

        lls_arr = np.array(lls)
        grads_arr = np.array(grads)
        step = float(sigma2_grid[1] - sigma2_grid[0])

        # Centered-difference slope of ll for comparison vs autodiff grad.
        # Endpoints use forward/backward difference.
        fd_slope = np.empty_like(lls_arr)
        fd_slope[1:-1] = (lls_arr[2:] - lls_arr[:-2]) / (2.0 * step)
        fd_slope[0] = (lls_arr[1] - lls_arr[0]) / step
        fd_slope[-1] = (lls_arr[-1] - lls_arr[-2]) / step

        # Autodiff-vs-FD disagreement. Large values near jumps = signature (a).
        autodiff_fd_diff = np.abs(grads_arr - fd_slope)

        # Jump statistic on ll itself (|Δll/Δσ²| at grid scale).
        ll_slope_abs = np.abs(np.diff(lls_arr)) / step
        max_jump = float(ll_slope_abs.max())
        median_jump = float(np.median(ll_slope_abs))

        # Jump statistic on autodiff gradient.
        grad_jump = np.abs(np.diff(grads_arr)) / step
        max_grad_jump = float(grad_jump.max())
        median_grad_jump = float(np.median(grad_jump))

        print(
            f"\n  [LEDH+OT SV2D HMC-freeze surface scan summary]"
        )
        print(
            f"    sigma2 grid: {sigma2_grid[0]:.2f}..{sigma2_grid[-1]:.2f} "
            f"step={step}"
        )
        print(
            f"    ll range:    [{lls_arr.min():.4f}, {lls_arr.max():.4f}]"
        )
        print(
            f"    grad range:  [{grads_arr.min():.4f}, {grads_arr.max():.4f}]"
        )
        print(f"    max  |Δll/Δσ²|:          {max_jump:.4f}")
        print(f"    median |Δll/Δσ²|:        {median_jump:.4f}")
        print(
            f"    max/median (ll jumps):   "
            f"{max_jump / max(median_jump, 1e-12):.2f}"
        )
        print(f"    max  |Δgrad/Δσ²|:        {max_grad_jump:.4f}")
        print(f"    median |Δgrad/Δσ²|:      {median_grad_jump:.4f}")
        print(
            f"    max autodiff-vs-fd diff: {float(autodiff_fd_diff.max()):.4f}"
        )
        print(
            f"    mean autodiff-vs-fd diff:{float(autodiff_fd_diff.mean()):.4f}"
        )
        # Highlight the top-5 points with largest autodiff-vs-fd disagreement
        top5_idx = np.argsort(autodiff_fd_diff)[-5:][::-1]
        print(f"    top-5 autodiff-vs-fd disagreement points:")
        for idx in top5_idx:
            print(
                f"      sigma2={sigma2_grid[idx]:.3f}  ll={lls_arr[idx]:.4f}  "
                f"autodiff={grads_arr[idx]:.4f}  fd={fd_slope[idx]:.4f}  "
                f"diff={autodiff_fd_diff[idx]:.4f}"
            )

        save_result(__file__, {
            'case_name': (
                f'LEDH+OT SV2D HMC-freeze surface scan '
                f'N={N_PARTICLES} T={T} n_lambda={N_LAMBDA_STEPS} '
                f'always_resample=False'
            ),
            'model_name': MODEL,
            'filter_name': FILTER,
            'n_particles': N_PARTICLES,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'T': T,
            'seed': [42, 0],
            'always_resample': False,
            'sigma2_grid': sigma2_grid.tolist(),
            'log_lik': lls,
            'autodiff_grad': grads,
            'fd_slope_of_ll': fd_slope.tolist(),
            'autodiff_minus_fd': autodiff_fd_diff.tolist(),
            'll_slope_abs': ll_slope_abs.tolist(),
            'grad_jump_abs': grad_jump.tolist(),
            'max_jump_per_unit_sigma2_ll': max_jump,
            'median_jump_per_unit_sigma2_ll': median_jump,
            'max_over_median_ratio_ll': float(
                max_jump / max(median_jump, 1e-12)
            ),
            'max_jump_per_unit_sigma2_grad': max_grad_jump,
            'median_jump_per_unit_sigma2_grad': median_grad_jump,
            'max_autodiff_vs_fd_diff': float(autodiff_fd_diff.max()),
            'mean_autodiff_vs_fd_diff': float(autodiff_fd_diff.mean()),
        })
