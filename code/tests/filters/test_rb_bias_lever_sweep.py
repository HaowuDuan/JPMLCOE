"""Cheap diagnostic: which lever actually moves the RB bearing-likelihood peak?

Hypothesis
----------
The current RB MAP gives sigma_bearing = 0.113 (truth 0.1). An anisotropy
diagnostic (tests/filters/test_rb_particle_cloud_anisotropy.py) showed
~0.105 is a geometric floor from the 1/r bearing Jacobian. The remaining
~0.008 is some combination of:
  (a) finite-particle PF log-likelihood bias (decays ~1/N),
  (b) OT smoothing at epsilon=0.5 blurring transport,
  (c) LEDH/EKF linearization residual across n_lambda=15 inner flow steps.

This test probes which lever moves the argmax of log p_hat(sigma_b) toward
truth, WITHOUT running HMC or MAP. For each lever setting, evaluate
log p_hat at 5 sigma_b grid points at fixed PF seed. Read off the
argmax per setting.

Settings
--------
- default:       N=500,  eps=0.5, n_lambda=15   (matches current HMC config)
- more_particles: N=2000, eps=0.5, n_lambda=15   (tests (a))
- lower_epsilon:  N=500,  eps=0.1, n_lambda=15   (tests (b))
- more_lambda:    N=500,  eps=0.5, n_lambda=29   (tests (c))

Grid: sigma_b in {0.09, 0.10, 0.113, 0.13, 0.15}. Covers truth, current
MAP, and bracket above/below.

Fixed:
- sigma_range = 0.1 (truth)
- T = 50, data seed = 42, PF seed = [42, 0]
- always_resample = True
- float32
- CPU-only (runs locally)

Output: per-setting table of log p_hat(sigma_b) and argmax sigma_b,
saved to tests/filters/results/rb_bias_lever_sweep.json.

No assertion. Prints + saves.

Runtime: 4 settings * 5 sigma_b points = 20 forward PF evaluations.
Approx 10-15 minutes on CPU.

Run:
  python -m pytest tests/filters/test_rb_bias_lever_sweep.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel


DTYPE = tf.float32
T = 50
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1
DATA_SEED = 42
PF_SEED = tf.constant([42, 0], dtype=tf.int32)

SIGMA_B_GRID = [
    0.090, 0.092, 0.094, 0.096, 0.098,
    0.100, 0.102, 0.104, 0.106, 0.108,
    0.110, 0.112, 0.114, 0.116, 0.118,
    0.120, 0.122, 0.124, 0.126, 0.128,
    0.130,
]

SETTINGS = [
    # (name, n_particles, ot_epsilon, n_lambda_steps)
    ('default',             500,  0.5, 15),
    ('more_particles',      2000, 0.5, 15),
    ('lower_epsilon_0.3',   500,  0.3, 15),
    ('lower_epsilon_0.1',   500,  0.1, 15),
    ('more_lambda',         500,  0.5, 29),
]


def _make_filter(n_particles, ot_epsilon, n_lambda_steps, sigma_b_val):
    """Build filter with given settings and model at (TRUE_SIGMA_RANGE, sigma_b_val)."""
    base_model = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=sigma_b_val,
        dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma_bearing'])
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=n_particles,
        n_lambda_steps=n_lambda_steps,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': ot_epsilon},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


class TestRBBiasLeverSweep:
    def test_argmax_per_setting(self):
        # Data at truth, same seed the HMC/MAP use
        truth_model = RangeBearingModel(
            sigma_range=TRUE_SIGMA_RANGE,
            sigma_bearing=TRUE_SIGMA_BEARING,
            dtype=DTYPE,
        )
        rng = np.random.default_rng(DATA_SEED)
        _, _, obs = generate_data(truth_model, T=T, rng=rng)
        obs_tf = tf.constant(obs, dtype=DTYPE)

        print(f"\n  [RB bias lever sweep  T={T}  data seed={DATA_SEED}  PF seed=[42,0]]")
        print(f"  sigma_b grid: {SIGMA_B_GRID}")

        all_results = {}
        for name, n_particles, ot_eps, n_lambda in SETTINGS:
            print(f"\n  === setting: {name}  "
                  f"(N={n_particles}, eps={ot_eps}, n_lambda={n_lambda}) ===")
            ll_values = []
            for sb in SIGMA_B_GRID:
                diff_model, filt = _make_filter(n_particles, ot_eps, n_lambda, sb)
                ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
                ll_val = float(ll.numpy())
                ll_values.append(ll_val)
                print(f"    sigma_b={sb:.3f}  log_p_hat={ll_val:10.4f}")
            # argmax sigma_b
            argmax_idx = int(np.argmax(ll_values))
            argmax_sb = SIGMA_B_GRID[argmax_idx]
            dist_to_truth = argmax_sb - TRUE_SIGMA_BEARING
            print(f"    argmax sigma_b = {argmax_sb:.3f}  "
                  f"(distance from truth 0.1: {dist_to_truth:+.3f})")
            all_results[name] = {
                'n_particles': n_particles,
                'ot_epsilon': ot_eps,
                'n_lambda_steps': n_lambda,
                'sigma_b_grid': list(SIGMA_B_GRID),
                'log_p_hat': ll_values,
                'argmax_sigma_b': argmax_sb,
                'distance_from_truth': dist_to_truth,
            }

        # Summary
        print(f"\n  === SUMMARY ===")
        print(f"  {'setting':<18} {'argmax':>8} {'dist_from_truth':>16}")
        for name in all_results:
            r = all_results[name]
            print(f"  {name:<18} {r['argmax_sigma_b']:>8.3f} "
                  f"{r['distance_from_truth']:>+16.3f}")

        # Save
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / 'rb_bias_lever_sweep.json'
        payload = {
            'case_name': 'RB bias lever sweep',
            'model': 'range_bearing',
            'filter': 'ledh_invertible_hmc',
            'T': T,
            'data_seed': DATA_SEED,
            'pf_seed': [42, 0],
            'sigma_range': TRUE_SIGMA_RANGE,
            'true_sigma_bearing': TRUE_SIGMA_BEARING,
            'settings': all_results,
        }
        with out_path.open('w') as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {out_path}")
