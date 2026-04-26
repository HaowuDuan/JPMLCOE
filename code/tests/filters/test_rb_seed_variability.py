"""Cheap diagnostic: does the PF seed shift the RB bearing-likelihood peak?

Background
----------
The lever sweep (`test_rb_bias_lever_sweep.py`) showed none of N, epsilon,
or n_lambda_steps moves the argmax of log p_hat(sigma_b) — it sits at
0.113 regardless. But the filter is stochastic in its PF seed. Fixed-seed
HMC samples ONE deterministic surface determined by one PF seed. If the
argmax is seed-dependent (different seeds place the peak at different
sigma_b values), the current single-seed HMC result may be unrepresentative.

This test probes seed variability: for 10 different PF seeds, evaluate
log p_hat at the same 5-point sigma_b grid and compute the argmax per
seed.

Two possible outcomes:
  - Argmax clusters at 0.113 across all seeds -> peak is data-intrinsic,
    seed choice does not help, single-seed HMC is fine.
  - Argmax spreads across a range -> seed choice is a live dial, and a
    single-seed HMC result is one random draw from the distribution of
    possible targets.

Fixed:
- sigma_range = 0.1 (truth)
- data seed = 42 (same data as MAP/HMC)
- T = 50
- Filter settings: default (N=500, eps=0.5, n_lambda=15)
- always_resample = True
- float32
- CPU-only

Varying: PF seed [42, k] for k = 0..9.
Grid: sigma_b in {0.09, 0.10, 0.113, 0.13, 0.15}.

Output: per-seed argmax + summary stats, saved to
tests/filters/results/rb_seed_variability.json.

Runtime: 10 seeds * 5 sigma_b points = 50 forward PF evaluations.
Approx 15-25 minutes on CPU.

Run:
  python -m pytest tests/filters/test_rb_seed_variability.py -v -s
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

SIGMA_B_GRID = [0.0989, 0.101, 0.103, 0.105, 0.107, 0.109, 0.111, 0.113]

# Default filter settings (match current HMC config)
N_PARTICLES = 500
OT_EPSILON = 0.5
N_LAMBDA_STEPS = 15

# PF seeds to test: [42, k] for k = 0..9
PF_SEEDS = [(42, k) for k in range(10)]


def _make_filter(sigma_b_val):
    base_model = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=sigma_b_val,
        dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma_bearing'])
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': OT_EPSILON},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


class TestRBSeedVariability:
    def test_argmax_per_pf_seed(self):
        # Data at truth, same seed the HMC/MAP use
        truth_model = RangeBearingModel(
            sigma_range=TRUE_SIGMA_RANGE,
            sigma_bearing=TRUE_SIGMA_BEARING,
            dtype=DTYPE,
        )
        rng = np.random.default_rng(DATA_SEED)
        _, _, obs = generate_data(truth_model, T=T, rng=rng)
        obs_tf = tf.constant(obs, dtype=DTYPE)

        print(f"\n  [RB PF-seed variability  T={T}  data seed={DATA_SEED}]")
        print(f"  filter: N={N_PARTICLES}, eps={OT_EPSILON}, n_lambda={N_LAMBDA_STEPS}")
        print(f"  sigma_b grid: {SIGMA_B_GRID}")
        print(f"  pf seeds: {PF_SEEDS}")

        per_seed = []
        for seed_pair in PF_SEEDS:
            pf_seed = tf.constant(list(seed_pair), dtype=tf.int32)
            print(f"\n  === pf_seed = {seed_pair} ===")
            ll_values = []
            for sb in SIGMA_B_GRID:
                diff_model, filt = _make_filter(sb)
                ll = filt.log_marginal_likelihood_tf(obs_tf, seed=pf_seed)
                ll_val = float(ll.numpy())
                ll_values.append(ll_val)
                print(f"    sigma_b={sb:.3f}  log_p_hat={ll_val:10.4f}")
            argmax_idx = int(np.argmax(ll_values))
            argmax_sb = SIGMA_B_GRID[argmax_idx]
            print(f"    argmax sigma_b = {argmax_sb:.3f}")
            per_seed.append({
                'pf_seed': list(seed_pair),
                'log_p_hat': ll_values,
                'argmax_sigma_b': argmax_sb,
                'distance_from_truth': argmax_sb - TRUE_SIGMA_BEARING,
            })

        # Summary
        argmax_list = [r['argmax_sigma_b'] for r in per_seed]
        unique_vals, counts = np.unique(argmax_list, return_counts=True)

        print(f"\n  === SEED VARIABILITY SUMMARY ===")
        print(f"  argmax distribution across {len(PF_SEEDS)} seeds:")
        for v, c in zip(unique_vals, counts):
            bar = '#' * int(c)
            print(f"    sigma_b={v:.3f}  count={c:2d}  {bar}")
        print(f"  argmax min={min(argmax_list):.3f}  max={max(argmax_list):.3f}  "
              f"mean={np.mean(argmax_list):.4f}  median={np.median(argmax_list):.3f}")

        # Verdict
        spread = max(argmax_list) - min(argmax_list)
        at_truth = sum(1 for v in argmax_list if v == TRUE_SIGMA_BEARING)
        print(f"\n  spread: {spread:.3f}  "
              f"({at_truth}/{len(PF_SEEDS)} seeds have argmax = truth = 0.1)")
        if spread == 0:
            print(f"  => peak is seed-INDEPENDENT at sigma_b = {argmax_list[0]:.3f}. "
                  "No seed-choice dial available.")
        else:
            print(f"  => peak IS seed-dependent. Seed choice is a live dial.")

        # Save
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / 'rb_seed_variability.json'
        payload = {
            'case_name': 'RB PF-seed variability',
            'model': 'range_bearing',
            'filter': 'ledh_invertible_hmc',
            'T': T,
            'data_seed': DATA_SEED,
            'n_particles': N_PARTICLES,
            'ot_epsilon': OT_EPSILON,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'sigma_range': TRUE_SIGMA_RANGE,
            'true_sigma_bearing': TRUE_SIGMA_BEARING,
            'sigma_b_grid': list(SIGMA_B_GRID),
            'per_seed': per_seed,
            'summary': {
                'argmax_min': float(min(argmax_list)),
                'argmax_max': float(max(argmax_list)),
                'argmax_mean': float(np.mean(argmax_list)),
                'argmax_median': float(np.median(argmax_list)),
                'spread': float(spread),
                'at_truth_count': at_truth,
                'num_seeds': len(PF_SEEDS),
            },
        }
        with out_path.open('w') as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {out_path}")
