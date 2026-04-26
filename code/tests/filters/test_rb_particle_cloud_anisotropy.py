"""Diagnostic: particle-cloud anisotropy on the Range-Bearing model.

Hypothesis
----------
The MAP bias in sigma_bearing (0.113 vs truth 0.1, while sigma_range is
exact at 0.0999) is not a general architecture/LEDH bug. It is a
geometric asymmetry in the RB observation model. The two observation
Jacobians have different magnitudes at r ~= 5:

  |d range / d x|   = 1/r   ~= 0.2    (unit radial)
  |d bearing / d x| = 1/r^2 ~= 0.04   (unit tangential, divided by r)

LEDH flow pulls particles in proportion to the observation Jacobian.
So the radial direction (range-informed) concentrates strongly, the
tangential direction (bearing-informed) remains relatively spread.
The residual tangential particle spread is projected onto the bearing
channel as additional apparent noise, inflating the inferred
sigma_bearing.

This test runs the LEDH+OT filter at truth parameters, extracts the
weighted particle covariance at each time step, projects it onto
radial and tangential directions, and reports std(radial) vs
std(tangential). If tangential >> radial consistently, the geometric
anisotropy hypothesis is supported.

Expected numbers (order of magnitude):
  radial std     ~= sigma_r                  = 0.10      (range informed)
  tangential std ~= sigma_b * r              = 0.10 * 5  = 0.50 (bearing informed at r=5)
  tangential/radial ratio ~= 5

Observed tangential std, projected back to angular residual at the mean
(divide by r), adds approximately 0.5 / 5 = 0.1 rad of apparent bearing
noise. Combined with the true 0.1 rad, inferred sigma_b ~ sqrt(0.1^2 +
0.1^2) = 0.141. That matches the observed HMC posterior mean (0.140).

No assertion. Prints and saves per-step table to stdout.

Run:
  python -m pytest tests/filters/test_rb_particle_cloud_anisotropy.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Force CPU-only: avoids the Metal "could not find registered platform" error on macOS
# tensorflow-metal. Harmless on Linux/CUDA machines (won't hide a real GPU if one is there
# and the env var isn't set, since we only clobber it inside the tf.config call below).
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
from src.filters.particle.ledh_invertible import LEDHParticleFlowFilter


DTYPE = tf.float32
N_PARTICLES = 500
N_LAMBDA_STEPS = 15     # matches configs/dpf/hmc/range_bearing/ledh_ot.yaml
T = 50
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1
DATA_SEED = 42
PF_SEED = 42
SENSOR = np.array([0.0, 0.0])


def _project_cov_radial_tangential(mean, cov, sensor):
    """Project 2x2 cov onto radial (toward sensor) and tangential axes.

    Returns (radial_std, tangential_std, r).
    """
    q = mean - sensor
    r = float(np.linalg.norm(q))
    d = q / r                                   # radial unit vector
    t = np.array([-d[1], d[0]])                 # tangential unit vector (90 deg CCW)
    radial_var = float(d @ cov @ d)
    tangential_var = float(t @ cov @ t)
    return (float(np.sqrt(max(radial_var, 0.0))),
            float(np.sqrt(max(tangential_var, 0.0))),
            r)


class TestRBParticleCloudAnisotropy:
    def test_radial_vs_tangential_spread_at_truth(self):
        # Model at truth
        model = RangeBearingModel(
            sigma_range=TRUE_SIGMA_RANGE,
            sigma_bearing=TRUE_SIGMA_BEARING,
            dtype=DTYPE,
        )

        # Generate data at same seed as the HMC / MAP configs use
        rng = np.random.default_rng(DATA_SEED)
        _, states, obs = generate_data(model, T=T, rng=rng)

        # LEDH filter, no HMC wrapper -- we need the per-step covs from
        # the parent class which populates FilterResult.
        filt = LEDHParticleFlowFilter(
            model,
            n_particles=N_PARTICLES,
            n_lambda_steps=N_LAMBDA_STEPS,
            regularization=1e-8,
            weight_clip_range=50.0,
        )
        result = filt.filter(obs, random_seed=PF_SEED)

        means = result.means          # (T, 2) weighted particle means
        covs = result.covs            # (T, 2, 2) weighted particle covariances

        print()
        print(f"  [RB particle-cloud anisotropy at truth "
              f"sigma_range={TRUE_SIGMA_RANGE} sigma_bearing={TRUE_SIGMA_BEARING}"
              f" N={N_PARTICLES} T={T} n_lambda={N_LAMBDA_STEPS}]")
        print(f"  {'t':>3} {'r':>8} {'state_r':>8} "
              f"{'rad_std':>10} {'tang_std':>10} {'ratio':>8} "
              f"{'bearing_err_from_spread':>22}")

        rad_stds, tang_stds, ratios, bearing_equiv = [], [], [], []
        for t in range(T):
            m = means[t]
            P = covs[t]
            rad_std, tang_std, r = _project_cov_radial_tangential(m, P, SENSOR)
            true_r = float(np.linalg.norm(states[t]))
            # Tangential cov, divided by r, gives equivalent bearing-residual std
            bearing_residual_from_spread = tang_std / r if r > 0 else float('nan')
            ratio = tang_std / rad_std if rad_std > 0 else float('nan')

            rad_stds.append(rad_std)
            tang_stds.append(tang_std)
            ratios.append(ratio)
            bearing_equiv.append(bearing_residual_from_spread)

            if t % 5 == 0 or t == T - 1:
                print(f"  {t:>3d} {r:>8.3f} {true_r:>8.3f} "
                      f"{rad_std:>10.5f} {tang_std:>10.5f} {ratio:>8.2f} "
                      f"{bearing_residual_from_spread:>22.5f}")

        rad_stds = np.array(rad_stds)
        tang_stds = np.array(tang_stds)
        ratios = np.array(ratios)
        bearing_equiv = np.array(bearing_equiv)

        # Summary stats
        print()
        print(f"  SUMMARY (over T={T} steps):")
        print(f"    radial std:           mean={rad_stds.mean():.5f}  "
              f"median={np.median(rad_stds):.5f}  max={rad_stds.max():.5f}")
        print(f"    tangential std:       mean={tang_stds.mean():.5f}  "
              f"median={np.median(tang_stds):.5f}  max={tang_stds.max():.5f}")
        print(f"    tang/rad ratio:       mean={ratios.mean():.2f}  "
              f"median={np.median(ratios):.2f}  max={ratios.max():.2f}")
        print(f"    bearing eq from tang spread: "
              f"mean={bearing_equiv.mean():.5f} rad  "
              f"median={np.median(bearing_equiv):.5f} rad")
        print()
        print(f"  PREDICTIONS IF HYPOTHESIS HOLDS:")
        print(f"    radial std     ~ sigma_range   = {TRUE_SIGMA_RANGE}")
        print(f"    tangential std ~ sigma_b * r   = {TRUE_SIGMA_BEARING} * "
              f"{np.mean([np.linalg.norm(m) for m in means]):.2f}")
        print(f"    Inferred sigma_b from combined spread + true noise:")
        print(f"      sqrt({TRUE_SIGMA_BEARING}^2 + ({bearing_equiv.mean():.4f})^2) "
              f"= {np.sqrt(TRUE_SIGMA_BEARING**2 + bearing_equiv.mean()**2):.4f}")
        print(f"      Observed MAP sigma_bearing = 0.1129")
        print(f"      Observed HMC posterior mean sigma_bearing = 0.140")

        # Save result
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / 'rb_particle_cloud_anisotropy.json'
        payload = {
            'case_name': 'RB particle-cloud anisotropy at truth',
            'model': 'range_bearing',
            'filter': 'ledh_invertible',
            'n_particles': N_PARTICLES,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'T': T,
            'sigma_range': TRUE_SIGMA_RANGE,
            'sigma_bearing': TRUE_SIGMA_BEARING,
            'data_seed': DATA_SEED,
            'pf_seed': PF_SEED,
            'per_step': {
                't': list(range(T)),
                'radial_std': rad_stds.tolist(),
                'tangential_std': tang_stds.tolist(),
                'ratio': ratios.tolist(),
                'bearing_equiv': bearing_equiv.tolist(),
            },
            'summary': {
                'radial_std_mean': float(rad_stds.mean()),
                'tangential_std_mean': float(tang_stds.mean()),
                'ratio_mean': float(ratios.mean()),
                'bearing_equiv_mean': float(bearing_equiv.mean()),
                'inferred_sigma_b_if_combined': float(
                    np.sqrt(TRUE_SIGMA_BEARING**2 + bearing_equiv.mean()**2)
                ),
            },
        }
        with out_path.open('w') as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {out_path}")
