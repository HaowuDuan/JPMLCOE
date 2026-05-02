"""Check if RB particles ever predict bearings near the +/-pi branch cut.

If yes, the missing angle-wrapping in log_observation_prob_batch is being triggered.
If no, the bug is real but doesn't affect this dataset, and the bearing R-hat issue
has another cause.

Output:
    tests/hmc/results/rb_bearing_branch_cut_check.json

Usage:
    python -m tests.hmc.rb_bearing_branch_cut_check
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import tensorflow as tf

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data


DTYPE = tf.float32
N_PARTICLES = 500
T = 50
DATA_SEED = 42
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1
PI = float(np.pi)
DANGER_BAND = 0.3       # within 0.3 rad of +/-pi
DANGER_BAND_TIGHT = 0.1

OUTPUT_PATH = Path(__file__).parent / "results" / "rb_bearing_branch_cut_check.json"


def main():
    print("=== RB bearing branch-cut check ===")
    print(f"  N={N_PARTICLES}, T={T}, true sigma=({TRUE_SIGMA_RANGE}, {TRUE_SIGMA_BEARING})")
    print()

    model = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=TRUE_SIGMA_BEARING,
        dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    states, _, obs = generate_data(model, T=T, rng=rng)
    states = np.asarray(states)
    obs = np.asarray(obs)
    print(f"  observed bearings: min={obs[:, 1].min():.3f}, max={obs[:, 1].max():.3f}, "
          f"mean={obs[:, 1].mean():.3f}")
    if states.ndim == 2:
        print(f"  true state min/max: x in [{states[:, 0].min():.2f}, {states[:, 0].max():.2f}], "
              f"y in [{states[:, 1].min():.2f}, {states[:, 1].max():.2f}]")
    else:
        print(f"  true state shape: {states.shape}, dtype={states.dtype}")
    print()

    obs_tf = tf.constant(obs, dtype=DTYPE)

    # Sample N particles from initial distribution
    L = tf.linalg.cholesky(model.Sigma_0)
    z = tf.random.stateless_normal([N_PARTICLES, 2], seed=[42, 0], dtype=DTYPE)
    particles = model.mu_0 + tf.linalg.matmul(z, L, transpose_b=True)

    bearings_per_step = []
    in_danger_band = []
    in_tight_band = []
    state_extents = []

    for t in range(T):
        dx = particles[:, 0] - model.sensor_pos[0]
        dy = particles[:, 1] - model.sensor_pos[1]
        bearings = tf.atan2(dy, dx).numpy()
        bearings_per_step.append({
            "t": t,
            "min": float(bearings.min()),
            "max": float(bearings.max()),
            "mean": float(bearings.mean()),
            "std": float(bearings.std()),
            "n_near_branch_cut": int(np.sum(np.abs(np.abs(bearings) - PI) < DANGER_BAND)),
        })
        in_danger_band.append(np.sum(np.abs(np.abs(bearings) - PI) < DANGER_BAND))
        in_tight_band.append(np.sum(np.abs(np.abs(bearings) - PI) < DANGER_BAND_TIGHT))
        state_extents.append({
            "t": t,
            "x_min": float(particles[:, 0].numpy().min()),
            "x_max": float(particles[:, 0].numpy().max()),
            "y_min": float(particles[:, 1].numpy().min()),
            "y_max": float(particles[:, 1].numpy().max()),
        })

        # Propagate via state transition (additive Gaussian noise)
        Q = model.Q.numpy() if not hasattr(model.Q, 'numpy') else model.Q.numpy()
        # F is identity by default for RB; use state_transition_mean
        means = particles  # static state model: X_{t+1} = X_t + noise
        L_Q = tf.linalg.cholesky(model.Q)
        z = tf.random.stateless_normal([N_PARTICLES, 2], seed=[42, t + 1], dtype=DTYPE)
        particles = means + tf.linalg.matmul(z, L_Q, transpose_b=True)

    total_near_cut = int(sum(in_danger_band))
    total_near_cut_tight = int(sum(in_tight_band))
    total_evaluations = N_PARTICLES * T

    print(f"  total particle-bearing evaluations: {total_evaluations}")
    print(f"  particles within {DANGER_BAND} rad of +/-pi: {total_near_cut} "
          f"({100*total_near_cut/total_evaluations:.4f}%)")
    print(f"  particles within {DANGER_BAND_TIGHT} rad of +/-pi: {total_near_cut_tight} "
          f"({100*total_near_cut_tight/total_evaluations:.4f}%)")

    if total_near_cut == 0:
        verdict = "BUG NOT TRIGGERED — no particles approach +/-pi in this dataset"
    elif total_near_cut_tight == 0:
        verdict = f"MARGINAL — particles approach but do not cross +/-pi"
    else:
        verdict = f"BUG TRIGGERED — {total_near_cut_tight} particle-bearing evaluations within {DANGER_BAND_TIGHT} rad of branch cut"
    print(f"\n  VERDICT: {verdict}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "config": {
                "n_particles": N_PARTICLES,
                "T": T,
                "data_seed": DATA_SEED,
                "true_sigma_range": TRUE_SIGMA_RANGE,
                "true_sigma_bearing": TRUE_SIGMA_BEARING,
                "danger_band_rad": DANGER_BAND,
                "danger_band_tight_rad": DANGER_BAND_TIGHT,
                "dtype": DTYPE.name,
            },
            "verdict": verdict,
            "total_particle_bearings": total_evaluations,
            "n_near_branch_cut_loose": total_near_cut,
            "n_near_branch_cut_tight": total_near_cut_tight,
            "obs_bearing_range": [float(obs[:, 1].min()), float(obs[:, 1].max())],
            "per_step": bearings_per_step,
            "state_extents": state_extents,
        }, f, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
