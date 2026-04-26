"""1D scan of log p_hat(sigma_bearing | sigma_range=truth) with gradients.

Diagnostic for non-convergence of sigma_bearing in 4-chain HMC (R-hat=1.094 vs
sigma_range R-hat=1.003). Fixes sigma_range at truth (0.1), sweeps sigma_bearing,
records log_p_hat and d/dsigma_bearing log_p_hat at each grid point with
multiple PF seeds. Reads off whether the conditional surface is sharp/smooth
or noisy.

Output:
    tests/hmc/results/rb_bearing_likelihood_scan.json

Usage:
    python -m tests.hmc.rb_bearing_likelihood_scan
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import tensorflow as tf

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel


DTYPE = tf.float32

N_PARTICLES = 500
N_LAMBDA_STEPS = 15
T = 50
DATA_SEED = 42

TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1

COARSE_GRID = np.linspace(0.05, 0.30, 30)
DENSE_GRID = np.linspace(0.08, 0.20, 20)
DENSE_FOCUS = np.array([0.10, 0.13, 0.16])
N_SEEDS_COARSE = 5
N_SEEDS_FOCUS = 15

OUTPUT_PATH = Path(__file__).parent / "results" / "rb_bearing_likelihood_scan.json"


def make_filter(sigma_range_val, sigma_bearing_val):
    base_model = RangeBearingModel(
        sigma_range=sigma_range_val,
        sigma_bearing=sigma_bearing_val,
        dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma_bearing'])
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


def eval_ll_and_grad(obs_tf, sigma_bearing_val, pf_seed):
    """Compute log_p_hat and d log_p_hat / d sigma_bearing at the given point."""
    diff_model, filt = make_filter(TRUE_SIGMA_RANGE, sigma_bearing_val)
    x_b = tf.Variable(sigma_bearing_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        diff_model.update_parameters({'sigma_bearing': x_b})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=pf_seed)
    grad = tape.gradient(ll, x_b)
    return float(ll.numpy()), float(grad.numpy()) if grad is not None else float('nan')


def main():
    print(f"=== RB sigma_bearing 1D scan ===")
    print(f"  N={N_PARTICLES}, T={T}, n_lambda={N_LAMBDA_STEPS}, dtype={DTYPE.name}")
    print(f"  sigma_range fixed at truth = {TRUE_SIGMA_RANGE}")
    print(f"  data seed = {DATA_SEED}, true sigma_bearing = {TRUE_SIGMA_BEARING}")
    print()

    base_for_data = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=TRUE_SIGMA_BEARING,
        dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(base_for_data, T=T, rng=rng)
    obs_tf = tf.constant(obs, dtype=DTYPE)

    points = []
    for sb in COARSE_GRID:
        points.append(("coarse", float(sb), N_SEEDS_COARSE))
    for sb in DENSE_GRID:
        points.append(("dense", float(sb), N_SEEDS_COARSE))
    for sb in DENSE_FOCUS:
        points.append(("focus", float(sb), N_SEEDS_FOCUS))

    results = []
    t_start = time.time()
    total_evals = sum(p[2] for p in points)
    eval_done = 0

    for tag, sb_val, n_seeds in points:
        per_seed_ll = []
        per_seed_grad = []
        for k in range(n_seeds):
            pf_seed = tf.constant([42, k], dtype=tf.int32)
            ll, grad = eval_ll_and_grad(obs_tf, sb_val, pf_seed)
            per_seed_ll.append(ll)
            per_seed_grad.append(grad)
            eval_done += 1
        elapsed = time.time() - t_start
        eta = elapsed / eval_done * (total_evals - eval_done)
        print(
            f"  [{tag:6s}] sigma_bearing={sb_val:.4f}  "
            f"ll mean={np.mean(per_seed_ll):.3f} std={np.std(per_seed_ll):.3f}  "
            f"grad mean={np.mean(per_seed_grad):+.3f} std={np.std(per_seed_grad):.3f}  "
            f"({eval_done}/{total_evals}, ETA {eta/60:.1f} min)"
        )
        results.append({
            "tag": tag,
            "sigma_bearing": sb_val,
            "n_seeds": n_seeds,
            "ll_per_seed": per_seed_ll,
            "grad_per_seed": per_seed_grad,
            "ll_mean": float(np.mean(per_seed_ll)),
            "ll_std": float(np.std(per_seed_ll)),
            "grad_mean": float(np.mean(per_seed_grad)),
            "grad_std": float(np.std(per_seed_grad)),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "config": {
                "n_particles": N_PARTICLES,
                "n_lambda_steps": N_LAMBDA_STEPS,
                "T": T,
                "data_seed": DATA_SEED,
                "true_sigma_range": TRUE_SIGMA_RANGE,
                "true_sigma_bearing": TRUE_SIGMA_BEARING,
                "sigma_range_fixed_at": TRUE_SIGMA_RANGE,
                "dtype": DTYPE.name,
            },
            "results": results,
            "elapsed_sec": time.time() - t_start,
        }, f, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total elapsed: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
