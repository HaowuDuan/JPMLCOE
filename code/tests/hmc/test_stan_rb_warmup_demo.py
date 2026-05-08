"""4-chain RB warmup-only demo for the Stan-style HMC runner.

Runs ONLY the warmup phase (no sampling) for 4 chains on the range-bearing
model with LEDH+OT filter. Logs (eps, accept_prob) per iteration and the
adapted (eps, mass_matrix) at the end of each adaptation window. Saves to
JSON for plotting.

Purpose: demonstrate that the Stan-style windowed adaptation works on the
real RB pipeline — mass matrix evolves through windows, step size adapts via
DA, FindReasonableEpsilon reseeds DA after each metric update.

Output:
    tests/hmc/results/stan_rb_warmup_demo.json

Usage:
    python -m tests.hmc.test_stan_rb_warmup_demo
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
import tensorflow_probability as tfp

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel
from src.DF.parameter_handler import ParameterHandler
from src.DF.types import ParameterSpec
from src.DF.stan_hmc_runner import (
    DiagonalMetric,
    WarmupInit,
    default_warmup_init,
    stan_warmup,
)


# -----------------------------------------------------------------------------
# RB setup matching ledh_ot_axisstep_l10 configs (for direct comparability)
# -----------------------------------------------------------------------------

DTYPE = tf.float32
DATA_SEED = 42
T = 50
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1
N_PARTICLES = 500
N_LAMBDA_STEPS = 15

# 4 chains with varied inits, same as existing axisstep configs
CHAIN_CONFIGS = [
    {'seed': 42, 'init_sigma_range': 0.05, 'init_sigma_bearing': 0.05},
    {'seed': 43, 'init_sigma_range': 0.10, 'init_sigma_bearing': 0.10},
    {'seed': 44, 'init_sigma_range': 0.20, 'init_sigma_bearing': 0.20},
    {'seed': 45, 'init_sigma_range': 0.30, 'init_sigma_bearing': 0.30},
]

# Stan-style warmup
NUM_WARMUP = 200          # short enough to fit in user's time budget; matches existing RB num_burnin
NUM_LEAPFROG = 5          # avoid round numbers if you suspect resonance; 5 is fine for RB
TARGET_ACCEPT = 0.8

# FindReasonableEpsilon starting point. eps_init=1.0 (the default) drives the
# OT backward straight into the singular regime on LEDH+OT and triggers the
# safety net every iteration. Start small and let FindReasonableEpsilon double
# upward — the existing axisstep_l10 chains adapted to ~0.035 for range, so
# 0.005 is well below stability and gets to a sane eps in 2-3 doublings.
EPS_INIT = 0.005

# Same prior as ledh_ot_axisstep_l10 configs: LogNormal(loc=-2.05, scale=0.5)
PRIOR_LOC = -2.05
PRIOR_SCALE = 0.5

OUTPUT_PATH = Path(__file__).parent / "results" / "stan_rb_warmup_demo.json"


# -----------------------------------------------------------------------------
# Per-chain warmup runner
# -----------------------------------------------------------------------------

def build_target_log_prob_fn(filter_obj, diff_model, param_handler, observations_tf):
    """Construct the same target_log_prob_fn as StanDPFRunner.run_inference."""

    def target_log_prob_fn(unconstrained_params):
        constrained_params = param_handler.constrain(unconstrained_params)
        diff_model.update_parameters(constrained_params)
        seed = tf.constant([42, 0], dtype=tf.int32)
        log_likelihood = filter_obj.log_marginal_likelihood_tf(
            observations_tf, seed=seed
        )
        log_prior = param_handler.log_prior(constrained_params)
        return log_likelihood + log_prior

    return target_log_prob_fn


def run_one_chain(chain_idx, chain_cfg, observations):
    """Run warmup-only for one chain. Returns dict with full M/eps trajectory."""
    print(f"\n=== Chain {chain_idx + 1}/4 (seed={chain_cfg['seed']}, "
          f"init=({chain_cfg['init_sigma_range']:.2f}, "
          f"{chain_cfg['init_sigma_bearing']:.2f})) ===")

    base_model = RangeBearingModel(
        sigma_range=chain_cfg['init_sigma_range'],
        sigma_bearing=chain_cfg['init_sigma_bearing'],
        dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma_range', 'sigma_bearing'])

    param_specs = {
        'sigma_range': ParameterSpec(
            name='sigma_range',
            init_value=chain_cfg['init_sigma_range'],
            constraint='positive',
            prior=tfp.distributions.LogNormal(loc=PRIOR_LOC, scale=PRIOR_SCALE),
        ),
        'sigma_bearing': ParameterSpec(
            name='sigma_bearing',
            init_value=chain_cfg['init_sigma_bearing'],
            constraint='positive',
            prior=tfp.distributions.LogNormal(loc=PRIOR_LOC, scale=PRIOR_SCALE),
        ),
    }
    param_handler = ParameterHandler(param_specs, dtype=DTYPE)

    filter_obj = LEDHParticleFlowFilterHMC(
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

    obs_tf = tf.constant(observations, dtype=DTYPE)
    target_log_prob_fn = build_target_log_prob_fn(
        filter_obj, diff_model, param_handler, obs_tf
    )

    q0 = param_handler.unconstrained_init

    # Per-iteration log
    iter_log = {'iter': [], 'eps': [], 'accept': [], 'wall_seconds': []}
    t_start = time.time()

    def progress_cb(it, total, eps, accept):
        iter_log['iter'].append(int(it))
        iter_log['eps'].append(float(eps))
        iter_log['accept'].append(float(accept))
        iter_log['wall_seconds'].append(float(time.time() - t_start))
        # Print one line per 10% of progress
        if it == 1 or it % max(1, total // 20) == 0:
            print(f"  [warmup {it}/{total}] eps={eps:.5f} accept={accept:.0%} "
                  f"wall={time.time() - t_start:.0f}s")

    rng_seed = tf.constant([chain_cfg['seed'], 0], dtype=tf.int32)
    warmup_init = WarmupInit(
        metric=DiagonalMetric(tf.ones(2, dtype=DTYPE)),
        eps_init=EPS_INIT,
    )

    result = stan_warmup(
        target_log_prob_fn=target_log_prob_fn,
        q0=q0,
        num_warmup=NUM_WARMUP,
        num_leapfrog=NUM_LEAPFROG,
        target_accept=TARGET_ACCEPT,
        warmup_init=warmup_init,
        seed=rng_seed,
        progress_callback=progress_cb,
    )

    chain_total_seconds = time.time() - t_start
    print(f"  done. final eps={result.eps:.5f}, "
          f"final M={result.metric.M.numpy().tolist()}, "
          f"n_divergences={result.n_divergences}, "
          f"wall={chain_total_seconds:.0f}s")

    return {
        'chain_idx': chain_idx,
        'config': chain_cfg,
        'iter_log': iter_log,
        'window_summaries': [
            {
                'window_idx': w.window_idx,
                'n_iter': w.n_iter,
                'accept_rate': w.accept_rate,
                'final_eps': w.final_eps,
                'metric_M': w.metric_M,
                'is_adapt_window': w.is_adapt_window,
                'n_divergences_in_window': w.n_divergences_in_window,
                'median_dh': w.median_dh,
                'max_dh': w.max_dh,
            }
            for w in result.window_summaries
        ],
        'final_eps': result.eps,
        'final_metric_M': result.metric.M.numpy().tolist(),
        'n_divergences': result.n_divergences,
        'wall_seconds': chain_total_seconds,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=== Stan-style 4-chain RB warmup demonstration ===")
    print(f"  N={N_PARTICLES}, T={T}, n_lambda={N_LAMBDA_STEPS}")
    print(f"  num_warmup={NUM_WARMUP}, num_leapfrog={NUM_LEAPFROG}, "
          f"target_accept={TARGET_ACCEPT}, eps_init={EPS_INIT}")
    print(f"  Stan windowed adaptation: schedule will be computed by stan_warmup\n")

    # Generate observations once (shared across chains)
    truth_model = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=TRUE_SIGMA_BEARING,
        dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(truth_model, T=T, rng=rng)

    chains = []
    for i, chain_cfg in enumerate(CHAIN_CONFIGS):
        chain_result = run_one_chain(i, chain_cfg, obs)
        chains.append(chain_result)

    out = {
        'config': {
            'n_particles': N_PARTICLES,
            'T': T,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'data_seed': DATA_SEED,
            'true_sigma_range': TRUE_SIGMA_RANGE,
            'true_sigma_bearing': TRUE_SIGMA_BEARING,
            'num_warmup': NUM_WARMUP,
            'num_leapfrog': NUM_LEAPFROG,
            'target_accept': TARGET_ACCEPT,
            'eps_init': EPS_INIT,
            'prior_loc': PRIOR_LOC,
            'prior_scale': PRIOR_SCALE,
        },
        'chains': chains,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 60)
    print("\nSummary across 4 chains:")
    print(f"{'chain':>6}  {'final_eps':>10}  {'final_M[0]':>12}  {'final_M[1]':>12}  "
          f"{'n_div':>6}  {'wall':>8}")
    for c in chains:
        print(f"{c['chain_idx']+1:>6}  {c['final_eps']:>10.5f}  "
              f"{c['final_metric_M'][0]:>12.4f}  {c['final_metric_M'][1]:>12.4f}  "
              f"{c['n_divergences']:>6d}  {c['wall_seconds']:>8.0f}s")


if __name__ == '__main__':
    main()
