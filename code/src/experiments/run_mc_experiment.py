"""Monte Carlo experiment runner for single-filter evaluation.

Reproduces the MC comparison in Section 4 of:
  "Stiffness Mitigation in Stochastic Particle Flow Filters"

Runs one filter N times with a fixed observation and fixed truth,
saving per-run MSE and tr(P). Run both linear and optimal configs
separately then compare the two summary.json files.

Run:
    python -m src.experiments.run_mc_experiment \
        +experiment=two_sensor_bearing/two_sensor_bearing_mc_linear

    python -m src.experiments.run_mc_experiment \
        +experiment=two_sensor_bearing/two_sensor_bearing_mc_optimal
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import sys
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))))

import hydra
from omegaconf import DictConfig
import numpy as np
from pathlib import Path
import json


def run_mc_experiment(cfg: DictConfig) -> dict:
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(cfg.get('tf_log_level', 2))

    _dtype_map = {'float32': tf.float32, 'float64': tf.float64}
    dtype_tf = _dtype_map[cfg.get('dtype', 'float64')]

    truth   = np.array(cfg.truth, dtype=np.float64)
    z_fixed = np.array(cfg.z_fixed, dtype=np.float64).reshape(1, -1)
    n_mc    = int(cfg.n_mc_runs)

    filter_name = cfg.filter._target_.split('.')[-1]
    print(f"Model:  {cfg.model._target_.split('.')[-1]}")
    print(f"Filter: {filter_name}  (n_particles={cfg.filter.n_particles}, "
          f"n_lambda_steps={cfg.filter.n_lambda_steps}, "
          f"schedule_mu={cfg.filter.get('schedule_mu', 0.0)})")
    print(f"Truth:  {truth},  z_fixed: {z_fixed.flatten()}")
    print(f"MC runs: {n_mc}\n")

    model      = hydra.utils.instantiate(cfg.model, dtype=dtype_tf)
    filter_obj = hydra.utils.instantiate(cfg.filter, model=model)

    mse_list, trP_list = [], []

    for run in range(n_mc):
        rng    = np.random.default_rng(run)
        result = filter_obj.filter(z_fixed, random_state=rng)

        mse  = float(np.sum((result.means[0] - truth) ** 2))
        tr_P = float(np.trace(result.covs[0]))
        mse_list.append(mse)
        trP_list.append(tr_P)

        print(f"  Run {run+1:2d} | MSE={mse:10.4f}  tr(P)={tr_P:10.2f}")

    print()
    print("=" * 45)
    print(f"  Average | MSE={np.mean(mse_list):10.4f}  tr(P)={np.mean(trP_list):10.2f}")
    print("=" * 45)

    return {'mse': mse_list, 'trP': trP_list}


def save_and_plot(results: dict, output_dir: Path, cfg: DictConfig):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'filter': cfg.filter._target_.split('.')[-1],
        'schedule_mu': float(cfg.filter.get('schedule_mu', 0.0)),
        'n_particles': int(cfg.filter.n_particles),
        'truth': list(cfg.truth),
        'z_fixed': list(cfg.z_fixed),
        'mse': results['mse'],
        'trP': results['trP'],
        'mse_mean': float(np.mean(results['mse'])),
        'trP_mean': float(np.mean(results['trP'])),
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        n_mc   = len(results['mse'])
        x      = np.arange(n_mc)
        label  = f"schedule_mu={summary['schedule_mu']}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.bar(x, results['mse'])
        ax1.axhline(np.mean(results['mse']), color='r', linestyle='--', label=f"mean={summary['mse_mean']:.2f}")
        ax1.set_xlabel('MC run')
        ax1.set_ylabel('MSE')
        ax1.set_title(f'MSE — {label}')
        ax1.legend()

        ax2.bar(x, results['trP'])
        ax2.axhline(np.mean(results['trP']), color='r', linestyle='--', label=f"mean={summary['trP_mean']:.2f}")
        ax2.set_xlabel('MC run')
        ax2.set_ylabel('tr(P)')
        ax2.set_title(f'tr(P) — {label}')
        ax2.legend()

        fig.tight_layout()
        fig.savefig(output_dir / 'mc_results.png', dpi=150)
        plt.close(fig)
        print(f"Plot saved to {output_dir / 'mc_results.png'}")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"Results saved to {output_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    results = run_mc_experiment(cfg)

    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    save_and_plot(results, output_dir, cfg)


if __name__ == '__main__':
    main()
