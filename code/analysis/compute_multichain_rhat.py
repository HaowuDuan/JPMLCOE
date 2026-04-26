"""Post-hoc multi-chain R-hat / ESS extractor.

Reads `samples_<param>.npy` from the per-chain output directories produced
by `run_hmc_multichain_rhat.sh`, stacks them into `[num_chains, num_samples]`
arrays, and computes multi-chain R-hat and ESS using TFP's
`potential_scale_reduction` and `effective_sample_size`.

Expected layout under `outputs/dpf/hmc/`:
    <model>/<filter>_c1/samples_<param>.npy    shape (num_samples,)
    <model>/<filter>_c2/samples_<param>.npy
    <model>/<filter>_c3/samples_<param>.npy
    <model>/<filter>_c4/samples_<param>.npy

Output: prints a per-experiment, per-parameter table and writes
    analysis/multichain_rhat_summary.json

Usage:
    cd code
    .venv/bin/python analysis/compute_multichain_rhat.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs' / 'dpf' / 'hmc'
ANALYSIS_OUT = ROOT / 'analysis' / 'multichain_rhat_summary.json'


EXPERIMENTS = [
    ('linear_gaussian',       'ledh_ot',      ['obs_noise_std']),
    ('linear_gaussian',       'bpf_ot',       ['obs_noise_std']),
    ('linear_gaussian',       'bpf_ot_long',  ['obs_noise_std']),
    ('range_bearing',         'ledh_ot',      ['sigma_range', 'sigma_bearing']),
    ('stochastic_volatility', 'ledh_ot',      ['alpha']),
    ('stochastic_volatility_2d', 'ledh_ot_sigma2', ['sigma2']),
]
NUM_CHAINS = 4


def load_chain_samples(model, filter_tag, param):
    """Return np.ndarray of shape (num_chains, num_samples)."""
    chains = []
    for i in range(1, NUM_CHAINS + 1):
        path = OUTPUTS / model / f'{filter_tag}_c{i}' / f'samples_{param}.npy'
        if not path.exists():
            raise FileNotFoundError(path)
        chains.append(np.asarray(np.load(path), dtype=np.float64))
    lengths = {c.shape[0] for c in chains}
    if len(lengths) != 1:
        raise ValueError(
            f'{model}/{filter_tag} {param}: chain lengths differ: {lengths}'
        )
    return np.stack(chains, axis=0)   # (num_chains, num_samples)


def rhat_ess(stacked):
    """TFP expects sample_axis first; chain_axis second.
    stacked: (num_chains, num_samples).
    """
    x = tf.constant(stacked.T, dtype=tf.float64)   # (num_samples, num_chains)
    rhat = tfp.mcmc.potential_scale_reduction(
        x, independent_chain_ndims=1
    ).numpy()
    ess = tfp.mcmc.effective_sample_size(
        x, cross_chain_dims=1
    ).numpy()
    return float(rhat), float(ess)


def main():
    summary = []
    print(f"{'experiment':<35} {'param':<18} {'R-hat':>8} {'ESS':>10} {'N/chain':>8}")
    print('-' * 85)
    for model, filter_tag, params in EXPERIMENTS:
        exp_entry = {
            'model': model,
            'filter': filter_tag,
            'num_chains': NUM_CHAINS,
            'params': {},
        }
        try:
            for p in params:
                stacked = load_chain_samples(model, filter_tag, p)
                rhat, ess = rhat_ess(stacked)
                exp_entry['params'][p] = {
                    'rhat': rhat,
                    'ess': ess,
                    'num_samples_per_chain': int(stacked.shape[1]),
                    'posterior_mean': float(np.mean(stacked)),
                    'posterior_std': float(np.std(stacked)),
                    'chain_means': [float(np.mean(c)) for c in stacked],
                    'chain_stds': [float(np.std(c)) for c in stacked],
                }
                print(
                    f"{model+'/'+filter_tag:<35} {p:<18} "
                    f"{rhat:>8.4f} {ess:>10.2f} {stacked.shape[1]:>8d}"
                )
        except FileNotFoundError as e:
            exp_entry['error'] = f'missing file: {e}'
            print(f"{model+'/'+filter_tag:<35} SKIP: missing {e.filename}")
        except Exception as e:
            exp_entry['error'] = str(e)
            print(f"{model+'/'+filter_tag:<35} ERROR: {e}")
        summary.append(exp_entry)

    ANALYSIS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ANALYSIS_OUT.open('w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {ANALYSIS_OUT}")


if __name__ == '__main__':
    main()
