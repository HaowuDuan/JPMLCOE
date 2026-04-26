"""Estimate mass matrix from existing 4-chain RB samples.

Reads constrained samples (sigma_range, sigma_bearing) from each chain,
maps to unconstrained q-space via inverse softplus, computes pooled
covariance, and prints recommended mass_vector for next HMC run.

Usage:
    python -m analysis.rb_mass_matrix_from_chains
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


CHAIN_DIRS = [
    "outputs/dpf/hmc/range_bearing/ledh_ot_c1",
    "outputs/dpf/hmc/range_bearing/ledh_ot_c2",
    "outputs/dpf/hmc/range_bearing/ledh_ot_c3",
    "outputs/dpf/hmc/range_bearing/ledh_ot_c4",
]

ROOT = Path(__file__).parent.parent


def softplus_inverse(s):
    """Inverse softplus: q such that softplus(q) = s. q = log(exp(s) - 1)."""
    return np.log(np.expm1(s))


def main():
    print("=== RB mass matrix estimation from 4-chain samples ===\n")

    all_q_range = []
    all_q_bearing = []
    per_chain_stats = []

    for cd in CHAIN_DIRS:
        path = ROOT / cd
        sr = np.load(path / "samples_sigma_range.npy")
        sb = np.load(path / "samples_sigma_bearing.npy")
        # Map constrained -> unconstrained q-space
        q_r = softplus_inverse(sr)
        q_b = softplus_inverse(sb)

        per_chain_stats.append({
            "chain": cd.split("/")[-1],
            "n": len(sr),
            "sr_mean": sr.mean(), "sr_std": sr.std(),
            "sb_mean": sb.mean(), "sb_std": sb.std(),
            "q_r_mean": q_r.mean(), "q_r_std": q_r.std(),
            "q_b_mean": q_b.mean(), "q_b_std": q_b.std(),
            "q_corr": np.corrcoef(q_r, q_b)[0, 1],
        })
        all_q_range.append(q_r)
        all_q_bearing.append(q_b)

    print(f"{'chain':10s}  {'n':>5s}  {'sr_mean':>9s} {'sr_std':>8s}  {'sb_mean':>9s} {'sb_std':>8s}  {'q_r_std':>8s} {'q_b_std':>8s}  {'q_corr':>7s}")
    for s in per_chain_stats:
        print(
            f"{s['chain']:10s}  {s['n']:>5d}  "
            f"{s['sr_mean']:>9.4f} {s['sr_std']:>8.4f}  "
            f"{s['sb_mean']:>9.4f} {s['sb_std']:>8.4f}  "
            f"{s['q_r_std']:>8.4f} {s['q_b_std']:>8.4f}  "
            f"{s['q_corr']:>+7.3f}"
        )

    q_pooled = np.column_stack([
        np.concatenate(all_q_range),
        np.concatenate(all_q_bearing),
    ])
    print(f"\nPooled samples: shape = {q_pooled.shape}")

    Sigma = np.cov(q_pooled, rowvar=False)
    print(f"\nCovariance Sigma (unconstrained q-space):")
    print(f"  [[{Sigma[0,0]:.6f}, {Sigma[0,1]:+.6f}],")
    print(f"   [{Sigma[1,0]:+.6f}, {Sigma[1,1]:.6f}]]")

    corr = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
    print(f"  correlation rho = {corr:+.3f}")

    std_r = np.sqrt(Sigma[0, 0])
    std_b = np.sqrt(Sigma[1, 1])
    print(f"\nMarginal std (q-space): range={std_r:.4f}, bearing={std_b:.4f}")

    m_r = 1.0 / (std_r ** 2)
    m_b = 1.0 / (std_b ** 2)
    print(f"\nDiagonal mass = 1/std^2:")
    print(f"  mass_vector: [{m_r:.4f}, {m_b:.4f}]")

    mean_mass = 0.5 * (m_r + m_b)
    old_step = 0.001
    eq_step = old_step * np.sqrt(mean_mass)
    print(f"\nStep size guidance:")
    print(f"  old step (unit mass) = {old_step}")
    print(f"  equivalent step with new mass ~ old × sqrt(mean(mass))")
    print(f"  = {old_step} × sqrt({mean_mass:.2f}) = {eq_step:.4f}")
    print(f"  recommend new step_size = {eq_step * 0.5:.4f}  (half eq, gives DA headroom)")

    print(f"\nPaste into RB YAMLs (under dpf.hmc):")
    print(f"    step_size: {eq_step * 0.5:.4f}")
    print(f"    mass_vector: [{m_r:.4f}, {m_b:.4f}]")


if __name__ == "__main__":
    main()
