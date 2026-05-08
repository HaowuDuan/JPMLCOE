"""Generate multi-chain trace + histogram figures and stats JSON for converged
4-chain HMC runs. Used for the Multi-Chain Convergence Diagnostics subsection
of report/main_reorganized.tex.

Outputs:
    report/figures/hmc_lg_4chain_combined.png
    report/figures/hmc_sv1d_4chain_combined.png
    report/figures/hmc_rb_4chain_combined.png
    report/converged_chain_stats.json

Stats per parameter:
    truth, posterior mean / std, 90% CI, truth offset / posterior sd,
    split R-hat, bulk ESS, tail ESS (Vehtari 2021 rank-normalized).

Usage:
    python -m analysis.plot_converged_chains
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "code" / "outputs" / "dpf" / "hmc"
FIG_DIR = REPO / "report" / "figures"
STATS_PATH = REPO / "report" / "converged_chain_stats.json"

# -----------------------------------------------------------------------------
# Convergence diagnostics: split R-hat, bulk ESS, tail ESS (Vehtari et al. 2021)
# -----------------------------------------------------------------------------

def autocorr(x: np.ndarray) -> np.ndarray:
    """Autocorrelation function via FFT for one 1-D chain."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    acf /= acf[0]
    return acf


def ess_geyer(chains: np.ndarray) -> float:
    """Effective sample size using Geyer's initial monotone sequence on
    pooled-chain autocorrelation. Standard recipe."""
    M, N = chains.shape
    if N < 4:
        return float('nan')
    rho_per_chain = np.array([autocorr(chains[m]) for m in range(M)])
    rho_mean = rho_per_chain.mean(axis=0)
    if len(rho_mean) >= 3:
        odd = rho_mean[1::2]
        even = rho_mean[2::2]
        L = min(len(odd), len(even))
        pair_sums = odd[:L] + even[:L]
    else:
        pair_sums = np.array([])
    cumulative_min = np.minimum.accumulate(pair_sums)
    cutoff = np.argmax(cumulative_min < 0)
    if cumulative_min[cutoff] >= 0:
        cutoff = len(pair_sums)
    tau = 1.0 + 2.0 * pair_sums[:cutoff].sum()
    if tau < 1.0:
        tau = 1.0
    return float(M * N / tau)


def split_rhat(chains: np.ndarray) -> float:
    """Split R-hat: split each chain in half, treat as 2M chains, compute Gelman-Rubin."""
    M, N = chains.shape
    half = N // 2
    split = np.empty((2 * M, half))
    for m in range(M):
        split[2 * m] = chains[m, :half]
        split[2 * m + 1] = chains[m, half:2 * half]
    Mp, Np = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    B = Np * np.var(chain_means, ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return float('nan')
    var_hat = ((Np - 1) / Np) * W + B / Np
    return float(np.sqrt(var_hat / W))


def rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Rank-normalize all samples across chains, then map to standard normal quantiles."""
    M, N = chains.shape
    flat = chains.flatten()
    ranks = rankdata(flat)
    z = (ranks - 0.375) / (M * N + 0.25)  # plotting positions
    from scipy.stats import norm
    z = norm.ppf(z)
    return z.reshape(M, N)


def ess_bulk(chains: np.ndarray) -> float:
    """Bulk ESS: ESS on rank-normalized samples (Vehtari 2021)."""
    return ess_geyer(rank_normalize(chains))


def fold_around_median(chains: np.ndarray) -> np.ndarray:
    """Fold each sample as |y - median(pooled)|. Used for tail R-hat."""
    med = float(np.median(chains))
    return np.abs(chains - med)


def rhat_bulk(chains: np.ndarray) -> float:
    """Rank-normalized split R-hat (Vehtari 2021). Split-R-hat applied to the
    rank-normalized chain — sensitive to any difference in distribution, not
    just mean/variance, since ranks are uniform under the null."""
    return split_rhat(rank_normalize(chains))


def rhat_tail(chains: np.ndarray) -> float:
    """Rank-normalized split R-hat on |y - median(pooled)| (Vehtari 2021).
    Catches scale/tail mismatches that bulk R-hat can miss."""
    return split_rhat(rank_normalize(fold_around_median(chains)))


def rhat_modern(chains: np.ndarray) -> float:
    """Stan's default: max(rhat_bulk, rhat_tail)."""
    return max(rhat_bulk(chains), rhat_tail(chains))


def ess_tail(chains: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> float:
    """Tail ESS: min(ESS on indicator(samples < q5), ESS on indicator(samples > q95))."""
    flat = chains.flatten()
    lo = np.quantile(flat, q_lo)
    hi = np.quantile(flat, q_hi)
    ind_lo = (chains < lo).astype(float)
    ind_hi = (chains > hi).astype(float)
    return float(min(ess_geyer(ind_lo), ess_geyer(ind_hi)))


# -----------------------------------------------------------------------------
# Stats per case
# -----------------------------------------------------------------------------

def stats_for_param(chains: np.ndarray, truth: float) -> dict:
    flat = chains.flatten()
    mean = float(flat.mean())
    std = float(flat.std(ddof=1))
    q5 = float(np.quantile(flat, 0.05))
    q95 = float(np.quantile(flat, 0.95))
    truth_offset = (mean - truth) / std if std > 0 else float('nan')
    rhat_split = split_rhat(chains)
    rhat_bulk_v = rhat_bulk(chains)
    rhat_tail_v = rhat_tail(chains)
    bulk = ess_bulk(chains)
    tail = ess_tail(chains)
    chain_means = [float(m) for m in chains.mean(axis=1)]
    return {
        "truth": truth,
        "mean": mean,
        "std": std,
        "q5": q5,
        "q95": q95,
        "truth_offset_in_std": truth_offset,
        "split_rhat": rhat_split,
        "rhat_bulk": rhat_bulk_v,
        "rhat_tail": rhat_tail_v,
        "rhat_max": max(rhat_bulk_v, rhat_tail_v),
        "ess_bulk": bulk,
        "ess_tail": tail,
        "chain_means": chain_means,
        "chain_stds": [float(s) for s in chains.std(axis=1, ddof=1)],
        "n_chains": int(chains.shape[0]),
        "n_per_chain": int(chains.shape[1]),
    }


def load_chains(model_dir: str, chain_glob: str, n_chains: int, params: list) -> dict:
    """Load samples_<param>.npy from each chain directory. Returns {param: ndarray (M, N)}."""
    out = {p: [] for p in params}
    for c in range(1, n_chains + 1):
        cdir = OUTPUTS / model_dir / f"{chain_glob}{c}"
        for p in params:
            arr = np.load(cdir / f"samples_{p}.npy")
            out[p].append(arr)
    return {p: np.array(out[p]) for p in params}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_trace_hist_pair(axs, chains: np.ndarray, truth: float, label: str,
                         param_name: str, colors=None):
    """Two axes: (left) trace overlay 4 chains; (right) pooled histogram + truth."""
    M, N = chains.shape
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, M))

    # Trace
    ax_trace = axs[0]
    for m in range(M):
        ax_trace.plot(chains[m], color=colors[m], alpha=0.7, linewidth=0.8,
                      label=f"chain {m+1}")
    ax_trace.axhline(truth, color='black', linestyle='--', linewidth=1, label=f"truth={truth}")
    ax_trace.set_xlabel("iteration")
    ax_trace.set_ylabel(param_name)
    ax_trace.set_title(f"{label} — trace")
    ax_trace.legend(fontsize=7, loc='best', ncol=2)

    # Histogram pooled
    ax_hist = axs[1]
    pooled = chains.flatten()
    ax_hist.hist(pooled, bins=40, density=True, alpha=0.7, color='C0',
                 edgecolor='white')
    ax_hist.axvline(truth, color='black', linestyle='--', linewidth=1.5,
                    label=f"truth={truth}")
    ax_hist.set_xlabel(param_name)
    ax_hist.set_ylabel("density")
    ax_hist.set_title(f"{label} — pooled posterior")
    ax_hist.legend(fontsize=8)


def plot_lg_combined(stats: dict):
    chains_bpf = load_chains("linear_gaussian", "bpf_ot_long_c", 4, ["obs_noise_std"])["obs_noise_std"]
    chains_ledh = load_chains("linear_gaussian", "ledh_ot_c", 4, ["obs_noise_std"])["obs_noise_std"]
    truth = 1.0

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_trace_hist_pair(axs[0], chains_bpf, truth, "LG, BPF+OT-long (N=2000/chain)",
                         r"$\sigma_{\mathrm{obs}}$")
    plot_trace_hist_pair(axs[1], chains_ledh, truth, "LG, LEDH+OT (N=400/chain)",
                         r"$\sigma_{\mathrm{obs}}$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hmc_lg_4chain_combined.png", dpi=140)
    plt.close(fig)

    stats["lg"] = {
        "bpf_ot_long": {"obs_noise_std": stats_for_param(chains_bpf, truth)},
        "ledh_ot": {"obs_noise_std": stats_for_param(chains_ledh, truth)},
    }


def plot_lg_kalman_family_combined(stats: dict, filters: list = None):
    """4-chain figure for KF / EKF / UKF on 1D LG.

    `filters` is a list of (config-prefix, label) tuples. Defaults to all three;
    pass a subset (e.g. KF+EKF only) when UKF chains are not yet ready.
    """
    if filters is None:
        filters = [
            ("kalman_c", "Kalman"),
            ("ekf_c", "EKF"),
            ("ukf_c", "UKF"),
        ]
    truth = 1.0
    n = len(filters)
    fig, axs = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axs = np.array([axs])

    if "lg" not in stats:
        stats["lg"] = {}
    for i, (prefix, label) in enumerate(filters):
        chains = load_chains("linear_gaussian", prefix, 4, ["obs_noise_std"])["obs_noise_std"]
        plot_trace_hist_pair(axs[i], chains, truth,
                             f"LG, {label} (N={chains.shape[1]}/chain)",
                             r"$\sigma_{\mathrm{obs}}$")
        # strip trailing "_c" so the JSON key is e.g. "kalman" not "kalman_c"
        key = prefix.rstrip("_c").rstrip("_")
        stats["lg"][key] = {"obs_noise_std": stats_for_param(chains, truth)}

    fig.tight_layout()
    out = FIG_DIR / "hmc_lg_kalman_family_4chain.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_sv1d_combined(stats: dict):
    chains = load_chains("stochastic_volatility", "ledh_ot_c", 4, ["alpha"])["alpha"]
    truth = 0.91

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    plot_trace_hist_pair(axs, chains, truth, "SV1D, LEDH+OT (N=400/chain)", r"$\alpha$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hmc_sv1d_4chain_combined.png", dpi=140)
    plt.close(fig)

    stats["sv1d"] = {
        "ledh_ot": {"alpha": stats_for_param(chains, truth)},
    }


def plot_rb_combined(stats: dict):
    data = load_chains("range_bearing", "ledh_ot_axisstep_l10_c", 4,
                       ["sigma_range", "sigma_bearing"])
    chains_r = data["sigma_range"]
    chains_b = data["sigma_bearing"]
    truth = 0.10

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_trace_hist_pair(axs[0], chains_r, truth, r"RB, $\sigma_{\mathrm{range}}$",
                         r"$\sigma_{\mathrm{range}}$")
    plot_trace_hist_pair(axs[1], chains_b, truth, r"RB, $\sigma_{\mathrm{bearing}}$",
                         r"$\sigma_{\mathrm{bearing}}$")
    fig.suptitle("Range-bearing, LEDH+OT, per-axis step (4x bearing) + L=10",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hmc_rb_4chain_combined.png", dpi=140)
    plt.close(fig)

    stats["rb"] = {
        "ledh_ot_axisstep_l10": {
            "sigma_range": stats_for_param(chains_r, truth),
            "sigma_bearing": stats_for_param(chains_b, truth),
        },
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Run all canonical figures. Pass --kalman-family to regenerate just the
    KF/EKF/UKF figure (useful while UKF chains are still running)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kalman-family", action="store_true",
                        help="Only regenerate the KF/EKF/UKF 4-chain figure.")
    parser.add_argument("--filters", nargs="+", default=None,
                        help="Subset of kalman_family filters (e.g. kalman_c ekf_c).")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # If --kalman-family, merge into existing stats rather than overwriting.
    if args.kalman_family:
        stats = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
        filters = None
        if args.filters:
            label_map = {"kalman_c": "Kalman", "ekf_c": "EKF", "ukf_c": "UKF"}
            filters = [(p, label_map.get(p, p)) for p in args.filters]
        print("Generating KF/EKF/UKF combined figure…")
        out = plot_lg_kalman_family_combined(stats, filters)
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nFigure: {out}")
        print(f"Stats:  {STATS_PATH}")
        for filt, by_param in stats.get("lg", {}).items():
            for p, s in by_param.items():
                print(f"  lg/{filt}/{p}: mean={s['mean']:.4f} ± {s['std']:.4f}, "
                      f"R̂={s['split_rhat']:.4f} (bulk={s['rhat_bulk']:.4f}, tail={s['rhat_tail']:.4f}, "
                      f"max={s['rhat_max']:.4f}), bulk_ESS={s['ess_bulk']:.0f}, "
                      f"tail_ESS={s['ess_tail']:.0f}, offset={s['truth_offset_in_std']:+.2f}σ")
        return

    stats = {}
    print("Generating LG combined figure…")
    plot_lg_combined(stats)
    print("Generating KF/EKF/UKF combined figure…")
    plot_lg_kalman_family_combined(stats)
    print("Generating SV1D combined figure…")
    plot_sv1d_combined(stats)
    print("Generating RB combined figure…")
    plot_rb_combined(stats)

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nFigures in {FIG_DIR}/")
    print(f"Stats in {STATS_PATH}")
    print()
    for model, by_filter in stats.items():
        for filt, by_param in by_filter.items():
            for p, s in by_param.items():
                print(f"  {model}/{filt}/{p}: mean={s['mean']:.4f} ± {s['std']:.4f}, "
                      f"R̂={s['split_rhat']:.4f} (bulk={s['rhat_bulk']:.4f}, tail={s['rhat_tail']:.4f}, "
                      f"max={s['rhat_max']:.4f}), bulk_ESS={s['ess_bulk']:.0f}, "
                      f"tail_ESS={s['ess_tail']:.0f}, offset={s['truth_offset_in_std']:+.2f}σ")


if __name__ == "__main__":
    main()
