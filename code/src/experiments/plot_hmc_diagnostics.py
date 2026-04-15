#!/usr/bin/env python3
"""Build HMC trace, posterior, and diagnostics outputs for the report."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HMC = ROOT / "code" / "outputs" / "dpf" / "hmc"
REPORT = ROOT / "report"
TABLE_PATH = Path("/tmp/hmc_diagnostics_table.tex")


LG_RUNS = {
    "kalman": ("Kalman", HMC / "linear_gaussian" / "kalman", "black"),
    "ekf": ("EKF", HMC / "linear_gaussian" / "ekf", "tab:blue"),
    "ukf": ("UKF", HMC / "linear_gaussian" / "ukf", "tab:green"),
    "ledh_ot": ("LEDH+OT", HMC / "linear_gaussian" / "ledh_ot", "tab:orange"),
    "bpf_ot": ("BPF+OT", HMC / "linear_gaussian" / "bpf_ot", "tab:purple"),
}

OTHER_RUNS = {
    "rb": ("Range-bearing LEDH+OT", HMC / "range_bearing" / "ledh_ot", ["sigma_range", "sigma_bearing"]),
    "sv1d": ("SV1D LEDH+OT", HMC / "stochastic_volatility" / "ledh_ot", ["alpha"]),
    "sv2d": ("SV2D LEDH+OT", HMC / "stochastic_volatility_2d" / "ledh_ot_sigma2", ["sigma2"]),
}

FALLBACK_TRUE = {
    "obs_noise_std": 1.0,
    "sigma_range": 0.1,
    "sigma_bearing": 0.1,
    "alpha": 0.91,
    "sigma2": 1.0,
}


def load_summary(path: Path) -> dict:
    with (path / "summary.json").open() as f:
        return json.load(f)


def load_samples(path: Path, param: str) -> np.ndarray:
    arr = np.load(path / f"samples_{param}.npy")
    return np.asarray(arr, dtype=float).squeeze()


def true_param(summary: dict, param: str) -> float:
    if param in summary.get("true_params", {}):
        return float(summary["true_params"][param])
    if param in summary.get("summary", {}) and "true" in summary["summary"][param]:
        return float(summary["summary"][param]["true"])
    model = summary.get("config", {}).get("model", {})
    if param in model:
        return float(model[param])
    return float(FALLBACK_TRUE[param])


def save(fig, name: str) -> Path:
    path = REPORT / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lg_trace(summaries: dict, samples: dict) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, (label, _path, color) in LG_RUNS.items():
        y = samples[key]["obs_noise_std"]
        ax.plot(np.arange(len(y)), y, color=color, linewidth=1.2, label=label)
    truth = true_param(summaries["kalman"], "obs_noise_std")
    ax.axhline(truth, color="black", linestyle="--", linewidth=1, label="true")
    ax.set_title("HMC trace: 1D linear Gaussian, obs_noise_std")
    ax.set_xlabel("Post-burnin sample index")
    ax.set_ylabel("obs_noise_std")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    return save(fig, "hmc_lg_trace.png")


def plot_lg_histograms(summaries: dict, samples: dict) -> list[Path]:
    out = []
    kalman = samples["kalman"]["obs_noise_std"]
    truth = true_param(summaries["kalman"], "obs_noise_std")
    for key, (label, _path, color) in LG_RUNS.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = samples[key]["obs_noise_std"]
        ax.hist(vals, bins=30, density=True, color=color, alpha=0.45, label=label)
        if key != "kalman":
            ax.hist(kalman, bins=30, density=True, histtype="step", color="0.65", linewidth=2, label="Kalman ref")
        ax.axvline(truth, color="black", linestyle="--", linewidth=1, label="true")
        ax.set_title(f"HMC posterior: 1D LG, {label}")
        ax.set_xlabel("obs_noise_std")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        out.append(save(fig, f"hmc_lg_histogram_{key}.png"))
    return out


def plot_rb(summaries: dict, samples: dict) -> list[Path]:
    params = ["sigma_range", "sigma_bearing"]
    labels = [r"$\sigma_{\mathrm{range}}$", r"$\sigma_{\mathrm{bearing}}$"]
    colors = ["tab:blue", "tab:green"]
    truths = [true_param(summaries["rb"], p) for p in params]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=False)
    for ax, p, lab, color, truth in zip(axes, params, labels, colors, truths):
        y = samples["rb"][p]
        ax.plot(np.arange(len(y)), y, color=color, linewidth=1.2)
        ax.axhline(truth, color="black", linestyle="--", linewidth=1)
        ax.set_title(lab)
        ax.set_xlabel("Post-burnin sample index")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Parameter value")
    fig.suptitle("HMC trace: range-bearing, LEDH+OT")
    trace_path = save(fig, "hmc_rb_trace.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, p, lab, color, truth in zip(axes, params, labels, colors, truths):
        ax.hist(samples["rb"][p], bins=30, density=True, color=color, alpha=0.5)
        ax.axvline(truth, color="black", linestyle="--", linewidth=1)
        ax.set_title(lab)
        ax.set_xlabel("Parameter value")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Density")
    fig.suptitle("HMC posterior: range-bearing, LEDH+OT")
    hist_path = save(fig, "hmc_rb_histogram.png")
    return [trace_path, hist_path]


def plot_single(key: str, param: str, title_name: str, summaries: dict, samples: dict) -> list[Path]:
    truth = true_param(summaries[key], param)
    y = samples[key][param]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(y)), y, color="tab:blue", linewidth=1.2)
    ax.axhline(truth, color="black", linestyle="--", linewidth=1, label="true")
    ax.set_title(f"HMC trace: {title_name}")
    ax.set_xlabel("Post-burnin sample index")
    ax.set_ylabel(param)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    trace = save(fig, f"hmc_{key}_trace.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=30, density=True, color="tab:blue", alpha=0.5)
    ax.axvline(truth, color="black", linestyle="--", linewidth=1, label="true")
    ax.set_title(f"HMC posterior: {title_name}")
    ax.set_xlabel(param)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    hist = save(fig, f"hmc_{key}_histogram.png")
    return [trace, hist]


def dict_mean(value, missing: list[str], field: str) -> float:
    if value is None:
        missing.append(field)
        return float("nan")
    if isinstance(value, dict):
        vals = [float(v) for v in value.values()]
        return float(np.mean(vals)) if vals else float("nan")
    if isinstance(value, (list, tuple)):
        return float(np.mean([float(v) for v in value]))
    return float(value)


def diag_row(model_filter: str, path: Path) -> tuple[dict, list[str]]:
    summary = load_summary(path)
    missing: list[str] = []
    diag = summary.get("diagnostics", {})
    meta = summary.get("metadata", {})
    timing = meta.get("timing", {})
    acceptance = diag.get("acceptance_rate")
    if acceptance is None:
        missing.append("acceptance_rate")
        acceptance = float("nan")
    ess = dict_mean(diag.get("ess"), missing, "ess")
    rhat = dict_mean(diag.get("rhat"), missing, "rhat")
    burn = timing.get("burnin_time_seconds")
    samp = timing.get("sampling_time_seconds")
    if burn is None or samp is None:
        missing.append("burnin_time_seconds/sampling_time_seconds")
        wall = timing.get("total_time_seconds", float("nan"))
    else:
        wall = float(burn) + float(samp)
    mean_step = timing.get("mean_step_time")
    if mean_step is None:
        missing.append("mean_step_time")
        mean_step = float("nan")
    row = {
        "Model / filter": model_filter,
        "Acceptance": float(acceptance),
        "ESS": ess,
        "split-Rhat": rhat,
        "Wall (s)": float(wall),
        "Time/step (s)": float(mean_step),
        "N_samples": int(meta.get("num_samples", -1)),
        "N_burnin": int(meta.get("num_burnin", -1)),
        "L": int(meta.get("num_leapfrog_steps", -1)),
    }
    for key in ["num_samples", "num_burnin", "num_leapfrog_steps"]:
        if key not in meta:
            missing.append(key)
    return row, missing


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&")


def write_diag_table(rows: list[dict]) -> Path:
    lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Model / filter & Acceptance & ESS & split-$\hat R$ & Wall (s) & Time/step (s) & $N_{\text{samples}}$ & $N_{\text{burnin}}$ & $L$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{latex_escape(r['Model / filter'])} & "
            f"{r['Acceptance']:.2f} & {r['ESS']:.1f} & {r['split-Rhat']:.4f} & "
            f"{r['Wall (s)']:.0f} & {r['Time/step (s)']:.2f} & "
            f"{r['N_samples']} & {r['N_burnin']} & {r['L']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    TABLE_PATH.write_text("\n".join(lines))
    return TABLE_PATH


def main() -> None:
    summaries: dict[str, dict] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    for key, (_label, path, _color) in LG_RUNS.items():
        summaries[key] = load_summary(path)
        samples[key] = {"obs_noise_std": load_samples(path, "obs_noise_std")}
    for key, (_label, path, params) in OTHER_RUNS.items():
        summaries[key] = load_summary(path)
        samples[key] = {p: load_samples(path, p) for p in params}

    outputs = [plot_lg_trace(summaries, samples)]
    outputs.extend(plot_lg_histograms(summaries, samples))
    outputs.extend(plot_rb(summaries, samples))
    outputs.extend(plot_single("sv1d", "alpha", "stochastic volatility 1D, LEDH+OT", summaries, samples))
    outputs.extend(plot_single("sv2d", "sigma2", "stochastic volatility 2D, LEDH+OT", summaries, samples))

    diag_specs = [
        ("LG Kalman", LG_RUNS["kalman"][1]),
        ("LG EKF", LG_RUNS["ekf"][1]),
        ("LG UKF", LG_RUNS["ukf"][1]),
        ("LG LEDH+OT", LG_RUNS["ledh_ot"][1]),
        ("LG BPF+OT", LG_RUNS["bpf_ot"][1]),
        ("RB LEDH+OT", OTHER_RUNS["rb"][1]),
        ("SV1D LEDH+OT", OTHER_RUNS["sv1d"][1]),
        ("SV2D LEDH+OT", OTHER_RUNS["sv2d"][1]),
    ]
    rows, missing = [], {}
    for label, path in diag_specs:
        row, miss = diag_row(label, path)
        rows.append(row)
        if miss:
            missing[label] = sorted(set(miss))
    table_path = write_diag_table(rows)

    true_values = {
        "LG obs_noise_std": true_param(summaries["kalman"], "obs_noise_std"),
        "RB sigma_range": true_param(summaries["rb"], "sigma_range"),
        "RB sigma_bearing": true_param(summaries["rb"], "sigma_bearing"),
        "SV1D alpha": true_param(summaries["sv1d"], "alpha"),
        "SV2D sigma2": true_param(summaries["sv2d"], "sigma2"),
    }
    print("True parameters:", true_values)
    for path in outputs:
        print(f"{path}: {path.stat().st_size} bytes")
    print(f"{table_path}: {table_path.stat().st_size} bytes")
    print("Diagnostics rows:")
    for row in rows:
        print(row)
    print("Missing fields:", missing)


if __name__ == "__main__":
    main()
