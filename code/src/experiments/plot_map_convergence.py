#!/usr/bin/env python3
"""Build MAP convergence figures for the report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "code" / "outputs" / "dpf" / "map"
REPORT = ROOT / "report"


RUNS = {
    "lg_bpf": {
        "label": "LG BPF+OT",
        "dir": OUT_ROOT / "linear_gaussian" / "bpf_ot",
        "color": "tab:blue",
        "param": "obs_noise_std",
    },
    "lg_ledh": {
        "label": "LG LEDH+OT",
        "dir": OUT_ROOT / "linear_gaussian" / "ledh_ot",
        "color": "tab:green",
        "param": "obs_noise_std",
    },
    "rb_ledh": {
        "label": "RB LEDH+OT",
        "dir": OUT_ROOT / "range_bearing" / "ledh",
        "color": "tab:blue",
        "params": ["sigma_range", "sigma_bearing"],
    },
    "sv1d_ledh": {
        "label": "SV1D LEDH+OT",
        "dir": OUT_ROOT / "stochastic_volatility" / "ledh_ot",
        "color": "tab:blue",
        "param": "alpha",
    },
    "sv2d_ledh": {
        "label": "SV2D LEDH+OT",
        "dir": OUT_ROOT / "stochastic_volatility_2d" / "ledh_ot_sigma2",
        "color": "tab:blue",
        "param": "sigma2",
    },
}

FALLBACK_TRUE = {
    "obs_noise_std": 0.3,
    "sigma_range": 0.1,
    "sigma_bearing": 0.1,
    "alpha": 0.91,
    "sigma2": 1.0,
}


def load_trace(run: dict) -> dict[str, np.ndarray]:
    path = run["dir"] / "map_trace.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        out[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return out


def load_summary(run: dict) -> dict:
    with (run["dir"] / "summary.json").open() as f:
        return json.load(f)


def true_param(summary: dict, name: str) -> float:
    if name in summary.get("true_params", {}):
        return float(summary["true_params"][name])
    model = summary.get("config", {}).get("model", {})
    if name in model:
        return float(model[name])
    return float(FALLBACK_TRUE[name])


def verify_rb_uses_ot() -> str:
    summary = load_summary(RUNS["rb_ledh"])
    filt = summary.get("config", {}).get("filter", {})
    method = filt.get("hmc_resampling_method") or filt.get("resampling_method")
    if method != "ot_entropy":
        raise RuntimeError(
            "Range-bearing ledh run is not OT resampling: "
            f"hmc_resampling_method/resampling_method={method!r}"
        )
    return str(method)


def maybe_log_grad_axis(ax, grad: np.ndarray) -> None:
    positive = grad[grad > 0]
    if positive.size and positive.max() / max(positive.min(), 1e-300) > 100:
        ax.set_yscale("log")


def save(fig, name: str) -> Path:
    path = REPORT / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_learning_rate(traces: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"]
    for color, key in zip(colors, ["lg_bpf", "lg_ledh", "rb_ledh", "sv1d_ledh", "sv2d_ledh"]):
        t = traces[key]
        ax.plot(t["step"], t["learning_rate"], label=RUNS[key]["label"], color=color)
    ax.set_title("MAP learning-rate schedules (cosine warm-down)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return save(fig, "map_learning_rate.png")


def plot_lg(traces: dict[str, dict[str, np.ndarray]], summaries: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for key in ["lg_bpf", "lg_ledh"]:
        t = traces[key]
        label = "BPF+OT" if key == "lg_bpf" else "LEDH+OT"
        axes[0].plot(t["step"], -t["log_likelihood"], color=RUNS[key]["color"], label=label)
        axes[1].plot(t["step"], t["grad_norm"], color=RUNS[key]["color"])
        axes[2].plot(t["step"], t["param_obs_noise_std"], color=RUNS[key]["color"])
    truth = true_param(summaries["lg_bpf"], "obs_noise_std")
    axes[2].axhline(truth, color="black", linestyle="--", linewidth=1, label="true")
    finish_composite(fig, axes, "MAP convergence: linear Gaussian", "$\\theta$")
    axes[0].legend(fontsize=8)
    return save(fig, "map_lg.png")


def plot_rb(traces: dict[str, dict[str, np.ndarray]], summaries: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    t = traces["rb_ledh"]
    axes[0].plot(t["step"], -t["log_likelihood"], color="tab:blue", label="LEDH+OT")
    axes[1].plot(t["step"], t["grad_norm"], color="tab:blue")
    axes[2].plot(t["step"], t["param_sigma_range"], color="tab:blue", label=r"$\sigma_{\mathrm{range}}$")
    axes[2].plot(t["step"], t["param_sigma_bearing"], color="tab:green", label=r"$\sigma_{\mathrm{bearing}}$")
    for name, color in [("sigma_range", "tab:blue"), ("sigma_bearing", "tab:green")]:
        axes[2].axhline(true_param(summaries["rb_ledh"], name), color=color, linestyle="--", linewidth=1)
    finish_composite(fig, axes, "MAP convergence: range-bearing", "$\\sigma$")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    return save(fig, "map_rb.png")


def plot_single(
    traces: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, dict],
    key: str,
    title: str,
    param: str,
    ylabel: str,
    output: str,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    t = traces[key]
    axes[0].plot(t["step"], -t["log_likelihood"], color="tab:blue", label="LEDH+OT")
    axes[1].plot(t["step"], t["grad_norm"], color="tab:blue")
    axes[2].plot(t["step"], t[f"param_{param}"], color="tab:blue")
    axes[2].axhline(true_param(summaries[key], param), color="black", linestyle="--", linewidth=1, label="true")
    finish_composite(fig, axes, title, ylabel)
    axes[0].legend(fontsize=8)
    return save(fig, output)


def finish_composite(fig, axes, title: str, param_label: str) -> None:
    fig.suptitle(title, y=0.995)
    axes[0].set_ylabel(r"$-\log p(z_{1:T} \mid \theta)$")
    axes[1].set_ylabel(r"$\|\nabla_\theta U\|$")
    axes[2].set_ylabel(param_label)
    axes[2].set_xlabel("Optimizer step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    maybe_log_grad_axis(axes[1], axes[1].lines[0].get_ydata())


def lr_schedule_status(traces: dict[str, dict[str, np.ndarray]]) -> str:
    base = traces["lg_bpf"]["learning_rate"]
    diffs = []
    for key in ["lg_ledh", "rb_ledh", "sv1d_ledh", "sv2d_ledh"]:
        arr = traces[key]["learning_rate"]
        diffs.append((key, float(np.max(np.abs(base - arr)))))
    if all(diff == 0.0 for _, diff in diffs):
        return "overlap"
    return ", ".join(f"{RUNS[key]['label']} max diff {diff:.3g}" for key, diff in diffs if diff)


def main() -> None:
    rb_method = verify_rb_uses_ot()
    traces = {key: load_trace(run) for key, run in RUNS.items()}
    summaries = {key: load_summary(run) for key, run in RUNS.items()}
    outputs = [
        plot_learning_rate(traces),
        plot_lg(traces, summaries),
        plot_rb(traces, summaries),
        plot_single(traces, summaries, "sv1d_ledh", "MAP convergence: stochastic volatility 1D", "alpha", r"$\alpha$", "map_sv1d.png"),
        plot_single(traces, summaries, "sv2d_ledh", "MAP convergence: stochastic volatility 2D", "sigma2", r"$\sigma_2$", "map_sv2d.png"),
    ]
    true_values = {
        "LG obs_noise_std": true_param(summaries["lg_bpf"], "obs_noise_std"),
        "RB sigma_range": true_param(summaries["rb_ledh"], "sigma_range"),
        "RB sigma_bearing": true_param(summaries["rb_ledh"], "sigma_bearing"),
        "SV1D alpha": true_param(summaries["sv1d_ledh"], "alpha"),
        "SV2D sigma2": true_param(summaries["sv2d_ledh"], "sigma2"),
    }
    print(f"RB resampling method: {rb_method}")
    print("True parameters:", true_values)
    print("Learning-rate schedules:", lr_schedule_status(traces))
    for path in outputs:
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
