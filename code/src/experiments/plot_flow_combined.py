"""One-off combined flow comparison plots for report figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/Users/haowuduan/Documents/githubrepos/HMC-DPF-OT")
BETA = 0.5
SV2D_B = 1.0
SENSOR_POS = np.array([0.0, 0.0], dtype=np.float64)
LG_H = 1.0
CORE_REQUIRED = ("means", "covs", "states", "observations")
INITIAL_FILES = ("initial_state", "initial_mean", "initial_cov")


def load_and_prepend(run_dir: Path) -> tuple[dict[str, np.ndarray] | None, str | None]:
    missing = [run_dir / f"{name}.npy" for name in CORE_REQUIRED if not (run_dir / f"{name}.npy").exists()]
    if missing:
        return None, f"missing {missing[0]}"
    data = {name: np.load(run_dir / f"{name}.npy") for name in CORE_REQUIRED}
    has_initial = all((run_dir / f"{name}.npy").exists() for name in INITIAL_FILES)
    if has_initial:
        data.update({name: np.load(run_dir / f"{name}.npy") for name in INITIAL_FILES})
    for name, arr in data.items():
        if not np.isfinite(arr).all():
            return None, f"{run_dir / (name + '.npy')} contains non-finite values"
    if has_initial:
        data["states_full"] = np.vstack([data["initial_state"][None, :], data["states"]])
        data["means_full"] = np.vstack([data["initial_mean"][None, :], data["means"]])
        data["covs_full"] = np.concatenate([data["initial_cov"][None, :, :], data["covs"]])
        data["has_initial"] = True
    else:
        print(f"WARNING: {run_dir} has no complete initial_*.npy set; plotting from states[0]")
        data["states_full"] = data["states"]
        data["means_full"] = data["means"]
        data["covs_full"] = data["covs"]
        data["has_initial"] = False
    return data, None


def validate_pair(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> str | None:
    for key in ("states", "observations"):
        if not np.allclose(left[key], right[key]):
            return f"{key}.npy differs between runs"
    if left["has_initial"] and right["has_initial"] and not np.allclose(left["initial_state"], right["initial_state"]):
        return "initial_state.npy differs between runs"
    if left["means"].shape != right["means"].shape or left["covs"].shape != right["covs"].shape:
        return "means/covs shapes differ between runs"
    return None


def validate_many(runs: list[dict[str, np.ndarray]]) -> str | None:
    for run in runs[1:]:
        err = validate_pair(runs[0], run)
        if err:
            return err
    return None


def ci95(means: np.ndarray, covs: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    delta = 1.96 * np.sqrt(np.maximum(covs[:, dim, dim], 0.0))
    return means[:, dim] - delta, means[:, dim] + delta


def sv_observation_proxy(observations_log_y2: np.ndarray) -> np.ndarray:
    return observations_log_y2[:, 0] - np.log(BETA ** 2)


def rb_observation_position(observations: np.ndarray) -> np.ndarray:
    ranges = observations[:, 0]
    bearings = observations[:, 1]
    return np.column_stack([SENSOR_POS[0] + ranges * np.cos(bearings), SENSOR_POS[1] + ranges * np.sin(bearings)])


def sv2d_observation_proxy(observations: np.ndarray) -> np.ndarray:
    proxy = np.full((observations.shape[0], 2), np.nan, dtype=np.float64)
    proxy[:, 0] = observations[:, 0] / SV2D_B
    return proxy


def lg_observation_proxy(observations: np.ndarray) -> np.ndarray:
    return observations[:, 0] / LG_H


def plot_line_band(ax, t, data, dim, color, label):
    lo, hi = ci95(data["means_full"], data["covs_full"], dim)
    ax.plot(t, data["means_full"][:, dim], color=color, linewidth=1.5, label=f"{label} mean")
    ax.fill_between(t, lo, hi, color=color, alpha=0.2, label=f"{label} 95% CI")


def plot_sv(left, right, labels, output: Path, title: str) -> float:
    t_full = np.arange(left["means_full"].shape[0])
    t_obs = np.arange(1, left["observations"].shape[0] + 1)
    obs_proxy = sv_observation_proxy(left["observations"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    ax.plot(t_full, left["states_full"][:, 0], color="black", linewidth=1.8, label="Truth")
    plot_line_band(ax, t_full, left, 0, "tab:blue", labels[0])
    plot_line_band(ax, t_full, right, 0, "tab:green", labels[1])
    ax.plot(t_obs, obs_proxy, linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation proxy")
    ax.set_xlabel("Time")
    ax.set_ylabel("x_t")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return float(np.max(np.abs(left["means_full"][:, 0] - right["means_full"][:, 0])))


def plot_rb(left, right, labels, output: Path, title: str) -> float:
    t_full = np.arange(left["means_full"].shape[0])
    t_obs = np.arange(1, left["observations"].shape[0] + 1)
    obs_pos = rb_observation_position(left["observations"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    for dim, ax in enumerate(axes):
        ax.plot(t_full, left["states_full"][:, dim], color="black", linewidth=1.8, label="Truth")
        plot_line_band(ax, t_full, left, dim, "tab:blue", labels[0])
        plot_line_band(ax, t_full, right, dim, "tab:green", labels[1])
        ax.plot(t_obs, obs_pos[:, dim], linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation back-projection")
        ax.set_title(f"State {dim + 1} ({'x' if dim == 0 else 'y'} position)")
        ax.set_ylabel("Position")
        ax.grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set_xlabel("Time")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return float(np.max(np.abs(left["means_full"] - right["means_full"])))


def plot_stochastic_edh_rb_comparison(left, right, labels, output: Path, title: str) -> float:
    t_full = np.arange(left["means_full"].shape[0])
    t_obs = np.arange(1, left["observations"].shape[0] + 1)
    obs_pos = rb_observation_position(left["observations"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    ylabels = ("Position x", "Position y")
    for dim, ax in enumerate(axes):
        ax.plot(t_full, left["states_full"][:, dim], color="black", linewidth=1.8, label="Truth")
        plot_line_band(ax, t_full, left, dim, "tab:blue", labels[0])
        plot_line_band(ax, t_full, right, dim, "tab:green", labels[1])
        ax.plot(t_obs, obs_pos[:, dim], linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation")
        ax.set_title(f"State {dim + 1}")
        ax.set_ylabel(ylabels[dim])
        ax.grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set_xlabel("Time")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return float(np.max(np.abs(left["means_full"] - right["means_full"])))


def plot_sv2d(runs, labels, colors, output: Path, title: str) -> np.ndarray:
    t_full = np.arange(runs[0]["means_full"].shape[0])
    t_obs = np.arange(1, runs[0]["observations"].shape[0] + 1)
    obs_proxy = sv2d_observation_proxy(runs[0]["observations"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    ylabels = ("x^(1) (level)", "x^(2) (log-vol)")
    for dim, ax in enumerate(axes):
        ax.plot(t_full, runs[0]["states_full"][:, dim], color="black", linewidth=1.8, label="Truth")
        for run, label, color in zip(runs, labels, colors):
            plot_line_band(ax, t_full, run, dim, color, label)
        finite = np.isfinite(obs_proxy[:, dim])
        if np.any(finite):
            ax.plot(t_obs[finite], obs_proxy[finite, dim], linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation back-projection")
        ax.set_title(f"State {dim + 1}")
        ax.set_ylabel(ylabels[dim])
        ax.grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set_xlabel("Time")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    ref = runs[0]["means_full"]
    return np.max([np.max(np.abs(run["means_full"] - ref), axis=0) for run in runs[1:]], axis=0)


def mean_spread(runs: list[dict[str, np.ndarray]]) -> np.ndarray:
    stacked = np.stack([run["means_full"] for run in runs], axis=0)
    return np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0), axis=0)


def plot_pf_edh_ledh_rb(runs, labels, colors, output: Path, title: str) -> np.ndarray:
    t_full = np.arange(runs[0]["means_full"].shape[0])
    t_obs = np.arange(1, runs[0]["observations"].shape[0] + 1)
    obs_pos = rb_observation_position(runs[0]["observations"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    ylabels = ("Position x", "Position y")
    for dim, ax in enumerate(axes):
        ax.plot(t_full, runs[0]["states_full"][:, dim], color="black", linewidth=1.8, label="Truth")
        for run, label, color in zip(runs, labels, colors):
            plot_line_band(ax, t_full, run, dim, color, label)
        ax.plot(t_obs, obs_pos[:, dim], linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation")
        ax.set_title(f"State {dim + 1}")
        ax.set_ylabel(ylabels[dim])
        ax.grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set_xlabel("Time")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return mean_spread(runs)


def plot_pf_edh_ledh_sv_log(runs, labels, colors, output: Path, title: str) -> np.ndarray:
    t_full = np.arange(runs[0]["means_full"].shape[0])
    t_obs = np.arange(1, runs[0]["observations"].shape[0] + 1)
    obs_proxy = sv_observation_proxy(runs[0]["observations"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    ax.plot(t_full, runs[0]["states_full"][:, 0], color="black", linewidth=1.8, label="Truth")
    for run, label, color in zip(runs, labels, colors):
        plot_line_band(ax, t_full, run, 0, color, label)
    ax.plot(t_obs, obs_proxy, linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation")
    ax.set_xlabel("Time")
    ax.set_ylabel("x_t (log-volatility)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return mean_spread(runs)


def make_plot(spec: dict) -> tuple[bool, float | None, str | None]:
    left, err = load_and_prepend(spec["left"])
    if err:
        print(f"WARNING: skipping {spec['output']}: {err}")
        return False, None, err
    right, err = load_and_prepend(spec["right"])
    if err:
        print(f"WARNING: skipping {spec['output']}: {err}")
        return False, None, err
    err = validate_pair(left, right)
    if err:
        print(f"WARNING: skipping {spec['output']}: {err}")
        return False, None, err
    diff = spec["plotter"](left, right, spec["labels"], spec["output"], spec["title"])
    print(f"produced {spec['output']} | max abs mean diff={diff:.6g}")
    return True, diff, None


def make_multi_plot(spec: dict) -> tuple[bool, np.ndarray | None, str | None]:
    runs = []
    for run_dir in spec["runs"]:
        run, err = load_and_prepend(run_dir)
        if err:
            print(f"WARNING: skipping {spec['output']}: {err}")
            return False, None, err
        runs.append(run)
    err = validate_many(runs)
    if err:
        print(f"WARNING: skipping {spec['output']}: {err}")
        return False, None, err
    diff = spec["plotter"](runs, spec["labels"], spec["colors"], spec["output"], spec["title"])
    print(f"produced {spec['output']} | max abs mean diff per dim={diff}")
    return True, diff, None


def plot_kalman_1d_linear() -> tuple[bool, float | None, tuple[tuple[int, ...], ...] | None, str | None]:
    run_dir = REPO / "code/outputs/1d_linear/1d_linear_Kalman"
    output = REPO / "report/kalman_1d_linear.png"
    data, err = load_and_prepend(run_dir)
    if err:
        print(f"WARNING: skipping {output}: {err}")
        return False, None, None, err

    t_full = np.arange(data["means_full"].shape[0])
    t_obs = np.arange(1, data["observations"].shape[0] + 1)
    obs_proxy = lg_observation_proxy(data["observations"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    fig.suptitle("1D Linear Gaussian: Kalman Filter", fontsize=14)
    ax.plot(t_full, data["states_full"][:, 0], color="black", linewidth=1.8, label="Truth")
    plot_line_band(ax, t_full, data, 0, "tab:blue", "Kalman")
    ax.plot(t_obs, obs_proxy, linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation")
    ax.set_xlabel("Time")
    ax.set_ylabel("x_t")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    max_abs_err = float(np.max(np.abs(data["means_full"][:, 0] - data["states_full"][:, 0])))
    shapes = tuple(data[name].shape for name in CORE_REQUIRED)
    print(f"produced {output} | max abs Kalman mean truth diff={max_abs_err:.6g}")
    return True, max_abs_err, shapes, None


def main() -> None:
    base_sv = REPO / "code/outputs/stochastic_volatility"
    base_sv2d = REPO / "code/outputs/stochastic_volatility_2d"
    base_rb = REPO / "code/outputs/range_bearing"
    produced = 0
    specs = [
        {
            "left": base_sv / "stochastic_volatility_edh_flow_log",
            "right": base_sv / "stochastic_volatility_ledh_flow_log",
            "labels": ("EDH flow", "LEDH flow"),
            "title": "SV (log transform): EDH flow vs LEDH flow",
            "output": REPO / "report/edh_vs_ledh_flow_sv_log.png",
            "plotter": plot_sv,
        },
        {
            "left": base_rb / "range_bearing_edh_flow",
            "right": base_rb / "range_bearing_edh_invertible",
            "labels": ("EDH flow", "EDH invertible"),
            "title": "Range-bearing: EDH flow vs EDH invertible",
            "output": REPO / "report/edh_flow_vs_invertible_rb.png",
            "plotter": plot_rb,
        },
        {
            "left": base_rb / "range_bearing_ledh_flow",
            "right": base_rb / "range_bearing_ledh_invertible",
            "labels": ("LEDH flow", "LEDH invertible"),
            "title": "Range-bearing: LEDH flow vs LEDH invertible",
            "output": REPO / "report/ledh_flow_vs_invertible_rb.png",
            "plotter": plot_rb,
        },
        {
            "left": base_rb / "range_bearing_stochastic_edh",
            "right": base_rb / "range_bearing_sde_local_correction",
            "labels": ("No correction", "With local correction"),
            "title": "Stochastic EDH flow on range-bearing: global vs local correction",
            "output": REPO / "report/stochastic_edh_rb_combined.png",
            "plotter": plot_stochastic_edh_rb_comparison,
        },
    ]
    for spec in specs:
        ok, _, _ = make_plot(spec)
        produced += int(ok)

    sv2d_specs = [
        {
            "runs": [
                base_rb / "range_bearing_pf",
                base_rb / "range_bearing_edh_flow",
                base_rb / "range_bearing_ledh_flow",
            ],
            "labels": ("BPF", "EDH flow", "LEDH flow"),
            "colors": ("tab:blue", "tab:green", "tab:orange"),
            "title": "Range-bearing: BPF vs EDH flow vs LEDH flow",
            "output": REPO / "report/pf_edh_ledh_rb.png",
            "plotter": plot_pf_edh_ledh_rb,
        },
        {
            "runs": [
                base_sv / "stochastic_volatility_pf_log",
                base_sv / "stochastic_volatility_edh_flow_log",
                base_sv / "stochastic_volatility_ledh_flow_log",
            ],
            "labels": ("BPF", "EDH flow", "LEDH flow"),
            "colors": ("tab:blue", "tab:green", "tab:orange"),
            "title": "SV (log transform): BPF vs EDH flow vs LEDH flow",
            "output": REPO / "report/pf_edh_ledh_sv_log.png",
            "plotter": plot_pf_edh_ledh_sv_log,
        },
        {
            "runs": [
                base_sv2d / "stochastic_volatility_2d_pf",
                base_sv2d / "stochastic_volatility_2d_edh_flow",
                base_sv2d / "stochastic_volatility_2d_ledh_flow",
            ],
            "labels": ("BPF", "EDH flow", "LEDH flow"),
            "colors": ("tab:blue", "tab:green", "tab:orange"),
            "title": "SV2D: BPF vs EDH flow vs LEDH flow",
            "output": REPO / "report/pf_edh_ledh_sv2d.png",
            "plotter": plot_sv2d,
        },
        {
            "runs": [
                base_sv2d / "stochastic_volatility_2d_edh_flow",
                base_sv2d / "stochastic_volatility_2d_edh_invertible",
            ],
            "labels": ("EDH flow", "EDH invertible"),
            "colors": ("tab:blue", "tab:green"),
            "title": "SV2D: EDH flow vs EDH invertible",
            "output": REPO / "report/edh_flow_vs_invertible_sv2d.png",
            "plotter": plot_sv2d,
        },
        {
            "runs": [
                base_sv2d / "stochastic_volatility_2d_edh_invertible",
                base_sv2d / "stochastic_volatility_2d_ledh_invertible",
            ],
            "labels": ("EDH invertible", "LEDH invertible"),
            "colors": ("tab:blue", "tab:green"),
            "title": "SV2D: EDH invertible vs LEDH invertible",
            "output": REPO / "report/edh_vs_ledh_invertible_sv2d.png",
            "plotter": plot_sv2d,
        },
    ]
    for spec in sv2d_specs:
        ok, _, _ = make_multi_plot(spec)
        produced += int(ok)

    ok, _, _, _ = plot_kalman_1d_linear()
    produced += int(ok)
    print(f"produced {produced} plots total")


if __name__ == "__main__":
    main()
