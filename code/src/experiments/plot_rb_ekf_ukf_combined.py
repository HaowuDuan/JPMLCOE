"""One-off combined EKF/UKF range-bearing report plot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/Users/haowuduan/Documents/githubrepos/JPMLCOE")
EKF_DIR = REPO / "code/outputs/range_bearing/range_bearing_ekf"
UKF_DIR = REPO / "code/outputs/range_bearing/range_bearing_ukf"
OUTPUT_PATH = REPO / "report/ekf_ukf_range_bearing_combined.png"

# Confirmed from code/src/models/range_bearing.py:
# sensor_pos defaults to np.array([0.0, 0.0]); observations are [range, bearing],
# and sigma_bearing is documented/used in radians.
SENSOR_POS = np.array([0.0, 0.0], dtype=np.float64)


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    data = {
        name: np.load(run_dir / f"{name}.npy")
        for name in (
            "initial_state", "initial_mean", "initial_cov",
            "means", "covs", "states", "observations",
        )
    }
    for name, arr in data.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{run_dir / (name + '.npy')} contains non-finite values")
    t_steps, dim = data["means"].shape
    if data["covs"].shape != (t_steps, dim, dim):
        raise ValueError(f"covs shape {data['covs'].shape} incompatible with means {data['means'].shape}")
    if data["states"].shape != data["means"].shape:
        raise ValueError(f"states shape {data['states'].shape} incompatible with means {data['means'].shape}")
    if data["observations"].shape != (t_steps, 2):
        raise ValueError(f"observations must be (T, 2), got {data['observations'].shape}")
    if data["initial_state"].shape != (dim,) or data["initial_mean"].shape != (dim,):
        raise ValueError("initial_state/initial_mean shape incompatible with state dimension")
    if data["initial_cov"].shape != (dim, dim):
        raise ValueError("initial_cov shape incompatible with state dimension")
    return data


def ci95(means: np.ndarray, covs: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    std = np.sqrt(np.maximum(covs[:, dim, dim], 0.0))
    delta = 1.96 * std
    return means[:, dim] - delta, means[:, dim] + delta


def observations_to_position(observations: np.ndarray) -> np.ndarray:
    ranges = observations[:, 0]
    bearings = observations[:, 1]
    return np.column_stack([SENSOR_POS[0] + ranges * np.cos(bearings), SENSOR_POS[1] + ranges * np.sin(bearings)])


def prepend_initial(run: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    full = dict(run)
    full["states"] = np.vstack([run["initial_state"][None, :], run["states"]])
    full["means"] = np.vstack([run["initial_mean"][None, :], run["means"]])
    full["covs"] = np.concatenate([run["initial_cov"][None, :, :], run["covs"]])
    return full


def plot_dimension(ax, t_full, t_obs, dim, title, truth, obs_pos, ekf, ukf, show_legend=False):
    ekf_lo, ekf_hi = ci95(ekf["means"], ekf["covs"], dim)
    ukf_lo, ukf_hi = ci95(ukf["means"], ukf["covs"], dim)

    ax.plot(t_full, truth[:, dim], color="black", linewidth=1.8, label="Truth")
    ax.plot(t_full, ekf["means"][:, dim], color="tab:blue", linewidth=1.5, label="EKF mean")
    ax.fill_between(t_full, ekf_lo, ekf_hi, color="tab:blue", alpha=0.2, label="EKF 95% CI")
    ax.plot(t_full, ukf["means"][:, dim], color="tab:green", linewidth=1.5, label="UKF mean")
    ax.fill_between(t_full, ukf_lo, ukf_hi, color="tab:green", alpha=0.2, label="UKF 95% CI")
    ax.plot(
        t_obs, obs_pos[:, dim],
        linestyle="None",
        marker="x", markersize=4.0, markeredgewidth=1.0, color="red", alpha=0.85,
        label="Observation back-projection",
    )

    ax.set_title(title)
    ax.set_ylabel("Position")
    ax.grid(True, alpha=0.35)
    if show_legend:
        ax.legend(loc="best", fontsize=9)


def main() -> None:
    ekf = load_run(EKF_DIR)
    ukf = load_run(UKF_DIR)

    if ekf["means"].shape[1] != 2:
        raise ValueError(f"Expected d=2 for range-bearing, got {ekf['means'].shape[1]}")
    if ekf["means"].shape != ukf["means"].shape or ekf["covs"].shape != ukf["covs"].shape:
        raise ValueError("EKF and UKF output shapes do not match")
    if not np.allclose(ekf["states"], ukf["states"]):
        raise ValueError("EKF and UKF states.npy differ; refusing to mix truth arrays")
    if not np.allclose(ekf["initial_state"], ukf["initial_state"]):
        raise ValueError("EKF and UKF initial_state.npy differ; refusing to mix truth arrays")
    if not np.allclose(ekf["observations"], ukf["observations"]):
        raise ValueError("EKF and UKF observations.npy differ; refusing to mix observations")

    obs_pos = observations_to_position(ekf["observations"])
    ekf_full = prepend_initial(ekf)
    ukf_full = prepend_initial(ukf)
    t_full = np.arange(ekf_full["means"].shape[0])
    t_obs = np.arange(1, ekf["observations"].shape[0] + 1)

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    fig.suptitle("Range-bearing: EKF vs UKF", fontsize=14)

    plot_dimension(axes[0], t_full, t_obs, 0, "State 1 (x position)", ekf_full["states"], obs_pos, ekf_full, ukf_full, True)
    plot_dimension(axes[1], t_full, t_obs, 1, "State 2 (y position)", ekf_full["states"], obs_pos, ekf_full, ukf_full, False)
    axes[1].set_xlabel("Time")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    saved = plt.imread(OUTPUT_PATH)
    print(f"EKF full means/covs/states, observations: {ekf_full['means'].shape}, {ekf_full['covs'].shape}, {ekf_full['states'].shape}, {ekf['observations'].shape}")
    print(f"UKF full means/covs/states, observations: {ukf_full['means'].shape}, {ukf_full['covs'].shape}, {ukf_full['states'].shape}, {ukf['observations'].shape}")
    print(f"Truth t=0: {ekf_full['states'][0]}")
    print(f"Saved figure image shape: {saved.shape}")
    print(f"Saved figure path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
