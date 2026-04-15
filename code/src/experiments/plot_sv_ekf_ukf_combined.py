"""One-off combined EKF/UKF plots for stochastic volatility."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/Users/haowuduan/Documents/githubrepos/JPMLCOE")
BASE = REPO / "code/outputs/stochastic_volatility"
BETA = 0.5  # Confirmed from StochasticVolatilityModel default and saved configs.
Y_FLOOR = 1e-8


RUNS = {
    "raw": {
        "ekf_dir": BASE / "stochastic_volatility_ekf",
        "ukf_dir": BASE / "stochastic_volatility_ukf",
        "output": REPO / "report/ekf_ukf_sv_raw_combined.png",
        "title": "Stochastic volatility (raw): EKF vs UKF",
        "observations_are_log": False,
    },
    "log": {
        "ekf_dir": BASE / "stochastic_volatility_ekf_log",
        "ukf_dir": BASE / "stochastic_volatility_ukf_log",
        "output": REPO / "report/ekf_ukf_sv_log_combined.png",
        "title": "Stochastic volatility (log-squared transform): EKF vs UKF",
        "observations_are_log": True,
    },
}


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    names = ("initial_state", "initial_mean", "initial_cov", "means", "covs", "states", "observations")
    data = {name: np.load(run_dir / f"{name}.npy") for name in names}
    for name, arr in data.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{run_dir / (name + '.npy')} contains non-finite values")
    t_steps, dim = data["means"].shape
    if dim != 1:
        raise ValueError(f"Expected 1D SV means, got {data['means'].shape}")
    if data["covs"].shape != (t_steps, 1, 1) or data["states"].shape != (t_steps, 1):
        raise ValueError(f"Bad state/cov shapes in {run_dir}")
    if data["observations"].shape != (t_steps, 1):
        raise ValueError(f"Bad observation shape in {run_dir}: {data['observations'].shape}")
    return data


def prepend_initial(run: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    full = dict(run)
    full["states"] = np.vstack([run["initial_state"][None, :], run["states"]])
    full["means"] = np.vstack([run["initial_mean"][None, :], run["means"]])
    full["covs"] = np.concatenate([run["initial_cov"][None, :, :], run["covs"]])
    return full


def ci95(means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = 1.96 * np.sqrt(np.maximum(covs[:, 0, 0], 0.0))
    return means[:, 0] - delta, means[:, 0] + delta


def observation_proxy(observations: np.ndarray, observations_are_log: bool) -> np.ndarray:
    if observations_are_log:
        log_y2 = observations[:, 0]
    else:
        y_abs = np.maximum(np.abs(observations[:, 0]), Y_FLOOR)
        log_y2 = np.log(y_abs ** 2)
    return log_y2 - np.log(BETA ** 2)


def validate_pair(ekf: dict[str, np.ndarray], ukf: dict[str, np.ndarray]) -> None:
    for key in ("initial_state", "initial_mean", "initial_cov", "states", "observations"):
        if not np.allclose(ekf[key], ukf[key]):
            raise ValueError(f"EKF and UKF {key}.npy differ")
    if ekf["means"].shape != ukf["means"].shape or ekf["covs"].shape != ukf["covs"].shape:
        raise ValueError("EKF and UKF means/covs shapes differ")


def plot_case(name: str, cfg: dict) -> None:
    ekf_raw = load_run(cfg["ekf_dir"])
    ukf_raw = load_run(cfg["ukf_dir"])
    validate_pair(ekf_raw, ukf_raw)

    ekf = prepend_initial(ekf_raw)
    ukf = prepend_initial(ukf_raw)
    t_full = np.arange(ekf["means"].shape[0])
    t_obs = np.arange(1, ekf_raw["observations"].shape[0] + 1)
    obs_proxy = observation_proxy(ekf_raw["observations"], cfg["observations_are_log"])
    ekf_lo, ekf_hi = ci95(ekf["means"], ekf["covs"])
    ukf_lo, ukf_hi = ci95(ukf["means"], ukf["covs"])

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    fig.suptitle(cfg["title"] + "\nred x: log-squared observation back-projection, visualisation only", fontsize=13)

    ax.plot(t_full, ekf["states"][:, 0], color="black", linewidth=1.8, label="Truth")
    ax.plot(t_full, ekf["means"][:, 0], color="tab:blue", linewidth=1.5, label="EKF mean")
    ax.fill_between(t_full, ekf_lo, ekf_hi, color="tab:blue", alpha=0.2, label="EKF 95% CI")
    ax.plot(t_full, ukf["means"][:, 0], color="tab:green", linewidth=1.5, label="UKF mean")
    ax.fill_between(t_full, ukf_lo, ukf_hi, color="tab:green", alpha=0.2, label="UKF 95% CI")
    ax.plot(t_obs, obs_proxy, linestyle="None", marker="x", markersize=4, color="red", alpha=0.85, label="Observation proxy")
    ax.set_xlabel("Time")
    ax.set_ylabel("x_t (log-volatility)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    cfg["output"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["output"], dpi=150)
    plt.close(fig)

    image_shape = plt.imread(cfg["output"]).shape
    mean_diff = float(np.max(np.abs(ekf["means"][:, 0] - ukf["means"][:, 0])))
    cov_diff = float(np.max(np.abs(ekf["covs"][:, 0, 0] - ukf["covs"][:, 0, 0])))
    print(f"{name}: EKF means/covs/states/observations {ekf_raw['means'].shape}, {ekf_raw['covs'].shape}, {ekf_raw['states'].shape}, {ekf_raw['observations'].shape}")
    print(f"{name}: UKF means/covs/states/observations {ukf_raw['means'].shape}, {ukf_raw['covs'].shape}, {ukf_raw['states'].shape}, {ukf_raw['observations'].shape}")
    print(f"{name}: full plotted length {ekf['means'].shape[0]}, max |EKF-UKF mean| {mean_diff:.6g}, max cov diag diff {cov_diff:.6g}")
    print(f"{name}: saved {cfg['output']} with image shape {image_shape}")


def main() -> None:
    print(f"Using beta={BETA}")
    for name, cfg in RUNS.items():
        plot_case(name, cfg)


if __name__ == "__main__":
    main()
