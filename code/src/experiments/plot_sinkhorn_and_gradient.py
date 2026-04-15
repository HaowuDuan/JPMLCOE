"""Plot Sinkhorn convergence and OT gradient validation figures."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "report"
SINKHORN_JSON = ROOT / "code/tests/hmc/results/test_sinkhorn_convergence_trajectory.json"
GRAD_APPROX_JSON = ROOT / "code/tests/hmc/results/test_gradient_vs_numerical_lg_ot_approx.json"
GRAD_IMPLICIT_JSON = ROOT / "code/tests/hmc/results/test_gradient_vs_numerical_lg_ot_implicit.json"
SINKHORN_OUT = REPORT_DIR / "sinkhorn_convergence.png"
GRAD_OUT = REPORT_DIR / "gradient_validation.png"


def _load_cases(path):
    with path.open("r") as f:
        return json.load(f)["cases"]


def plot_sinkhorn_convergence():
    cases = sorted(_load_cases(SINKHORN_JSON), key=lambda c: c["epsilon"])
    epsilons = np.array([case["epsilon"] for case in cases], dtype=float)
    colors = plt.cm.viridis((epsilons - epsilons.min()) / (epsilons.max() - epsilons.min()))

    fig, ax = plt.subplots(figsize=(8, 5))
    for case, color in zip(cases, colors):
        y = np.asarray(case["row_err_trajectory"], dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=color, linewidth=2.0, label=rf"$\epsilon = {case['epsilon']:g}$")

    ax.axhline(1e-6, color="0.35", linestyle="--", linewidth=1.2, label=r"$10^{-6}$ reference")
    ax.set_yscale("log")
    ax.set_title(r"Sinkhorn convergence: marginal violation vs iteration ($N = 200$)")
    ax.set_xlabel(r"Iteration number $\ell$")
    ax.set_ylabel(r"$\|\mathbf{P}\mathbf{1} - \mathbf{a}\|_\infty$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SINKHORN_OUT, dpi=150)
    plt.close(fig)
    return cases


def _gradient_rows(path):
    return sorted(_load_cases(path), key=lambda c: (c["T"], c["param_value"]))


def plot_gradient_validation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = [
        ("approx", "Extrapolation (approximate)", GRAD_APPROX_JSON),
        ("implicit", "Implicit differentiation", GRAD_IMPLICIT_JSON),
    ]
    medians = {}
    all_cases = []

    for ax, (method, title, path) in zip(axes, panels):
        rows = _gradient_rows(path)
        all_cases.extend(rows)
        theta = np.asarray([row["param_value"] for row in rows], dtype=float)
        fd_grad = np.asarray([row["fd_grad_median"] for row in rows], dtype=float)
        ad_grad = np.asarray([row["autodiff_grad"] for row in rows], dtype=float)
        rel_err = np.asarray([row["relative_error"] for row in rows], dtype=float)
        med_rel = float(np.nanmedian(rel_err))
        medians[method] = med_rel

        ax.plot(theta, fd_grad, "k-", marker="o", linewidth=2.0, markersize=4, label="Finite difference")
        ax.plot(theta, ad_grad, color="tab:blue", marker="o", linewidth=2.0, markersize=4, label="Autodiff")
        ax.set_title(f"{title}\nmedian rel. error = {med_rel:.2e}")
        ax.set_xlabel(r"obs\_noise\_std sweep value")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$\partial \log p(z_{1:T}) / \partial \sigma_{\mathrm{obs}}$")
    axes[0].legend(frameon=False)
    fig.suptitle("LG LEDH+OT gradient validation: autodiff vs finite difference")
    fig.tight_layout()
    fig.savefig(GRAD_OUT, dpi=150)
    plt.close(fig)
    return all_cases, medians


def main():
    sinkhorn_cases = plot_sinkhorn_convergence()
    gradient_cases, medians = plot_gradient_validation()

    for path in (SINKHORN_OUT, GRAD_OUT):
        print(f"{path} {path.stat().st_size} bytes")
    print("sinkhorn:", [(case["epsilon"], case["n_iter"], case["row_err_trajectory"][-1]) for case in sinkhorn_cases])
    print("gradient median relerr:", medians)
    print(f"gradient cases: {len(gradient_cases)}")


if __name__ == "__main__":
    main()
