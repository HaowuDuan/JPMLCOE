"""Visualization utilities for HMC inference results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict


def plot_trace(
    samples: Dict[str, np.ndarray],
    true_params: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    title: str = "HMC Trace Plot",
    burn_in_samples: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Trace plot of posterior samples for each parameter.

    Args:
        samples: Dict mapping parameter names to sample arrays, shape (num_samples,).
        true_params: Optional dict of true parameter values (drawn as dashed lines).
        save_path: If provided, save figure to this path.
        title: Plot title.
        burn_in_samples: Optional dict mapping parameter names to burn-in sample arrays.
            If provided, burn-in is shown before the sampling phase with a vertical separator.
    """
    param_names = list(samples.keys())
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), squeeze=False)

    for i, name in enumerate(param_names):
        ax = axes[i, 0]
        s = np.asarray(samples[name]).flatten()

        if burn_in_samples and name in burn_in_samples:
            bi = np.asarray(burn_in_samples[name]).flatten()
            num_burnin = len(bi)
            # Plot burn-in in gray, post-burn-in in blue
            ax.plot(np.arange(num_burnin), bi, linewidth=0.5, alpha=0.4, color='gray')
            ax.plot(np.arange(num_burnin, num_burnin + len(s)), s,
                    linewidth=0.5, alpha=0.7, color='C0')
            # Vertical separator
            ax.axvline(num_burnin, color='k', linestyle='--', linewidth=1.0,
                        alpha=0.6, label=f'End burn-in ({num_burnin})')
            ax.set_xlabel('Step')
        else:
            ax.plot(s, linewidth=0.5, alpha=0.7)
            ax.set_xlabel('Sample')

        if true_params and name in true_params:
            ax.axhline(true_params[name], color='r', linestyle='--', linewidth=1.5,
                        label=f'True = {true_params[name]}')

        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trace plot saved to {save_path}")

    plt.close(fig)


def plot_map_convergence(
    param_history: Dict[str, list],
    loss_history: list,
    true_params: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    title: str = "MAP Optimization",
):
    """
    Plot parameter convergence and loss curve for MAP estimation.

    Args:
        param_history: Dict mapping parameter names to list of values per step.
        loss_history: List of loss values per step.
        true_params: Optional dict of true parameter values.
        save_path: If provided, save figure to this path.
        title: Plot title.
    """
    param_names = list(param_history.keys())
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params + 1, 1, figsize=(10, 3 * (n_params + 1)), squeeze=False)

    # Loss curve
    ax = axes[0, 0]
    ax.plot(loss_history, linewidth=1.0, alpha=0.8, color='C1')
    ax.set_xlabel('Step')
    ax.set_ylabel('Neg. Log Posterior')
    ax.set_title('Loss')
    ax.grid(True, alpha=0.3)

    # Parameter trajectories
    for i, name in enumerate(param_names):
        ax = axes[i + 1, 0]
        vals = param_history[name]
        ax.plot(vals, linewidth=1.0, alpha=0.7, color='C0')

        if true_params and name in true_params:
            ax.axhline(true_params[name], color='r', linestyle='--', linewidth=1.5,
                        label=f'True = {true_params[name]}')

        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"MAP convergence plot saved to {save_path}")

    plt.close(fig)


def plot_posterior_histograms(
    samples: Dict[str, np.ndarray],
    true_params: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Posterior Histograms",
):
    """
    Histogram of posterior samples for each parameter.

    Args:
        samples: Dict mapping parameter names to sample arrays, shape (num_samples,).
        true_params: Optional dict of true parameter values (drawn as dashed lines).
        save_path: If provided, save figure to this path.
        title: Plot title.
    """
    param_names = list(samples.keys())
    n_params = len(param_names)

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4), squeeze=False)

    for i, name in enumerate(param_names):
        ax = axes[0, i]
        s = np.asarray(samples[name]).flatten()

        ax.hist(s, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='white')

        mean_val = np.mean(s)
        ax.axvline(mean_val, color='steelblue', linestyle='-', linewidth=1.5,
                    label=f'Mean = {mean_val:.4f}')

        if true_params and name in true_params:
            ax.axvline(true_params[name], color='r', linestyle='--', linewidth=1.5,
                        label=f'True = {true_params[name]}')

        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Posterior histogram saved to {save_path}")

    plt.close(fig)
