"""Visualization utilities for HMC and MAP inference results."""

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
    grad_norm_history: Optional[list] = None,
    grad_history: Optional[Dict[str, list]] = None,
    log_likelihood_history: Optional[list] = None,
    log_prior_history: Optional[list] = None,
    learning_rate_history: Optional[list] = None,
):
    """
    Plot optimizer diagnostics for MAP estimation.

    Args:
        param_history: Dict mapping parameter names to list of values per step.
        loss_history: List of loss values per step.
        true_params: Optional dict of true parameter values.
        save_path: If provided, save figure to this path.
        title: Plot title.
        grad_norm_history: Optional gradient L2 norm per step.
        grad_history: Optional per-parameter gradient values per step. Gradients
            are on the unconstrained optimizer scale.
        log_likelihood_history: Optional log-likelihood values per step.
        log_prior_history: Optional log-prior values per step.
        learning_rate_history: Optional optimizer learning rate per step.
    """
    param_names = list(param_history.keys())
    n_params = len(param_names)

    def _has_values(values) -> bool:
        return values is not None and len(values) > 0

    has_objective_components = (
        _has_values(log_likelihood_history) or _has_values(log_prior_history)
    )
    has_grad_norm = _has_values(grad_norm_history)
    has_learning_rate = _has_values(learning_rate_history)
    has_grad_history = grad_history is not None and any(
        len(grad_history.get(name, [])) > 0 for name in param_names
    )

    n_rows = 1 + n_params
    n_rows += int(has_objective_components)
    n_rows += int(has_grad_norm)
    n_rows += int(has_learning_rate)
    n_rows += n_params if has_grad_history else 0

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.6 * n_rows), squeeze=False)
    row = 0

    # Loss curve
    ax = axes[row, 0]
    ax.plot(loss_history, linewidth=1.0, alpha=0.8, color='C1')
    ax.set_xlabel('Step')
    ax.set_ylabel('Neg. Log Posterior')
    ax.set_title('Loss')
    ax.grid(True, alpha=0.3)
    row += 1

    # Objective decomposition
    if has_objective_components:
        ax = axes[row, 0]
        if _has_values(log_likelihood_history):
            ax.plot(log_likelihood_history, linewidth=1.0, alpha=0.8,
                    label='Log likelihood')
        if _has_values(log_prior_history):
            ax.plot(log_prior_history, linewidth=1.0, alpha=0.8,
                    label='Log prior')
        if _has_values(loss_history):
            ax.plot(-np.asarray(loss_history), linewidth=1.0, alpha=0.5,
                    linestyle='--', label='Log posterior')
        ax.set_xlabel('Step')
        ax.set_ylabel('Log value')
        ax.set_title('Objective Components')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        row += 1

    # Gradient norm
    if has_grad_norm:
        ax = axes[row, 0]
        grad_norm = np.asarray(grad_norm_history, dtype=float)
        ax.semilogy(np.maximum(grad_norm, np.finfo(float).tiny),
                    linewidth=1.0, alpha=0.8, color='C2')
        ax.set_xlabel('Step')
        ax.set_ylabel('L2 norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)
        row += 1

    # Learning rate
    if has_learning_rate:
        ax = axes[row, 0]
        ax.plot(learning_rate_history, linewidth=1.0, alpha=0.8, color='C3')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning rate')
        ax.set_title('Learning Rate')
        ax.grid(True, alpha=0.3)
        row += 1

    # Parameter trajectories
    for name in param_names:
        ax = axes[row, 0]
        vals = param_history[name]
        ax.plot(vals, linewidth=1.0, alpha=0.7, color='C0')

        if true_params and name in true_params:
            ax.axhline(true_params[name], color='r', linestyle='--', linewidth=1.5,
                        label=f'True = {true_params[name]}')
            ax.legend(loc='upper right')

        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Parameter')
        ax.grid(True, alpha=0.3)
        row += 1

    # Per-parameter gradients on optimizer scale
    if has_grad_history:
        for name in param_names:
            ax = axes[row, 0]
            vals = grad_history.get(name, [])
            ax.plot(vals, linewidth=1.0, alpha=0.8, color='C4')
            ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel(f'grad {name}')
            ax.set_title(f'{name} Gradient (Unconstrained)')
            ax.grid(True, alpha=0.3)
            row += 1

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
