"""Visualization utilities for filter results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any


def plot_filter_results(
    states: np.ndarray,
    observations: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    initial_state: np.ndarray,
    initial_mean: np.ndarray,
    initial_cov: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Filter Results"
):
    """
    Plot filter estimation results.

    Prepends initial conditions so the plot shows t=0 through t=T.

    Args:
        states: True states [x_1,...,x_T], shape (T, state_dim)
        observations: Observations [y_1,...,y_T], shape (T, obs_dim)
        means: Filtered means [x̂_1,...,x̂_T], shape (T, state_dim)
        covs: Filtered covariances [P_1,...,P_T], shape (T, state_dim, state_dim)
        initial_state: True initial state x_0, shape (state_dim,)
        initial_mean: Filter initial mean x̂_0, shape (state_dim,)
        initial_cov: Filter initial covariance P_0, shape (state_dim, state_dim)
        save_path: Optional path to save figure
        title: Plot title
    """
    T, state_dim = states.shape
    obs_dim = observations.shape[1]

    # Prepend initial conditions: [x_0, x_1, ..., x_T]
    full_states = np.vstack([initial_state[np.newaxis, :], states])
    full_means = np.vstack([initial_mean[np.newaxis, :], means])
    full_covs = np.concatenate([initial_cov[np.newaxis, :, :], covs])

    # Time axis: t=0 through t=T for states/means, t=1 through t=T for obs
    time_full = np.arange(T + 1)
    time_obs = np.arange(1, T + 1)

    # Compute 95% confidence intervals
    stds = np.sqrt(np.array([np.diag(full_covs[t]) for t in range(T + 1)]))
    lower = full_means - 1.96 * stds
    upper = full_means + 1.96 * stds

    # Create subplots
    n_plots = state_dim + min(obs_dim, 2)  # Show at most 2 observation dimensions
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    if n_plots == 1:
        axes = [axes]

    # Plot states
    for i in range(state_dim):
        ax = axes[i]
        ax.plot(time_full, full_states[:, i], 'k-', label='True state', alpha=0.7, linewidth=2)
        ax.plot(time_full, full_means[:, i], 'b-', label='Filtered mean', alpha=0.7)
        ax.fill_between(time_full, lower[:, i], upper[:, i], alpha=0.3, label='95% CI')

        ax.set_xlabel('Time')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot observations (t=1 through t=T, no observation at t=0)
    for i in range(min(obs_dim, 2)):
        ax = axes[state_dim + i]
        ax.plot(time_obs, observations[:, i], 'k.', label='Observations', alpha=0.5, markersize=3)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Observation {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_comparison(
    states: np.ndarray,
    observations: np.ndarray,
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None,
    title: str = "Filter Comparison"
):
    """
    Compare multiple filter results on the same plot.

    Args:
        states: True states, shape (T, state_dim)
        observations: Observations, shape (T, obs_dim)
        results_dict: Dictionary mapping filter names to their results
                     Each result should have 'means' and optionally 'covs'
        save_path: Optional path to save figure
        title: Plot title
    """
    T, state_dim = states.shape
    time = np.arange(T)

    # Create subplots
    fig, axes = plt.subplots(state_dim, 1, figsize=(12, 3*state_dim))
    if state_dim == 1:
        axes = [axes]

    colors = ['b', 'r', 'g', 'm', 'c', 'y']

    for dim in range(state_dim):
        ax = axes[dim]

        # Plot true state
        ax.plot(time, states[:, dim], 'k-', label='True state',
                alpha=0.7, linewidth=2, zorder=100)

        # Plot each filter's estimates
        for idx, (filter_name, result) in enumerate(results_dict.items()):
            means = result['means']
            color = colors[idx % len(colors)]

            ax.plot(time, means[:, dim], color=color, label=filter_name,
                   alpha=0.7, linewidth=1.5)

            # Add confidence interval if available
            if 'covs' in result:
                covs = result['covs']
                stds = np.sqrt(np.array([covs[t, dim, dim] for t in range(T)]))
                lower = means[:, dim] - 1.96 * stds
                upper = means[:, dim] + 1.96 * stds
                ax.fill_between(time, lower, upper, alpha=0.15, color=color)

        ax.set_xlabel('Time')
        ax.set_ylabel(f'State {dim+1}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def plot_ess_history(
    ess_histories: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Effective Sample Size"
):
    """
    Plot ESS history for particle filters.

    Args:
        ess_histories: Dictionary mapping filter names to ESS histories
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['b', 'r', 'g', 'm', 'c', 'y']

    for idx, (filter_name, ess_hist) in enumerate(ess_histories.items()):
        color = colors[idx % len(colors)]
        time = np.arange(len(ess_hist))
        ax.plot(time, ess_hist, color=color, label=filter_name, alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('ESS')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ESS plot saved to {save_path}")
    else:
        plt.show()


def plot_log_likelihood_comparison(
    log_likelihoods: Dict[str, float],
    save_path: Optional[Path] = None,
    title: str = "Log-Likelihood Comparison"
):
    """
    Bar plot comparing log-likelihoods across filters.

    Args:
        log_likelihoods: Dictionary mapping filter names to log-likelihood values
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    filters = list(log_likelihoods.keys())
    values = list(log_likelihoods.values())

    colors = plt.cm.viridis(np.linspace(0, 1, len(filters)))
    bars = ax.bar(filters, values, color=colors, alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Log-Likelihood')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Log-likelihood comparison saved to {save_path}")
    else:
        plt.show()


def plot_high_dim_results(
    states: np.ndarray,
    observations: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    initial_state: np.ndarray,
    initial_mean: np.ndarray,
    initial_cov: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Filter Results (High-Dim)",
    plot_indices: Optional[List[int]] = None,
    obs_spacing: int = 4
):
    """
    Plot filter results for high-dimensional systems (e.g., Lorenz 96).

    Prepends initial conditions so the plot shows t=0 through t=T.
    Shows only selected state components to avoid creating huge plots.

    Args:
        states: True states [x_1,...,x_T], shape (T, state_dim)
        observations: Observations [y_1,...,y_T], shape (T, obs_dim)
        means: Filtered means [x̂_1,...,x̂_T], shape (T, state_dim)
        covs: Filtered covariances [P_1,...,P_T], shape (T, state_dim, state_dim)
        initial_state: True initial state x_0, shape (state_dim,)
        initial_mean: Filter initial mean x̂_0, shape (state_dim,)
        initial_cov: Filter initial covariance P_0, shape (state_dim, state_dim)
        save_path: Optional path to save figure
        title: Plot title
        plot_indices: List of state indices to plot (default: [18, 19] for x^19, x^20)
        obs_spacing: Observation spacing to determine observed/unobserved
    """
    T, state_dim = states.shape

    # Prepend initial conditions: [x_0, x_1, ..., x_T]
    full_states = np.vstack([initial_state[np.newaxis, :], states])
    full_means = np.vstack([initial_mean[np.newaxis, :], means])
    full_covs = np.concatenate([initial_cov[np.newaxis, :, :], covs])

    # Time axis: t=0 through t=T
    time_full = np.arange(T + 1)

    # Default: show x^(19) (unobserved) and x^(20) (observed) as in paper
    if plot_indices is None:
        plot_indices = [18, 19]  # 0-indexed: x^(19) and x^(20)

    # Compute 95% confidence intervals for selected indices
    stds = np.sqrt(np.array([np.diag(full_covs[t]) for t in range(T + 1)]))

    n_plots = len(plot_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax_idx, state_idx in enumerate(plot_indices):
        ax = axes[ax_idx]

        # Better labels for multi-target tracking (state_dim % 4 == 0)
        if state_dim % 4 == 0 and state_dim >= 16:
            target_idx = state_idx // 4
            component_idx = state_idx % 4
            component_names = ['x', 'y', 'vx', 'vy']
            ylabel = f'Target {target_idx + 1}: {component_names[component_idx]}'
        else:
            is_observed = (state_idx + 1) % obs_spacing == 0
            obs_label = "observed" if is_observed else "unobserved"
            ylabel = f'x^({state_idx + 1}) [{obs_label}]'

        # Plot true state
        ax.plot(time_full, full_states[:, state_idx], 'k-', label='True state',
                alpha=0.7, linewidth=2)

        # Plot filtered mean
        ax.plot(time_full, full_means[:, state_idx], 'b-', label='Filtered mean', alpha=0.7)

        # Plot 95% CI
        lower = full_means[:, state_idx] - 1.96 * stds[:, state_idx]
        upper = full_means[:, state_idx] + 1.96 * stds[:, state_idx]
        ax.fill_between(time_full, lower, upper, alpha=0.3, label='95% CI')

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
