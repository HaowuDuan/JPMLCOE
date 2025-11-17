"""
Visualization utilities for filter comparison.

This module provides plotting functions to visualize filter performance,
trajectories, errors, and particle distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from scipy.stats import norm


def plot_trajectories(true_states: np.ndarray, filter_results: Dict[str, np.ndarray],
                     title: str = "State Trajectories", save_path: Optional[str] = None):
    """
    Plot true states vs estimated states for multiple filters.
    
    Args:
        true_states: True states, shape (T, state_dim)
        filter_results: Dictionary mapping filter names to estimated means
            {'filter_name': means_array}, where means_array has shape (T, state_dim)
        title: Plot title
        save_path: Optional path to save the figure
    """
    T = len(true_states)
    time_steps = np.arange(T)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true states
    ax.plot(time_steps, true_states[:, 0], 'k-', linewidth=2, label='True State', alpha=0.8)
    
    # Plot estimates for each filter
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    for i, (filter_name, means) in enumerate(filter_results.items()):
        color = colors[i % len(colors)]
        ax.plot(time_steps, means[:, 0], '--', color=color, linewidth=1.5,
               label=f'{filter_name} Estimate', alpha=0.7)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_errors(true_states: np.ndarray, filter_results: Dict[str, np.ndarray],
               filter_covs: Dict[str, np.ndarray],
               title: str = "Estimation Errors", confidence: float = 0.95,
               save_path: Optional[str] = None):
    """
    Plot estimation errors with confidence bounds.
    
    Args:
        true_states: True states, shape (T, state_dim)
        filter_results: Dictionary mapping filter names to estimated means
        filter_covs: Dictionary mapping filter names to estimated covariances
        title: Plot title
        confidence: Confidence level for error bounds (default 0.95)
        save_path: Optional path to save the figure
    """
    T = len(true_states)
    time_steps = np.arange(T)
    
    # Number of filters
    n_filters = len(filter_results)
    
    fig, axes = plt.subplots(n_filters, 1, figsize=(12, 4 * n_filters))
    if n_filters == 1:
        axes = [axes]
    
    # Compute confidence interval multiplier (for 1D Gaussian)
    z_score = norm.ppf((1 + confidence) / 2)
    
    for idx, (filter_name, means) in enumerate(filter_results.items()):
        ax = axes[idx]
        covs = filter_covs[filter_name]
        
        # Compute errors
        errors = true_states[:, 0] - means[:, 0]
        
        # Compute error bounds (±z_score * std)
        std_errors = np.sqrt(covs[:, 0, 0])
        upper_bound = z_score * std_errors
        lower_bound = -z_score * std_errors
        
        # Plot errors
        ax.plot(time_steps, errors, 'b-', linewidth=1.5, label='Error', alpha=0.7)
        ax.fill_between(time_steps, lower_bound, upper_bound,
                        alpha=0.3, color='blue', label=f'{int(confidence*100)}% Confidence')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Error')
        ax.set_title(f'{filter_name} - Estimation Errors')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_particle_evolution(particles: np.ndarray, weights: np.ndarray,
                          true_states: Optional[np.ndarray] = None,
                          selected_steps: Optional[np.ndarray] = None,
                          title: str = "Particle Evolution",
                          save_path: Optional[str] = None):
    """
    Plot particle distributions at selected time steps.
    
    Args:
        particles: Array of particles, shape (T, n_particles, state_dim)
        weights: Array of weights, shape (T, n_particles)
        true_states: Optional true states for reference, shape (T, state_dim)
        selected_steps: Optional array of time steps to plot. If None, plots
            evenly spaced steps covering the full trajectory.
        title: Plot title
        save_path: Optional path to save the figure
    """
    T = particles.shape[0]
    
    if selected_steps is None:
        # Select 6 evenly spaced time steps
        selected_steps = np.linspace(0, T - 1, 6, dtype=int)
    
    n_steps = len(selected_steps)
    n_cols = 3
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, t in enumerate(selected_steps):
        ax = axes[idx]
        
        # Get particles and weights at time t
        pts = particles[t, :, 0]
        wts = weights[t, :]
        
        # Plot weighted histogram
        ax.hist(pts, weights=wts, bins=50, density=True, alpha=0.6,
               color='blue', edgecolor='black', label='Particle Distribution')
        
        # Plot true state if available
        if true_states is not None:
            ax.axvline(true_states[t, 0], color='red', linewidth=2,
                      linestyle='--', label='True State')
        
        ax.set_xlabel('State')
        ax.set_ylabel('Density')
        ax.set_title(f'Time Step {t}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_ess(ess_history: np.ndarray, title: str = "Effective Sample Size",
            save_path: Optional[str] = None):
    """
    Plot Effective Sample Size (ESS) over time.
    
    Args:
        ess_history: Array of ESS values, shape (T,)
        title: Plot title
        save_path: Optional path to save the figure
    """
    T = len(ess_history)
    time_steps = np.arange(T)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_steps, ess_history, 'b-', linewidth=1.5, label='ESS')
    
    # Add threshold line (if resample threshold is known)
    # This would need to be passed as a parameter, but for now we'll skip it
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_performance(performance_metrics: Dict[str, Dict[str, float]],
                    title: str = "Filter Performance Comparison",
                    save_path: Optional[str] = None):
    """
    Plot runtime and memory usage comparison.
    
    Args:
        performance_metrics: Dictionary mapping filter names to metrics:
            {'filter_name': {'runtime': ..., 'memory_mb': ...}}
        title: Plot title
        save_path: Optional path to save the figure
    """
    filter_names = list(performance_metrics.keys())
    runtimes = [performance_metrics[name].get('runtime', 0) for name in filter_names]
    memories = [performance_metrics[name].get('memory_mb', 0) for name in filter_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Runtime plot
    ax1.bar(filter_names, runtimes, color=['blue', 'red', 'green'][:len(filter_names)])
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Memory plot
    ax2.bar(filter_names, memories, color=['blue', 'red', 'green'][:len(filter_names)])
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_metrics_comparison(comparison: Dict[str, Dict[str, float]],
                          metrics: Optional[list] = None,
                          title: str = "Metrics Comparison",
                          save_path: Optional[str] = None):
    """
    Plot comparison of multiple metrics across filters.
    
    Args:
        comparison: Dictionary with metrics for each filter
        metrics: List of metric names to plot. If None, plots common metrics.
        title: Plot title
        save_path: Optional path to save the figure
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'coverage']
    
    filter_names = list(comparison.keys())
    
    # Prepare data
    x = np.arange(len(filter_names))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [comparison[name].get(metric, 0) for name in filter_names]
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric)
    
    ax.set_xlabel('Filter')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(filter_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

