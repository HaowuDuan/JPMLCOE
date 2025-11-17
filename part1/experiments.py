"""
Helper functions for numerical experiments with filters.

This module provides:
1. Benchmarking utilities (runtime, memory)
2. Filter comparison metrics (RMSE, MAE, coverage)
3. Filter runner functions
"""

import numpy as np
import time
import tracemalloc
from typing import Dict, Tuple, Optional, Callable, Any
from scipy.stats import norm


def benchmark_filter(filter_func: Callable, *args, **kwargs) -> Dict[str, float]:
    """
    Benchmark a filter function's runtime and memory usage.
    
    Args:
        filter_func: Filter function to benchmark
        *args, **kwargs: Arguments to pass to filter_func
        
    Returns:
        Dictionary with 'runtime' (seconds) and 'memory_mb' (MB)
    """
    # Start memory tracking
    tracemalloc.start()
    
    # Time the filter
    start_time = time.time()
    result = filter_func(*args, **kwargs)
    end_time = time.time()
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'runtime': end_time - start_time,
        'memory_mb': peak / (1024 * 1024)  # Convert to MB
    }


def compute_rmse(true_states: np.ndarray, estimated_means: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE).
    
    Args:
        true_states: True states, shape (T, state_dim)
        estimated_means: Estimated means, shape (T, state_dim)
        
    Returns:
        RMSE value
    """
    errors = true_states - estimated_means
    return np.sqrt(np.mean(errors ** 2))


def compute_mae(true_states: np.ndarray, estimated_means: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        true_states: True states, shape (T, state_dim)
        estimated_means: Estimated means, shape (T, state_dim)
        
    Returns:
        MAE value
    """
    errors = true_states - estimated_means
    return np.mean(np.abs(errors))


def compute_coverage(true_states: np.ndarray, estimated_means: np.ndarray,
                    estimated_covs: np.ndarray, confidence: float = 0.95) -> float:
    """
    Compute coverage: fraction of time true state is within confidence interval.
    
    Args:
        true_states: True states, shape (T, state_dim)
        estimated_means: Estimated means, shape (T, state_dim)
        estimated_covs: Estimated covariances, shape (T, state_dim, state_dim)
        confidence: Confidence level (default 0.95)
        
    Returns:
        Coverage fraction (between 0 and 1)
    """
    T = len(true_states)
    z_score = norm.ppf((1 + confidence) / 2)
    
    in_interval = 0
    for t in range(T):
        # For 1D case, check if true state is within ±z_score * std
        std = np.sqrt(estimated_covs[t, 0, 0])
        lower = estimated_means[t, 0] - z_score * std
        upper = estimated_means[t, 0] + z_score * std
        
        if lower <= true_states[t, 0] <= upper:
            in_interval += 1
    
    return in_interval / T


def compare_filters(filter_results: Dict[str, Dict[str, np.ndarray]],
                   performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple filters using various metrics.
    
    Args:
        filter_results: Dictionary mapping filter names to results:
            {'filter_name': {'means': ..., 'covs': ..., 'true_states': ...}}
        performance_metrics: Dictionary mapping filter names to performance:
            {'filter_name': {'runtime': ..., 'memory_mb': ...}}
        
    Returns:
        Dictionary mapping filter names to comparison metrics:
            {'filter_name': {'rmse': ..., 'mae': ..., 'coverage': ...}}
    """
    comparison = {}
    
    for filter_name, results in filter_results.items():
        true_states = results['true_states']
        means = results['means']
        covs = results['covs']
        
        comparison[filter_name] = {
            'rmse': compute_rmse(true_states, means),
            'mae': compute_mae(true_states, means),
            'coverage': compute_coverage(true_states, means, covs),
            'runtime': performance_metrics[filter_name].get('runtime', 0),
            'memory_mb': performance_metrics[filter_name].get('memory_mb', 0)
        }
    
    return comparison


def print_comparison_summary(comparison: Dict[str, Dict[str, float]]):
    """
    Print a formatted summary of filter comparison.
    
    Args:
        comparison: Dictionary from compare_filters()
    """
    print("\n" + "="*80)
    print("FILTER COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Filter':<10} {'RMSE':<12} {'MAE':<12} {'Coverage':<12} {'Runtime (s)':<15} {'Memory (MB)':<15}")
    print("-"*80)
    
    # Data rows
    for filter_name, metrics in comparison.items():
        print(f"{filter_name:<10} "
              f"{metrics['rmse']:<12.6f} "
              f"{metrics['mae']:<12.6f} "
              f"{metrics['coverage']:<12.4f} "
              f"{metrics['runtime']:<15.4f} "
              f"{metrics['memory_mb']:<15.2f}")
    
    print("="*80)


def run_ekf(model, observations: np.ndarray, true_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Extended Kalman Filter.
    
    Args:
        model: State-space model instance
        observations: Observations, shape (T, obs_dim)
        true_states: True states (for storing in results)
        
    Returns:
        Tuple of (means, covs, log_likelihoods)
    """
    from filters import ExtendedKalmanFilter
    
    ekf = ExtendedKalmanFilter(model)
    means, covs = ekf.filter(observations)
    log_likelihoods = np.array(ekf.log_likelihoods)
    return means, covs, log_likelihoods


def run_ukf(model, observations: np.ndarray, true_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Unscented Kalman Filter.
    
    Args:
        model: State-space model instance
        observations: Observations, shape (T, obs_dim)
        true_states: True states (for storing in results)
        
    Returns:
        Tuple of (means, covs, log_likelihoods)
    """
    from filters import UnscentedKalmanFilter
    
    ukf = UnscentedKalmanFilter(model)
    means, covs = ukf.filter(observations)
    log_likelihoods = np.array(ukf.log_likelihoods)
    return means, covs, log_likelihoods


def run_pf(model, observations: np.ndarray, true_states: np.ndarray,
          n_particles: int = 1000,
          random_state: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Particle Filter.
    
    Args:
        model: State-space model instance
        observations: Observations, shape (T, obs_dim)
        true_states: True states (for storing in results)
        n_particles: Number of particles
        random_state: Optional numpy random generator
        
    Returns:
        Tuple of (means, covs, log_likelihoods, ess_history)
    """
    from filters import ParticleFilter
    
    pf = ParticleFilter(model, n_particles=n_particles)
    means, covs = pf.filter(observations, random_state=random_state)
    log_likelihoods = np.array(pf.log_likelihoods)
    ess_history = np.array(pf.ess_history)
    return means, covs, log_likelihoods, ess_history

