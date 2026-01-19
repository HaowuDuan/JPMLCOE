"""Hydra-based experiment runner for filter evaluation."""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
import json
import time
import tracemalloc
import psutil
from typing import Dict, Any

from ..models.utils import generate_data


def _is_valid_initial_state(state: np.ndarray, model) -> bool:
    """
    Check if all target positions are in [0,40]×[0,40].
    
    Args:
        state: Initial state vector
        model: State-space model
        
    Returns:
        True if all positions are within bounds
    """
    n_targets = model.state_dim // 4
    for i in range(n_targets):
        x = state[i * 4]
        y = state[i * 4 + 1]
        if not (0 <= x <= 40 and 0 <= y <= 40):
            return False
    return True


def run_filter_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run a filtering experiment with Hydra configuration.

    Args:
        cfg: Hydra configuration containing model, filter, and experiment settings

    Returns:
        Dictionary with results and metadata
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(cfg.seed)

    # Instantiate model
    print(f"Creating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    # Generate data
    T = cfg.get('T', 100)
    
    # Predict-then-update filters need observe_initial=False
    # These filters: propagate particles first, then update with observation
    filter_target = cfg.filter._target_.split('.')[-1]
    predict_then_update_filters = {
        'KernelMappingPF',
        'ExactDaumHuangFlow', 'LocalExactDaumHuangFlow',  # Flow filters do predict-then-update
    }
    observe_initial = filter_target not in predict_then_update_filters
    
    print(f"Generating {T} observations... (observe_initial={observe_initial})")
    states, observations = generate_data(model, T=T, rng=rng, observe_initial=observe_initial)

    # Instantiate filter
    print(f"Creating filter: {cfg.filter._target_}")
    filter_name = cfg.filter._target_.split('.')[-1]
    
    # For Kalman filters (EKF/UKF), sample perturbed initial state as per paper
    # Paper: "mean of initial distributions... sampled from Gaussian centered at true initial states"
    # with std = 10 for positions, std = 1 for velocities
    # Only do this for tracking models (state_dim divisible by 4), otherwise let filter handle initialization
    if filter_name in ['ExtendedKalmanFilter', 'UnscentedKalmanFilter'] and model.state_dim >= 4 and model.state_dim % 4 == 0:
        true_initial_state = states[0]

        # Sample perturbed initial mean (paper specification)
        initial_noise_std = np.array([10.0, 10.0, 1.0, 1.0])
        if model.state_dim == 16:  # Multi-target (4 targets)
            initial_noise_std = np.tile(initial_noise_std, 4)
        elif model.state_dim > 4:  # General multi-target
            n_targets = model.state_dim // 4
            initial_noise_std = np.tile(initial_noise_std, n_targets)

        # Sample with rejection (must be in [0,40]×[0,40])
        perturbed_mean = true_initial_state + rng.normal(0, initial_noise_std)
        max_attempts = 100
        attempts = 0
        while not _is_valid_initial_state(perturbed_mean, model) and attempts < max_attempts:
            perturbed_mean = true_initial_state + rng.normal(0, initial_noise_std)
            attempts += 1

        if attempts >= max_attempts:
            print(f"Warning: Could not find valid initial state after {max_attempts} attempts. Using true state.")
            perturbed_mean = true_initial_state

        # Build initial covariance
        initial_cov_std = np.array([10.0, 10.0, 1.0, 1.0])
        if model.state_dim == 16:
            initial_cov_std = np.tile(initial_cov_std, 4)
        elif model.state_dim > 4:
            n_targets = model.state_dim // 4
            initial_cov_std = np.tile(initial_cov_std, n_targets)
        initial_cov = np.diag(initial_cov_std ** 2)

        print(f"Initialized {filter_name} with perturbed state (noise std: position=10, velocity=1)")
        filter_obj = hydra.utils.instantiate(
            cfg.filter,
            model=model,
            mean_0=perturbed_mean,
            Sigma_0=initial_cov,
            sample_initial_mean=False,  # Don't sample since we're providing explicit initial state
            random_seed=cfg.seed
        )
    # For EKF/UKF with non-tracking models, let them handle their own initialization
    elif filter_name in ['ExtendedKalmanFilter', 'UnscentedKalmanFilter']:
        print(f"Initialized {filter_name} with sampled initial state from model distribution")
        filter_obj = hydra.utils.instantiate(
            cfg.filter,
            model=model,
            random_seed=cfg.seed
        )
    # Pass model only if filter needs it (EKF, UKF, PF need it; KalmanFilter doesn't)
    elif 'Kalman' in cfg.filter._target_ and 'Extended' not in cfg.filter._target_ and 'Unscented' not in cfg.filter._target_:
        filter_obj = hydra.utils.instantiate(cfg.filter)
    else:
        filter_obj = hydra.utils.instantiate(cfg.filter, model=model)

    # Run filter
    print(f"Running {filter_name}...")

    # Start performance tracking
    tracemalloc.start()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024**2  # MB
    start_time = time.perf_counter()

    # All filters now return FilterResult for consistency
    # EDH/LEDH non-invertible flow filters inherit from FlowFilterBase and accept random_state
    # EDH/LEDH invertible filters don't accept random_state (use internal RNG)
    # Particle filters and Kalman filters don't accept random_state
    if filter_name in ['ExactDaumHuangFlow', 'LocalExactDaumHuangFlow']:
        # Non-invertible flow filters from FlowFilterBase
        result = filter_obj.filter(observations, random_state=rng)
    else:
        # All other filters (invertible flows, particle filters, Kalman filters)
        result = filter_obj.filter(observations)

    means = result.means
    covs = result.covs
    log_likelihood = result.log_likelihood
    metadata = result.metadata

    # Extract diagnostics if available (works for all filter types)
    pf_diagnostics = {}
    if hasattr(result, 'log_likelihoods') and result.log_likelihoods is not None:
        pf_diagnostics['log_likelihoods'] = result.log_likelihoods
    if hasattr(result, 'ess') and result.ess is not None:
        pf_diagnostics['ess'] = result.ess
    if hasattr(result, 'weights_history') and result.weights_history is not None:
        pf_diagnostics['weights_history'] = result.weights_history
    if hasattr(result, 'resampled_at') and result.resampled_at is not None:
        pf_diagnostics['resampled_at'] = result.resampled_at
    if hasattr(result, 'n_unique') and result.n_unique is not None:
        pf_diagnostics['n_unique'] = result.n_unique

    # End performance tracking
    end_time = time.perf_counter()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_memory = process.memory_info().rss / 1024**2  # MB

    # Compute performance metrics
    wall_time = end_time - start_time
    time_per_step = wall_time / T
    memory_increase = end_memory - start_memory
    peak_memory_mb = peak_memory / 1024**2

    # Add performance metrics to metadata
    metadata['performance'] = {
        'wall_time_seconds': float(wall_time),
        'time_per_timestep_ms': float(time_per_step * 1000),
        'peak_memory_mb': float(peak_memory_mb),
        'memory_increase_mb': float(memory_increase),
        'start_memory_mb': float(start_memory),
        'end_memory_mb': float(end_memory),
        'num_timesteps': int(T)
    }

    # Compute RMSE
    rmse = np.sqrt(np.mean((means - states)**2))

    # Prepare results
    results = {
        'means': means,
        'covs': covs,
        'states': states,
        'observations': observations,
        'log_likelihood': log_likelihood,
        'rmse': float(rmse),
        'metadata': metadata,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    # Add particle filter diagnostics if available (from local scope)
    if 'pf_diagnostics' in locals():
        results.update(pf_diagnostics)

    # Print summary
    print("\n" + "="*60)
    print(f"RESULTS: {filter_name}")
    print("="*60)
    print(f"RMSE: {rmse:.4f}")
    if log_likelihood is not None:
        print(f"Log-likelihood: {log_likelihood:.2f}")
    # Print mean ESS if available (from diagnostics)
    if 'ess' in pf_diagnostics and pf_diagnostics['ess'] is not None:
        mean_ess = float(np.mean(pf_diagnostics['ess']))
        print(f"Mean ESS: {mean_ess:.1f}")
    if 'mean_ess' in metadata:
        print(f"Mean ESS: {metadata['mean_ess']:.1f}")
    print(f"\nPerformance:")
    print(f"  Wall time: {metadata['performance']['wall_time_seconds']:.3f} seconds")
    print(f"  Time per step: {metadata['performance']['time_per_timestep_ms']:.2f} ms")
    print(f"  Peak memory: {metadata['performance']['peak_memory_mb']:.1f} MB")
    print(f"  Memory increase: {metadata['performance']['memory_increase_mb']:.1f} MB")
    print("="*60)

    return results


def save_results(results: Dict[str, Any], output_dir: Path):
    """
    Save experiment results to disk.

    Args:
        results: Results dictionary from run_filter_experiment
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main arrays
    np.save(output_dir / 'means.npy', results['means'])
    np.save(output_dir / 'covs.npy', results['covs'])
    np.save(output_dir / 'states.npy', results['states'])
    np.save(output_dir / 'observations.npy', results['observations'])

    # Save particle filter diagnostics as separate .npy files (from result object, not metadata)
    if 'ess' in results and results['ess'] is not None:
        np.save(output_dir / 'ess.npy', results['ess'])
    if 'weights_history' in results and results['weights_history'] is not None:
        np.save(output_dir / 'weights_history.npy', results['weights_history'])
    if 'log_likelihoods' in results and results['log_likelihoods'] is not None:
        np.save(output_dir / 'log_likelihoods.npy', results['log_likelihoods'])
    if 'n_unique' in results and results['n_unique'] is not None:
        np.save(output_dir / 'n_unique.npy', results['n_unique'])
    if 'resampled_at' in results and results['resampled_at'] is not None:
        # Save resampled_at as numpy array for consistency
        np.save(output_dir / 'resampled_at.npy', np.array(results['resampled_at']))

    # Save kernel flow particle arrays as .npy files (not in JSON - too large)
    metadata_for_json = results['metadata'].copy()
    if 'prior_particles' in metadata_for_json:
        np.save(output_dir / 'prior_particles.npy', metadata_for_json.pop('prior_particles'))
    if 'posterior_particles' in metadata_for_json:
        np.save(output_dir / 'posterior_particles.npy', metadata_for_json.pop('posterior_particles'))

    # Save scalars and metadata
    summary = {
        'rmse': results['rmse'],
        'log_likelihood': results['log_likelihood'],
        'metadata': metadata_for_json,
        'config': results['config']
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for Hydra experiments."""
    results = run_filter_experiment(cfg)

    # Get Hydra's output directory (it changes cwd to this directory)
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)

    save_results(results, output_dir)

    # Optional: create plots
    if cfg.get('plot', True):
        state_dim = results['states'].shape[1]
        
        # Use high-dim visualization for large state spaces (> 10 dimensions)
        if state_dim > 10:
            from .visualization import plot_high_dim_results
            # Get obs_spacing from model config if available
            obs_spacing = cfg.model.get('obs_spacing', 4)
            
            # For multi-target acoustic tracking (16-d), plot position components
            # Plot x,y positions for all 4 targets (indices: 0,1, 4,5, 8,9, 12,13)
            if state_dim == 16:  # Multi-target acoustic tracking
                plot_indices = [0, 1, 4, 5, 8, 9, 12, 13]  # x,y for each of 4 targets
            else:
                # Default for other high-dim systems (e.g., Lorenz96)
                plot_indices = None  # Will use default [18, 19]
            
            plot_high_dim_results(
                results['states'],
                results['observations'],
                results['means'],
                results['covs'],
                save_path=output_dir / 'plot.png',
                obs_spacing=obs_spacing,
                plot_indices=plot_indices
            )
        else:
            from .visualization import plot_filter_results
            plot_filter_results(
                results['states'],
                results['observations'],
                results['means'],
                results['covs'],
                save_path=output_dir / 'plot.png'
            )


if __name__ == '__main__':
    main()
