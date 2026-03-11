"""Hydra-based experiment runner for filter evaluation."""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # 0=all, 1=no DEBUG, 2=no INFO, 3=ERROR only

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
import json
import time
import tracemalloc
import psutil
from typing import Dict, Any, Tuple, Optional, Callable

from ..models.utils import generate_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_initial_state(state: np.ndarray, model) -> bool:
    """Check if all target positions are in [0,40]x[0,40]."""
    n_targets = model.state_dim // 4
    for i in range(n_targets):
        x = state[i * 4]
        y = state[i * 4 + 1]
        if not (0 <= x <= 40 and 0 <= y <= 40):
            return False
    return True


def _create_filter(cfg: DictConfig, model, initial_state: np.ndarray,
                   rng: np.random.Generator) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Instantiate a filter from Hydra config.

    Returns:
        (filter_obj, initial_mean, initial_cov) — the initial_mean and
        initial_cov are the values actually used to initialize the filter,
        needed for visualization.
    """
    filter_name = cfg.filter._target_.split('.')[-1]

    # --- EKF / UKF with tracking models: perturbed initial state ---
    if (filter_name in ('ExtendedKalmanFilter', 'UnscentedKalmanFilter')
            and model.state_dim >= 4 and model.state_dim % 4 == 0):

        # Paper: perturb true initial with std=10 (pos), std=1 (vel)
        noise_std = np.tile([10.0, 10.0, 1.0, 1.0], model.state_dim // 4)

        perturbed_mean = initial_state + rng.normal(0, noise_std)
        for _ in range(100):
            if _is_valid_initial_state(perturbed_mean, model):
                break
            perturbed_mean = initial_state + rng.normal(0, noise_std)
        else:
            print("Warning: Could not find valid perturbed initial state. Using true state.")
            perturbed_mean = initial_state.copy()

        initial_cov = np.diag(noise_std ** 2)

        filter_obj = hydra.utils.instantiate(
            cfg.filter, model=model,
            mean_0=perturbed_mean, Sigma_0=initial_cov,
            sample_initial_mean=False, random_seed=cfg.seed,
        )
        print(f"Initialized {filter_name} with perturbed state (noise std: position=10, velocity=1)")
        return filter_obj, perturbed_mean, initial_cov

    # --- EKF / UKF with non-tracking models ---
    if filter_name in ('ExtendedKalmanFilter', 'UnscentedKalmanFilter'):
        filter_obj = hydra.utils.instantiate(
            cfg.filter, model=model, random_seed=cfg.seed,
        )
        # Retrieve what the filter actually initialized with
        initial_mean = np.asarray(filter_obj.mean_0)
        initial_cov = np.asarray(filter_obj.Sigma_0)
        print(f"Initialized {filter_name} with sampled initial state from model distribution")
        return filter_obj, initial_mean, initial_cov

    # --- Plain Kalman (no model needed) ---
    if ('Kalman' in cfg.filter._target_
            and 'Extended' not in cfg.filter._target_
            and 'Unscented' not in cfg.filter._target_):
        filter_obj = hydra.utils.instantiate(cfg.filter)
        initial_mean = filter_obj.mean_0
        initial_cov = filter_obj.Sigma_0
        return filter_obj, initial_mean, initial_cov

    # --- Everything else (particle filters, flow filters) ---
    filter_obj = hydra.utils.instantiate(cfg.filter, model=model)
    initial_mean = np.asarray(model.mu_0)
    initial_cov = np.asarray(model.Sigma_0)
    return filter_obj, initial_mean, initial_cov


def _make_progress_callback(T: int) -> Callable[[int, int, float], None]:
    """Build a progress callback that prints step-by-step timing."""
    step_times = []
    def cb(t: int, total: int, step_time: float) -> None:
        step_times.append(step_time)
        elapsed = sum(step_times)
        print(f"\r  Step {t+1}/{total} | {step_time*1000:.1f} ms | elapsed {elapsed:.2f}s", end="", flush=True)
        if t + 1 == total:
            print()  # newline after last step
    return cb


def _run_filter(filter_obj, filter_name: str, observations: np.ndarray,
                rng: np.random.Generator,
                progress_callback: Optional[Callable] = None):
    """Dispatch filter.filter() with the correct signature."""
    import inspect
    sig = inspect.signature(filter_obj.filter)
    has_cb = 'progress_callback' in sig.parameters

    if filter_name in ('ExactDaumHuangFlow', 'LocalExactDaumHuangFlow'):
        if has_cb:
            return filter_obj.filter(observations, random_state=rng,
                                     progress_callback=progress_callback)
        return filter_obj.filter(observations, random_state=rng)

    if has_cb:
        return filter_obj.filter(observations, progress_callback=progress_callback)
    return filter_obj.filter(observations)


def _extract_diagnostics(result) -> Dict[str, Any]:
    """Extract non-None diagnostic fields from FilterResult."""
    fields = ('log_likelihoods', 'ess', 'weights_history', 'resampled_at', 'n_unique')
    return {f: getattr(result, f) for f in fields if getattr(result, f, None) is not None}


def _get_gpu_memory_mb():
    """Get current GPU memory usage in MB. Returns None if no GPU available."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info = tf.config.experimental.get_memory_info('GPU:0')
            return {
                'current_mb': float(info['current'] / 1024**2),
                'peak_mb': float(info['peak'] / 1024**2),
            }
    except Exception:
        pass
    return None


class _PerfTracker:
    """Context manager for wall-time and memory tracking."""

    def __enter__(self):
        tracemalloc.start()
        self._process = psutil.Process()
        self._start_mem = self._process.memory_info().rss / 1024**2
        self._start_time = time.perf_counter()
        self._gpu_start = _get_gpu_memory_mb()
        return self

    def __exit__(self, *exc):
        self._end_time = time.perf_counter()
        _, self._peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._end_mem = self._process.memory_info().rss / 1024**2
        self._gpu_end = _get_gpu_memory_mb()

    def as_dict(self, T: int) -> Dict[str, float]:
        wall = self._end_time - self._start_time
        d = {
            'wall_time_seconds': float(wall),
            'time_per_timestep_ms': float(wall / T * 1000),
            'peak_memory_mb': float(self._peak_mem / 1024**2),
            'memory_increase_mb': float(self._end_mem - self._start_mem),
            'start_memory_mb': float(self._start_mem),
            'end_memory_mb': float(self._end_mem),
            'num_timesteps': int(T),
        }
        if self._gpu_end is not None:
            d['gpu_peak_memory_mb'] = self._gpu_end['peak_mb']
            d['gpu_current_memory_mb'] = self._gpu_end['current_mb']
        return d


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_filter_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run a filtering experiment with Hydra configuration.

    Returns:
        Dictionary with results, diagnostics, and metadata.
    """
    # Allow config to override TF log level (must happen before TF import)
    tf_log_level = str(cfg.get('tf_log_level', 2))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level

    rng = np.random.default_rng(cfg.seed)

    # Resolve dtype from config
    import tensorflow as tf
    _dtype_map = {'float32': tf.float32, 'float64': tf.float64}
    dtype_tf = _dtype_map[cfg.get('dtype', 'float64')]

    # Instantiate model
    print(f"Creating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model, dtype=dtype_tf)

    # Generate data (standard convention: x_0 separate, states=[x_1..x_T])
    T = cfg.get('T', 100)
    print(f"Generating {T} time steps...")
    initial_state, states, observations = generate_data(model, T=T, rng=rng)

    # Create filter
    filter_name = cfg.filter._target_.split('.')[-1]
    print(f"Creating filter: {filter_name}")
    filter_obj, initial_mean, initial_cov = _create_filter(cfg, model, initial_state, rng)

    # Build progress callback if enabled
    progress_cb = None
    if cfg.get('show_progress', False):
        progress_cb = _make_progress_callback(T)

    # Run filter with performance tracking
    print(f"Running {filter_name}...")
    with _PerfTracker() as perf:
        result = _run_filter(filter_obj, filter_name, observations, rng,
                             progress_callback=progress_cb)

    metadata = result.metadata
    metadata['performance'] = perf.as_dict(T)

    # Compute RMSE (both means and states cover [x_1, ..., x_T])
    rmse = np.sqrt(np.mean((result.means - states) ** 2))

    # Assemble results
    diagnostics = _extract_diagnostics(result)

    results = {
        'initial_state': initial_state,
        'initial_mean': initial_mean,
        'initial_cov': initial_cov,
        'means': result.means,
        'covs': result.covs,
        'states': states,
        'observations': observations,
        'log_likelihood': result.log_likelihood,
        'rmse': float(rmse),
        'metadata': metadata,
        'config': OmegaConf.to_container(cfg, resolve=True),
        **diagnostics,
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {filter_name}")
    print("=" * 60)
    print(f"RMSE: {rmse:.4f}")
    if result.log_likelihood is not None:
        print(f"Log-likelihood: {result.log_likelihood:.2f}")
    if 'ess' in diagnostics:
        print(f"Mean ESS: {float(np.mean(diagnostics['ess'])):.1f}")
    if 'mean_ess' in metadata:
        print(f"Mean ESS: {metadata['mean_ess']:.1f}")
    perf_d = metadata['performance']
    print(f"\nPerformance:")
    print(f"  Wall time: {perf_d['wall_time_seconds']:.3f} seconds")
    print(f"  Time per step: {perf_d['time_per_timestep_ms']:.2f} ms")
    print(f"  Peak memory: {perf_d['peak_memory_mb']:.1f} MB")
    print(f"  Memory increase: {perf_d['memory_increase_mb']:.1f} MB")
    if 'gpu_peak_memory_mb' in perf_d:
        print(f"  Peak memory (GPU): {perf_d['gpu_peak_memory_mb']:.1f} MB")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Save / Plot
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Any], output_dir: Path):
    """Save experiment results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main arrays
    np.save(output_dir / 'initial_state.npy', results['initial_state'])
    np.save(output_dir / 'initial_mean.npy', results['initial_mean'])
    np.save(output_dir / 'initial_cov.npy', results['initial_cov'])
    np.save(output_dir / 'means.npy', results['means'])
    np.save(output_dir / 'covs.npy', results['covs'])
    np.save(output_dir / 'states.npy', results['states'])
    np.save(output_dir / 'observations.npy', results['observations'])

    # Particle filter diagnostics
    diagnostic_fields = ('ess', 'weights_history', 'log_likelihoods', 'n_unique')
    for field in diagnostic_fields:
        if field in results and results[field] is not None:
            np.save(output_dir / f'{field}.npy', results[field])
    if 'resampled_at' in results and results['resampled_at'] is not None:
        np.save(output_dir / 'resampled_at.npy', np.array(results['resampled_at']))

    # Kernel flow particle arrays (too large for JSON)
    metadata_for_json = results['metadata'].copy()
    for key in ('prior_particles', 'posterior_particles'):
        if key in metadata_for_json:
            np.save(output_dir / f'{key}.npy', metadata_for_json.pop(key))

    # Summary JSON
    summary = {
        'rmse': results['rmse'],
        'log_likelihood': results['log_likelihood'],
        'metadata': metadata_for_json,
        'config': results['config'],
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for Hydra experiments."""
    results = run_filter_experiment(cfg)

    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    save_results(results, output_dir)

    # Create plots
    if cfg.get('plot', True):
        state_dim = results['states'].shape[1]

        if state_dim > 10:
            from .visualization import plot_high_dim_results
            obs_spacing = cfg.model.get('obs_spacing', 4)

            if state_dim == 16:
                plot_indices = [0, 1, 4, 5, 8, 9, 12, 13]
            else:
                plot_indices = None

            plot_high_dim_results(
                results['states'], results['observations'],
                results['means'], results['covs'],
                initial_state=results['initial_state'],
                initial_mean=results['initial_mean'],
                initial_cov=results['initial_cov'],
                save_path=output_dir / 'plot.png',
                obs_spacing=obs_spacing,
                plot_indices=plot_indices,
            )
        else:
            from .visualization import plot_filter_results
            plot_filter_results(
                results['states'], results['observations'],
                results['means'], results['covs'],
                initial_state=results['initial_state'],
                initial_mean=results['initial_mean'],
                initial_cov=results['initial_cov'],
                save_path=output_dir / 'plot.png',
            )


if __name__ == '__main__':
    main()
