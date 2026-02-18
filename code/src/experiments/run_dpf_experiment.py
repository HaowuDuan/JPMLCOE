"""Hydra-based DPF experiment runner for HMC parameter inference."""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Add project root (code/) to sys.path so absolute imports work when running as a script
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))))

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
import json
import time
import tracemalloc
import psutil
from typing import Dict, Any

from src.models.utils import generate_data


class _PerfTracker:
    """Context manager for wall-time and memory tracking."""

    def __enter__(self):
        tracemalloc.start()
        self._process = psutil.Process()
        self._start_mem = self._process.memory_info().rss / 1024**2
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._end_time = time.perf_counter()
        _, self._peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._end_mem = self._process.memory_info().rss / 1024**2

    def as_dict(self) -> Dict[str, float]:
        wall = self._end_time - self._start_time
        return {
            'wall_time_seconds': float(wall),
            'peak_memory_mb': float(self._peak_mem / 1024**2),
            'memory_increase_mb': float(self._end_mem - self._start_mem),
        }


def _build_param_specs(cfg_trainable: DictConfig, cfg_model: DictConfig) -> dict:
    """Build ParameterSpec objects from Hydra config.

    init_value is taken from the model config (the inference model's initial
    parameter), so it only needs to be specified once.  An explicit
    init_value in the trainable_params section overrides this.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    from src.DF.types import ParameterSpec

    param_specs = {}
    for name, spec_cfg in cfg_trainable.items():
        prior = hydra.utils.instantiate(spec_cfg.prior)
        # Use explicit init_value if provided, otherwise pull from model config
        if 'init_value' in spec_cfg:
            init_val = float(spec_cfg.init_value)
        else:
            init_val = float(cfg_model[name])
        param_specs[name] = ParameterSpec(
            name=name,
            init_value=init_val,
            constraint=spec_cfg.get('constraint', None),
            prior=prior,
        )
    return param_specs


def _create_true_model(cfg: DictConfig, dtype_tf):
    """Create true model with true parameters for data generation."""
    import tensorflow as tf

    # Build model kwargs from config, overriding with true params
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop('_target_', None)

    true_params = OmegaConf.to_container(cfg.data.true_params, resolve=True)
    model_cfg.update(true_params)

    return hydra.utils.instantiate(cfg.model, dtype=dtype_tf, **true_params)


def _create_filter(cfg: DictConfig, model):
    """Instantiate filter from config."""
    filter_name = cfg.filter._target_.split('.')[-1]

    # EKF/UKF: pass model + sample_initial_mean
    if filter_name in ('ExtendedKalmanFilter', 'UnscentedKalmanFilter'):
        return hydra.utils.instantiate(
            cfg.filter, model=model,
            sample_initial_mean=cfg.filter.get('sample_initial_mean', False),
            random_seed=cfg.get('seed', 0),
        )

    # Particle filters: pass model
    return hydra.utils.instantiate(cfg.filter, model=model)


def run_dpf_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """Run DPF (HMC parameter inference) experiment."""
    import tensorflow as tf

    tf_log_level = str(cfg.get('tf_log_level', 2))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level

    _dtype_map = {'float32': tf.float32, 'float64': tf.float64}
    dtype_tf = _dtype_map[cfg.get('dtype', 'float32')]

    rng = np.random.default_rng(cfg.data.seed)

    # 1. Create true model and generate data
    print(f"Creating true model: {cfg.model._target_}")
    true_model = _create_true_model(cfg, dtype_tf)

    T = cfg.data.T
    print(f"Generating {T} time steps with true params: {OmegaConf.to_container(cfg.data.true_params)}")
    initial_state, states, observations = generate_data(true_model, T=T, rng=rng)

    # 2. Create inference model (with wrong initial guesses)
    print(f"Creating inference model")
    inference_model = hydra.utils.instantiate(cfg.model, dtype=dtype_tf)

    # 3. Build parameter specs
    param_specs = _build_param_specs(cfg.dpf.trainable_params, cfg.model)
    print(f"Trainable params: {list(param_specs.keys())}")

    # 4. Import filter class
    filter_class_path = cfg.filter._target_
    module_path, class_name = filter_class_path.rsplit('.', 1)
    import importlib
    filter_module = importlib.import_module(module_path)
    filter_class = getattr(filter_module, class_name)

    # Build filter kwargs (everything except _target_)
    filter_kwargs = OmegaConf.to_container(cfg.filter, resolve=True)
    filter_kwargs.pop('_target_', None)

    sampler = cfg.dpf.get('sampler', 'hmc')
    print(f"\nRunning {sampler.upper()} inference...")

    if sampler == 'pmmh':
        # PMMH: no DifferentiableModel, no gradients
        from src.DF import PMMHRunner

        runner = PMMHRunner(
            base_model=inference_model,
            filter_class=filter_class,
            filter_kwargs=filter_kwargs,
            param_specs=param_specs,
        )

        pmmh_cfg = cfg.dpf.get('pmmh', cfg.dpf.get('hmc', {}))
        proposal_std = None
        if 'proposal_std' in pmmh_cfg:
            proposal_std = OmegaConf.to_container(pmmh_cfg.proposal_std, resolve=True)

        with _PerfTracker() as perf:
            result = runner.run_inference(
                observations=observations,
                num_samples=int(pmmh_cfg.get('num_samples', 1000)),
                num_burnin=int(pmmh_cfg.get('num_burnin', 500)),
                proposal_std=proposal_std,
                seed=int(pmmh_cfg.get('seed', 42)),
            )
    else:
        # HMC / NUTS / custom_hmc: needs DifferentiableModel + gradients
        from src.DF import DPFRunner

        runner = DPFRunner(
            base_model=inference_model,
            filter_class=filter_class,
            filter_kwargs=filter_kwargs,
            param_specs=param_specs,
            sampler=sampler,
        )

        hmc_cfg = cfg.dpf.hmc

        with _PerfTracker() as perf:
            result = runner.run_inference(
                observations=observations,
                num_samples=hmc_cfg.num_samples,
                num_burnin=hmc_cfg.num_burnin,
                step_size=hmc_cfg.step_size,
                num_leapfrog_steps=hmc_cfg.get('num_leapfrog_steps', 10),
                adaptation_rate=hmc_cfg.get('adaptation_rate', 0.8),
                target_accept_prob=hmc_cfg.get('target_accept_prob', 0.75),
                seed=hmc_cfg.get('seed', 42),
                max_tree_depth=hmc_cfg.get('max_tree_depth', 10),
                grad_clip_norm=float(hmc_cfg.get('grad_clip_norm', 100.0)),
            )

    # 6. Print results
    true_params = OmegaConf.to_container(cfg.data.true_params, resolve=True)

    print("\n" + "=" * 60)
    print("DPF INFERENCE RESULTS")
    print("=" * 60)

    for name, stats in result.summary.items():
        true_val = true_params.get(name, '?')
        in_ci = stats['q5'] <= float(true_val) <= stats['q95'] if true_val != '?' else None
        print(f"\n  {name}:")
        print(f"    True:           {true_val}")
        print(f"    Posterior mean:  {stats['mean']:.4f} +/- {stats['std']:.4f}")
        print(f"    90% CI:         [{stats['q5']:.4f}, {stats['q95']:.4f}]")
        if in_ci is not None:
            print(f"    True in 90% CI: {'YES' if in_ci else 'NO'}")

    print(f"\nDiagnostics:")
    print(f"  Acceptance rate: {result.diagnostics['acceptance_rate']:.1%}")
    print(f"  Final step size: {result.diagnostics['final_step_size']:.6f}")
    print(f"  ESS: {result.diagnostics['ess']}")
    print(f"  R-hat: {result.diagnostics['rhat']}")

    perf_d = perf.as_dict()
    print(f"\nPerformance:")
    print(f"  Wall time: {perf_d['wall_time_seconds']:.1f} seconds")
    print(f"  Peak memory: {perf_d['peak_memory_mb']:.1f} MB")
    print("=" * 60)

    return {
        'result': result,
        'observations': observations,
        'states': states,
        'initial_state': initial_state,
        'true_params': true_params,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'performance': perf_d,
    }


def save_dpf_results(results: Dict[str, Any], output_dir: Path):
    """Save DPF results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = results['result']

    # Save posterior samples
    for name, samples in result.samples.items():
        np.save(output_dir / f'samples_{name}.npy', samples)

    # Save data
    np.save(output_dir / 'observations.npy', results['observations'])
    np.save(output_dir / 'states.npy', results['states'])
    np.save(output_dir / 'initial_state.npy', results['initial_state'])

    # Summary JSON
    summary = {
        'summary': result.summary,
        'diagnostics': result.diagnostics,
        'metadata': result.metadata,
        'true_params': results['true_params'],
        'config': results['config'],
        'performance': results['performance'],
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config_dpf")
def main(cfg: DictConfig):
    """Main entry point for DPF experiments."""
    results = run_dpf_experiment(cfg)

    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    save_dpf_results(results, output_dir)


if __name__ == '__main__':
    main()
