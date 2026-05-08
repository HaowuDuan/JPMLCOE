"""Minimum profiling harness for the LEDH+OT DPF HMC pipeline.

What it does:
  - Reuses the same Hydra config + model/filter building blocks as
    run_dpf_experiment.py.
  - Builds the DPFRunner once. Generates one observation sequence at the
    config's data seed.
  - Loops `n_measure` times computing log-posterior twice on the SAME
    parameter point: once forward only (no gradient tape), once with the
    gradient tape. Records paired (forward_ms, full_ms) per iteration so
    backward = full - forward is well-defined per state.
  - Wraps the loop with tf.profiler so TensorBoard sees a single trace.
  - Logs GPU memory and process RSS each iteration.
  - Saves summary JSON.

Usage (Hydra-style):
  python run_profile_minimal.py dpf=hmc/range_bearing/ledh_ot_axisstep_l10_c1 \
      n_warmup=2 n_measure=10 logdir=outputs/profile/rb_ledh_axisstep

Smoke-test on 1D LG:
  python run_profile_minimal.py dpf=hmc/linear_gaussian/ledh_ot_c1 \
      n_warmup=2 n_measure=5 logdir=outputs/profile/lg_smoke

Reading the output:
  tensorboard --logdir=outputs/profile --bind_all
  cat outputs/profile/*/summary.json
"""

import os
import sys
import json
import time
import importlib
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Ensure code/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import psutil
import tensorflow as tf

from src.models.utils import generate_data
from src.experiments.run_dpf_experiment import _create_true_model, _build_param_specs
from src.DF.hmc_runner import DPFRunner


def _gpu_mem_mb():
    """Current/peak GPU memory in MB, or None if no GPU."""
    try:
        info = tf.config.experimental.get_memory_info('GPU:0')
        return {
            'current_mb': float(info['current']) / 1024**2,
            'peak_mb': float(info['peak']) / 1024**2,
        }
    except Exception:
        return None


def _sync():
    """Force a host-device sync so wall-time reads device completion, not just
    kernel launch. Falls back to a tiny .numpy() round-trip on older TF."""
    try:
        tf.test.experimental.sync_devices()
    except (AttributeError, NotImplementedError):
        _ = tf.constant(0.0).numpy()


@hydra.main(version_base=None, config_path="configs", config_name="config_dpf")
def main(cfg: DictConfig):
    n_warmup = int(cfg.get('n_warmup', 2))
    n_measure = int(cfg.get('n_measure', 10))
    logdir = str(cfg.get('logdir', 'outputs/profile/minimum'))
    Path(logdir).mkdir(parents=True, exist_ok=True)

    dtype_map = {'float32': tf.float32, 'float64': tf.float64}
    dtype_tf = dtype_map[cfg.get('dtype', 'float32')]

    print("=== Minimum profiling harness ===")
    print(f"Filter:  {cfg.filter._target_}")
    print(f"Model:   {cfg.model._target_}")
    print(f"T:       {cfg.data.T}")
    print(f"GPU:     {tf.config.list_physical_devices('GPU')}")
    print(f"Logdir:  {logdir}")
    print(f"Warmup:  {n_warmup}   Measure: {n_measure}")

    # 1. Build true model + observations
    rng = np.random.default_rng(cfg.data.seed)
    true_model = _create_true_model(cfg, dtype_tf)
    initial_state, states, observations = generate_data(true_model, T=cfg.data.T, rng=rng)
    if hasattr(true_model, 'transform_observations'):
        observations = true_model.transform_observations(observations)

    # 2. Build inference model + filter + DPF runner
    inference_model = hydra.utils.instantiate(cfg.model, dtype=dtype_tf)
    param_specs = _build_param_specs(cfg.dpf.trainable_params, cfg.model)

    filter_class_path = cfg.filter._target_
    module_path, class_name = filter_class_path.rsplit('.', 1)
    filter_module = importlib.import_module(module_path)
    filter_class = getattr(filter_module, class_name)

    filter_kwargs = OmegaConf.to_container(cfg.filter, resolve=True)
    filter_kwargs.pop('_target_', None)

    runner = DPFRunner(
        base_model=inference_model,
        filter_class=filter_class,
        filter_kwargs=filter_kwargs,
        param_specs=param_specs,
    )
    runner._observations_tf = tf.constant(observations, dtype=dtype_tf)

    # 3. Initial parameter point in unconstrained space.
    # ParameterHandler builds this from the model's initial values.
    q = tf.cast(runner.param_handler.unconstrained_init, dtype_tf)
    print(f"\nTrainable params: {list(param_specs.keys())}")
    print(f"q (unconstrained): {q.numpy()}")

    # 4. Forward and full callables. The full call mirrors what one leapfrog
    # half-step does inside HMC: a single (value, grad) evaluation.
    def forward_call(q_):
        return runner._negative_log_posterior(q_)

    def full_call(q_):
        with tf.GradientTape() as tape:
            tape.watch(q_)
            nlp = runner._negative_log_posterior(q_)
        grad = tape.gradient(nlp, q_)
        return nlp, grad

    # 5. Warm up each mode separately. First call triggers tracing/XLA compile;
    # later calls use the cached graph. Discard their times.
    print(f"\nWarmup ({n_warmup} iter forward, {n_warmup} iter full)")
    for _ in range(n_warmup):
        _ = forward_call(q)
    for _ in range(n_warmup):
        _ = full_call(q)
    _sync()

    # 6. Reset peak memory so the measured window's peak is isolated.
    try:
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception:
        pass
    proc = psutil.Process()

    # 7. Profiled loop. Paired forward + full per iteration on the same q.
    # Subtraction backward = full - forward is well-defined per iteration.
    use_profiler = bool(cfg.get('profile', True))
    print(f"\nMeasuring {n_measure} paired iterations  (profiler={use_profiler})")
    if use_profiler:
        tf.profiler.experimental.start(logdir)

    fwd_times, full_times = [], []
    mems, rss_mb = [], []

    for i in range(n_measure):
        with tf.profiler.experimental.Trace("forward", step_num=i, _r=1):
            _sync()
            t0 = time.perf_counter()
            _ = forward_call(q)
            _sync()
            t1 = time.perf_counter()

        with tf.profiler.experimental.Trace("full", step_num=i, _r=1):
            t2 = time.perf_counter()
            _ = full_call(q)
            _sync()
            t3 = time.perf_counter()

        fwd_times.append(t1 - t0)
        full_times.append(t3 - t2)
        mems.append(_gpu_mem_mb())
        rss_mb.append(proc.memory_info().rss / 1024**2)

        bwd = (full_times[-1] - fwd_times[-1]) * 1000
        print(f"  iter {i:2d}:  fwd {fwd_times[-1]*1000:7.1f} ms"
              f"  full {full_times[-1]*1000:7.1f} ms"
              f"  bwd {bwd:7.1f} ms"
              f"  rss {rss_mb[-1]:7.0f} MB"
              f"  gpu_cur {mems[-1]['current_mb']:6.0f} MB" if mems[-1] else "")

    if use_profiler:
        tf.profiler.experimental.stop()

    # 8. Summary
    fwd_arr = np.array(fwd_times) * 1000   # ms
    full_arr = np.array(full_times) * 1000
    diff_arr = full_arr - fwd_arr           # paired backward time

    def _stats(a):
        return {
            'median_ms': float(np.median(a)),
            'iqr_ms': float(np.quantile(a, 0.75) - np.quantile(a, 0.25)),
            'min_ms': float(np.min(a)),
            'max_ms': float(np.max(a)),
        }

    summary = {
        'config': {
            'filter': cfg.filter._target_,
            'model': cfg.model._target_,
            'T': int(cfg.data.T),
            'data_seed': int(cfg.data.seed),
            'dtype': cfg.get('dtype', 'float32'),
            'n_warmup': n_warmup,
            'n_measure': n_measure,
            'trainable_params': list(param_specs.keys()),
            'q_unconstrained': q.numpy().tolist(),
        },
        'per_iter_ms': {
            'forward': fwd_arr.tolist(),
            'full': full_arr.tolist(),
            'backward_paired': diff_arr.tolist(),
        },
        'forward_stats': _stats(fwd_arr),
        'full_stats': _stats(full_arr),
        'backward_stats': _stats(diff_arr),
        'backward_fraction_median': float(np.median(diff_arr) / np.median(full_arr))
            if np.median(full_arr) > 0 else None,
        'memory': {
            'gpu_per_iter_mb': mems,
            'rss_per_iter_mb': rss_mb,
            'gpu_growth_mb': float(mems[-1]['peak_mb'] - mems[0]['peak_mb'])
                if mems[0] is not None and mems[-1] is not None else None,
            'rss_growth_mb': float(rss_mb[-1] - rss_mb[0]),
        },
    }

    out_path = Path(logdir) / 'summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results ===")
    print(f"Forward:  median {summary['forward_stats']['median_ms']:.1f} ms"
          f" (IQR {summary['forward_stats']['iqr_ms']:.1f})")
    print(f"Full:     median {summary['full_stats']['median_ms']:.1f} ms"
          f" (IQR {summary['full_stats']['iqr_ms']:.1f})")
    print(f"Backward: median {summary['backward_stats']['median_ms']:.1f} ms"
          f" (IQR {summary['backward_stats']['iqr_ms']:.1f})  [paired]")
    if summary['backward_fraction_median'] is not None:
        print(f"Backward fraction (of full): {summary['backward_fraction_median']:.1%}")
    print(f"\nGPU peak growth across measured window: "
          f"{summary['memory']['gpu_growth_mb']} MB" if summary['memory']['gpu_growth_mb']
          is not None else "")
    print(f"RSS growth across measured window: "
          f"{summary['memory']['rss_growth_mb']:.1f} MB")
    print(f"\nProfiler trace: tensorboard --logdir={logdir}")
    print(f"Summary JSON:   {out_path}")


if __name__ == '__main__':
    main()
