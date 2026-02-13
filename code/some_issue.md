---
name: TF Logs Progress SDE
overview: Suppress TensorFlow/Metal info messages, add optional progress tracking with time per step, and address StochasticEDH on MPS (debugging stale code, RNG differences, force CPU for reproducibility).
todos: []
isProject: false
---

# Plan: Suppress TF Logs, Progress Tracking, and SDE/MPS Reproducibility

## Part 1: Suppress TensorFlow/Metal Info Messages

**Problem:** Messages like `Note the GPU implementation does not produce the same series as CPU implementation` from `metal_plugin/src/kernels/stateless_random_op.cc` clutter output.

**Approach:** Set `TF_CPP_MIN_LOG_LEVEL` before any TensorFlow import. Must happen before `setup_tensorflow_device()` (which imports `tf`).

**Change:** In [code/src/experiments/run_experiment.py](code/src/experiments/run_experiment.py), add at the very top (before other imports):

```python
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 0=all, 1=no DEBUG, 2=no INFO, 3=ERROR only
```

Or make it config-driven: add `tf_log_level: 2` to [code/configs/config.yaml](code/configs/config.yaml) and use `cfg.get("tf_log_level", 0)` so users can override when needed.

---

## Part 2: Progress Tracking with Time per Step

**Problem:** Filter runs as a black box; no live feedback. Only total wall time and average ms/step are printed after completion.

**Approach:** Add optional progress callback to filters with Python loops. Callback receives `(t, T, step_time)` each step.

### 2.1 Filter interface

- Add optional `progress_callback: Optional[Callable[[int, int, float], None]] = None` to `filter()` in filters that use Python loops.
- Callback signature: `(t: int, T: int, step_time_sec: float)`.
- Call after each predict/update, before appending results.

**Files to modify:**


| File                                                                                         | Loop location |
| -------------------------------------------------------------------------------------------- | ------------- |
| [code/src/filters/particle/flow_base.py](code/src/filters/particle/flow_base.py)             | Lines 144–148 |
| [code/src/filters/particle/edh_invertible.py](code/src/filters/particle/edh_invertible.py)   | Lines 362–368 |
| [code/src/filters/particle/ledh_invertible.py](code/src/filters/particle/ledh_invertible.py) | Lines 399–405 |
| [code/src/filters/kalman/extended_kalman.py](code/src/filters/kalman/extended_kalman.py)     | Lines 247–252 |
| [code/src/filters/kalman/unscented_kalman.py](code/src/filters/kalman/unscented_kalman.py)   | Lines 321–326 |
| [code/src/filters/kalman/kalman.py](code/src/filters/kalman/kalman.py)                       | Lines 250–255 |
| [code/src/filters/particle/kernel_flow.py](code/src/filters/particle/kernel_flow.py)         | Lines 455+    |


**Note:** `bootstrap_pf_tf.py` uses a `tf.range` loop inside a `@tf.function`; adding per-step callbacks would require a larger refactor. Leave it without progress for now.

### 2.2 run_experiment integration

- Add `show_progress: false` to [code/configs/config.yaml](code/configs/config.yaml).
- When `cfg.get("show_progress", False)`, build a callback that prints e.g. `Step t/T (X.X ms)` or uses `tqdm`.
- Pass callback to `filter_obj.filter(..., progress_callback=cb)` when the filter supports it (check `callable(getattr(filter_obj.filter, '__wrapped__', None))` or use `hasattr` / `inspect` to detect the param; simpler: try/except or explicit filter list).

**Suggested callback (no tqdm):**

```python
def _make_progress_callback(T):
    step_times = []
    def cb(t, total, step_time):
        step_times.append(step_time)
        elapsed = sum(step_times)
        avg_ms = (step_time * 1000) if step_times else 0
        print(f"\r  Step {t+1}/{total} | {step_time*1000:.1f} ms | elapsed {elapsed:.2f}s", end="", flush=True)
    return cb
```

---

## Part 3: StochasticEDH and MPS — "Code on MPS Not Updated?"

**Problem:** "edh flow initialization is correct, but stochastic edh is not." Possibilities: (a) cached/stale code on MPS, (b) MPS-specific behavior (RNG, etc.).

### 3.1 Debugging: Is Stale Code Running on MPS?

If StochasticEDH behaves differently despite sharing `initialize()` with ExactDaumHuangFlow, check:


| Cause                     | Symptom                                | Fix                                                                                                          |
| ------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Python bytecode cache** | Old logic after edits                  | `find code -type d -name __pycache__ -exec rm -rf {} +` or `python -B -m src.experiments.run_experiment ...` |
| **TF graph caching**      | Cached `tf.function` traces per device | Call `tf.keras.backend.clear_session()` before filter run; restart Python process                            |
| **Module import order**   | Different module version loaded        | Restart Python; avoid `runpy` / interactive reload; run as fresh process                                     |
| **MPS vs CPU**            | Different results on MPS               | Run with `device=cpu` to confirm behavior matches CPU                                                        |


**Verification:** Add a temporary `print("StochasticEDH.initialize", id(self.model))` in `edh_flow.initialize()` (or in `StochasticEDHFlow` if it ever overrides it) and run both EDH flow and StochasticEDH. If the print appears for both, the same code path is used; if not, an override or import issue exists.

### 3.2 Root Cause: MPS RNG Differs from CPU

StochasticEDH inherits `initialize()` from ExactDaumHuangFlow — same logic. The difference is in **update** using `tf.random.stateless_normal` for SDE diffusion. Metal logs: *"GPU implementation does not produce the same series as CPU implementation"* — so on MPS, diffusion noise differs from CPU, leading to different outputs.

**Implications:**

- Same initialization, different first-step output on MPS vs CPU due to RNG.
- Results are non-reproducible across MPS vs CPU for StochasticEDH.

### 3.3 Mitigation Options

1. **Force CPU for StochasticEDH** (recommended for reproducibility): In `run_filter_experiment`, when filter is `StochasticEDHFlow` and `device == "auto"`, call `setup_tensorflow_device(device="cpu")` or `force_cpu()` before creating the filter.
2. **Config override**: Add `stochastic_edh_use_cpu: true` to config; when True and filter is StochasticEDH, force CPU regardless of `device`.
3. **Document only**: Add a note in config or README that StochasticEDH on MPS may differ from CPU due to RNG. No code change.

**Recommended:** Option 1 — automatically use CPU for StochasticEDH when `device=auto`.

---

## Summary of Changes

| Item              | File(s)                                                                                            | Change                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| TF log level      | run_experiment.py, config.yaml                                                                     | Set `TF_CPP_MIN_LOG_LEVEL` before TF import; optional config                            |
| Progress callback | flow_base, edh_invertible, ledh_invertible, kalman, extended_kalman, unscented_kalman, kernel_flow | Add optional `progress_callback` to `filter()`, call each step with `(t, T, step_time)` |
| Progress wiring   | run_experiment.py, config.yaml                                                                     | Add `show_progress`, build callback when True, pass to filter                           |
| SDE on MPS        | run_experiment.py (or device.py)                                                                   | When filter is StochasticEDH and device=auto, force CPU                                 |

**Manual debugging (no code change):** If StochasticEDH behaves wrong but EDH flow is fine, run: `find code -type d -name __pycache__ -exec rm -rf {} +`, then `python -B -m src.experiments.run_experiment device=cpu experiment=...` to rule out bytecode cache and MPS-specific behavior.


