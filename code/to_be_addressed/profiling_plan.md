# Profiling plan: LEDH+OT HMC (revised)

Goal: produce defensible numbers for where one HMC leapfrog step spends its time
on the LEDH+OT pipeline, plus confirm the marginal-likelihood memory leak is
gone. The only speedup claim already on the table is **XLA 1.3× forward at
T=20**. Don't extend that claim until measured.

This rewrite addresses the issues Codex flagged on the first draft: section
non-disjointness, XLA fusion across `Trace` scopes, `tracemalloc` blindness to
TF/CUDA bytes, CUPTI discoverability, and statistical defensibility.

## Definitions (don't drift on these)

- **Unit of measurement**: wall-clock for **one leapfrog step** with the seed
  fixed and HMC step-size adaptation **off**. One leapfrog step = one
  log-posterior evaluation + one gradient evaluation. We do *not* report
  per-HMC-iteration or per-proposal numbers; those mix accept/reject branches.
- **Largest model**: range-bearing LEDH+OT at $N=500$ particles, $T=50$ steps,
  $L=5$ leapfrog steps. Fixed PF seed `[42, 0]`.
- **Forward time**: log-posterior eval, no gradient.
- **Backward time**: total leapfrog step minus forward time. This is the only
  defensible way to attribute gradient cost — `Trace` scopes alone do not
  separate forward/backward inside an XLA-compiled graph.

## Preflight (must pass before any timing run)

1. **CUPTI discoverable.** With pip CUDA + venv, `libcupti.so` must be
   importable by TF profiler. Check:
   ```
   python -c "import tensorflow as tf; tf.profiler.experimental.start('/tmp/prof_test'); tf.profiler.experimental.stop(); import os; print(os.listdir('/tmp/prof_test'))"
   ```
   Expect a `plugins/profile/<timestamp>/` directory with `*.trace.json.gz`
   inside. If the trace file is empty or missing GPU streams, fix
   `LD_LIBRARY_PATH` to include the venv's `nvidia/cuda_cupti/lib` path before
   doing anything else.

2. **Profiler plugin installed.** `pip install tensorboard_plugin_profile`.
   Without it, TensorBoard shows the data but no Profile tab.

3. **Graph stability.** Three consecutive calls of the production HMC step
   (after one warm-up) must trace identically. Check by enabling retracing
   warnings:
   ```python
   tf.config.run_functions_eagerly(False)
   tf.get_logger().setLevel("INFO")
   # call hmc_step three times; no "WARNING:tensorflow: ... has been retraced"
   ```
   If retracing fires, fix it before profiling — retraces invalidate timing.

4. **No competing GPU work.** `nvidia-smi` should show only this run on the
   3090. Kill leftover Python processes from other shells.

## Setup (~30–45 min, one-time)

### Add a single outer trace per leapfrog step

Inside the leapfrog body (in the HMC loop, wherever the gradient tape is
opened), wrap each step:

```python
with tf.profiler.experimental.Trace("leapfrog", step_num=i, _r=1):
    # the existing leapfrog half-kick / full-drift / half-kick
```

This gives TensorBoard step-aware analysis (per-step traces, step-time
distribution). Do **not** add nested `Trace` scopes for sub-sections — those
spans are host-side annotations and do not respect XLA fusion. Use them only
as visual bookmarks if you want, never as a source of GPU-time numbers.

### Forward-only vs full-step ablation

The defensible breakdown comes from running two configurations:

- **Forward-only**: log-posterior eval only, no gradient tape.
- **Full step**: log-posterior + gradient w.r.t. parameters.

Backward cost = full minus forward. This is the cleanest attribution we have
under XLA. Add a `forward_only` flag to the entry point that toggles the
gradient tape.

### Harness script

`code/run_profile_ledh_ot.py`. One run per `(model, mode, repeat_index)` tuple.
Each run is a fresh Python process (call this script repeatedly from a shell
loop) — XLA cache contamination across configurations is real.

```python
import argparse, json, os, time
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--mode", choices=["forward", "full"], required=True)
parser.add_argument("--n-warmup", type=int, default=2)
parser.add_argument("--n-measure", type=int, default=10)
parser.add_argument("--logdir", required=True)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

# Build the leapfrog step. Disable adaptation, fix seed.
step = build_leapfrog_step(
    config_path=args.config,
    forward_only=(args.mode == "forward"),
    adapt=False,
    seed=42,
)

state = step.initial_state()

# Warm-up: trigger compile, throw away timing.
for _ in range(args.n_warmup):
    state = step(state)

tf.test.experimental.sync_devices()

# Reset memory stats so peak reflects only the measured window.
tf.config.experimental.reset_memory_stats('GPU:0')

# Per-iteration timing with explicit device sync.
times = []
mem = []
if args.profile:
    tf.profiler.experimental.start(args.logdir)
for i in range(args.n_measure):
    t0 = time.perf_counter()
    with tf.profiler.experimental.Trace("leapfrog", step_num=i, _r=1):
        state = step(state)
    tf.test.experimental.sync_devices()
    times.append(time.perf_counter() - t0)
    mem.append(tf.config.experimental.get_memory_info('GPU:0'))
if args.profile:
    tf.profiler.experimental.stop()

# Persist for offline analysis.
out = dict(
    times_seconds=times,
    gpu_mem_per_iter=mem,
    config=args.config,
    mode=args.mode,
)
with open(os.path.join(args.logdir, "summary.json"), "w") as f:
    json.dump(out, f, indent=2)
```

The `sync_devices()` calls are essential: without them, host-side timing
measures kernel-launch latency, not device completion.

## Pass 1: timing breakdown on the largest model (~30 min)

Goal: report forward-step time, backward-step time (= full − forward), and
backward fraction, with median + IQR over 5 fresh-process repeats per mode.

```bash
LOGROOT=outputs/profile/ledh_ot_rb
mkdir -p $LOGROOT
for mode in forward full; do
  for rep in 1 2 3 4 5; do
    python -u run_profile_ledh_ot.py \
      --config configs/dpf/range_bearing/ledh_ot.yaml \
      --mode $mode \
      --logdir $LOGROOT/${mode}_rep${rep} \
      $( [ "$rep" = "1" ] && echo --profile )
  done
done
```

Only repeat 1 enables `--profile` (the profiler adds modest overhead and we
only need one trace for kernel-level analysis). Repeats 2–5 measure raw
wall-time without profiler overhead.

**Reported numbers** (per leapfrog step):
- forward median + IQR (5 reps × 10 measured iters = 50 samples)
- full median + IQR
- backward = full − forward, propagated IQR

## Pass 2: dimension scan on the LEDH+OT code path (~45 min)

Goal: distinguish op-launch overhead from algorithmic scaling.

Run the **same LEDH+OT pipeline** at synthetic linear-Gaussian data with
$d_x \in \{1, 2, 5, 10\}$. Do *not* swap to a different filter or model —
keep the exact LEDH+OT path so conclusions transfer. Vary only $d_x$.

```bash
for d in 1 2 5 10; do
  for rep in 1 2 3; do
    python -u run_profile_ledh_ot.py \
      --config configs/dpf/synth_lg_dx${d}/ledh_ot.yaml \
      --mode full \
      --logdir outputs/profile/dimscan/d${d}_rep${rep} \
      $( [ "$rep" = "1" ] && echo --profile )
  done
done
```

(Need to write the four `synth_lg_dx*.yaml` configs. Same $T$, same $N$, only
$d_x$ changes.)

**Diagnostic**: plot median wall-time vs $d_x$ on log-log axes.
- Flat between $d_x=1$ and $d_x=2$, then a knee around 5: launch-bound regime
  at low $d$.
- Smooth power law throughout: never launch-bound.

Cross-check from the profile trace at $d_x=1$ and $d_x=10$: TensorBoard
GPU Kernel Stats. Look at:
- **Kernel duration distribution** — if 90 %+ of kernels are < 50 µs, the run
  is launch-bound.
- **Host launch gaps** — if there are ~10 µs idle gaps between kernels, host
  is the bottleneck.
- **Occupancy** — for the dominant compute kernel (MatMul, TriangularSolve),
  look at SM occupancy. Low occupancy = under-utilised GPU regardless of which
  side is "bound".

A single wall-time-vs-$d_x$ curve is not enough on its own.

## Pass 3: memory steady-state (~20 min, separate from Pass 1/2)

The leak we're checking is the 13–18 GB-per-call spike that came from particle
initialisation inside the marginal-likelihood routine. That's TF/CUDA bytes,
not Python objects, so:

**Primary signals**:
1. `tf.config.experimental.get_memory_info('GPU:0')` `current` and `peak`,
   logged every iteration (already in the harness above).
2. `nvidia-smi --query-gpu=memory.used --format=csv` polled every 2 s to a log
   file.
3. Process RSS via `ps -o pid,rss,vsz -p $PID` polled every 2 s.

**Secondary signal** (only for confirming whether a residual leak is on the
Python side):
4. `tracemalloc` snapshot diff. Useful if (1)–(3) plateau and you still see
   slow Python heap growth.

Run the harness for 50 leapfrog steps (no profiler, no extra overhead) and
record all four signals per iteration. Steady-state criterion:

- TF `peak` plateaus within 3–5 iterations and does not climb after that.
- `nvidia-smi` `memory.used` plateaus likewise.
- RSS plateau within 5–10 iterations.

If any of (1)–(3) climb monotonically for 50 iterations, the leak is back.
Use the iteration-by-iteration log to localise: which iteration first
exceeded the plateau? Was it a resampling step, an OT solve, a gradient eval?

## Smoke test (5 min, run before any expensive run)

Same harness on **1D linear Gaussian** with $N=200$, 3 warm-up + 5 measured
leapfrog steps, both `--mode forward` and `--mode full`. Verifies:

- The script runs to completion without error.
- The profile log dir contains a `*.trace.json.gz` with both CPU and GPU
  streams (open in TensorBoard, look for the GPU device track).
- Steps 2–5 in `times_seconds` are within 5 % of each other (no retracing on
  iteration 2).
- `summary.json` parses.

If any of these fail, fix and re-smoke. Don't move to RB until smoke is clean.

## Analysis (TensorBoard, ~30–45 min)

```
tensorboard --logdir=outputs/profile --bind_all
```

Profile tab → Trace Viewer:
- **Step time** view: median + distribution per step. The outer
  `Trace("leapfrog", step_num=i)` registers each iteration.
- **Streams**: confirm GPU streams are populated. If only CPU stream is shown,
  CUPTI is not loading — go back to Preflight.

Profile tab → GPU Kernel Stats:
- Sort by total self-time. Top 5 should be the algorithm-dominant kernels
  (MatMul, TriangularSolve / Cholesky, Cast / MemcpyHtoD).
- For each, note self-time, SM occupancy, kernel duration distribution.

Profile tab → Op Profile (TF op level):
- Aggregate by op type. Cross-check with kernel stats — TF ops that are
  fused into a single XLA cluster show up as `XlaRun` with the cluster
  identity. Don't try to attribute cluster time back to the original TF ops.

## Deliverables

A markdown writeup `code/to_be_addressed/profiling_results.md` with:

1. **Preflight verdict** — one line: "CUPTI loaded, GPU streams populated,
   3 retrace-free runs confirmed."
2. **Pass 1 table**:
   | Model | Forward median (ms) | Forward IQR | Full median (ms) | Full IQR | Backward = Full − Forward (ms) | Backward % |
3. **Pass 2 dim-scan plot** — log-log wall-time vs $d_x$, with annotation of
   the knee (or absence). Two-sentence interpretation.
4. **Pass 3 memory verdict** — "Steady state at peak X.Y GB, no growth across
   50 iterations" OR "Leak still present, RSS climbed from X to Y GB across N
   iterations, first jump at iteration K".
5. **XLA win restated**: "1.3× forward at T=20" is the only number defensible
   right now. Pass 1 should *not* claim a backward-side speedup unless the
   forward and full timings on the same hardware show it.

## Honest cost

| Step | Wall-clock | Human attention |
|---|---|---|
| Preflight | 10 min | 10 min |
| Setup (Trace + harness + ablation flag) | 30–45 min | 30–45 min |
| Smoke test | 5 min run | 10 min reading |
| Pass 1 (RB, 2 modes × 5 reps) | 30 min run | 5 min babysit |
| Pass 2 (4 dims × 3 reps) | 45 min run | 5 min babysit |
| Pass 3 (memory) | 20 min run | 10 min reading |
| Analysis + writeup | 60 min | 60 min |
| **Total** | **~3.5 hours wall-clock** | **~3 hours focused** |

Most of the wall-clock is unattended (multiple repeats running in series).
The focused time is mostly in setup and analysis.

## Traps to avoid

1. **Don't trust nested `Trace` scopes for GPU time under `jit_compile=True`.**
   XLA fuses across them. Use forward-vs-full ablation instead.
2. **Don't combine `tracemalloc` with the timing pass.** 2–5× slowdown
   contaminates wall-time.
3. **Don't reuse a Python process across configurations.** XLA cache hangs on,
   the second config benefits from a warm cache and looks faster than a real
   run would be.
4. **Don't read host-side `Trace` durations as device time.** Always
   `sync_devices()` before reading wall-clock; for deeper attribution, read
   from GPU streams in the trace viewer, not from Python timestamps.
5. **Don't extrapolate the XLA win.** The 1.3× number is at $T=20$. At
   $T=50$ or $T=100$, op-launch overhead amortises differently. Re-measure if
   you want to claim a different $T$.
6. **One warm-up call is not enough to prove no retracing.** Confirm graph
   stability via the retrace-warning log in Preflight, not by inspection.
