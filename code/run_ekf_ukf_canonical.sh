#!/bin/bash
# Re-run the 6 canonical EKF/UKF experiments on SV and range-bearing in parallel.
#
# Context: run_experiment.py was patched so that non-tracking EKF/UKF filters
# start from the model's deterministic mu_0 rather than a random draw of the
# prior distribution. This script regenerates the canonical results under the
# new (correct) initialization.
#
# Runs 2 filters (EKF, UKF) x 3 model variants (SV raw, SV log-transform, RB) = 6 jobs.
# All jobs run in parallel. Per-job logs are written to /tmp/ekf_ukf_rerun_<timestamp>/.

cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

LOG_DIR="/tmp/ekf_ukf_rerun_$(date +%s)"
mkdir -p "$LOG_DIR"
START_MARKER="$LOG_DIR/start.marker"
touch "$START_MARKER"
echo "Parallel logs: $LOG_DIR"

EXPERIMENTS=(
  "stochastic_volatility/stochastic_volatility_ekf"
  "stochastic_volatility/stochastic_volatility_ekf_log"
  "stochastic_volatility/stochastic_volatility_ukf"
  "stochastic_volatility/stochastic_volatility_ukf_log"
  "range_bearing/range_bearing_ekf"
  "range_bearing/range_bearing_ukf"
)

declare -a PIDS
# Force CPU for each parallel job. These filters are tiny (<3 MB, <2 s) and
# running 6 concurrent TF-CUDA processes on a single GPU causes init contention
# (OOM / DNN init failed / InternalError). CPU is both correct and faster here
# because it avoids the GPU init cost per process.
for exp in "${EXPERIMENTS[@]}"; do
  name=$(basename "$exp")
  echo "  [launch] $name"
  CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
    "$PYTHON_BIN" -m src.experiments.run_experiment experiment="$exp" \
    > "$LOG_DIR/$name.log" 2>&1 &
  PIDS+=($!)
done

fail=0
for i in "${!PIDS[@]}"; do
  name=$(basename "${EXPERIMENTS[$i]}")
  if wait "${PIDS[$i]}"; then
    echo "  [ok]     $name"
  else
    echo "  [FAIL]   $name  (see $LOG_DIR/$name.log)"
    fail=$((fail + 1))
  fi
done

if [ "$fail" -eq 0 ]; then
  "$PYTHON_BIN" - "$START_MARKER" <<'PY'
import sys
from pathlib import Path

import numpy as np

marker_mtime_ns = Path(sys.argv[1]).stat().st_mtime_ns
checks = [
    ("stochastic_volatility_ekf",
     Path("outputs/stochastic_volatility/stochastic_volatility_ekf/initial_mean.npy"),
     np.array([0.0])),
    ("stochastic_volatility_ekf_log",
     Path("outputs/stochastic_volatility/stochastic_volatility_ekf_log/initial_mean.npy"),
     np.array([0.0])),
    ("stochastic_volatility_ukf",
     Path("outputs/stochastic_volatility/stochastic_volatility_ukf/initial_mean.npy"),
     np.array([0.0])),
    ("stochastic_volatility_ukf_log",
     Path("outputs/stochastic_volatility/stochastic_volatility_ukf_log/initial_mean.npy"),
     np.array([0.0])),
    ("range_bearing_ekf",
     Path("outputs/range_bearing/range_bearing_ekf/initial_mean.npy"),
     np.array([5.0, 5.0])),
    ("range_bearing_ukf",
     Path("outputs/range_bearing/range_bearing_ukf/initial_mean.npy"),
     np.array([5.0, 5.0])),
]

errors = []
for name, path, expected in checks:
    if not path.exists():
        errors.append(f"{name}: missing {path}")
        continue
    if path.stat().st_mtime_ns <= marker_mtime_ns:
        errors.append(f"{name}: {path} was not rewritten by this rerun")
        continue
    actual = np.load(path)
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-10):
        errors.append(f"{name}: initial_mean={actual.tolist()}, expected {expected.tolist()}")

if errors:
    print("Post-run verification failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("Verified all 6 initial_mean.npy files are fresh and deterministic.")
PY
  verify_status=$?
  if [ "$verify_status" -ne 0 ]; then
    exit "$verify_status"
  fi
  echo "All 6 experiments finished successfully."
  echo "Results updated in outputs/stochastic_volatility/ and outputs/range_bearing/."
  exit 0
else
  echo "$fail experiment(s) failed. See logs in $LOG_DIR."
  exit 1
fi
