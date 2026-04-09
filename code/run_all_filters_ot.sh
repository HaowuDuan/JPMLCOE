#!/usr/bin/env bash
# Run filtering experiments that use OT resampling.
# Usage: bash run_all_filters_ot.sh [MAX_PARALLEL]
#   MAX_PARALLEL defaults to 4 because OT resampling is relatively expensive.

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-4}
export TF_FORCE_GPU_ALLOW_GROWTH=true

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    1d_linear/1d_linear_ledh_invertible_ot
    5d_linear/5d_linear_ledh_invertible_ot
    5d_linear/5d_linear_pf_ot
    5d_linear_partial_strong/5d_partial_strong_ledh_invertible_ot
    5d_linear_partial_strong/5d_partial_strong_pf_ot
    5d_linear_partial_weak/5d_partial_weak_ledh_invertible_ot
    5d_linear_partial_weak/5d_partial_weak_pf_ot
    cubic_sensor/cubic_sensor_pf_ot
    stochastic_volatility/stochastic_volatility_ledh_invertible_ot_log
    stochastic_volatility/stochastic_volatility_ledh_invertible_ot_log_ukf
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.1
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.3
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.5
    stochastic_volatility/stochastic_volatility_pf_ot_eps1.0
)

TOTAL=${#EXPERIMENTS[@]}
LOGDIR="outputs/all_filters_ot_logs"
mkdir -p "$LOGDIR"

echo "=== Running $TOTAL OT filtering experiments (max $MAX_PARALLEL parallel) ==="
echo "Logs: $LOGDIR/"
echo "Started: $(date)"
echo ""

PIDS=()
NAMES=()

for i in "${!EXPERIMENTS[@]}"; do
    while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done

    exp="${EXPERIMENTS[$i]}"
    name=$(basename "$exp")
    model=$(dirname "$exp")
    n=$((i + 1))
    logfile="$LOGDIR/${model}_${name}.log"

    echo "[$n/$TOTAL] Launching: $exp"
    "$PYTHON_BIN" -m src.experiments.run_experiment experiment="$exp" > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$exp")
done

echo ""
echo "All jobs launched. Waiting for remaining jobs..."

FAILED=0
SUCCEEDED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  OK: ${NAMES[$i]}"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        status=$?
        echo "  FAILED: ${NAMES[$i]} (exit $status)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Done: $SUCCEEDED/$TOTAL succeeded, $FAILED failed ==="
echo "Finished: $(date)"

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
