#!/usr/bin/env bash
# Run all filters on 5D Linear Gaussian (partial obs, strong coupling) IN PARALLEL.
# Usage: bash run_5d_partial_strong_filters.sh [MAX_PARALLEL]

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-999}
export TF_FORCE_GPU_ALLOW_GROWTH=true

DIR=5d_linear_partial_strong
EXPERIMENTS=(
    ${DIR}/5d_partial_strong_Kalman
    ${DIR}/5d_partial_strong_ekf
    ${DIR}/5d_partial_strong_ukf
    ${DIR}/5d_partial_strong_pf
    ${DIR}/5d_partial_strong_pf_ot
    ${DIR}/5d_partial_strong_pf_soft
    ${DIR}/5d_partial_strong_edh_flow
    ${DIR}/5d_partial_strong_edh_invertible
    ${DIR}/5d_partial_strong_stochastic_edh
    ${DIR}/5d_partial_strong_ledh_flow
    ${DIR}/5d_partial_strong_ledh_invertible
    ${DIR}/5d_partial_strong_ledh_invertible_ot
    ${DIR}/5d_partial_strong_ledh_invertible_soft
)

TOTAL=${#EXPERIMENTS[@]}
LOGDIR="outputs/${DIR}/logs"
mkdir -p "$LOGDIR"

echo "=== Running $TOTAL 5D partial-strong experiments (max $MAX_PARALLEL parallel) ==="
echo "Logs: $LOGDIR/"
echo ""

RUNNING=0
PIDS=()
NAMES=()

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    name=$(basename "$exp")
    n=$((i + 1))
    logfile="$LOGDIR/${name}.log"

    echo "[$n/$TOTAL] Launching: $exp"
    python -m src.experiments.run_experiment experiment="$exp" > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$exp")
    RUNNING=$((RUNNING + 1))

    if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
        wait -n 2>/dev/null || true
        RUNNING=$((RUNNING - 1))
    fi
done

echo ""
echo "All jobs launched. Waiting for remaining to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "  FAILED: ${NAMES[$i]} (exit $status) — see $LOGDIR/$(basename ${NAMES[$i]}).log"
        FAILED=$((FAILED + 1))
    else
        echo "  OK: ${NAMES[$i]}"
    fi
done

echo ""
echo "=== Done: $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed ==="
