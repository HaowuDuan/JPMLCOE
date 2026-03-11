#!/usr/bin/env bash
# Run all filters on acoustic tracking model IN PARALLEL.
# Usage: bash run_acoustic_tracking_filters.sh [MAX_PARALLEL]

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-4}
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    acoustic_tracking/acoustic_tracking_edh_flow
    acoustic_tracking/acoustic_tracking_edh_invertible
    acoustic_tracking/acoustic_tracking_ekf
    acoustic_tracking/acoustic_tracking_ledh_flow
    acoustic_tracking/acoustic_tracking_ledh_invertible
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n25
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n29
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n33
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n37
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n42
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n25
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n29
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n33
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n37
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n42
    acoustic_tracking/acoustic_tracking_pf
    acoustic_tracking/acoustic_tracking_pf_1e6
    acoustic_tracking/acoustic_tracking_pf_tf
    acoustic_tracking/acoustic_tracking_stochastic_edh
    acoustic_tracking/acoustic_tracking_ukf
)

TOTAL=${#EXPERIMENTS[@]}
LOGDIR="outputs/$(dirname ${EXPERIMENTS[0]})/logs"
mkdir -p "$LOGDIR"

echo "=== Running $TOTAL experiments (max $MAX_PARALLEL parallel) ==="
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
