#!/usr/bin/env bash
# Run all filters on range uearing model IN PARALLEL.
# Usage: bash run_range_bearing_filters.sh [MAX_PARALLEL]

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-999}
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    range_bearing/range_bearing_edh_flow
    range_bearing/range_bearing_edh_flow_global
    range_bearing/range_bearing_edh_flow_global_100steps
    range_bearing/range_bearing_edh_invertible
    range_bearing/range_bearing_ekf
    range_bearing/range_bearing_kernel_matrix
    range_bearing/range_bearing_kernel_scalar
    range_bearing/range_bearing_ledh_flow
    range_bearing/range_bearing_ledh_invertible
    range_bearing/range_bearing_pf
    range_bearing/range_bearing_sde_local_correction
    range_bearing/range_bearing_sde_local_correction_100steps
    range_bearing/range_bearing_sde_local_correction_optimal
    range_bearing/range_bearing_stochastic_edh
    range_bearing/range_bearing_stochastic_edh_100steps
    range_bearing/range_bearing_stochastic_edh_optimal
    range_bearing/range_bearing_stochastic_edhtest
    range_bearing/range_bearing_ukf
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
