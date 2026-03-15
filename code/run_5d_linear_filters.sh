#!/usr/bin/env bash
# Run all filters on 5D Linear Gaussian model IN PARALLEL.
# Classical (Kalman, EKF, UKF), Bootstrap PF variants, EDH variants, LEDH variants.
# Usage: bash run_5d_linear_filters.sh [MAX_PARALLEL]
#   MAX_PARALLEL: max concurrent jobs (default: 4)

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-999}

# Required for multiple TF processes to share the GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    # Classical baselines
    5d_linear/5d_linear_Kalman
    5d_linear/5d_linear_ekf
    5d_linear/5d_linear_ukf
    # Bootstrap PF variants
    5d_linear/5d_linear_pf
    5d_linear/5d_linear_pf_ot
    5d_linear/5d_linear_pf_soft
    # EDH variants
    5d_linear/5d_linear_edh_flow
    5d_linear/5d_linear_edh_invertible
    5d_linear/5d_linear_stochastic_edh
    # LEDH variants
    5d_linear/5d_linear_ledh_flow
    5d_linear/5d_linear_ledh_invertible
    5d_linear/5d_linear_ledh_invertible_ot
    5d_linear/5d_linear_ledh_invertible_soft
)

TOTAL=${#EXPERIMENTS[@]}
LOGDIR="outputs/5d_linear/logs"
mkdir -p "$LOGDIR"

echo "=== Running $TOTAL 5D Linear Gaussian experiments (max $MAX_PARALLEL parallel) ==="
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

    # Throttle: wait for a slot if at max
    if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
        # Wait for any one job to finish
        wait -n 2>/dev/null || true
        RUNNING=$((RUNNING - 1))
    fi
done

echo ""
echo "All jobs launched. Waiting for remaining to finish..."

# Wait for all remaining jobs and collect results
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
