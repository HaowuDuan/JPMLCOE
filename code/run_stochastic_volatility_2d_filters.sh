#!/usr/bin/env bash
# Run all filters on 2D Stochastic Volatility model.
# Tests EKF vs UKF as global filter for flow filters.
# Usage: bash run_stochastic_volatility_2d_filters.sh [MAX_PARALLEL]

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-999}
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    # Kalman filters (standalone)
    stochastic_volatility_2d/stochastic_volatility_2d_ekf
    stochastic_volatility_2d/stochastic_volatility_2d_ukf
    # Particle filter (baseline)
    stochastic_volatility_2d/stochastic_volatility_2d_pf
    # Flow filters (EKF global)
    stochastic_volatility_2d/stochastic_volatility_2d_edh_flow
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_flow
    stochastic_volatility_2d/stochastic_volatility_2d_stochastic_edh
    stochastic_volatility_2d/stochastic_volatility_2d_edh_invertible
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_invertible
    # Flow filters (UKF global)
    stochastic_volatility_2d/stochastic_volatility_2d_edh_flow_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_flow_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_stochastic_edh_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_edh_invertible_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_invertible_ukf
)

TOTAL=${#EXPERIMENTS[@]}
LOGDIR="outputs/stochastic_volatility_2d/logs"
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
