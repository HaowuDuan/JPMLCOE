#!/usr/bin/env bash
# Run ALL filters on stochastic volatility model in LOG-SPACE mode.
# Compares Kalman, particle, and flow filters with the log(y^2) transform.
# Usage: bash run_stochastic_volatility_log_filters.sh [MAX_PARALLEL]

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-999}
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    # Kalman filters
    # stochastic_volatility/stochastic_volatility_ekf_log
    # stochastic_volatility/stochastic_volatility_ukf_log
    # # Particle filter (baseline)
    # stochastic_volatility/stochastic_volatility_pf_log
    # Flow filters (EKF global)
    # stochastic_volatility/stochastic_volatility_edh_flow_log
    # stochastic_volatility/stochastic_volatility_ledh_flow_log
    # stochastic_volatility/stochastic_volatility_stochastic_edh_log
    # stochastic_volatility/stochastic_volatility_edh_invertible_log
    # stochastic_volatility/stochastic_volatility_ledh_invertible_log
    # Flow filters (UKF global)
    stochastic_volatility/stochastic_volatility_edh_flow_log_ukf
    stochastic_volatility/stochastic_volatility_ledh_flow_log_ukf
    stochastic_volatility/stochastic_volatility_stochastic_edh_log_ukf
    stochastic_volatility/stochastic_volatility_edh_invertible_log_ukf
    stochastic_volatility/stochastic_volatility_ledh_invertible_log_ukf
    # Kernel flow filters (no global filter)
    # stochastic_volatility/stochastic_volatility_kernel_scalar_log
    # stochastic_volatility/stochastic_volatility_kernel_matrix_log
)

TOTAL=${#EXPERIMENTS[@]}
LOGDIR="outputs/stochastic_volatility/logs"
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
