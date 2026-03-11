#!/usr/bin/env bash
# Run all filters on stochastic volatility model IN PARALLEL.
# Usage: bash run_stochastic_volatility_filters.sh [MAX_PARALLEL]

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-4}
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    stochastic_volatility/stochastic_volatility_edh_flow
    stochastic_volatility/stochastic_volatility_ekf
    stochastic_volatility/stochastic_volatility_ekf_sampled_init
    stochastic_volatility/stochastic_volatility_kernel_matrix
    stochastic_volatility/stochastic_volatility_kernel_scalar
    stochastic_volatility/stochastic_volatility_ledh_flow
    stochastic_volatility/stochastic_volatility_pf
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.1
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.3
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.5
    stochastic_volatility/stochastic_volatility_pf_ot_eps1.0
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.5
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.7
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.9
    stochastic_volatility/stochastic_volatility_pf_tf
    stochastic_volatility/stochastic_volatility_stochastic_edh
    stochastic_volatility/stochastic_volatility_ukf
    stochastic_volatility/stochastic_volatility_ukf_sampled_init
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
