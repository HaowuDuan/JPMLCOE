#!/usr/bin/env bash
# Run all filters on kitagawa model IN PARALLEL.
# Usage: bash run_kitagawa_filters.sh [MAX_PARALLEL]

cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-4}
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    kitagawa/kitagawa_edh_flow
    kitagawa/kitagawa_edh_invertible
    kitagawa/kitagawa_ekf
    kitagawa/kitagawa_ledh_flow
    kitagawa/kitagawa_ledh_flow_1e4
    kitagawa/kitagawa_ledh_invertible
    kitagawa/kitagawa_ledh_invertible_1e4
    kitagawa/kitagawa_ledh_invertible_bimodal
    kitagawa/kitagawa_ledh_invertible_bimodal_k1
    kitagawa/kitagawa_ledh_invertible_bimodal_k10
    kitagawa/kitagawa_ledh_invertible_bimodal_k5
    kitagawa/kitagawa_ledh_invertible_bimodal_k7
    kitagawa/kitagawa_pf
    kitagawa/kitagawa_pf_1e4
    kitagawa/kitagawa_stochastic_edh
    kitagawa/kitagawa_ukf
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
