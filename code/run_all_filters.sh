#!/usr/bin/env bash
# Run ALL filter experiments (T=100) with bounded parallelism.
# Usage: bash run_all_filters.sh [MAX_PARALLEL]
#   MAX_PARALLEL defaults to 10

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

MAX_PARALLEL=${1:-10}
export TF_FORCE_GPU_ALLOW_GROWTH=true

EXPERIMENTS=(
    # --- 1d_linear (9) ---
    1d_linear/1d_linear_Kalman
    1d_linear/1d_linear_edh_flow
    1d_linear/1d_linear_edh_invertible
    1d_linear/1d_linear_ledh_flow
    1d_linear/1d_linear_ledh_invertible
    1d_linear/1d_linear_ledh_invertible_ot
    1d_linear/1d_linear_ledh_invertible_soft
    1d_linear/1d_linear_pf
    1d_linear/1d_linear_stochastic_edh

    # --- 5d_linear (13) ---
    5d_linear/5d_linear_Kalman
    5d_linear/5d_linear_edh_flow
    5d_linear/5d_linear_edh_invertible
    5d_linear/5d_linear_ekf
    5d_linear/5d_linear_ledh_flow
    5d_linear/5d_linear_ledh_invertible
    5d_linear/5d_linear_ledh_invertible_ot
    5d_linear/5d_linear_ledh_invertible_soft
    5d_linear/5d_linear_pf
    5d_linear/5d_linear_pf_ot
    5d_linear/5d_linear_pf_soft
    5d_linear/5d_linear_stochastic_edh
    5d_linear/5d_linear_ukf

    # --- 5d_linear_partial_strong (13) ---
    5d_linear_partial_strong/5d_partial_strong_Kalman
    5d_linear_partial_strong/5d_partial_strong_edh_flow
    5d_linear_partial_strong/5d_partial_strong_edh_invertible
    5d_linear_partial_strong/5d_partial_strong_ekf
    5d_linear_partial_strong/5d_partial_strong_ledh_flow
    5d_linear_partial_strong/5d_partial_strong_ledh_invertible
    5d_linear_partial_strong/5d_partial_strong_ledh_invertible_ot
    5d_linear_partial_strong/5d_partial_strong_ledh_invertible_soft
    5d_linear_partial_strong/5d_partial_strong_pf
    5d_linear_partial_strong/5d_partial_strong_pf_ot
    5d_linear_partial_strong/5d_partial_strong_pf_soft
    5d_linear_partial_strong/5d_partial_strong_stochastic_edh
    5d_linear_partial_strong/5d_partial_strong_ukf

    # --- 5d_linear_partial_weak (13) ---
    5d_linear_partial_weak/5d_partial_weak_Kalman
    5d_linear_partial_weak/5d_partial_weak_edh_flow
    5d_linear_partial_weak/5d_partial_weak_edh_invertible
    5d_linear_partial_weak/5d_partial_weak_ekf
    5d_linear_partial_weak/5d_partial_weak_ledh_flow
    5d_linear_partial_weak/5d_partial_weak_ledh_invertible
    5d_linear_partial_weak/5d_partial_weak_ledh_invertible_ot
    5d_linear_partial_weak/5d_partial_weak_ledh_invertible_soft
    5d_linear_partial_weak/5d_partial_weak_pf
    5d_linear_partial_weak/5d_partial_weak_pf_ot
    5d_linear_partial_weak/5d_partial_weak_pf_soft
    5d_linear_partial_weak/5d_partial_weak_stochastic_edh
    5d_linear_partial_weak/5d_partial_weak_ukf

    # --- range_bearing (14) ---
    range_bearing/range_bearing_kf
    range_bearing/range_bearing_ekf
    range_bearing/range_bearing_ukf
    range_bearing/range_bearing_pf
    range_bearing/range_bearing_edh_flow
    range_bearing/range_bearing_edh_flow_global
    range_bearing/range_bearing_edh_invertible
    range_bearing/range_bearing_ledh_flow
    range_bearing/range_bearing_ledh_invertible
    range_bearing/range_bearing_kernel_scalar
    range_bearing/range_bearing_kernel_matrix
    range_bearing/range_bearing_stochastic_edh
    range_bearing/range_bearing_sde_local_correction
    range_bearing/range_bearing_sde_local_correction_optimal

    # --- stochastic_volatility (28) ---
    stochastic_volatility/stochastic_volatility_kf
    stochastic_volatility/stochastic_volatility_ekf
    stochastic_volatility/stochastic_volatility_ekf_log
    stochastic_volatility/stochastic_volatility_ukf
    stochastic_volatility/stochastic_volatility_ukf_log
    stochastic_volatility/stochastic_volatility_pf
    stochastic_volatility/stochastic_volatility_pf_log
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.1
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.3
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.5
    stochastic_volatility/stochastic_volatility_pf_ot_eps1.0
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.5
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.7
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.9
    stochastic_volatility/stochastic_volatility_edh_flow
    stochastic_volatility/stochastic_volatility_edh_flow_log
    stochastic_volatility/stochastic_volatility_edh_flow_log_ukf
    stochastic_volatility/stochastic_volatility_edh_invertible_log
    stochastic_volatility/stochastic_volatility_edh_invertible_log_ukf
    stochastic_volatility/stochastic_volatility_ledh_flow
    stochastic_volatility/stochastic_volatility_ledh_flow_log
    stochastic_volatility/stochastic_volatility_ledh_flow_log_ukf
    stochastic_volatility/stochastic_volatility_ledh_invertible_log
    stochastic_volatility/stochastic_volatility_ledh_invertible_log_ukf
    stochastic_volatility/stochastic_volatility_stochastic_edh
    stochastic_volatility/stochastic_volatility_stochastic_edh_log
    stochastic_volatility/stochastic_volatility_kernel_scalar
    stochastic_volatility/stochastic_volatility_kernel_matrix

    # --- stochastic_volatility_2d (14) ---
    stochastic_volatility_2d/stochastic_volatility_2d_ekf
    stochastic_volatility_2d/stochastic_volatility_2d_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_augmented_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_pf
    stochastic_volatility_2d/stochastic_volatility_2d_edh_flow
    stochastic_volatility_2d/stochastic_volatility_2d_edh_flow_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_edh_invertible
    stochastic_volatility_2d/stochastic_volatility_2d_edh_invertible_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_flow
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_flow_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_invertible
    stochastic_volatility_2d/stochastic_volatility_2d_ledh_invertible_ukf
    stochastic_volatility_2d/stochastic_volatility_2d_stochastic_edh
    stochastic_volatility_2d/stochastic_volatility_2d_stochastic_edh_ukf

    # --- acoustic_tracking (8) ---
    acoustic_tracking/acoustic_tracking_ekf
    acoustic_tracking/acoustic_tracking_ukf
    acoustic_tracking/acoustic_tracking_pf
    acoustic_tracking/acoustic_tracking_pf_tf
    acoustic_tracking/acoustic_tracking_edh_flow
    acoustic_tracking/acoustic_tracking_edh_invertible
    acoustic_tracking/acoustic_tracking_ledh_flow
    acoustic_tracking/acoustic_tracking_ledh_invertible
    acoustic_tracking/acoustic_tracking_stochastic_edh

    # --- kitagawa (8) ---
    kitagawa/kitagawa_ekf
    kitagawa/kitagawa_ukf
    kitagawa/kitagawa_pf
    kitagawa/kitagawa_edh_flow
    kitagawa/kitagawa_edh_invertible
    kitagawa/kitagawa_ledh_flow
    kitagawa/kitagawa_ledh_invertible
    kitagawa/kitagawa_stochastic_edh
)

TOTAL=${#EXPERIMENTS[@]}
LOGDIR="outputs/all_filters_logs"
mkdir -p "$LOGDIR"

echo "=== Running $TOTAL filter experiments (max $MAX_PARALLEL parallel) ==="
echo "Logs: $LOGDIR/"
echo "Started: $(date)"
echo ""

RUNNING=0
PIDS=()
NAMES=()

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    name=$(basename "$exp")
    model=$(dirname "$exp")
    n=$((i + 1))
    logfile="$LOGDIR/${model}_${name}.log"

    echo "[$n/$TOTAL] Launching: $exp"
    python -m src.experiments.run_experiment experiment="$exp" > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$exp")
    RUNNING=$((RUNNING + 1))

    if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
        wait -n 2>/dev/null || {
            # wait -n not available (bash < 4.3): poll loop fallback
            while [ "$RUNNING" -ge "$MAX_PARALLEL" ]; do
                for j in "${!PIDS[@]}"; do
                    if ! kill -0 "${PIDS[$j]}" 2>/dev/null; then
                        wait "${PIDS[$j]}" 2>/dev/null || true
                        unset 'PIDS[$j]'
                        RUNNING=$((RUNNING - 1))
                        break
                    fi
                done
                sleep 1
            done
        }
        RUNNING=$((RUNNING - 1))
    fi
done

echo ""
echo "All jobs launched. Waiting for remaining to finish..."

FAILED=0
SUCCEEDED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "  FAILED: ${NAMES[$i]} (exit $status)"
        FAILED=$((FAILED + 1))
    else
        echo "  OK: ${NAMES[$i]}"
        SUCCEEDED=$((SUCCEEDED + 1))
    fi
done

echo ""
echo "=== Done: $SUCCEEDED/$TOTAL succeeded, $FAILED failed ==="
echo "Finished: $(date)"
