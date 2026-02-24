#!/usr/bin/env bash
# Run acoustic tracking LEDH invertible q/n sweep experiments.
# Usage: bash run_at_qn.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    # --- q=1.2 sweep ---
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n25
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n29
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n33
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n37
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.2_n42
    # --- q=1.5 sweep ---
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n25
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n29
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n33
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n37
    acoustic_tracking/acoustic_tracking_ledh_invertible_q1.5_n42
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL experiments ==="
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    echo "========================================"
    echo "[$n/$TOTAL] $exp"
    echo "========================================"
    python -m src.experiments.run_experiment experiment="$exp" || {
        echo "FAILED: $exp"
        echo ""
        continue
    }
    echo ""
done

echo "=== All $TOTAL experiments complete ==="
