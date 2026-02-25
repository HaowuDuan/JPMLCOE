#!/usr/bin/env bash
# Run EKF, UKF, and BPF (sys, soft, OT) for Range-Bearing model.
# Usage: bash run_range_bearing.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/range_bearing/ekf
    hmc/range_bearing/ukf
    hmc/range_bearing/bpf_sys
    hmc/range_bearing/bpf_soft
    hmc/range_bearing/bpf_ot
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
    python -m src.experiments.run_dpf_experiment dpf="$exp" || {
        echo "FAILED: $exp"
        echo ""
        continue
    }
    echo ""
done

echo "=== All $TOTAL experiments complete ==="
