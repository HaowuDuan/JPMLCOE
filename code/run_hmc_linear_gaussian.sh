#!/usr/bin/env bash
# Run all HMC experiments for Linear Gaussian model.
# Usage: bash run_hmc_linear_gaussian.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/kalman
    hmc/linear_gaussian/ekf
    hmc/linear_gaussian/ukf
    hmc/linear_gaussian/bpf_sys
    hmc/linear_gaussian/bpf_ot
    hmc/linear_gaussian/bpf_soft
    hmc/linear_gaussian/ledh_sys
    hmc/linear_gaussian/ledh_ot
    hmc/linear_gaussian/ledh_soft
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL Linear Gaussian HMC experiments ==="
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
