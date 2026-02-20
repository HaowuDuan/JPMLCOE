#!/usr/bin/env bash
# Run all DPF experiments with soft resampling.
# Usage: bash run_dpf_soft.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    linear_gaussian_ledh_hmc_soft
    linear_gaussian_bpf_hmc_soft
    cubic_sensor_ledh_hmc_soft
    cubic_sensor_bpf_hmc_soft
    kitagawa_ledh_hmc_soft
    kitagawa_bpf_hmc_soft
    range_bearing_ledh_hmc_soft
    range_bearing_bpf_hmc_soft
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL soft resampling experiments ==="
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
