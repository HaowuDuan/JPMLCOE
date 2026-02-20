#!/usr/bin/env bash
# Run all DPF experiments with OT entropy resampling.
# Usage: bash run_dpf_ot.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    linear_gaussian_ledh_hmc_ot
    linear_gaussian_bpf_hmc_ot
    cubic_sensor_ledh_hmc_ot
    cubic_sensor_bpf_hmc_ot
    kitagawa_ledh_hmc_ot
    kitagawa_bpf_hmc_ot
    range_bearing_ledh_hmc_ot
    range_bearing_bpf_hmc_ot
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL OT entropy resampling experiments ==="
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
