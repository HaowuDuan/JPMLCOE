#!/usr/bin/env bash
# Run all DPF experiments with systematic resampling.
# Usage: bash run_dpf_sys.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    linear_gaussian_ledh_hmc_sys
    linear_gaussian_bpf_hmc_sys
    cubic_sensor_ledh_hmc_sys
    cubic_sensor_bpf_hmc_sys
    kitagawa_ledh_hmc_sys
    kitagawa_bpf_hmc_sys
    range_bearing_ledh_hmc_sys
    range_bearing_bpf_hmc_sys
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL systematic resampling experiments ==="
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
