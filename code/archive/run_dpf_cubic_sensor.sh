#!/usr/bin/env bash
# Run all DPF experiments for Cubic Sensor model.
# Usage: bash run_dpf_cubic_sensor.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/cubic_sensor/ledh_sys
    hmc/cubic_sensor/ledh_soft
    hmc/cubic_sensor/ledh_ot
    hmc/cubic_sensor/bpf_sys
    hmc/cubic_sensor/bpf_soft
    hmc/cubic_sensor/bpf_ot
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL Cubic Sensor DPF experiments ==="
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
