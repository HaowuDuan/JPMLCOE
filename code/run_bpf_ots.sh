#!/usr/bin/env bash
# Run BPF (OT and soft resampling) for Linear Gaussian and Cubic Sensor.
# Usage: bash run_bpf_ots.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/bpf_ot
    hmc/linear_gaussian/bpf_soft
    hmc/cubic_sensor/bpf_ot
    hmc/cubic_sensor/bpf_soft
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
