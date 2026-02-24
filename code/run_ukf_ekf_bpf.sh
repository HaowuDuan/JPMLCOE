#!/usr/bin/env bash
# Run UKF, EKF, and BPF (systematic) for Linear Gaussian and Cubic Sensor.
# Usage: bash run_ukf_ekf_bpf.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/ukf
    hmc/linear_gaussian/ekf
    hmc/linear_gaussian/bpf_sys
    hmc/cubic_sensor/ekf
    hmc/cubic_sensor/ukf
    hmc/cubic_sensor/bpf_sys
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
