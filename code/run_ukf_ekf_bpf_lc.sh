#!/usr/bin/env bash
# Run UKF, EKF, and BPF variants (lc configs) for Linear Gaussian and Cubic Sensor.
# Usage: bash run_ukf_ekf_bpf_lc.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/ukf_lc
    hmc/linear_gaussian/ekf_lc
    hmc/linear_gaussian/bpf_sys_lc
    hmc/linear_gaussian/bpf_soft_lc
    hmc/linear_gaussian/bpf_ot_lc
    hmc/cubic_sensor/ekf_lc
    hmc/cubic_sensor/ukf_lc
    hmc/cubic_sensor/bpf_sys_lc
    hmc/cubic_sensor/bpf_soft_lc
    hmc/cubic_sensor/bpf_ot_lc
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
