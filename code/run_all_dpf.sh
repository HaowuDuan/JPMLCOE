#!/usr/bin/env bash
# Run all DPF experiments one by one.
# Usage: bash run_all_dpf.sh
# Results saved to outputs/dpf/<config_name>/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENTS=(
    # Linear Gaussian (1D, trivial baseline)
    linear_gaussian_ledh_hmc_sys
    linear_gaussian_ledh_hmc_soft
    linear_gaussian_ledh_hmc_ot
    linear_gaussian_bpf_hmc_sys
    linear_gaussian_bpf_hmc_soft
    linear_gaussian_bpf_hmc_ot

    # Cubic Sensor (1D, mildly nonlinear)
    cubic_sensor_ledh_hmc_sys
    cubic_sensor_ledh_hmc_soft
    cubic_sensor_ledh_hmc_ot
    cubic_sensor_bpf_hmc_sys
    cubic_sensor_bpf_hmc_soft
    cubic_sensor_bpf_hmc_ot

    # Kitagawa (1D, highly nonlinear)
    kitagawa_ledh_hmc_sys
    kitagawa_ledh_hmc_soft
    kitagawa_ledh_hmc_ot
    kitagawa_bpf_hmc_sys
    kitagawa_bpf_hmc_soft
    kitagawa_bpf_hmc_ot

    # Range-Bearing (2D, nonlinear observation)
    range_bearing_ledh_hmc_sys
    range_bearing_ledh_hmc_soft
    range_bearing_ledh_hmc_ot
    range_bearing_bpf_hmc_sys
    range_bearing_bpf_hmc_soft
    range_bearing_bpf_hmc_ot
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL DPF experiments ==="
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
