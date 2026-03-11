#!/usr/bin/env bash
# Run HMC experiments: Linear Gaussian (2 noise params) + Range Bearing.
# Usage: bash run_hmc_linear_gaussian_full.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian_full/ekf
    hmc/linear_gaussian_full/ukf
    hmc/linear_gaussian_full/bpf_sys
    hmc/linear_gaussian_full/bpf_ot
    hmc/linear_gaussian_full/bpf_soft
    hmc/linear_gaussian_full/ledh_sys
    hmc/linear_gaussian_full/ledh_ot
    hmc/linear_gaussian_full/ledh_soft
    hmc/range_bearing/ekf
    hmc/range_bearing/ukf
    hmc/range_bearing/bpf_ot
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL Linear Gaussian (2-noise-param) HMC experiments ==="
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
