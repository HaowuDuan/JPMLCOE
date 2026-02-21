#!/usr/bin/env bash
# Run all DPF experiments with soft resampling.
# Usage: bash run_dpf_soft.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/ledh_soft
    hmc/linear_gaussian/bpf_soft
    hmc/cubic_sensor/ledh_soft
    hmc/cubic_sensor/bpf_soft
    hmc/kitagawa/ledh_soft
    hmc/kitagawa/bpf_soft
    hmc/range_bearing/ledh_soft
    hmc/range_bearing/bpf_soft
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
