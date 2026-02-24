#!/usr/bin/env bash
# Run BPF experiments with systematic resampling only.
# Usage: bash run_dpf_sys_bpf.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/bpf_sys
    hmc/cubic_sensor/bpf_sys
    hmc/kitagawa/bpf_sys
    hmc/range_bearing/bpf_sys
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
