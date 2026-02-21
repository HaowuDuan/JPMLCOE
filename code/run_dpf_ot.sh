#!/usr/bin/env bash
# Run all DPF experiments with OT entropy resampling.
# Usage: bash run_dpf_ot.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/ledh_ot
    hmc/linear_gaussian/bpf_ot
    hmc/cubic_sensor/ledh_ot
    hmc/cubic_sensor/bpf_ot
    hmc/kitagawa/ledh_ot
    hmc/kitagawa/bpf_ot
    hmc/range_bearing/ledh_ot
    hmc/range_bearing/bpf_ot
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
