#!/usr/bin/env bash
# Run MH experiments for Linear Gaussian cross-check.
# Compares MH+Kalman (exact) vs MH+BPF (approximate) vs HMC+Kalman (existing).
# Usage: bash run_mh_linear_gaussian.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    mh/linear_gaussian/kalman
    mh/linear_gaussian/bpf_sys
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL MH experiments ==="
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
