#!/usr/bin/env bash
# Sweep OT epsilon and soft alpha for BPF + HMC on linear Gaussian.
# Usage: bash run_bpf_epsilon_sweep.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/linear_gaussian/bpf_ot_0.2
    hmc/linear_gaussian/bpf_ot_0.4
    hmc/linear_gaussian/bpf_ot_0.8
    hmc/linear_gaussian/bpf_ot_1.0
    hmc/linear_gaussian/bpf_soft_0.2
    hmc/linear_gaussian/bpf_soft_0.4
    hmc/linear_gaussian/bpf_soft_0.8
    hmc/linear_gaussian/bpf_soft_1.0
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL epsilon/alpha sweep experiments ==="
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
