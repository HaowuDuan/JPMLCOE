#!/usr/bin/env bash
# Run BPF (OT, soft, systematic resampling) small-N runs for Stochastic Volatility model.
# T=300, N=100 particles, 2000 samples, 1000 burn-in, target_accept_prob=0.7
# Usage: bash run_sv_small.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    hmc/stochastic_volatility/bpf_sys_small
    hmc/stochastic_volatility/bpf_soft_small
    hmc/stochastic_volatility/bpf_ot_small
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
