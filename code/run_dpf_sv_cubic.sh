#!/usr/bin/env bash
# Run BPF DPF experiments for Stochastic Volatility and Cubic Sensor.
# Usage: bash run_dpf_sv_cubic.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    # Cubic Sensor (1D, mildly nonlinear)
    hmc/cubic_sensor/bpf_sys
    hmc/cubic_sensor/bpf_soft
    hmc/cubic_sensor/bpf_ot

    # Stochastic Volatility (1D, non-Gaussian observation)
    hmc/stochastic_volatility/bpf_sys
    hmc/stochastic_volatility/bpf_soft
    hmc/stochastic_volatility/bpf_ot
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL BPF experiments (Cubic Sensor + Stochastic Volatility) ==="
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
