#!/usr/bin/env bash
# Run all DPF experiments one by one.
# Usage: bash run_all_dpf.sh
# Results saved to outputs/dpf/<config_name>/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENTS=(
    # Linear Gaussian (1D, trivial baseline)
    hmc/linear_gaussian/ledh_sys
    hmc/linear_gaussian/ledh_soft
    hmc/linear_gaussian/ledh_ot
    hmc/linear_gaussian/bpf_sys
    hmc/linear_gaussian/bpf_soft
    hmc/linear_gaussian/bpf_ot

    # Cubic Sensor (1D, mildly nonlinear)
    hmc/cubic_sensor/ledh_sys
    hmc/cubic_sensor/ledh_soft
    hmc/cubic_sensor/ledh_ot
    hmc/cubic_sensor/bpf_sys
    hmc/cubic_sensor/bpf_soft
    hmc/cubic_sensor/bpf_ot

    # Kitagawa (1D, highly nonlinear)
    hmc/kitagawa/ledh_sys
    hmc/kitagawa/ledh_soft
    hmc/kitagawa/ledh_ot
    hmc/kitagawa/bpf_sys
    hmc/kitagawa/bpf_soft
    hmc/kitagawa/bpf_ot

    # Range-Bearing (2D, nonlinear observation)
    hmc/range_bearing/ledh_sys
    hmc/range_bearing/ledh_soft
    hmc/range_bearing/ledh_ot
    hmc/range_bearing/bpf_sys
    hmc/range_bearing/bpf_soft
    hmc/range_bearing/bpf_ot

    # Stochastic Volatility (1D, non-Gaussian observation)
    hmc/stochastic_volatility/bpf_sys
    hmc/stochastic_volatility/bpf_soft
    hmc/stochastic_volatility/bpf_ot
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
