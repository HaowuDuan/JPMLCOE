#!/usr/bin/env bash
# Run HMC inference for all OT-resampling experiments:
#   Linear-Gaussian (BPF + LEDH) and Nonlinear (SV1D, SV2D, Range-Bearing) LEDH+OT.
# Usage: bash run_hmc_all_ot.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

export TF_FORCE_GPU_ALLOW_GROWTH=true

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    # Linear-Gaussian
    hmc/linear_gaussian/bpf_ot
    hmc/linear_gaussian/ledh_ot
    # Nonlinear
    hmc/stochastic_volatility/ledh_ot
    hmc/stochastic_volatility_2d/ledh_ot_sigma2
    hmc/range_bearing/ledh_ot
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== Running $TOTAL HMC OT experiments ==="
echo "Started: $(date)"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    echo "========================================"
    echo "[$n/$TOTAL] $exp"
    echo "========================================"
    "$PYTHON_BIN" -u -m src.experiments.run_dpf_experiment dpf="$exp"
    echo ""
done

echo "=== All $TOTAL HMC OT experiments complete ==="
echo "Finished: $(date)"
