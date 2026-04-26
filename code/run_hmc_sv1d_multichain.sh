#!/usr/bin/env bash
# 1D SV multi-chain HMC for R-hat.
# LEDH+OT, 4 chains. Infers alpha only (Beta prior centered at truth 0.91).
# Seeds 42-45; initial alpha spread across [0.6, 0.97].
#
# Usage: bash run_hmc_sv1d_multichain.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    hmc/stochastic_volatility/ledh_ot_c1
    hmc/stochastic_volatility/ledh_ot_c2
    hmc/stochastic_volatility/ledh_ot_c3
    hmc/stochastic_volatility/ledh_ot_c4
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== SV1D multi-chain HMC: $TOTAL runs ==="
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

echo "=== SV1D multi-chain complete ==="
echo "Finished: $(date)"
