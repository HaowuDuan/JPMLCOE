#!/usr/bin/env bash
# SV2D MAP epsilon sweep: epsilon=2.0 then epsilon=5.0
# Test whether raising OT Sinkhorn epsilon smooths the likelihood surface
# enough for MAP to converge from sigma2=2.0 initial guess.
# Usage: bash run_map_sv2d_epsilon_sweep.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    map/stochastic_volatility_2d/ledh_ot_sigma2_eps2
    map/stochastic_volatility_2d/ledh_ot_sigma2_eps5
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== SV2D MAP epsilon sweep: $TOTAL experiments ==="
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

echo "=== SV2D MAP epsilon sweep complete ==="
echo "Finished: $(date)"
