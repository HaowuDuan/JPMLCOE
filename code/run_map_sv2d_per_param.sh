#!/usr/bin/env bash
# MAP for each SV2D parameter individually.
# Diagnoses which params are well-identified vs biased by PF noise.
#
# Usage:
#   bash run_map_sv2d_per_param.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    map/stochastic_volatility_2d/ledh_ot_a1
    map/stochastic_volatility_2d/ledh_ot_a2
    map/stochastic_volatility_2d/ledh_ot_sigma1



    
    #map/stochastic_volatility_2d/ledh_ot_sigma2
    map/stochastic_volatility_2d/ledh_ot_b
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== SV2D MAP per-parameter sweep ($TOTAL runs) ==="
echo "Started: $(date)"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    echo "========================================"
    echo "[$n/$TOTAL] $exp  (started $(date))"
    echo "========================================"
    "$PYTHON_BIN" -u -m src.experiments.run_dpf_experiment dpf="$exp"
    echo ""
done

echo "=== All $TOTAL MAPs complete ==="
echo "Finished: $(date)"
