#!/usr/bin/env bash
# Run MAP for nonlinear models with LEDH + OT resampling.
# Models: SV1D, SV2D (sigma2 only), Range Bearing
# Usage: bash run_map_nonlinear_ledh_ot.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    map/stochastic_volatility/ledh_ot
    map/stochastic_volatility_2d/ledh_ot_sigma2
    map/range_bearing/ledh
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== Running $TOTAL nonlinear LEDH+OT MAP experiments ==="
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

echo "=== All $TOTAL experiments complete ==="
echo "Finished: $(date)"
