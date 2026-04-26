#!/usr/bin/env bash
# 4-chain HMC for RB with diagonal mass matrix [47.26, 82.09].
# Outputs to outputs/dpf/hmc/range_bearing/ledh_ot_mass_c{1..4} (separate from the no-mass runs).
#
# Usage:
#   bash run_hmc_4chain_rb_mass.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    hmc/range_bearing/ledh_ot_mass_c1
    hmc/range_bearing/ledh_ot_mass_c2
    hmc/range_bearing/ledh_ot_mass_c3
    hmc/range_bearing/ledh_ot_mass_c4
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== 4-chain HMC: RB with mass matrix ($TOTAL runs) ==="
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

echo "=== All $TOTAL chains complete ==="
echo "Finished: $(date)"
