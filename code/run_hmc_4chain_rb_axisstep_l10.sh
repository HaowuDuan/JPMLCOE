#!/usr/bin/env bash
# 4-chain HMC for RB with per-axis step_size [0.001, 0.004] and num_leapfrog=10.
# Goal: drive sigma_bearing R-hat from 1.025 (axisstep at L=5) below 1.01 by
# doubling trajectory length per HMC iteration.
# Outputs to outputs/dpf/hmc/range_bearing/ledh_ot_axisstep_l10_c{1..4}.
#
# Usage:
#   bash run_hmc_4chain_rb_axisstep_l10.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    hmc/range_bearing/ledh_ot_axisstep_l10_c1
    hmc/range_bearing/ledh_ot_axisstep_l10_c2
    hmc/range_bearing/ledh_ot_axisstep_l10_c3
    hmc/range_bearing/ledh_ot_axisstep_l10_c4
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== 4-chain HMC: RB axisstep + num_leapfrog=10 ($TOTAL runs) ==="
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
