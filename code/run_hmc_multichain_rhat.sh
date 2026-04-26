#!/usr/bin/env bash
# Multi-chain HMC for R-hat diagnostics.
# 3 experiments x 4 chains each = 12 runs.
# Each chain has its own seed and own initial parameter value.
# Outputs go to outputs/dpf/hmc/<model>/<filter>_c{1..4}/.
# Post-run: stack samples across chains, compute R-hat per parameter.
#
# Usage: bash run_hmc_multichain_rhat.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    hmc/linear_gaussian/ledh_ot_c1
    hmc/linear_gaussian/ledh_ot_c2
    hmc/linear_gaussian/ledh_ot_c3
    hmc/linear_gaussian/ledh_ot_c4
    hmc/linear_gaussian/bpf_ot_c1
    hmc/linear_gaussian/bpf_ot_c2
    hmc/linear_gaussian/bpf_ot_c3
    hmc/linear_gaussian/bpf_ot_c4
    hmc/range_bearing/ledh_ot_c1
    hmc/range_bearing/ledh_ot_c2
    hmc/range_bearing/ledh_ot_c3
    hmc/range_bearing/ledh_ot_c4
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== Multi-chain HMC R-hat sweep: $TOTAL runs ==="
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

echo "=== Multi-chain sweep complete ==="
echo "Finished: $(date)"
