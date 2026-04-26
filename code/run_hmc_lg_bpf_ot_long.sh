#!/usr/bin/env bash
# LG + BPF+OT 4 chains, LONG version: num_samples=2000, num_burnin=500.
# Purpose: chain 3 of the short run got stuck (acceptance 0.97, mean 1.29
# vs others ~1.20). Longer chains should let it escape and R-hat drop.
#
# Parallelism: runs 2 chains at a time on one GPU. Per-chain peak ~885 MB,
# so 2x = ~1.8 GB on a 24 GB card — fine under TF_FORCE_GPU_ALLOW_GROWTH.
# If OOM or CUDA context errors, change PARALLEL=1 below to run sequentially.
#
# Wall: per chain ~2.9 hours at 2000+500 steps. 4 chains, 2 in parallel:
# ~5.8 hours total. 4 chains sequential: ~11.5 hours.
#
# Usage: bash run_hmc_lg_bpf_ot_long.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

# Allow multiple TF processes to share the GPU
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}

PARALLEL=${PARALLEL:-2}   # set to 1 for sequential if parallel crashes

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

CHAINS=(
    hmc/linear_gaussian/bpf_ot_long_c1
    hmc/linear_gaussian/bpf_ot_long_c2
    hmc/linear_gaussian/bpf_ot_long_c3
    hmc/linear_gaussian/bpf_ot_long_c4
)

TOTAL=${#CHAINS[@]}

echo "=== LG+BPF+OT long chains: $TOTAL runs, parallelism=$PARALLEL ==="
echo "Started: $(date)"
echo ""

run_one () {
    local exp="$1"
    local n="$2"
    echo "========================================"
    echo "[$n/$TOTAL] $exp  (pid=$$)"
    echo "========================================"
    "$PYTHON_BIN" -u -m src.experiments.run_dpf_experiment dpf="$exp"
}

i=0
while [ $i -lt $TOTAL ]; do
    pids=()
    for ((k=0; k<PARALLEL && i<TOTAL; k++, i++)); do
        run_one "${CHAINS[$i]}" $((i + 1)) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done

echo ""
echo "=== LG+BPF+OT long chains complete ==="
echo "Finished: $(date)"
