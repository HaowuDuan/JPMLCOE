#!/usr/bin/env bash
# 4-chain HMC for UKF on 1D Linear Gaussian — GPU.
# Companion to run_hmc_4chain_lg_kalman_family.sh; runs alongside that script
# (which should have ukf_* commented out).
# All 4 UKF chains share one GPU via TF memory growth.
#
# Usage: bash run_hmc_4chain_lg_ukf_gpu.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

# Without memory growth, the first TF process grabs all 24 GB and the rest OOM.
export TF_FORCE_GPU_ALLOW_GROWTH=true

LOG_DIR="outputs/dpf/hmc/linear_gaussian/_4chain_logs"
mkdir -p "$LOG_DIR"

run_chain() {
    local exp="$1"
    local label="$2"
    local logfile="$LOG_DIR/${label}.log"
    echo "[start] $label  -> $logfile"
    "$PYTHON_BIN" -u -m src.experiments.run_dpf_experiment dpf="$exp" \
        > "$logfile" 2>&1
    echo "[done ] $label  (exit $?)"
}

CHAINS=(ukf_c1 ukf_c2 ukf_c3 ukf_c4)

echo "=== 4-chain UKF LG HMC sweep on GPU ==="
echo "Started: $(date)"
echo "Launching ${#CHAINS[@]} UKF chains in parallel on GPU…"
echo ""

pids=()
for chain in "${CHAINS[@]}"; do
    run_chain "hmc/linear_gaussian/${chain}" "${chain}" &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=1
    fi
done

if [[ $fail -ne 0 ]]; then
    echo ""
    echo "[fail] one or more UKF chains failed; see $LOG_DIR/" >&2
    exit 1
fi

echo ""
echo "=== UKF sweep complete ==="
echo "Finished: $(date)"
