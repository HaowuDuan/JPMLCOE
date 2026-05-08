#!/usr/bin/env bash
# 4-chain HMC for Kalman / EKF / UKF on 1D Linear Gaussian.
# 12 chains total (3 filters × 4 chains), all run in parallel on CPU.
# GPU launch overhead dominates these 1D, T=50 ops — CPU is faster.
# Each TF process pegs ~1 core; 20-thread box absorbs all 12.
#
# Usage: bash run_hmc_4chain_lg_kalman_family.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

export CUDA_VISIBLE_DEVICES=""

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

CHAINS=(
    kalman_c1 kalman_c2 kalman_c3 kalman_c4
    ekf_c1    ekf_c2    ekf_c3    ekf_c4
)

echo "=== 4-chain LG HMC sweep: kalman, ekf ==="
echo "Started: $(date)"
echo "Launching ${#CHAINS[@]} chains in parallel…"
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
    echo "[fail] one or more chains failed; see $LOG_DIR/" >&2
    exit 1
fi

echo ""
echo "=== Sweep complete ==="
echo "Finished: $(date)"
