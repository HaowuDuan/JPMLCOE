#!/usr/bin/env bash
# Run MAP for 1D Linear-Gaussian with OT-resampling BPF and LEDH filters.
# Forces CPU only (no GPU).
# Usage: bash run_map_linear_gaussian_ot_cpu.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

# Force CPU — hide all GPUs from TensorFlow
export CUDA_VISIBLE_DEVICES=""
export TF_FORCE_GPU_ALLOW_GROWTH=true

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    map/linear_gaussian/bpf_ot
    map/linear_gaussian/ledh_ot
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== Running $TOTAL Linear-Gaussian MAP OT experiments (CPU ONLY) ==="
echo "CUDA_VISIBLE_DEVICES=\"\" (GPU disabled)"
echo "Started: $(date)"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    echo "========================================"
    echo "[$n/$TOTAL] $exp"
    echo "========================================"
    "$PYTHON_BIN" -m src.experiments.run_dpf_experiment dpf="$exp"
    echo ""
done

echo "=== All $TOTAL MAP OT experiments complete (CPU) ==="
echo "Finished: $(date)"
