#!/usr/bin/env bash
# Run LEDH CSMC PGibbs experiments (~8 hours total).
#   LEDH+MH:  10s/step × 720 = ~2h
#   LEDH+HMC: 100s/step × 216 = ~6h
# Usage: bash run_pgibbs.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

TOTAL=2
echo "=== Running $TOTAL LEDH CSMC PGibbs experiments (~8h) ==="
echo ""

# 1. LEDH CSMC + MH: ~2h
echo "========================================"
echo "[1/$TOTAL] pgibbs/kitagawa_ledh_csmc_mh (250 burn-in + 470 samples, ~2h)"
echo "========================================"
python -m src.experiments.run_dpf_experiment \
    dpf=pgibbs/kitagawa_ledh_csmc_mh \
    dpf.pgibbs.num_burnin=250 \
    dpf.pgibbs.num_samples=470 \
    || echo "FAILED: pgibbs/kitagawa_ledh_csmc_mh"
echo ""

# 2. LEDH CSMC + HMC: ~6h
echo "========================================"
echo "[2/$TOTAL] pgibbs/kitagawa_ledh_csmc_hmc (70 burn-in + 146 samples, ~6h)"
echo "========================================"
python -m src.experiments.run_dpf_experiment \
    dpf=pgibbs/kitagawa_ledh_csmc_hmc \
    dpf.pgibbs.num_burnin=70 \
    dpf.pgibbs.num_samples=146 \
    || echo "FAILED: pgibbs/kitagawa_ledh_csmc_hmc"
echo ""

echo "=== All $TOTAL experiments complete ==="
