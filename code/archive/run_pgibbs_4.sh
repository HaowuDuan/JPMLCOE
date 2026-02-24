#!/usr/bin/env bash
# Run all 4 PGibbs experiments overnight with full sample counts.
# Time estimates based on test run:
#   BPF+MH:   ~3s/step × 3000 = ~2.5h
#   BPF+HMC:  ~93s/step × 3000 = ~77h → reduced: 250+470 = ~19h
#   LEDH+MH:  ~10s/step × 3000 = ~8h
#   LEDH+HMC: ~100s/step × 3000 = ~83h → reduced: 70+146 = ~6h
# Usage: bash run_pgibbs_4.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

TOTAL=4
echo "=== Running $TOTAL PGibbs experiments ==="
echo ""

# 1. BPF CSMC + MH (~2.5h)
echo "========================================"
echo "[1/$TOTAL] pgibbs/kitagawa_bpf_mh (1000 burn-in + 2000 samples)"
echo "========================================"
python -m src.experiments.run_dpf_experiment \
    dpf=pgibbs/kitagawa_bpf_mh \
    dpf.pgibbs.num_burnin=1000 \
    dpf.pgibbs.num_samples=2000 \
    || echo "FAILED: pgibbs/kitagawa_bpf_mh"
echo ""

# 2. BPF CSMC + HMC (~19h)
echo "========================================"
echo "[2/$TOTAL] pgibbs/kitagawa_bpf_hmc (250 burn-in + 470 samples)"
echo "========================================"
python -m src.experiments.run_dpf_experiment \
    dpf=pgibbs/kitagawa_bpf_hmc \
    dpf.pgibbs.num_burnin=250 \
    dpf.pgibbs.num_samples=470 \
    || echo "FAILED: pgibbs/kitagawa_bpf_hmc"
echo ""

# 3. LEDH CSMC + MH (~8h)
echo "========================================"
echo "[3/$TOTAL] pgibbs/kitagawa_ledh_csmc_mh (250 burn-in + 470 samples)"
echo "========================================"
python -m src.experiments.run_dpf_experiment \
    dpf=pgibbs/kitagawa_ledh_csmc_mh \
    dpf.pgibbs.num_burnin=250 \
    dpf.pgibbs.num_samples=470 \
    || echo "FAILED: pgibbs/kitagawa_ledh_csmc_mh"
echo ""

# 4. LEDH CSMC + HMC (~6h)
echo "========================================"
echo "[4/$TOTAL] pgibbs/kitagawa_ledh_csmc_hmc (70 burn-in + 146 samples)"
echo "========================================"
python -m src.experiments.run_dpf_experiment \
    dpf=pgibbs/kitagawa_ledh_csmc_hmc \
    dpf.pgibbs.num_burnin=70 \
    dpf.pgibbs.num_samples=146 \
    || echo "FAILED: pgibbs/kitagawa_ledh_csmc_hmc"
echo ""

echo "=== All $TOTAL experiments complete ==="
