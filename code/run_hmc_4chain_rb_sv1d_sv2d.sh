#!/usr/bin/env bash
# 4-chain HMC for RB, SV1D, SV2D sequentially. Budget ~86 hours.
# Per-step cost from existing summaries:
#   RB:   37.2 s/step x 600 steps x 4 chains = 24.8 h
#   SV1D: 21.4 s/step x 700 steps x 4 chains = 16.6 h
#   SV2D: 52.2 s/step x 700 steps x 4 chains = 40.6 h
# Total: ~82 h.
#
# Usage:
#   bash run_hmc_4chain_rb_sv1d_sv2d.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

EXPERIMENTS=(
    # hmc/range_bearing/ledh_ot_c1
    # hmc/range_bearing/ledh_ot_c2
    # hmc/range_bearing/ledh_ot_c3
    # hmc/range_bearing/ledh_ot_c4
    # 
    
    
    hmc/stochastic_volatility_2d/ledh_ot_sigma2_c1
    hmc/stochastic_volatility_2d/ledh_ot_sigma2_c2
    hmc/stochastic_volatility_2d/ledh_ot_sigma2_c3
    hmc/stochastic_volatility_2d/ledh_ot_sigma2_c4
)

TOTAL=${#EXPERIMENTS[@]}

echo "=== 4-chain HMC: RB + SV1D + SV2D ($TOTAL runs) ==="
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
