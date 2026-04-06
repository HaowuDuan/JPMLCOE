#!/bin/bash
# Smoke tests for 2D Stochastic Volatility HMC parameter estimation
# Run MAP first (fast), then HMC if MAP succeeds.
# Usage: bash run_sv2d_smoke.sh [map|hmc|all]

set -e

MODE=${1:-all}
RUNNER="python src/experiments/run_dpf_experiment.py"

echo "=========================================="
echo " 2D SV Smoke Tests — mode: $MODE"
echo "=========================================="

# --- MAP ---
if [[ "$MODE" == "map" || "$MODE" == "all" ]]; then
    echo ""
    echo "=== MAP Level 0: sigma2 only ==="
    $RUNNER dpf=map/stochastic_volatility_2d/ledh_ot_sigma2

    echo ""
    echo "=== MAP Level 1: a2 only ==="
    $RUNNER dpf=map/stochastic_volatility_2d/ledh_ot_a2

    echo ""
    echo "=== MAP Level 2: a2 + sigma2 ==="
    $RUNNER dpf=map/stochastic_volatility_2d/ledh_ot_a2_sigma2

    echo ""
    echo "=== MAP Level 3: all 4 params ==="
    $RUNNER dpf=map/stochastic_volatility_2d/ledh_ot_all
fi

# --- HMC ---
if [[ "$MODE" == "hmc" || "$MODE" == "all" ]]; then
    echo ""
    echo "=== HMC Level 0: sigma2 only ==="
    $RUNNER dpf=hmc/stochastic_volatility_2d/ledh_ot_sigma2

    echo ""
    echo "=== HMC Level 1: a2 only ==="
    $RUNNER dpf=hmc/stochastic_volatility_2d/ledh_ot_a2

    echo ""
    echo "=== HMC Level 2: a2 + sigma2 ==="
    $RUNNER dpf=hmc/stochastic_volatility_2d/ledh_ot_a2_sigma2

    echo ""
    echo "=== HMC Level 3: all 4 params ==="
    $RUNNER dpf=hmc/stochastic_volatility_2d/ledh_ot_all
fi

echo ""
echo "=========================================="
echo " Done."
echo "=========================================="
