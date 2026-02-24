#!/usr/bin/env bash
# Run all stochastic volatility experiments + selected range-bearing experiments.
# Usage: bash run_sv_rb.sh

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    # --- Stochastic Volatility: Kalman ---
    stochastic_volatility/stochastic_volatility_ekf
    stochastic_volatility/stochastic_volatility_ekf_sampled_init
    stochastic_volatility/stochastic_volatility_ukf
    stochastic_volatility/stochastic_volatility_ukf_sampled_init
    # --- Stochastic Volatility: Bootstrap PF ---
    stochastic_volatility/stochastic_volatility_pf
    stochastic_volatility/stochastic_volatility_pf_tf
    # --- Stochastic Volatility: OT resampling ---
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.1
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.3
    stochastic_volatility/stochastic_volatility_pf_ot_eps0.5
    stochastic_volatility/stochastic_volatility_pf_ot_eps1.0
    # --- Stochastic Volatility: Soft resampling ---
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.5
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.7
    stochastic_volatility/stochastic_volatility_pf_soft_alpha0.9
    # --- Stochastic Volatility: Flow filters ---
    stochastic_volatility/stochastic_volatility_edh_flow
    stochastic_volatility/stochastic_volatility_ledh_flow
    stochastic_volatility/stochastic_volatility_stochastic_edh
    stochastic_volatility/stochastic_volatility_kernel_matrix
    stochastic_volatility/stochastic_volatility_kernel_scalar
    # --- Range-Bearing ---
    range_bearing/range_bearing_sde_local_correction_optimal
    range_bearing/range_bearing_stochastic_edh_optimal
)

TOTAL=${#EXPERIMENTS[@]}
echo "=== Running $TOTAL experiments ==="
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    echo "========================================"
    echo "[$n/$TOTAL] $exp"
    echo "========================================"
    python -m src.experiments.run_experiment experiment="$exp" || {
        echo "FAILED: $exp"
        echo ""
        continue
    }
    echo ""
done

echo "=== All $TOTAL experiments complete ==="
