#!/usr/bin/env bash
# Run KF baseline on SV and Range-Bearing models.
set -e
cd "$(cd "$(dirname "$0")" && pwd)"
export TF_FORCE_GPU_ALLOW_GROWTH=true

python -m src.experiments.run_experiment experiment=stochastic_volatility/stochastic_volatility_kf
python -m src.experiments.run_experiment experiment=range_bearing/range_bearing_kf
