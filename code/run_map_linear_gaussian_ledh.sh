#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== MAP: linear_gaussian/ledh (LEDH + OT, grad through) ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/linear_gaussian/ledh'

echo ""
echo "=== MAP: linear_gaussian/ledh_soft (LEDH + soft, grad through) ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/linear_gaussian/ledh_soft'

echo ""
echo "=== MAP: linear_gaussian/ledh_sys (LEDH + OT, stop_gradient) ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/linear_gaussian/ledh_sys'
