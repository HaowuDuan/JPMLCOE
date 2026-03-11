#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== MAP: range_bearing/ledh (LEDH + OT) ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/range_bearing/ledh'
