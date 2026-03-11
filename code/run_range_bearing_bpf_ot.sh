#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== MAP: range_bearing/bpf_ot ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/range_bearing/bpf_ot'

echo ""
echo "=== HMC: range_bearing/bpf_ot ==="
python src/experiments/run_dpf_experiment.py 'dpf=hmc/range_bearing/bpf_ot'
