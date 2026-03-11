#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== MAP: kitagawa/bpf_ot (BPF + OT) ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/kitagawa/bpf_ot'

echo ""
echo "=== MAP: kitagawa/ledh (LEDH + OT) ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/kitagawa/ledh'
