#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== MAP: range_bearing/bpf_ot (both params) ==="
python src/experiments/run_dpf_experiment.py 'dpf=map/range_bearing/bpf_ot'

#echo ""
#echo "=== MAP: range_bearing/bpf_ot_range_only (freeze bearing=0.1) ==="
# python src/experiments/run_dpf_experiment.py 'dpf=map/range_bearing/bpf_ot_range_only'

# echo ""
# echo "=== MAP: range_bearing/bpf_ot_bearing_only (freeze range=0.1) ==="
# python src/experiments/run_dpf_experiment.py 'dpf=map/range_bearing/bpf_ot_bearing_only'

# echo ""
# echo "=== MAP: range_bearing/bpf_ot_range_only T=200 (freeze bearing=0.1) ==="
# python src/experiments/run_dpf_experiment.py 'dpf=map/range_bearing/bpf_ot_range_only_T200'

# echo ""
# echo "=== MAP: range_bearing/bpf_ot_bearing_only T=200 (freeze range=0.1) ==="
# python src/experiments/run_dpf_experiment.py 'dpf=map/range_bearing/bpf_ot_bearing_only_T200'
