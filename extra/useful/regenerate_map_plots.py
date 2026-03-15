"""Regenerate MAP convergence plots from existing summary.json files."""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.visualization_hmc import plot_map_convergence


def main():
    map_root = Path(__file__).parent / "outputs" / "dpf" / "map"
    summary_files = sorted(map_root.rglob("summary.json"))

    if not summary_files:
        print("No MAP summary.json files found.")
        return

    for sf in summary_files:
        out_dir = sf.parent
        with open(sf) as f:
            data = json.load(f)

        param_history = data.get("metadata", {}).get("param_history", {})
        loss_history = data.get("diagnostics", {}).get("loss_history", [])
        true_params = data.get("true_params", {})

        if not param_history or not loss_history:
            print(f"SKIP {out_dir.relative_to(map_root)} — missing data")
            continue

        plot_map_convergence(
            param_history, loss_history, true_params,
            save_path=out_dir / "map_convergence.png",
        )

        # Remove old HMC-style plots
        for old in ("trace_plot.png", "posterior_histograms.png"):
            old_path = out_dir / old
            if old_path.exists():
                old_path.unlink()
                print(f"  Removed {old}")

    print(f"\nDone — processed {len(summary_files)} directories.")


if __name__ == "__main__":
    main()
