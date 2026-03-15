# OT Resampling: Make Epsilon Scaling Optional

## Problem

The current `ot_entropy_resample` in `code/src/resampling/ot_entropy.py` always uses **epsilon scaling** — a warm-starting strategy that anneals epsilon from `diameter²` down to the target value through a `tf.while_loop`.

However, the code *already* normalizes particles (centering + scaling to O(1)) inside the `@tf.custom_gradient` wrapper (lines 426-432). After normalization, the cost matrix is O(1) and a fixed epsilon (e.g. 0.5) is well-conditioned — Sinkhorn converges fine from zero initialization without annealing.

The epsilon scaling is therefore redundant and introduces **θ-dependent behavior**: the number of annealing steps varies with particle spread (which varies with θ), affecting the quality of potentials fed to the final solve. While gradient stitching (`tf.stop_gradient` + extrapolation, lines 207-212) correctly blocks gradients through the loops, the forward-pass potential values still vary with θ through this path, indirectly affecting the gradient evaluated at the extrapolation step.

The Corenflos et al. (2021) paper explicitly recommends **fixed epsilon with data normalization, no scaling**.

## What's Changing

### File: `code/src/resampling/ot_entropy.py`

**1. New parameter on `ot_entropy_resample` (line 375):**
- `use_epsilon_scaling: bool = False`
- Default `False` = paper-recommended behavior (fixed epsilon, no annealing)
- Set `True` to get the old behavior

**2. Branching inside `@tf.custom_gradient` (around line 434):**

When `use_epsilon_scaling=False` (new default):
- Compute cost matrix from normalized particles
- Initialize potentials to zeros
- Call `sinkhorn_iteration` directly with the fixed target epsilon
- No annealing loop, no diameter computation

When `use_epsilon_scaling=True` (old behavior):
- Call `sinkhorn_with_epsilon_scaling` as before
- Everything unchanged

The branch is a Python-level `if/else` (not `tf.cond`), which is correct because `use_epsilon_scaling` is a Python bool known at graph-construction time.

### Config files: No changes required
Existing configs only pass `epsilon` in `resampling_config`. The new parameter defaults to `False`, so all existing configs automatically get the new behavior. To opt back in:
```yaml
resampling_config:
  epsilon: 0.5
  use_epsilon_scaling: true
```

## What's NOT Changing

- `sinkhorn_iteration` — untouched
- `sinkhorn_with_epsilon_scaling` — untouched (still available when `use_epsilon_scaling=True`)
- `compute_transport_matrix_from_potentials` — untouched
- Data normalization (centering + scaling) — stays regardless of setting
- Gradient stitching (`stop_gradient` + extrapolation) — stays regardless of setting
- `soft_resample`, `systematic_resample` — untouched

## How to Revert

If the change causes issues, revert by either:

1. **Config-level revert**: Add `use_epsilon_scaling: true` to any `resampling_config` in YAML
2. **Code-level revert**: Change the default from `False` to `True` on line ~383:
   ```python
   use_epsilon_scaling: bool = True  # revert to old behavior
   ```
3. **Full git revert**: `git diff code/src/resampling/ot_entropy.py` to see exact changes, `git checkout code/src/resampling/ot_entropy.py` to restore

## Verification

1. Unit tests: `python -m pytest code/tests/unit/test_resampling.py -v`
2. Quick experiment: `python code/src/experiments/run_dpf_experiment.py --config-name dpf/hmc/linear_gaussian/bpf_ot`
3. Old path still works: add `use_epsilon_scaling: true` to config and re-run

## Architecture Reference

```
ot_entropy_resample()                    # entry point (line 375)
  ├── @tf.custom_gradient wrapper        # lines 422-453
  │   ├── data normalization             # lines 426-432 (always)
  │   ├── if use_epsilon_scaling:
  │   │   └── sinkhorn_with_epsilon_scaling()   # line 218 (annealing loop)
  │   │       ├── compute_diameter()             # θ-dependent init
  │   │       ├── tf.while_loop (scaling)        # variable steps
  │   │       └── sinkhorn_iteration()           # final solve
  │   └── else:                                  # NEW PATH
  │       └── sinkhorn_iteration()               # direct solve, fixed epsilon
  │           ├── tf.while_loop (iterations)     # gradient blocked by stop_gradient
  │           └── extrapolation step             # gradient flows here
  └── compute_transport_matrix_from_potentials() # line 317 (same for both)
```
