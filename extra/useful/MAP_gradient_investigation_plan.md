# MAP Gradient Investigation Plan: LEDH + OT doesn't converge

## Problem Statement

Running MAP optimization on `linear_gaussian/ledh` (LEDH + OT, gradients through resampling),
the parameter `obs_noise_std` passes through the true value (1.0) but the gradient doesn't go
to zero at the optimum. The optimizer overshoots instead of settling.

---

## Root Causes Identified

### 1. Prior gradient at the true value
- Prior: `LogNormal(loc=0, scale=1)` has mode at ~0.37, not 1.0.
- At `obs_noise_std=1.0`: `d/dx log p(x) = -1/x - log(x)/x = -1`.
- Contributes a constant gradient of -1 pulling the MAP away from the true value.
- With T=100 observations the likelihood dominates, but the effect is nonzero.

### 2. Finite-particle bias with fixed seed
- Config: `random_seed: false`, so every step uses seed `[42, 0]`.
- N=200 particles: this single realization's log-likelihood surface has a minimum
  that doesn't exactly coincide with `theta_true=1.0`.
- The gradient of this specific realization at `theta=1.0` is generically nonzero.

### 3. Gradient bias from OT normalization `stop_gradient`
- In `ot_entropy.py:496-500`:
  ```python
  centered = particles - tf.stop_gradient(mean)
  scale_factor = tf.stop_gradient(std * sqrt(dim) + 1e-8)
  scaled = centered / scale_factor
  ```
- When `obs_noise_std` changes, all particles shift together. The true derivative
  of the normalized particles should nearly cancel (mean subtraction removes the
  uniform shift). But `stop_gradient(mean)` prevents this cancellation in the
  backward pass, so the gradient through T overestimates the effect.
- Creates systematic gradient bias in the transport matrix derivative.

### 4. Extrapolation damping attenuates gradient signal
- Sinkhorn extrapolation (`ot_entropy.py:218-229`):
  ```python
  alpha_extra = d * alpha_stop + (1 - d) * alpha_new   # d=0.5
  ```
- With `d=0.5`, only 50% of the gradient flows through (the `alpha_stop` term
  is dead). This attenuates the gradient signal from the Sinkhorn potentials.

### 5. LEDH 29-step Jacobian accumulation (numerical)
- 29 sequential `tf.linalg.det(I + d_lambda * A)` calls compound numerical error.
- Jacobians affect weights, weights affect log-likelihood, log-likelihood gradient
  carries the accumulated numerical noise.

---

## Experiment Plan

### Experiment 1: Isolate prior effect
**Goal**: Confirm the prior isn't the dominant issue.

- **A**: Use a flat (uniform) prior — remove the prior term or set `scale: 100.0`.
- **B**: Keep `LogNormal(0, 1)` as baseline.
- Compare: if gradient at `theta=1.0` is still large without the prior, it's not
  the prior causing the problem.

### Experiment 2: Stochastic vs fixed seed
**Goal**: Determine if fixed-seed bias is the main issue.

- **A**: `random_seed: true` (different PF seed each step → stochastic gradient).
- **B**: `random_seed: false` (current baseline).
- If stochastic gradients converge better, the fixed-seed single-realization
  bias is the dominant effect.
- May need to reduce `learning_rate` for noisy gradients (try 0.005 or 0.001).

### Experiment 3: BPF+OT vs LEDH+OT
**Goal**: Isolate whether the LEDH flow or OT resampling causes gradient bias.

- **A**: `linear_gaussian/bpf_ot` (BPF with OT resampling, no LEDH flow).
- **B**: `linear_gaussian/ledh` (current, LEDH + OT).
- If BPF+OT also shows non-converging gradient → problem is in OT resampling.
- If BPF+OT converges fine → problem is in LEDH Jacobian accumulation.

### Experiment 4: Remove `stop_gradient` on OT normalization
**Goal**: Test if the normalization `stop_gradient` causes the bias.

- Modify `ot_entropy.py:492-500`: remove `tf.stop_gradient` on `mean` and
  `scale_factor`.
  ```python
  # Before:
  centered = particles - tf.stop_gradient(mean)
  scale_factor = tf.stop_gradient(std * tf.sqrt(dimension) + 1e-8)

  # After:
  centered = particles - mean
  scale_factor = std * tf.sqrt(dimension) + 1e-8
  ```
- Risk: may cause gradient instability (the normalization becomes part of the
  backward pass, creating circular dependencies). Monitor for NaN/Inf gradients.
- If gradient bias disappears → confirms this is the root cause.
- May need `clip_gradients: true` or manual gradient clipping for stability.

### Experiment 5: Increase particles
**Goal**: Check if gradient bias decreases with N.

- **A**: N=200 (current)
- **B**: N=500
- **C**: N=1000
- If bias decreases with N, it's a finite-particle effect.
- If bias is constant, it's a systematic algorithmic issue.

### Experiment 6: Float64 precision
**Goal**: Rule out numerical precision as the cause.

- Set `dtype: float64` in the config.
- If gradient at `theta=1.0` becomes closer to zero → numerical precision in the
  29-step Jacobian is the bottleneck.

---

## Execution Order

Run experiments in this order (each informs the next):

1. **Exp 1** (flat prior) — quick config change, rules out trivial cause
2. **Exp 3** (BPF+OT vs LEDH+OT) — isolates OT vs LEDH as the source
3. **Exp 2** (stochastic seed) — tests if averaging fixes it
4. **Exp 4** (remove stop_gradient) — tests the OT normalization hypothesis
5. **Exp 5** (more particles) — confirms scaling behavior
6. **Exp 6** (float64) — rules out numerical precision

## Success Criteria

- The gradient norm at `obs_noise_std ~= 1.0` should be < 1.0 (dominated by prior).
- The parameter should converge to within 0.05 of 1.0 and stabilize.
- The loss curve should flatten at convergence (no oscillation).

---

## Key Files

| File | Role |
|------|------|
| `configs/dpf/map/linear_gaussian/ledh.yaml` | MAP config for LEDH+OT |
| `src/resampling/ot_entropy.py:492-536` | OT resampling with `stop_gradient` on normalization |
| `src/resampling/ot_entropy.py:209-231` | Sinkhorn extrapolation with damping |
| `src/filters/particle/ledh_invertible_hmc.py:198-215` | LEDH 29-step Jacobian accumulation |
| `src/filters/particle/ledh_invertible_hmc.py:318-362` | Compiled filter entry point |
| `src/DF/hmc_runner.py:274-423` | MAP optimizer loop |
