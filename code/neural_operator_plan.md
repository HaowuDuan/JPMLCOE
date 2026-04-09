# Neural Operator Resampling: Action Plan

## Overview

Train a neural operator to replace OT resampling in the particle filter. Separate training codebase, load checkpoints in the main code for inference.

## Directory Structure

```
neural_operator/              # Separate from code/, at repo root
  README.md
  pyproject.toml
  configs/                    # Hydra configs for training
    config.yaml
    data/
      pf_rollouts.yaml
    model/
      transport_set_model.yaml
    train/
      default.yaml
    optimizer/
      adamw.yaml
  src/neural_operator/
    data/
      generate.py             # Run PF rollouts, collect (particles, weights) snapshots
      dataset.py              # TF dataset pipeline
      preprocess.py           # Centering, scaling
    models/
      transport_model.py      # Neural operator architecture
      blocks.py               # Reusable layers
    losses/
      supervised_ot.py        # MSE against OT transport target
      distributional.py       # Moment matching, MMD auxiliaries
      constraints.py          # Row/column sum constraints on T
    training/
      train.py                # Training loop
      evaluate.py             # Validation metrics
      checkpointing.py        # Save/load checkpoints
      tb_logging.py           # TensorBoard logging
    export/
      export_inference.py     # Export model + metadata for loading in code/
      metadata.py             # Checkpoint metadata schema
  scripts/
    generate_dataset.py       # Entry point: PF rollouts -> dataset
    train.py                  # Entry point: Hydra + TensorBoard training
    evaluate.py               # Entry point: evaluate on held-out data
    export.py                 # Entry point: export for inference
  outputs/                    # Hydra run dirs (gitignored)
  data_cache/                 # Generated dataset shards (gitignored)
```

## Loader in Main Codebase

File: `code/src/resampling/neural_operator_resample.py`

```python
class NeuralOperatorResampler:
    """Load trained neural operator for resampling."""
    returns_transport_matrix = True

    def __init__(self, export_dir, dtype):
        self.spec = load_metadata(export_dir)
        self.model = build_model_from_spec(self.spec, dtype=dtype)
        self.model.load_weights(...)
        self.model.trainable = False

    @tf.function
    def __call__(self, particles, weights, seed):
        # seed kept for interface compatibility
        T = self.model(particles, weights)
        T = project_to_transport_constraints(T, weights)
        resampled_particles = T @ particles
        uniform_weights = tf.ones_like(weights) / tf.cast(tf.shape(weights)[0], weights.dtype)
        return ResampleResult(
            particles=resampled_particles,
            weights=uniform_weights,
            ancestor_indices=None,
            transport_matrix=T,
        )
```

Key points:
- Load once at filter construction time, not inside the traced call
- Return transport_matrix (needed for covariance transport in LEDH HMC)
- All ops must be TF-native for @tf.function compatibility
- One checkpoint per (N, d) unless explicitly designed for dynamic shapes

## Integration Points

1. `code/src/resampling/__init__.py` — export NeuralOperatorResampler
2. `code/src/utils/resampling_config.py` — add 'neural_operator' option
3. `code/src/filters/particle/ledh_invertible_hmc.py` line 279 — currently hard-codes transport behavior by method name. Change to check `result.transport_matrix is not None`

## Input/Output Specification

- Input: `particles (N, d)`, `weights (N,)` normalized, `seed (2,)` for compatibility
- Output: `ResampleResult` with `transport_matrix (N, N)`, uniform weights
- Internal: model outputs `T_hat (N, N)`, then `x_tilde = T_hat @ x`

## Model Design Constraints

- Permutation equivariant in particle axis
- Weights as per-particle features
- Output parameterizes a valid transport matrix (row/column sum constraints)
- Deterministic inference (no dropout)
- Learn transport plan T, not unordered particle cloud (needed for covariance transport)

## Training Data

1. Run PF rollouts on target models (LG, SV, RB, etc.)
2. At each timestep BEFORE resampling, save `(particles, weights)`
3. Compute OT transport target `T_ot` using existing `ot_entropy_resample`
4. Save both `T_ot` and `x_ot = T_ot @ x` as ground truth

Note: current experiment runner saves `weights_history` but not particle clouds. Need a dedicated data collector.

## Training

- Supervised: MSE on `T_ot` or `x_ot`
- Auxiliary losses: moment matching, transport marginal penalties
- TensorBoard for logging loss, gradient norms, transport quality metrics
- Hydra for hyperparameter management

## Integration Concerns

- Model must be loaded before @tf.function tracing
- Gradient flow: exact through the learned surrogate (not exact OT gradients)
- Dtype: training in float32, inference may need float64 — handle mismatch
- Dense T is O(N^2) memory — acceptable at N=200-1000
- Add gradient tests through the full compiled filter after integration

## Implementation Order

1. Set up `neural_operator/` directory structure
2. Implement data generation (PF rollout snapshots + OT targets)
3. Implement model architecture
4. Train on LG model data first (simplest)
5. Evaluate: compare learned T vs OT T on held-out data
6. Implement loader (`neural_operator_resample.py`)
7. Integrate into filter pipeline
8. Run gradient tests through compiled filter
9. Run MAP/HMC with neural operator resampling
10. Scale to nonlinear models (SV, RB)

## Update: Monge-Ampere Continuous Transport Map (Current Direction)

This section supersedes the supervised OT plan above for the current research direction. The earlier sections are kept as reference because they match the existing Sinkhorn-based integration assumptions, but the current target is different:

- No amortized OT against Sinkhorn labels
- No discrete `N x N` transport-plan prediction
- Solve a continuous transport-map problem between KDE-smoothed densities

### Problem Setup

Given particles `X = {x_i}_{i=1}^N` and normalized particle-filter weights `w_i`, define:

```text
q_h(x) = (1/N) sum_i K_h(x - x_i)
p_h(x) = sum_i w_i K_h(x - x_i)
```

Seek the quadratic-cost Brenier map `T = grad(phi)` with `phi` convex such that `T# q_h = p_h`. For smooth positive KDEs, the Monge-Ampere equation is:

```text
q_h(x) = p_h(T(x)) * det nabla T(x)
```

or, with `T = grad(phi)`:

```text
q_h(x) = p_h(grad(phi)(x)) * det nabla^2 phi(x)
```

At resampling time, apply the continuous map to the current particle locations:

```text
x_tilde_i = T(x_i),    weights_tilde_i = 1/N
```

This produces a map-based resampler, not a discrete OT plan.

### Architecture: GradNet (mGradNet-M)

Use GradNet (Pronav et al. 2024, "Gradient Networks") with the GradNetOT extension
(Shreyas et al. 2025, arXiv:2507.13191) for solving Monge-Ampere.

- Code: https://github.com/cShreyas/GradNetOT
- Foundation: https://github.com/SPronav/GradientNetworks

**mGradNet-M** (modular variant) directly parameterizes the transport map `T(x)`:

- Each module: `g_m(f_m(x)) * ∇f_m(x)`
- Full map: `T(x) = sum_m g_m(f_m(x)) * ∇f_m(x)`
- Jacobian `J_T(x)` is guaranteed **symmetric PSD** by construction
- PSD Jacobian means T is the gradient of a convex potential — no penalty needed

**Advantages over ICNN:**

- Directly parameterizes the map, not the potential
- No second-order differentiation needed (ICNN requires `∇φ` which needs Hessians for training)
- Convexity/monotonicity guaranteed architecturally
- Only needs Jacobian of T (first-order autodiff), not Hessian of φ
- `mGradNet-M` outperforms ICNN/ICGN baselines in GradNetOT experiments

**Adaptation for particle filter resampling:**

- Context: particle cloud `{(x_i, w_i)}` encoded by a permutation-invariant set encoder (DeepSets or Set Transformer)
- The set encoder produces context `c`
- `T(x; c)` is the conditional mGradNet-M map
- Near-identity initialization (important: when weights are uniform, T should be close to identity)
- Bandwidth annealing: start with larger h, decrease

### Training Objective

The main objective should be physics-informed / self-supervised on `(q_h, p_h)`, not supervised on Sinkhorn outputs.

Recommended collocation strategy:

- Sample `x ~ q_h` by choosing a particle index uniformly and adding kernel noise `xi ~ K_h`.
- Evaluate `q_h(x)`, `p_h(T(x))`, and `log det J_theta(x)`.

Recommended losses:

```text
L_push = E_{x ~ q_h} [ -log p_h(T(x)) - log det J_theta(x) ]
L_MA   = E_{x ~ q_h} [ (log q_h(x) - log p_h(T(x)) - log det J_theta(x))^2 ]
```

Interpretation:

- `L_push` is the KL objective for matching the pushforward `T# q_h` to `p_h` up to an additive constant from `q_h`.
- `L_MA` is the pointwise Monge-Ampere / change-of-variables residual.

Add regularizers:

- Convexity / positive-definiteness regularization if the ICNN implementation is only approximately convex
- Small transport-cost penalty `E_q ||T(x) - x||^2` as a stabilizer
- Identity prior when `w` is nearly uniform, since then `p_h` is close to `q_h`

Recommended total loss:

```text
L = L_push + lambda_MA * L_MA + lambda_reg * L_reg + lambda_id * L_id
```

Use supervised targets only for:

- Synthetic pretraining on cases with known analytic maps, e.g. Gaussian-to-Gaussian
- Unit tests and sanity checks

Do not train against Sinkhorn labels if the goal is the Monge-Ampere map itself. That would pull the model back toward a different discrete, entropic problem.

### Bandwidth `h`

`h` is a core modeling choice, not a minor implementation detail.

Effects of `h`:

- Large `h`: densities are smoother, the PDE is easier, gradients are better conditioned, but the transport problem is biased toward an oversmoothed map, often close to identity.
- Small `h`: densities are closer to the empirical measures, but `q_h` and `p_h` become sharply peaked, `log p_h` and `log det J` become stiff, and training / optimization becomes unstable.

Practical rules:

- Use the same kernel family and the same bandwidth matrix `H` for both `q_h` and `p_h`.
- Start from a KDE rule-of-thumb bandwidth, e.g. Scott / Silverman with a shrinkage covariance estimate:

```text
H = c^2 * Sigma_x * N^(-2 / (d + 4))
```

- Treat the scalar multiplier `c` as a tunable regularization parameter.
- Prefer the smallest `c` that still keeps the PDE solve numerically stable.
- Use continuation: solve first at a larger bandwidth, then anneal `h` downward.

What is principled here:

- Pure density-estimation cross-validation is not enough because the objective is transport quality, not density fit.
- Choose `h` using downstream validation: Monge-Ampere residual, conditioning of `log det J`, fit of mapped particles to `p_h`, and filter-level metrics.

Adaptive bandwidths may help later, but the first implementation should use a single global `H`.

### Applying a Continuous Map to Discrete Particles

Applying `T` only to the particle centers is an approximation to transporting the full KDE mass.

What is being approximated:

- The continuous model transports the whole kernel mixture `q_h`.
- The implementation replaces each source kernel by a single mapped point `T(x_i)`.

When this is mild:

- If `T` is smooth on the kernel scale `h`, the center-only approximation error is typically second-order in `h`.
- In that regime, `T(x_i)` is a reasonable summary of how the local kernel mass moves.

When this can be significant:

- If `h` is very small, the continuous problem is already close to a singular discrete one.
- If `T` has large curvature on the scale of a kernel, moving only the center misses the spread and distortion of that kernel.
- If the target has sharp local structure, the mapped point cloud can match the PDE while still being a poor discrete particle approximation.

Recommended tests:

1. Compare the mapped empirical cloud `{T(x_i)}` against `p_h` using KDE discrepancy, MMD, and moment error.
2. Jitter a subset of particles with `xi ~ K_h` and measure how much `T(x_i + xi)` differs from `T(x_i)`.
3. Evaluate filter-level metrics; if those are good, small residual particle mismatch may not matter.

If center-only transport is too crude, the next step is not a transport matrix. It is to transport local sigma points or jittered samples from each kernel.

### Output Structure: Map vs. Transport Matrix

This approach does not naturally produce an `N x N` transport matrix.

Natural outputs:

- Mapped particles `x_tilde_i = T(x_i)`
- Local Jacobians `J_i = nabla T(x_i)`
- Optionally local Hessians if diagnostics need them

What it does not give:

- A canonical discrete coupling `T_ij`
- Exact ancestry weights or discrete mass splits between original particles

LEDH covariance transport:

The current OT path transports covariances via an N×N transport matrix:
```python
covs_new = einsum('ij,jkl->ikl', T_mat, covs)  # mix old covs by transport weights
```

GradNet provides per-particle Jacobians `J_i = ∇T(x_i)` of shape `(d, d)`. Covariance
transport through a nonlinear map is the standard propagation:
```python
covs_new_i = J_i @ covs_old_i @ J_i^T  # local geometric stretching/rotation
```

This is actually more principled than the N×N matrix mixing — it says "the local
geometry around particle i was stretched/rotated by J_i" rather than "particle i's
covariance is a weighted average of all old covariances."

The LEDH code change is straightforward:
```python
# Old (OT transport matrix):
covs = tf.einsum('ij,jkl->ikl', T_mat, covs)

# New (Jacobian-based):
J = compute_jacobians(T, particles)  # (N, d, d)
covs = tf.einsum('nij,njk,nlk->nil', J, covs, J)  # J @ cov @ J^T per particle
```

GradNet's Jacobian is guaranteed symmetric PSD, so this always produces valid
(PSD) covariance matrices.

### Comparison to the Existing Implicit Sinkhorn Path

The two methods solve different problems.

Implicit Sinkhorn:

- Solves the exact discrete entropy-regularized OT problem used by the current filter
- Returns a discrete transport matrix
- Implicit backward gives gradients of the implemented discrete forward map up to solver tolerance
- Best option when fidelity to the current OT resampler matters

Monge-Ampere with KDEs:

- Solves a continuous quadratic-cost transport problem between smoothed densities
- Returns a transport map and local Jacobians, not a transport matrix
- If solved to convergence, gradients are correct for the smoothed continuous problem
- Adds bias from KDE smoothing and from converting a continuous map back to discrete particles

Gradient-quality assessment:

- The Monge-Ampere path should give smoother, better-conditioned gradients because KDE smoothing removes part of the discrete OT non-smoothness.
- Those gradients are not gradients of the current discrete Sinkhorn resampler. They are gradients of a different approximation.
- The existing implicit Sinkhorn path is higher fidelity to the current filter objective and is the right baseline for HMC correctness.
- The Monge-Ampere path may still be useful for MAP or optimization settings if the smoother gradient and lower variance outweigh the modeling bias.

So the comparison is:

- Better fidelity: implicit Sinkhorn
- Potentially smoother gradients: Monge-Ampere KDE map
- Built-in transport matrix: implicit Sinkhorn
- Natural continuous-map structure: Monge-Ampere

### Revised Implementation Order

1. Clone GradNetOT repo, study the code, port core mGradNet-M architecture to TensorFlow.
2. Implement TF-native KDE utilities for `q_h`, `p_h`, score evaluation, and sampling collocation points from `q_h`.
3. Implement conditional mGradNet-M with permutation-invariant set encoder for context `(x_i, w_i, h)`.
4. Implement autodiff-based evaluation of `T(x)`, `J(x)`, and `log det J(x)`.
5. Train on analytic sanity checks: 1D monotone cases, Gaussian-to-Gaussian maps.
6. Add physics-informed losses `L_push` and `L_MA`; validate pushforward KDE matches `p_h`.
7. Implement map-based resampler in `code/src/resampling/` returning mapped particles + Jacobians.
8. Implement Jacobian-based covariance transport in LEDH: `covs_new_i = J_i @ covs_i @ J_i^T`.
9. Benchmark against implicit Sinkhorn on:
   - state-estimation error
   - gradient finite-difference ratios
   - HMC / MAP behavior
   - runtime
10. Scale to nonlinear models (SV, RB).
