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

### Framework: Pure TensorFlow

All training and inference in TensorFlow. No PyTorch.

**Why this works:**
- d=1-4: analytic Jacobians are cheap and avoid nested tapes entirely
- Existing codebase already hand-writes batched Jacobians (cubic_sensor.py, stochastic_volatility.py)
- Existing graph-safe logdet utilities in linalg.py
- Same Hydra + TensorBoard infrastructure as the rest of the project

**Jacobian strategy:** Analytic, not autodiff.

Each mGradNet-M module: `T_m(x) = W_m^T phi(s_beta * (W_m x + b_m))`

Module Jacobian: `J_m = W_m^T diag(phi'(...) * s_beta) W_m`

For WSoftmax activation: `J_softmax = diag(p) - p p^T`, so `J_m = W_m^T (diag(p) - p p^T) * s_beta W_m`

Full map Jacobian: `J_T = Σ_m a_m J_m` (plus identity from residual parameterization)

Use `tape.batch_jacobian` only as unit test oracle to verify analytic Jacobians.

**Residual parameterization:** `T(x,c) = x + f(x,c)` with near-zero init.
- Makes `J = I + J_f`
- `logdet(I + J_f)` is safer than `logdet(J)` for near-identity maps
- When weights are uniform, T should be close to identity — residual parameterization gives this for free

**Architecture port:**
- Custom `tf.keras.layers.Layer` with one weight matrix W and explicit matmuls (W x, z W^T)
- Do NOT use two Dense layers for tied transpose
- Positive scales via `tf.nn.softplus(raw_variable)`
- WTanh, WSoftmax as custom activation layers

**Training loop:**
```python
schedule = tf.keras.optimizers.schedules.CosineDecay(...)
opt = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=cfg.wd)
writer = tf.summary.create_file_writer(logdir)

@tf.function(reduce_retracing=True)
def train_step(batch, seed):
    with tf.GradientTape() as tape:
        # Branch: encode (particles, weights) -> context c
        c = encoder(batch['particles'], batch['weights'])
        # Trunk: evaluate T and analytic J at query points
        T_x, J_x = trunk.forward_and_jacobian(batch['query_points'], c)
        # Loss: Monge-Ampere residual + regularizers
        loss, metrics = compute_ma_loss(T_x, J_x, batch, cfg)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, grad_norm = tf.clip_by_global_norm(grads, cfg.clip_norm)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return metrics
```

TensorBoard logging outside compiled step. Hydra for config.

**Gotchas:**
- Keep encoder and pointwise map separate (correct Jacobian semantics)
- Reuse existing `graph_safe_log_abs_det` from `linalg.py`
- Build model before tracing (avoid lazy variable creation)
- `tf.linalg.slogdet` can fail on singular matrices — use custom safe version

### Conditioning Architecture

**Branch (set encoder):** Weighted DeepSets.
- Token: `t_i = [x_i, log w_i, w_i]` (dimension d+2)
- Per-particle encoder: `h_i = φ(t_i)` (small MLP)
- Weighted pool: `m = Σ_i w_i h_i`
- Add summary stats: weighted mean/cov of x, entropy of w
- Output: context vector c

**Trunk (conditional mGradNet-M):** Context modulates gates and biases.
```
u_m(x,c) = W_m x + U_m c + b_m(c)
T(x;c) = x + Σ_m a_m(c) W_m^T act(τ_m(c) u_m(x,c))
```
With `a_m(c), τ_m(c) > 0` (via softplus), convexity in x preserved for any c.

**Training data:** Hybrid.
1. Online synthetic weighted particle clouds for diversity
2. Offline OT teacher (Sinkhorn) on smaller subset for supervision
3. Fine-tune on real PF snapshots

### Smoothness and Regularization

Smooth activations (WTanh, WSoftmax) guarantee T(x) is analytic, but NOT that:
- J(x) is well-conditioned (λ_min bounded away from 0)
- J(x) varies slowly on the particle scale

**Regularization strategy (prioritized):**

1. Residual architecture `T(x,c) = x + S(x,c)` — J = I + J_S, safe for logdet
2. Bandwidth continuation in h — smooth source/target → smooth optimal T
3. Local-linearity penalty (directly targets discrete particle transport validity):
   ```
   L_lin = E[||T(x+δ) - T(x) - J(x)δ||²],  δ ~ K_h
   ```
4. Eigenvalue barrier on J — penalize small λ_min(J)
5. Spectral norm on W, weight decay — mild scale control

Prefer WTanh over WSoftmax for activation — WSoftmax can saturate → near-singular J.

### Gradient Supervision

For pretraining and validation:
- **Gaussian-to-Gaussian**: analytic T and ∇T known. Best teacher.
- **Sinkhorn teacher**: gives T(x_i) at particle locations. Local affine fits over kNN give noisy J estimates. Teaches entropic map, not Brenier map.
- **MA constraint**: `log det J = log q - log p(T(x))` constrains determinant only, not full J. Useful but not sufficient for covariance transport.

Full Jacobian supervision is hard to get. The local-linearity penalty (above) is a practical substitute — it doesn't require target J values, just penalizes non-linearity.

### What We Are Building

A **neural operator** that solves Monge-Ampere:
- **Input**: (particles, weights) — any configuration from any model
- **Output**: transport map T(·; c) satisfying q_h(x) = p_h(T(x)) det ∇T(x)
- **Architecture**: conditional mGradNet-M (convex by construction)
- **One network, trained once, applied to any resampling problem at inference**

NOT a per-instance solver like GradNetOT. NOT amortized Sinkhorn prediction.

### Validated Design Parameters

**Flexible N:** DeepSets pools across particles, trunk is pointwise. Variable N
supported architecturally. Include `log N` and `ESS = 1/Σw²` in context vector so
the operator knows about the effective sample size. Train across a range of N.

**KDE:** Gaussian kernel. Compute bandwidth using weighted covariance and
`N_eff = 1/Σw²`, not raw N. Prefer covariance-scaled bandwidth over isotropic scalar.
Anneal h downward during training (start larger, end smaller).

**Model size:**

| d | M (modules) | e (embed width) | Trunk params |
|---|-------------|-----------------|-------------|
| 1 | 8 | 32-64 | ~500-1K |
| 2 | 16 | 64 | ~4K |
| 3-4 | 16-32 | 64 | ~4K-8K |

**DeepSets encoder:** hidden [64, 64], context dim 32-64. ~5K-10K params.

Total model size: ~10K-20K parameters. Very small, trains fast.

**Conditioning rule (implementation constraint):**

The context `c` from the set encoder can modulate:
- Per-module positive gates `a_m(c)` (via softplus on a linear output)
- Per-module positive slopes `τ_m(c)` (via softplus)
- Per-module biases/shifts `b_m(c)`, `U_m c`
- A global output shift `d(c)`

The context MUST NOT modify the `W_m` weight matrices. These are fixed learned
parameters shared across all contexts. This keeps the architecture's built-in
convexity guarantee intact, which is needed for the Monge-Ampere transport map
to be valid. See Brenier's theorem: the OT map must be the gradient of a convex
potential.

**Training budget (3 hours on RTX 3090):**

- 15K-30K optimizer steps
- B = 16-32 particle clouds per step
- Q = 128 query/collocation points per cloud
- Total queries per step: B × Q ≈ 2K-4K

**Evaluation metrics:**

- MA residual on held-out particle clouds
- KL divergence between target p_h and KDE of transported particles
- Moment error and MMD on transported particles (KL is bandwidth-sensitive)
- J conditioning: min eigenvalue, failure rate of `log det J`
- Unit test: analytic J matches `tape.batch_jacobian` on small random cases
- End-to-end: autodiff vs numerical gradient of filter likelihood
- Downstream: filter RMSE, ESS with neural operator resampling

**Training data for d=1 LG prototype:**

1. Start: analytic Gaussian-to-Gaussian teacher (known monotone map)
2. Then: synthetic weighted particle clouds with varied ESS, spread, multimodality
3. Finally: real PF snapshots from LG runs for fine-tuning

### Implementation Order

1. Port mGradNet-M architecture to TF (`tf.keras.layers.Layer` with tied weights, analytic Jacobian)
2. Implement TF-native KDE utilities (weighted bandwidth, q_h/p_h evaluation, sampling from q_h)
3. Implement weighted DeepSets branch encoder (tokens include log N, ESS)
4. Implement conditional mGradNet-M trunk (context modulates gates/slopes/shifts, NOT W_m)
5. Implement `forward_and_jacobian(x, c)` with analytic J
6. Unit test: verify analytic J matches `tape.batch_jacobian` on random inputs
7. Train on 1D Gaussian-to-Gaussian with analytic teacher
8. Train on 1D synthetic weighted particle clouds with MA loss + local-linearity penalty
9. Add bandwidth annealing and eigenvalue barrier
10. Evaluate: MA residual, KL, moment error, J conditioning
11. Implement map-based resampler in `code/src/resampling/` returning mapped particles + Jacobians
12. Implement Jacobian-based covariance transport: `covs_new_i = J_i @ covs_i @ J_i^T`
13. End-to-end gradient test through compiled filter
14. Benchmark against implicit Sinkhorn
15. Scale to d=2 (SV2D, RB) with real PF snapshots
