# HMC in Differentiable Particle Filters: Key Trade-offs

## Numerical Instability in HMC-DPF: Root Cause and Manifestations

### Root cause: HMC explores dangerous parameter regions

The fundamental problem is a mismatch between HMC's exploration strategy and the particle filter's numerical requirements.

HMC operates in **unconstrained space** $\phi \in \mathbb{R}^d$. A bijector (e.g., Softplus) maps to constrained space:

$$\theta = \text{Softplus}(\phi) = \log(1 + e^\phi)$$

During leapfrog integration, HMC proposes extreme values of $\phi$. When $\phi \ll 0$ (e.g., $\phi = -20$), we get:

$$\theta = \text{Softplus}(-20) \approx e^{-20} \approx 2 \times 10^{-9}$$

These near-zero parameter values propagate into matrices that the filter assumes are well-conditioned, triggering a cascade of numerical failures.

### The numerical pipeline and where it breaks

The LEDH particle flow filter computes, for each of $N$ particles at each of $T$ timesteps:

**1. EKF predict/update** — produces predicted covariance $P_k$ and innovation covariance $S_k$:

$$S_k = H_k P_k H_k^\top + R$$

where $R = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots)$ is the observation noise covariance and $H_k$ is the observation Jacobian.

**2. Flow parameter computation** — solves a linear system via Cholesky:

$$L_S L_S^\top = S, \quad S^{-1} H = \text{cholesky\_solve}(L_S, H)$$

$$A = -\frac{1}{2} P H^\top S^{-1} H$$

**3. Particle transport** — 29 Euler steps with Jacobian accumulation:

$$\eta^{(j+1)} = \eta^{(j)} + \Delta\lambda \cdot (A \eta^{(j)} + b)$$

$$M^{(j)} = I + \Delta\lambda \cdot A$$

$$\log\theta = \sum_{j=1}^{29} \log|\det(M^{(j)})|$$

**4. Weight computation** — uses accumulated Jacobian $\theta = e^{\log\theta}$ to correct particle weights.

When $\theta \to 0$ for most particles: **weight collapse** (one particle dominates, ESS $\to 1$).

### Three manifestations of the same root cause

All three failures stem from extreme HMC proposals producing near-singular matrices:

#### Manifestation 1: `MatrixInverse` crash in forward pass

```
InvalidArgumentError: Input is not invertible. [Op:MatrixInverse]
```

**Where:** `R_inv = tf.linalg.inv(R)` in `log_marginal_likelihood_tf`.

**Math:** When $\sigma \to 0$, $R = \text{diag}(\sigma^2) \to 0$, so $R^{-1}$ does not exist.

$$\text{cond}(R) = \frac{\sigma_{\max}^2}{\sigma_{\min}^2} \to \infty$$

**Eager vs graph mode:** In eager mode, `tf.linalg.inv` returns NaN for singular inputs. In graph mode on GPU (inside TFP's leapfrog `tf.while_loop`), the `MatrixInverse` kernel raises `InvalidArgumentError`.

**Fix:** `graph_safe_inv` uses `tf.linalg.pinv` (SVD-based pseudoinverse), which never raises.

#### Manifestation 2: `MatrixInverse` crash in backward pass

```
InvalidArgumentError: Input is not invertible.
  [Op:MatrixInverse] in MatrixDeterminant_13_grad
```

**Where:** The gradient of $\log|\det(M)|$ inside the flow loop.

**Math:** TensorFlow's auto-generated gradient of $\log|\det(M)|$ is:

$$\frac{\partial}{\partial M} \log|\det(M)| = M^{-\top}$$

TF computes $M^{-\top}$ via the `MatrixInverse` op. When any of the $N \times 29$ Jacobian matrices $M^{(j)} = I + \Delta\lambda \cdot A$ is singular, the backward pass crashes — even if the forward pass (`slogdet`) succeeded.

**Why this only affects HMC:** Regular filtering never computes gradients. HMC requires `GradientTape` to differentiate through the entire filter for the leapfrog energy gradient $\nabla_\phi U(\phi)$.

**Fix:** `@tf.custom_gradient` replaces TF's auto-generated backward pass:

- `graph_safe_log_abs_det`: uses `pinv` in backward (robust, ~2x slower)
- `graph_safe_log_abs_det_fast`: uses NaN-guarded `inv` in backward (near-original speed):
  - Detect NaN matrices: replace with $I$ before calling `inv`
  - Add backward jitter: $M^{-\top} \approx (M + 10^{-6}I)^{-\top}$
  - Zero out gradient for NaN matrices (HMC will reject these proposals anyway)
- `graph_safe_log_abs_det_svd`: full SVD, $\log|\det(M)| = \sum_i \log \sigma_i$ (most robust, ~4x slower)

#### Manifestation 3: Cholesky decomposition failure (GPU warning)

```
W cholesky_op_gpu.cu.cc:205] Cholesky decomposition was not successful for batch 0.
The input might not be valid. Filling lower-triangular output with NaNs.
```

**Where:** `safe_cholesky(S)` inside `batched_ekf_update` and `compute_flow_params_batch`.

**Math:** The innovation covariance is:

$$S = H P H^\top + R$$

For $S$ to be positive definite, we need $R \succ 0$ (positive definite). When $\sigma \to 0$, $R \to 0$, and $S$ can become non-PD due to:

- $R$ itself being near-zero (dominant cause)
- Accumulated numerical error in $P$ from prior flow steps
- NaN propagation from an earlier failed operation in the same HMC proposal

Cholesky requires $S \succ 0$. When $S$ is not PD, `tf.linalg.cholesky` on GPU fills the output with NaN (does not crash). The NaN propagates through the rest of the computation, producing $\log p \to$ NaN, which causes HMC to reject the proposal.

**Current status:** Not a crash, but generates noisy warnings. The NaN propagation → rejection mechanism is correct behavior — bad proposals are rejected as they should be. Suppress with `TF_CPP_MIN_LOG_LEVEL=2`.

### Summary: failure chain

```
HMC proposes extreme φ
  → Softplus(φ) ≈ 0                    (near-zero σ)
  → R = diag(σ²) ≈ 0                   (singular noise covariance)
  ├→ inv(R) crashes                     [Manifestation 1 — FIXED: graph_safe_inv]
  ├→ S = HPH' + R ≈ HPH'               (S loses positive-definiteness)
  │  └→ cholesky(S) fails               [Manifestation 3 — warning, not crash]
  │     └→ NaN in flow params A, b
  │        └→ M = I + ΔλA contains NaN
  │           ├→ det(M) forward: OK (slogdet handles NaN)
  │           └→ det(M) backward: inv(M) crashes
  │              [Manifestation 2 — FIXED: graph_safe_log_abs_det_fast]
  └→ All NaN paths → log p = NaN → HMC rejects proposal ✓
```

### Weight collapse (related but distinct)

Weight collapse is not a crash but a statistical failure. After the 29-step flow transport:

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot \theta^{(i)} \cdot \frac{p(y_k | \eta_1^{(i)}) \, p(\eta_1^{(i)} | x_{k-1}^{(i)})}{q(\eta_1^{(i)} | \eta_0^{(i)})}$$

If the accumulated Jacobian $\theta^{(i)} = \exp(\sum_j \log|\det(M^{(j)})|)$ varies wildly across particles (due to numerical errors amplified over 29 steps), one particle dominates:

$$\text{ESS} = \frac{1}{\sum_i (w^{(i)})^2} \to 1$$

This can happen even without NaN — just from float32 precision loss in the 29-step Jacobian accumulation (observed on CUDA but not MPS, likely due to different float32 rounding behavior).

---

## The Singular R Crash

During HMC inference on `range_bearing_ledh_hmc_soft`, the filter crashes with:

```
InvalidArgumentError: Input is not invertible. [Op:MatrixInverse]
```

at `ledh_invertible_hmc.py:304` where `R_inv = safe_inv(R)`.

**Root cause:** HMC's leapfrog integrator proposes extreme unconstrained parameter values. Softplus maps these to near-zero sigmas, making `R = diag([sigma_range^2, sigma_bearing^2])` singular.

This crash occurs with soft resampling (`stop_gradient_resampling: false`) but not with systematic resampling (`stop_gradient_resampling: true`). The reason is the differentiability-bias trade-off (see below).

### Why HMC crashes but regular filtering doesn't

The same `safe_inv` and `safe_log_abs_det` functions work fine during standalone filtering but crash during HMC inference. Three reasons:

1. **Eager vs graph mode.** Regular filtering runs in eager mode (Python for-loop). In eager mode, `tf.linalg.inv` and `tf.linalg.det` return NaN/Inf for singular inputs — no crash, just bad values that propagate silently. HMC runs inside TFP's leapfrog `tf.while_loop`, which traces the body into graph mode. In graph mode on GPU, the `MatrixInverse` op raises `InvalidArgumentError` instead of returning NaN. Same op, different execution mode, different failure behavior.

2. **Fixed vs explored parameters.** During filtering, parameters are fixed (user-specified) — R is always well-conditioned. During HMC, the sampler explores unconstrained space freely. Leapfrog can propose extreme values that map through Softplus to near-zero sigmas, making R singular. The filter never sees bad parameters; HMC does by design.

3. **No gradients vs gradients needed.** Regular filtering doesn't compute gradients — the backward pass never runs. HMC needs gradients via GradientTape. Even when the forward pass of `det(M)` succeeds, the backward pass inserts a `MatrixInverse` node for the gradient computation (`d/dM log|det(M)| = M^{-T}`). If any of the N_particles × 29 Jacobian matrices is singular, that backward `MatrixInverse` crashes.

### Fix: graph-safe variants in `linalg.py`

Added separate graph-safe functions (used only in `ledh_invertible_hmc.py`):

- `graph_safe_inv`: uses `tf.linalg.pinv` (SVD-based pseudoinverse) instead of `tf.linalg.inv`. Never crashes.
- `graph_safe_log_abs_det`: uses `slogdet` (fast LU) for the forward pass + `@tf.custom_gradient` with `pinv` for the backward pass. Avoids `MatrixInverse` in both directions.
- `graph_safe_log_abs_det_svd`: full SVD approach (`log|det| = sum(log(singular_values))`). Most robust but ~4x slower. Kept as fallback.

The original `safe_inv` and `safe_log_abs_det` are unchanged — other filters that run in eager mode continue using them.

---

## Differentiability-Bias Trade-off

### "Not differentiable" does not mean "crashes"

A common claim is that systematic resampling is "not differentiable" and therefore cannot be used in differentiable particle filters. This is misleading. The code runs fine — what fails is the **gradient quality**, not the computation.

Systematic resampling consists of four steps:

1. Compute cumulative weights: $C_i = \sum_{j=1}^{i} w_j$
2. Generate uniform points: $u_k = \frac{k - 1 + U}{N}$, where $U \sim \text{Uniform}(0,1)$
3. Find indices via `searchsorted`: $I_k = \min\{i : C_i \geq u_k\}$
4. Gather particles: $x'_k = x_{I_k}$

Step 3 is the problem. It maps continuous weights to **integer indices** — a piecewise-constant function. Tiny changes in weights do not change which indices are selected, so:

$$\frac{\partial I_k}{\partial w_j} = 0 \quad \text{almost everywhere}$$

TensorFlow can compute a gradient through this — it's just **zero**. The code executes, no crash, no error. But the gradient carries no information about how resampling depends on weights.

Step 4 (`tf.gather`) does have a well-defined gradient: it scatters gradients back to the selected particles. So the gradient flows through the *values* of the particles that were selected, but not through the *selection process itself*.

### What `stop_gradient` actually does

With `stop_gradient_resampling: true`, the code explicitly wraps the resampled particles in `tf.stop_gradient()`. This cuts **all** gradient through resampling — both the (zero) gradient through index selection and the (nonzero) gradient through particle values. The effect is almost identical to the natural zero-gradient behavior, but more explicit and controlled.

### What "differentiable resampling" provides — and its limits

Soft resampling and OT Sinkhorn both aim to provide gradient flow through resampling, but they achieve this to **different degrees**.

**Soft resampling is only partially differentiable.** It blends weights with a uniform distribution:

$$\tilde{w}_i = \alpha \, w_i + (1 - \alpha) / N$$

The blending is differentiable ($\partial \tilde{w}_i / \partial w_i = \alpha$), but the particle *selection* still uses discrete index operations:

```python
# soft.py — the non-differentiable core
cumsum = tf.cumsum(q_weights)
u_vals = u + tf.range(N) / N                    # systematic sampling points
indices = tf.searchsorted(cumsum, u_vals)        # discrete: ∂indices/∂weights = 0
resampled_particles = tf.gather(particles, indices)  # routes gradient to selected particles only
```

`tf.searchsorted` maps continuous weights to integer indices — a piecewise-constant function with zero gradient almost everywhere. `tf.gather` passes gradient through particle *values* at fixed indices, but cannot capture how changing weights would change *which* particles are selected.

The "soft" part is the importance weight correction `w_new = w / q`, which is differentiable. So the autodiff gradient captures:
- How observation log-probs change with parameters (through `log_observation_prob_batch`) ✓
- How importance weights `w/q` change with parameters ✓
- How particle *selection* changes with parameters ✗ (discrete, zero gradient)

This makes autodiff through soft resampling a **biased** gradient estimate. Finite difference captures the full effect (including discrete index changes), so FD and autodiff disagree — and this gets **worse with more particles** because the cumulative-sum bins narrow (width ~1/N), making `searchsorted` indices more sensitive to tiny parameter perturbations.

**OT Sinkhorn is fully differentiable.** It replaces discrete index selection with a continuous transport matrix:

$$x'_i = \sum_j T_{ij} \, x_j$$

where $T$ is an $N \times N$ doubly-stochastic matrix computed via Sinkhorn iterations (all differentiable ops). Every input particle contributes to every output particle, weighted by $T_{ij}$. There is **no discrete index selection** — no `searchsorted`, no `gather`. The gradient flows smoothly through the matrix multiplication.

This makes OT the only resampling method suitable for validating autodiff gradients against finite difference.

Both approaches give nonzero gradients through resampling, but at different costs: soft resampling is cheap but partially differentiable (biased gradient), while OT Sinkhorn is fully differentiable but computationally expensive (Sinkhorn iterations + $O(N^2)$ transport matrix).

### Summary table

| Resampling Strategy | Fully Differentiable? | Gradient Bias | Gradient Variance | Notes |
|---|---|---|---|---|
| Systematic + stop_gradient | No | High (ignores resampling dependency) | Low | Smooth energy surface |
| Systematic without stop_gradient | No | High (zero gradient through indices) | Low | Nearly same as stop_gradient |
| Soft resampling (PF-net) | **Partial** (weight correction yes, index selection no) | Medium | High | `searchsorted`+`gather` is discrete; FD/autodiff mismatch grows with N |
| OT Sinkhorn | **Yes** (continuous transport matrix, no discrete indices) | Low | Medium | Expensive per step; only method suitable for FD vs autodiff validation |

### Why this matters for HMC

HMC is **not** SGD. SGD handles noisy gradients naturally by averaging over many update steps. HMC's leapfrog integrator simulates Hamiltonian dynamics, which assumes the gradient is a **smooth, deterministic force field**.

- **stop_gradient**: biased-but-smooth gradient gives a valid (if slightly wrong) Hamiltonian to simulate. Leapfrog trajectories stay well-behaved.
- **Soft resampling with gradient flow**: the gradient through resampling of N particles is high-variance. Leapfrog trajectories accumulate noise at each step and can diverge into extreme parameter regions.

### Practical implications

- Soft resampling is theoretically better (less biased gradient).
- But the right choice depends on **what consumes the gradient**:
  - **SGD/Adam** (point estimation): handles noisy gradients fine. Soft resampling is preferred.
  - **HMC/NUTS** (posterior sampling): needs smooth gradients. stop_gradient is more stable.
  - **More particles**: reduces gradient variance, making soft resampling more viable for HMC.
- Systematic resampling with stop_gradient is a perfectly valid choice for HMC — the biased gradient still points roughly in the right direction, which is sufficient for HMC to find the posterior (just slightly wrong about its shape).

---

## `num_leapfrog_steps` Tuning

Controls how far HMC travels along the Hamiltonian trajectory before proposing. Each step requires a full filter forward pass + gradient computation.

### Effect of trajectory length

| Range | Behavior | Acceptance Rate | Exploration |
|---|---|---|---|
| **1-3** (too few) | Near random-walk; proposals close to current state | High | Slow — paying gradient cost for little movement |
| **5-20** (sweet spot) | Proposals far enough to explore efficiently; Hamiltonian approximately conserved | Good | Efficient — this is where HMC beats random-walk |
| **50+** (too many) | Trajectory U-turns back near start; wasted computation | Drops | Poor — many gradient evals for no net movement |

### Interaction with noisy gradients

With soft resampling (`stop_gradient_resampling: false`), each leapfrog step accumulates gradient error. More steps = more accumulated error = higher chance of trajectory divergence.

Mitigations:
- Reduce `num_leapfrog_steps` to 3-5 (limits error accumulation, but slower mixing)
- Reduce `step_size` (more conservative steps)
- Use NUTS (`sampler: nuts`) which auto-detects U-turns and adapts trajectory length per sample
- Increase `n_particles` (reduces gradient variance at the source)

### Current default

`num_leapfrog_steps: 5` (changed from 10). For 1-2 parameter models, 3-5 is appropriate. Rule of thumb: scale roughly with $\sqrt{d}$ where $d$ is parameter dimension.

---

## Leapfrog Divergence: The Real Reason HMC Gets Stuck

### Observed symptom

Running with TFP's HMC sampler, the chain accepts only the first proposal then rejects everything:

```
[burn-in 1/3] accept=100% | step_size=0.1618 | obs_noise_std=2.0081
[burn-in 2/3] accept=50%  | step_size=0.0660 | obs_noise_std=2.0081  ← stuck
[burn-in 3/3] accept=33%  | step_size=0.0660 | obs_noise_std=2.0081
[sample 1/2]  accept=25%  | step_size=0.0660 | obs_noise_std=2.0081
[sample 2/2]  accept=20%  | step_size=0.0660 | obs_noise_std=2.0081
```

The acceptance rates (100%, 50%, 33%, 25%, 20%) are **cumulative** — only 1 out of 5 proposals was actually accepted. Parameters are completely frozen.

### What the diagnostic tf.print revealed

No NaN, no -inf. All log-likelihoods are finite. The problem is the leapfrog trajectory **diverging** due to extreme curvature in the posterior:

```
Leapfrog step 1: q=[1.577]   → ll=-173    (obs_noise_std ≈ 2.0, reasonable)
Leapfrog step 2: q=[0.229]   → ll=-107    (obs_noise_std ≈ 1.3, better!)
Leapfrog step 3: q=[-2.712]  → ll=+137    (obs_noise_std ≈ 0.064, cliff edge)
Leapfrog step 4: q=[-7.830]  → ll=-481    (obs_noise_std ≈ 0.0004, past the cliff)
Leapfrog step 5: q=[128.218] → ll=-577    (obs_noise_std ≈ 128, wildly diverged)
```

The trajectory starts reasonable, passes through a **likelihood cliff** near small noise values (where the Gaussian observation density becomes infinitely peaked), and the enormous gradient there launches the particle to q=128. The final proposal has `nlp ≈ 595` vs current `nlp ≈ 181`, giving `accept_prob ≈ exp(-414) ≈ 0`.

### Why the cliff exists

The observation log-likelihood for Gaussian noise is:

$$\log p(y \mid x, \sigma) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y - h(x))^2}{2\sigma^2}$$

As $\sigma \to 0$:
- The $-\log\sigma$ term → $+\infty$ (likelihood increases)
- The $(y - h(x))^2 / \sigma^2$ term → $+\infty$ (likelihood crashes)

There is a narrow ridge where these balance, creating a **sharp cliff** in the log-likelihood surface. The gradient at the cliff edge is enormous — far too large for the leapfrog step size to handle.

### Why step size adaptation fails

The dual-averaging adaptation has too few steps to converge:

```
num_adapt = int(0.8 × num_burnin) = int(0.8 × 3) = 2 steps
```

After 2 adaptation steps, the step size is frozen. The sequence is:
1. Step 1 uses initial `step_size=0.01` → accepted → adaptation increases to 0.16
2. Step 2 uses 0.16 → diverges → adaptation drops to 0.066
3. Adaptation ends. Step size 0.066 is still too large but cannot be reduced further.

With proper burn-in (100+ steps), dual averaging would eventually find a small enough step size. But 3 burn-in steps is insufficient.

### Step size printed vs step size used

The printed `step_size` is the **adapted value after each step**, not the step size that was used:

| Step | Step size **used** | Accepted? | Step size **printed** |
|---|---|---|---|
| burn-in 1 | 0.01 (config) | Yes | 0.1618 (adapted up) |
| burn-in 2 | 0.1618 | No | 0.0660 (adapted down) |
| burn-in 3+ | 0.0660 | No | 0.0660 (frozen) |

---

## Gradient Clipping vs NUTS: Two Approaches to Leapfrog Divergence

### The problem both solve

Standard HMC's leapfrog integrator uses the **full gradient** for momentum updates:

$$p \leftarrow p + \epsilon \cdot \nabla \log p(\theta \mid y)$$

When $|\nabla \log p|$ is enormous (at the likelihood cliff), the momentum becomes huge, and the next position update $q \leftarrow q + \epsilon \cdot p$ overshoots into a terrible region. This is a **leapfrog divergence**.

### Approach 1: Gradient clipping (custom_hmc)

Caps the gradient magnitude before each momentum update:

$$\tilde{g} = g \cdot \min\left(1, \frac{C}{|g|}\right)$$

where $C$ is the clip norm (default 100.0). The direction is preserved; only the magnitude is limited.

**How it helps:** At the cliff where $|g| = 10000$, clipping reduces it to 100. The leapfrog takes a modest step in the correct direction instead of rocketing past the cliff.

**Properties:**
- The trajectory no longer follows true Hamiltonian dynamics (modified potential)
- But the Metropolis accept/reject still uses the **true** log-posterior for the energy comparison
- So detailed balance is preserved — no bias in the stationary distribution
- The proposal quality degrades (clipped trajectory doesn't conserve energy as well), lowering acceptance rate
- Simple to implement, no additional compute cost

**When to use:** When you want a simple, robust sampler and can tolerate lower acceptance rates. Good for exploratory runs.

**Limitation:** The clip norm is a hyperparameter. Too aggressive (small $C$) makes proposals near-random-walk. Too loose (large $C$) doesn't prevent divergence.

### Approach 2: NUTS (No U-Turn Sampler)

Instead of fixing the number of leapfrog steps, NUTS **adaptively chooses** the trajectory length by detecting when the trajectory starts doubling back (the "U-turn" criterion).

**How it works:**
1. Start a leapfrog trajectory
2. At each step, check: is the trajectory still moving away from the start? (dot product of momentum with displacement)
3. If the trajectory reverses direction → stop (U-turn detected)
4. Uses a binary tree to efficiently build the trajectory while maintaining detailed balance

**How it handles the cliff:** When the leapfrog hits q=-2.7 (the cliff), the enormous gradient reverses the momentum. NUTS detects this as a U-turn and **truncates the trajectory** before the particle rockets to q=128. The proposal is somewhere near the cliff edge — much more reasonable than the diverged trajectory.

**Properties:**
- Theoretically clean — no modified dynamics, no hyperparameters for trajectory length
- Adapts trajectory length per sample: short near high curvature, long in flat regions
- Reports "divergent transitions" when the integrator still can't handle the curvature even with truncation (diagnostic signal)
- More gradient evaluations per sample than fixed HMC (builds a binary tree), but fewer wasted evaluations from diverged trajectories

**When to use:** Default choice for production runs. Used by Stan, PyMC, NumPyro as the standard sampler.

**Limitation:** Each sample may require varying numbers of gradient evaluations (up to $2^{\text{max\_tree\_depth}}$), making wall-clock time per sample unpredictable. Controlled by `max_tree_depth` (default 10, meaning up to 1024 gradient evals per sample in extreme cases).

### Why TFP provides NUTS but not gradient clipping

TFP implements textbook algorithms. Gradient clipping is a practical hack — effective but not part of the HMC theory. NUTS is the principled solution: instead of modifying the dynamics to prevent divergence, it detects divergence and truncates. The HMC community (Stan, etc.) standardized on NUTS for this reason.

For particle filter likelihoods (our case), the posterior has unusually sharp curvature that challenges even NUTS. Both approaches are worth trying:
- `sampler: custom_hmc` with `grad_clip_norm: 100.0` — simple, robust
- `sampler: nuts` with `max_tree_depth: 10` — principled, adaptive

### Summary

| | Gradient Clipping | NUTS |
|---|---|---|
| Handles cliff | Caps gradient, prevents overshoot | Detects U-turn, truncates trajectory |
| Trajectory length | Fixed (`num_leapfrog_steps`) | Adaptive (up to $2^{\text{max\_tree\_depth}}$) |
| Hyperparameters | `grad_clip_norm` | `max_tree_depth` |
| Theory | Practical hack (preserves detailed balance) | Principled (part of HMC theory) |
| Compute | Predictable per sample | Variable per sample |
| Best for | Noisy/sharp likelihoods, exploratory runs | General use, production runs |

---

## How HMC Produces a Posterior Distribution

### The Bayesian setup

HMC is a Bayesian inference method. The goal is to characterize the **posterior distribution** over parameters given observed data:

$$p(\theta \mid y_{1:T}) = \frac{p(y_{1:T} \mid \theta) \, p(\theta)}{p(y_{1:T})}$$

The three ingredients:

- **Prior** $p(\theta)$: encodes beliefs before seeing data. In our configs this is LogNormal, e.g., $\sigma_{\text{range}} \sim \text{LogNormal}(\mu = -2.3, \, s = 0.5)$, centered near the true value 0.1.
- **Likelihood** $p(y_{1:T} \mid \theta)$: how probable the observed data is under parameters $\theta$. This is the particle filter's output — run the LEDH filter with parameters $\theta$, get the log marginal likelihood $\sum_t \log p(y_t \mid y_{1:t-1}, \theta)$.
- **Evidence** $p(y_{1:T}) = \int p(y_{1:T} \mid \theta) \, p(\theta) \, d\theta$: the normalizing constant. This integral is intractable for nonlinear models — and this is precisely why we need MCMC.

### What HMC assumes (and what it doesn't)

HMC makes **no assumption about the shape of the posterior**. It does not assume the posterior is Gaussian, unimodal, or any parametric form. It only requires the ability to evaluate, at any point $\theta$:

1. The unnormalized log posterior: $\log p(\theta \mid y) \propto \log p(y \mid \theta) + \log p(\theta)$
2. Its gradient: $\nabla_\theta \left[ \log p(y \mid \theta) + \log p(\theta) \right]$

The normalizing constant $p(y_{1:T})$ cancels out — HMC never needs it.

This is fundamentally different from other approaches:

| Method | What it assumes about the posterior |
|---|---|
| **MLE / MAP** | Nothing — finds a single point (mode), no distribution |
| **Laplace approximation** | Posterior $\approx$ Gaussian centered at the MAP |
| **Variational inference** | Posterior $\in$ some parametric family (e.g., mean-field Gaussian) |
| **HMC / NUTS** | **Nothing** — discovers the shape by sampling |

### How samples become a distribution

HMC generates a sequence of parameter samples $\theta_1, \theta_2, \ldots, \theta_N$ by simulating Hamiltonian dynamics on the "energy landscape" $U(\theta) = -\log p(\theta \mid y)$. Each sample is one point in parameter space where the posterior has mass.

The collection of samples forms an **empirical approximation** of the posterior:

$$p(\theta \mid y) \approx \frac{1}{N} \sum_{i=1}^{N} \delta(\theta - \theta_i)$$

From these samples, any posterior quantity can be estimated:

- **Posterior mean**: $\hat{\theta} = \frac{1}{N} \sum_i \theta_i$
- **Posterior standard deviation**: $\hat{\sigma} = \sqrt{\frac{1}{N-1} \sum_i (\theta_i - \hat{\theta})^2}$
- **95% credible interval**: sort $\{\theta_i\}$, take the 2.5th and 97.5th percentiles
- **Posterior probability of a hypothesis**: fraction of samples satisfying the condition

The more samples, the better the approximation. With enough samples, HMC converges to the **exact posterior** regardless of its shape — this is the theoretical guarantee of MCMC.

### Why hundreds of samples are needed

Each HMC sample is one "snapshot" of where the posterior has mass. The burn-in phase serves two purposes:

1. **Find the high-probability region**: the chain starts at the initial guess (e.g., $\sigma = 0.3$) and needs to migrate toward the true posterior mode.
2. **Adapt the step size**: the dual-averaging adapter learns a step size that achieves the target acceptance rate (~70%).

After burn-in, the sampling phase collects draws that represent the posterior. For a 2D parameter space ($\sigma_{\text{range}}, \sigma_{\text{bearing}}$), 200–500 post-burn-in samples typically suffice for reliable mean and credible interval estimates.

---

## Reading the Diagnostic Output

The `tf.print` in `_negative_log_posterior` produces:

```
[nlp] ll= -194.77  prior= -5.73  nlp= 200.50  q= [4.99 1.69]
```

| Field | What it is | How to interpret |
|---|---|---|
| `q` | Unconstrained params $\phi$ (HMC operates here) | Convert via softplus: $\theta = \log(1 + e^\phi)$. E.g., $\phi = [4.99, 1.69] \to \theta \approx [5.0, 2.0]$ |
| `ll` | Log-likelihood $\log p(y \mid \theta)$ from the particle filter | More negative = worse fit. Typical: $-150$ to $-250$. Positive values ($+137$) = cliff edge. `nan` = filter blew up |
| `prior` | Log prior + Jacobian: $\log p(\theta) + \log\|d\theta/d\phi\|$ | Penalizes extreme params. $-5$ is mild; $-30+$ means far from prior center |
| `nlp` | Negative log posterior: $-(\text{ll} + \text{prior})$ | **Lower = better.** The "energy" HMC minimizes. $\text{nlp} = 200$ means posterior $\propto e^{-200}$ |

### What to look for in a leapfrog trajectory

Each HMC step evaluates `nlp` multiple times (once per leapfrog step). A healthy trajectory:

```
q=[1.85] → nlp=185.7    (start)
q=[1.87] → nlp=186.4    (small change — good)
q=[1.89] → nlp=187.0    (still close — energy conserved)
q=[1.90] → nlp=187.3    (proposal)
```

Energy difference $\Delta H \approx 1.6$ → accept prob $\approx e^{-1.6} \approx 0.20$. Reasonable.

A diverged trajectory:

```
q=[1.58]    → nlp=175     (start)
q=[0.23]    → nlp=108     (better — moving toward mode)
q=[-2.71]   → nlp=-133    (CLIFF — ll went positive)
q=[-7.83]   → nlp=512     (past cliff — terrible)
q=[128.22]  → nlp=595     (launched into space)
```

Energy difference $\Delta H \approx 420$ → accept prob $\approx e^{-420} \approx 0$. Rejected.

### Red flags

- `ll` jumps by $> 100$ between consecutive leapfrog steps → curvature too sharp for step size
- `ll` becomes positive → noise parameter near zero, on the cliff edge
- `q` changes by $> 10$ in one leapfrog step → momentum is huge, trajectory diverging
- `prior = -inf` → parameter hit a boundary (e.g., $\phi \to -\infty$, $\theta \to 0$)
- `ll = nan` or `nlp = nan` → numerical failure in the filter (Cholesky, singular matrix)

---

## SGD-based DPF vs HMC-based DPF

### Two ways to use differentiable particle filter gradients

The DPF literature primarily uses **SGD-based optimization** (point estimation), not HMC (posterior sampling). The distinction matters because the two approaches have opposite gradient requirements.

### SGD-based DPF (point estimation)

**Goal:** Find the single best $\theta$ (MLE or MAP).

**Method:**
```python
q = tf.Variable(initial_unconstrained_params)
optimizer = Adam(learning_rate=0.01)

for step in range(500):
    seed = random_seed()                    # different seed each step
    with tf.GradientTape() as tape:
        loss = -log_marginal_likelihood(observations, q, seed)
    grad = tape.gradient(loss, [q])
    optimizer.apply_gradients(zip(grad, [q]))
```

Each step uses a **different random seed**, making the gradient stochastic. SGD doesn't care — it averages over hundreds of steps and converges to the optimum.

**This is what PF-net does** (Jonschkowski & Brock 2018). Soft resampling was invented specifically for this: it gives nonzero gradient through resampling so Adam can learn which particles matter. Corenflos et al. (2021) use OT resampling for the same purpose.

**Key property of SGD:** needs gradients that are **correct on average** (unbiased). Tolerates high variance.

### HMC-based DPF (posterior sampling)

**Goal:** Sample from the full posterior $p(\theta \mid y)$ — not just the mode, but uncertainty.

**Method:**
```python
seed = fixed_seed(42)                       # MUST be fixed
def log_posterior(q):
    return log_marginal_likelihood(observations, q, seed) + log_prior(q)

samples = HMC(log_posterior, ...)           # Hamiltonian dynamics on fixed surface
```

The seed **must be fixed** — HMC simulates physics on a deterministic energy landscape. If the landscape changed randomly at each evaluation, the leapfrog integrator would see contradictory forces and diverge.

**Key property of HMC:** needs gradients that are **smooth at every point**. One cliff = trajectory explosion.

### Why LEDH hurts HMC but helps SGD

LEDH adds 29 flow steps, each with Jacobian $M = I + \Delta\lambda \cdot A(\theta)$. The mapping is:

$$\theta \to A(\theta) \to \prod_{j=1}^{29} \det(M_j) \to \text{weights} \to \log p(y|\theta)$$

This chain of 29 determinant products **amplifies** small changes in $\theta$ into large changes in the likelihood — creating sharp curvature (cliffs) in the likelihood surface.

For SGD: the gradient is more informative (lower bias through resampling) → faster convergence to the optimum. Sharp curvature doesn't matter because each step is small and averaged.

For HMC: the sharp curvature causes leapfrog divergence. BPF gives a smoother (biased but navigable) likelihood surface.

| | SGD-based DPF | HMC-based DPF |
|---|---|---|
| Goal | Point estimate (MLE/MAP) | Full posterior |
| Seed | Random (different each step) | Fixed (deterministic surface) |
| Gradient need | Unbiased (correct on average) | Smooth (no cliffs) |
| Handles noise | Yes (averages over steps) | No (one cliff = divergence) |
| Best filter | LEDH + soft/OT (richest gradient) | BPF + systematic (smoothest gradient) |

### Practical implication: two-stage approach

Given the experimental results (BPF+systematic: 100% acceptance, correct posterior, 1s/step; LEDH+any: stuck, diverging, 23s/step), a practical approach is:

1. **Stage 1 — Parameter inference:** Use BPF + HMC (or BPF + SGD for MAP) to estimate $\theta$
2. **Stage 2 — State estimation:** Use LEDH with the inferred $\theta$ for high-quality filtering

This leverages each method's strength: BPF for smooth gradients, LEDH for accurate state tracking.

### Feasibility of adding SGD to current codebase

The existing `DPFRunner` infrastructure supports SGD with minimal changes:

- `ParameterHandler`: manages unconstrained $\leftrightarrow$ constrained bijectors ✓
- `DifferentiableModel`: wraps model params with gradient chain ✓
- `_negative_log_posterior`: computes differentiable loss ✓
- `log_marginal_likelihood_tf`: compiled filter forward pass ✓

Only change needed: a `run_map()` method that replaces the HMC loop with an Adam loop. The model wrapper, bijectors, filter, and compiled graph are all reused. SGD can optionally randomize the particle filter seed each step (reduces overfitting to a single realization) or keep it fixed (simpler, works for MAP).

---

## Experimental Results: Filter × Resampling Comparison (Linear Gaussian, 5 HMC steps)

First systematic run across all filter-resampling combinations. `num_burnin=3, num_samples=2` — only testing for crashes and basic behavior, not convergence.

### Results table

| Config | Filter | Resamp | Accept | Converged? | Time/step | Notes |
|---|---|---|---|---|---|---|
| lg_ledh_sys | LEDH | systematic | 1/5 | No (stuck 2.04) | 23.5s | Leapfrog diverges at cliff |
| lg_ledh_soft | LEDH | soft | 1/5 | No (stuck 2.04) | 23.8s | Worst divergence: q→32M |
| lg_ledh_ot | LEDH | OT | 1/5 | No (stuck 2.06) | 28s | OT slower, still diverges |
| **lg_bpf_sys** | **BPF** | **systematic** | **5/5** | **Yes (0.96±0.20)** | **1.0s** | **Only working config** |
| lg_bpf_soft | BPF | soft | 3/5 | Partial (0.84) | 1.1s | Some movement |
| lg_bpf_ot | BPF | OT | 1/5 | No (stuck 2.07) | 6.5s | OT distorts likelihood |

### Key findings

1. **BPF is 20-30× faster than LEDH** per HMC step (1s vs 23-29s). The 29-step flow with Jacobian accumulation dominates compute.

2. **BPF + systematic is the only working combination.** 100% acceptance, correct posterior (true=1.0, est=0.96±0.20), true value inside 90% CI.

3. **Soft resampling causes the worst divergence.** With LEDH+soft, the leapfrog launched to q=32,225,134 — the noisy gradient through soft resampling amplifies curvature massively.

4. **OT resampling hurts BPF.** BPF+OT has nlp values in a narrow range (232-296) with little gradient signal, while BPF+systematic has a clear optimum around q≈0.2 (nlp≈184). The Sinkhorn transport plan flattens the likelihood surface.

5. **NaN appeared in cubic_sensor_ledh_sys** (2-param model). Trajectory diverged to q=[-59080, 420407], then `prior=-inf → nlp=inf → q=[nan, 840820]`. Rejected safely, but near the edge of a crash.

6. **Cubic sensor LEDH accepted moves toward σ_W→0** (the cliff). With more samples, the chain would crash or freeze at near-zero noise.

---

## Particle Gibbs + HMC: Two Orthogonal Tricks for LEDH

### Motivation

The experiments above demonstrate that direct HMC through the particle filter fails for all but the simplest cases. The core problems are:

1. **Gradient cliffs** from resampling discontinuities
2. **Weight collapse** from accumulated Jacobian products (LEDH's 15+ flow steps)
3. **Bimodality trapping** (Kitagawa's $y = x^2/20$ admits $\pm\sqrt{20y}$)

Particle Gibbs + CSMC solves these via **two independent tricks**, each targeting a different problem.

---

### Trick 1: Gibbs Decomposition (Efficiency)

#### The Idea

Instead of targeting the marginal posterior $p(\theta \mid y_{1:T})$ by integrating out $x$ (which requires differentiating through the PF), target the **joint** posterior $p(\theta, x_{0:T} \mid y_{1:T})$ via Gibbs sampling:

**$\theta$-step**: Sample $\theta \mid x_{0:T}, y_{1:T}$ using HMC
**$x$-step**: Sample $x_{0:T} \mid \theta, y_{1:T}$ using a particle filter (no gradients)

#### $\theta$-step: Closed-Form Log-Posterior

Given a fixed trajectory $x_{0:T}$, the conditional posterior of $\theta$ is:

$$\log p(\theta \mid x_{0:T}, y_{1:T}) = \underbrace{\sum_{t=1}^{T} \log p(x_t \mid x_{t-1}, \theta)}_{\text{transition terms}} + \underbrace{\sum_{t=1}^{T} \log p(y_t \mid x_t, \theta)}_{\text{observation terms}} + \log p(x_0) + \log p(\theta) + \text{const}$$

Each term is a Gaussian log-density. **No particle filter is involved.**

**Transition term** ($t = 1, \ldots, T$):
$$\log p(x_t \mid x_{t-1}, \theta) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|Q(\theta)| - \frac{1}{2}(x_t - f(x_{t-1}, t))^T Q(\theta)^{-1} (x_t - f(x_{t-1}, t))$$

**Observation term** ($t = 1, \ldots, T$):
$$\log p(y_t \mid x_t, \theta) = -\frac{m}{2}\log(2\pi) - \frac{1}{2}\log|R(\theta)| - \frac{1}{2}(y_t - h(x_t))^T R(\theta)^{-1} (y_t - h(x_t))$$

**For Kitagawa** ($d = m = 1$, $Q = \sigma_V^2$, $R = \sigma_W^2$, $f(x,t) = x/2 + 25x/(1+x^2) + 8\cos(1.2t)$, $h(x) = x^2/20$):

$$\log p(\sigma_V, \sigma_W \mid x_{0:T}, y_{1:T}) = -T\log\sigma_V - \frac{1}{2\sigma_V^2}\sum_{t=1}^{T}(x_t - f(x_{t-1}, t))^2 - T\log\sigma_W - \frac{1}{2\sigma_W^2}\sum_{t=1}^{T}(y_t - x_t^2/20)^2 + \log p(\sigma_V, \sigma_W)$$

**Key properties:**
- Smooth and differentiable — $\nabla_\theta$ is a closed-form sum of Gaussian score functions
- No resampling, no Jacobians, no weight collapse
- HMC can use large step sizes ($\epsilon \sim 10^{-1}$) and get 60-80% acceptance
- Cost: $O(T)$ scalar operations (vs $O(T \cdot N \cdot n_\lambda)$ for LEDH forward+backward)

#### $x$-step: Particle Filter (No Gradients)

Sample a new trajectory $x_{0:T} \sim p(x_{0:T} \mid \theta, y_{1:T})$ using **any** particle filter:
- Run the PF forward with fixed $\theta$
- Store full particle genealogy (particles + ancestor indices at each $t$)
- At $t = T$, sample one trajectory index $k \sim \text{Categorical}(w_T^{1:N})$
- Trace ancestry backward to reconstruct $x_{0:T}^{(k)}$

**No gradients are needed** — this is pure forward simulation. No GradientTape, no backprop through resampling, no Jacobian accumulation issues.

#### Why Gibbs Helps

| Aspect | Direct HMC through PF | Gibbs $\theta$-step |
|--------|----------------------|---------------------|
| Target surface | Ragged, discontinuous | Smooth Gaussian sum |
| Gradient | Backprop through PF (expensive, unstable) | Closed-form sum (cheap, exact) |
| Step size | $\epsilon \sim 10^{-3}$ (avoid divergence) | $\epsilon \sim 10^{-1}$ (smooth surface) |
| Acceptance rate | 0-10% | 60-80% |
| Weight collapse | 15+ Jacobian determinants | None |
| Cost per step | $O(T \cdot N \cdot n_\lambda)$ (LEDH forward+backward) | $O(T)$ (Gaussian sums) |

---

### Trick 2: Conditional SMC / Reference Pinning (Bimodality)

#### The Problem

In the $x$-step, a standard PF can sample trajectories, but for bimodal models (Kitagawa), it may collapse to one mode. If the PF always picks the same mode, the Gibbs sampler gets trapped.

#### CSMC Algorithm

CSMC pins one particle (the **reference**) to the previous iteration's trajectory $x^*_{0:T}$:

**Initialization** ($t = 0$):
- Sample $x_0^i \sim p(x_0)$ for $i = 1, \ldots, N-1$
- Set $x_0^N = x_0^*$ (reference pinned)
- Set $w_0^i = 1/N$ for all $i$

**For** $t = 1, \ldots, T$:

1. **Resample** (reference-protected):
   - Sample ancestors $a_i \sim \text{Categorical}(w_{t-1}^{1:N})$ for $i = 1, \ldots, N-1$
   - Set $a_N = N$ (or use ancestor sampling, see below)

2. **Propagate**:
   - Free particles: $x_t^i \sim p(x_t \mid x_{t-1}^{a_i}, \theta)$ for $i = 1, \ldots, N-1$
   - Reference: $x_t^N = x_t^*$ (deterministic, follows prescribed path)

3. **Weight**:
   - For all $i$: $w_t^i \propto p(y_t \mid x_t^i, \theta)$
   - Normalize: $w_t^i = w_t^i / \sum_j w_t^j$

**Output** ($t = T$):
- Sample $k \sim \text{Categorical}(w_T^{1:N})$
- Trace ancestry of particle $k$ → full trajectory $x_{0:T}^{(k)}$
- This becomes $x^*_{0:T}$ for the next Gibbs iteration

#### Why Reference Pinning Handles Bimodality

- N-1 free particles explore both modes of $p(x_t \mid y_t)$
- Reference anchors the sampler (guarantees ergodicity — the chain can always "stay put")
- At $t = T$, the sampler can choose ANY particle's trajectory, including one that took a completely different path from the reference
- Over Gibbs iterations, output trajectories can switch between modes

#### Ancestor Sampling (Lindsten et al. 2014)

Without ancestor sampling, CSMC suffers from **path degeneracy**: early in the trajectory, all free particles share a common ancestor, so the output trajectory almost always coincides with the reference. Mixing is slow.

**Ancestor sampling** resamples the reference particle's ancestor at each $t$. For each candidate $j \in \{1, \ldots, N\}$:

$$\tilde{w}_{t|T}^j \propto w_{t-1}^j \cdot p(x_t^* \mid x_{t-1}^j, \theta)$$

Sample $a_N \sim \text{Categorical}(\tilde{w}_{t|T}^{1:N})$.

The transition density for Gaussian models:
$$p(x_t^* \mid x_{t-1}^j, \theta) = \mathcal{N}(x_t^*;\; f(x_{t-1}^j, t),\; Q(\theta))$$

This allows the reference's history to "detach" and graft onto any particle's past, dramatically improving mixing. Cost: $N$ extra transition density evaluations per timestep.

---

### Full PGibbs + HMC Algorithm

**Initialize:**
- $\theta^{(0)} = \theta_{\text{init}}$
- Run bootstrap PF with $\theta^{(0)}$, sample one trajectory → $x_{0:T}^{(0)}$

**For iteration** $m = 1, 2, \ldots, M$:

1. **$\theta$-step** (HMC on smooth surface):
   - Target: $\log p(\theta \mid x_{0:T}^{(m-1)}, y_{1:T})$ (closed-form, see above)
   - Run $L$ leapfrog steps with step size $\epsilon$
   - Accept/reject via Metropolis criterion → $\theta^{(m)}$

2. **$x$-step** (CSMC):
   - Run CSMC with $\theta^{(m)}$ and reference $x_{0:T}^{(m-1)}$
   - Ancestor sampling for better mixing
   - Sample output trajectory → $x_{0:T}^{(m)}$

---

### CSMC with LEDH Flow

#### Motivation

The $x$-step can use **any** particle filter. Using LEDH instead of bootstrap PF gives:
- Better proposals for free particles (flow moves particles toward likelihood → higher ESS)
- More diverse trajectory candidates at $t = T$
- Especially beneficial for high-dimensional models where bootstrap PF struggles

#### Algorithm: Conditional LEDH

Apply LEDH flow only to the N-1 free particles. The reference stays pinned.

**For** $t = 1, \ldots, T$:

1. **Resample** (reference-protected):
   - Resample $i = 1, \ldots, N-1$ from $w_{t-1}^{1:N}$
   - $a_N = N$ (standard) or ancestor-sampled
   - Resample per-particle EKF covariances accordingly

2. **Predict** (EKF + transition):
   - Batched EKF predict for all $N$ particles → predictive covariances $P_t^i$
   - Free: $\eta_0^i \sim p(x_t \mid x_{t-1}^{a_i}, \theta)$ for $i = 1, \ldots, N-1$
   - Reference: $\eta_0^N = x_t^*$ (pinned, no stochastic sampling)
   - Deterministic means: $\bar{\eta}_0^i = f(x_{t-1}^{a_i}, t)$ for all $i$

3. **LEDH Flow** (free particles only, $i = 1, \ldots, N-1$):
   - Initialize: $\lambda = 0$, $\log\theta^i = 0$
   - For $j = 1, \ldots, n_\lambda$:
     - $d\lambda = \lambda_j$ (exponentially decaying)
     - $\lambda \leftarrow \lambda + d\lambda$
     - $(A^i, b^i) = \text{FlowParams}(\bar{\eta}^i, \lambda, y_t, P_t^i, R, R^{-1})$
     - $\bar{\eta}^i \leftarrow \bar{\eta}^i + d\lambda(A^i\bar{\eta}^i + b^i)$
     - $\eta_1^i \leftarrow \eta_1^i + d\lambda(A^i\eta_1^i + b^i)$
     - $\log\theta^i \leftarrow \log\theta^i + \log|\det(I + d\lambda \cdot A^i)|$
   - Reference: $\eta_1^N = x_t^*$ (unchanged, no flow)

4. **Weight** (heterogeneous proposals):

   **Free particles** ($i = 1, \ldots, N-1$):
   $$w_t^i \propto w_{t-1}^i \cdot p(y_t \mid \eta_1^i, \theta) \cdot \frac{p(\eta_1^i \mid x_{t-1}^{a_i}, \theta)}{p(\eta_0^i \mid x_{t-1}^{a_i}, \theta)} \cdot |\det J^i|$$

   where $J^i = \prod_{j=1}^{n_\lambda}(I + d\lambda_j A_j^i)$.

   The ratio $p(\eta_1^i \mid \cdot) / p(\eta_0^i \mid \cdot)$ corrects for the flow moving the particle:
   - Proposal: sample $\eta_0^i \sim p(x_t \mid x_{t-1}^{a_i})$, then transform $\eta_1^i = T(\eta_0^i)$
   - Effective proposal density: $q(\eta_1^i) = p(\eta_0^i \mid x_{t-1}^{a_i}) / |\det J^i|$ (change of variables)
   - Importance weight = target / proposal = $\frac{p(y_t | \eta_1^i) \cdot p(\eta_1^i | x_{t-1}^{a_i})}{p(\eta_0^i | x_{t-1}^{a_i}) / |\det J^i|}$

   **Reference particle** ($i = N$):
   $$w_t^N \propto w_{t-1}^N \cdot p(y_t \mid x_t^*, \theta)$$

   No flow → no Jacobian, no transition density ratio. The reference is weighted by the raw likelihood only.

   **All $N$ weights normalized together.**

5. **EKF update** for all particles' covariances.

#### Why Skip the Flow for the Reference?

CSMC's theoretical guarantee (ergodicity, correct target distribution) requires: **if we always select the reference's ancestors, we reproduce exactly $x^*_{0:T}$**.

If the flow moved the reference from $x_t^*$ to some $\tilde{x}_t^*$, this invariant breaks — the chain would no longer have $x^*_{0:T}$ as a fixed point. Ergodicity is lost.

By skipping the flow for the reference, we ensure:
- Reference path is exactly $x^*_{0:T}$ (deterministic)
- The Gibbs sampler can always "stay put" (choose the reference trajectory)
- Detailed balance is preserved

#### Why Heterogeneous Weights Are Valid

Different particles use different proposals:
- Free particles: $q_{\text{LEDH}}^i(\eta_1) = p(\eta_0 \mid x_{t-1}^{a_i}) / |\det J^i|$ (flow-adjusted transition)
- Reference: $q_{\text{ref}}(x_t^N) = \delta(x_t^N - x_t^*)$ (point mass)

Importance sampling allows heterogeneous proposals: each particle's weight must correctly account for **its own** proposal. The formulas above satisfy this. The point-mass proposal for the reference is handled by the CSMC conditional construction (Andrieu et al. 2010, Proposition 1).

#### Ancestor Sampling in LEDH CSMC

Same as bootstrap CSMC — ancestor sampling does NOT use the flow. At time $t$, for each candidate ancestor $j$:

$$\tilde{w}_{t|T}^j \propto w_{t-1}^j \cdot p(x_t^* \mid x_{t-1}^j, \theta) = w_{t-1}^j \cdot \mathcal{N}(x_t^*;\; f(x_{t-1}^j, t),\; Q(\theta))$$

Sample $a_N \sim \text{Categorical}(\tilde{w}_{t|T}^{1:N})$.

This is a direct model evaluation (not a flow computation), so it's cheap.

---

### Trajectory Storage and Output Sampling

#### Genealogy Tracking

CSMC stores the full particle genealogy to reconstruct trajectories:
- `particles_history[t]`: $(N, d)$ — particle values at time $t$
- `ancestors_history[t]`: $(N,)$ — ancestor indices at time $t$

Storage: $O(T \cdot N \cdot d)$ for particles, $O(T \cdot N)$ for ancestors.

#### Backward Trajectory Sampling

At $t = T$:
1. Sample final index: $k_T \sim \text{Categorical}(w_T^{1:N})$
2. Trace backward: $k_t = \text{ancestors\_history}[t+1][k_{t+1}]$ for $t = T-1, \ldots, 0$
3. Extract: $x_t^{\text{out}} = \text{particles\_history}[t][k_t]$

Result: full trajectory $x_{0:T}^{\text{out}}$ sampled from (approximate) smoothing distribution.

---

### Summary: What Each Trick Provides

| | Without Trick | With Trick |
|---|---|---|
| **Trick 1 (Gibbs)** | HMC differentiates through PF: gradient cliffs, weight collapse, 0-10% acceptance | $\theta$-step on smooth surface: 60-80% acceptance, no PF in gradient path |
| **Trick 2 (CSMC)** | Standard PF collapses to one mode for bimodal models | Reference particle preserves one mode; free particles explore others |

**Combined**: PGibbs + HMC with CSMC solves both efficiency (smooth $\theta$-step) and bimodality ($x$-step explores all modes).

### Implementation Phases

| Phase | $x$-step filter | When to use |
|-------|----------------|-------------|
| **Phase 1** | Bootstrap CSMC (`conditional_smc.py`) | Start here — simplest, validates framework |
| **Phase 2** | LEDH CSMC (`ledh_invertible_csmc.py`) | When bootstrap PF struggles (high-dim, strong nonlinearity) |

---

### HMC vs Metropolis-Hastings for the θ-step

#### Convergence guarantee

The Particle Gibbs convergence proof (Andrieu, Doucet & Holenstein 2010) requires that the θ-step is a **valid MCMC kernel** that leaves $p(\theta \mid x_{0:T}, y_{1:T})$ invariant. The theorem is stated for a generic kernel — any MCMC method that satisfies this invariance condition works. The proof relies on:

1. CSMC targets $p(x_{0:T} \mid \theta, y_{1:T})$ correctly (reference particle ensures invariance)
2. The θ-step kernel leaves $p(\theta \mid x_{0:T}, y_{1:T})$ invariant
3. Together, the two-step Gibbs kernel leaves the joint $p(\theta, x_{0:T} \mid y_{1:T})$ invariant → ergodicity

Both MH and HMC satisfy condition 2: MH via its accept/reject construction, HMC via leapfrog dynamics + Metropolis correction. So **convergence is guaranteed for both**. The original paper used MH as the concrete example, but the theorem applies to HMC equally.

The practical question is not correctness but efficiency: HMC's gradient-guided exploration only pays off for high-dimensional θ ($d \gtrsim 10$). For the low-dimensional models in this project (1-2 parameters), MH is simpler and sufficient.

#### Key difference

HMC uses **gradient information** to propose parameters that follow the posterior's curvature. MH uses a **blind random walk** — propose $\theta' = \theta + \epsilon$, accept/reject based on density ratio.

#### When HMC wins

HMC's advantage is in **high-dimensional parameter spaces** ($d \gtrsim 10$). In $d$ dimensions:

- **MH random walk**: To maintain reasonable acceptance (~23% optimal), the proposal std must scale as $\sigma \propto d^{-1/2}$. The chain moves $O(d^{-1/2})$ per step, needing $O(d)$ steps to traverse the posterior. Total cost to get one independent sample: $O(d^2)$ density evaluations.

- **HMC**: The leapfrog integrator follows the gradient, so proposals can travel $O(1)$ distance in parameter space regardless of $d$. With properly tuned trajectory length, HMC needs $O(d^{1/4})$ steps per independent sample — dramatically better scaling.

#### When MH is sufficient

For **low-dimensional** problems ($d \leq 5$), MH works fine:

- The random walk can explore a 2D or 3D space efficiently
- Each MH step is cheap: just one density evaluation (no gradient, no leapfrog)
- No risk of leapfrog divergence or step size tuning issues
- Simpler implementation, fewer hyperparameters

**For Kitagawa** ($d = 2$: $\sigma_V, \sigma_W$), MH is the pragmatic choice. The closed-form $\theta$-step posterior (smooth Gaussian sum) is easy for MH to navigate in 2D. HMC's gradient overhead (multiple leapfrog steps with `GradientTape`) makes each iteration more expensive for negligible mixing benefit.

#### Summary

| | MH | HMC |
|---|---|---|
| Cost per step | 1 density evaluation | $L$ gradient evaluations ($L$ = leapfrog steps) |
| Scaling with $d$ | $O(d^2)$ | $O(d^{5/4})$ |
| Hyperparameters | proposal_std | step_size, num_leapfrog, grad_clip_norm |
| Divergence risk | None | Leapfrog can diverge at curvature cliffs |
| Sweet spot | $d \leq 5$ | $d \geq 10$ |

**Recommendation**: Use MH for Kitagawa and other low-parameter models. Reserve HMC for models with 10+ parameters where MH mixing becomes prohibitively slow.

---

### PGibbs σ-Stalling: Variance Drift from Poor Initialization

#### Observed symptom

Running PGibbs + MH on Kitagawa with initial guesses σ_V=2, σ_W=2 (true: σ_V=3.162, σ_W=1.0), the chain drifts monotonically upward past burn-in:

```
[sample 71/470]  sigma_V=12.13, sigma_W=5.63  | theta_accept=72%
[sample 80/470]  sigma_V=12.26, sigma_W=5.55  | theta_accept=74%
[sample 90/470]  sigma_V=12.89, sigma_W=5.52  | theta_accept=72%
```

The 72% acceptance rate looks healthy, but the parameters are at 4× the true values and still climbing. This is not slow convergence — it is a stable wrong equilibrium.

#### Root cause: prior-sampled initial trajectory

When `init_filter_class` is not provided, `PGibbsRunner._initialize_trajectory()` samples the initial trajectory $x_{0:T}$ from the prior — purely from $p(x_0)$ and $p(x_t \mid x_{t-1}, \theta_\text{init})$ with no conditioning on observations $y_{1:T}$.

This creates a positive feedback loop:

1. **Prior trajectory has huge observation residuals.** Since $x_t$ is independent of $y_t$, the residuals $(y_t - x_t^2/20)$ are enormous.

2. **θ-step inflates σ_W.** The closed-form posterior for σ_W given fixed x is:
   $$\log p(\sigma_W \mid x, y) \ni -T \log \sigma_W - \frac{1}{2\sigma_W^2} \sum_t (y_t - x_t^2/20)^2$$
   When the sum of squared residuals is huge, larger σ_W reduces the Mahalanobis term faster than the log-determinant penalty grows. MH accepts the increase.

3. **CSMC produces diffuse trajectories.** With inflated σ_W, the observation weights $p(y_t \mid x_t) = \mathcal{N}(y_t; x_t^2/20, \sigma_W^2)$ become nearly flat — weights don't discriminate between particles. The CSMC trajectory barely improves over the reference.

4. **θ-step inflates σ_V.** The new trajectory (still bad) has large transition residuals. The same logic applies: larger σ_V absorbs the residuals. σ_V climbs.

5. **Self-reinforcing equilibrium.** With σ_V ≈ 13, the CSMC's bootstrap proposal $\mathcal{N}(f(x_{t-1}), 169)$ is so diffuse that particles go everywhere. With σ_W ≈ 5.5, the observation likelihood is nearly flat ($\text{var} = 30.25$). The CSMC is essentially drawing from the prior, which reinforces the large σ estimates.

This is a **stable fixed point** of the Gibbs sampler: the trajectory is consistent with large σ, and large σ is consistent with the trajectory. Ergodicity guarantees the chain will eventually escape, but the mixing time can be astronomically long.

#### Why HMC does not fix this

Both MH and HMC condition on the same fixed trajectory $x_{0:T}$ in the θ-step. HMC would find the mode of $p(\theta \mid x, y)$ faster (gradient-guided), but that mode is at large σ when $x$ has large residuals. The Gibbs decomposition is the bottleneck — no θ-sampler can overcome a bad trajectory.

#### Why PMMH does not have this problem

PMMH proposes $\theta^*$ and runs a **fresh particle filter** to estimate $p(y \mid \theta^*)$. The PF integrates out $x$: the likelihood estimate reflects how well $\theta^*$ explains the **observations**, not some fixed trajectory.

If σ_V = 13, the PF's bootstrap particles spread out wildly and most miss the observations → $\hat{p}(y \mid \theta^*) \approx 0$ → proposal rejected. PMMH has a **built-in correction**: the marginal likelihood naturally penalizes inflated variance.

The tradeoff:
- **PMMH**: robust to initialization (integrates out $x$), expensive per step (full PF each iteration)
- **PGibbs**: cheap per step (closed-form θ-step), fragile to initialization (conditions on $x$)

#### Fix: BPF-initialized trajectory

Initialize the trajectory by running a bootstrap PF with the initial parameter guesses, then sampling one trajectory from the particle genealogy. This gives a trajectory that is at least consistent with the observations under the initial parameters.

The initial parameters (σ_V=2, σ_W=2) don't need to be accurate — they just need to be reasonable enough that the BPF produces a trajectory where $x_t$ tracks $y_t$. The first CSMC sweep then improves this trajectory, and the feedback loop runs in the right direction (toward the truth, not away from it).

In `run_dpf_experiment.py`, pass `init_filter_class=ParticleFilterTF` and `init_filter_kwargs={'n_particles': N}` when constructing `PGibbsRunner`.

---

## Resampling Gradient Mechanisms: Why OT Requires Graph Mode

### The three resampling methods handle gradients differently

Each resampling method uses a fundamentally different mechanism for backward-pass gradients. This determines whether it can run in eager mode or requires graph mode.

### Systematic resampling — no gradient through resampling

Systematic resampling uses `tf.searchsorted` (piecewise-constant → zero gradient) and `tf.gather` (scatters gradient to selected particles only). The gradient through index selection is zero almost everywhere.

In practice, `stop_gradient_resampling=True` is always used with systematic, which explicitly cuts all gradient through resampling. TF's standard autodiff handles everything — **no custom gradient needed**.

**Works in eager mode.**

### Soft resampling — standard differentiable ops

Soft resampling (PF-net) blends weights with a uniform distribution:

$$\tilde{w}_i = \alpha \, w_i + (1 - \alpha) / N$$

Then resamples from $\tilde{w}$ and computes importance weights $w'_i = w_i / \tilde{w}_i$.

All operations — arithmetic, `tf.gather`, division, normalization — are standard TF ops with well-defined eager-mode gradients. TF's built-in `GradientTape` autodiff handles the full backward pass natively.

**Works in eager mode.**

### OT (Sinkhorn) resampling — `@tf.custom_gradient` with `tf.gradients()`

OT resampling solves an entropy-regularized optimal transport problem via Sinkhorn iteration to produce a transport matrix $T_{ij}$. The Sinkhorn loop involves non-differentiable internal operations (iterative convergence, log-domain stabilization), so the result is wrapped in `@tf.custom_gradient`:

```python
# ot_entropy.py:422-453
@tf.custom_gradient
def _compute_transport_matrix(particles, log_weights, ...):
    # Forward: Sinkhorn iteration → transport matrix T
    T = compute_transport_matrix_from_potentials(...)

    def gradient(dT):
        # Backward: differentiate T w.r.t. inputs
        dparticles, dlog_weights = tf.gradients(T, [particles, log_weights], dT)
        return dparticles, dlog_weights, None, None, None, None

    return T, gradient
```

The custom gradient function calls `tf.gradients()` — **a graph-mode-only API**. In eager mode, `tf.gradients()` raises:

```
RuntimeError: tf.gradients is not supported when eager execution is enabled.
Use tf.GradientTape instead.
```

This is a TensorFlow limitation: `tf.gradients()` builds symbolic gradient ops in a graph, while `tf.GradientTape` records operations eagerly. They are fundamentally different mechanisms. The `@tf.custom_gradient` decorator bridges them in graph mode (the custom gradient function is called during graph construction), but in eager mode TF tries to execute `tf.gradients()` immediately, which fails.

**Requires graph mode (`eager_mode=False`).**

### Summary

| Resampling | Gradient mechanism | `tf.gradients()`? | Eager mode? |
|---|---|---|---|
| Systematic | `stop_gradient` cuts all gradient | No | Yes |
| Soft | Standard TF ops (arithmetic, gather) | No | Yes |
| OT (Sinkhorn) | `@tf.custom_gradient` → `tf.gradients()` | **Yes** | **No — graph only** |

### Implication for testing

Tests that compute gradients through OT resampling must use `eager_mode=False`. The filter's internal loop runs inside `tf.while_loop` (graph context), where `tf.gradients()` works. The outer `GradientTape` (from TFP HMC or manual test code) traces through the compiled graph.

Tests for systematic and soft can use `eager_mode=True` for simpler debugging.

---

## Critical Bug: `@tf.function` Caching Breaks Eager-Mode Parameter Updates

### Discovery

Diagnostic tests comparing autodiff vs finite-difference (FD) gradients revealed a 16× mismatch for BPF + systematic:

| Component | Autodiff | FD | Ratio |
|---|---|---|---|
| Combined (prior + lik) | +9.998 | +0.597 | 16.7× |
| Prior only | ~+0.6 | ~+0.6 | ~1× |
| Likelihood only | ~+9.4 | **0.000** | ∞ |

The likelihood FD is **exactly zero** — changing `obs_noise_std` via `update_parameters` does not change the filter's output. The entire combined FD (+0.597) comes from the prior gradient alone.

### Root cause: `@tf.function` on model batch methods

In `linear_gaussian.py`, the observation log-probability method is decorated with `@tf.function`:

```python
@tf.function
def log_observation_prob_batch(self, observation, particles):
    L_R = tf.linalg.cholesky(self.R)  # self.R reads self.obs_noise_std
    ...
```

The `self.R` property dynamically computes `obs_noise_std² × D@Dᵀ`. This works correctly on the first call. The problem is **what happens on subsequent calls**.

### How `@tf.function` caching works with object attributes

When `@tf.function` traces a method:

1. TF executes the Python code once to build a computation graph
2. `self.R` is called during tracing — it reads `self.obs_noise_std` and computes `R`
3. The resulting tensor is **embedded in the cached graph**
4. On subsequent calls with the same input shapes, TF **reuses the cached graph** without re-executing the Python code
5. `self.R` is **never called again** — the stale R value persists

This is because `self.obs_noise_std` is a regular eager tensor (not a `tf.Variable`). TF's `@tf.function` does not track mutations to Python object attributes — it only checks function argument shapes/dtypes for cache invalidation.

### Why the compiled path works but eager doesn't

**Compiled path** (`eager_mode=False`, used for OT):

```python
@tf.function
def compiled_filter(observations, particles, weights, seed_start, param_values):
    for i, name in enumerate(param_names):
        setattr(model, name, param_values[i])  # symbolic tensor in graph
    ...
    log_obs = model.log_observation_prob_batch(y, particles)
```

Here `param_values` is an **explicit function argument**. The inner `@tf.function` call is inlined during the outer trace, so `self.obs_noise_std` is a symbolic graph tensor. Different `param_values` → different graph execution → correct R.

**Eager path** (`eager_mode=True`, used for systematic/soft):

```python
def _run_eager(self, observations, particles, weights, rng_key):
    for t in range(T):
        log_obs = self.model.log_observation_prob_batch(y, particles)  # @tf.function!
```

Here `log_observation_prob_batch` is called directly from Python. The first call traces it with the current `self.R`. All 50 subsequent calls (one per timestep) and **all future calls from different HMC iterations** reuse the cached graph with the stale R.

### Impact on HMC production code

In `DPFRunner._negative_log_posterior`, each HMC iteration:

1. `update_parameters({'obs_noise_std': softplus(q_new)})` — updates the model attribute
2. `filter_obj.log_marginal_likelihood_tf(observations, seed)` → `_run_eager` → calls `log_observation_prob_batch`

Step 2 uses the **cached** `@tf.function` from the first-ever call. The observation noise covariance R is frozen at its initial value. Consequences:

- **Likelihood forward pass is constant** — does not respond to parameter changes
- **Only the prior changes** with the parameter
- HMC effectively samples from the **prior only**, not the posterior
- LogNormal(0,1) prior mode ≈ 0.37 → explains drift to `obs_noise_std ≈ 0.33`

The autodiff gradient appears non-zero (+9.4) because the **first trace's graph** correctly captured the dependency on `obs_noise_std`. But subsequent forward passes don't honor this dependency — the gradient is a "phantom" that describes the first trace, not the current evaluation.

### Impact on finite-difference tests

The FD test helper `finite_difference_gradient` creates ONE runner and calls `_negative_log_posterior(q+eps)` then `_negative_log_posterior(q-eps)`. The first call traces `@tf.function`. The second call uses the cached graph. Both get the same R → same likelihood → FD of likelihood = 0.

The FD result of +0.597 is entirely the prior's finite difference:

```
FD_total = FD_prior + FD_likelihood = ~0.6 + 0.0 = ~0.6
```

### Fix

**For the eager path**: bypass `@tf.function` on model methods that depend on trainable parameters. Options:

1. **Remove `@tf.function`** from `log_observation_prob_batch` and other batch methods that read dynamic properties (R, Q). The eager path already runs a Python for-loop, so `@tf.function` per-call overhead is minimal.

2. **Call the underlying Python function** in `_run_eager`, bypassing the cache:
   ```python
   # Instead of: model.log_observation_prob_batch(y, particles)
   # Use the unwrapped function:
   log_obs = type(model).log_observation_prob_batch.python_function(model, y, particles)
   ```

3. **Pass R as an argument** to `log_observation_prob_batch` instead of reading from `self`. This makes R a traced input, not a captured constant.

**For test helpers**: use a **fresh model/filter instance** for each FD evaluation point. Different model instances have different `@tf.function` cache entries, so each traces with the correct R.

### Affected model classes

Any model with `@tf.function` on batch methods that read dynamic properties (`self.R`, `self.Q`) is affected. Currently:

- `LinearGaussianModel.log_observation_prob_batch` — reads `self.R` (depends on `obs_noise_std`)
- `LinearGaussianModel.state_transition_cov_batch` — reads `self.Q` (depends on `process_noise_std`)
- Other model classes need auditing for the same pattern

---

## `tf.cond` Resampling Discontinuity and `always_resample` Fix

### The problem: HMC step size collapses to zero

When running HMC with the compiled filter path (`@tf.function` + `tf.while_loop`), the dual-averaging step size adaptation drives the step size to zero. All proposals are rejected and the chain is frozen.

### Root cause: `tf.cond` branch switching

The compiled filter uses `tf.cond` for conditional resampling:

```python
particles, weights, ... = tf.cond(
    ess < resample_thresh,
    do_resample,
    no_resample
)
```

During HMC's leapfrog integration, each step perturbs the parameters slightly. This changes the particle weights, which changes the ESS. At some timesteps, the ESS crosses the resampling threshold — the `tf.cond` switches from `do_resample` to `no_resample` (or vice versa).

The two branches produce **different computation graphs** with different outputs. When the branch switches, the log-likelihood jumps discontinuously. This breaks the leapfrog integrator's energy conservation:

1. Leapfrog step $k$: ESS at timestep $t=17$ is above threshold → no resample
2. Leapfrog step $k+1$: parameters perturbed slightly → ESS at $t=17$ drops below threshold → resample
3. The log-likelihood changes by a large amount (different particles survive)
4. The energy $H = U + K$ is no longer conserved → proposal has $\Delta H \gg 1$ → rejected
5. Dual averaging sees low acceptance → shrinks step size
6. Smaller step size → still hits discontinuities (they exist at every scale) → step size → 0

This is a fundamental incompatibility: `tf.cond` creates a piecewise-smooth likelihood surface, but HMC's leapfrog assumes a smooth surface.

### Exception: OT resampling does not cause step size collapse

Empirically, OT (Sinkhorn) resampling does **not** cause the step size to collapse to zero, even with `tf.cond` conditional resampling. The reason is that OT resampling uses a continuous transport matrix $x'_i = \sum_j T_{ij} x_j$ rather than discrete index selection. When ESS is high and `tf.cond` switches from `do_resample` to `no_resample`, the `do_resample` branch produces particles nearly identical to the originals (the transport matrix $T \approx I/N$ when weights are uniform). The branch outputs are close, so the log-likelihood jump is small and the leapfrog energy is approximately conserved.

In contrast, systematic and soft resampling use discrete `searchsorted` + `gather`. When the branch switches, the discrete index selection can produce a very different particle set, causing a large log-likelihood jump.

### Fix: `always_resample` option

Both `BootstrapPFHMC` and `LEDHParticleFlowFilterHMC` now accept `always_resample: bool = False`:

```yaml
filter:
  _target_: src.filters.particle.bootstrap_pf_hmc.BootstrapPFHMC
  always_resample: true
  ...
```

When `always_resample=True`:
- **Compiled path**: a Python `if` at trace time selects a branch that resamples unconditionally (no `tf.cond` in the graph). The computation graph is uniform across all timesteps.
- **Eager path**: sets `resample_threshold = 1.0` so `ESS < N` is always true. The Python `if` always triggers resampling. (Python `if` doesn't create graph discontinuities in eager mode.)

When `always_resample=False` (default): the original `tf.cond` path is preserved unchanged.

### Why always-resampling is safe with soft resampling

Soft resampling blends weights with a uniform distribution:

$$\tilde{w}_i = \alpha \, w_i + (1 - \alpha) / N$$

When ESS is already high (weights $\approx 1/N$), the blended weights are nearly uniform, and resampling is a near-no-op: particles barely move, importance correction $w_i / \tilde{w}_i \approx 1$. The computational cost is the same, but the graph structure is uniform.

### Affected files

- `bootstrap_pf_hmc.py`: `__init__` accepts `always_resample`, compiled path has both branches
- `ledh_invertible_hmc.py`: same pattern, also handles covariance resampling in both branches
- Cubic sensor configs (`bpf_sys.yaml`, `bpf_soft.yaml`, `bpf_ot.yaml`): `always_resample: true`

### MAP initialization (future work)

A complementary approach: find the posterior mode via Adam optimization first, then use it as the HMC starting point. This avoids the burn-in phase exploring bad parameter regions where the likelihood surface is most pathological. The `DPFRunner.run_map()` method already exists for this purpose.
