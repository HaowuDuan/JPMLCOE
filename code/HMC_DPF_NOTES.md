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

### What "differentiable resampling" provides

Soft resampling and OT Sinkhorn replace the discrete index selection with a **continuous relaxation**. Instead of picking one particle per index, they compute a weighted mixture:

**Soft resampling** blends weights with a uniform distribution:

$$\tilde{w}_i = \alpha \, w_i + (1 - \alpha) / N$$

then resamples from $\tilde{w}$. The blending is differentiable, so $\partial \tilde{w}_i / \partial w_i = \alpha \neq 0$.

**OT Sinkhorn** solves an entropy-regularized optimal transport problem to produce a differentiable transport plan $T_{ij}$:

$$x'_i = \sum_j T_{ij} \, x_j$$

Both approaches give nonzero gradients through resampling, but at a cost: soft resampling introduces high gradient variance, and OT Sinkhorn is computationally expensive.

### Summary table

| Resampling Strategy | Gradient Bias | Gradient Variance | Notes |
|---|---|---|---|
| Systematic + stop_gradient | High (ignores resampling dependency) | Low | Smooth energy surface |
| Systematic without stop_gradient | High (zero gradient through indices) | Low | Nearly same as stop_gradient |
| Soft resampling (PF-net) | Low (gradient flows through) | High | Noisy energy surface |
| OT Sinkhorn | Low | Medium | Expensive per step |

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
