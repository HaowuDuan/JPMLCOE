# Improving LEDH Invertible on the Kitagawa Model

## 1. Diagnosis

The Kitagawa observation model $y_n = x_n^2/20 + w_n$ squares the state, destroying sign information. The true posterior $p(x_n | y_{1:n})$ is **bimodal** at every timestep: both $+x$ and $-x$ produce the same expected observation $x^2/20$.

Current results (T=100, seed=42):

| Filter              | N particles | RMSE  | MAE (x<0) | MAE (x>0) | Sign correct |
|---------------------|-------------|-------|-----------|-----------|-------------|
| EKF                 | —           | 33.06 | 30.9      | 4.4       | 55/100      |
| Bootstrap PF        | 1000        | 19.41 | 22.4      | 2.6       | 52/100      |
| EDH Flow            | 500         | 20.69 | 25.1      | 1.2       | 50/100      |
| EDH Invertible      | 500         | 20.67 | 25.1      | 1.0       | 52/100      |
| LEDH Flow           | 500         | 18.93 | 23.7      | 3.0       | 52/100      |

Key observations:
- **All filters track positive states well** (MAE 1–4) but fail on negative states (MAE 22–31).
- **Sign recovery is at chance level** (~50%) for every method — the observation provides zero sign information.
- Flow filters (EDH/LEDH) have perfect ESS=500 but are **locked into one mode** because the Daum-Huang flow is derived from a Gaussian (unimodal) posterior approximation.
- The observation Jacobian $\partial h/\partial x = x/10$ **vanishes at $x=0$**, making the flow "blind" near the origin.

## 2. Root Cause

The EDH flow equations (Li & Coates 2017) solve for the optimal transport from prior $\mathcal{N}(\bar\eta_0, P)$ to the Gaussian approximation of the posterior. When the true posterior is bimodal, this Gaussian approximation collapses both modes into one. The flow herds **all** particles toward a single mode.

The transition dynamics $f(x,t) = x/2 + 25x/(1+x^2) + 8\cos(1.2t)$ are sign-sensitive — they carry real information about which mode is correct. But the standard flow never uses this information to resolve the sign ambiguity.

## 3. Proposed Improvement: Post-Flow Sign-Flip Correction

### Idea

After the LEDH flow moves particles to $\eta_1$ but **before weight computation**, propose flipping each particle's sign using a Metropolis-like step. Since $p(y|x) = p(y|-x)$ exactly (observation symmetry), the acceptance ratio depends only on the **transition prior**:

$$\text{flip\_prob}_i = \frac{p(-\eta_{1,i} \mid x_{k-1,i})}{p(\eta_{1,i} \mid x_{k-1,i}) + p(-\eta_{1,i} \mid x_{k-1,i})}$$

where $p(\eta \mid x_{k-1}) = \mathcal{N}(\eta; f(x_{k-1}, t), Q)$.

### Why this works

1. **Exploits transition dynamics**: The Kitagawa transition $f(x,t)$ is odd-symmetric in $x$ (sign-preserving for large $|x|$). If $x_{t-1} < 0$, then $f(x_{t-1}, t) < 0$ typically, so $p(-\eta_1 | x_{t-1}) > p(\eta_1 | x_{t-1})$ when $\eta_1 > 0$. The flip probability will be high — correctly recovering the negative mode.

2. **Zero overhead on easy problems**: For models where the posterior is unimodal, `flip_prob ≈ 0` everywhere and no particles flip. The correction is a no-op.

3. **Weights self-correct**: Since `compute_flow_weights` evaluates $p(y|\eta_1) \cdot p(\eta_1|x_{k-1}) \cdot \theta / p(\eta_0|x_{k-1})$ for the *final* (possibly flipped) particle, the importance weights automatically adjust. No extra reweighting needed.

4. **Preserves LEDH properties**: The flow itself is unchanged. The Jacobian $\theta$ from the flow is still valid as the mapping from $\eta_0$ to the pre-flip $\eta_1$. The flip is a simple post-hoc proposal correction.

### Implementation

In `ledh_invertible.py`, method `update()`, after line 260 (`self.particles.assign(eta_1)`) and before `compute_flow_weights`:

```python
# --- Post-flow sign-flip for symmetric observation models ---
f_prev = self.model.state_transition_mean_batch(self.particles_prev.value())
Q = self.model.state_transition_cov_batch(self.particles_prev.value())
Q_inv = tf.linalg.inv(Q)

diff_plus = eta_1 - f_prev
diff_minus = -eta_1 - f_prev

# Log transition probabilities (up to normalizing constant)
log_p_plus = -0.5 * tf.reduce_sum(
    diff_plus * tf.linalg.matvec(Q_inv, diff_plus), axis=1
)
log_p_minus = -0.5 * tf.reduce_sum(
    diff_minus * tf.linalg.matvec(Q_inv, diff_minus), axis=1
)

# Flip probability via numerically stable sigmoid
flip_prob = tf.nn.sigmoid(log_p_minus - log_p_plus)

seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
self.seed_counter += 1
u = tf.random.stateless_uniform([self.n_particles], seed=seed, dtype=self.dtype)
flip_mask = tf.cast(u < flip_prob, self.dtype)

# Flip sign where mask = 1: x -> -x
eta_1 = eta_1 * (1.0 - 2.0 * flip_mask[:, tf.newaxis])
self.particles.assign(eta_1)
```

### Configuration

Add a boolean flag `sign_flip: bool = False` to the constructor. Enable it for models with observation symmetry:

```yaml
# configs/filter/ledh_invertible.yaml
filter:
  _target_: src.filters.particle.ledh_invertible.LEDHParticleFlowFilter
  n_particles: 500
  n_lambda_steps: 29
  resample_threshold: 0.5
  sign_flip: true   # Enable for Kitagawa
```

## 4. Complementary Ideas

### 4a. Second-Order Flow Correction (Hessian-Augmented)

The flow uses $H = \partial h/\partial x = x/10$, which vanishes at $x=0$. The Hessian $\partial^2 h/\partial x^2 = 1/10$ is constant and nonzero. Augmenting the linearization with a second-order term:

$$h(x) \approx h(\bar{x}) + H(x - \bar{x}) + \frac{1}{2}(x-\bar{x})^T \nabla^2 h \, (x-\bar{x})$$

would give the flow information even when particles cross zero. This is a deeper change to the flow equations (modifies `compute_flow_params_batch`) but would help any model with vanishing first-order observability.

### 4b. Inflated Proposal Variance

Multiply the process noise covariance $Q$ by a factor $\kappa > 1$ during the predict step:

$$\eta_0 \sim \mathcal{N}(f(x_{k-1}), \kappa Q)$$

This spreads particles wider before the flow, increasing the chance of maintaining coverage across both modes. The weight correction in LEDH invertible accounts for the mismatch between proposal and true $Q$. Start with $\kappa = 1.5$–$2.0$.

### 4c. Stratified Initialization

At $t=0$, initialize half the particles with positive values and half with negative values (stratified from $\mathcal{N}(0, \sigma_0^2)$). This doesn't help long-term (without sign-flip the flow still collapses to one mode) but improves early-timestep tracking.

## 5. Expected Impact

The sign-flip correction should:
- Improve sign recovery from ~50% (chance) to 70–85% (depending on SNR)
- Reduce RMSE on negative states by roughly half (MAE from ~25 to ~12)
- Bring overall RMSE to ~12–14 range
- Keep ESS high (flow maintains particle diversity, flips don't collapse weights)

The biggest gains will be at timesteps where the state has been consistently negative for several steps (strong transition prior toward negative mode).

## 6. Limitations

- The sign-flip is specific to **discrete observation symmetries** ($x \mapsto -x$). It doesn't generalize to arbitrary multimodal posteriors.
- For high-dimensional states, identifying the symmetry group is model-specific. But for Kitagawa (1D, known $x^2$ symmetry), it's clean.
- If $Q$ is very large relative to the mode separation, the transition prior becomes uninformative and the flip probability approaches 0.5 (random flipping). In that regime, more particles or a mixture approach would be needed.
