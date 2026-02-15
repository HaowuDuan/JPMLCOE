# Weight Collapse in LEDH + HMC: Diagnosis and Solutions

## 1. Problem Statement

When running HMC with the Kitagawa model and LEDH particle filter, **every time step** triggers:
```
WARNING: weight collapse detected, falling back to uniform weights
```
This means the importance weights become NaN/Inf at every step, and the filter falls back to uniform weights $w_i = 1/N$. This destroys the gradient signal HMC needs.

---

## 2. The LEDH Weight Formula

### 2.1 Importance Weights (Eq. from Li & Coates 2017)

After flowing particles from $\eta_0^{(i)}$ (predicted) to $\eta_1^{(i)}$ (flowed) through the Daum-Huang ODE, the importance weight for particle $i$ is:

$$
w_i \propto \frac{p(z_k \mid \eta_1^{(i)}) \cdot p(\eta_1^{(i)} \mid x_{k-1}^{(i)}) \cdot |\det J^{(i)}|}{p(\eta_0^{(i)} \mid x_{k-1}^{(i)})} \cdot w_{k-1}^{(i)}
$$

where:
- $p(z_k \mid \eta_1^{(i)})$ — observation likelihood at the flowed particle
- $p(\eta_1^{(i)} \mid x_{k-1}^{(i)})$ — transition density evaluated at the flowed position
- $p(\eta_0^{(i)} \mid x_{k-1}^{(i)})$ — transition density evaluated at the predicted (pre-flow) position
- $|\det J^{(i)}|$ — Jacobian determinant of the flow map $\eta_0 \mapsto \eta_1$
- $w_{k-1}^{(i)}$ — previous weight

**Code location:** [distributions.py:103-202](code/src/utils/distributions.py#L103-L202), function `compute_flow_weights`

### 2.2 In Log-Space

The code computes:

$$
\log w_i = \log p(\eta_1^{(i)} \mid x_{k-1}^{(i)}) + \log p(z_k \mid \eta_1^{(i)}) + \log |\det J^{(i)}| - \log p(\eta_0^{(i)} \mid x_{k-1}^{(i)}) + \log w_{k-1}^{(i)}
$$

**Code location:** [distributions.py:182-189](code/src/utils/distributions.py#L182-L189)

### 2.3 Transition Density Terms

Both $p(\eta_1 \mid x_{k-1})$ and $p(\eta_0 \mid x_{k-1})$ are Gaussian:

$$
\log p(\eta \mid x_{k-1}) = -\frac{1}{2}\left[ d \log(2\pi) + \log|Q| + (\eta - f(x_{k-1}))^T Q^{-1} (\eta - f(x_{k-1})) \right]
$$

where $Q = \sigma_V^2 I$ is the process noise covariance and $f(x, n) = \frac{x}{2} + \frac{25x}{1+x^2} + 8\cos(1.2n)$ is the Kitagawa transition.

**Code location:**
- Transition density: [distributions.py:159-180](code/src/utils/distributions.py#L159-L180)
- Kitagawa $f(x,n)$: [kitagawa.py:103-105](code/src/models/kitagawa.py#L103-L105)
- $Q = \sigma_V^2$: [kitagawa.py:186-189](code/src/models/kitagawa.py#L186-L189)

### 2.4 Observation Likelihood

$$
\log p(z_k \mid \eta_1^{(i)}) = -\frac{1}{2}\left[\log(2\pi\sigma_W^2) + \frac{(z_k - (\eta_1^{(i)})^2/20)^2}{\sigma_W^2}\right]
$$

**Code location:** [kitagawa.py:308-330](code/src/models/kitagawa.py#L308-L330), `log_observation_prob_batch`

### 2.5 Jacobian of the Flow

The LEDH flow integrates the ODE $d\eta/d\lambda = A(\lambda)\eta + b(\lambda)$ via Euler steps. Each step's Jacobian is $M_j = I + \Delta\lambda_j A_j$. The total Jacobian is:

$$
\log|\det J^{(i)}| = \sum_{j=1}^{L} \log|\det(I + \Delta\lambda_j A_j^{(i)})|
$$

**Code location:** [ledh_invertible.py:246-248](code/src/filters/particle/ledh_invertible.py#L246-L248)

---

## 3. Why Weights Collapse

### 3.1 The Ratio $p(\eta_1 \mid x_{k-1}) / p(\eta_0 \mid x_{k-1})$

The key problematic term is the **ratio of transition densities**:

$$
\log\frac{p(\eta_1 \mid x_{k-1})}{p(\eta_0 \mid x_{k-1})} = -\frac{1}{2\sigma_V^2}\left[(\eta_1 - f)^2 - (\eta_0 - f)^2\right]
$$

where $f = f(x_{k-1}, n)$.

When $\sigma_V$ is **too small** (init guess 5.0 vs true 10.0), the factor $1/(2\sigma_V^2) = 1/50$ instead of $1/200$. This amplifies the difference $(\eta_1 - f)^2 - (\eta_0 - f)^2$ by **4x**.

The flow moves $\eta_0 \to \eta_1$ toward the observation, so typically $|\eta_1 - f| > |\eta_0 - f|$ (the flow pushes particles away from the transition prior toward the likelihood). Under a too-tight prior ($\sigma_V$ too small), this movement is heavily penalized:

$$
\frac{1}{2\sigma_V^2}(\eta_1 - f)^2 \gg \frac{1}{2\sigma_V^2}(\eta_0 - f)^2
$$

This causes the log-weight ratio to become **large and negative** for some particles and **large and positive** for others, leading to extreme variance in log-weights.

### 3.2 The Observation Likelihood Blowup

Similarly, with $\sigma_W = 2.0$ (init) vs true $1.0$:

$$
\frac{(z_k - x^2/20)^2}{\sigma_W^2}
$$

The Kitagawa observation is $h(x) = x^2/20$, which is highly nonlinear. For particles far from the true state, the squared error $(z_k - x^2/20)^2$ can be huge. With wrong $\sigma_W$, the scale is mismatched: either too forgiving (missing the true peak) or too punishing (killing all particles).

### 3.3 The Normalization Catastrophe

After computing log-weights, normalization happens via:

$$
w_i = \frac{\exp(\log w_i - \max_j \log w_j)}{\sum_k \exp(\log w_k - \max_j \log w_j)}
$$

**Code location:** [distributions.py:56-77](code/src/utils/distributions.py#L56-L77), `normalize_log_weights`

When the spread of log-weights exceeds ~88 (the range of float32), even after subtracting the max, some terms underflow to exactly 0 while one term dominates. This gives $w_i \approx 0$ for all but one particle — or if the max log-weight is Inf/NaN, the entire normalization produces NaN.

**The fallback:** When weights contain NaN, the code replaces them with uniform $w_i = 1/N$.

**Code location:** [distributions.py:194-200](code/src/utils/distributions.py#L194-L200)

---

## 4. Why Uniform Weights Kill HMC Gradients

### 4.1 The Gradient Chain

HMC differentiates the log-posterior:

$$
\nabla_\theta \log p(\theta \mid y) = \nabla_\theta \log p(y \mid \theta) + \nabla_\theta \log p(\theta)
$$

The critical term is $\nabla_\theta \log p(y \mid \theta)$, which is the gradient of the log marginal likelihood through the filter. In the LEDH filter:

$$
\log p(y_{1:T} \mid \theta) = \sum_{t=1}^{T} \log \hat{p}(y_t \mid y_{1:t-1})
$$

where each per-step estimate depends on particle weights and positions:

$$
\hat{p}(y_t \mid y_{1:t-1}) = \sum_i w_i^{(t)} p(y_t \mid \eta_1^{(i,t)})
$$

**Code location:** [ledh_invertible.py:273-278](code/src/filters/particle/ledh_invertible.py#L273-L278) — but note this implementation uses a simplified log-likelihood estimate (max + log-mean-exp over raw likelihoods, not weighted).

### 4.2 How the Gradient Flows Through Weights

The gradient of the log-likelihood w.r.t. $\theta = (\sigma_V, \sigma_W)$ flows through:

1. **$\sigma_V \to Q = \sigma_V^2 I$** — affects the transition density and the flow ODE via $P$ (predicted covariance)
2. **$\sigma_W \to R = \sigma_W^2 I$** — affects the observation likelihood and the flow ODE via $R^{-1}$
3. **$Q, R \to$ flow params $A(\lambda), b(\lambda)$** — the flow ODE coefficients depend on both
4. **$A, b \to \eta_1$** — the flowed particle positions
5. **$\eta_1, Q, R \to w_i$** — the importance weights
6. **$w_i \to$ resampled particles** — through OT transport matrix $T$
7. **Resampled particles carry forward to next time step**

### 4.3 The Uniform Weight Disaster

When weights collapse and are replaced by $w_i = 1/N$:

$$
w_i = \frac{1}{N} \quad \Rightarrow \quad \frac{\partial w_i}{\partial \theta} = 0
$$

The uniform constant $1/N$ has **no dependence on $\theta$**. This severs the gradient chain at step (5) above. The gradient $\partial \log p(y|\theta) / \partial \theta$ becomes:

- **Zero through the weight channel**: weights don't depend on $\theta$
- **Partially alive through particle positions**: $\eta_1$ still depends on $\theta$ through the flow ODE, but the per-step log-likelihood uses the **raw** `log_observation_prob_batch` without weighting (see [ledh_invertible.py:274](code/src/filters/particle/ledh_invertible.py#L274)), so position gradients survive but are biased
- **Dead through resampling**: OT resampling uses $T(\text{particles}, \text{weights})$ — with constant weights, $\partial T / \partial \theta = 0$ through the weight channel

Effectively, HMC sees a **nearly flat** log-posterior landscape. The leapfrog integrator computes near-zero gradients, so momentum dominates and the chain random-walks instead of exploring the posterior. With near-zero gradients, HMC degenerates to random-walk Metropolis with very small steps.

### 4.4 The Compounding Effect

Weight collapse at time $t$ corrupts all subsequent time steps $t+1, \ldots, T$ because:
1. Uniform weights $\to$ OT transport does nothing meaningful $\to$ particles are not reweighted correctly
2. Subsequent predict/update cycles start from incorrectly distributed particles
3. This typically causes weight collapse at every subsequent time step too

This is exactly what we observe: the warning fires at every single time step (100 times per filter pass).

---

## 5. Solutions

### 5.1 Better Initial Guess (Quick Fix)

Set initial parameter guesses closer to truth:
```yaml
# In kitagawa_ledh.yaml
model:
  sigma_V: 8.0    # Closer to true 10.0 (was 5.0)
  sigma_W: 1.0    # Exact true value (was 2.0)
dpf:
  trainable_params:
    sigma_V:
      init_value: 8.0
    sigma_W:
      init_value: 1.0
```
**Pros:** Simple, may work.
**Cons:** Defeats the purpose — real inference shouldn't need good initial guesses.

### 5.2 More Particles

More particles reduce variance of the weight estimate:
```yaml
filter:
  n_particles: 500   # Was 200
```
**Pros:** Reduces weight degeneracy probabilistically.
**Cons:** 2.5x slower per HMC step. Still doesn't fix the fundamental mismatch.

### 5.3 Tempered/Annealed Likelihood (Recommended)

Instead of evaluating the full likelihood $p(y|\theta)$ at wrong parameters, use a **tempered likelihood** during early HMC burn-in:

$$
p_\beta(y \mid \theta) = p(y \mid \theta)^\beta, \quad \beta \in (0, 1]
$$

This flattens the likelihood surface, preventing weight collapse. Gradually increase $\beta \to 1$ during burn-in.

**Implementation approach:** In `_negative_log_posterior`, scale the log-likelihood:
```python
log_likelihood = beta * self.filter_obj.log_marginal_likelihood_tf(...)
```
Start with $\beta = 0.1$ and anneal to $\beta = 1.0$ over the burn-in period.

**Pros:** Principled, widely used in SMC and particle MCMC literature.
**Cons:** Requires annealing schedule tuning; early samples are from a different distribution.

### 5.4 Weight Clipping / Soft Clamping (Practical Fix)

Instead of falling back to uniform, **clip the log-weights** before normalization:

$$
\log \tilde{w}_i = \text{clip}(\log w_i, -C, C)
$$

with $C \sim 20\text{--}50$. This prevents any single weight from dominating while preserving the gradient direction.

**Implementation:** Pass `clip_range=(-30, 30)` to `normalize_log_weights` in `compute_flow_weights`:
```python
weights = normalize_log_weights(log_weights, clip_range=(-30.0, 30.0))
```

**Code location to modify:** [distributions.py:192](code/src/utils/distributions.py#L192)

**Pros:** Preserves gradient signal even when weights are extreme. Simple to implement.
**Cons:** Introduces bias — clipped weights don't integrate to the correct posterior.

### 5.5 Gradient-Friendly Log-Likelihood Estimate

The current per-step log-likelihood in `log_marginal_likelihood_tf` ([ledh_invertible.py:273-278](code/src/filters/particle/ledh_invertible.py#L273-L278)) computes:

$$
\log \hat{p}(y_t) = \max_i \ell_i + \log\left(\frac{1}{N}\sum_i \exp(\ell_i - \max_i \ell_i)\right)
$$

where $\ell_i = \log p(y_t \mid \eta_1^{(i)})$. This does **not** use the weights at all. A **weighted** version would be:

$$
\log \hat{p}(y_t) = \log\left(\sum_i w_i \exp(\ell_i)\right)
$$

But this requires non-degenerate weights, circling back to the collapse problem.

### 5.6 Particle MCMC (PMCMC) with Fixed Random Numbers

Use the same random seed for particle generation across all HMC evaluations (already done via `seed=tf.constant([42, 0])`). This ensures the log-likelihood is a **deterministic** function of $\theta$, which is critical for HMC to work with particle filters. (**Already implemented** at [hmc_runner.py:79](code/src/DF/hmc_runner.py#L79).)

### 5.7 Increase Flow Steps

More flow integration steps ($L$) improve the quality of the flow transport, reducing the $|\eta_1 - f|$ gap that causes weight divergence:
```yaml
filter:
  n_lambda_steps: 49   # Was 29
```
Better flow $\Rightarrow$ $\eta_1$ closer to optimal $\Rightarrow$ smaller transition density ratio $\Rightarrow$ more stable weights.

**Code location:** [ledh_invertible.py:51](code/src/filters/particle/ledh_invertible.py#L51)

### 5.8 Float64 Precision

Float32 has $\sim$7 decimal digits and overflows at $e^{88}$. Float64 has $\sim$15 digits and overflows at $e^{709}$. This dramatically extends the range of log-weights that can be handled without NaN:

```yaml
dtype: float64
```

**Pros:** 8x wider log-weight range before overflow.
**Cons:** 2x memory, potentially slower on GPU (though often the same speed on modern NVIDIA GPUs with FP64 support).

---

## 6. Recommended Action Plan

**Immediate (get it running):**
1. **Clip log-weights** with `clip_range=(-50, 50)` — preserves gradient, prevents NaN
2. **Switch to float64** — wider numerical range
3. **Better initial guess** — sigma_V=8.0 instead of 5.0

**Medium-term (proper solution):**
4. **Tempered likelihood** with annealing schedule during burn-in
5. **More particles** (500+) for reduced variance

**Long-term (research-quality):**
6. **Particle Gibbs** or **SMC^2** instead of naive HMC — proper particle MCMC methods that handle weight degeneracy theoretically
