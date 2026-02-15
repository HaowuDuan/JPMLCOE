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
| Bootstrap PF        | 10000       | 19.27 | —         | —         | —           |
| LEDH Invertible     | 10000       | 19.50 | —         | —         | —           |

Key observations:
- **All filters track positive states well** (MAE 1–4) but fail on negative states (MAE 22–31).
- **Sign recovery is at chance level** (~50%) for every method — the observation provides zero sign information.
- **20x more particles (10,000) gives < 1% RMSE improvement** — the problem is structural, not sample size.
- Flow filters (EDH/LEDH) have perfect ESS=500 but are **locked into one mode** because the Daum-Huang flow is derived from a Gaussian (unimodal) posterior approximation.
- The observation Jacobian $\partial h/\partial x = x/10$ **vanishes at $x=0$**, making the flow "blind" near the origin.

## 2. Root Cause

The EDH flow equations (Li & Coates 2017) solve for the optimal transport from prior $\mathcal{N}(\bar\eta_0, P)$ to the Gaussian approximation of the posterior. When the true posterior is bimodal, this Gaussian approximation collapses both modes into one. The flow herds **all** particles toward a single mode.

There is **no way to distinguish the two modes from a single observation** $y_t$ alone. Reverting all particle signs $x \to -x$ leaves $p(y_t | x)$ unchanged. The only asymmetry is in the transition dynamics $f(x,t) = x/2 + 25x/(1+x^2) + 8\cos(1.2t)$, but this signal is too weak at a single timestep (the transition prior with $\sigma_V = 10$ is very diffuse).

## 3. Failed Approach: Post-Flow Sign-Flip

### What we tried

After the LEDH flow, propose flipping each particle's sign using a Metropolis-like step based on the transition probability ratio:

$$\text{flip\_prob}_i = \sigma\!\left(\log p(-\eta_{1,i} \mid x_{k-1,i}) - \log p(\eta_{1,i} \mid x_{k-1,i})\right)$$

### Why it failed (RMSE 20.12 → essentially unchanged)

**Chicken-and-egg problem**: The flip probability depends on `particles_prev` ($x_{k-1}$) being in the correct mode. But if the previous step's flow also collapsed to the wrong mode, then $x_{k-1}$ is wrong, $f(x_{k-1})$ points toward the wrong sign, and `flip_prob ≈ 0`. The filter stays locked in the wrong mode across consecutive timesteps.

With $\sigma_V = 10$, the transition prior $p(\eta | x_{k-1}) = \mathcal{N}(\eta; f(x_{k-1}), 100)$ is very flat. Even when $x_{k-1}$ is correct, the Mahalanobis distance difference is only $\sim 2\text{--}3$, giving flip probabilities of 0.85–0.95 for genuinely wrong particles — but those particles almost never exist because `particles_prev` is already homogeneously in one mode.

## 4. Proposed Improvement: Look-Ahead Sign Correction

### Key insight

A single observation $y_t$ cannot distinguish $+x$ from $-x$ (since $h(x) = x^2/20$ is symmetric). But the **next** observation $y_{t+1}$ can, indirectly.

If we propagate $+\eta_1$ and $-\eta_1$ one step forward through the transition:
- $x_{t+1}^+ = f(+\eta_1, t+1)$
- $x_{t+1}^- = f(-\eta_1, t+1)$

Then the predicted observations are:
- $\hat{y}^+ = (x_{t+1}^+)^2/20 = f(+\eta_1)^2/20$
- $\hat{y}^- = (x_{t+1}^-)^2/20 = f(-\eta_1)^2/20$

These are **different** because $f(x,t) \neq -f(-x,t)$. The transition has an odd component ($x/2 + 25x/(1+x^2)$) and an even component ($8\cos(1.2t)$). The even cosine term means $|f(+x)| \neq |f(-x)|$ in general, so $h(f(+x)) \neq h(f(-x))$.

Comparing $\hat{y}^+$ and $\hat{y}^-$ against the actual $y_{t+1}$ tells us which sign is more consistent with the future.

### Concrete example

At $t = 50$, $\cos(1.2 \times 51) \approx -0.95$:
- $f(+15) = 7.5 + 1.66 + 8(-0.95) = 1.56 \implies \hat{y}^+ = 1.56^2/20 = 0.12$
- $f(-15) = -7.5 - 1.66 + 8(-0.95) = -16.76 \implies \hat{y}^- = 16.76^2/20 = 14.05$

If $y_{t+1} \approx 14$, the negative mode is overwhelmingly favored. The signal is strong (difference of ~14 vs ~0.1) even though a single observation couldn't tell $+15$ from $-15$.

### Why this avoids the chicken-and-egg problem

The look-ahead does **not** depend on `particles_prev` being in the correct mode. It uses:
1. The **current** flowed particle $\eta_1$ (which has the right magnitude from the flow, just possibly wrong sign)
2. A **future** observation $y_{t+1}$ (which is independent data, not derived from previous filter state)
3. The **transition dynamics** $f(x, t+1)$ (which are sign-asymmetric)

Even if the filter has been locked in the wrong mode for 10 consecutive steps, the look-ahead can detect the sign from the next observation alone.

### Multi-step look-ahead (depth $K$)

The signal can be amplified by looking $K$ steps ahead:
1. Propagate deterministically: $x_{t+k}^{\pm} = f(f(\ldots f(\pm\eta_1)\ldots))$ for $k = 1, \ldots, K$
2. Score each trajectory: $\sum_{k=1}^{K} \log p(y_{t+k} \mid x_{t+k}^{\pm})$
3. Flip if the negative trajectory scores better

Multi-step look-ahead helps because nonlinear dynamics amplify the sign difference over time. The cost is $K$ extra transition evaluations per particle per timestep.

### Predictive scoring with process noise correction

The point predictions $x_{t+k}^{\pm} = f(\ldots)$ ignore process noise $\sigma_V = 10$. To avoid overconfident flip decisions, the log-score should account for predictive uncertainty:

$$\sigma_{\text{pred}}^2(x) = \left(\frac{\partial h}{\partial x}\bigg|_{f(x)}\right)^2 \sigma_V^2 + \sigma_W^2 = \frac{f(x)^2}{100} \cdot 100 + 1 = f(x)^2 + 1$$

The flip probability at each look-ahead step becomes:

$$\log r_k = -\frac{(y_{t+k} - \hat{y}_k^-)^2}{2\sigma_{\text{pred}}^2(x_k^-)} - \frac{1}{2}\log\sigma_{\text{pred}}^2(x_k^-) + \frac{(y_{t+k} - \hat{y}_k^+)^2}{2\sigma_{\text{pred}}^2(x_k^+)} + \frac{1}{2}\log\sigma_{\text{pred}}^2(x_k^+)$$

$$\text{flip\_prob} = \sigma\!\left(\sum_{k=1}^{K} \log r_k\right)$$

## 5. Implementation Plan for `ledh_invertible_bimodal.py`

### Constructor changes

Replace `flip_fraction` with:

```python
def __init__(self, ...,
             lookahead_steps: int = 1,      # K: how many future observations to use (0 = disabled)
             **filter_kwargs):
```

### Override `filter()`

The parent's `filter()` calls `self.update(obs_tf[t])` with only the current observation. The look-ahead needs access to future observations and the current timestep index. Override to pass these:

```python
def filter(self, observations, ...):
    self.initialize(initial_mean, initial_cov, random_seed)
    T = len(observations)
    obs_tf = tf.constant(observations, dtype=self.dtype)

    for t in range(T):
        t0 = time.perf_counter()

        # Set model time for Kitagawa's cos(1.2*t) term
        if hasattr(self.model, 't'):
            self.model.t = t + 1

        self.predict()

        # Pass future observations for look-ahead (up to K steps)
        y_future = obs_tf[t+1 : t+1+self.lookahead_steps] if t + 1 < T else None
        self.update(obs_tf[t], y_future=y_future, current_t=t+1)

        mean, cov = self._estimate_mean_cov()
        self.means.append(mean)
        self.covs.append(cov)
        if progress_callback is not None:
            progress_callback(t, T, time.perf_counter() - t0)

    # ... rest identical to parent
```

### Override `update()`

Add `y_future` and `current_t` parameters. After the flow loop and Jacobian computation, insert the look-ahead sign correction before weight computation:

```python
def update(self, y, y_future=None, current_t=None):
    # --- Flow loop (identical to parent) ---
    # ... produces eta_1, theta ...

    # --- Look-ahead sign correction ---
    if y_future is not None and tf.shape(y_future)[0] > 0:
        eta_1 = self._lookahead_sign_correction(eta_1, y_future, current_t)

    self.particles.assign(eta_1)

    # --- Weight computation (identical to parent, uses TRUE model) ---
    weights_new = compute_flow_weights(...)
    # ... rest identical to parent
```

### New method: `_lookahead_sign_correction()`

```python
def _lookahead_sign_correction(self, eta_1, y_future, current_t):
    """
    Score +eta_1 vs -eta_1 against future observations using
    deterministic multi-step prediction through the transition dynamics.
    """
    K = tf.minimum(tf.shape(y_future)[0], self.lookahead_steps)

    # Two candidate trajectories: positive and negative sign
    x_plus = eta_1                                    # (N, sd)
    x_minus = -eta_1                                  # (N, sd)

    log_score_plus = tf.zeros(self.n_particles, dtype=self.dtype)
    log_score_minus = tf.zeros(self.n_particles, dtype=self.dtype)

    saved_t = self.model.t

    for k in range(self.lookahead_steps):
        if k >= K:
            break

        # Propagate one step: x_{t+k+1} = f(x_{t+k}, t+k+1)
        self.model.t = current_t + k + 1
        x_plus = self.model.state_transition_mean_batch(x_plus)
        x_minus = self.model.state_transition_mean_batch(x_minus)

        # Predicted observations: h(x_{t+k+1})
        y_pred_plus = self.model.observation_function_batch(x_plus)   # (N, od)
        y_pred_minus = self.model.observation_function_batch(x_minus)  # (N, od)

        # Predictive variance: (dh/dx|_x)^2 * sigma_V^2 + sigma_W^2
        H_plus = self.model.observation_jacobian_batch(x_plus)    # (N, od, sd)
        H_minus = self.model.observation_jacobian_batch(x_minus)
        Q = self.model.process_noise_cov       # (sd, sd)
        R = self.model.observation_noise_cov   # (od, od)

        # For 1D: var_pred = H^2 * Q + R (scalars)
        # General: var_pred = H @ Q @ H^T + R per particle
        HQHt_plus = tf.einsum('nij,jk,nlk->nil', H_plus, Q, H_plus)  # (N, od, od)
        HQHt_minus = tf.einsum('nij,jk,nlk->nil', H_minus, Q, H_minus)
        var_pred_plus = HQHt_plus + tf.expand_dims(R, 0)   # (N, od, od)
        var_pred_minus = HQHt_minus + tf.expand_dims(R, 0)

        # Score: log N(y_future[k]; y_pred, var_pred)
        y_k = y_future[k]  # (od,)
        diff_plus = y_k - y_pred_plus[:, 0]    # (N,) for 1D obs
        diff_minus = y_k - y_pred_minus[:, 0]

        vp = var_pred_plus[:, 0, 0]    # (N,) diagonal for 1D
        vm = var_pred_minus[:, 0, 0]

        log_score_plus += -0.5 * (diff_plus**2 / vp + tf.math.log(vp))
        log_score_minus += -0.5 * (diff_minus**2 / vm + tf.math.log(vm))

    self.model.t = saved_t

    # Flip probability
    flip_prob = tf.nn.sigmoid(log_score_minus - log_score_plus)

    # Stochastic flip
    seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
    self.seed_counter += 1
    u = tf.random.stateless_uniform([self.n_particles], seed=seed, dtype=self.dtype)
    flip_mask = tf.cast(u < flip_prob, self.dtype)

    eta_1 = eta_1 * (1.0 - 2.0 * flip_mask[:, tf.newaxis])
    return eta_1
```

### Config

```yaml
# configs/filter/ledh_invertible_bimodal.yaml
_target_: src.filters.particle.ledh_invertible_bimodal.LEDHInvertibleBimodal

n_particles: 500
n_lambda_steps: 29
resample_threshold: 0.5
lookahead_steps: 2    # K=2 gives good signal-to-noise for Kitagawa
```

## 6. Expected Impact

With $K = 1$ look-ahead:
- The sign discrimination has a z-score of ~0.7 per step (predictive mean difference ~14 vs predictive std ~19).
- Expected sign recovery: ~75% (up from 50%).

With $K = 2$:
- Accumulated z-score ~1.0, sign recovery ~85%.
- Expected RMSE improvement: ~25–35% (from ~20 to ~13–15).

With $K = 3$:
- Accumulated z-score ~1.2, sign recovery ~89%.
- Diminishing returns beyond $K = 3$ due to prediction noise accumulation.

The last timestep ($t = T$) has no future observation, so no correction is applied there. For $t = T - 1$, only $K = 1$ is available, etc.

## 7. Side Note: `model.t` Bug

During investigation, discovered that `model.t` is not correctly managed during the regular `filter()` loop:
- After `generate_data()`, `model.t = T` (because `sample_state_transition` increments it).
- The `filter()` method never resets or updates `model.t`.
- All calls to `state_transition_batch` use the stale `t = T`, making the cosine term $8\cos(1.2T)$ constant across all timesteps.
- Only `log_marginal_likelihood_tf` correctly sets `model.t = t + 1` per step.
- The bimodal filter's `filter()` override should fix this by setting `model.t = t + 1` at each step (shown in Section 5).

## 8. Limitations

- **Introduces a fixed lag**: The estimate at time $t$ uses $y_{t+1}, \ldots, y_{t+K}$. For offline filtering (all observations available) this is fine. For online use, it adds $K$-step latency.
- **Cost**: $K$ extra transition + observation evaluations per particle per timestep. For Kitagawa (1D), this is negligible. For high-dimensional models, it could be significant.
- **Model-specific**: The look-ahead exploits the fact that $f(x,t) \neq -f(-x,t)$. For models where the transition IS sign-symmetric, the look-ahead provides no signal. However, most realistic transitions break this symmetry.
- **Process noise degrades multi-step predictions**: With $\sigma_V = 10$, deterministic $K$-step predictions become unreliable for $K > 3$. The predictive variance correction (Section 4) partially accounts for this.
