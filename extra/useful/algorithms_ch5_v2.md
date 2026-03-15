# Algorithms from Särkkä, Chapter 5: Extended and Unscented Kalman Filtering

## Gaussian Approximations to Nonlinear Transforms

### Algorithm 5.1 — Linear approximation, additive transform

**Setup:** x ~ N(m, P), q ~ N(0, Q), y = g(x) + q

**Joint approximation:**

(x, y) ~ N( (m, μ_L), [[P, C_L], [C_L^T, S_L]] )

where:
- μ_L = g(m)
- S_L = G_x(m) P G_x(m)^T + Q
- C_L = P G_x(m)^T

G_x(m) is the Jacobian of g w.r.t. x evaluated at x = m:

[G_x(m)]_{j,j'} = ∂g_j(x)/∂x_{j'} |_{x=m}

---

### Algorithm 5.2 — Linear approximation, non-additive transform

**Setup:** x ~ N(m, P), q ~ N(0, Q), y = g(x, q)

**Joint approximation:**

(x, y) ~ N( (m, μ_L), [[P, C_L], [C_L^T, S_L]] )

where:
- μ_L = g(m, 0)
- S_L = G_x(m) P G_x(m)^T + G_q(m) Q G_q(m)^T
- C_L = P G_x(m)^T

Jacobians evaluated at x = m, q = 0:

[G_x(m)]_{j,j'} = ∂g_j(x,q)/∂x_{j'} |_{x=m, q=0}

[G_q(m)]_{j,j'} = ∂g_j(x,q)/∂q_{j'} |_{x=m, q=0}

---

### Algorithm 5.3 — Quadratic (second-order) approximation, additive transform

**Setup:** x ~ N(m, P), q ~ N(0, Q), y = g(x) + q

**Joint approximation:**

(x, y) ~ N( (m, μ_Q), [[P, C_Q], [C_Q^T, S_Q]] )

where:
- μ_Q = g(m) + (1/2) Σ_i  e_i tr( G^(i)_xx(m) P )
- S_Q = G_x(m) P G_x(m)^T + (1/2) Σ_{i,i'} e_i e_{i'}^T tr( G^(i)_xx(m) P G^(i')_xx(m) P ) + Q
- C_Q = P G_x(m)^T

G^(i)_xx(m) is the Hessian of the i-th component g_i evaluated at m:

[G^(i)_xx(m)]_{j,j'} = ∂²g_i(x)/(∂x_j ∂x_{j'}) |_{x=m}

e_i is the i-th standard basis vector.

**Note:** Särkkä does not present the quadratic approximation for the non-additive case.

---

### Algorithm 5.7 — Statistical linearization, additive transform

**Setup:** x ~ N(m, P), q ~ N(0, Q), y = g(x) + q

**Joint approximation:**

(x, y) ~ N( (m, μ_S), [[P, C_S], [C_S^T, S_S]] )

where:
- μ_S = E[g(x)]
- S_S = E[g(x) δx^T] P^{-1} E[g(x) δx^T]^T + Q
- C_S = E[g(x) δx^T]^T

with δx = x - m, expectations taken w.r.t. x ~ N(m, P).

---

### Algorithm 5.8 — Statistical linearization, non-additive transform

**Setup:** x ~ N(m, P), q ~ N(0, Q), y = g(x, q)

**Joint approximation:**

(x, y) ~ N( (m, μ_S), [[P, C_S], [C_S^T, S_S]] )

where:
- μ_S = E[g(x, q)]
- S_S = E[g(x,q) δx^T] P^{-1} E[g(x,q) δx^T]^T + E[g(x,q) q^T] Q^{-1} E[g(x,q) q^T]^T
- C_S = E[g(x,q) δx^T]^T

Expectations taken w.r.t. x ~ N(m, P) and q ~ N(0, Q).

---

### Algorithm 5.9 — Statistical linearization (Jacobian form), additive transform

**Setup:** Same as Algorithm 5.7.

Alternative form using the identity E[g(x) δx^T] = E[G_x(x)] P:

- μ_S = E[g(x)]
- S_S = E[G_x(x)] P E[G_x(x)]^T + Q
- C_S = P E[G_x(x)]^T

where G_x(x) is the Jacobian of g. Expectations w.r.t. x ~ N(m, P).

**Key identity used:** For x ~ N(m, P) and differentiable g:

E[g(x)(x - m)^T] = E[G_x(x)] P

---

### Algorithm 5.12 — Unscented transform, additive transform

**Setup:** x ~ N(m, P), q ~ N(0, Q), y = g(x) + q, x ∈ R^n

**Joint approximation:**

(x, y) ~ N( (m, μ_U), [[P, C_U], [C_U^T, S_U]] )

**Procedure:**

1. Form 2n+1 sigma points:
   - X^(0) = m
   - X^(i) = m + √(n+λ) [√P]_i,  i = 1,...,n
   - X^(i+n) = m - √(n+λ) [√P]_i,  i = 1,...,n

   where [√P]_i denotes the i-th column, λ = α²(n+κ) - n.

2. Propagate: Y^(i) = g(X^(i)), i = 0,...,2n

3. Compute:
   - μ_U = Σ_{i=0}^{2n} W_i^(m) Y^(i)
   - S_U = Σ_{i=0}^{2n} W_i^(c) (Y^(i) - μ_U)(Y^(i) - μ_U)^T + Q
   - C_U = Σ_{i=0}^{2n} W_i^(c) (X^(i) - m)(Y^(i) - μ_U)^T

**Weights:**
- W_0^(m) = λ/(n+λ)
- W_0^(c) = λ/(n+λ) + (1 - α² + β)
- W_i^(m) = W_i^(c) = 1/(2(n+λ)),  i = 1,...,2n

---

### Algorithm 5.13 — Unscented transform, non-additive transform

**Setup:** x ~ N(m, P), q ~ N(0, Q), y = g(x, q), x ∈ R^n, q ∈ R^{n_q}, n' = n + n_q

**Joint approximation:**

(x, y) ~ N( (m, μ_U), [[P, C_U], [C_U^T, S_U]] )

**Procedure:**

1. Form sigma points for augmented variable x̃ = (x, q):
   - m̃ = (m, 0),  P̃ = diag(P, Q)
   - X̃^(0) = m̃
   - X̃^(i) = m̃ + √(n'+λ') [√P̃]_i,  i = 1,...,n'
   - X̃^(i+n') = m̃ - √(n'+λ') [√P̃]_i,  i = 1,...,n'

   where λ' is defined as λ but with n replaced by n'.

2. Propagate: Ŷ^(i) = g(X̃^(i)_x, X̃^(i)_q), i = 0,...,2n'

3. Compute:
   - μ_U = Σ_{i=0}^{2n'} W_i^(m)' Ŷ^(i)
   - S_U = Σ_{i=0}^{2n'} W_i^(c)' (Ŷ^(i) - μ_U)(Ŷ^(i) - μ_U)^T
   - C_U = Σ_{i=0}^{2n'} W_i^(c)' (X̃^(i)_x - m)(Ŷ^(i) - μ_U)^T

   (No separate +Q; noise absorbed through augmentation.)

---

## Filters (Applying the Transforms to the Bayesian Recursion)

All filters assume p(x_{k-1} | y_{1:k-1}) ≈ N(m_{k-1}, P_{k-1}) and produce p(x_k | y_{1:k}) ≈ N(m_k, P_k).

**Common update structure (all filters):**

Given predicted mean m_k^-, predicted covariance P_k^-, predicted observation ŷ_k, innovation covariance S_k, cross-covariance C_k:

- K_k = C_k S_k^{-1}
- v_k = y_k - ŷ_k
- m_k = m_k^- + K_k v_k
- P_k = P_k^- - K_k S_k K_k^T

The filters differ only in how they compute (m_k^-, P_k^-, ŷ_k, S_k, C_k).

---

### Algorithm 5.4 — EKF I (first order, additive noise)

**Model:** x_k = f(x_{k-1}) + q_{k-1}, y_k = h(x_k) + r_k

**Prediction:**
- m_k^- = f(m_{k-1})
- P_k^- = F_x(m_{k-1}) P_{k-1} F_x(m_{k-1})^T + Q_{k-1}

**Update:**
- ŷ_k = h(m_k^-)
- S_k = H_x(m_k^-) P_k^- H_x(m_k^-)^T + R_k
- C_k = P_k^- H_x(m_k^-)^T

where F_x, H_x are Jacobians of f, h w.r.t. x.

---

### Algorithm 5.5 — EKF II (first order, non-additive noise)

**Model:** x_k = f(x_{k-1}, q_{k-1}), y_k = h(x_k, r_k)

**Prediction:**
- m_k^- = f(m_{k-1}, 0)
- P_k^- = F_x(m_{k-1}) P_{k-1} F_x(m_{k-1})^T + F_q(m_{k-1}) Q_{k-1} F_q(m_{k-1})^T

**Update:**
- ŷ_k = h(m_k^-, 0)
- S_k = H_x(m_k^-) P_k^- H_x(m_k^-)^T + H_r(m_k^-) R_k H_r(m_k^-)^T
- C_k = P_k^- H_x(m_k^-)^T

where:
- F_x = ∂f/∂x |_{x=m, q=0}
- F_q = ∂f/∂q |_{x=m, q=0}
- H_x = ∂h/∂x |_{x=m, r=0}
- H_r = ∂h/∂r |_{x=m, r=0}

---

### Algorithm 5.6 — EKF III (second order, additive noise only)

**Model:** x_k = f(x_{k-1}) + q_{k-1}, y_k = h(x_k) + r_k

**Prediction:**
- m_k^- = f(m_{k-1}) + (1/2) Σ_i e_i tr( F^(i)_xx(m_{k-1}) P_{k-1} )
- P_k^- = F_x(m_{k-1}) P_{k-1} F_x(m_{k-1})^T + (1/2) Σ_{i,i'} e_i e_{i'}^T tr( F^(i)_xx(m_{k-1}) P_{k-1} F^(i')_xx(m_{k-1}) P_{k-1} ) + Q_{k-1}

**Update:**
- ŷ_k = h(m_k^-) + (1/2) Σ_i e_i tr( H^(i)_xx(m_k^-) P_k^- )
- S_k = H_x(m_k^-) P_k^- H_x(m_k^-)^T + (1/2) Σ_{i,i'} e_i e_{i'}^T tr( H^(i)_xx(m_k^-) P_k^- H^(i')_xx(m_k^-) P_k^- ) + R_k
- C_k = P_k^- H_x(m_k^-)^T

v_k = y_k - ŷ_k  (note: ŷ_k now includes the bias correction)

where F^(i)_xx and H^(i)_xx are Hessians of f_i and h_i.

**Non-additive case:** Not presented in Särkkä.

---

### Algorithm 5.10 — SLF I (statistical linearization, additive noise)

**Model:** x_k = f(x_{k-1}) + q_{k-1}, y_k = h(x_k) + r_k

**Prediction:** (expectations w.r.t. x_{k-1} ~ N(m_{k-1}, P_{k-1}))
- m_k^- = E[f(x_{k-1})]
- P_k^- = E[f(x_{k-1}) δx_{k-1}^T] P_{k-1}^{-1} E[f(x_{k-1}) δx_{k-1}^T]^T + Q_{k-1}

**Update:** (expectations w.r.t. x_k ~ N(m_k^-, P_k^-))
- ŷ_k = E[h(x_k)]
- S_k = E[h(x_k) δ̃x_k^T] (P_k^-)^{-1} E[h(x_k) δ̃x_k^T]^T + R_k
- C_k = E[h(x_k) δ̃x_k^T]^T

where δx_{k-1} = x_{k-1} - m_{k-1} and δ̃x_k = x_k - m_k^-.

---

### Algorithm 5.11 — SLF II (statistical linearization, non-additive noise)

**Model:** x_k = f(x_{k-1}, q_{k-1}), y_k = h(x_k, r_k)

**Prediction:** (expectations w.r.t. x_{k-1} ~ N(m_{k-1}, P_{k-1}) and q_{k-1} ~ N(0, Q_{k-1}))
- m_k^- = E[f(x_{k-1}, q_{k-1})]
- P_k^- = E[f(x_{k-1}, q_{k-1}) δx_{k-1}^T] P_{k-1}^{-1} E[f(x_{k-1}, q_{k-1}) δx_{k-1}^T]^T
         + E[f(x_{k-1}, q_{k-1}) q_{k-1}^T] Q_{k-1}^{-1} E[f(x_{k-1}, q_{k-1}) q_{k-1}^T]^T

**Update:** (expectations w.r.t. x_k ~ N(m_k^-, P_k^-) and r_k ~ N(0, R_k))
- ŷ_k = E[h(x_k, r_k)]
- S_k = E[h(x_k, r_k) δ̃x_k^T] (P_k^-)^{-1} E[h(x_k, r_k) δ̃x_k^T]^T
       + E[h(x_k, r_k) r_k^T] R_k^{-1} E[h(x_k, r_k) r_k^T]^T
- C_k = E[h(x_k, r_k) δ̃x_k^T]^T

---

### Algorithm 5.14 — UKF I (additive noise)

**Model:** x_k = f(x_{k-1}) + q_{k-1}, y_k = h(x_k) + r_k, x ∈ R^n

**Prediction:**

1. Sigma points from N(m_{k-1}, P_{k-1}):
   - X^(0) = m_{k-1}
   - X^(i) = m_{k-1} + √(n+λ) [√P_{k-1}]_i,  i = 1,...,n
   - X^(i+n) = m_{k-1} - √(n+λ) [√P_{k-1}]_i,  i = 1,...,n

2. Propagate: X̂^(i) = f(X^(i)),  i = 0,...,2n

3. Predicted moments:
   - m_k^- = Σ W_i^(m) X̂^(i)
   - P_k^- = Σ W_i^(c) (X̂^(i) - m_k^-)(X̂^(i) - m_k^-)^T + Q_{k-1}

**Update:**

1. Sigma points from N(m_k^-, P_k^-): same construction

2. Propagate: Ŷ^(i) = h(X^(i)),  i = 0,...,2n

3. Moments:
   - ŷ_k = Σ W_i^(m) Ŷ^(i)
   - S_k = Σ W_i^(c) (Ŷ^(i) - ŷ_k)(Ŷ^(i) - ŷ_k)^T + R_k
   - C_k = Σ W_i^(c) (X^(i) - m_k^-)(Ŷ^(i) - ŷ_k)^T

4. Standard Kalman update with K_k, v_k, m_k, P_k.

**Weights:** W_0^(m) = λ/(n+λ), W_0^(c) = λ/(n+λ) + (1 - α² + β), W_i^(m) = W_i^(c) = 1/(2(n+λ)) for i ≥ 1.

---

### Algorithm 5.15 — UKF II (non-additive noise, augmented)

**Model:** x_k = f(x_{k-1}, q_{k-1}), y_k = h(x_k, r_k), x ∈ R^n, q ∈ R^{n_q}, r ∈ R^{n_r}

**Prediction:** (augmented with q, n' = n + n_q)

1. Augmented sigma points from m̃ = (m_{k-1}, 0), P̃ = diag(P_{k-1}, Q_{k-1}):
   - X̃^(0) = m̃
   - X̃^(i) = m̃ + √(n'+λ') [√P̃]_i,  i = 1,...,n'
   - X̃^(i+n') = m̃ - √(n'+λ') [√P̃]_i,  i = 1,...,n'

2. Propagate: X̂^(i) = f(X̃^(i)_x, X̃^(i)_q),  i = 0,...,2n'

3. Predicted moments:
   - m_k^- = Σ W_i^(m)' X̂^(i)
   - P_k^- = Σ W_i^(c)' (X̂^(i) - m_k^-)(X̂^(i) - m_k^-)^T

   (No separate +Q; noise absorbed through augmentation.)

**Update:** (augmented with r, n'' = n + n_r)

1. Augmented sigma points from m̃ = (m_k^-, 0), P̃ = diag(P_k^-, R_k):
   - same construction with n'', λ''

2. Propagate: Ŷ^(i) = h(X̃^(i)_x, X̃^(i)_r),  i = 0,...,2n''

3. Moments:
   - ŷ_k = Σ W_i^(m)'' Ŷ^(i)
   - S_k = Σ W_i^(c)'' (Ŷ^(i) - ŷ_k)(Ŷ^(i) - ŷ_k)^T
   - C_k = Σ W_i^(c)'' (X̃^(i)_x - m_k^-)(Ŷ^(i) - ŷ_k)^T

   (No separate +R.)

4. Standard Kalman update.

---

## Log-Space Approach for Multiplicative Noise (Not in Särkkä)

### Motivation

Algorithms 5.5 and 5.15 handle non-additive noise generically (via Jacobian F_q or augmentation). But for models where the noise enters **multiplicatively**, a coordinate change can convert the problem to additive noise, after which the cheaper additive-noise algorithms (5.4, 5.6, 5.14) apply directly. This is the standard practitioner approach for stochastic volatility models.

**The key idea:** if x_{t+1} = x_t · w_t with w_t lognormal, then log x_{t+1} = log x_t + log w_t, which is additive Gaussian in log-space.

**This works for any filter** — EKF, SOEKF, UKF, SLF — because the coordinate change is done before the filter, not inside it.

---

### Example: Discrete Stochastic Volatility Model

**Original model (multiplicative observation noise):**

State (log-volatility, already additive):
  v_t = α + β v_{t-1} + σ_η η_t,    η_t ~ N(0, 1)

Observation (returns, multiplicative):
  y_t = exp(v_t / 2) · ε_t,    ε_t ~ N(0, 1)

The observation has non-additive noise: y = h(v, ε) = exp(v/2) · ε.

**Log-squared transformation of observation:**

Define z_t = log(y_t²) = v_t + log(ε_t²).

Now log(ε_t²) = log(χ²_1), which is non-Gaussian. Standard approximation:
  log(χ²_1) ≈ N(-1.2704, π²/2)

So the transformed model is:

State:    v_t = α + β v_{t-1} + σ_η η_t
Obs:      z_t = v_t + ξ_t,    ξ_t ~ N(-1.2704, π²/2)

This is now a **linear-Gaussian** model (up to the log-χ² approximation), solvable by the plain Kalman filter.

---

### Algorithm: Log-Space EKF for SV

**Transformed model:**
  v_t = α + β v_{t-1} + σ_η η_t
  z_t = v_t + ξ_t,    ξ_t ~ N(μ_ξ, σ²_ξ)

where μ_ξ = -1.2704, σ²_ξ = π²/2.

Since this is linear-Gaussian, we use the **plain Kalman filter:**

**Prediction:**
- m_t^- = α + β m_{t-1}
- P_t^- = β² P_{t-1} + σ²_η

**Update:**
- v_t = z_t - (m_t^- + μ_ξ)
- S_t = P_t^- + σ²_ξ
- K_t = P_t^- / S_t
- m_t = m_t^- + K_t v_t
- P_t = (1 - K_t) P_t^-

where z_t = log(y_t²) is computed from the raw observation y_t.

**Note:** The approximation quality depends entirely on how well N(μ_ξ, σ²_ξ) approximates the log-χ²_1 distribution. Kim, Shephard & Chib (1998) use a 7-component Gaussian mixture for better accuracy.

---

### Algorithm: Log-Space UKF for General Multiplicative Dynamics

**Original model with multiplicative noise:**
  x_t = g(x_{t-1}) · exp(σ_q ε_t - σ²_q/2),    ε_t ~ N(0,1)
  y_t = h(x_t) + r_t,    r_t ~ N(0, R)

**Coordinate change:** Define ζ_t = log x_t.

**Transformed model (additive noise):**
  ζ_t = log g(exp(ζ_{t-1})) + σ_q ε_t - σ²_q/2    ← define f̃(ζ) = log g(exp(ζ)) - σ²_q/2
  y_t = h(exp(ζ_t)) + r_t                            ← define h̃(ζ) = h(exp(ζ))

So:  ζ_t = f̃(ζ_{t-1}) + σ_q ε_t,    y_t = h̃(ζ_t) + r_t

Now apply the **standard additive-noise UKF (Algorithm 5.14)** to f̃ and h̃:

**Prediction:**

1. Sigma points from N(m_{t-1}, P_{t-1}) in log-space:
   - Z^(0) = m_{t-1}
   - Z^(i) = m_{t-1} ± √((n+λ) P_{t-1})   (1-d: just ± √((1+λ) P_{t-1}))

2. Propagate: Ẑ^(i) = f̃(Z^(i)) = log g(exp(Z^(i))) - σ²_q/2

3. m_t^- = Σ W_i^(m) Ẑ^(i),    P_t^- = Σ W_i^(c) (Ẑ^(i) - m_t^-)² + σ²_q

**Update:**

1. Sigma points from N(m_t^-, P_t^-) in log-space

2. Propagate: Ŷ^(i) = h̃(Z^(i)) = h(exp(Z^(i)))

3. ŷ_t = Σ W_i^(m) Ŷ^(i)
   S_t = Σ W_i^(c) (Ŷ^(i) - ŷ_t)² + R
   C_t = Σ W_i^(c) (Z^(i) - m_t^-)(Ŷ^(i) - ŷ_t)

4. Standard Kalman update: K_t, v_t, m_t, P_t.

**Note:** m_t and P_t are the mean and variance of **log x_t**, not x_t. To recover moments of x_t, use lognormal identities: E[x_t] = exp(m_t + P_t/2), etc.

---

### Algorithm: Log-Space EKF for General Multiplicative Dynamics

**Same transformed model as above:**
  ζ_t = f̃(ζ_{t-1}) + σ_q ε_t,    y_t = h̃(ζ_t) + r_t

Apply **Algorithm 5.4** (first-order additive EKF) to f̃ and h̃:

**Prediction:**
- m_t^- = f̃(m_{t-1}) = log g(exp(m_{t-1})) - σ²_q/2
- P_t^- = (f̃'(m_{t-1}))² P_{t-1} + σ²_q

where f̃'(ζ) = g'(exp(ζ)) exp(ζ) / g(exp(ζ)).

**Update:**
- ŷ_t = h̃(m_t^-) = h(exp(m_t^-))
- S_t = (h̃'(m_t^-))² P_t^- + R
- C_t = P_t^- h̃'(m_t^-)
- K_t = C_t / S_t

where h̃'(ζ) = h'(exp(ζ)) exp(ζ).

- v_t = y_t - ŷ_t
- m_t = m_t^- + K_t v_t
- P_t = (1 - K_t h̃'(m_t^-)) P_t^-

**Trade-off:** The original model had multiplicative noise but perhaps simpler h. The transformed model has additive noise but h̃(ζ) = h(exp(ζ)) is more nonlinear, so the EKF linearization may be less accurate.

---

## Summary Table

| Filter | Additive noise | Non-additive noise | Log-space for mult. noise |
|---|---|---|---|
| 1st-order EKF | Alg 5.4 | Alg 5.5 (needs F_q, H_r) | Transform + Alg 5.4 |
| 2nd-order EKF | Alg 5.6 | Not in Särkkä | Transform + Alg 5.6 |
| SLF | Alg 5.10 | Alg 5.11 | Transform + Alg 5.10 |
| UKF | Alg 5.14 | Alg 5.15 (augmented) | Transform + Alg 5.14 |

**Key point:** The log-space column always uses the additive-noise algorithm, because the coordinate change has already made the noise additive. The cost is increased nonlinearity in the transformed functions f̃ and h̃.
