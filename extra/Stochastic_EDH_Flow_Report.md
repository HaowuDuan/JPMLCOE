# Stochastic EDH Flow: Implementation Cross-Check Report

**Date**: 2026-02-11
**Reference Paper**: Dai & Daum, "A New Parameterized Family of Stochastic Particle Flow Filters" (arXiv:2103.09676v3, 2021)

---

## 1. Paper Summary

### Core Idea

The paper derives a parameterized family of stochastic particle flows that unifies several existing flows (Exact Flow, Stochastic Flow with Fixed Q, Diagnostic Noise, Approximate Flow). The key insight: adding a diffusion term Q to the flow SDE **stabilizes** the particle evolution without changing the posterior distribution (under Gaussian assumptions).

### The Stochastic Flow SDE

$$dx = f(x, \lambda) d\lambda + \sqrt{Q} \, dw_\lambda$$

where $w_\lambda$ is Brownian motion in pseudo-time $\lambda \in [0,1]$.

### Key Equations (Theorem 2.1)

Under assumptions (A1) linear gradients, (A2) non-singular Hessian:

**Drift** (Eq. 6):
$$f = (\nabla^2 \log p)^{-1} \left[ -\nabla \log h + K (\nabla^2 \log p)^{-1} (\nabla \log p) \right]$$

**Diffusion** (Eq. 7):
$$Q = (\nabla^2 \log p)^{-1} (-\nabla^2 \log h + K + K^T) (\nabla^2 \log p)^{-1}$$

The parameter $K$ controls which flow you get:
- **K = 0**: Stochastic Flow with Fixed Q (Theorem 4.1.2)
- **Q = 0** (K chosen to make Q vanish): Exact (deterministic) Flow (Theorem 4.1.1)
- **Arbitrary Q > 0**: Choose K via Corollary 2.1: $K = \frac{1}{2}(\nabla^2 \log p) Q (\nabla^2 \log p) + \frac{1}{2}(\nabla^2 \log h)$

### Linear Gaussian Specialization

Under Gaussian prior $g(x) \sim \mathcal{N}(x_{\text{prior}}, P_g)$ and linear measurement $z = Hx + v$, $v \sim \mathcal{N}(0, R)$:

$$\nabla^2 \log g = -P_g^{-1}, \quad \nabla^2 \log h = -H^T R^{-1} H$$

$$\nabla^2 \log p = -P_g^{-1} - \lambda H^T R^{-1} H = -P(\lambda)^{-1}$$

where $P(\lambda) = (P_g^{-1} + \lambda H^T R^{-1} H)^{-1}$ is the intermediate covariance.

The SDE linearizes to (Eq. 11):
$$dx = [A(\lambda) x + b(\lambda)] d\lambda + \sqrt{Q} \, dw_\lambda$$

with (Eq. 12):
$$A(\lambda) = (\nabla^2 \log p)^{-1} [-\nabla^2 \log h + K] = P(\lambda)[H^T R^{-1} H + K]$$

### Special Cases Summary

| Flow | K | Q |
|------|---|---|
| Exact (deterministic) | Makes Q = 0 | 0 |
| K=0 stochastic | 0 | $P(\lambda) H^T R^{-1} H \, P(\lambda)$ |
| Arbitrary Q | $\frac{1}{2} P^{-1}(\lambda) Q P^{-1}(\lambda) + \frac{1}{2} \nabla^2 \log h$ | User-chosen |

---

## 2. Pseudocode

### Algorithm: Stochastic EDH Particle Flow Filter

**Given**: Prior particles $\{x_i^{k-1}\}$, observation $z_k$, EKF/UKF for covariance guidance

**Predict step**:
```
1. Update EKF/UKF mean ← ensemble mean of particles
2. Run EKF/UKF predict → get P_{k|k-1}, μ_{k|k-1}
3. Store η̄_0 = μ_{k|k-1}  (deterministic predicted mean)
4. Propagate particles: η_i^0 = f(x_i^{k-1}) + process_noise  for i=1..N
```

**Update step** (flow from λ=0 to λ=1):
```
5. Set Q = q·I  (user-chosen diffusion scale)
6. For j = 1, ..., n_steps:
     a. λ ← λ + ε_j
     b. Compute flow parameters using Gaussian approximation:
        - P = P_{k|k-1}  (from EKF/UKF, fixed throughout flow)
        - H = ∂h/∂x evaluated at η̄_0  (fixed, NOT re-linearized)
        - Compute A(λ), b(λ) from drift formula
     c. For each particle i:
        - dW_i ~ N(0, ε_j · I)
        - η_i ← η_i + [A·η_i + b]·ε_j + √Q · dW_i
7. Particles at λ=1 represent posterior samples
8. Update EKF/UKF with observation z_k
```

### Computing the Evolution Kernel (A and b)

For a user-chosen scalar $q$ with $Q = qI$, the drift has the form $f(x) = K_1 \nabla \log g(x) + K_2 \nabla \log h(x)$ where:

**Option A: Use Corollary 2.1** (arbitrary Q, solve for K):

$$K = \frac{1}{2} P^{-1}(\lambda) \cdot Q \cdot P^{-1}(\lambda) + \frac{1}{2}(-H^T R^{-1} H)$$

Then $A(\lambda) = P(\lambda)[H^T R^{-1} H + K]$ and $b = f(\bar{x}) - A \bar{x}$.

**Option B: Direct computation** (what the current implementation does):

$$K_1 = \frac{1}{2}Q + \frac{1}{2}(-P)(-H^T R^{-1} H)(-P) = \frac{1}{2}Q + \frac{1}{2} P H^T R^{-1} H P$$

$$K_2 = -(\nabla^2 \log p)^{-1} = P(\lambda)$$

$$A = K_1 \nabla^2 \log g + K_2 \nabla^2 \log h = K_1 (-P_g^{-1}) + P(\lambda)(-H^T R^{-1} H)$$

$$b = f(\bar{x}) - A \bar{x}$$

**Option C: Simplification for Exact Flow + additive noise** (recommended approach):

Since the paper proves that Q does not affect the posterior distribution (Theorem 3.1), the simplest correct implementation is:

1. Compute $A(\lambda)$, $b(\lambda)$ exactly as the deterministic Exact Flow (Eqs. 21-22):

$$A = -\frac{1}{2} P H^T (\lambda H P H^T + R)^{-1} H$$

$$b = (I + 2\lambda A)[(I + \lambda A) P H^T R^{-1}(z - e) + A \mu_0]$$

2. Add isotropic diffusion noise at each step:

$$\eta_i \leftarrow \eta_i + [A \eta_i + b] \epsilon_j + \sqrt{q} \cdot dW_i$$

This is valid because: the Exact Flow is a special case (K=0 → Q=0), and Theorem 3.1 shows that for ANY choice of Q (or K), the posterior distribution is the same. The noise simply stabilizes the numerics.

---

## 3. Cross-Check: Current Implementation vs Paper

### 3.1 Framework Mismatch

| Aspect | `edh_flow.py` (deterministic) | `stochastic_edh.py` (SDE) |
|--------|-------------------------------|---------------------------|
| Language | TensorFlow | NumPy/SciPy |
| Base class | `FlowFilterBase` (shared) | `FlowFilterBase` (shared) |
| Flow params | `compute_flow_params()` (shared utility) | Custom `_compute_drift_coefficients()` |
| Integration | `euler_step` (deterministic) | `euler_maruyama_step` (SDE) |
| Re-linearization | Yes, at each λ step (`eta_bar` updated) | No (fixed at η̄_0) |
| P source | EKF predicted cov (fixed) | EKF or empirical (configurable) |
| Covariance blending | No | Yes (α=0.5 blend in predict) |

**Key observation**: The two implementations are inconsistent in several ways. The stochastic version re-derives the drift from scratch using Hessian formulas rather than reusing the Exact Flow A, b with added noise.

### 3.2 Issue #1: HIGH - Drift Computation Uses Hessians Instead of Exact Flow Formulas

**Current code** (`stochastic_edh.py:150-223`):

The implementation computes K1, K2 from the paper's Theorem 2.1 Hessian formulas:
```python
# K₁ = 0.5*Q + 0.5*(∇²log p)⁻¹(∇²log h)(∇²log p)⁻¹
K1 = 0.5 * Q + 0.5 * (inv_hess_log_p @ hess_log_h @ inv_hess_log_p)
# K₂ = -(∇²log p)⁻¹ = P
K2 = -inv_hess_log_p  # = P
# A = K₁(∇²log p) + K₂(∇²log h)
A = K1 @ hess_log_p + K2 @ hess_log_h
```

**Problem**: This is mathematically correct for the Gaussian case, but:

1. It uses `P` as `self.predicted_cov` (the EKF-predicted $P_{k|k-1}$), NOT the λ-dependent $P(\lambda) = (P_g^{-1} + \lambda H^T R^{-1} H)^{-1}$. The paper's $\nabla^2 \log p = -P(\lambda)^{-1}$ varies with λ, but the code uses a **fixed** $P$. This means the Hessian $\nabla^2 \log p$ is approximated as $-P_{k|k-1}^{-1}$ at all λ values.

2. The Exact Flow formulas (Eqs. 21-22) already encode the correct λ-dependence through $S(\lambda) = \lambda H P H^T + R$. Using those formulas directly is simpler AND more correct in the Gaussian-guided regime.

**Recommendation**: Replace `_compute_drift_coefficients` with the Exact Flow's `compute_flow_params` + additive noise. The paper explicitly proves (Theorem 4.1.1) that the Exact Flow is a special case with Q=0, and Theorem 3.1 shows Q doesn't change the posterior.

### 3.3 Issue #2: HIGH - Hessian of log p Does NOT Account for λ

**Current code** (`stochastic_edh.py:197-198`):
```python
# Hessian of log p (Gaussian approximation - no analytical prior)
hess_log_p = -P_inv
```

**Paper** (Section 2, Eq. 3):
$$\nabla^2 \log p(x, \lambda) = \nabla^2 \log g + \lambda \nabla^2 \log h = -P_g^{-1} - \lambda H^T R^{-1} H$$

The code uses `hess_log_p = -P_inv` where `P = self.predicted_cov` (a **fixed** matrix). This means $\nabla^2 \log p$ doesn't vary with λ, which is incorrect. The correct formulation should be:

```python
hess_log_p = -P_inv - lambda_val * H.T @ R_inv @ H  # = -P(λ)^{-1}
```

or equivalently, use $P(\lambda)^{-1} = P_g^{-1} + \lambda H^T R^{-1} H$.

However, since we are already using a Gaussian approximation with EKF-guided P (not exact P_g), this distinction is somewhat academic. The Exact Flow formulas (Eqs. 21-22) handle the λ-dependence correctly through S(λ), which is why reusing them is the cleaner approach.

### 3.4 Issue #3: MEDIUM - Observation Model Re-Linearization in Intermediate Steps

**Current `edh_flow.py` behavior**: Re-linearizes H at each λ step by recomputing `eta_bar = mean(particles_flow)` and passing it to `compute_flow_params`, which calls `model.observation_jacobian(eta_bar)`.

**Current `stochastic_edh.py` behavior**: Computes H at `eta_bar` (which is the mean of flowing particles) at each step, but this is the **Gaussian-guided mean**, not the particles themselves.

**Paper's assumption**: The derivation assumes $\nabla \log h$ is linear in x (Assumption A1), which means H is **constant** (not re-linearized). The Exact Flow formulas (Eqs. 21-22) use a fixed H evaluated at the prior mean.

**For the stochastic flow**: Since the full derivation assumes constant H, the SDE version should use a **fixed** H evaluated once at η̄_0. Re-linearization during intermediate λ steps would invalidate the theoretical guarantees (unbiasedness, consistency, stability).

**For the deterministic EDH flow** (`edh_flow.py`): Re-linearization at intermediate steps is a practical heuristic that often improves performance for nonlinear problems, even though it goes beyond the paper's assumptions. This is fine for the ODE case where there are no stochastic stability concerns.

**Recommendation**: The stochastic flow should NOT re-linearize at intermediate steps (and currently it effectively doesn't, since it computes H at the flowing mean which changes slowly). But this should be made explicit by computing H once.

### 3.5 Issue #4: MEDIUM - Observation Hessian Term (Second-Order)

**Current code** (`stochastic_edh.py:191-195`):
```python
if hasattr(self.model, 'observation_hessian'):
    obs_hessian = self.model.observation_hessian(eta_bar)
    for i in range(self.obs_dim):
        hess_log_h += weighted_innovation[i] * obs_hessian[i]
```

The code includes a second-order correction to $\nabla^2 \log h$ using the observation Hessian. This goes beyond the paper's Assumption (A1) which requires $\nabla \log h$ to be linear in x (i.e., $\nabla^2 \log h$ is constant, meaning no second-order term).

For a nonlinear observation model $h(x)$, the full Hessian of $\log h$ is:
$$\nabla^2 \log h = -H^T R^{-1} H + \sum_i [R^{-1}(z - h(x))]_i \frac{\partial^2 h_i}{\partial x^2}$$

While including this term is a valid extension for nonlinear problems, it violates the assumption under which the stochastic flow was derived. The second-order term makes $\nabla^2 \log h$ depend on x, so the Q matrix in Eq. 7 would also depend on x — contradicting the requirement that Q is state-independent.

**Recommendation**: For the stochastic flow, use only the first-order term: `hess_log_h = -H.T @ R_inv @ H`. The second-order correction can be kept as an optional enhancement but should not be the default.

### 3.6 Issue #5: LOW - Covariance Blending in Predict

**Current code** (`stochastic_edh.py:121-122`):
```python
alpha = 0.5
self.global_filter.cov = alpha * ensemble_cov + (1 - alpha) * self.global_filter.cov
```

The deterministic `edh_flow.py` does NOT blend covariances (line 200-205) — it just updates the mean and lets the EKF maintain its own covariance. The stochastic version's blending is an ad-hoc modification not supported by theory.

### 3.7 Issue #6: LOW - Euler-Maruyama Noise Uses No Seed

**Current code** (`stochastic_edh.py:287-290`):
```python
particles_flow = euler_maruyama_step(
    particles_flow, self._compute_drift, d_lambda, A, b,
    diffusion_coeff=self.diffusion_scale
)
```

Looking at `euler_maruyama_step` (ode_solvers.py:79-84):
```python
if diffusion_coeff > 0 and seed is not None:
    noise = tf.random.stateless_normal(...)
    ...
else:
    return x + drift * dt  # Falls through to deterministic!
```

The `seed` parameter is not passed, so the function falls through to the deterministic branch even when `diffusion_scale > 0`. **The diffusion noise is never actually applied.**

**Wait** — `stochastic_edh.py` uses NumPy arrays, not TensorFlow tensors. The `euler_maruyama_step` in ode_solvers.py is TF-based. Let me re-check... Actually, the import at line 14 is `from ...utils.ode_solvers import euler_maruyama_step`, but the implementation expects TF tensors. If `particles_flow` is a NumPy array (from `flow_base.py`'s predict), this would either error or silently convert.

This is a significant bug: either the noise is not applied (because seed=None), or there's a type mismatch between NumPy and TensorFlow.

---

## 4. Can StochasticEDHFlow Inherit from ExactDaumHuangFlow?

**Short answer: Yes, with the "Option C" approach (Exact Flow + additive noise).**

### What they share:
1. Same base class (`FlowFilterBase`)
2. Same EKF/UKF covariance guidance mechanism
3. Same predict step (propagate particles + EKF predict)
4. Same global filter update after flow
5. Same λ step schedule generation
6. Same flow structure: loop over λ steps, compute A/b, migrate particles

### What differs:
1. **Integration**: ODE (Euler) vs SDE (Euler-Maruyama) — just add noise term
2. **Re-linearization**: `edh_flow.py` re-linearizes H at each step; stochastic version should NOT (per paper theory). However, for practical nonlinear problems, re-linearization is a heuristic that helps. If we want maximum code reuse, we can make re-linearization a configurable option.
3. **Q matrix**: Stochastic version has `Q = q·I` as additional parameter

### Proposed Inheritance Structure

```python
class StochasticEDHFlow(ExactDaumHuangFlow):
    """Stochastic extension of EDH flow: replaces ODE with SDE."""

    def __init__(self, model, diffusion_scale=0.001,
                 relinearize=False,   # False = paper-correct for SDE
                 **kwargs):
        super().__init__(model, **kwargs)
        self.diffusion_scale = diffusion_scale
        self.relinearize = relinearize

    def update(self, y):
        """Same as EDH flow update, but with Euler-Maruyama instead of Euler."""
        # Reuse parent's setup (convert to TF, compute R_inv, etc.)
        # Override the integration step to add √Q · dW noise
        # Optionally skip re-linearization based on self.relinearize
        ...
```

### Key Design Decision: Re-linearization

| Approach | Re-linearize H? | Theory | Practice |
|----------|-----------------|--------|----------|
| Paper-correct SDE | No (fix H at η̄_0) | Guaranteed unbiased, stable | May underperform for highly nonlinear h(x) |
| Practical SDE | Yes (update at each step) | No formal guarantees | Often better empirically |
| Exact Flow (ODE) | Yes (current edh_flow.py) | Heuristic extension | Standard practice |

**Recommendation**: Inherit from `ExactDaumHuangFlow`, add `relinearize` flag (default False for paper correctness), and add the noise term in the integration step.

---

## 5. Summary of Issues

| # | Severity | Issue | Recommendation |
|---|----------|-------|----------------|
| 1 | HIGH | Drift uses raw Hessian formulas instead of Exact Flow A,b | Use `compute_flow_params` + additive noise |
| 2 | HIGH | Hessian of log p doesn't vary with λ | Fixed by using Exact Flow formulas |
| 3 | MEDIUM | Re-linearization policy unclear | Make explicit: default NO for SDE |
| 4 | MEDIUM | Second-order obs Hessian violates paper assumptions | Use first-order only by default |
| 5 | LOW | Covariance blending in predict (ad-hoc) | Remove blending, match edh_flow.py |
| 6 | HIGH | Euler-Maruyama never applies noise (seed=None) | Pass seed or fix integration |

---

## 6. Recommended Refactored Implementation

The cleanest approach is **Option C**: inherit from `ExactDaumHuangFlow`, reuse its `compute_flow_params`-based A/b computation, and simply add SDE noise:

```python
class StochasticEDHFlow(ExactDaumHuangFlow):
    """
    Stochastic EDH particle flow filter.

    Extends the deterministic Exact Flow with isotropic diffusion noise
    for improved numerical stability (Dai & Daum, 2021).

    SDE: dx = [A(λ)x + b(λ)]dλ + √(q·I) dw_λ

    Theorem 3.1 guarantees that the posterior distribution is unchanged
    by the diffusion term Q under Gaussian assumptions.
    """

    def __init__(self, model, diffusion_scale=0.001, relinearize=False, **kwargs):
        super().__init__(model, **kwargs)
        self.diffusion_scale = diffusion_scale
        self.relinearize = relinearize  # False = paper-correct

    def update(self, y):
        """Flow with Euler-Maruyama: reuse parent's A,b + add √Q·dW."""
        # Same setup as parent...
        # In the integration loop, replace:
        #   particles = euler_step(particles, drift, dλ, A, b)
        # with:
        #   particles = euler_step(particles, drift, dλ, A, b)
        #   particles += sqrt(q * dλ) * randn(...)  # SDE noise
        #
        # If self.relinearize is False:
        #   Fix H = model.observation_jacobian(η̄_0) once
        #   Fix eta_bar = η̄_0 throughout (don't update from flowing particles)
```

This approach:
- Eliminates ~200 lines of duplicate/divergent code
- Reuses the battle-tested `compute_flow_params` utility
- Makes the SDE extension minimal and transparent
- Preserves theoretical guarantees from the paper

---

## 7. StiffnessMitigationSolver Analysis

**Reference**: Dai & Daum, "Stiffness Mitigation in Stochastic Particle Flow Filters" (arXiv:2107.04672, 2021), Section 3.

### 7.1 What It Does

The solver finds an optimal schedule β(λ) that minimizes stiffness of the flow SDE. Instead of the linear schedule β=λ (where measurement information is incorporated uniformly), the optimal schedule reshapes the incorporation rate to reduce the condition number of the intermediate Hessian matrix M(β).

**Paper formulation** (Eqs. 23–25, Section 3):

The problem is an optimal control problem with α+β=1 normalization:
- State equation (Eq. 23): dβ/dλ = u(λ)
- Boundary conditions (Eq. 24): β(0)=0, β(1)=1
- Cost functional (Eq. 25): J = ∫₀¹ [½u² + μ·κ(M(β))] dλ

The Pontryagin maximum principle (Theorem 3.1, Eq. 26) gives the optimal β*:
$$\ddot{\beta}^* = \mu \cdot \frac{\partial \kappa(M)}{\partial \beta}\bigg|_{\beta=\beta^*}$$

with Dirichlet boundary conditions (Eq. 27): β*(0)=0, β*(1)=1.

### 7.2 Code Correctness — Cross-Check Against Paper

#### 7.2.1 M(β) definition — ✓ CORRECT (matches Eq. 17 with α+β=1)

```python
def _compute_M(self, beta):
    return self.J_prior + beta * self.J_meas
```

From Eq. 17: M(λ) = -(α+β)∇²log p₀ - β·∇²log h.

With the normalization α+β=1 (paper Section 3): M = -∇²log p₀ - β·∇²log h.

For Gaussian: -∇²log p₀ = P₀⁻¹ = J_prior, and -∇²log h = H^T R⁻¹ H = J_meas.

So M(β) = J_prior + β·J_meas ✓

#### 7.2.2 Condition number derivative — ✓ CORRECT (paper Eq. 28 has sign typo)

The code uses the nuclear norm condition number (Remark 3.2):

κ*(M) = tr(M)·tr(M⁻¹) (valid for positive definite M since ||M||* = tr(M))

The code computes:
$$\frac{\partial \kappa}{\partial \beta} = \text{tr}(M') \text{tr}(M^{-1}) - \text{tr}(M) \text{tr}(M^{-1} M' M^{-1})$$

where M' = ∂M/∂β = J_meas.

**Derivation verification using Lemma A.1** (d(A⁻¹)/dθ = -A⁻¹(dA/dθ)A⁻¹):
- κ = tr(M) · tr(M⁻¹)
- d/dβ tr(M) = tr(M') = tr(J_meas) ✓
- d/dβ M⁻¹ = -M⁻¹ M' M⁻¹ (Lemma A.1) ✓
- d/dβ tr(M⁻¹) = -tr(M⁻¹ M' M⁻¹) ✓
- Product rule: dκ/dβ = tr(M')·tr(M⁻¹) + tr(M)·(-tr(M⁻¹ M' M⁻¹)) ✓

**Sign verification against Eq. 28**: The paper writes (Remark 3.2):

> d²β*/dλ² = -μ[tr(∇²log h)·tr(M⁻¹) + tr(M)·tr(M⁻² ∇²log h)]

Converting to our notation with ∂M/∂β = -∇²log h = J_meas:
- tr(∇²log h) = -tr(J_meas)
- The correct calculus gives: dκ/dβ = -tr(∇²log h)·tr(M⁻¹) **+** tr(M)·tr(M⁻¹·∇²log h·M⁻¹)
- Which equals: tr(J_meas)·tr(M⁻¹) **-** tr(M)·tr(M⁻¹·J_meas·M⁻¹)

**The paper's Eq. 28 has a sign error**: it shows `+` between the two terms inside the brackets, but the correct sign from Lemma A.1 is `-`. The code is correct; the paper appears to have dropped the negative sign from the matrix inverse derivative.

#### 7.2.3 ODE dynamics — ✓ CORRECT (matches Eq. 26)

```python
def _ode_dynamics(self, lam, y, mu):
    beta, beta_dot = y
    beta_double_dot = mu * self._condition_number_derivative_nuclear(beta)
    return [beta_dot, beta_double_dot]
```

This is the first-order system form of Eq. 26: d²β/dλ² = μ·∂κ/∂β. ✓

#### 7.2.4 BVP shooting method — ✓ CORRECT (matches paper Section 4)

```python
# ODE: [β', β''] = [β_dot, μ * ∂κ/∂β]
# BC: β(0) = 0, β(1) = 1
# Shooting: vary β'(0) = u₀ until β(1) = 1
```

The paper (Section 4) states: "The simple bisection method, a special case of the shooting method, is used to find β̇*(0) such that β*(1) = 1."

The code:
1. Parameterizes by initial slope u₀ = β̇(0) — matches Eq. 27 BC
2. Integrates with RK45 from λ=0 to λ=1 — paper uses ode45
3. Uses Brent's method (a refinement of bisection) to find u₀ such that β(1)=1

✓ Consistent with paper's approach.

#### 7.2.5 μ=0 special case — ✓ CORRECT (matches Remark 3.1)

```python
if mu <= 1e-8:
    return lambda lam: (lam, 1.0)
```

Remark 3.1: "Consider a special case of μ=0... Its solution is the straight line β*=λ."

Code returns β=λ, β̇=1 ✓

#### 7.2.6 Spline interpolation — ✓ CORRECT

The solver returns a cubic spline interpolant of the computed β(λ), with its derivative β̇(λ). This provides smooth (β, β̇) values at arbitrary λ for use in the flow loop.

### 7.3 Issues Found

### 7.3 Refactored Implementation (TensorFlow, integrated)

The solver has been rewritten in TensorFlow and integrated directly into `StochasticEDHFlow` as class methods, controlled by the `schedule_mu` keyword:

- `schedule_mu = 0.0` → linear schedule β=λ (Remark 3.1)
- `schedule_mu > 0` → BVP solver runs at each `update()` call

**Key methods**:
- `_dkappa_dbeta()`: ∂κ/∂β using `tf.linalg.inv`, `tf.linalg.trace` (Remark 3.2 + Lemma A.1)
- `_rk4_schedule_step()`: RK4 for the 2D ODE system [β, β̇]
- `_shoot()`: integrate ODE from λ=0→1, return β(1) for shooting method
- `_compute_optimal_schedule()`: bisection shooting + final integration to get β at each λ step

**Integration with flow loop**: The optimal schedule reparameterizes the flow. Since A_opt(λ) = β̇(λ)·A_exact(β(λ)), using A_exact(β) with step size dβ = β̇·dλ is mathematically equivalent. The noise term still uses dλ (Brownian motion is in λ-time).

**Previous issues resolved**:
- Issue #7 (missing imports): eliminated — no scipy dependency, uses TF ops
- Issue #8 (beta clipping): removed — no clipping in new implementation
- Issue #9 (float cache): eliminated — no caching in new implementation
- Issue #10 (narrow bracket): improved — bracket auto-widens + fallback to linear schedule
- Issue #11 (not connected): resolved — fully integrated via `schedule_mu` keyword

### 7.4 Summary

The stiffness mitigation solver is **mathematically correct** when compared against the paper's Section 3, Theorem 3.1, and Remark 3.2. It is now fully integrated into `StochasticEDHFlow` using TensorFlow, controlled by `schedule_mu`. All previous issues (#7-#11) are resolved.

---

## References

- Dai, L. & Daum, F. (2021). "A New Parameterized Family of Stochastic Particle Flow Filters." arXiv:2103.09676v3.
- Dai, L. & Daum, F. (2021). "Stiffness Mitigation in Stochastic Particle Flow Filters." arXiv:2107.04672.
- Li, Y. & Coates, M. (2017). "Particle filtering with invertible particle flow." IEEE TSP.
- Daum, F. & Huang, J. (2010). "Exact particle flow for nonlinear filters." Proc. SPIE.
