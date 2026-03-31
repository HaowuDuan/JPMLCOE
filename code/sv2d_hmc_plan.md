# 2D Stochastic Volatility: HMC Parameter Estimation Plan

## Model Recap

**State:** `x_t = [x_1t, x_2t]`

**Transition:** `x_{t+1} = A x_t + e_t`, `e_t ~ N(0, Sigma)`
- Diagonal case: `A = diag(a1, a2)`, `Sigma = diag(sigma1^2, sigma2^2)`

**Observation:** `y_t = b * x_1t + exp(x_2t / 2) * v_t`, `v_t ~ N(0,1)`

**5 parameters:** `a1, a2, sigma1, sigma2, b`

---

## Critical Finding: Model Not HMC-Compatible (Must Fix First)

The current `StochasticVolatility2DModel` stores **pre-computed TF constants** (`self._A`, `self._Sigma`, `self._b`, `self._L_Sigma`, `self._P0`). The DPF pipeline updates parameters via `setattr(model, 'a1', tensor)`, but all model methods read from the pre-computed constants (e.g., `self._A`), **not** from `self.a1`.

**Result:** Gradients do NOT flow through the parameters. HMC would sample randomly.

The same issue exists in the 1D SV model (`_alpha_tf`, `_sigma_tf`, `_beta_tf`).

The `LinearGaussianModel` works because it reads `self.obs_noise_std` directly in its property methods.

### Required Refactoring

All model methods that are called during filtering must compute derived quantities from the raw scalar attributes (`self.a1`, `self.a2`, `self.sigma1`, `self.sigma2`, `self.b`) on the fly, not from pre-computed constants. This preserves the gradient chain.

**Example change for `state_transition_mean`:**
```python
# BEFORE (gradient-broken):
def state_transition_mean(self, x):
    return tf.linalg.matvec(self._A, x)  # _A is a tf.constant, no gradient

# AFTER (gradient-preserving):
def state_transition_mean(self, x):
    A = tf.stack([[self.a1, 0.0], [0.0, self.a2]])  # recomputed from live attrs
    return tf.linalg.matvec(A, x)
```

**Methods to refactor (critical for BPF HMC path):**
- `state_transition_batch` — uses `self._A`, `self._L_Sigma`
- `log_observation_prob_batch` — uses `self._b`
- `sample_initial_state_batch` — uses `self._L_P0`
- `state_transition_mean_batch` — uses `self._A`
- `observation_jacobian_batch` — uses `self._b`
- `observation_function_batch` — uses `self._b`

**For EKF/UKF path also refactor:** `state_transition_mean`, `state_jacobian`, `state_transition_cov`, `observation_mean`, `observation_jacobian`, `observation_function`, `observation_cov`, `log_observation_prob`, `process_noise_cov`.

---

## Initial Condition: Does It Update With Parameters?

### Current Behavior
- `Sigma_0 = P0` where `P0` solves the discrete Lyapunov equation `P0 = A P0 A^T + Sigma`
- Computed **once** at `__init__` via `scipy_linalg.solve_discrete_lyapunov` (NumPy, not differentiable)
- Particle filter samples initial particles **outside** the compiled filter loop, then passes them in
- **P0 is NOT recomputed when parameters change during HMC**

### Should It Be Updated?

**Theoretically yes** — when you change `a1` or `sigma1`, the stationary distribution changes, and the initial condition should reflect the proposed parameters. Otherwise the likelihood has a bias from the mismatched initial distribution.

**Practically:**
- For long time series (T >> 1), the effect of the initial condition washes out quickly
- For the bootstrap PF, particles are sampled once and then propagated — the initial sample is fixed
- Making P0 differentiable w.r.t. parameters requires solving Lyapunov in TF (non-trivial for general A)

### Diagonal Case: Closed-Form (Differentiable!)

For diagonal `A = diag(a1, a2)` and `Sigma = diag(s1^2, s2^2)`:
```
P0 = diag(s1^2 / (1 - a1^2), s2^2 / (1 - a2^2))
```
This is a simple formula that can be computed in TF with full gradient support.

### Recommendation
- **Phase 1 (toy):** Fix A and Sigma → P0 is constant → no issue
- **Phase 2+:** When inferring A or Sigma, add a property that recomputes P0 from current parameter values using the closed-form. This ensures correct gradients. For non-diagonal A, would need a TF-native Lyapunov solver (future work).

---

## Stationarity Constraint on A

For the system to have a stationary distribution, all eigenvalues of A must satisfy `|lambda| < 1`.

**Diagonal case:** `|a1| < 1` and `|a2| < 1`.

**For HMC:** Use interval constraint `(-1+eps, 1-eps)` or if assuming positive persistence, use `(0, 1)` which maps to the `unit` constraint (Sigmoid bijector).

**Typical financial models:** `a1, a2 in (0.8, 0.99)` — high persistence. A prior like `Beta(9, 1)` shifted to `(0, 1)` works well (peaked near 0.9).

**Non-diagonal case:** Constraint is on eigenvalues, not individual entries. Much harder to enforce via bijectors. Would need parameterizing A via its eigendecomposition or Schur form. **Defer to future work.**

---

## Prior Choices

| Parameter | Constraint | Bijector | Recommended Prior | Rationale |
|-----------|-----------|----------|-------------------|-----------|
| `a1` | `(0, 1)` or `unit` | Sigmoid | `Beta(9, 1)` or `Uniform(0, 1)` | Persistence ∈ (0,1), peaked near 0.9 for finance |
| `a2` | `(0, 1)` or `unit` | Sigmoid | `Beta(9, 1)` or `Uniform(0, 1)` | Same as a1 |
| `sigma1` | `positive` | Softplus | `LogNormal(loc=-0.7, scale=0.5)` | Centered near 0.5 (true value) |
| `sigma2` | `positive` | Softplus | `LogNormal(loc=0.0, scale=0.5)` | Centered near 1.0 (true value) |
| `b` | `positive` (if b>0) or unconstrained | Softplus / Identity | `LogNormal(0, 1)` or `Normal(0, 2)` | Depends on sign assumption |

**Note:** Priors should be weakly informative — broad enough to not dominate the likelihood, but concentrated enough to avoid numerical issues at extreme values.

---

## Can the Code Set Arbitrary Parameters as Learnable?

**Yes**, with caveats:
- Any parameter name listed in `dpf.trainable_params` in the config becomes learnable
- The `ParameterHandler` creates bijectors for each
- The `DifferentiableModel` wraps the model and updates those specific attributes via setattr
- Other parameters remain at their initial values

**BUT:** The model must read from the raw attribute (not pre-computed constants) for gradients to flow. This is the refactoring described above.

---

## Difficulty Ladder

### Level 0: Infer `b` only (fix A, Sigma)
**Why start here:** Simplest case. `b` only appears in observation equation. P0 doesn't depend on `b`. No stationarity constraint needed for `b`.

- True params: `a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0`
- Trainable: `b` only
- Init guess: `b=0.5` (away from true)
- Prior: `LogNormal(0, 1)` (positive, weakly informative)
- Filter: `BootstrapPFHMC` with systematic resampling + stop_gradient
- Model refactoring needed: only `self._b` → `self.b` in obs-related methods

### Level 1: Infer `sigma2` only (fix A, b, sigma1)
**Why:** `sigma2` controls log-volatility noise — directly affects observation variance. P0 depends on `sigma2` but we can fix P0 for now.

- Trainable: `sigma2`
- Init guess: `sigma2=2.0`
- Prior: `LogNormal(0, 0.5)`
- Constraint: `positive`
- Model refactoring: `self._Sigma` → computed from `self.sigma1`, `self.sigma2`
- Also need `self._L_Sigma` recomputed (Cholesky of diagonal = just sqrt of diagonal entries)

### Level 2: Infer `a2` only (fix sigma, b, a1)
**Why:** Tests transition parameter inference. Must handle stationarity constraint.

- Trainable: `a2`
- Init guess: `a2=0.7`
- Prior: `Beta(9, 1)` mapped to `(0, 1)` via `unit` constraint
- Model refactoring: `self._A` → computed from `self.a1`, `self.a2`
- P0 now depends on `a2` — should recompute for correctness

### Level 3: Infer `a2` + `sigma2` (2 parameters)
**Why:** Both transition parameters from the log-volatility component.

- Trainable: `a2, sigma2`
- P0 depends on both — use closed-form diagonal Lyapunov

### Level 4: Infer all 5 parameters
**Why:** Full inference. Hardest.

- Trainable: `a1, a2, sigma1, sigma2, b`
- Need all model refactoring complete
- P0 recomputed from all parameters
- May need NUTS (adaptive leapfrog) instead of fixed-step HMC
- Mass matrix preconditioning likely needed

---

## Action Items

### Phase 1: Model Refactoring (REQUIRED)

1. **Refactor `StochasticVolatility2DModel`** so all methods compute derived quantities from `self.a1`, `self.a2`, `self.sigma1`, `self.sigma2`, `self.b` on the fly
   - Keep pre-computed constants for non-HMC usage (data generation), but all filter-facing methods must be parameter-aware
   - For diagonal case, no matrix operations needed: `A @ x = [a1*x1, a2*x2]`
   - `L_Sigma` for diagonal case: `diag(sigma1, sigma2)` (trivial)
   - `P0` for diagonal case: `diag(sigma1^2/(1-a1^2), sigma2^2/(1-a2^2))`

2. **Ensure `sample_initial_state_batch` uses recomputed P0** or accepts P0 as argument

3. **Unit test:** Verify gradients flow through model parameters using `tf.GradientTape` + finite difference comparison

### Phase 2: Config Files

4. **Create MAP configs** for each difficulty level:
   - `code/configs/dpf/map/stochastic_volatility_2d/bpf_b_only.yaml` (Level 0)
   - `code/configs/dpf/map/stochastic_volatility_2d/bpf_sigma2.yaml` (Level 1)
   - `code/configs/dpf/map/stochastic_volatility_2d/bpf_a2.yaml` (Level 2)

5. **Create HMC configs** for each difficulty level:
   - `code/configs/dpf/hmc/stochastic_volatility_2d/bpf_b_only.yaml` (Level 0)
   - `code/configs/dpf/hmc/stochastic_volatility_2d/bpf_sigma2.yaml` (Level 1)
   - `code/configs/dpf/hmc/stochastic_volatility_2d/bpf_a2.yaml` (Level 2)

### Phase 3: Run and Validate

6. **MAP first** — faster iteration, verify loss decreases toward true parameter
7. **HMC second** — verify posterior concentrates around true value
8. **Check diagnostics:** acceptance rate, ESS, R-hat, trace plots
9. **Scale up:** combine parameters (Level 3 → Level 4)

### Phase 4: Generalization (Future)

10. Non-diagonal A, Sigma (eigenvalue-based stationarity constraint)
11. EKF-based likelihood (instead of BPF) — needs `observation_cov_corrected` to be differentiable
12. LEDH flow filter for better gradient signal

---

## Suggested True Parameters for Experiments

```
a1    = 0.95    # high persistence for level
a2    = 0.91    # moderate persistence for log-vol
sigma1 = 0.5   # moderate level noise
sigma2 = 1.0   # substantial vol-of-vol
b     = 1.0    # unit observation coefficient
T     = 200    # time series length (longer = more info)
seed  = 42
```

These match the existing model config defaults. For HMC, use:
- `n_particles = 1000` (BPF)
- `num_samples = 500, num_burnin = 200`
- `step_size = 0.001` (will adapt)
- `num_leapfrog_steps = 5`

---

## Open Questions

1. **Should the 1D SV model also be refactored?** Same pre-computed constant issue. The existing 1D SV HMC configs (`dpf/hmc/stochastic_volatility/`) likely have broken gradients.

2. **EKF likelihood for 2D SV?** The EKF can compute `log p(y|theta)` analytically (no particle noise). But `observation_cov(x)` is state-dependent, requiring the corrected EKF. Worth investigating as a smoother likelihood surface for HMC.

3. **Particle count vs gradient quality:** BPF log-likelihood is noisy. More particles = smoother gradients but slower. 1000 particles is a reasonable start.
