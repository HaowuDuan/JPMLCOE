# Test Plan: src/utils/ — Shared Utilities

All tests go in `code/tests/utils/` as separate files per module.

## Setup

```python
# conftest.py in tests/utils/
import pytest
import numpy as np
import tensorflow as tf

DTYPES = [tf.float32, tf.float64]

@pytest.fixture(params=[tf.float32, tf.float64])
def dtype(request):
    return request.param
```

---

## File 1: `test_linalg.py`  →  `src/utils/linalg.py`

### `safe_cholesky`

**T1.1 — Well-conditioned matrix: L @ L^T == A**
```python
A = diag([4.0, 9.0, 16.0])
L = safe_cholesky(A, jitter=0.0, adaptive=False)
assert_allclose(L @ L.T, A, rtol=1e-5)
```
Expected: L = diag([2, 3, 4])

**T1.2 — Adaptive jitter scales with matrix magnitude**
```python
A_small = diag([1e-4, 1e-4])   # avg_diag = 1e-4 < 1.0 → uses max(avg, 1.0)=1.0
A_large = diag([1e4, 1e4])     # avg_diag = 1e4 → jitter = 1e-10 * 1e4 = 1e-6
# Both must succeed without exception
```

**T1.3 — Near-singular matrix: recovers without NaN**
```python
A = [[1.0, 1.0], [1.0, 1.0]]   # rank-1, not PD
L = safe_cholesky(A, jitter=1e-6, adaptive=False)
assert not np.any(np.isnan(L.numpy()))
assert np.all(np.isfinite(L.numpy()))
```

**T1.4 — Batch input (3D tensor)**
```python
A_batch = tf.stack([diag([1.0, 2.0]), diag([3.0, 4.0])])  # (2, 2, 2)
L_batch = safe_cholesky(A_batch)
assert L_batch.shape == (2, 2, 2)
```

---

### `safe_solve`

**T1.5 — Vector RHS: A @ x = b**
```python
A = [[2.0, 0.0], [0.0, 4.0]]
b = [6.0, 8.0]
x = safe_solve(A, b)
assert_allclose(x, [3.0, 2.0], rtol=1e-5)
```

**T1.6 — Matrix RHS: A @ X = B**
```python
A = diag([2.0, 4.0])
B = [[1.0, 0.0], [0.0, 1.0]]
X = safe_solve(A, B)
assert_allclose(X, diag([0.5, 0.25]), rtol=1e-5)
```

**T1.7 — Cholesky method matches default for PD input**
```python
A = [[4.0, 1.0], [1.0, 3.0]]  # PD
b = [5.0, 7.0]
x_default  = safe_solve(A, b, method='default')
x_cholesky = safe_solve(A, b, method='cholesky')
assert_allclose(x_default, x_cholesky, rtol=1e-4)
```

**T1.8 — Output shape: vector in → vector out; matrix in → matrix out**
```python
x = safe_solve(A_2x2, b_2)     # → shape (2,)
X = safe_solve(A_2x2, B_2x3)   # → shape (2, 3)
```

---

### `symmetrize`

**T1.9 — Asymmetric input → symmetric output**
```python
A = [[1.0, 2.0], [4.0, 3.0]]
S = symmetrize(A)
assert_allclose(S, [[1.0, 3.0], [3.0, 3.0]], rtol=1e-6)
assert_allclose(S, S.T, atol=1e-10)  # exactly symmetric
```

**T1.10 — Already symmetric → unchanged**
```python
S = diag([5.0, 7.0])
assert_allclose(symmetrize(S), S, rtol=1e-10)
```

---

### `log_det`

**T1.11 — Known determinant**
```python
A = diag([2.0, 3.0, 5.0])   # det = 30
ld = log_det(A)
assert_allclose(ld, np.log(30.0), rtol=1e-5)
```

---

### `safe_inv`

**T1.12 — A @ safe_inv(A) ≈ I**
```python
A = [[4.0, 1.0], [1.0, 3.0]]
assert_allclose(A @ safe_inv(A), eye(2), atol=1e-5)
```

**T1.13 — Near-singular: returns finite result (no NaN)**
```python
A = eye(2) * 1e-8   # very small, near-singular
A_inv = safe_inv(A)
assert np.all(np.isfinite(A_inv.numpy()))
```

---

### `safe_log_abs_det`

**T1.14 — Matches slogdet for regular matrix**
```python
A = [[3.0, 1.0], [1.0, 4.0]]
_, expected = tf.linalg.slogdet(A)
result = safe_log_abs_det(A, jitter=0.0)
assert_allclose(result, expected, rtol=1e-5)
```

**T1.15 — Handles negative determinant (takes absolute value)**
```python
A = [[-1.0, 0.0], [0.0, 2.0]]   # det = -2, log|det| = log(2)
result = safe_log_abs_det(A)
assert_allclose(result, np.log(2.0), rtol=1e-5)
```

---

### `matrix_sqrt`

**T1.16 — Cholesky: sqrt(A) @ sqrt(A)^T == A**
```python
A = [[4.0, 2.0], [2.0, 3.0]]   # PD
S = matrix_sqrt(A, method='cholesky')
assert_allclose(S @ S.T, A, rtol=1e-5)
```

**T1.17 — Eig: S @ S^T == A and S is symmetric**
```python
S = matrix_sqrt(A, method='eig')
assert_allclose(S @ S.T, A, atol=1e-5)
assert_allclose(S, S.T, atol=1e-6)   # eig sqrt is symmetric
```

---

## File 2: `test_ode_solvers.py`  →  `src/utils/ode_solvers.py`

### `euler_step`

**T2.1 — Scalar linear ODE x' = -x, exact solution x(dt) = x0 * exp(-dt)**
```python
def f(x, a):
    return a * x
x0 = tf.constant([2.0], dtype=tf.float64)
dt = 0.01
x1 = euler_step(x0, f, dt, tf.constant(-1.0, dtype=tf.float64))
expected = 2.0 * np.exp(-0.01)
assert_allclose(x1.numpy(), [expected], rtol=dt)  # O(dt) error
```

**T2.2 — Batch particles: (N, d) → (N, d)**
```python
particles = tf.ones([50, 2], dtype=tf.float64)
A = -0.1 * tf.eye(2, dtype=tf.float64)
b = tf.zeros(2, dtype=tf.float64)
result = euler_step(particles, drift_fn, dt, A, b)
assert result.shape == (50, 2)
```

**T2.3 — Zero drift: state unchanged**
```python
x = tf.constant([3.0, -1.0], dtype=tf.float64)
result = euler_step(x, lambda x: tf.zeros_like(x), 0.1)
assert_allclose(result.numpy(), x.numpy())
```

---

### `rk4_step`

**T2.4 — RK4 is more accurate than Euler for x' = -x with large dt**
```python
dt = 0.5
x0 = tf.constant([1.0], dtype=tf.float64)
x_euler = euler_step(x0, f, dt, ...)
x_rk4   = rk4_step(x0, f, dt, ...)
exact    = np.exp(-0.5)
euler_err = abs(x_euler - exact)
rk4_err   = abs(x_rk4 - exact)
assert rk4_err < euler_err
```

**T2.5 — Time-dependent drift: x'(t) = -t (exact: x(t) = x0 - t^2/2)**
```python
def f_t(x, t):
    return -t
x0 = tf.constant([5.0], dtype=tf.float64)
x_rk4 = rk4_step(x0, f_t, dt=1.0, t=0.0)
expected = 5.0 - 0.5
assert_allclose(x_rk4, [expected], rtol=1e-3)
```

---

### `euler_maruyama_step`

**T2.6 — Zero diffusion: reduces to deterministic Euler**
```python
result = euler_maruyama_step(x0, f, dt, diffusion_coeff=0.0, seed=None)
assert_allclose(result, euler_step(x0, f, dt))
```

**T2.7 — Noise has correct variance scaling: Var[noise] = diffusion_coeff * dt**
```python
N = 10000
results = [euler_maruyama_step(x0, zero_drift, dt=1.0,
                               diffusion_coeff=4.0, seed=[i,0]).numpy()
           for i in range(N)]
# Empirical variance ≈ 4.0
assert_allclose(np.var(results), 4.0, rtol=0.1)
```

---

### `integrate_ode`

**T2.8 — Euler: solution of x' = -x reaches correct endpoint**
```python
x_final = integrate_ode(x0=1.0, f=f_decay, t_span=(0, 1), n_steps=1000, method='euler')
assert_allclose(x_final, np.exp(-1.0), rtol=1e-2)
```

**T2.9 — RK4: same ODE, tighter tolerance**
```python
x_final = integrate_ode(x0=1.0, f=f_decay, t_span=(0, 1), n_steps=100, method='rk4')
assert_allclose(x_final, np.exp(-1.0), rtol=1e-6)
```

---

## File 3: `test_distributions.py`  →  `src/utils/distributions.py`

### `log_gaussian_prob`

**T3.1 — 1D known value: x=0, μ=0, σ²=1 → log p = -0.5*log(2π)**
```python
x    = tf.constant([0.0], dtype=tf.float64)
mean = tf.constant([0.0], dtype=tf.float64)
cov  = tf.eye(1, dtype=tf.float64)
logp = log_gaussian_prob(x, mean, cov)
assert_allclose(logp, -0.5 * np.log(2 * np.pi), rtol=1e-6)
```

**T3.2 — 2D diagonal covariance: value matches scipy**
```python
from scipy.stats import multivariate_normal
x  = [1.0, 2.0];  mean = [0.0, 0.0];  cov = np.diag([2.0, 3.0])
expected = multivariate_normal.logpdf(x, mean, cov)
result   = log_gaussian_prob(...)
assert_allclose(result, expected, rtol=1e-5)
```

**T3.3 — Batch: (batch, n) input → (batch,) output**
```python
x_batch = tf.constant([[0.0, 0.0], [1.0, 1.0]], dtype=tf.float64)  # (2, 2)
logp = log_gaussian_prob(x_batch, mean_batch, cov_broadcast)
assert logp.shape == (2,)
```

---

### `log_sum_exp`

**T3.4 — Numerically stable with large values**
```python
log_vals = tf.constant([1000.0, 1001.0, 1002.0], dtype=tf.float64)
# Naive exp would overflow; log_sum_exp must handle this
result = log_sum_exp(log_vals)
expected = 1002.0 + np.log(1 + np.exp(-1) + np.exp(-2))
assert_allclose(result, expected, rtol=1e-6)
```

**T3.5 — Single element: log_sum_exp([a]) == a**
```python
assert_allclose(log_sum_exp([5.0]), 5.0)
```

---

### `normalize_log_weights`

**T3.6 — Output sums to 1**
```python
log_w = tf.constant([-1.0, -2.0, -0.5, -3.0], dtype=tf.float64)
w = normalize_log_weights(log_w)
assert_allclose(tf.reduce_sum(w).numpy(), 1.0, rtol=1e-6)
assert np.all(w.numpy() >= 0)
```

**T3.7 — Uniform input → uniform output**
```python
log_w = tf.zeros(5, dtype=tf.float64)
w = normalize_log_weights(log_w)
assert_allclose(w.numpy(), np.ones(5) / 5, rtol=1e-6)
```

**T3.8 — Clip range prevents overflow**
```python
log_w = tf.constant([1e6, 0.0, -1e6], dtype=tf.float64)
w = normalize_log_weights(log_w, clip_range=(-30.0, 30.0))
assert np.all(np.isfinite(w.numpy()))
assert_allclose(tf.reduce_sum(w).numpy(), 1.0, rtol=1e-6)
```

---

### `multivariate_normal_sample`

**T3.9 — Sample mean and covariance match distribution (Monte Carlo)**
```python
N = 50000
mean = [2.0, -1.0];  cov = [[3.0, 1.0], [1.0, 2.0]]
samples = multivariate_normal_sample(mean_tf, cov_tf, N, seed=[0,0])
assert_allclose(samples.numpy().mean(axis=0), mean, atol=0.05)
assert_allclose(np.cov(samples.numpy().T), cov, rtol=0.05)
```

**T3.10 — Output shape is (n_samples, d)**
```python
samples = multivariate_normal_sample(mean_2d, cov_2x2, 100, seed=[0,0])
assert samples.shape == (100, 2)
```

---

### `compute_flow_weights`

**T3.11 — Identity flow (eta_1 == eta_0): weight reduces to likelihood**

When eta_1 == eta_0, numerator and denominator transition terms cancel.
Weights should be proportional to p(z | eta_1).

```python
eta_0 = eta_1 = particles_prev  # no flow
weights = compute_flow_weights(eta_1, eta_0, particles_prev, obs, model)
assert_allclose(tf.reduce_sum(weights), 1.0, rtol=1e-5)
```

**T3.12 — Uniform prior weights → normalized weights sum to 1**
```python
weights = compute_flow_weights(eta_1, eta_0, particles_prev, obs, model,
                               prev_weights=None, jacobians=None)
assert_allclose(tf.reduce_sum(weights), 1.0, rtol=1e-6)
assert np.all(weights.numpy() >= 0)
```

**T3.13 — NaN inputs get zero weight (fallback to uniform)**

When some particles produce NaN log-likelihood, weights must still be finite.
```python
eta_1_bad = tf.concat([eta_1[:1] * float('nan'), eta_1[1:]], axis=0)
weights = compute_flow_weights(eta_1_bad, eta_0, ...)
assert np.all(np.isfinite(weights.numpy()))
```

---

## File 4: `test_flow_params.py`  →  `src/utils/flow_params.py`

### `compute_flow_params_global`

**T4.1 — A formula: A = -0.5 * P @ H^T @ (λ*HPH^T + R)^{-1} @ H**
```python
# Compute manually and compare
P = diag([4.0, 1.0]);  H = [[1.0, 0.0]];  R = [[0.1]]
R_inv = [[10.0]];  lam = 0.5
A_code, _ = compute_flow_params_global(H_tf, lam_tf, z_tf, P_tf, R_tf, R_inv_tf, eta_tf, 2)
HPH = H @ P @ H.T
S = lam * HPH + R
A_expected = -0.5 * P @ H.T @ inv(S) @ H
assert_allclose(A_code, A_expected, rtol=1e-5)
```

**T4.2 — b formula at λ=0: b = A @ η̄₀  (since I+0*A = I, I+0*A = I)**
```python
lam = 0.0
A_code, b_code = compute_flow_params_global(...)
# At lambda=0: A = -0.5*P@H^T@R^{-1}@H (independent of lambda)
# b = (I+0)[](I+0)@P@H^T@R^{-1}@z + A@eta] = P@H^T@R^{-1}@z + A@eta
b_expected = P @ H.T @ R_inv @ z + A_code.numpy() @ eta_bar
assert_allclose(b_code, b_expected, rtol=1e-5)
```

**T4.3 — Deterministic flow converges to posterior mean**

For linear-Gaussian model with known posterior:
- Prior: N(0, sigma_p^2)
- Likelihood: N(z; x, sigma_r^2)
- Posterior mean: sigma_p^2 * z / (sigma_p^2 + sigma_r^2)

Run 200 Euler steps of dx/dt = Ax + b with dt=1/200 from single particle at prior mean.
Final particle position must be ≈ posterior mean.

```python
# 1D case: state_dim=1, obs_dim=1
P = [[100.0]];  R = [[1.0]];  H = [[1.0]]
z = tf.constant([5.0], dtype=tf.float64)
eta_bar = tf.constant([0.0], dtype=tf.float64)
posterior_mean = 100.0 * 5.0 / (100.0 + 1.0)  # ≈ 4.95

x = tf.constant([[0.0]], dtype=tf.float64)  # start at prior mean
for i in range(200):
    lam = (i + 1) / 200.0
    A, b = compute_flow_params_global(H_tf, lam_tf, z, P_tf, R_tf, R_inv_tf, eta_bar, 1)
    x = x + (x @ A.T + b) * (1.0/200.0)

assert_allclose(x.numpy()[0, 0], posterior_mean, rtol=1e-2)
```

**T4.4 — A has non-positive eigenvalues (stable flow)**
```python
eigvals = np.linalg.eigvals(A_code.numpy())
assert np.all(eigvals.real <= 1e-10), f"Positive A eigenvalues: {eigvals}"
```

**T4.5 — `compute_flow_params` (local, with e-correction) vs global agree when H is linear**

For linear observation h(x) = H @ x, the e-correction e = h(x) - H@x = 0.
Both functions must return same A and b.

```python
# Use LinearGaussian model where h(x) = H @ x exactly
A_local, b_local   = compute_flow_params(model, eta_bar, lam, z, P, R, R_inv, eta_bar, d)
A_global, b_global = compute_flow_params_global(H, lam, z, P, R, R_inv, eta_bar, d)
assert_allclose(A_local, A_global, rtol=1e-5)
assert_allclose(b_local, b_global, rtol=1e-5)
```

---

## Running All Tests

```bash
cd /Users/haowuduan/Documents/githubrepos/JPMLCOE/code
python -m pytest tests/utils/ -v --tb=short
```

Float64 variants of all tests run by default. Float32 tests via the `dtype` fixture where applicable.

## Test File Structure

```
code/tests/utils/
├── __init__.py
├── conftest.py         # dtype fixture, shared model helpers
├── test_linalg.py      # T1.1 – T1.17
├── test_ode_solvers.py # T2.1 – T2.9
├── test_distributions.py # T3.1 – T3.13
└── test_flow_params.py # T4.1 – T4.5
```
