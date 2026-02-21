# Test Plan: src/utils/ — Shared Utilities

All test files go directly under `code/tests/`, one per utils module.

## Setup

```python
# conftest.py already exists at tests/ level.
# Each test file imports what it needs directly.
import numpy as np
import tensorflow as tf
```

---

## File 1: `test_utils_linalg.py`  →  `src/utils/linalg.py`

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

### `graph_safe_inv` (NEW — GPU graph-safe SVD-based inverse)

**T1.18 — Matches safe_inv for well-conditioned PD matrix**
```python
A = [[4.0, 1.0], [1.0, 3.0]]
A_inv_safe = safe_inv(A)
A_inv_graph = graph_safe_inv(A)
assert_allclose(A_inv_graph, A_inv_safe, rtol=1e-5)
```

**T1.19 — Near-singular: returns finite result**
```python
A = eye(2) * 1e-8
A_inv = graph_safe_inv(A)
assert np.all(np.isfinite(A_inv.numpy()))
```

---

### `graph_safe_log_abs_det` (NEW — slogdet forward + pinv backward)

**T1.20 — Forward matches safe_log_abs_det for regular matrix**
```python
A = [[3.0, 1.0], [1.0, 4.0]]
result_safe = safe_log_abs_det(A, jitter=0.0)
result_graph = graph_safe_log_abs_det(A, jitter=0.0)
assert_allclose(result_graph, result_safe, rtol=1e-5)
```

**T1.21 — Gradient is finite (custom gradient backward pass works)**
```python
A = tf.Variable([[3.0, 1.0], [1.0, 4.0]], dtype=tf.float64)
with tf.GradientTape() as tape:
    y = graph_safe_log_abs_det(A, jitter=1e-8)
grad = tape.gradient(y, A)
assert np.all(np.isfinite(grad.numpy()))
# Gradient of log|det(A)| = A^{-T}, verify shape and sign
assert grad.shape == (2, 2)
```

---

### `graph_safe_log_abs_det_fast` (NEW — NaN-guarded fast backward)

**T1.22 — Forward matches safe_log_abs_det for regular matrix**
```python
A = [[3.0, 1.0], [1.0, 4.0]]
result = graph_safe_log_abs_det_fast(A, jitter=0.0)
expected = safe_log_abs_det(A, jitter=0.0)
assert_allclose(result, expected, rtol=1e-5)
```

**T1.23 — NaN input: gradient is zero (NaN guard fires)**

The NaN guard replaces NaN matrices with identity before calling inv,
then zeros out the gradient. This is designed for HMC where diverged
leapfrog proposals produce NaN Jacobians — those proposals get rejected
anyway, so zero gradient is correct.

```python
A_nan = tf.Variable([[float('nan'), 0.0], [0.0, 1.0]], dtype=tf.float64)
with tf.GradientTape() as tape:
    y = graph_safe_log_abs_det_fast(A_nan, jitter=1e-8)
grad = tape.gradient(y, A_nan)
assert np.all(grad.numpy() == 0.0)  # NaN input → zero gradient
```

---

### `graph_safe_log_abs_det_svd` (NEW — pure SVD log|det|)

**T1.24 — Matches slogdet for regular PD matrix**
```python
A = [[3.0, 1.0], [1.0, 4.0]]
_, expected = tf.linalg.slogdet(A)
result = graph_safe_log_abs_det_svd(A, jitter=0.0)
assert_allclose(result, expected, rtol=1e-5)
```

**T1.25 — Handles negative determinant via SVD absolute value**
```python
A = [[-1.0, 0.0], [0.0, 2.0]]   # det = -2
# SVD computes sum(log(singular_values)), singular values are always positive
result = graph_safe_log_abs_det_svd(A)
assert_allclose(result, np.log(2.0), rtol=1e-4)
```

---

### `to_numpy` (NEW — trivial converter)

**T1.26 — Converts TF tensor to numpy**
```python
t = tf.constant([1.0, 2.0])
assert isinstance(to_numpy(t), np.ndarray)
```

**T1.27 — Passthrough for numpy array and scalar**
```python
a = np.array([1.0])
assert to_numpy(a) is a  # same object
assert to_numpy(3.14) == 3.14
```

---

## File 2: `test_utils_ode_solvers.py`  →  `src/utils/ode_solvers.py`

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
result = euler_step(particles, drift_fn, dt, ...)
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
exact = np.exp(-0.5)
euler_err = abs(euler_step(x0, f, dt, ...) - exact)
rk4_err   = abs(rk4_step(x0, f, dt, ...) - exact)
assert rk4_err < euler_err
```

**T2.5 — Time-dependent drift: x'(t) = -t (exact: x(t) = x0 - t^2/2)**
```python
def f_t(x, t):
    return -t * tf.ones_like(x)
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
x_final = integrate_ode(x0=..., f=f_decay, t_span=(0, 1), n_steps=1000, method='euler')
assert_allclose(x_final, np.exp(-1.0), rtol=1e-2)
```

**T2.9 — RK4: same ODE, tighter tolerance**
```python
x_final = integrate_ode(x0=..., f=f_decay, t_span=(0, 1), n_steps=100, method='rk4')
assert_allclose(x_final, np.exp(-1.0), rtol=1e-6)
```

---

## File 3: `test_utils_distributions.py`  →  `src/utils/distributions.py`

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

### `sample_particles_cholesky` (NEW — particle sampling via Cholesky)

**T3.11 — Output shape is (N, d)**
```python
samples = sample_particles_cholesky(mean, cov, n_particles=100, state_dim=2, seed=[0,0])
assert samples.shape == (100, 2)
```

**T3.12 — Sample mean approaches distribution mean (Monte Carlo)**
```python
N = 50000
mean = [3.0, -2.0];  cov = 2.0 * eye(2)
samples = sample_particles_cholesky(mean_tf, cov_tf, N, 2, seed=[0,0])
assert_allclose(samples.numpy().mean(axis=0), mean, atol=0.05)
```

---

### `compute_flow_weights`

**T3.13 — Identity flow (eta_1 == eta_0): weights sum to 1**

When eta_1 == eta_0, numerator and denominator transition terms cancel.
```python
eta_0 = eta_1 = particles_prev  # no flow
weights = compute_flow_weights(eta_1, eta_0, particles_prev, obs, model)
assert_allclose(tf.reduce_sum(weights), 1.0, rtol=1e-5)
```

**T3.14 — Uniform prior weights → normalized weights sum to 1**
```python
weights = compute_flow_weights(eta_1, eta_0, particles_prev, obs, model,
                               prev_weights=None, jacobians=None)
assert_allclose(tf.reduce_sum(weights), 1.0, rtol=1e-6)
assert np.all(weights.numpy() >= 0)
```

**T3.15 — NaN inputs get zero weight (fallback to uniform)**
```python
eta_1_bad = tf.concat([eta_1[:1] * float('nan'), eta_1[1:]], axis=0)
weights = compute_flow_weights(eta_1_bad, eta_0, ...)
assert np.all(np.isfinite(weights.numpy()))
```

---

## File 4: `test_utils_flow_params.py`  →  `src/utils/flow_params.py`

### `compute_flow_params_global`

**T4.1 — A formula: A = -0.5 * P @ H^T @ (λ*HPH^T + R)^{-1} @ H**
```python
P = diag([4.0, 1.0]);  H = [[1.0, 0.0]];  R = [[0.1]]
lam = 0.5
A_code, _ = compute_flow_params_global(H, lam, z, P, R, R_inv, eta_bar, 2)
S = lam * H @ P @ H.T + R
A_expected = -0.5 * P @ H.T @ inv(S) @ H
assert_allclose(A_code, A_expected, rtol=1e-5)
```

**T4.2 — b formula at λ=0**

At λ=0: I+0A = I, so b = P@H^T@R^{-1}@z + A@η̄₀
```python
lam = 0.0
A_code, b_code = compute_flow_params_global(...)
b_expected = P @ H.T @ R_inv @ z + A_code @ eta_bar
assert_allclose(b_code, b_expected, rtol=1e-5)
```

**T4.3 — Deterministic flow converges to posterior mean**

For 1D linear-Gaussian: prior N(0, σ_p²), likelihood N(z; x, σ_r²).
Posterior mean = σ_p² z / (σ_p² + σ_r²).
Run 200 Euler steps of dx/dt = Ax + b from prior mean → should reach posterior mean.
```python
P = [[100.0]];  R = [[1.0]];  H = [[1.0]]
posterior_mean = 100.0 * 5.0 / 101.0  # ≈ 4.95
# Euler integration...
assert_allclose(x_final, posterior_mean, rtol=1e-2)
```

**T4.4 — A has non-positive eigenvalues (stable flow)**
```python
eigvals = np.linalg.eigvals(A_code.numpy())
assert np.all(eigvals.real <= 1e-10)
```

---

### `compute_flow_params` (local, with e-correction)

**T4.5 — Matches global when h(x) = H@x (linear model, e=0)**
```python
A_local, b_local   = compute_flow_params(model, eta_bar, lam, z, P, R, R_inv, eta_bar, d)
A_global, b_global = compute_flow_params_global(H, lam, z, P, R, R_inv, eta_bar, d)
assert_allclose(A_local, A_global, rtol=1e-5)
assert_allclose(b_local, b_global, rtol=1e-5)
```

---

### `compute_flow_params_batch` (NEW — batched local flow params)

**T4.6 — Matches per-particle loop of compute_flow_params**

Run compute_flow_params for each particle individually, then run
compute_flow_params_batch for all particles at once. Results must match.
```python
for i in range(N):
    A_i, b_i = compute_flow_params(model, particles[i], lam, z, P, R, R_inv, eta_bar, d)
A_batch, b_batch = compute_flow_params_batch(model, particles, lam, z, P, R, R_inv, eta_bar, d)
assert_allclose(A_batch[i], A_i, rtol=1e-5)
assert_allclose(b_batch[i], b_i, rtol=1e-5)
```

---

### `compute_flow_params_batch_global` (NEW — batched global flow params)

**T4.7 — Matches per-particle loop of compute_flow_params_global**

Same strategy as T4.6 but with the global (no e-correction) variant.
```python
H_batch = model.observation_jacobian_batch(particles)
A_batch, b_batch = compute_flow_params_batch_global(H_batch, lam, z, P, R, R_inv, eta_bar, d)
# Compare to single-particle calls
for i in range(N):
    A_i, b_i = compute_flow_params_global(H_batch[i], lam, z, P, R, R_inv, eta_bar, d)
    assert_allclose(A_batch[i], A_i, rtol=1e-5)
```

**T4.8 — All A matrices have non-positive eigenvalues**
```python
for i in range(N):
    eigvals = np.linalg.eigvals(A_batch[i].numpy())
    assert np.all(eigvals.real <= 1e-10)
```

---

## Running All Tests

```bash
cd /Users/haowuduan/Documents/githubrepos/JPMLCOE/code

# Run all utils tests:
.venv/bin/python -m pytest tests/test_utils_*.py -v --tb=short

# Save results to file:
.venv/bin/python -m pytest tests/test_utils_*.py -v 2>&1 | tee tests/utils_test_results.txt
```

## Test File Structure

```
code/tests/
├── test_utils_linalg.py         # T1.1 – T1.27
├── test_utils_ode_solvers.py    # T2.1 – T2.9
├── test_utils_distributions.py  # T3.1 – T3.15
└── test_utils_flow_params.py    # T4.1 – T4.8
```
