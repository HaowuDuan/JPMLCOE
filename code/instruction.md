# TensorFlow & TensorFlow Probability API Reference

A comprehensive guide to every TensorFlow (`tf`) and TensorFlow Probability (`tfp`) function used in this codebase. Each entry includes what the function does, why it is used, and a real code snippet from this project.

---

## Table of Contents

1. [What Are TensorFlow and TensorFlow Probability?](#1-what-are-tensorflow-and-tensorflow-probability)
2. [Tensor Creation](#2-tensor-creation)
3. [Data Types](#3-data-types)
4. [Shape Manipulation](#4-shape-manipulation)
5. [Math and Element-wise Operations](#5-math-and-element-wise-operations)
6. [Reductions (Aggregating Values)](#6-reductions-aggregating-values)
7. [Clipping and Conditionals](#7-clipping-and-conditionals)
8. [Linear Algebra](#8-linear-algebra)
9. [Random Number Generation](#9-random-number-generation)
10. [Indexing and Selection](#10-indexing-and-selection)
11. [Iteration and Loops](#11-iteration-and-loops)
12. [Graph Compilation with @tf.function](#12-graph-compilation-with-tffunction)
13. [Custom Gradients](#13-custom-gradients)
14. [Automatic Differentiation (Gradient Tape)](#14-automatic-differentiation-gradient-tape)
15. [Debugging Tools](#15-debugging-tools)
16. [Device Configuration (GPU/CPU)](#16-device-configuration-gpucpu)
17. [NumPy Interop](#17-numpy-interop)
18. [TFP Distributions](#18-tfp-distributions)
19. [TFP Bijectors (Parameter Transforms)](#19-tfp-bijectors-parameter-transforms)
20. [TFP MCMC (Markov Chain Monte Carlo)](#20-tfp-mcmc-markov-chain-monte-carlo)

---

## 1. What Are TensorFlow and TensorFlow Probability?

**TensorFlow (TF)** is a numerical computing library. Think of it as a GPU-accelerated replacement for NumPy. The core object is a **tensor** -- a multi-dimensional array of numbers (like a NumPy `ndarray`). TensorFlow can run computations on CPU, NVIDIA GPU (CUDA), or Apple Silicon GPU (Metal/MPS).

**TensorFlow Probability (TFP)** is a library built on top of TensorFlow for probabilistic modeling. It provides probability distributions (Gaussian, LogNormal, etc.), bijectors (parameter transformations), and MCMC samplers (HMC, NUTS). Everything in TFP is differentiable, meaning you can compute gradients through probability computations.

**Key concepts:**
- **Tensor**: A multi-dimensional array. A scalar is a 0-D tensor, a vector is a 1-D tensor, a matrix is a 2-D tensor.
- **Shape**: The dimensions of a tensor. E.g., shape `(200, 3)` = 200 particles, each with 3 state dimensions.
- **dtype**: The numeric type. `tf.float32` = 32-bit float, `tf.float64` = 64-bit float (more precise but slower).
- **Eager mode**: Operations execute immediately (like NumPy). This is the default.
- **Graph mode**: Operations are compiled into a computation graph first, then executed. Faster for repeated computations. Activated with `@tf.function`.

**Imports used throughout this codebase:**
```python
import tensorflow as tf               # Core TensorFlow
import tensorflow_probability as tfp   # TensorFlow Probability
```

---

## 2. Tensor Creation

### `tf.constant(value, dtype=...)`

Creates an **immutable** tensor from a Python value, list, or NumPy array. Once created, the value cannot change.

**What it does:** Wraps a fixed value into a TensorFlow tensor so it can participate in TF computations.

**Why it's used:** To convert Python/NumPy values into TF tensors. Required when passing data into `@tf.function`-compiled code.

```python
# src/utils/flow_params.py:55
regularization = tf.constant(0.0, dtype=P.dtype)

# src/filters/kalman/extended_kalman.py:59
seed = tf.constant([random_seed if random_seed is not None else 0, 0], dtype=tf.int32)

# src/DF/hmc_runner.py:86
seed = tf.constant([42, 0], dtype=tf.int32)

# src/filters/particle/bootstrap_pf_tf.py:233
observations_tf = tf.constant(observations, dtype=self.dtype)
```

---

### `tf.zeros(shape, dtype=...)`

Creates a tensor filled with zeros.

**What it does:** Makes an all-zero tensor of the given shape.

```python
# src/DF/hmc_runner.py:106
grad = tf.zeros_like(q)  # Zero gradient fallback
```

---

### `tf.ones(shape, dtype=...)`

Creates a tensor filled with ones.

**What it does:** Makes an all-one tensor of the given shape. Often used to create uniform particle weights: each particle starts with weight `1/N`.

```python
# src/filters/particle/bootstrap_pf_tf.py:133
weights = tf.ones(self.n_particles, dtype=self.dtype) / tf.cast(self.n_particles, self.dtype)
# Creates [1/N, 1/N, ..., 1/N] — uniform weights for N particles
```

---

### `tf.ones_like(tensor)` and `tf.zeros_like(tensor)`

Creates a tensor of the same shape and dtype as `tensor`, filled with ones (or zeros).

**What it does:** Copies the shape/dtype of an existing tensor but fills it with ones or zeros. Useful when you need a tensor that matches another tensor's shape.

```python
# src/utils/linalg.py:234
tf.zeros_like(M_inv_T)  # Zero-gradient for NaN matrices

# src/resampling/ot_entropy.py:112
g = tf.ones_like(weights)   # Initialize Sinkhorn potential
f = tf.zeros_like(weights)  # Initialize Sinkhorn potential
```

---

### `tf.eye(n, dtype=...)`

Creates an identity matrix of size `n x n`.

**What it does:** Makes a matrix with ones on the diagonal and zeros elsewhere. The identity matrix `I` satisfies `I @ A = A` for any matrix `A`.

**Why it's used:** Added to matrices for regularization (`A + jitter * I`), and used in flow equations where `I + lambda * A` appears.

```python
# src/utils/linalg.py:29
eye = tf.eye(n, dtype=A.dtype)
# ...
A_reg = A + eye * scaled_jitter  # Regularize matrix for numerical stability

# src/utils/flow_params.py:91
I = tf.eye(state_dim, dtype=P.dtype)
b = tf.linalg.matvec(I + 2 * lambda_val * A, term1 + term2)
```

---

### `tf.fill(dims, value)`

Creates a tensor of shape `dims` filled with `value`.

**What it does:** Like `tf.ones` or `tf.zeros` but with an arbitrary fill value.

```python
# src/models/cubic_sensor.py:214
tf.fill([n_particles], value)
```

---

### `tf.Variable(initial_value, dtype=...)`

Creates a **mutable** tensor whose value can change over time.

**What it does:** Unlike `tf.constant`, a `tf.Variable` can be updated via `.assign()` or `.assign_add()`. Used for state that changes during filtering (e.g., the current mean and covariance).

**Why it's used:** Filters need to update their state (mean, covariance) at each timestep. `tf.Variable` allows this.

```python
# src/filters/kalman/extended_kalman.py:83-84
self.mean = tf.Variable(self.mean_0, dtype=self.dtype)   # Mutable filter mean
self.cov = tf.Variable(self.Sigma_0, dtype=self.dtype)   # Mutable filter covariance

# Later, update via:
# self.mean.assign(new_mean)
# self.cov.assign(new_cov)
```

---

### `tf.TensorArray(dtype, size, ...)`

A dynamically-sized array of tensors. Works inside `@tf.function`.

**What it does:** Think of it as an appendable list that works inside compiled TF graphs. You `.write(index, value)` to store values and `.stack()` to convert the whole array into a single tensor at the end.

**Why it's used:** Inside `@tf.function`, Python lists don't work for accumulating tensors. `TensorArray` is the graph-compatible alternative.

```python
# src/filters/particle/bootstrap_pf_tf.py:136-144
means_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[self.state_dim])
covs_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[self.state_dim, self.state_dim])
log_liks_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[])

# Write to it inside a loop:
means_list = means_list.write(t, mean)     # Store mean at timestep t

# Convert to a regular tensor at the end:
means = means_list.stack()  # Shape: (T, state_dim)
```

- `size=T`: Pre-allocate for `T` elements (fixed-size).
- `dynamic_size=True`: Allow the array to grow (used when the number of writes isn't known ahead of time).
- `element_shape`: Shape of each element (enables shape checking).

---

## 3. Data Types

### `tf.float32`, `tf.float64`, `tf.int32`

Numeric types for tensors.

| Type | Bits | Precision | Use case |
|------|------|-----------|----------|
| `tf.float32` | 32 | ~7 decimal digits | Default, faster on GPU |
| `tf.float64` | 64 | ~15 decimal digits | When float32 is not precise enough |
| `tf.int32` | 32 | Integer | Indices, seeds, counters |

**Why precision matters in this project:** The LEDH particle flow filter accumulates Jacobian determinants over 29 steps. On CUDA (NVIDIA GPU), `float32` can cause "weight collapse" (all weights go to zero) due to accumulated rounding errors. On MPS (Apple Silicon), the same `float32` code works fine due to different internal precision handling.

```python
# src/filters/particle/ledh_invertible.py:61
self.dtype = getattr(model, 'dtype', tf.float64)  # Default to float64 for stability
```

---

### `tf.cast(tensor, dtype)`

Converts a tensor from one numeric type to another.

**What it does:** Changes the dtype of a tensor. Like `int(x)` or `float(x)` in Python, but for tensors.

**Why it's used:** Many TF operations require all inputs to have the same dtype. `tf.cast` converts between types.

```python
# src/utils/linalg.py:34
n_float = tf.cast(n, A.dtype)  # Convert int shape to float for division

# src/utils/distributions.py:25
n = tf.cast(tf.shape(x)[-1], x.dtype)  # Convert dimension (int) to float for log(2*pi)

# src/resampling/systematic.py:30
N_float = tf.cast(N, weights.dtype)  # Convert particle count to float
```

---

## 4. Shape Manipulation

### `tf.shape(tensor)`

Returns the shape of a tensor as a 1-D integer tensor.

**What it does:** Gets the dimensions at runtime. Unlike `tensor.shape` (which gives the static shape known at graph-build time), `tf.shape()` works even when dimensions are unknown.

```python
# src/utils/linalg.py:28
n = tf.shape(A)[-1]  # Get the last dimension (matrix size) dynamically

# src/filters/particle/bootstrap_pf_tf.py:129
T = tf.shape(observations)[0]  # Number of timesteps
```

---

### `tf.reshape(tensor, shape)`

Changes the shape of a tensor without changing its data.

**What it does:** Rearranges the same data into a different shape. The total number of elements must remain the same.

```python
# src/utils/linalg.py:43
scaled_jitter = tf.reshape(scaled_jitter, tf.concat([tf.shape(scaled_jitter), [1, 1]], axis=0))
# Reshapes a scalar to (..., 1, 1) for broadcasting with a matrix
```

---

### `tf.expand_dims(tensor, axis)`

Inserts a dimension of size 1 at the specified position.

**What it does:** Adds a "dummy" dimension. A shape `(5,)` vector becomes `(1, 5)` (axis=0) or `(5, 1)` (axis=1). This is essential for broadcasting -- making shapes compatible for element-wise operations.

```python
# src/utils/flow_params.py:205
P_b = tf.expand_dims(P, 0)  # (sd, sd) -> (1, sd, sd) for broadcasting with (N, sd, sd)

# src/filters/particle/bootstrap_pf_tf.py:170
mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
# weights[:, tf.newaxis] reshapes (N,) -> (N, 1) to multiply with (N, state_dim)
```

---

### `tf.newaxis`

An alias for `None` used in indexing to add a dimension. Equivalent to `tf.expand_dims`.

```python
# src/utils/linalg.py:65
b_rhs = b[..., tf.newaxis]  # (n,) -> (n, 1), turning a vector into a column matrix

# src/utils/linalg.py:182
dy[..., tf.newaxis, tf.newaxis] * M_inv_T  # Scalar (...,) -> (..., 1, 1) for matrix broadcast
```

**`...` (Ellipsis):** In tensor indexing, `...` means "all preceding dimensions." So `b[..., tf.newaxis]` adds a dimension at the end regardless of how many dimensions `b` has.

---

### `tf.squeeze(tensor, axis=...)`

Removes dimensions of size 1.

**What it does:** The opposite of `expand_dims`. Removes "dummy" dimensions.

```python
# src/resampling/ot_entropy.py:83
result = -tf.squeeze(epsilon, axis=[1, 2]) * log_sum_exp
# Removes the extra dimensions added for broadcasting

# src/resampling/ot_entropy.py:87
result = tf.squeeze(result, axis=0)  # Remove batch dimension when input was 1D
```

---

### `tf.transpose(tensor)` / `tf.linalg.matrix_transpose(tensor)`

Swaps dimensions of a tensor.

- `tf.transpose(M)`: For 2-D tensors, swaps rows and columns (standard matrix transpose).
- `tf.linalg.matrix_transpose(M)`: Transposes the last two dimensions. Works with batched matrices like `(N, 3, 3)`.

```python
# src/utils/flow_params.py:73
HPH = H @ P_reg @ tf.transpose(H)  # H * P * H^T

# src/utils/flow_params.py:225
H_T = tf.linalg.matrix_transpose(H_batch)  # (N, obs_dim, state_dim) -> (N, state_dim, obs_dim)
```

---

### `tf.stack(tensors, axis=0)`

Combines a list of tensors into one tensor by stacking along a new axis.

**What it does:** Takes `[tensor1, tensor2, ...]` and creates a single tensor with an extra dimension.

```python
# src/DF/parameter_handler.py:94
return tf.stack(unconstrained_values)
# Combines [scalar1, scalar2, ...] into a 1-D tensor of shape (num_params,)

# src/filters/particle/bootstrap_pf_tf.py:206
means = means_list.stack()  # TensorArray -> single tensor of shape (T, state_dim)
```

---

### `tf.unstack(tensor, axis=0)`

The opposite of `tf.stack`: splits a tensor into a list of tensors along an axis.

```python
# src/filters/particle/bootstrap_pf_tf.py:150-151
seed, trans_seed, resample_seed = tf.unstack(
    tf.random.experimental.stateless_split(seed, num=3)
)
# Splits (3, 2) tensor into three (2,) seed tensors
```

---

### `tf.concat(tensors, axis)`

Joins tensors along an existing axis (unlike `tf.stack` which creates a new axis).

```python
# src/utils/linalg.py:43
tf.concat([tf.shape(scaled_jitter), [1, 1]], axis=0)
# Concatenates two 1-D shape tensors: e.g., [batch_dims..., 1, 1]
```

---

### `tf.tile(tensor, multiples)`

Repeats a tensor along each dimension.

**What it does:** Like copy-pasting a tensor. `tf.tile([[1,2]], [3, 1])` gives `[[1,2],[1,2],[1,2]]`.

```python
# src/utils/flow_params.py:354
z_broadcast = tf.tile(tf.expand_dims(observation, 0), [N, 1])
# Repeats observation (obs_dim,) into (N, obs_dim) so each particle sees the same observation
```

---

## 5. Math and Element-wise Operations

These operate element-by-element on tensors. If `a = [1, 2, 3]` and `b = [4, 5, 6]`, then `a + b = [5, 7, 9]`.

### Arithmetic operators: `+`, `-`, `*`, `/`, `**`

Standard math on tensors. These work element-wise, just like NumPy.

```python
# src/utils/ode_solvers.py:21
return x + f(x, *args) * dt  # Euler step: x_new = x + derivative * dt

# src/utils/distributions.py:33
mahalanobis = tf.reduce_sum(y**2, axis=-1)  # Squared elements, then sum
```

---

### `tf.math.log(x)` (also written as `tf.math.log()`)

Natural logarithm (base e).

**Why it's used:** Probability computations work in log-space to avoid numerical underflow. Probabilities can be extremely small (e.g., 1e-300), but their log is a manageable number (-690).

```python
# src/utils/distributions.py:35
-0.5 * (n * tf.math.log(2.0 * tf.constant(math.pi, dtype=x.dtype)) + logdet + mahalanobis)
# The log-probability of a multivariate Gaussian
```

---

### `tf.exp(x)`

Exponential function: `e^x`. The inverse of `tf.math.log`.

```python
# src/utils/distributions.py:76
weights_unnorm = tf.exp(log_weights_normalized)
# Convert log-weights back to regular weights
```

---

### `tf.sqrt(x)`

Square root.

```python
# src/utils/ode_solvers.py:81
noise = noise * tf.sqrt(tf.cast(dt, x.dtype)) * tf.sqrt(tf.cast(diffusion_coeff, x.dtype))
# Brownian motion noise scales with sqrt(dt)
```

---

### `tf.abs(x)`

Absolute value.

```python
# src/utils/linalg.py:140
tf.math.log(tf.abs(tf.linalg.det(M_reg)))
# log|det(M)| — the log of the absolute value of the determinant
```

---

### `tf.maximum(a, b)` and `tf.minimum(a, b)`

Element-wise maximum (or minimum) of two tensors.

**What it does:** For each element, picks the larger (or smaller) of the two values.

```python
# src/utils/linalg.py:38
scaled_jitter = jitter * tf.maximum(avg_diag, tf.constant(1.0, dtype=avg_diag.dtype))
# Ensure jitter is at least `jitter * 1.0`, even if the matrix diagonal is tiny

# src/resampling/ot_entropy.py:41
squared_dist = tf.maximum(diff, tf.zeros((), dtype=diff.dtype))
# Clamp negative values (from numerical error) to zero
```

---

### `tf.norm(tensor, ...)`

Computes the norm (length/magnitude) of a tensor.

**What it does:** By default, the L2 (Euclidean) norm: `sqrt(sum(x_i^2))`.

```python
# src/DF/hmc_runner.py:108
tf.print("  [grad] nlp=", nlp, " |grad|=", tf.norm(grad))
# Print the magnitude of the gradient vector for debugging
```

---

## 6. Reductions (Aggregating Values)

Reductions collapse one or more dimensions by aggregating values (sum, mean, max, etc.).

### `tf.reduce_sum(tensor, axis=...)`

Sums elements along the specified axis. If `axis=None`, sums everything into a scalar.

**What it does:** Collapses a dimension by adding up values.

```python
# src/utils/distributions.py:33
mahalanobis = tf.reduce_sum(y**2, axis=-1)
# Sum of squared elements along last axis: (N, d) -> (N,)
# This computes the Mahalanobis distance for each particle

# src/utils/distributions.py:77
weights_unnorm / tf.reduce_sum(weights_unnorm, axis=-1, keepdims=True)
# Normalize by dividing each weight by the total sum
```

- `axis=-1`: Sum along the last dimension.
- `keepdims=True`: Keep the reduced dimension as size 1 (for broadcasting).

---

### `tf.reduce_mean(tensor, axis=...)`

Computes the mean along the specified axis.

```python
# src/DF/hmc_runner.py:606
tf.reduce_mean(tf.cast(is_accepted, dtype))
# Compute acceptance rate: fraction of True values
```

---

### `tf.reduce_max(tensor, axis=...)`

Maximum value along the specified axis.

```python
# src/utils/distributions.py:68
max_log = tf.reduce_max(log_weights, axis=-1, keepdims=True)
log_weights_normalized = log_weights - max_log
# The "log-sum-exp trick": subtract the max before exp() to prevent overflow
```

---

### `tf.reduce_logsumexp(tensor, axis=...)`

Computes `log(sum(exp(x)))` in a numerically stable way.

**What it does:** This is the most important numerical trick in this codebase. Naively computing `log(sum(exp(x)))` can overflow (if x has large values) or underflow (if x has very negative values). This function handles it by internally subtracting the max.

**Why it's used:** Computing marginal likelihoods and normalizing weights in log-space.

```python
# src/utils/distributions.py:52
return tf.reduce_logsumexp(log_values, axis=axis)

# src/utils/distributions.py:214
log_marginal_lik = tf.reduce_logsumexp(log_weights)
# log p(y_t | y_{1:t-1}) = logsumexp(log_weights)
# This gives the log marginal likelihood for importance-weighted particle filters
```

---

### `tf.cumsum(tensor, axis=...)`

Cumulative sum along an axis.

**What it does:** For `[a, b, c]`, returns `[a, a+b, a+b+c]`.

**Why it's used:** Resampling algorithms use the cumulative distribution function (CDF), which is the cumulative sum of weights.

```python
# src/resampling/systematic.py:33
cumsum = tf.cumsum(weights)
# Weights: [0.1, 0.3, 0.2, 0.4] -> CDF: [0.1, 0.4, 0.6, 1.0]
```

---

## 7. Clipping and Conditionals

### `tf.clip_by_value(tensor, clip_value_min, clip_value_max)`

Clamps all values to be within `[min, max]`.

**What it does:** Values below `min` become `min`, values above `max` become `max`, values in between are unchanged.

**Why it's used:** Prevents numerical overflow/underflow. For example, clipping log-weights to `[-30, 30]` prevents `exp(-1000)` = 0 or `exp(1000)` = infinity.

```python
# src/utils/distributions.py:72-73
log_weights_normalized = tf.clip_by_value(
    log_weights_normalized, clip_range[0], clip_range[1]
)

# src/resampling/systematic.py:45
indices = tf.clip_by_value(indices, 0, N - 1)
# Ensure indices are within valid bounds [0, N-1]
```

---

### `tf.where(condition, x, y)`

Element-wise conditional: returns `x` where condition is True, `y` where False.

**What it does:** Like a vectorized if-else. For each element: `if condition[i]: result[i] = x[i] else: result[i] = y[i]`.

```python
# src/utils/distributions.py:210
log_weights = tf.where(
    tf.math.is_finite(log_weights),
    log_weights,
    tf.constant(-1e30, dtype=log_weights.dtype)
)
# Replace NaN/Inf log-weights with -1e30 (effectively zero weight after exp)

# src/utils/linalg.py:224-225
M_safe = tf.where(
    is_finite[..., tf.newaxis, tf.newaxis], M_reg, eye
)
# Replace NaN matrices with identity matrix
```

---

### `tf.cond(predicate, true_fn, false_fn)`

Graph-compatible if-else. Executes `true_fn()` if predicate is True, `false_fn()` otherwise.

**What it does:** Unlike Python `if`, `tf.cond` works inside `@tf.function` compiled graphs. Both branches must return tensors of the same type and shape.

**Why it's used:** The weight collapse detector needs to work inside compiled graph code.

```python
# src/utils/distributions.py:225
weights = tf.cond(is_finite, lambda: weights, _warn_and_uniform)
# If weights are finite, use them. Otherwise, fall back to uniform weights.
```

---

### `tf.math.is_finite(tensor)`

Returns a boolean tensor: True where values are finite (not NaN, not Inf).

```python
# src/utils/linalg.py:220
is_finite = tf.reduce_all(tf.math.is_finite(M_reg), axis=[-2, -1])
# Check if ALL elements in each matrix are finite

# src/DF/hmc_runner.py:107
n_bad = tf.reduce_sum(tf.cast(~tf.math.is_finite(grad), tf.int32))
# Count how many gradient elements are NaN or Inf
```

---

## 8. Linear Algebra

The `tf.linalg` module provides matrix operations. These are heavily used for covariance manipulation, Kalman filter updates, and particle flow computations.

### `tf.linalg.cholesky(A)`

Computes the Cholesky decomposition: finds lower triangular matrix `L` such that `L @ L^T = A`.

**What it does:** Decomposes a symmetric positive-definite matrix into a "square root." This is more numerically stable and faster than computing the full inverse.

**Why it's used:** Gaussian sampling requires `L` (the Cholesky factor of the covariance). Solving `Ax = b` via Cholesky is 2x faster than general solve.

```python
# src/utils/linalg.py:45
return tf.linalg.cholesky(A_reg)
# A_reg = A + jitter * I (regularized to prevent failure on near-singular matrices)

# src/utils/distributions.py:96
L = tf.linalg.cholesky(cov)
# Cholesky of covariance matrix for sampling: samples = mean + L @ z
```

**Note:** Cholesky only works on symmetric positive-definite matrices. If the matrix is near-singular, this can fail. That's why `safe_cholesky()` adds a small jitter to the diagonal.

---

### `tf.linalg.cholesky_solve(L, b)`

Solves `A @ x = b` where `L` is the Cholesky factor of `A` (i.e., `A = L @ L^T`).

**What it does:** Two back-substitutions: first solve `L @ y = b`, then solve `L^T @ x = y`. Much faster than computing `A^{-1} @ b` directly.

```python
# src/utils/flow_params.py:233
S_inv_H = tf.linalg.cholesky_solve(L_S, H_batch)
# Solves S @ X = H for X, where L_S is the Cholesky factor of S
# This computes S^{-1} @ H without explicitly computing S^{-1}
```

---

### `tf.linalg.solve(A, b)`

Solves the linear system `A @ x = b` for `x`.

**What it does:** Finds the vector (or matrix) `x` that satisfies `A @ x = b`. Internally uses LU decomposition.

```python
# src/utils/linalg.py:76
result = tf.linalg.solve(A, b_rhs)
# Default solver when no specific method is requested
```

---

### `tf.linalg.lstsq(A, b, fast=False)`

Least-squares solution to `A @ x ≈ b`.

**What it does:** If the system is overdetermined (more equations than unknowns), finds the `x` that minimizes `||A @ x - b||^2`. If `A` is square and invertible, gives the exact solution.

```python
# src/utils/linalg.py:73
result = tf.linalg.lstsq(A, b_rhs, fast=False)
# Fallback solver when Cholesky and direct solve might fail
```

---

### `tf.linalg.inv(A)`

Computes the matrix inverse `A^{-1}`.

**What it does:** Finds the matrix such that `A @ A^{-1} = I`. Computationally expensive and numerically sensitive.

**Warning:** This can crash in graph mode on GPU if the matrix is singular (contains NaN). That's why `safe_inv()` adds regularization and `graph_safe_inv()` uses `pinv` instead.

```python
# src/utils/linalg.py:118
return tf.linalg.inv(A + jitter * eye)
# Add small jitter to diagonal before inverting for numerical stability
```

---

### `tf.linalg.pinv(A)`

Computes the Moore-Penrose pseudo-inverse (SVD-based).

**What it does:** Like `inv`, but works on any matrix (even singular or non-square). Uses SVD internally, so it never crashes, but is slower than `inv`.

**Why it's used:** As a safe fallback in GPU graph mode where `inv` can crash on singular matrices.

```python
# src/utils/linalg.py:166
return tf.linalg.pinv(A + jitter * eye)
# graph_safe_inv: uses pinv instead of inv to avoid GPU crashes
```

---

### `tf.linalg.det(A)`

Computes the determinant of a matrix.

**What it does:** Returns a scalar that measures the "volume scaling factor" of the matrix. For a 2x2 matrix `[[a,b],[c,d]]`, the determinant is `ad - bc`.

**Why it's used:** Jacobian determinants track how the particle flow stretches/compresses probability mass.

```python
# src/utils/linalg.py:140
tf.math.log(tf.abs(tf.linalg.det(M_reg)))
# log|det(M)| — used for Jacobian accumulation in invertible flows
```

---

### `tf.linalg.slogdet(A)`

Computes the sign and log of the absolute determinant.

**What it does:** Returns `(sign, log|det(A)|)`. More numerically stable than `log(abs(det(A)))` because it avoids computing the potentially huge/tiny determinant directly.

```python
# src/utils/linalg.py:94
sign, logdet = tf.linalg.slogdet(A)
# sign is +1 or -1, logdet is log|det(A)|

# src/utils/distributions.py:28
sign, logdet = tf.linalg.slogdet(cov)
# Used in Gaussian log-probability: the -0.5*log|Sigma| term
```

---

### `tf.linalg.svd(A, compute_uv=False)`

Singular Value Decomposition.

**What it does:** Decomposes `A = U @ diag(s) @ V^T`. When `compute_uv=False`, only returns the singular values `s`.

**Why it's used:** `log|det| = sum(log(s))`. SVD-based determinant is the most robust approach (never crashes), but ~4x slower than LU-based `slogdet`.

```python
# src/utils/linalg.py:281
s = tf.linalg.svd(M_reg, compute_uv=False)
return tf.reduce_sum(tf.math.log(tf.maximum(s, tf.constant(1e-30, dtype=s.dtype))), axis=-1)
```

---

### `tf.linalg.eigh(A)`

Eigenvalue decomposition for symmetric (Hermitian) matrices.

**What it does:** Returns eigenvalues and eigenvectors such that `A = V @ diag(lambda) @ V^T`.

```python
# src/utils/linalg.py:315
eigvals, eigvecs = tf.linalg.eigh(A)
sqrt_eigvals = tf.sqrt(eigvals)
# Matrix square root via eigendecomposition: sqrt(A) = V @ diag(sqrt(lambda)) @ V^T
```

---

### `tf.linalg.trace(A)`

Sum of diagonal elements of a matrix.

**What it does:** For matrix `A`, computes `A[0,0] + A[1,1] + ... + A[n,n]`.

**Why it's used:** The trace of a covariance matrix is the total variance. Used for adaptive regularization scaling.

```python
# src/utils/linalg.py:33
trace_A = tf.linalg.trace(A)
avg_diag = trace_A / n_float
scaled_jitter = jitter * tf.maximum(avg_diag, 1.0)
# Scale regularization by the matrix's average diagonal value
```

---

### `tf.linalg.diag(diagonal)` and `tf.linalg.diag_part(matrix)`

- `tf.linalg.diag(v)`: Creates a diagonal matrix from a vector.
- `tf.linalg.diag_part(M)`: Extracts the diagonal from a matrix.

```python
# src/utils/linalg.py:319
eigvecs @ tf.linalg.diag(sqrt_eigvals) @ tf.linalg.matrix_transpose(eigvecs)
# Build a diagonal matrix from eigenvalues for matrix square root

# src/utils/distributions.py:185
tf.linalg.diag_part(L_Q)
# Extract diagonal of Cholesky factor: used to compute log|det| = 2 * sum(log(diag(L)))
```

---

### `tf.matmul(a, b)` / `tf.linalg.matmul(a, b)` / `a @ b`

Matrix multiplication. All three forms are equivalent.

**What it does:** Standard matrix multiplication. For (m, n) @ (n, p) = (m, p). Also supports batched: (B, m, n) @ (B, n, p) = (B, m, p).

```python
# src/utils/distributions.py:100
return mean + tf.linalg.matmul(z, L, transpose_b=True)
# Gaussian sampling: samples = mean + z @ L^T, where z ~ N(0, I) and L is Cholesky factor

# src/utils/flow_params.py:222
HP = tf.matmul(H_batch, P_b)  # (N, obs_dim, state_dim) @ (N, state_dim, state_dim)
```

- `transpose_b=True`: Transposes the second argument before multiplying. Equivalent to `a @ tf.transpose(b)`.

---

### `tf.linalg.matvec(matrix, vector)`

Matrix-vector multiplication: `M @ v`.

**What it does:** Multiplies a matrix by a vector. Unlike `matmul`, the result is a vector (not a matrix). Saves you from having to reshape the vector into a column matrix.

```python
# src/utils/flow_params.py:87
e = h_x - tf.linalg.matvec(H, linearization_point)
# H @ x, where H is (obs_dim, state_dim) and x is (state_dim,), result is (obs_dim,)

# src/utils/flow_params.py:93-95
term1 = tf.linalg.matvec((I + lambda_val * A) @ P_reg @ tf.transpose(H) @ R_inv, observation - e)
term2 = tf.linalg.matvec(A, eta_bar_0)
b = tf.linalg.matvec(I + 2 * lambda_val * A, term1 + term2)
```

---

### `tf.linalg.triangular_solve(L, b, lower=True)`

Solves `L @ x = b` where `L` is a triangular matrix.

**What it does:** Back-substitution for triangular systems. Much faster than general solve because it exploits the triangular structure.

**Why it's used:** Computing Mahalanobis distance via Cholesky: instead of `(x - mu)^T @ Sigma^{-1} @ (x - mu)`, compute `y = L^{-1} @ (x - mu)` via triangular solve, then `||y||^2`. This avoids computing the full matrix inverse.

```python
# src/utils/distributions.py:32
y = tf.linalg.triangular_solve(L, diff[..., tf.newaxis], lower=True)[..., 0]
mahalanobis = tf.reduce_sum(y**2, axis=-1)
# Mahalanobis distance = (x-mu)^T @ Sigma^{-1} @ (x-mu)
# Computed as ||L^{-1} @ (x-mu)||^2 without forming Sigma^{-1}
```

---

### `tf.einsum(equation, *tensors)`

Einstein summation: a flexible way to express tensor contractions.

**What it does:** Uses string notation to describe arbitrary tensor operations (matrix multiply, trace, outer product, batch operations, etc.). Think of it as a compact language for "sum over repeated indices."

**Notation:** Each letter is a dimension index. Repeated letters on the input side are summed over.

```python
# src/filters/particle/bootstrap_pf_tf.py:174
cov = tf.reduce_sum(
    weights[:, tf.newaxis, tf.newaxis] *
    tf.einsum('ij,ik->ijk', diff, diff),
    axis=0
)
# 'ij,ik->ijk' = outer product for each particle:
#   diff[i,j] * diff[i,k] -> outer_product[i,j,k]
# This computes the (N, d, d) array of outer products, then sums with weights to get covariance

# src/utils/flow_params.py:241
Hx = tf.einsum('nij,nj->ni', H_batch, linearization_points)
# 'nij,nj->ni' = batched matrix-vector multiply:
#   For each particle n: H[n,:,:] @ x[n,:] -> result[n,:]

# src/utils/flow_params.py:262
A_eta = tf.einsum('nij,j->ni', A_batch, eta_bar_0)
# 'nij,j->ni' = batched matrix-vector multiply with shared vector:
#   For each particle n: A[n,:,:] @ eta_bar_0[:] -> result[n,:]
```

---

## 9. Random Number Generation

This codebase uses **stateless** random operations throughout. Unlike regular random functions that rely on a global seed, stateless operations take an explicit seed tensor, making results fully reproducible and compatible with `@tf.function`.

### `tf.random.stateless_normal(shape, seed, dtype=...)`

Samples from a standard normal distribution N(0, 1).

**What it does:** Generates random numbers from a Gaussian distribution. The seed ensures reproducibility: same seed = same random numbers.

```python
# src/utils/distributions.py:97
z = tf.random.stateless_normal([n_samples, d], seed=seed, dtype=mean.dtype)
# Draw n_samples standard normal vectors, each of dimension d
# These are then transformed: samples = mean + L @ z
```

---

### `tf.random.stateless_uniform(shape, seed, minval=0, maxval=1, dtype=...)`

Samples from a uniform distribution.

```python
# src/resampling/systematic.py:37
u = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0/N_float, dtype=weights.dtype)
# Single uniform random number in [0, 1/N] for systematic resampling
```

---

### `tf.random.experimental.stateless_split(seed, num=2)`

Splits a single seed into multiple independent seeds.

**What it does:** From one seed, generates `num` new seeds that produce independent random streams. This is how you get multiple independent random operations from a single starting seed.

**Why it's used:** Each timestep needs fresh, independent random numbers for prediction (state transitions) and resampling. Splitting the seed ensures independence.

```python
# src/filters/particle/bootstrap_pf_tf.py:150-152
seed, trans_seed, resample_seed = tf.unstack(
    tf.random.experimental.stateless_split(seed, num=3)
)
# Split one seed into three: one for next iteration, one for state transitions, one for resampling
```

---

### `tf.random.uniform(shape, minval, maxval, dtype=...)`

Non-stateless (global-seed) uniform random sampling.

**What it does:** Same as `stateless_uniform` but uses TF's global random state. Not reproducible across runs unless you set `tf.random.set_seed()`.

```python
# src/filters/particle/bootstrap_pf_tf.py:234
seed = tf.random.uniform([2], minval=0, maxval=2**31 - 1, dtype=tf.int32)
# Generate a random initial seed for the stateless operations in the filter loop
```

---

### `tf.random.set_seed(seed)`

Sets the global random seed for TensorFlow.

```python
# src/DF/hmc_runner.py:181
if seed is not None:
    tf.random.set_seed(seed)
# Set global seed before HMC sampling for reproducibility
```

---

## 10. Indexing and Selection

### `tf.gather(params, indices)`

Selects elements from a tensor by index.

**What it does:** Like NumPy fancy indexing `params[indices]`. Returns the elements at the specified indices.

**Why it's used:** Resampling selects particles by index -- high-weight particles get duplicated, low-weight particles get dropped.

```python
# src/resampling/systematic.py:48
resampled_particles = tf.gather(particles, indices)
# Select particles by index: if indices = [0, 0, 2, 3], particle 0 is duplicated, particle 1 is dropped
```

---

### `tf.searchsorted(sorted_sequence, values, side='right')`

Binary search: finds insertion points in a sorted sequence.

**What it does:** For each value in `values`, finds the index in `sorted_sequence` where it would be inserted to maintain sorted order.

**Why it's used:** In systematic resampling, the CDF (cumulative weights) is a sorted sequence. For each uniform sample point, `searchsorted` finds which particle it maps to.

```python
# src/resampling/systematic.py:42
indices = tf.searchsorted(cumsum, u_vals, side='right')
# cumsum = [0.1, 0.4, 0.6, 1.0]  (CDF of weights)
# u_vals = [0.05, 0.30, 0.55, 0.80]  (systematic sample points)
# result = [0, 1, 2, 3]  (which particle each sample point falls into)
```

---

### `tf.unique(x)`

Returns unique elements and their indices.

```python
# src/filters/particle/bootstrap_pf_tf.py:198
unique_first_dim = tf.unique(particles[:, 0])[0]
n_unique = tf.shape(unique_first_dim)[0]
# Count unique particles after resampling (diagnostic for particle diversity)
```

---

### Tensor slicing and indexing

TensorFlow tensors support NumPy-style indexing:

```python
# src/DF/parameter_handler.py:110
unconstrained_value = unconstrained_params[i]  # Index into 1-D tensor

# src/filters/particle/bootstrap_pf_tf.py:158
observations[t]  # Get observation at timestep t

# src/filters/particle/bootstrap_pf_tf.py:198
particles[:, 0]  # All particles, first state dimension
```

---

## 11. Iteration and Loops

### `tf.range(n)`

Creates a sequence of integers `[0, 1, 2, ..., n-1]`.

**What it does:** The TF equivalent of Python's `range()`. Works inside `@tf.function`.

```python
# src/filters/particle/bootstrap_pf_tf.py:148
for t in tf.range(T):
    # Process each timestep
    ...
```

**Why not use Python `range()`?** Inside `@tf.function`, `tf.range` creates a proper TF loop that works with dynamic shapes (when `T` is only known at runtime). Python `range(T)` would require `T` to be known at graph-build time.

---

### `tf.while_loop(cond, body, loop_vars)`

A loop primitive for TensorFlow graphs.

**What it does:** Repeatedly calls `body(loop_vars)` as long as `cond(loop_vars)` returns True. All loop variables must be tensors and maintain consistent shapes.

```python
# src/resampling/ot_entropy.py:199 (Sinkhorn iteration)
# Iterates the Sinkhorn algorithm until convergence or max iterations
```

---

## 12. Graph Compilation with @tf.function

### `@tf.function`

Decorator that compiles a Python function into a TensorFlow computation graph.

**What it does:** The first time you call the function, TF "traces" it: it executes the Python code once, recording all TF operations into a graph. On subsequent calls, it runs the graph directly (skipping Python), which is much faster.

**Why it's used:** Critical for performance. A particle filter loop with 1000 timesteps runs ~10-100x faster when compiled.

```python
# src/utils/linalg.py:7-8
@tf.function
def safe_cholesky(A: tf.Tensor, jitter: float = 1e-10, adaptive: bool = True) -> tf.Tensor:
    ...

# src/filters/particle/bootstrap_pf_tf.py:117-118
@tf.function
def filter_tf(self, observations: tf.Tensor, seed: tf.Tensor):
    ...
```

**Important caveats:**
- Python `if/else` is only evaluated once during tracing. Use `tf.cond` for runtime conditionals.
- Python `print()` only runs during tracing. Use `tf.print()` for runtime output.
- Python lists can't accumulate tensors. Use `tf.TensorArray`.
- Side effects (writing to variables outside the function) behave differently.

### `@tf.function(reduce_retracing=True)`

A variant that reduces redundant retracing when the function is called with different input shapes.

```python
# src/filters/kalman/extended_kalman.py:87
@tf.function(reduce_retracing=True)
def _predict_step(self, mean: tf.Tensor, cov: tf.Tensor):
    ...
# Prevents retracing every time the EKF is called with slightly different shapes
```

---

## 13. Custom Gradients

### `@tf.custom_gradient`

Defines a function with a manually-specified backward pass (gradient).

**What it does:** Normally, TF computes gradients automatically. But sometimes the automatic gradient is numerically unstable or undefined. `@tf.custom_gradient` lets you replace the automatic gradient with a custom one.

**Why it's used:** The gradient of `log|det(M)|` uses `M^{-1}`, which crashes on GPU in graph mode for singular matrices. The custom gradient uses `pinv` (pseudo-inverse) instead, which never crashes.

```python
# src/utils/linalg.py:171-183
@tf.custom_gradient
def _graph_safe_log_abs_det_impl(M_reg):
    """log|det(M)| with robust backward pass."""
    # Forward pass: fast LU-based slogdet
    sign, logabsdet = tf.linalg.slogdet(M_reg)

    def grad(dy):
        # Backward pass: use pinv instead of inv to avoid GPU crash
        M_inv_T = tf.linalg.matrix_transpose(tf.linalg.pinv(M_reg))
        return dy[..., tf.newaxis, tf.newaxis] * M_inv_T

    return logabsdet, grad
```

**How it works:**
1. The function returns `(output, grad_function)`.
2. During forward pass, TF uses the output.
3. During backward pass (gradient computation), TF calls `grad(dy)` where `dy` is the upstream gradient.
4. `grad` must return the gradient with respect to each input.

---

## 14. Automatic Differentiation (Gradient Tape)

### `tf.GradientTape`

Records operations for automatic differentiation.

**What it does:** Creates a "tape" that records every TF operation performed on watched tensors. After recording, you call `tape.gradient(output, inputs)` to compute the gradient of `output` with respect to `inputs`.

**Why it's used:** HMC/NUTS requires the gradient of the log-posterior with respect to parameters. `GradientTape` computes this automatically through the entire filter forward pass.

```python
# src/DF/hmc_runner.py:99-102
with tf.GradientTape() as tape:
    tape.watch(q)                              # Watch the parameter vector
    nlp = self._negative_log_posterior(q)       # Forward pass: run the entire filter
grad = tape.gradient(nlp, q)                   # Backward pass: compute gradient
```

- `tape.watch(q)`: Tells the tape to track operations involving `q`. Needed for plain tensors (not `tf.Variable`s, which are watched automatically).
- `tape.gradient(loss, variables)`: Computes `d(loss)/d(variables)` using the chain rule through all recorded operations.
- `with` context: The tape only records operations inside the `with` block.

---

### `tf.stop_gradient(tensor)`

Prevents gradient from flowing through a tensor.

**What it does:** Returns the same tensor value, but during backpropagation, gradients are not computed through it. Effectively treats the tensor as a constant during differentiation.

```python
# src/resampling/ot_entropy.py:207-208
# Used in Sinkhorn iterations to control gradient flow
```

---

## 15. Debugging Tools

### `tf.print(*values)`

Prints tensor values at runtime (including inside `@tf.function`).

**What it does:** Unlike Python `print()`, which only runs during tracing, `tf.print()` executes every time the graph runs. Essential for debugging compiled code.

```python
# src/utils/distributions.py:223
tf.print("WARNING: weight collapse detected, falling back to uniform weights")
# Prints a warning when all particle weights become NaN/Inf

# src/DF/hmc_runner.py:108
tf.print("  [grad] nlp=", nlp, " |grad|=", tf.norm(grad),
         " grad=", grad, " q=", q, " n_bad=", n_bad)
# Print gradient diagnostics during HMC sampling
```

---

### `tf.debugging.assert_positive(tensor, message=...)`

Asserts that all elements of a tensor are positive. Raises an error otherwise.

```python
# src/utils/linalg.py:95
tf.debugging.assert_positive(sign, message="log_det: matrix is not positive definite (sign <= 0)")
# Ensure the determinant is positive (matrix is positive definite)
```

---

## 16. Device Configuration (GPU/CPU)

These functions configure which hardware TensorFlow uses.

### `tf.config.list_physical_devices(device_type)`

Lists available hardware devices.

```python
# src/utils/device.py:61
gpus = tf.config.list_physical_devices('GPU')  # List all GPUs
cpus = tf.config.list_physical_devices('CPU')  # List all CPUs
```

---

### `tf.config.experimental.set_memory_growth(device, True)`

Enables dynamic GPU memory allocation.

**What it does:** By default, TF grabs ALL GPU memory at startup. With memory growth enabled, TF only allocates memory as needed and grows incrementally.

**Why it's used:** Prevents out-of-memory errors when other processes also need GPU memory.

```python
# src/utils/device.py:71-72
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

### `tf.config.set_visible_devices(devices, device_type)`

Controls which devices TF can see.

**What it does:** Setting visible devices to `[]` for GPUs forces TF to use CPU only.

```python
# src/utils/device.py:129
tf.config.set_visible_devices([], 'GPU')  # Force CPU-only mode
```

---

### `tf.config.experimental.get_device_details(device)`

Gets detailed information about a hardware device.

```python
# src/utils/device.py:78
details = tf.config.experimental.get_device_details(gpus[0])
device_name = details.get('device_name', 'Unknown')
# Distinguish CUDA (NVIDIA) from MPS (Apple Silicon)
```

---

## 17. NumPy Interop

### `.numpy()`

Converts a TensorFlow tensor to a NumPy array.

**What it does:** Copies data from TF (possibly on GPU) to a NumPy array (always on CPU). Only works in eager mode (not inside `@tf.function`).

**Why it's used:** Results from TF computations are converted to NumPy for storage, plotting, and logging.

```python
# src/filters/particle/bootstrap_pf_tf.py:240-246
means = means_tf.numpy()
covs = covs_tf.numpy()
log_likelihoods = log_liks_tf.numpy()
# Convert all filter results from TF tensors to NumPy arrays

# src/utils/linalg.py:326
return x.numpy() if isinstance(x, tf.Tensor) else x
# Helper: convert to numpy, pass through if already numpy
```

---

### `tf.constant(numpy_array, dtype=...)`

Converts a NumPy array (or Python value) to a TensorFlow tensor.

```python
# src/filters/particle/bootstrap_pf_tf.py:233
observations_tf = tf.constant(observations, dtype=self.dtype)
# Convert NumPy observation array to TF tensor before passing to @tf.function
```

---

## 18. TFP Distributions

TensorFlow Probability distributions represent probability distributions as objects. You can sample from them and compute their log-probabilities.

### `tfp.distributions.LogNormal(loc, scale)`

A log-normal distribution: if `X ~ LogNormal(loc, scale)`, then `log(X) ~ Normal(loc, scale)`.

**What it does:** Creates a distribution over positive real numbers. The parameters are the mean and std of the underlying normal distribution on the log scale.

**Why it's used:** Prior distribution for positive-valued parameters (like noise variances). LogNormal(0, 1) peaks near 1 and has a long right tail.

```python
# src/DF/smoke_test_linear_gaussian.py:103-105
prior=tfp.distributions.LogNormal(
    tf.constant(0.0, dtype=tf.float32),
    tf.constant(1.0, dtype=tf.float32)
)
# Prior for a positive parameter: most likely around 1, but allows large values
```

---

### `tfp.distributions.Beta(concentration1, concentration0)`

A Beta distribution over the interval (0, 1).

```python
# src/DF/example_usage.py:58
prior=tfp.distributions.Beta(9.0, 1.0)
# Beta(9, 1) is peaked near 0.9 — a prior that says "this parameter is probably close to 1"
```

---

### `.log_prob(value)`

Computes the log probability density at a given value.

**What it does:** Evaluates `log p(x)` where `p` is the probability density function of the distribution.

**Why it's used:** Prior contributions to the log-posterior are computed via `.log_prob()`.

```python
# src/DF/parameter_handler.py:160
log_prob_constrained = tf.cast(spec.prior.log_prob(value_for_prior), self.dtype)
# Evaluate log-prior at the current parameter value
```

---

### `tfp.distributions.Distribution` (base class)

The abstract base class for all TFP distributions. Used as a type hint.

```python
# src/DF/types.py:27
prior: tfp.distributions.Distribution
# Type annotation: any TFP distribution can be used as a prior
```

---

## 19. TFP Bijectors (Parameter Transforms)

Bijectors are invertible transformations. They map between "constrained" space (where parameters have physical meaning, like positive variance) and "unconstrained" space (all of R^n, where HMC can operate freely).

### `tfp.bijectors.Identity()`

The identity transformation: `f(x) = x`.

**What it does:** No transformation. Input = output. Used for parameters with no constraints.

```python
# src/DF/parameter_handler.py:54
bijectors[name] = tfp.bijectors.Identity()
# Unconstrained parameter: no transformation needed
```

---

### `tfp.bijectors.Softplus()`

The softplus transformation: `f(x) = log(1 + exp(x))`.

**What it does:** Maps R -> (0, infinity). Like `exp(x)` but grows linearly for large x instead of exponentially, preventing overflow.

**Why it's used:** Ensures parameters like variance stay positive during HMC, where the sampler explores unconstrained space.

```python
# src/DF/parameter_handler.py:60
bijectors[name] = tfp.bijectors.Softplus()
# Positive constraint: unconstrained R -> (0, infinity)
# softplus(-5) ≈ 0.007, softplus(0) ≈ 0.693, softplus(5) ≈ 5.007
```

---

### `tfp.bijectors.Sigmoid(low=..., high=...)`

The sigmoid transformation: `f(x) = 1 / (1 + exp(-x))`, optionally scaled to interval `(low, high)`.

**What it does:** Maps R -> (0, 1) by default. With `low` and `high`, maps R -> (low, high).

```python
# src/DF/parameter_handler.py:64
bijectors[name] = tfp.bijectors.Sigmoid()
# Unit interval constraint: R -> (0, 1)

# src/DF/parameter_handler.py:69
bijectors[name] = tfp.bijectors.Sigmoid(low=a, high=b)
# Bounded constraint: R -> (a, b)
```

---

### `.forward(x)` and `.inverse(x)`

Apply the bijector or its inverse.

- `.forward(x)`: unconstrained -> constrained (e.g., softplus maps R -> R+)
- `.inverse(x)`: constrained -> unconstrained (e.g., inverse softplus maps R+ -> R)

```python
# src/DF/parameter_handler.py:111
constrained[name] = bijector.forward(unconstrained_value)
# Transform from HMC's unconstrained space to the parameter's natural space

# src/DF/parameter_handler.py:91
unconstrained = bijector.inverse(constrained)
# Transform from natural parameter space to HMC's unconstrained space
```

---

### `.forward_log_det_jacobian(x, event_ndims)`

Computes the log absolute determinant of the Jacobian of the forward transformation.

**What it does:** When you change variables (unconstrained -> constrained), the probability density changes by a factor of `|det(d_constrained / d_unconstrained)|`. The log of this factor is the "log det Jacobian."

**Why it's used:** Correct probability computation requires accounting for the Jacobian when using bijector transforms. Without it, the posterior would be biased.

```python
# src/DF/parameter_handler.py:164-166
log_det_jacobian = bijector.forward_log_det_jacobian(
    bijector.inverse(constrained_value),
    event_ndims=0  # Scalar parameter
)
# Total log prob in unconstrained space = log_prior(constrained) + log_det_jacobian
```

---

## 20. TFP MCMC (Markov Chain Monte Carlo)

TFP provides implementations of MCMC samplers for Bayesian inference. These generate samples from a target distribution (the posterior) by constructing a Markov chain.

### `tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn, step_size, num_leapfrog_steps)`

Hamiltonian Monte Carlo (HMC) sampler.

**What it does:** Simulates a physical system where the parameter is a "ball" rolling on the "surface" of the negative log-posterior. Uses gradient information (from `tf.GradientTape`) to make informed proposals, achieving much higher acceptance rates than random-walk methods.

**Parameters:**
- `target_log_prob_fn`: Function that computes `log p(theta | y)` for a given parameter vector.
- `step_size`: Size of each leapfrog integration step. Too large = rejected proposals. Too small = slow exploration.
- `num_leapfrog_steps`: Number of leapfrog steps per proposal. More steps = longer trajectories = more distant proposals.

```python
# src/DF/hmc_runner.py:166-170
inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=step_size,
    num_leapfrog_steps=num_leapfrog_steps
)
```

---

### `tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size, max_tree_depth)`

No-U-Turn Sampler (NUTS) -- an adaptive extension of HMC.

**What it does:** Automatically determines the optimal number of leapfrog steps by building a binary tree of states and stopping when the trajectory starts to "turn around" (make a U-turn). Eliminates the need to tune `num_leapfrog_steps`.

```python
# src/DF/hmc_runner.py:159-163
inner_kernel = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=target_log_prob_fn,
    step_size=step_size,
    max_tree_depth=max_tree_depth  # Limits the binary tree depth (2^depth max steps)
)
```

---

### `tfp.mcmc.DualAveragingStepSizeAdaptation(inner_kernel, num_adaptation_steps, target_accept_prob)`

Wraps an HMC/NUTS kernel with adaptive step size tuning.

**What it does:** During burn-in, automatically adjusts the step size to achieve a target acceptance rate. Uses the dual averaging algorithm from Nesterov (2009).

**Parameters:**
- `inner_kernel`: The HMC or NUTS kernel to adapt.
- `num_adaptation_steps`: Number of steps to adapt (usually a fraction of burn-in).
- `target_accept_prob`: Desired acceptance rate (typically 0.65-0.85).

```python
# src/DF/hmc_runner.py:174-178
adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel,
    num_adaptation_steps=num_adaptation_steps,
    target_accept_prob=target_accept_prob  # e.g., 0.75
)
```

---

### `tfp.mcmc.effective_sample_size(chain)`

Computes the effective sample size (ESS) of an MCMC chain.

**What it does:** MCMC samples are autocorrelated (consecutive samples are similar). ESS estimates how many independent samples the chain is equivalent to. ESS << total samples means the chain is mixing poorly.

```python
# src/DF/hmc_runner.py:618-620
param_tf = tf.constant(s[:, np.newaxis], dtype=dtype)
ess = tfp.mcmc.effective_sample_size(param_tf)
ess_dict[name] = float(ess.numpy()[0])
# If 1000 samples have ESS = 200, they carry as much information as 200 independent samples
```

---

### `tfp.mcmc.potential_scale_reduction(chains)`

Computes the R-hat statistic (Gelman-Rubin diagnostic).

**What it does:** Compares within-chain variance to between-chain variance. R-hat close to 1.0 indicates convergence. R-hat > 1.1 suggests the chain hasn't converged.

**How it works:** The chain is split into two halves (pseudo-chains), and the ratio of between-half to within-half variance is computed.

```python
# src/DF/hmc_runner.py:626-628
mid = len(s) // 2
chains = tf.constant([s[:mid], s[mid:2*mid]], dtype=dtype)  # Split into two halves
rhat_dict[name] = float(tfp.mcmc.potential_scale_reduction(chains).numpy())
# rhat ≈ 1.0 means converged, rhat > 1.1 means more samples needed
```

---

## 21. Additional Functions

### `tf.identity(tensor)`

Returns a tensor with the same value.

**What it does:** Creates a copy of the tensor in the computation graph. The value is unchanged.

**Why it's used:** Forces TensorFlow to treat the result as a new tensor, which is useful for snapshotting state (e.g., storing a copy of particles before they get modified in-place).

```python
# src/filters/kalman/extended_kalman.py:215-216
mean = tf.identity(self.mean_0)
cov = tf.identity(self.Sigma_0)
# Snapshot initial state (don't alias the Variable)

# src/filters/particle/conditional_smc.py:101
particles_history.append(tf.identity(particles))
# Store a copy so future modifications to `particles` don't alter stored values
```

---

### `tf.nn.softmax(logits)`

Computes softmax: `exp(x_i) / sum(exp(x_j))`.

**What it does:** Converts a vector of arbitrary real numbers into a probability distribution (non-negative values that sum to 1). Numerically stable (internally subtracts the max).

**Why it's used:** Converting log-weights to normalized weights in a single call.

```python
# src/filters/particle/conditional_smc.py:139
weights = tf.nn.softmax(log_w)
# Convert log-weights to normalized probability weights

# src/filters/particle/bootstrap_pf_hmc.py:143
weights = tf.nn.softmax(log_weights)
```

---

### `tf.nn.sigmoid(x)`

Computes the sigmoid function: `1 / (1 + exp(-x))`.

**What it does:** Maps any real number to the range (0, 1). Large positive x maps close to 1, large negative x maps close to 0.

```python
# src/filters/particle/ledh_invertible_bimodal.py:296
flip_prob = tf.nn.sigmoid(log_score_minus - log_score_plus)
# Convert log-odds to a probability for bimodal mode selection
```

---

### `tf.equal(a, b)`

Element-wise equality comparison. Returns a boolean tensor.

```python
# src/resampling/ot_entropy.py:112
diameter = tf.where(tf.equal(diameter, tf.zeros_like(diameter)), tf.ones_like(diameter), diameter)
# If diameter is 0 (all particles identical), replace with 1 to avoid division by zero
```

---

### `tf.logical_and(a, b)`

Element-wise logical AND on boolean tensors.

```python
# src/resampling/ot_entropy.py:195
return tf.logical_and(not_converged, not_max_iter)
# Continue Sinkhorn iteration only if both: not converged AND not at max iterations
```

---

### `tf.reduce_all(tensor, axis=...)`

Logical AND reduction: True only if ALL elements along the axis are True.

```python
# src/utils/distributions.py:220
is_finite = tf.reduce_all(tf.math.is_finite(weights))
# Check if ALL weights are finite (not NaN or Inf)
```

---

### Trigonometric Functions: `tf.sin(x)`, `tf.cos(x)`, `tf.atan2(y, x)`

Standard trigonometric functions, used in tracking models with bearing observations.

- `tf.sin(x)` / `tf.cos(x)`: Sine and cosine.
- `tf.atan2(y, x)`: Arctangent of `y/x`, returning the angle in `(-pi, pi)`. Handles quadrant correctly (unlike `atan(y/x)`).

```python
# src/models/range_bearing.py:170
bearing_true = tf.atan2(dy, dx)
# Compute bearing angle from position differences

# src/models/kitagawa.py:108
8.0 * tf.cos(1.2 * t_val)
# Time-varying component of the Kitagawa model dynamics

# src/models/two_sensor_bearing.py:260
diff = tf.atan2(tf.sin(diff), tf.cos(diff))
# Normalize angle difference to (-pi, pi) range
```

---

### `tf.pow(base, exponent)`

Raises base to the given exponent: `base^exponent`.

**What it does:** Like `**` operator but as a function call.

```python
# src/DF/hmc_runner.py:492
m_kappa = tf.pow(m, -kappa)
# Step size schedule: m^(-kappa) for dual averaging
```

---

### `tf.keras.optimizers.Adam(learning_rate)` and `tf.keras.optimizers.SGD(learning_rate)`

Gradient-based optimizers from Keras, used for MAP estimation (finding the parameter values that maximize the posterior).

```python
# src/DF/hmc_runner.py:264-266
opt = tf.keras.optimizers.Adam(learning_rate)
# or
opt = tf.keras.optimizers.SGD(learning_rate)
# Used in MAP warm-start: optimize parameters before starting HMC
```

---

## Quick Reference: Where Things Are Used

| Category | Key Functions | Primary Files |
|----------|--------------|---------------|
| Tensor creation | `tf.constant`, `tf.Variable`, `tf.eye`, `tf.TensorArray` | All files |
| Linear algebra | `cholesky`, `cholesky_solve`, `slogdet`, `triangular_solve` | `linalg.py`, `distributions.py`, `flow_params.py` |
| Reductions | `reduce_sum`, `reduce_logsumexp` | `distributions.py`, `ot_entropy.py` |
| Random | `stateless_normal`, `stateless_split` | `distributions.py`, `bootstrap_pf_tf.py` |
| Graph compilation | `@tf.function` | Nearly all files (21+ functions) |
| Auto-diff | `tf.GradientTape` | `hmc_runner.py` |
| Custom gradients | `@tf.custom_gradient` | `linalg.py`, `ot_entropy.py` |
| Distributions | `LogNormal`, `Beta` | `parameter_handler.py`, `example_usage.py` |
| Bijectors | `Softplus`, `Sigmoid`, `Identity` | `parameter_handler.py` |
| MCMC | `NoUTurnSampler`, `HMC`, `DualAveragingStepSizeAdaptation` | `hmc_runner.py` |
| Diagnostics | `effective_sample_size`, `potential_scale_reduction` | `hmc_runner.py`, `pgibbs_runner.py`, `pmmh_runner.py` |
| Device config | `list_physical_devices`, `set_memory_growth` | `device.py` |
| Neural net ops | `tf.nn.softmax`, `tf.nn.sigmoid` | `conditional_smc.py`, `bootstrap_pf_hmc.py`, `ledh_invertible_bimodal.py` |
| Trigonometry | `tf.sin`, `tf.cos`, `tf.atan2` | `range_bearing.py`, `two_sensor_bearing.py`, `kitagawa.py` |
| Optimizers | `tf.keras.optimizers.Adam`, `SGD` | `hmc_runner.py` |
