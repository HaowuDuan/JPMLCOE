# TensorFlow Migration Plan

## Goal
Create a clean, TensorFlow-only codebase for GPU acceleration and automatic differentiation. The NumPy version has been backed up separately.

**Strategy**: Convert existing NumPy implementation to pure TensorFlow without maintaining dual NumPy/TF code.

---

## Current State Analysis

### ✅ Already TensorFlow-Compatible (Reference)

**Resampling (100% TensorFlow):**
- `systematic_resample` - @tf.function
- `soft_resample` - @tf.function (differentiable)
- `ot_entropy_resample` - @tf.function (Sinkhorn algorithm)
- Utils: `effective_sample_size`, `normalize_log_weights`

**Partial TensorFlow:**
- `ParticleFilterTF` (`bootstrap_pf_tf.py`) - Can use as reference
- Models have some TF methods but mixed with NumPy

### ❌ Needs TensorFlow Conversion

**Everything will be converted to pure TensorFlow:**
- **Utilities**: All functions (distributions, linalg, ode_solvers)
- **Models**: All models converted to TensorFlow-only storage and methods
- **Filters**: All filters (Kalman, flow-based, particle)
- **Core**: Base classes

---

## Implementation Strategy

### Phase 1: Core Utilities (Foundation)

Convert utilities to pure TensorFlow (remove all NumPy code).

#### 1.1 Linear Algebra (`src/utils/linalg.py`)

**Replace** NumPy implementation with pure TensorFlow:

```python
import tensorflow as tf

@tf.function
def safe_cholesky_tf(matrix: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    """TensorFlow version of safe Cholesky decomposition."""
    # Add jitter to diagonal for numerical stability
    n = tf.shape(matrix)[-1]
    jitter = tf.eye(n, dtype=matrix.dtype) * epsilon
    return tf.linalg.cholesky(matrix + jitter)

@tf.function
def safe_solve_tf(A: tf.Tensor, b: tf.Tensor, method: str = 'cholesky') -> tf.Tensor:
    """TensorFlow version of safe linear solver."""
    if method == 'cholesky':
        L = safe_cholesky_tf(A)
        return tf.linalg.cholesky_solve(L, b[..., tf.newaxis])[..., 0]
    else:
        # tf.linalg.lstsq returns shape (n,) for input (n,), no need to index
        return tf.linalg.lstsq(A, b[..., tf.newaxis], fast=False)[..., 0]

@tf.function
def log_det_tf(matrix: tf.Tensor) -> tf.Tensor:
    """TensorFlow version of log determinant."""
    sign, logdet = tf.linalg.slogdet(matrix)
    return logdet

@tf.function
def symmetrize_tf(matrix: tf.Tensor) -> tf.Tensor:
    """TensorFlow version of matrix symmetrization."""
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))

@tf.function
def matrix_sqrt_tf(matrix: tf.Tensor) -> tf.Tensor:
    """TensorFlow version of matrix square root via eigendecomposition."""
    s, u, v = tf.linalg.svd(matrix)
    sqrt_s = tf.sqrt(tf.maximum(s, 0.0))
    return u @ tf.linalg.diag(sqrt_s) @ tf.linalg.matrix_transpose(v)
```

**Files to modify:**
- `code/src/utils/linalg.py` - **Replace** with TensorFlow-only functions (remove `_tf` suffix)
- `code/src/utils/__init__.py` - Export TensorFlow functions

---

#### 1.2 Distributions (`src/utils/distributions.py`)

**Replace** NumPy implementation with pure TensorFlow:

```python
import tensorflow as tf

@tf.function
def log_gaussian_prob(x: tf.Tensor, mean: tf.Tensor, cov: tf.Tensor) -> tf.Tensor:
    """
    TensorFlow version of log Gaussian probability.

    Args:
        x: Data point(s) of shape (..., n)
        mean: Mean of shape (..., n)
        cov: Covariance of shape (..., n, n)

    Returns:
        Log probability of shape (...)
    """
    diff = x - mean
    n = tf.cast(tf.shape(x)[-1], x.dtype)

    # Compute log determinant
    sign, logdet = tf.linalg.slogdet(cov)

    # Compute Mahalanobis distance
    L = tf.linalg.cholesky(cov)
    y = tf.linalg.triangular_solve(L, diff[..., tf.newaxis], lower=True)[..., 0]
    mahalanobis = tf.reduce_sum(y**2, axis=-1)

    return -0.5 * (n * tf.math.log(2.0 * tf.constant(3.14159265359, dtype=x.dtype)) + logdet + mahalanobis)

@tf.function
def log_sum_exp(log_values: tf.Tensor, axis: int = None) -> tf.Tensor:
    """TensorFlow version of log-sum-exp."""
    return tf.reduce_logsumexp(log_values, axis=axis)

@tf.function
def normalize_log_weights(log_weights: tf.Tensor, clip_range: tuple = None) -> tf.Tensor:
    """
    TensorFlow version of log weight normalization.

    Args:
        log_weights: Log weights of shape (..., N)
        clip_range: Optional (min, max) for clipping

    Returns:
        Normalized weights (not in log space)
    """
    max_log = tf.reduce_max(log_weights, axis=-1, keepdims=True)
    log_weights_normalized = log_weights - max_log

    if clip_range is not None:
        log_weights_normalized = tf.clip_by_value(
            log_weights_normalized, clip_range[0], clip_range[1]
        )

    weights_unnorm = tf.exp(log_weights_normalized)
    return weights_unnorm / tf.reduce_sum(weights_unnorm, axis=-1, keepdims=True)

@tf.function
def multivariate_normal_sample(mean: tf.Tensor, cov: tf.Tensor,
                                n_samples: int, seed: tf.Tensor) -> tf.Tensor:
    """
    TensorFlow version of multivariate normal sampling.

    Args:
        mean: Mean of shape (d,)
        cov: Covariance of shape (d, d)
        n_samples: Number of samples
        seed: Random seed for stateless sampling

    Returns:
        Samples of shape (n_samples, d)
    """
    d = tf.shape(mean)[0]
    L = tf.linalg.cholesky(cov)
    z = tf.random.stateless_normal([n_samples, d], seed=seed, dtype=mean.dtype)
    # Correct batch multiplication: z @ L^T
    return mean + tf.linalg.matmul(z, L, transpose_b=True)

@tf.function
def compute_flow_weights(
    eta_1: tf.Tensor,
    eta_0: tf.Tensor,
    particles_prev: tf.Tensor,
    observation: tf.Tensor,
    model,
    prev_weights: tf.Tensor = None,
    jacobians: tf.Tensor = None,
    clip_range: tuple = (-30.0, 30.0)
) -> tf.Tensor:
    """
    TensorFlow version of compute_flow_weights for invertible flow filters.

    This is a CRITICAL function - enables TensorFlow acceleration of flow filters.

    Args:
        eta_1: Flowed particles at λ=1, shape (N, d)
        eta_0: Sampled particles at λ=0, shape (N, d)
        particles_prev: Particles from previous timestep, shape (N, d)
        observation: Current observation, shape (obs_dim,)
        model: Model with TensorFlow batch methods
        prev_weights: Previous weights, shape (N,)
        jacobians: Jacobian determinants, shape (N,)
        clip_range: Clipping range for log weights

    Returns:
        Normalized weights, shape (N,)
    """
    n_particles = tf.shape(eta_1)[0]
    state_dim = tf.shape(eta_1)[1]

    if prev_weights is None:
        prev_weights = tf.ones(n_particles, dtype=eta_1.dtype) / tf.cast(n_particles, eta_1.dtype)

    if jacobians is None:
        jacobians = tf.ones(n_particles, dtype=eta_1.dtype)

    # Vectorized path (assumes state-independent Q)
    # 1. Batch transition means
    f_prev = model.state_transition_mean_batch(particles_prev)

    # 2. Single Q matrix
    Q = model.state_transition_cov_batch(particles_prev)

    # 3. Batch observation log-probs
    log_p_obs = model.log_observation_prob_batch(observation, eta_1)

    # 4. Vectorized log p(η₁ | x_{k-1})
    diff_1 = eta_1 - f_prev
    L_Q = tf.linalg.cholesky(Q)
    y_1 = tf.linalg.triangular_solve(L_Q, tf.transpose(diff_1), lower=True)
    y_1 = tf.transpose(y_1)

    log_p_eta1 = -0.5 * (
        tf.reduce_sum(y_1**2, axis=1) +
        2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Q))) +
        tf.cast(state_dim, eta_1.dtype) * tf.math.log(2.0 * tf.constant(3.14159265359, dtype=eta_1.dtype))
    )

    # 5. Vectorized log p(η₀ | x_{k-1})
    diff_0 = eta_0 - f_prev
    y_0 = tf.linalg.triangular_solve(L_Q, tf.transpose(diff_0), lower=True)
    y_0 = tf.transpose(y_0)

    log_p_eta0 = -0.5 * (
        tf.reduce_sum(y_0**2, axis=1) +
        2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Q))) +
        tf.cast(state_dim, eta_1.dtype) * tf.math.log(2.0 * tf.constant(3.14159265359, dtype=eta_1.dtype))
    )

    # 6. Combine in log space
    log_weights = (
        log_p_eta1 +
        log_p_obs +
        tf.math.log(tf.maximum(jacobians, 1e-300)) -
        log_p_eta0 +
        tf.math.log(tf.maximum(prev_weights, 1e-300))
    )

    # Normalize
    weights = normalize_log_weights(log_weights, clip_range=clip_range)

    # Check for weight collapse
    is_finite = tf.reduce_all(tf.math.is_finite(weights))
    uniform_weights = tf.ones(n_particles, dtype=eta_1.dtype) / tf.cast(n_particles, eta_1.dtype)
    weights = tf.cond(is_finite, lambda: weights, lambda: uniform_weights)

    return weights
```

**Files to modify:**
- `code/src/utils/distributions.py` - **Replace** with TensorFlow-only functions (remove `_tf` suffix)
- `code/src/utils/__init__.py` - Export TensorFlow functions

---

#### 1.3 ODE Solvers (`src/utils/ode_solvers.py`)

**Replace** NumPy implementation with pure TensorFlow:

```python
import tensorflow as tf
from typing import Callable

@tf.function
def euler_step(x: tf.Tensor, f: Callable, dt: float, *args) -> tf.Tensor:
    """Euler integration step."""
    return x + f(x, *args) * dt

@tf.function
def rk4_step(x: tf.Tensor, f: Callable, dt: float, *args,
             t: float = None) -> tf.Tensor:
    """RK4 integration step."""
    if t is not None:
        # Time-dependent drift
        k1 = f(x, t, *args)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt, *args)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt, *args)
        k4 = f(x + dt * k3, t + dt, *args)
    else:
        k1 = f(x, *args)
        k2 = f(x + 0.5 * dt * k1, *args)
        k3 = f(x + 0.5 * dt * k2, *args)
        k4 = f(x + dt * k3, *args)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@tf.function
def euler_maruyama_step(x: tf.Tensor, f: Callable, dt: float, *args,
                         diffusion_coeff: float = 0.0,
                         seed: tf.Tensor = None) -> tf.Tensor:
    """Euler-Maruyama SDE integration."""
    drift = f(x, *args)

    if diffusion_coeff > 0 and seed is not None:
        noise = tf.random.stateless_normal(tf.shape(x), seed=seed, dtype=x.dtype)
        noise = noise * tf.sqrt(dt) * tf.sqrt(diffusion_coeff)
        return x + drift * dt + noise
    else:
        return x + drift * dt
```

**Files to modify:**
- `code/src/utils/ode_solvers.py` - **Replace** with TensorFlow-only integrators (remove `_tf` suffix)
- `code/src/utils/__init__.py` - Export TensorFlow functions

---

### Phase 2: Models (TensorFlow-Only)

Convert all models to pure TensorFlow (remove NumPy storage and methods).

#### 2.1 Base Class (`src/models/base.py`)

**Convert to pure TensorFlow** - remove all NumPy, change abstract methods to expect TensorFlow tensors:

```python
import tensorflow as tf
from abc import ABC, abstractmethod

class StateSpaceModel(ABC):
    """Base class for TensorFlow state-space models."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state space."""
        pass

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of observation space."""
        pass

    @abstractmethod
    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        """Sample from initial state distribution (TensorFlow)."""
        pass

    @abstractmethod
    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample next state given current state (TensorFlow)."""
        pass

    @abstractmethod
    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample observation given state (TensorFlow)."""
        pass

    # Batch methods (all TensorFlow)
    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Default: loop over particles."""
        return tf.stack([self.state_transition_mean(x) for x in particles])

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Default: constant Q."""
        return self.state_transition_cov(particles[0])

    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Default: loop over particles."""
        return tf.stack([self.log_observation_prob(observation, x) for x in particles])
```

#### 2.2 Convert All Models to TensorFlow-Only

**Key changes for each model:**
1. **Remove NumPy storage** - only keep TensorFlow tensors (e.g., `self.F` becomes TensorFlow, remove `self.F_tf`)
2. **Remove `_tf` suffix** from all methods
3. **Remove `if TF_AVAILABLE:` guards**
4. **Use TensorFlow seed** instead of `np.random.Generator`

**Example: LinearGaussianModel**
```python
class LinearGaussianModel(StateSpaceModel):
    def __init__(self, F, B, H, D, mu_0=None, Sigma_0=None):
        # Convert to TensorFlow and compute dimensions
        self.F = tf.constant(F, dtype=tf.float32)
        self.B = tf.constant(B, dtype=tf.float32)
        self.H = tf.constant(H, dtype=tf.float32)
        self.D = tf.constant(D, dtype=tf.float32)

        # Store dimensions
        self.nx = self.F.shape[0]  # State dimension
        self.nv = self.B.shape[1]  # Process noise dimension
        self.ny = self.H.shape[0]  # Observation dimension
        self.nw = self.D.shape[1]  # Observation noise dimension

        # Compute covariances
        self.Q = self.B @ tf.transpose(self.B)
        self.R = self.D @ tf.transpose(self.D)

        # Initial state (compute dimensions first, then use)
        if mu_0 is None:
            self.mu_0 = tf.zeros(self.nx, dtype=tf.float32)
        else:
            self.mu_0 = tf.constant(mu_0, dtype=tf.float32)

        if Sigma_0 is None:
            self.Sigma_0 = tf.eye(self.nx, dtype=tf.float32)
        else:
            self.Sigma_0 = tf.constant(Sigma_0, dtype=tf.float32)

    @tf.function
    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Sample: X' = F·X + B·v, v ~ N(0, I)."""
        v = tf.random.stateless_normal([self.nv], seed=seed)
        return tf.linalg.matvec(self.F, x) + tf.linalg.matvec(self.B, v)

    @tf.function
    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized: particles @ F^T (more efficient than transposing twice)."""
        return particles @ tf.transpose(self.F)

    @tf.function
    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized Gaussian log-prob."""
        means = particles @ tf.transpose(self.H)
        diff = observation - means
        L_R = tf.linalg.cholesky(self.R)
        y = tf.linalg.triangular_solve(L_R, tf.transpose(diff), lower=True)
        mahalanobis = tf.reduce_sum(y**2, axis=0)
        logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_R)))
        return -0.5 * (tf.cast(self.obs_dim, observation.dtype) * tf.math.log(2.0 * 3.14159265359) + logdet + mahalanobis)
```

**Apply same pattern to:**
- RangeBearingModel
- AcousticTrackingModel (base, lite, full)
- TwoSensorBearingModel
- StochasticVolatilityModel
- Lorenz96Model

---

### Phase 3: Filters (TensorFlow-Only)

Convert all filters to pure TensorFlow.

#### 3.1 Kalman Filter (`src/filters/kalman/kalman.py`)

**Replace** NumPy implementation with pure TensorFlow:

```python
import tensorflow as tf

class KalmanFilter:
    """TensorFlow-only Kalman Filter."""

    def __init__(self, model):
        """
        Initialize Kalman Filter with a linear-Gaussian model.

        Args:
            model: LinearGaussianModel with TensorFlow constants
        """
        self.model = model
        self.mean = None
        self.cov = None

    @tf.function
    def predict(self, mean: tf.Tensor, cov: tf.Tensor) -> tuple:
        """
        Prediction step: x̂_{k|k-1} = F·x̂_{k-1|k-1}.

        Args:
            mean: Current mean estimate (state_dim,)
            cov: Current covariance (state_dim, state_dim)

        Returns:
            (predicted_mean, predicted_cov)
        """
        F = self.model.F
        Q = self.model.Q

        mean_pred = tf.linalg.matvec(F, mean)
        cov_pred = F @ cov @ tf.transpose(F) + Q

        return mean_pred, cov_pred

    @tf.function
    def update(self, mean_pred: tf.Tensor, cov_pred: tf.Tensor,
               observation: tf.Tensor) -> tuple:
        """
        Update step using observation.

        Args:
            mean_pred: Predicted mean (state_dim,)
            cov_pred: Predicted covariance (state_dim, state_dim)
            observation: Observation (obs_dim,)

        Returns:
            (updated_mean, updated_cov)
        """
        H = self.model.H
        R = self.model.R

        # Innovation
        y_pred = tf.linalg.matvec(H, mean_pred)
        innovation = observation - y_pred

        # Innovation covariance
        S = H @ cov_pred @ tf.transpose(H) + R

        # Kalman gain
        K = cov_pred @ tf.transpose(H) @ tf.linalg.inv(S)

        # Update
        mean_updated = mean_pred + tf.linalg.matvec(K, innovation)
        cov_updated = (tf.eye(self.model.state_dim, dtype=mean_pred.dtype) -
                       K @ H) @ cov_pred

        return mean_updated, cov_updated

    @tf.function
    def filter_tf(self, observations: tf.Tensor,
                   initial_mean: tf.Tensor = None,
                   initial_cov: tf.Tensor = None) -> tuple:
        """
        Run Kalman filter on observation sequence.

        Args:
            observations: Observations of shape (T, obs_dim)
            initial_mean: Initial mean (state_dim,). If None, uses model default.
            initial_cov: Initial covariance (state_dim, state_dim). If None, uses model default.

        Returns:
            (means, covs) each of shape (T, state_dim) and (T, state_dim, state_dim)
        """
        if initial_mean is None:
            initial_mean = self.model.mu_0
        if initial_cov is None:
            initial_cov = self.model.Sigma_0

        T = tf.shape(observations)[0]
        state_dim = self.model.state_dim

        # Allocate output
        means = tf.TensorArray(dtype=initial_mean.dtype, size=T)
        covs = tf.TensorArray(dtype=initial_cov.dtype, size=T)

        mean = initial_mean
        cov = initial_cov

        for t in tf.range(T):
            # Predict
            mean_pred, cov_pred = self.predict(mean, cov)

            # Update
            mean, cov = self.update(mean_pred, cov_pred, observations[t])

            means = means.write(t, mean)
            covs = covs.write(t, cov)

        return means.stack(), covs.stack()
```

**Files to modify:**
- `code/src/filters/kalman/kalman.py` - **Replace** with TensorFlow-only implementation

#### 3.2 Extended Kalman Filter (`src/filters/kalman/extended_kalman.py`)

**Replace** NumPy with TensorFlow. Remove `_tf` suffix from all methods.

#### 3.3 Unscented Kalman Filter (`src/filters/kalman/unscented_kalman.py`)

**Replace** NumPy with TensorFlow. Implement unscented transform using TensorFlow ops.

---

### Phase 4: Flow Filters (TensorFlow-Only)

Convert flow filters to pure TensorFlow.

#### 4.1 EDH Flow (`src/filters/particle/edh_flow.py`)

**Replace** NumPy implementation with pure TensorFlow:

```python
import tensorflow as tf
from ...utils.ode_solvers import euler_step, rk4_step

class EDHFlow:
    """TensorFlow-only Exact Daum-Huang Flow filter."""

    def __init__(self, model, n_particles: int, n_lambda_steps: int,
                 integration_method: str = 'euler',
                 filter_type: str = 'ekf'):
        self.model = model
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.integration_method = integration_method
        self.filter_type = filter_type

    @tf.function
    def _compute_flow_params(self, eta_bar: tf.Tensor, lambda_val: float,
                              observation: tf.Tensor) -> tuple:
        """
        Compute flow parameters A and b at given lambda.

        Uses global filter (EKF or UKF) for linearization.

        Returns:
            A: Flow matrix (state_dim, state_dim)
            b: Flow vector (state_dim,)
        """
        # Compute observation Jacobian H and mean h(eta_bar)
        H = self.model.observation_jacobian(eta_bar)  # (obs_dim, state_dim)
        h_mean = self.model.observation_mean(eta_bar)  # (obs_dim,)
        R = self.model.observation_cov(eta_bar)  # (obs_dim, obs_dim)

        # Compute P_bar (ensemble covariance) if using EKF-style linearization
        # Or use UKF/global filter covariance estimate

        # Flow parameters (simplified - actual implementation depends on filter_type)
        # A = lambda * H^T R^{-1} H
        # b = lambda * H^T R^{-1} (z - h(eta_bar))
        R_inv = tf.linalg.inv(R)
        A = lambda_val * tf.transpose(H) @ R_inv @ H
        innovation = observation - h_mean
        b = lambda_val * tf.transpose(H) @ R_inv @ innovation

        return A, b

    @tf.function
    def _compute_drift(self, particles: tf.Tensor, A: tf.Tensor,
                       b: tf.Tensor) -> tf.Tensor:
        """Drift function: dx/dλ = Ax + b."""
        return tf.linalg.matvec(A, particles, transpose_a=True) + b

    @tf.function
    def update_tf(self, particles: tf.Tensor, observation: tf.Tensor,
                   seed: tf.Tensor) -> tf.Tensor:
        """
        Flow particles from λ=0 to λ=1.

        Args:
            particles: Current particles (N, state_dim)
            observation: Observation (obs_dim,)
            seed: Random seed for stateless operations

        Returns:
            Flowed particles (N, state_dim)
        """
        # Initialize flow
        particles_flow = particles

        # Lambda schedule (exponential)
        lambda_vals = tf.exp(tf.linspace(tf.math.log(1e-6), 0.0, self.n_lambda_steps))

        for j in tf.range(self.n_lambda_steps - 1):
            lambda_j = lambda_vals[j]
            d_lambda = lambda_vals[j + 1] - lambda_j

            # Compute ensemble mean
            eta_bar = tf.reduce_mean(particles_flow, axis=0)

            # Compute flow parameters
            A, b = self._compute_flow_params(eta_bar, lambda_j, observation)

            # Integrate ODE
            if self.integration_method == 'euler':
                particles_flow = euler_step(
                    particles_flow, self._compute_drift, d_lambda, A, b
                )
            elif self.integration_method == 'rk4':
                particles_flow = rk4_step(
                    particles_flow, self._compute_drift_time_dependent,
                    d_lambda, observation, t=lambda_j
                )

        return particles_flow
```

**Files to modify:**
- `code/src/filters/particle/edh_flow.py` - **Replace** with TensorFlow-only

#### 4.2 EDH Invertible (`src/filters/particle/edh_invertible.py`)

**Replace** NumPy with TensorFlow:

```python
@tf.function
def update(self, observation: tf.Tensor, seed: tf.Tensor) -> None:
    """Update with invertible flow and importance resampling."""
    # Sample eta_0 from transition distribution
    seeds = tf.random.experimental.stateless_split(seed, self.n_particles)
    eta_0 = tf.stack([
        self.model.sample_state_transition(self.particles[i], seeds[i])
        for i in tf.range(self.n_particles)  # Use tf.range for graph compilation
    ])

    # Flow from λ=0 to λ=1
    eta_1 = self._flow_particles(eta_0, observation, seed)

    # Compute importance weights using TensorFlow
    from ...utils.distributions import compute_flow_weights
    self.weights = compute_flow_weights(
        eta_1=eta_1,
        eta_0=eta_0,
        particles_prev=self.particles,
        observation=observation,
        model=self.model,
        prev_weights=self.weights
    )

    # Update particles
    self.particles = eta_1
```

**Files to modify:**
- `code/src/filters/particle/edh_invertible.py` - **Replace** with TensorFlow-only

#### 4.3 Stochastic EDH (`src/filters/particle/stochastic_edh.py`)

**Replace** with TensorFlow. Use `euler_maruyama_step` for SDE integration.

#### 4.4 LEDH Variants

**Files to modify:**
- `code/src/filters/particle/ledh_flow.py` - **Replace** with TensorFlow-only
- `code/src/filters/particle/ledh_invertible.py` - **Replace** with TensorFlow-only

---

## File Structure

All files converted to pure TensorFlow (NumPy version backed up separately):

```
code/
├── src/
│   ├── utils/
│   │   ├── __init__.py                    [MODIFY]
│   │   ├── linalg.py                      [REPLACE with TensorFlow-only]
│   │   ├── distributions.py               [REPLACE with TensorFlow-only]
│   │   └── ode_solvers.py                 [REPLACE with TensorFlow-only]
│   ├── models/
│   │   ├── base.py                        [REPLACE with TensorFlow-only]
│   │   ├── linear_gaussian.py             [REPLACE with TensorFlow-only]
│   │   ├── range_bearing.py               [REPLACE with TensorFlow-only]
│   │   ├── acoustic_tracking.py           [REPLACE with TensorFlow-only]
│   │   ├── acoustic_tracking_lite.py      [REPLACE with TensorFlow-only]
│   │   ├── acoustic_tracking_full.py      [REPLACE with TensorFlow-only]
│   │   ├── two_sensor_bearing.py          [REPLACE with TensorFlow-only]
│   │   ├── stochastic_volatility.py       [REPLACE with TensorFlow-only]
│   │   └── lorenz96.py                    [REPLACE with TensorFlow-only]
│   └── filters/
│       ├── kalman/
│       │   ├── kalman.py                  [REPLACE with TensorFlow-only]
│       │   ├── extended_kalman.py         [REPLACE with TensorFlow-only]
│       │   └── unscented_kalman.py        [REPLACE with TensorFlow-only]
│       └── particle/
│           ├── edh_flow.py                [REPLACE with TensorFlow-only]
│           ├── edh_invertible.py          [REPLACE with TensorFlow-only]
│           ├── stochastic_edh.py          [REPLACE with TensorFlow-only]
│           ├── ledh_flow.py               [REPLACE with TensorFlow-only]
│           └── ledh_invertible.py         [REPLACE with TensorFlow-only]
```

---

## Implementation Priority

### Phase 1: Core Utilities
Priority: **CRITICAL** - All other phases depend on this.

1. Convert `linalg.py` to TensorFlow-only (safe_cholesky, safe_solve, etc.)
2. Convert `distributions.py` to TensorFlow-only (compute_flow_weights is most important!)
3. Convert `ode_solvers.py` to TensorFlow-only (euler_step, rk4_step, euler_maruyama_step)
4. Update exports in `__init__.py`

### Phase 2: Models
Priority: **HIGH** - Required for flow filters.

1. Convert `base.py` to TensorFlow-only
2. Convert all concrete models to TensorFlow-only:
   - LinearGaussianModel
   - RangeBearingModel
   - AcousticTrackingModel (base, lite, full)
   - TwoSensorBearingModel
   - StochasticVolatilityModel
   - Lorenz96Model

### Phase 3: Kalman Filters
Priority: **MEDIUM** - Independent from flow filters.

1. Convert `kalman.py` to TensorFlow-only
2. Convert `extended_kalman.py` to TensorFlow-only
3. Convert `unscented_kalman.py` to TensorFlow-only

### Phase 4: Flow Filters
Priority: **HIGH** - Main goal for performance.

1. Convert `EDHFlow` to TensorFlow-only
2. Convert `EDHInvertible` to TensorFlow-only (uses `compute_flow_weights`)
3. Convert `StochasticEDH` to TensorFlow-only
4. Convert LEDH variants to TensorFlow-only

---

## Expected Performance Gains

**TensorFlow Advantages:**
- GPU acceleration (10-100x for large particle counts)
- Automatic differentiation (enables gradient-based optimization)
- XLA compilation for further speedup
- Vectorization optimized by TF kernel fusion

**Specific Targets:**
- Kalman filters: 5-20x speedup on GPU
- Particle filters: 10-50x speedup (already proven with `ParticleFilterTF`)
- Flow filters: **NEW** - 20-100x speedup expected (currently NumPy-only)
- Utilities: 2-10x speedup (BLAS-optimized operations)

---

## Migration Strategy

**Clean TensorFlow-Only Approach:**
- **Replace all NumPy code** with pure TensorFlow
- **Remove `_tf` suffixes** - function names become clean (e.g., `log_gaussian_prob`, `filter`)
- **Remove `if TF_AVAILABLE:` guards** - TensorFlow is required
- **Use `tf.Tensor` everywhere** - no NumPy arrays, no dual storage
- **Stateless random sampling** - use `tf.random.stateless_*` with seed parameters
- **@tf.function decorators** - enable graph compilation and GPU acceleration

**Key Changes:**
1. **Storage**: `self.F = tf.constant(F)` instead of `self.F = np.array(F)` + `self.F_tf = tf.constant(...)`
2. **Methods**: `def sample(x: tf.Tensor, seed: tf.Tensor)` instead of dual NumPy/TF versions
3. **Imports**: `import tensorflow as tf` only, remove NumPy imports from core
4. **Token savings**: 40-50% reduction in code size

---

## Dependencies

**Required:**
- `tensorflow >= 2.10.0` (for GPU support and modern APIs)
- `tensorflow-probability >= 0.18.0` (for statistical distributions, already in use)

---

## Implementation Checklist

For each file conversion:

- [ ] Remove all NumPy imports and arrays
- [ ] Convert to `tf.Tensor` throughout
- [ ] Remove `_tf` suffixes from function/method names
- [ ] Add `@tf.function` decorators for performance
- [ ] Use stateless random sampling (`tf.random.stateless_*`)
- [ ] Remove `if TF_AVAILABLE:` guards
- [ ] Update function signatures to accept `tf.Tensor` and seed parameters
- [ ] Ensure GPU compatibility (avoid CPU-only ops)

---

## Expected Benefits

1. **Token Efficiency**: 40-50% reduction in code size
2. **Performance**: 10-100x speedup on GPU for large particle counts
3. **Maintainability**: Single clean implementation, no dual code paths
4. **Auto-differentiation**: Enable gradient-based optimization
5. **Graph compilation**: XLA optimization for further speedup
