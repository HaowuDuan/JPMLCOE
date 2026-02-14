# HMC Bayesian Inference — Comprehensive Action Plan

This document provides a phased implementation plan to make HMC parameter inference work end-to-end. Each step includes exact code changes, files affected, and reasoning.

**Prerequisite reading:** `HMC_READINESS_REPORT.md`, `FLOAT64_REGRESSION_PLAN.md`

---

## Three-Phase Strategy

| Phase | Goal | Model + Filter | Why |
|-------|------|----------------|-----|
| **Phase 1** | Validate HMC infrastructure | LinearGaussian + KalmanFilter (via EKF) | Both fully TF, KF is optimal for this model — guaranteed good fit. Gold standard validation. |
| **Phase 2** | Optimize LEDH numerics | (FLOAT64 regression plan) | Must fix float64 issues before LEDH can be trusted for HMC |
| **Phase 3** | Nonlinear HMC demo | Kitagawa + LEDH invertible | Highly nonlinear model, requires particle flow — the real target |

---

## Table of Contents

### Phase 1: LinearGaussian + KalmanFilter (Core HMC Validation)
1. [Step 1.1: Differentiable Filter Protocol](#step-1-1)
2. [Step 1.2: Fix DifferentiableModel — tf.Variable.assign()](#step-1-2)
3. [Step 1.3: Rewrite HMC Runner — Remove tf.py_function, Add NUTS](#step-1-3)
4. [Step 1.4: Modify LinearGaussianModel for Scalar Noise Parameters](#step-1-4)
5. [Step 1.5: Add log_marginal_likelihood_tf to KalmanFilter and EKF](#step-1-5)
6. [Step 1.6: Phase 1 Smoke Test — Infer noise scales](#step-1-6)
7. [Step 1.7: Phase 1 Hydra Config](#step-1-7)

### Phase 2: LEDH Float64 Optimization (from FLOAT64_REGRESSION_PLAN)
8. [Step 2.1: Controlled Float32/Float64 Comparison](#step-2-1)
9. [Step 2.2: Debug Per-Particle Covariances (if needed)](#step-2-2)
10. [Step 2.3: Multi-Seed Statistical Comparison](#step-2-3)
11. [Step 2.4: Parameter Re-tuning for TensorFlow](#step-2-4)

### Phase 3: Kitagawa + LEDH Invertible (Nonlinear HMC)
12. [Step 3.1: TF-ify KitagawaModel](#step-3-1)
13. [Step 3.2: Add log_marginal_likelihood_tf to LEDH Invertible](#step-3-2)
14. [Step 3.3: Phase 3 Smoke Test — Kitagawa + LEDH](#step-3-3)
15. [Step 3.4: Phase 3 Hydra Config](#step-3-4)

### Infrastructure (shared across phases)
16. [DPF Experiment Runner](#step-runner)
17. [Dependency Graph & Implementation Order](#dependency-graph)

---

# PHASE 1: LinearGaussian + KalmanFilter
**Goal:** Validate the entire HMC pipeline on the simplest possible case.

The Kalman filter is *optimal* for linear Gaussian models, so the log-likelihood surface is smooth and well-conditioned. If HMC can't recover the true noise parameters here, the infrastructure is broken.

**Parameters to infer:** `process_noise_std` (sigma_q) and `obs_noise_std` (sigma_r)

---

## Step 1.1: Differentiable Filter Protocol
<a name="step-1-1"></a>

### Problem
No filter exposes a TF-traceable function `(observations) -> log_likelihood_tensor`.

### File: `code/src/core/differentiable.py` (NEW)

```python
"""Protocol for filters that support differentiable log-likelihood computation."""

import tensorflow as tf
from typing import Protocol, runtime_checkable


@runtime_checkable
class DifferentiableFilter(Protocol):
    """
    Protocol for filters usable with HMC parameter inference.

    Any filter that implements this protocol can be plugged into the DPF
    framework. The key requirement: log_marginal_likelihood_tf() runs
    entirely inside TensorFlow's computation graph so tf.GradientTape
    can differentiate through it.

    Contract:
        - All operations must be TF ops (no .numpy(), no Python float)
        - Returns a tf.Tensor scalar, not a Python float
        - Deterministic given the same random seed
    """

    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor
    ) -> tf.Tensor:
        """
        Run the full filter, return total log p(y_{1:T}) as a TF scalar.

        Args:
            observations: (T, obs_dim), dtype matching model
            seed: TF random seed (2,), dtype int32

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        ...
```

---

## Step 1.2: Fix DifferentiableModel — tf.Variable.assign()
<a name="step-1-2"></a>

### Problem
`DifferentiableModel.update_parameters()` calls `.numpy()` and `float()`, severing the TF graph:
```python
# CURRENT (broken) — code/src/DF/differentiable_model.py:55-60
value = float(param_value.numpy())  # kills gradient tape
```

### File: `code/src/DF/differentiable_model.py` (REPLACE)

```python
"""Wrapper for models to support differentiable parameter updates during HMC."""

import tensorflow as tf
from typing import List, Dict, Any
import copy


class DifferentiableModel:
    """
    Wrapper that replaces trainable parameters with tf.Variable objects.
    When HMC proposes new values, tf.Variable.assign() updates them
    WITHOUT leaving the TF computation graph.

    The model's methods that reference self.sigma_q etc. will automatically
    see updated values because those attributes are now tf.Variable.
    """

    def __init__(self, base_model: Any, trainable_params: List[str]):
        object.__setattr__(self, '_base_model', base_model)
        object.__setattr__(self, '_trainable_params', trainable_params)
        object.__setattr__(self, '_original_values', {})
        object.__setattr__(self, '_variables', {})

        dtype = getattr(base_model, 'dtype', tf.float32)
        object.__setattr__(self, '_dtype', dtype)

        for param_name in trainable_params:
            if not hasattr(base_model, param_name):
                raise ValueError(f"Model does not have parameter: {param_name}")

            original_value = getattr(base_model, param_name)
            self._original_values[param_name] = copy.deepcopy(original_value)

            # Create tf.Variable — stays on the computation graph
            var = tf.Variable(
                tf.constant(float(original_value), dtype=dtype),
                name=f"dpf_{param_name}",
                trainable=False  # HMC manages updates, not TF optimizer
            )
            self._variables[param_name] = var

            # Replace model attribute with the Variable
            setattr(base_model, param_name, var)

    def update_parameters(self, param_dict: Dict[str, tf.Tensor]) -> None:
        """Update via tf.Variable.assign() — no .numpy(), stays on graph."""
        for param_name, param_value in param_dict.items():
            if param_name not in self._trainable_params:
                raise ValueError(f"Parameter {param_name} is not trainable")
            self._variables[param_name].assign(tf.cast(param_value, self._dtype))

    def restore_parameters(self) -> None:
        """Restore original Python values (after HMC completes)."""
        for param_name, original_value in self._original_values.items():
            setattr(self._base_model, param_name, copy.deepcopy(original_value))

    def get_variables(self) -> Dict[str, tf.Variable]:
        return dict(self._variables)

    def get_current_parameters(self) -> Dict[str, float]:
        return {name: float(var.numpy()) for name, var in self._variables.items()}

    def __getattr__(self, name: str) -> Any:
        if name in ('_base_model', '_trainable_params', '_original_values',
                     '_variables', '_dtype'):
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, '_base_model'), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_base_model', '_trainable_params', '_original_values',
                     '_variables', '_dtype'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_base_model'), name, value)

    def __repr__(self) -> str:
        base = object.__getattribute__(self, '_base_model')
        params = object.__getattribute__(self, '_trainable_params')
        return f"DifferentiableModel(base={base.__class__.__name__}, params={params})"
```

---

## Step 1.3: Rewrite HMC Runner — Remove tf.py_function, Add NUTS
<a name="step-1-3"></a>

### Problem
`_negative_log_posterior` wraps the filter in `tf.py_function` (not differentiable).

### File: `code/src/DF/hmc_runner.py` (REPLACE)

```python
"""Main HMC/NUTS runner for parameter inference with differentiable filters."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Any, Dict, Optional, Type
import warnings

from .types import ParameterSpec, DPFResult
from .parameter_handler import ParameterHandler
from .differentiable_model import DifferentiableModel


class DPFRunner:
    """
    Differentiable Filter runner for parameter inference via HMC/NUTS.

    Architecture:
        1. DifferentiableModel replaces trainable params with tf.Variable
        2. ParameterHandler manages bijector transforms
        3. Filter.log_marginal_likelihood_tf() runs entirely in TF graph
        4. _negative_log_posterior() calls filter directly — no tf.py_function
        5. HMC/NUTS differentiates through log posterior via tf.GradientTape
    """

    def __init__(
        self,
        base_model: Any,
        filter_class: Type,
        filter_kwargs: Dict[str, Any],
        param_specs: Dict[str, ParameterSpec],
        sampler: str = 'nuts'
    ):
        self.base_model = base_model
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        self.sampler = sampler.lower()

        # Wrap model: replaces params with tf.Variable
        trainable_param_names = list(param_specs.keys())
        self.diff_model = DifferentiableModel(base_model, trainable_param_names)

        # Bijectors and priors
        self.param_handler = ParameterHandler(param_specs)

        # Create filter ONCE with the wrapped model
        self.filter_obj = self.filter_class(self.diff_model, **self.filter_kwargs)

        self._observations_tf = None

    def _negative_log_posterior(self, unconstrained_params: tf.Tensor) -> tf.Tensor:
        """
        Compute -log p(theta | y) = -log p(y | theta) - log p(theta).

        Runs ENTIRELY inside TF's computation graph.
        No tf.py_function, no .numpy(), no Python float conversion.
        """
        # 1. Bijectors: unconstrained -> constrained
        constrained_params = self.param_handler.constrain(unconstrained_params)

        # 2. tf.Variable.assign() — stays on graph
        self.diff_model.update_parameters(constrained_params)

        # 3. Filter forward pass (entirely in TF)
        seed = tf.constant([42, 0], dtype=tf.int32)
        log_likelihood = self.filter_obj.log_marginal_likelihood_tf(
            self._observations_tf, seed=seed
        )

        # 4. Log prior with Jacobian adjustment
        log_prior = self.param_handler.log_prior(constrained_params)

        return -(log_likelihood + log_prior)

    def run_inference(
        self,
        observations: np.ndarray,
        num_samples: int = 1000,
        num_burnin: int = 500,
        step_size: float = 0.01,
        num_leapfrog_steps: int = 10,
        adaptation_rate: float = 0.8,
        target_accept_prob: float = 0.75,
        seed: Optional[int] = None,
        max_tree_depth: int = 10
    ) -> DPFResult:
        """Run HMC or NUTS to sample from posterior p(theta | y)."""
        dtype = getattr(self.base_model, 'dtype', tf.float32)
        self._observations_tf = tf.constant(observations, dtype=dtype)

        def target_log_prob_fn(unconstrained_params):
            return -self._negative_log_posterior(unconstrained_params)

        # Choose sampler
        if self.sampler == 'nuts':
            print(f"Using NUTS sampler (max_tree_depth={max_tree_depth})")
            inner_kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                max_tree_depth=max_tree_depth
            )
        else:
            print(f"Using HMC sampler (num_leapfrog_steps={num_leapfrog_steps})")
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps
            )

        # Adaptive step size
        num_adaptation_steps = int(adaptation_rate * num_burnin)
        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob
        )

        if seed is not None:
            tf.random.set_seed(seed)

        def trace_fn(_, pkr):
            return {
                'is_accepted': pkr.inner_results.is_accepted,
                'step_size': pkr.new_step_size
            }

        print(f"Running {self.sampler.upper()} with {num_samples} samples, "
              f"{num_burnin} burn-in...")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            samples_unconstrained, trace_results = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin,
                current_state=self.param_handler.unconstrained_init,
                kernel=adaptive_kernel,
                trace_fn=trace_fn
            )

        # Post-process
        samples_constrained = self._transform_samples(samples_unconstrained)
        diagnostics = self._compute_diagnostics(
            samples_constrained, trace_results['is_accepted'], trace_results['step_size']
        )
        summary = self._compute_summary(samples_constrained)
        self.diff_model.restore_parameters()

        print(f"{self.sampler.upper()} complete!")

        return DPFResult(
            samples=samples_constrained,
            summary=summary,
            diagnostics=diagnostics,
            metadata={
                'model_type': self.base_model.__class__.__name__,
                'filter_type': self.filter_class.__name__,
                'sampler': self.sampler,
                'num_samples': num_samples,
                'num_burnin': num_burnin,
                'num_observations': len(observations)
            }
        )

    # Backward compat
    run_hmc = run_inference

    def _transform_samples(self, samples_unconstrained):
        num_samples = samples_unconstrained.shape[0]
        result = {name: [] for name in self.param_handler.param_names}
        for i in range(num_samples):
            constrained = self.param_handler.constrain(samples_unconstrained[i])
            for name, value in constrained.items():
                result[name].append(float(value.numpy()))
        return {name: np.array(vals) for name, vals in result.items()}

    def _compute_diagnostics(self, samples, is_accepted, step_sizes):
        diagnostics = {}
        diagnostics['acceptance_rate'] = float(
            tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy()
        )
        diagnostics['final_step_size'] = float(step_sizes[-1].numpy())

        ess_dict = {}
        for name, s in samples.items():
            param_tf = tf.constant(s[np.newaxis, :], dtype=tf.float32)
            ess = tfp.mcmc.effective_sample_size(param_tf)
            ess_dict[name] = float(ess.numpy()[0])
        diagnostics['ess'] = ess_dict

        rhat_dict = {}
        for name, s in samples.items():
            if len(s) >= 4:
                mid = len(s) // 2
                chains = tf.constant([s[:mid], s[mid:2*mid]], dtype=tf.float32)
                rhat_dict[name] = float(tfp.mcmc.potential_scale_reduction(chains).numpy())
            else:
                rhat_dict[name] = np.nan
        diagnostics['rhat'] = rhat_dict

        return diagnostics

    def _compute_summary(self, samples):
        summary = {}
        for name, s in samples.items():
            summary[name] = {
                'mean': float(np.mean(s)), 'std': float(np.std(s)),
                'median': float(np.median(s)),
                'q5': float(np.percentile(s, 5)), 'q25': float(np.percentile(s, 25)),
                'q75': float(np.percentile(s, 75)), 'q95': float(np.percentile(s, 95)),
                'min': float(np.min(s)), 'max': float(np.max(s))
            }
        return summary
```

### Why remove `@tf.function` from `_negative_log_posterior`?
TFP's `sample_chain` traces the target function internally. Adding `@tf.function` on top causes retracing issues with `tf.Variable.assign()`. Let TFP handle it.

---

## Step 1.4: Modify LinearGaussianModel for Scalar Noise Parameters
<a name="step-1-4"></a>

### Problem
`LinearGaussianModel` stores `Q = B @ B^T` and `R = D @ D^T` as precomputed `tf.constant` in `__init__`. When `DifferentiableModel` replaces noise-scale parameters with `tf.Variable`, Q and R must **recompute dynamically** from the current parameter values.

### Design Decision
Add optional scalar parameters `process_noise_std` and `obs_noise_std`. When present:
- `Q = process_noise_std^2 * Q_base` (where `Q_base = B_normalized @ B_normalized^T`)
- `R = obs_noise_std^2 * R_base`

This keeps the existing API intact (models without these params work as before).

### File: `code/src/models/linear_gaussian.py` (MODIFY)

Changes to `__init__`:
```python
    def __init__(
        self,
        F, B, H, D,
        mu_0=None, Sigma_0=None,
        process_noise_std: float = None,   # NEW: scalar noise scale for Q
        obs_noise_std: float = None,       # NEW: scalar noise scale for R
        dtype=tf.float32
    ):
        # ... existing F, B, H, D setup ...

        # Store base noise matrices (always precomputed)
        self._Q_base = self.B @ tf.transpose(self.B)
        self._R_base = self.D @ tf.transpose(self.D)

        # Optional scalar noise parameters (for HMC inference)
        # When these are not None, Q and R become dynamic:
        #   Q = process_noise_std^2 * Q_base_normalized
        #   R = obs_noise_std^2 * R_base_normalized
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std

        if self.process_noise_std is not None:
            # Normalize Q_base so that process_noise_std controls the scale
            # Q_base_norm is Q_base / process_noise_std^2 at initialization
            init_std = float(process_noise_std)
            self._Q_base_norm = self._Q_base / (init_std ** 2)
        else:
            self._Q_base_norm = None

        if self.obs_noise_std is not None:
            init_std = float(obs_noise_std)
            self._R_base_norm = self._R_base / (init_std ** 2)
        else:
            self._R_base_norm = None

        # ... existing mu_0, Sigma_0 setup ...
```

Override the Q and R properties:
```python
    @property
    def Q(self):
        """Process noise covariance — dynamic if process_noise_std is set."""
        if self._Q_base_norm is not None:
            pns = self.process_noise_std
            if not isinstance(pns, tf.Tensor):
                pns = tf.constant(float(pns), dtype=self.dtype)
            return pns ** 2 * self._Q_base_norm
        return self._Q_base

    @Q.setter
    def Q(self, value):
        """Allow direct assignment for backward compat."""
        self._Q_base = value

    @property
    def R(self):
        """Observation noise covariance — dynamic if obs_noise_std is set."""
        if self._R_base_norm is not None:
            ons = self.obs_noise_std
            if not isinstance(ons, tf.Tensor):
                ons = tf.constant(float(ons), dtype=self.dtype)
            return ons ** 2 * self._R_base_norm
        return self._R_base

    @R.setter
    def R(self, value):
        """Allow direct assignment for backward compat."""
        self._R_base = value

    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Cov[X' | X] = Q (dynamic when process_noise_std is set)."""
        return self.Q

    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Cov[Y | X] = R (dynamic when obs_noise_std is set)."""
        return self.R

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        """For flow filters."""
        return self.R

    @property
    def process_noise_cov(self) -> tf.Tensor:
        """For flow filters."""
        return self.Q
```

Also update the batch methods that reference Q/R:
```python
    @tf.function
    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Vectorized Gaussian log-prob for all particles."""
        R = self.R  # Dynamic property — picks up tf.Variable changes
        means = particles @ tf.transpose(self.H)
        diff = observation - means
        L_R = tf.linalg.cholesky(R)
        y = tf.linalg.triangular_solve(L_R, tf.transpose(diff), lower=True)
        mahalanobis = tf.reduce_sum(y**2, axis=0)
        logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_R)))
        return -0.5 * (tf.cast(self.obs_dim, observation.dtype) *
                       tf.math.log(2.0 * 3.14159265359) + logdet + mahalanobis)

    @tf.function
    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Batch state transition using dynamic Q."""
        L_Q = tf.linalg.cholesky(self.Q)  # Dynamic
        z = tf.random.stateless_normal([tf.shape(particles)[0], self.nx], seed=seed, dtype=self.dtype)
        noise = tf.linalg.matmul(z, L_Q, transpose_b=True)
        return particles @ tf.transpose(self.F) + noise
```

### Why this design?
- **Backward compatible**: Without `process_noise_std`/`obs_noise_std`, model works exactly as before (Q = B@B^T constant).
- **DifferentiableModel-friendly**: When `DifferentiableModel` replaces `process_noise_std` with a `tf.Variable`, the `Q` property auto-recomputes.
- **Natural parameterization**: Inferring scalar noise levels is the most common Bayesian inference task for linear Gaussian models.

### Important: Remove `@tf.function` from methods that use dynamic Q/R
The `@tf.function` decorator on `log_observation_prob_batch` etc. will need to be removed or replaced with `reduce_retracing=True`, because Q and R change between calls. Alternatively, since these are called from within TFP's tracing, they'll be traced correctly.

**Simplest approach**: Remove `@tf.function` from methods that access `self.Q` or `self.R` when those are properties. Let TFP handle the tracing. The methods themselves are pure TF ops, so they'll be fast anyway.

---

## Step 1.5: Add `log_marginal_likelihood_tf` to KalmanFilter and EKF
<a name="step-1-5"></a>

### Problem
Both KalmanFilter and EKF compute per-step log-likelihoods in `_update_step` (already `@tf.function`), but `filter()` converts to numpy at the end.

### File: `code/src/filters/kalman/kalman.py` (ADD METHOD)

Add to `KalmanFilter` class:

```python
    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Total log marginal likelihood as a differentiable TF scalar.

        Runs predict/update loop using local tensor state (no side effects
        on self.mean/self.cov). Returns tf.Tensor, not Python float.

        Args:
            observations: (T, obs_dim), dtype matching filter
            seed: Unused (KF is deterministic)

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        T = tf.shape(observations)[0]

        # Use local tensors — no side effects on self.mean/self.cov
        mean = tf.identity(self.mean_0)
        cov = tf.identity(self.Sigma_0)
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in tf.range(T):
            # Predict
            mean, cov = self._predict_step(mean, cov)
            # Update — _update_step returns (mean, cov, K, innovation, log_lik)
            mean, cov, _, _, log_lik_t = self._update_step(mean, cov, observations[t])
            total_log_lik = total_log_lik + log_lik_t

        return total_log_lik
```

**IMPORTANT for KalmanFilter**: The KalmanFilter stores F, B, H, D and precomputes Q, R in its own `__init__` as `tf.constant`. It does NOT use the model object. For HMC, we need the KalmanFilter to read Q and R from the model dynamically.

**Two options:**

**Option A (Recommended)**: Use the EKF instead. The EKF takes a model object and calls `self.model.state_transition_cov()`, `self.model.observation_cov()`, etc. For a linear model, the EKF reduces to the exact Kalman filter. This is the path of least resistance.

**Option B**: Modify KalmanFilter to accept a model object and read Q/R dynamically. This requires more refactoring.

**We use Option A.** The EKF already works with the model:

### File: `code/src/filters/kalman/extended_kalman.py` (ADD METHOD)

```python
    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Total log marginal likelihood as a differentiable TF scalar.

        Uses local tensor state to avoid side effects.
        The EKF's _predict_step and _update_step call self.model.* methods,
        which read from tf.Variable when wrapped by DifferentiableModel.

        Args:
            observations: (T, obs_dim)
            seed: Unused (EKF is deterministic)

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        T = tf.shape(observations)[0]

        mean = tf.identity(self.mean_0)
        cov = tf.identity(self.Sigma_0)
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in tf.range(T):
            mean, cov = self._predict_step(mean, cov)
            mean, cov, log_lik_t = self._update_step(mean, cov, observations[t])
            total_log_lik = total_log_lik + log_lik_t

        return total_log_lik
```

### Why this works
- `_predict_step` calls `self.model.state_transition_cov(mean)` which returns `self.model.Q` — the dynamic property.
- `_update_step` calls `self.model.observation_cov(mean)` which returns `self.model.R` — the dynamic property.
- When `process_noise_std` is a `tf.Variable`, the entire chain stays differentiable.

---

## Step 1.6: Phase 1 Smoke Test
<a name="step-1-6"></a>

### File: `code/src/DF/smoke_test_linear_gaussian.py` (NEW)

```python
"""
Phase 1 Smoke Test: LinearGaussianModel + EKF + HMC

Infer process_noise_std and obs_noise_std of a linear Gaussian model.
The Kalman filter (via EKF) is optimal for this model, so the
log-likelihood surface is smooth and HMC should converge easily.

Expected outcome:
- Posterior means near true values (sigma_q=1.0, sigma_r=0.5)
- Acceptance rate > 0.6
- R-hat < 1.1
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.DF import DPFRunner, ParameterSpec
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data


def smoke_test_linear_gaussian():
    print("=" * 60)
    print("PHASE 1 SMOKE TEST: LinearGaussian + EKF")
    print("=" * 60)

    # ---------- True model ----------
    true_sigma_q = 1.0
    true_sigma_r = 0.5

    true_model = LinearGaussianModel(
        F=[[0.9, 0.1], [0.0, 0.8]],
        B=[[true_sigma_q], [0.5 * true_sigma_q]],
        H=[[1.0, 0.0]],
        D=[[true_sigma_r]],
        mu_0=[0.0, 0.0],
        Sigma_0=[[1.0, 0.0], [0.0, 1.0]],
        process_noise_std=true_sigma_q,
        obs_noise_std=true_sigma_r,
    )

    rng = np.random.default_rng(42)
    T = 200
    initial_state, states, observations = generate_data(true_model, T=T, rng=rng)
    print(f"Generated {T} observations")
    print(f"True params: sigma_q={true_sigma_q}, sigma_r={true_sigma_r}")

    # ---------- Inference model (wrong noise scales) ----------
    init_sigma_q = 0.5   # Wrong!
    init_sigma_r = 1.0   # Wrong!

    inference_model = LinearGaussianModel(
        F=[[0.9, 0.1], [0.0, 0.8]],
        B=[[1.0], [0.5]],  # B_base (will be scaled by process_noise_std)
        H=[[1.0, 0.0]],
        D=[[1.0]],         # D_base (will be scaled by obs_noise_std)
        mu_0=[0.0, 0.0],
        Sigma_0=[[1.0, 0.0], [0.0, 1.0]],
        process_noise_std=init_sigma_q,
        obs_noise_std=init_sigma_r,
    )

    # ---------- Parameter specs ----------
    param_specs = {
        'process_noise_std': ParameterSpec(
            name='process_noise_std',
            init_value=init_sigma_q,
            constraint='positive',
            prior=tfp.distributions.LogNormal(0.0, 1.0)
        ),
        'obs_noise_std': ParameterSpec(
            name='obs_noise_std',
            init_value=init_sigma_r,
            constraint='positive',
            prior=tfp.distributions.LogNormal(0.0, 1.0)
        ),
    }

    # ---------- DPF Runner ----------
    runner = DPFRunner(
        base_model=inference_model,
        filter_class=ExtendedKalmanFilter,
        filter_kwargs={},
        param_specs=param_specs,
        sampler='nuts'
    )

    # ---------- Run ----------
    result = runner.run_inference(
        observations=observations,
        num_samples=200,
        num_burnin=100,
        step_size=0.01,
        seed=42
    )

    # ---------- Validate ----------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name in ['process_noise_std', 'obs_noise_std']:
        true_val = true_sigma_q if name == 'process_noise_std' else true_sigma_r
        stats = result.summary[name]
        in_ci = stats['q5'] <= true_val <= stats['q95']
        print(f"\n  {name}:")
        print(f"    True:           {true_val:.4f}")
        print(f"    Posterior mean:  {stats['mean']:.4f} +/- {stats['std']:.4f}")
        print(f"    90% CI:         [{stats['q5']:.4f}, {stats['q95']:.4f}]")
        print(f"    True in 90% CI: {'YES' if in_ci else 'NO'}")

    print(f"\nDiagnostics:")
    print(f"  Accept rate: {result.diagnostics['acceptance_rate']:.1%}")
    print(f"  ESS: {result.diagnostics['ess']}")
    print(f"  R-hat: {result.diagnostics['rhat']}")

    # Sanity checks
    for name in ['process_noise_std', 'obs_noise_std']:
        mean = result.summary[name]['mean']
        assert not np.isnan(mean), f"{name} posterior mean is NaN!"
        assert mean > 0, f"{name} is non-positive: {mean}"

    accept = result.diagnostics['acceptance_rate']
    assert accept > 0.1, f"Acceptance rate too low: {accept}"

    print("\n>>> PHASE 1 SMOKE TEST PASSED <<<")
    return result


if __name__ == '__main__':
    smoke_test_linear_gaussian()
```

### How to run
```bash
cd code
python -m src.DF.smoke_test_linear_gaussian
```

### Expected output
- `process_noise_std` posterior mean near 1.0
- `obs_noise_std` posterior mean near 0.5
- Acceptance rate > 60%
- R-hat < 1.1 for both parameters

---

## Step 1.7: Phase 1 Hydra Config
<a name="step-1-7"></a>

### File: `code/configs/dpf/linear_gaussian_kf.yaml` (NEW)

```yaml
# DPF: Infer noise scales of LinearGaussianModel using EKF (=KF for linear models)

model:
  _target_: src.models.linear_gaussian.LinearGaussianModel
  F: [[0.9, 0.1], [0.0, 0.8]]
  B: [[1.0], [0.5]]      # B_base (scaled by process_noise_std)
  H: [[1.0, 0.0]]
  D: [[1.0]]             # D_base (scaled by obs_noise_std)
  mu_0: [0.0, 0.0]
  Sigma_0: [[1.0, 0.0], [0.0, 1.0]]
  process_noise_std: 0.5   # Initial guess (wrong)
  obs_noise_std: 1.0       # Initial guess (wrong)

filter:
  _target_: src.filters.kalman.extended_kalman.ExtendedKalmanFilter

dpf:
  sampler: nuts

  trainable_params:
    process_noise_std:
      init_value: 0.5
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 0.0
        scale: 1.0
    obs_noise_std:
      init_value: 1.0
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 0.0
        scale: 1.0

  hmc:
    num_samples: 1000
    num_burnin: 500
    step_size: 0.01
    max_tree_depth: 10
    adaptation_rate: 0.8
    target_accept_prob: 0.75
    seed: 42

data:
  T: 200
  seed: 42
  true_params:
    process_noise_std: 1.0
    obs_noise_std: 0.5

output:
  save_samples: true
  plot_posteriors: true
```

---

# PHASE 2: LEDH Float64 Optimization

**Goal:** Fix the float64 regression in LEDH before using it for HMC.

This phase is pulled directly from `FLOAT64_REGRESSION_PLAN.md` with concrete implementation details.

### Background

| Run | dtype | RMSE | Log-lik | Mean ESS |
|-----|-------|------|---------|----------|
| Old (pre-fix) | float32 | 14.16 | 1019.74 | — |
| New code | float32 | 16.68 | 1048.10 | smaller |
| New code | float64 | 21.17 | 104.53 | larger |

Float64 RMSE degrades to 21.17 and log-likelihood drops 10x. Root cause: `tf.random.stateless_normal` produces different values for different dtypes, confounding the comparison. A controlled experiment is needed.

---

## Step 2.1: Controlled Float32/Float64 Comparison
<a name="step-2-1"></a>

### File: `code/src/experiments/dtype_comparison.py` (NEW)

```python
"""Controlled dtype comparison: same data, different filter precision."""

import numpy as np
import tensorflow as tf
from pathlib import Path

from ..models.utils import generate_data


def run_dtype_comparison(model_cfg, filter_cfg, T=100, seed=42, n_runs=5):
    """
    Compare LEDH filter in float32 vs float64 on IDENTICAL data.

    1. Generate ground truth data ONCE in float64
    2. Run LEDH in float32 on the same data
    3. Run LEDH in float64 on the same data
    4. Compare RMSE and log-likelihood
    """
    import hydra

    results = {'float32': [], 'float64': []}

    for run_seed in range(seed, seed + n_runs):
        # Generate data in float64 (maximum precision)
        model_f64 = hydra.utils.instantiate(model_cfg, dtype=tf.float64)
        rng = np.random.default_rng(run_seed)
        initial_state, states, observations = generate_data(model_f64, T=T, rng=rng)

        # Save as float64 numpy (ground truth)
        states_f64 = states.astype(np.float64)
        obs_f64 = observations.astype(np.float64)

        for dtype_str in ['float32', 'float64']:
            dtype_tf = tf.float32 if dtype_str == 'float32' else tf.float64

            # Cast observations to filter dtype
            obs_cast = obs_f64.astype(np.float32 if dtype_str == 'float32' else np.float64)

            # Create model and filter in target dtype
            model = hydra.utils.instantiate(model_cfg, dtype=dtype_tf)
            filter_obj = hydra.utils.instantiate(filter_cfg, model=model)

            # Run filter
            result = filter_obj.filter(obs_cast, random_seed=run_seed)

            # Compute RMSE against float64 ground truth
            means_f64 = result.means.astype(np.float64)
            rmse = np.sqrt(np.mean((means_f64 - states_f64) ** 2))

            results[dtype_str].append({
                'seed': run_seed,
                'rmse': float(rmse),
                'log_likelihood': result.log_likelihood,
                'mean_ess': float(np.mean(result.ess)) if result.ess is not None else None,
            })

            print(f"  [{dtype_str}] seed={run_seed}: RMSE={rmse:.4f}, "
                  f"LL={result.log_likelihood:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (controlled comparison, same data)")
    print("=" * 60)
    for dtype_str in ['float32', 'float64']:
        rmses = [r['rmse'] for r in results[dtype_str]]
        lls = [r['log_likelihood'] for r in results[dtype_str] if r['log_likelihood'] is not None]
        print(f"\n  {dtype_str}:")
        print(f"    RMSE:   {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
        if lls:
            print(f"    LogLik: {np.mean(lls):.2f} +/- {np.std(lls):.2f}")

    return results
```

### Expected outcomes
- **If float64 = float32 on same data**: The regression was due to different random seeds. Proceed to Step 2.3.
- **If float64 < float32 on same data**: There's a genuine numerical issue. Proceed to Step 2.2.

---

## Step 2.2: Debug Per-Particle Covariances (if float64 is genuinely worse)
<a name="step-2-2"></a>

### What to check (in order of likelihood)

1. **Eigenvalue spectrum of per-particle covariances after `batched_ekf_predict`**
   - Float64 may preserve tiny eigenvalues that float32 rounds to zero
   - This can make the flow ill-conditioned (A matrix has huge eigenvalues)

2. **`safe_cholesky` jitter behavior**
   - Current jitter: `1e-10 * avg_diag`
   - In float32: near machine epsilon (1e-7), so it's meaningful
   - In float64: 6 orders above machine epsilon (1e-16), so it's excessive
   - **Fix**: Make jitter dtype-dependent:
     ```python
     eps = 1e-6 if dtype == tf.float32 else 1e-12
     ```

3. **Weight computation in `compute_flow_weights`**
   - Check if log-probabilities overflow differently in float64
   - The max-normalization should handle this, but verify

4. **Regularization consistency**
   - `regularization = 1e-8` for P (flow_params)
   - `safe_cholesky jitter = 1e-10` for S (innovation covariance)
   - These should be consistent: if P is regularized, S = lambda*H*P*H^T + R inherits it

### Files to instrument
- `code/src/filters/particle/ledh_invertible.py` — add eigenvalue logging in debug_mode
- `code/src/filters/kalman/batched_ekf.py` — log condition numbers
- `code/src/utils/flow_params.py` — log A matrix condition
- `code/src/utils/linalg.py` — log jitter amounts in safe_cholesky

---

## Step 2.3: Multi-Seed Statistical Comparison
<a name="step-2-3"></a>

If Step 2.1 shows float64 is fine on the same data, run a multi-seed comparison:

```bash
for seed in 42 43 44 45 46 47 48 49 50 51; do
  python -m code.src.experiments.run_experiment \
    experiment=acoustic_tracking/acoustic_tracking_ledh_invertible \
    seed=$seed dtype=float32

  python -m code.src.experiments.run_experiment \
    experiment=acoustic_tracking/acoustic_tracking_ledh_invertible \
    seed=$seed dtype=float64
done
```

Compare distributions of RMSE and log-likelihood across seeds. Float64 should be equal or slightly better on average.

---

## Step 2.4: Parameter Re-tuning for TensorFlow
<a name="step-2-4"></a>

MATLAB parameters were calibrated for MATLAB's numerical environment. Key candidates:

| Parameter | Current | Why re-tune | Sweep range |
|-----------|---------|-------------|-------------|
| `n_lambda_steps` | 29 | MATLAB float64 may tolerate fewer steps | 15, 20, 29, 40, 50 |
| `q` (exponential growth) | 1.2 | Controls step size distribution | 1.1, 1.2, 1.3, 1.5 |
| `regularization` | 1e-8 | Near float32 noise floor | 1e-10, 1e-8, 1e-6, 1e-4 |
| `safe_cholesky jitter` | 1e-10 | Same concern | dtype-dependent |
| `resample_threshold` | 0.5 | Currently resampling at rate=1.0 | 0.3, 0.5, 0.7, 0.9 |

### Approach
For each parameter, sweep values while holding others fixed on a reference dataset (acoustic tracking, seed=42, float32). Optimize RMSE first, then log-likelihood.

Start with `n_lambda_steps` and `q` as they directly control flow integration accuracy.

---

# PHASE 3: Kitagawa + LEDH Invertible

**Goal:** Run HMC parameter inference on a nonlinear model using LEDH particle flow.

The Kitagawa model is:
- **State**: `x_n = x_{n-1}/2 + 25*x_{n-1}/(1+x_{n-1}^2) + 8*cos(1.2*n) + V_n` (strongly nonlinear)
- **Observation**: `y_n = x_n^2/20 + W_n` (nonlinear, non-invertible)

Parameters to infer: `sigma_V` (process noise std) and `sigma_W` (observation noise std).

**Prerequisite:** Phase 2 complete (LEDH numerics stable in float64).

---

## Step 3.1: TF-ify KitagawaModel
<a name="step-3-1"></a>

### Problem
All methods in `KitagawaModel` use numpy. The model also has a time-dependent state transition (`8*cos(1.2*n)`), which requires tracking the time step.

### File: `code/src/models/kitagawa.py` (MODIFY)

Replace numpy methods with dual numpy/TF methods. Key pattern: use `tf.cast(self.sigma_V, self.dtype)` so it works whether `sigma_V` is a Python float or a `tf.Variable`.

```python
    # ----------------------------------------------------------------
    # TF-compatible methods (work with float, tf.Variable, tf.Tensor)
    # ----------------------------------------------------------------

    def state_transition_mean(self, x):
        """E[x_n | x_{n-1}] = f(x_{n-1}, n)."""
        if isinstance(x, tf.Tensor):
            t_val = tf.cast(self.t, self.dtype)
            x0 = x[0] if len(x.shape) == 1 else x
            mean = x0 / 2.0 + 25.0 * x0 / (1.0 + x0 ** 2) + 8.0 * tf.cos(1.2 * t_val)
            return tf.reshape(mean, tf.shape(x))
        return np.array([self._f(x[0], self.t)], dtype=self.np_dtype)

    def state_transition_cov(self, x):
        """Cov[x_n | x_{n-1}] = sigma_V^2."""
        sigma_V = self.sigma_V
        if isinstance(sigma_V, tf.Tensor):
            return tf.reshape(sigma_V ** 2, [1, 1])
        return np.array([[float(sigma_V) ** 2]], dtype=self.np_dtype)

    def state_jacobian(self, x):
        """df/dx = 0.5 + 25*(1-x^2)/(1+x^2)^2."""
        if isinstance(x, tf.Tensor):
            x0 = x[0] if len(x.shape) == 1 else x
            jac = 0.5 + 25.0 * (1.0 - x0 ** 2) / (1.0 + x0 ** 2) ** 2
            return tf.reshape(jac, [1, 1])
        return np.array([[self._df_dx(x[0])]], dtype=self.np_dtype)

    def observation_mean(self, x):
        """E[y_n | x_n] = x_n^2/20."""
        if isinstance(x, tf.Tensor):
            x0 = x[0] if len(x.shape) == 1 else x
            return tf.reshape(x0 ** 2 / 20.0, tf.shape(x))
        return np.array([self._h(x[0])], dtype=self.np_dtype)

    def observation_cov(self, x):
        """Cov[y_n | x_n] = sigma_W^2."""
        sigma_W = self.sigma_W
        if isinstance(sigma_W, tf.Tensor):
            return tf.reshape(sigma_W ** 2, [1, 1])
        return np.array([[float(sigma_W) ** 2]], dtype=self.np_dtype)

    def observation_jacobian(self, x):
        """dh/dx = x/10."""
        if isinstance(x, tf.Tensor):
            x0 = x[0] if len(x.shape) == 1 else x
            return tf.reshape(x0 / 10.0, [1, 1])
        return np.array([[self._dh_dx(x[0])]], dtype=self.np_dtype)

    def observation_hessian(self, x):
        """d^2h/dx^2 = 1/10."""
        if isinstance(x, tf.Tensor):
            return tf.constant([[[0.1]]], dtype=x.dtype)
        return np.array([[[1.0 / 10.0]]], dtype=self.np_dtype)

    def observation_function(self, x):
        """h(x) = x^2/20."""
        return self.observation_mean(x)

    @property
    def observation_noise_cov(self):
        sigma_W = self.sigma_W
        if isinstance(sigma_W, tf.Tensor):
            return tf.reshape(sigma_W ** 2, [1, 1])
        return np.array([[float(sigma_W) ** 2]], dtype=self.np_dtype)

    @property
    def process_noise_cov(self):
        sigma_V = self.sigma_V
        if isinstance(sigma_V, tf.Tensor):
            return tf.reshape(sigma_V ** 2, [1, 1])
        return np.array([[float(sigma_V) ** 2]], dtype=self.np_dtype)

    @property
    def mu_0(self):
        """Initial state mean."""
        if hasattr(self, '_mu_0_tf'):
            return self._mu_0_tf
        return np.array([0.0], dtype=self.np_dtype)

    @property
    def Sigma_0(self):
        """Initial state covariance."""
        iv = self.initial_var
        if isinstance(iv, tf.Tensor):
            return tf.reshape(iv, [1, 1])
        return np.array([[float(iv)]], dtype=self.np_dtype)

    # TF batch methods
    def state_transition_batch(self, particles, seed):
        """Batch state transition: x' = f(x, t) + sigma_V * w."""
        sigma_V = tf.cast(self.sigma_V, self.dtype) if not isinstance(self.sigma_V, tf.Tensor) else self.sigma_V
        t_val = tf.cast(self.t, self.dtype)
        x = particles[:, 0:1]
        mean = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
        w = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=self.dtype)
        return mean + sigma_V * w

    def state_transition_mean_batch(self, particles):
        """Batch transition mean."""
        if isinstance(particles, tf.Tensor):
            t_val = tf.cast(self.t, self.dtype)
            x = particles[:, 0:1]
            return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * tf.cos(1.2 * t_val)
        x = particles[:, 0]
        means = x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * self.t)
        return means[:, np.newaxis]

    def log_observation_prob_batch(self, observation, particles):
        """Batch log p(y|x)."""
        sigma_W = tf.cast(self.sigma_W, self.dtype) if not isinstance(self.sigma_W, tf.Tensor) else self.sigma_W
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            means = x ** 2 / 20.0
            var = sigma_W ** 2
            diff = observation[0] - means
            pi = tf.constant(np.pi, dtype=self.dtype)
            return -0.5 * (tf.math.log(2.0 * pi * var) + diff ** 2 / var)
        # numpy fallback
        x = particles[:, 0]
        means = x ** 2 / 20.0
        var = float(sigma_W) ** 2
        diff = observation[0] - means
        return -0.5 * (np.log(2 * np.pi * var) + diff ** 2 / var)

    def observation_jacobian_batch(self, particles):
        """Batch dh/dx = x/10."""
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            return tf.reshape(x / 10.0, [-1, 1, 1])
        x = particles[:, 0]
        return (x / 10.0)[:, np.newaxis, np.newaxis]

    def observation_function_batch(self, particles):
        """Batch h(x) = x^2/20."""
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            return tf.reshape(x ** 2 / 20.0, [-1, 1])
        x = particles[:, 0]
        return (x ** 2 / 20.0)[:, np.newaxis]

    def state_jacobian_batch(self, particles):
        """Batch df/dx."""
        if isinstance(particles, tf.Tensor):
            x = particles[:, 0]
            jac = 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2
            return tf.reshape(jac, [-1, 1, 1])
        x = particles[:, 0]
        jac = 0.5 + 25.0 * (1.0 - x ** 2) / (1.0 + x ** 2) ** 2
        return jac[:, np.newaxis, np.newaxis]

    def sample_initial_state_batch(self, n, seed):
        """Batch initial state sampling."""
        iv = tf.cast(self.initial_var, self.dtype) if not isinstance(self.initial_var, tf.Tensor) else self.initial_var
        std = tf.sqrt(iv)
        return tf.random.stateless_normal([n, 1], seed=seed, dtype=self.dtype) * std
```

### Time step management
The Kitagawa model has `self.t` for time-dependent dynamics. For the LEDH filter, `self.t` must be advanced correctly. Currently, `predict()` in LEDH does NOT call `model.sample_state_transition()` (which increments `self.t`). Instead it calls `model.state_transition_batch()`.

**Fix**: Increment `model.t` explicitly in the LEDH predict step, or pass `t` as a parameter to batch methods. For now, the simplest fix is to have LEDH's `filter()` method advance `model.t` at each timestep:

```python
# In LEDH filter() loop:
for t in range(T):
    self.model.t = t + 1  # Explicit time step management
    self.predict()
    self.update(obs_tf[t])
```

---

## Step 3.2: Add `log_marginal_likelihood_tf` to LEDH Invertible
<a name="step-3-2"></a>

### File: `code/src/filters/particle/ledh_invertible.py` (ADD METHOD)

```python
    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Total log marginal likelihood as a differentiable TF scalar.

        Runs the full LEDH filter (initialize, predict/update loop)
        and sums per-step log-likelihoods. All ops stay in TF.

        For HMC, the resampling step MUST be differentiable.
        Use soft_resample or ot_entropy_resample.

        Args:
            observations: (T, obs_dim), dtype matching model
            seed: TF random seed for initialization

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        # Initialize
        random_seed = int(seed[0].numpy()) if seed is not None else 42
        self.initialize(random_seed=random_seed)

        T = observations.shape[0]
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in range(T):
            # Advance time step for time-dependent models (e.g., Kitagawa)
            if hasattr(self.model, 't'):
                self.model.t = t + 1

            self.predict()
            self.update(observations[t])

            # Per-step log-likelihood (already computed in update())
            # But we need to accumulate as TF tensor, not append to list
            eta_1 = self.particles.value()
            log_obs_probs = self.model.log_observation_prob_batch(observations[t], eta_1)
            max_ll = tf.reduce_max(log_obs_probs)
            log_lik_t = max_ll + tf.math.log(tf.reduce_mean(tf.exp(log_obs_probs - max_ll)))
            total_log_lik = total_log_lik + log_lik_t

        return total_log_lik
```

### Differentiability concern: resampling
LEDH uses `systematic_resample` by default, which is NOT differentiable. For HMC:
```python
ledh = LEDHParticleFlowFilter(
    model, n_particles=200,
    resampling_method='soft',
    resampling_config={'alpha': 0.9}
)
```

---

## Step 3.3: Phase 3 Smoke Test
<a name="step-3-3"></a>

### File: `code/src/DF/smoke_test_kitagawa.py` (NEW)

```python
"""
Phase 3 Smoke Test: KitagawaModel + LEDH Invertible + HMC

Infer sigma_V and sigma_W of the Kitagawa model.
This is a challenging nonlinear model with strongly nonlinear dynamics
and a non-invertible observation function.

Prerequisites:
- Phase 1 passes (HMC infrastructure works)
- Phase 2 complete (LEDH numerics stable)
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.DF import DPFRunner, ParameterSpec
from src.filters.particle.ledh_invertible import LEDHParticleFlowFilter
from src.models.kitagawa import KitagawaModel
from src.models.utils import generate_data


def smoke_test_kitagawa_ledh():
    print("=" * 60)
    print("PHASE 3 SMOKE TEST: Kitagawa + LEDH Invertible")
    print("=" * 60)

    # True model
    true_sigma_V = 10.0
    true_sigma_W = 1.0

    true_model = KitagawaModel(sigma_V=true_sigma_V, sigma_W=true_sigma_W,
                               initial_var=5.0, dtype=tf.float32)

    rng = np.random.default_rng(42)
    T = 100
    initial_state, states, observations = generate_data(true_model, T=T, rng=rng)
    print(f"Generated {T} observations")
    print(f"True: sigma_V={true_sigma_V}, sigma_W={true_sigma_W}")

    # Inference model (wrong noise)
    inference_model = KitagawaModel(sigma_V=5.0, sigma_W=2.0,
                                    initial_var=5.0, dtype=tf.float32)

    param_specs = {
        'sigma_V': ParameterSpec(
            name='sigma_V',
            init_value=5.0,
            constraint='positive',
            prior=tfp.distributions.LogNormal(tf.math.log(10.0), 0.5)
        ),
        'sigma_W': ParameterSpec(
            name='sigma_W',
            init_value=2.0,
            constraint='positive',
            prior=tfp.distributions.LogNormal(0.0, 1.0)
        ),
    }

    runner = DPFRunner(
        base_model=inference_model,
        filter_class=LEDHParticleFlowFilter,
        filter_kwargs={
            'n_particles': 200,
            'n_lambda_steps': 29,
            'resampling_method': 'soft',
            'resampling_config': {'alpha': 0.9},
        },
        param_specs=param_specs,
        sampler='nuts'
    )

    result = runner.run_inference(
        observations=observations,
        num_samples=100,
        num_burnin=50,
        step_size=0.01,
        seed=42
    )

    # Display
    print("\n" + "=" * 60)
    for name in ['sigma_V', 'sigma_W']:
        true_val = true_sigma_V if name == 'sigma_V' else true_sigma_W
        stats = result.summary[name]
        print(f"  {name}: true={true_val:.2f}, "
              f"posterior={stats['mean']:.2f} +/- {stats['std']:.2f}")

    print(f"\n  Accept rate: {result.diagnostics['acceptance_rate']:.1%}")
    print(f"  ESS: {result.diagnostics['ess']}")

    print("\n>>> PHASE 3 SMOKE TEST COMPLETE <<<")
    return result


if __name__ == '__main__':
    smoke_test_kitagawa_ledh()
```

---

## Step 3.4: Phase 3 Hydra Config
<a name="step-3-4"></a>

### File: `code/configs/dpf/kitagawa_ledh.yaml` (NEW)

```yaml
# DPF: Infer noise parameters of Kitagawa model using LEDH

model:
  _target_: src.models.kitagawa.KitagawaModel
  sigma_V: 5.0       # Initial guess (true: 10.0)
  sigma_W: 2.0       # Initial guess (true: 1.0)
  initial_var: 5.0

filter:
  _target_: src.filters.particle.ledh_invertible.LEDHParticleFlowFilter
  n_particles: 200
  n_lambda_steps: 29
  resampling_method: soft
  resampling_config:
    alpha: 0.9

dpf:
  sampler: nuts

  trainable_params:
    sigma_V:
      init_value: 5.0
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 2.3          # log(10) ≈ 2.3
        scale: 0.5
    sigma_W:
      init_value: 2.0
      constraint: positive
      prior:
        _target_: tensorflow_probability.distributions.LogNormal
        loc: 0.0
        scale: 1.0

  hmc:
    num_samples: 500
    num_burnin: 250
    step_size: 0.01
    max_tree_depth: 8
    adaptation_rate: 0.8
    target_accept_prob: 0.7
    seed: 42

data:
  T: 200
  seed: 42
  true_params:
    sigma_V: 10.0
    sigma_W: 1.0

output:
  save_samples: true
  plot_posteriors: true
```

---

## DPF Experiment Runner (shared infrastructure)
<a name="step-runner"></a>

Same as the original action plan Step 10. Create `code/src/experiments/run_dpf_experiment.py` with Hydra integration, result saving, and posterior plotting. See the [original Step 10 code](#step-10-dpf-experiment-runner) — it works unchanged for all three phases.

---

## Dependency Graph & Implementation Order
<a name="dependency-graph"></a>

```
PHASE 1: LinearGaussian + KalmanFilter (validation)
═══════════════════════════════════════════════════

Step 1.1 (Protocol)──┐
Step 1.2 (DiffModel)─┤
                      ├──► Step 1.3 (HMC Runner)
Step 1.4 (LG Model)──┤         │
                      │    Step 1.5 (EKF log_marginal_likelihood_tf)
                      │         │
                      └─────────┴──► Step 1.6 (Smoke Test)
                                          │
                                     Step 1.7 (Config)


PHASE 2: LEDH Optimization (numerics)
═════════════════════════════════════

Step 2.1 (Controlled comparison) ──► Step 2.2 (Debug, if needed)
                                          │
                                     Step 2.3 (Multi-seed stats)
                                          │
                                     Step 2.4 (Parameter re-tuning)


PHASE 3: Kitagawa + LEDH (nonlinear HMC)
════════════════════════════════════════

     Phase 1 done ──┐
     Phase 2 done ──┤
                    │
               Step 3.1 (TF-ify Kitagawa) ──► Step 3.2 (LEDH log_marginal_lik)
                                                       │
                                                  Step 3.3 (Smoke test)
                                                       │
                                                  Step 3.4 (Config)
```

### Implementation checklist

| # | Step | Files | Status |
|---|------|-------|--------|
| | **PHASE 1** | | |
| 1.1 | Differentiable filter protocol | `core/differentiable.py` (NEW) | |
| 1.2 | Fix DifferentiableModel | `DF/differentiable_model.py` (REPLACE) | |
| 1.3 | Rewrite HMC runner + NUTS | `DF/hmc_runner.py` (REPLACE) | |
| 1.4 | LinearGaussianModel noise params | `models/linear_gaussian.py` (MODIFY) | |
| 1.5 | EKF log_marginal_likelihood_tf | `filters/kalman/extended_kalman.py` (ADD) | |
| 1.6 | Phase 1 smoke test | `DF/smoke_test_linear_gaussian.py` (NEW) | |
| 1.7 | Phase 1 config | `configs/dpf/linear_gaussian_kf.yaml` (NEW) | |
| | **PHASE 2** | | |
| 2.1 | Controlled dtype comparison | `experiments/dtype_comparison.py` (NEW) | |
| 2.2 | Debug covariances (conditional) | `filters/particle/ledh_invertible.py` etc. | |
| 2.3 | Multi-seed comparison | Shell script | |
| 2.4 | Parameter re-tuning | `ledh_invertible.py`, `linalg.py` etc. | |
| | **PHASE 3** | | |
| 3.1 | TF-ify KitagawaModel | `models/kitagawa.py` (MODIFY) | |
| 3.2 | LEDH log_marginal_likelihood_tf | `filters/particle/ledh_invertible.py` (ADD) | |
| 3.3 | Phase 3 smoke test | `DF/smoke_test_kitagawa.py` (NEW) | |
| 3.4 | Phase 3 config | `configs/dpf/kitagawa_ledh.yaml` (NEW) | |

---

## Summary: What Changed vs Original Plan

| Original Plan | Revised Plan | Why |
|---|---|---|
| StochasticVolatility + EKF first | **LinearGaussian + EKF first** | KF is optimal for LG — guaranteed good fit, cleanest validation |
| No LEDH float64 work | **Phase 2: full FLOAT64_REGRESSION_PLAN** | Must fix LEDH numerics before using for HMC |
| Flow filters (EDH/StochasticEDH) | **Kitagawa + LEDH invertible** | User's actual target; LEDH has weights + log-lik already |
| 12 steps, flat priority | **3 phases, sequential** | Clear milestones; each phase validates before proceeding |
| SV model TF-ification | **Kitagawa model TF-ification** | Kitagawa is the target, not SV |
| No dtype investigation | **Controlled dtype comparison script** | Concrete tool for FLOAT64_REGRESSION_PLAN |
