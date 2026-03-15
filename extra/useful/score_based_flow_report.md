# Score-Based Generalization of the Daum-Huang Particle Flow for Stochastic Volatility Models

## Table of Contents

1. [The Problem: Why Standard LEDH Fails for SV Models](#1-the-problem)
2. [Current Implementation: How H Enters the Pipeline](#2-current-implementation)
3. [Mathematical Foundation: From Jacobian to Score](#3-mathematical-foundation)
4. [The 2D Stochastic Volatility Model](#4-the-2d-sv-model)
5. [Fisher Information for the 2D SV Model](#5-fisher-information)
6. [Proposed Code Changes](#6-proposed-code-changes)
7. [Impact on Each Filter Type](#7-impact-on-each-filter)

---

## 1. The Problem: Why Standard LEDH Fails for SV Models <a name="1-the-problem"></a>

### 1.1 The Standard Assumption

The Daum-Huang exact flow (EDH/LEDH) is derived under the assumption that the
observation model has **additive Gaussian noise**:

$$
y_t = h(x_t) + v_t, \quad v_t \sim \mathcal{N}(0, R)
$$

where $h(x)$ is a (possibly nonlinear) observation function and $R$ is a
**state-independent** noise covariance. Under this assumption:

- The observation Jacobian $H = \partial h / \partial x$ captures **all**
  information about how the state affects the observation.
- The observation information matrix $H^\top R^{-1} H$ is always positive
  semidefinite.
- The flow equations from Li & Coates (2017) are well-posed.

### 1.2 Stochastic Volatility: A Multiplicative Noise Model

The stochastic volatility (SV) model violates this assumption. Consider even
the simplest 1D case:

$$
y_t = \beta \exp(x_t / 2) \cdot v_t, \quad v_t \sim \mathcal{N}(0, 1)
$$

This gives $y_t \mid x_t \sim \mathcal{N}(0, \beta^2 \exp(x_t))$. The
observation mean is:

$$
\mathbb{E}[y_t \mid x_t] = 0 \quad \text{for all } x_t
$$

Therefore the observation Jacobian $H = \partial \mathbb{E}[y|x] / \partial x = 0$
everywhere. Information about the state is carried entirely through the
**observation variance** $\text{Var}[y|x] = \beta^2 \exp(x_t)$, not through the
observation mean.

### 1.3 Consequence: Dead Flow

When $H = 0$, the LEDH flow equations (Eqs. 10-11 from Li & Coates 2017)
collapse:

$$
A(\lambda) = -\tfrac{1}{2} P H^\top (\lambda H P H^\top + R)^{-1} H = 0
$$
$$
b(\lambda) = (I + 2\lambda A)[(I + \lambda A) P H^\top R^{-1} (z - e) + A \bar{\eta}_0] = 0
$$

The flow $d\eta / d\lambda = A\eta + b = 0$ produces **zero displacement**.
Particles are not moved toward the posterior. The filter runs the prior
dynamics forward without assimilating observations.

### 1.4 This Affects LEDH, Not Just Standalone EKF/UKF

The LEDH filter uses EKF/UKF internally in two places:

1. **Flow computation** (`compute_flow_params_batch`): uses $H$ to compute
   $A(\lambda)$ and $b(\lambda)$
2. **Per-particle covariance tracking** (`batched_ekf_update`): uses $H$ to
   update covariances via Joseph form

Both are blind to the variance channel. The particle **weights** (computed via
`log_observation_prob_batch`) correctly use the full likelihood and can
distinguish good from bad particles, but without flow guidance, the particles
must find the posterior by random walk alone — defeating the purpose of the
flow filter.

---

## 2. Current Implementation: How H Enters the Pipeline <a name="2-current-implementation"></a>

### 2.1 Flow Parameter Computation

**File**: `src/utils/flow_params.py`

The core function `compute_flow_params_batch` (line 167) computes per-particle
$A(\lambda)$ and $b(\lambda)$ for all $N$ particles simultaneously:

```python
# Line 219: Batch Jacobians — THE critical call
H_batch = model.observation_jacobian_batch(linearization_points)  # (N, od, sd)

# Lines 221-229: Build S = λ H P H^T + R
HP = tf.matmul(H_batch, P_b)                    # (N, od, sd)
H_T = tf.linalg.matrix_transpose(H_batch)       # (N, sd, od)
HPH = tf.matmul(HP, H_T)                        # (N, od, od)
S = lambda_val * HPH + tf.expand_dims(R, 0)     # (N, od, od)

# Lines 231-237: Compute A(λ) = -0.5 P H^T (S)^{-1} H
L_S = safe_cholesky(S)
S_inv_H = tf.linalg.cholesky_solve(L_S, H_batch)
PH_T = tf.matmul(P_b, H_T)                      # (N, sd, od)
A_batch = -0.5 * tf.matmul(PH_T, S_inv_H)       # (N, sd, sd)

# Lines 239-242: Nonlinear correction e = h(x) - H @ x
h_batch = model.observation_function_batch(linearization_points)
Hx = tf.einsum('nij,nj->ni', H_batch, linearization_points)
e_batch = h_batch - Hx

# Lines 244-268: Compute b(λ) using H^T R^{-1} (z - e)
PHT_Rinv = tf.matmul(PH_T, tf.expand_dims(R_inv, 0))
z_minus_e = tf.expand_dims(observation, 0) - e_batch
inner = tf.einsum('nij,nj->ni', tf.matmul(I_lA, PHT_Rinv), z_minus_e)
# ... remainder of b computation
```

### 2.2 Batched EKF Covariance Update

**File**: `src/filters/kalman/batched_ekf.py`

Called after the flow (line 264 of `ledh_invertible.py`) to update per-particle
covariances for the next timestep:

```python
# Line 86: Jacobian computation
H_batch = model.observation_jacobian_batch(means)  # (N, od, sd)

# Lines 94-103: Standard Kalman update using H
H_cov = tf.matmul(H_batch, covs)
S = tf.matmul(H_cov, H_T) + tf.expand_dims(R, 0)
L_S = safe_cholesky(S)
K_T = tf.linalg.cholesky_solve(L_S, tf.matmul(H_batch, covs))
K = tf.linalg.matrix_transpose(K_T)
```

### 2.3 Weight Computation (Unaffected)

**File**: `src/utils/distributions.py`, function `compute_flow_weights`

The weight formula (line 201) uses `log_observation_prob_batch`, which
evaluates the **actual** observation log-density $\log p(y | x)$ — not the
Jacobian. This correctly captures both mean and variance channels:

```python
log_p_obs = model.log_observation_prob_batch(observation, eta_1)
```

This means particle weights are correct even when the flow is dead. The filter
still works — it just works as a bootstrap particle filter with wasted flow
computation.

### 2.4 Summary of H Usage

| Location | File:Line | What it computes | SV impact |
|----------|-----------|------------------|-----------|
| Flow A(λ) | `flow_params.py:219-237` | Per-particle flow matrix | **A = 0** (dead flow) |
| Flow b(λ) | `flow_params.py:244-268` | Per-particle flow drift | **b = 0** (dead flow) |
| Nonlinear correction e | `flow_params.py:239-242` | $h(x) - Hx$ | e = 0 (irrelevant) |
| Batched EKF update | `batched_ekf.py:86-117` | Per-particle cov tracking | K = 0 (no cov reduction for $x_2$) |
| Weights | `distributions.py:175` | Observation likelihood | **Correct** (uses full $\log p(y|x)$) |

---

## 3. Mathematical Foundation: From Jacobian to Score <a name="3-mathematical-foundation"></a>

### 3.1 The Key Identity

For the additive noise model $y = h(x) + v$, $v \sim \mathcal{N}(0, R)$, the
observation log-density is:

$$
\log p(y \mid x) = -\tfrac{1}{2}(y - h(x))^\top R^{-1}(y - h(x)) - \tfrac{1}{2}\log|2\pi R|
$$

The **score** (gradient of log-density w.r.t. state) is:

$$
g \triangleq \nabla_x \log p(y \mid x) = H^\top R^{-1}(y - h(x))
$$

The **negative Hessian** (under Gauss-Newton approximation, dropping the
second-order term $\nabla^2 h$) is:

$$
\Lambda \triangleq -\nabla^2_x \log p(y \mid x) \approx H^\top R^{-1} H
$$

This $\Lambda$ is always positive semidefinite (it's a Gram matrix).

### 3.2 Rewriting the Flow in Terms of Score and Information

The standard flow equations use $H$ and $R$ throughout. We can express
everything in terms of $g$ (score) and $\Lambda$ (negative Hessian /
information matrix):

**Identity 1**: $H^\top R^{-1} H = \Lambda$ (observation information)

**Identity 2** (push-through):

$$
H^\top(\lambda H P H^\top + R)^{-1} H = \Lambda (I + \lambda P \Lambda)^{-1}
$$

*Proof*: Using the matrix identity $(B^\top C^{-1} B)(I + \lambda A B^\top C^{-1} B)^{-1} = B^\top (C + \lambda B A B^\top)^{-1} B$ with $B = H$, $C = R$, $A = P$.

**Identity 3**: The term $H^\top R^{-1}(z - e)$ in the $b$ formula equals
the score plus an information-weighted linearization point:

$$
H^\top R^{-1}(z - e) = g + \Lambda \bar{x}
$$

where $\bar{x}$ is the linearization point and $e = h(\bar{x}) - H\bar{x}$.

*Proof*:

$$
H^\top R^{-1}(z - e) = H^\top R^{-1}(y - h(\bar{x}) + H\bar{x})
= \underbrace{H^\top R^{-1}(y - h(\bar{x}))}_{= g \text{ (score at } \bar{x})} + \underbrace{H^\top R^{-1} H}_{= \Lambda} \bar{x}
$$

### 3.3 The Generalized Flow Equations

Substituting Identities 1-3 into the Li & Coates (2017) equations:

**Equation (10) generalized**:

$$
\boxed{A(\lambda) = -\tfrac{1}{2} P \Lambda (I + \lambda P \Lambda)^{-1}}
$$

**Equation (11) generalized**:

$$
\boxed{b(\lambda) = (I + 2\lambda A)\bigl[(I + \lambda A) P (g + \Lambda \bar{x}) + A \bar{\eta}_0\bigr]}
$$

where:
- $g = \nabla_x \log p(y \mid x)\big|_{\bar{x}}$ is the score at the
  linearization point
- $\Lambda$ is the observation information matrix at $\bar{x}$ (must be PSD)
- $\bar{x}$ is the linearization point (the particle itself for LEDH)
- $\bar{\eta}_0$ is the prior mean

**For additive noise models**, these reduce exactly to the standard equations.
No existing model behavior changes.

### 3.4 Why the Negative Hessian Fails for SV

For a general observation density, the natural candidate for $\Lambda$ is the
negative Hessian $-\nabla^2_x \log p(y \mid x)$. However, this is **not
guaranteed to be PSD** for non-log-concave likelihoods.

For the 2D SV model $y \mid x \sim \mathcal{N}(bx_1, \exp(x_2))$:

$$
\log p(y \mid x) = -\tfrac{1}{2}\bigl[\log(2\pi) + x_2 + (y - bx_1)^2 \exp(-x_2)\bigr]
$$

The Hessian is:

$$
\nabla^2_x \log p(y \mid x) = \begin{pmatrix}
-b^2 e^{-x_2} & -b(y - bx_1)e^{-x_2} \\
-b(y - bx_1)e^{-x_2} & -\tfrac{1}{2}(y - bx_1)^2 e^{-x_2}
\end{pmatrix}
$$

The determinant of $-\nabla^2$:

$$
\det(\Lambda_{\text{Hess}}) = b^2 e^{-x_2} \cdot \tfrac{1}{2}(y-bx_1)^2 e^{-x_2} - b^2(y-bx_1)^2 e^{-2x_2}
= -\tfrac{1}{2} b^2 (y - bx_1)^2 e^{-2x_2} \leq 0
$$

The determinant is **negative** whenever $y \neq bx_1$, so $\Lambda_{\text{Hess}}$ is
**indefinite**. Using it in the flow equation $A(\lambda) = -\frac{1}{2} P \Lambda (I + \lambda P \Lambda)^{-1}$ would produce an
unstable flow.

### 3.5 The Fisher Information: A PSD Alternative

The **Fisher information matrix** is the *expected* negative Hessian, averaging
over $y \mid x$. For $y \mid x \sim \mathcal{N}(h(x), R(x))$, the Fisher
information has two terms (see Schervish, *Theory of Statistics*, 1995):

$$
\boxed{[\Lambda_{\text{FI}}]_{ij} = \left(\frac{\partial h}{\partial x_i}\right)^\top R^{-1} \frac{\partial h}{\partial x_j} + \frac{1}{2}\operatorname{tr}\!\left(R^{-1}\frac{\partial R}{\partial x_i} R^{-1}\frac{\partial R}{\partial x_j}\right)}
$$

- **First term**: the standard $H^\top R^{-1} H$, capturing information from
  the observation mean.
- **Second term**: information from the **state-dependent noise covariance**,
  capturing the variance channel.

For standard additive noise models ($R$ constant), $\partial R / \partial x_i = 0$, so the second term vanishes and $\Lambda_{\text{FI}} = H^\top R^{-1} H$ as usual.

The Fisher information is **always positive semidefinite** (it's an expectation
of outer products of the score). This guarantees flow stability.

---

## 4. The 2D Stochastic Volatility Model <a name="4-the-2d-sv-model"></a>

### 4.1 Model Specification

**State**: $x_t = (x_{1t}, x_{2t})^\top \in \mathbb{R}^2$

**State transition**:

$$
x_{t+1} = A x_t + e_t, \quad e_t \sim \mathcal{N}(0, \Sigma)
$$

where $A$ is a $2 \times 2$ matrix and $\Sigma$ is a $2 \times 2$ positive
definite covariance. For the diagonal case:

$$
A = \begin{pmatrix} a_1 & 0 \\ 0 & a_2 \end{pmatrix}, \quad
\Sigma = \begin{pmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{pmatrix}
$$

**Observation**:

$$
y_t = b \cdot x_{1t} + \exp(x_{2t}/2) \cdot v_t, \quad v_t \sim \mathcal{N}(0, 1)
$$

This gives:

$$
y_t \mid x_t \sim \mathcal{N}\bigl(b \cdot x_{1t},\; \exp(x_{2t})\bigr)
$$

- $x_{1t}$: the **level** component (mean of observations)
- $x_{2t}$: the **log-volatility** component (log-variance of observations)

### 4.2 Stationarity Conditions

The VAR(1) process $x_{t+1} = Ax_t + e_t$ is **covariance-stationary** iff
all eigenvalues of $A$ have modulus strictly less than 1:

$$
\rho(A) < 1 \quad \Longleftrightarrow \quad |\lambda_i(A)| < 1 \;\;\forall i
$$

For diagonal $A = \text{diag}(a_1, a_2)$, this simplifies to $|a_1| < 1$
and $|a_2| < 1$.

For general $A$, compute eigenvalues: $\lambda = \frac{1}{2}\bigl[\text{tr}(A) \pm \sqrt{\text{tr}(A)^2 - 4\det(A)}\bigr]$ and check their moduli.

### 4.3 Stationary Initial Distribution

For $x_0$ to make the process stationary, we need:

$$
x_0 \sim \mathcal{N}(0, \Sigma_\infty)
$$

where $\Sigma_\infty$ solves the **discrete Lyapunov equation**:

$$
\Sigma_\infty = A \Sigma_\infty A^\top + \Sigma
$$

**Diagonal case** (closed form):

$$
\Sigma_\infty = \text{diag}\!\left(\frac{\sigma_1^2}{1 - a_1^2},\; \frac{\sigma_2^2}{1 - a_2^2}\right)
$$

**General case**: solve via $\text{vec}(\Sigma_\infty) = (I - A \otimes A)^{-1}\text{vec}(\Sigma)$, or use `scipy.linalg.solve_discrete_lyapunov(A, Sigma)`.

### 4.4 Observation Density

$$
\log p(y \mid x) = -\frac{1}{2}\left[\log(2\pi) + x_2 + (y - bx_1)^2 \exp(-x_2)\right]
$$

### 4.5 Observation Mean and Jacobian (Standard Interface)

$$
h(x) = \mathbb{E}[y \mid x] = bx_1
$$

$$
H = \frac{\partial h}{\partial x} = \begin{pmatrix} b & 0 \end{pmatrix}
$$

The second column is zero: $H$ provides **no information** about $x_2$.

### 4.6 State-Dependent Observation Covariance

$$
R(x) = \exp(x_2), \qquad
\frac{\partial R}{\partial x_1} = 0, \qquad
\frac{\partial R}{\partial x_2} = \exp(x_2)
$$

---

## 5. Fisher Information for the 2D SV Model <a name="5-fisher-information"></a>

### 5.1 Derivation

Applying the general formula from Section 3.5:

**Mean-channel term** ($H^\top R^{-1} H$):

$$
H^\top R^{-1} H = \begin{pmatrix} b \\ 0 \end{pmatrix} e^{-x_2} \begin{pmatrix} b & 0 \end{pmatrix}
= \begin{pmatrix} b^2 e^{-x_2} & 0 \\ 0 & 0 \end{pmatrix}
$$

**Variance-channel term** ($\frac{1}{2}\text{tr}(R^{-1} \frac{\partial R}{\partial x_i} R^{-1} \frac{\partial R}{\partial x_j})$):

Since $R = \exp(x_2)$ is scalar:

$$
\frac{1}{2}\text{tr}\!\left(R^{-1}\frac{\partial R}{\partial x_i} R^{-1}\frac{\partial R}{\partial x_j}\right) = \frac{1}{2} R^{-2} \frac{\partial R}{\partial x_i}\frac{\partial R}{\partial x_j}
$$

- $(i, j) = (1, 1)$: $\frac{1}{2} e^{-2x_2} \cdot 0 \cdot 0 = 0$
- $(i, j) = (1, 2)$ or $(2, 1)$: $\frac{1}{2} e^{-2x_2} \cdot 0 \cdot e^{x_2} = 0$
- $(i, j) = (2, 2)$: $\frac{1}{2} e^{-2x_2} \cdot e^{x_2} \cdot e^{x_2} = \frac{1}{2}$

**Combined Fisher information**:

$$
\boxed{\Lambda_{\text{FI}} = \begin{pmatrix} b^2 e^{-x_2} & 0 \\ 0 & \frac{1}{2} \end{pmatrix}}
$$

This is:
- **Diagonal** and **positive definite** for all $x$
- $x_1$ gets information $b^2 e^{-x_2}$ from the mean channel (modulated by
  volatility — higher volatility means less information)
- $x_2$ gets **constant** information $\frac{1}{2}$ from the variance channel

### 5.2 Score Vector

$$
g = \nabla_x \log p(y \mid x) = \begin{pmatrix}
b(y - bx_1) e^{-x_2} \\
-\frac{1}{2} + \frac{1}{2}(y - bx_1)^2 e^{-x_2}
\end{pmatrix}
$$

**Interpretation**:
- $g_1$: proportional to the standardized residual $(y - bx_1) / \sigma$ scaled
  by $b$ — the standard Kalman "innovation signal"
- $g_2$: compares the squared residual $(y - bx_1)^2$ to the predicted variance
  $e^{x_2}$. If the squared residual exceeds the variance, $g_2 > 0$ (increase
  $x_2$); if below, $g_2 < 0$ (decrease $x_2$)

### 5.3 What the Flow Looks Like

With $\Lambda_{\text{FI}}$ diagonal, the generalized flow has clean structure:

$$
A(\lambda) = -\frac{1}{2} P \Lambda_{\text{FI}} (I + \lambda P \Lambda_{\text{FI}})^{-1}
$$

If $P$ is also diagonal (e.g., at initialization or for uncoupled states):

$$
A_{kk}(\lambda) = -\frac{P_{kk} \lambda_{k}}{2(1 + \lambda P_{kk} \lambda_{k})}
$$

where $\lambda_1 = b^2 e^{-x_2}$ and $\lambda_2 = 1/2$. Both components get
nonzero flow.

For the drift $b(\lambda)$, the score $g$ drives both components:
- $x_1$ particles move toward $x_1 + b^{-1}(y - bx_1)$ (match observation mean)
- $x_2$ particles move to match observation variance to $(y - bx_1)^2$

---

## 6. Proposed Code Changes <a name="6-proposed-code-changes"></a>

### 6.1 Model Interface: New Methods on `StateSpaceModel`

Add to `src/core/model_base.py`:

```python
# --- Score-based interface (for models with state-dependent noise) ---

def observation_score(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """Score of observation density: nabla_x log p(y|x).

    Default: H^T R^{-1} (y - h(x))  (additive noise).
    Override for state-dependent noise models.

    Args:
        y: observation (obs_dim,)
        x: state (state_dim,)
    Returns:
        score vector (state_dim,)
    """
    H = self.observation_jacobian(x)
    R = self.observation_cov(x)
    R_inv = tf.linalg.inv(R)
    residual = y - self.observation_mean(x)
    return tf.linalg.matvec(tf.transpose(H) @ R_inv, residual)

def observation_fisher_info(self, x: tf.Tensor) -> tf.Tensor:
    """Fisher information matrix for observation density.

    Default: H^T R^{-1} H  (additive noise, variance channel = 0).
    Override for state-dependent noise models.

    Args:
        x: state (state_dim,)
    Returns:
        Lambda_FI: (state_dim, state_dim), positive semidefinite
    """
    H = self.observation_jacobian(x)
    R_inv = tf.linalg.inv(self.observation_cov(x))
    return tf.transpose(H) @ R_inv @ H

# Batch versions
def observation_score_batch(self, y: tf.Tensor,
                            particles: tf.Tensor) -> tf.Tensor:
    """Batched score. Default: map_fn over observation_score.

    Args:
        y: observation (obs_dim,)
        particles: (N, state_dim)
    Returns:
        scores: (N, state_dim)
    """
    return tf.map_fn(
        lambda x: self.observation_score(y, x),
        particles,
        fn_output_signature=tf.TensorSpec([self.state_dim], particles.dtype)
    )

def observation_fisher_info_batch(self,
                                   particles: tf.Tensor) -> tf.Tensor:
    """Batched Fisher information. Default: map_fn.

    Args:
        particles: (N, state_dim)
    Returns:
        Lambda_batch: (N, state_dim, state_dim)
    """
    return tf.map_fn(
        self.observation_fisher_info,
        particles,
        fn_output_signature=tf.TensorSpec(
            [self.state_dim, self.state_dim], particles.dtype
        )
    )

@property
def has_state_dependent_obs_noise(self) -> bool:
    """Whether observation noise R depends on state. Default: False.
    When True, flow params use score/Fisher instead of H/R."""
    return False
```

**Key design point**: The default implementations reduce to $H^\top R^{-1} H$
and $H^\top R^{-1}(y - h(x))$, so **all existing models work unchanged**
without overriding anything.

### 6.2 2D Stochastic Volatility Model Implementation

New file `src/models/stochastic_volatility_2d.py`:

```python
"""2D Stochastic Volatility model with level and log-volatility."""

import numpy as np
import tensorflow as tf
from ..core.model_base import StateSpaceModel


class StochasticVolatility2D(StateSpaceModel):
    """
    2D Stochastic Volatility Model.

    State: x_t = [x_1t (level), x_2t (log-volatility)]

    Transition:
        x_{t+1} = A x_t + e_t,  e_t ~ N(0, Sigma)

    Observation:
        y_t = b * x_1t + exp(x_2t / 2) * v_t,  v_t ~ N(0, 1)
        => y_t | x_t ~ N(b * x_1t, exp(x_2t))

    Parameters:
        A: 2x2 transition matrix (eigenvalues inside unit circle)
        Sigma: 2x2 process noise covariance (positive definite)
        b: observation coefficient for level component
    """

    def __init__(self, A=None, Sigma=None, b: float = 1.0, dtype=None):
        if dtype is None:
            dtype = tf.float64
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        # Defaults: diagonal, stationary
        if A is None:
            A = np.array([[0.95, 0.0], [0.0, 0.91]], dtype=self.np_dtype)
        if Sigma is None:
            Sigma = np.array([[0.1, 0.0], [0.0, 1.0]], dtype=self.np_dtype)

        self._A = tf.constant(A, dtype=dtype)
        self._Sigma = tf.constant(Sigma, dtype=dtype)
        self._b = tf.constant(b, dtype=dtype)
        self._pi2 = tf.constant(2.0 * np.pi, dtype=dtype)

        # Verify stationarity: eigenvalues of A inside unit circle
        eigvals = np.linalg.eigvals(A)
        if np.any(np.abs(eigvals) >= 1.0):
            raise ValueError(
                f"A has eigenvalue(s) with |lambda| >= 1: {eigvals}. "
                "Process is not stationary."
            )

        # Compute stationary covariance (discrete Lyapunov equation)
        from scipy.linalg import solve_discrete_lyapunov
        Sigma_np = np.array(Sigma, dtype=np.float64)
        A_np = np.array(A, dtype=np.float64)
        self._Sigma_inf = tf.constant(
            solve_discrete_lyapunov(A_np, Sigma_np), dtype=dtype
        )

        # Cholesky of process noise (for sampling)
        self._L_Sigma = tf.linalg.cholesky(self._Sigma)

    # --- Properties ---
    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def mu_0(self) -> tf.Tensor:
        return tf.zeros([2], dtype=self.dtype)

    @property
    def Sigma_0(self) -> tf.Tensor:
        return self._Sigma_inf

    @property
    def observation_noise_cov(self) -> tf.Tensor:
        # Nominal R = 1 (the exp(x_2) factor is state-dependent)
        return tf.ones([1, 1], dtype=self.dtype)

    @property
    def process_noise_cov(self) -> tf.Tensor:
        return self._Sigma

    @property
    def has_state_dependent_obs_noise(self) -> bool:
        return True

    # --- Sampling ---
    def sample_initial_state(self, seed):
        L_inf = tf.linalg.cholesky(self._Sigma_inf)
        z = tf.random.stateless_normal([2], seed=seed, dtype=self.dtype)
        return L_inf @ z

    def sample_state_transition(self, x, seed):
        w = tf.random.stateless_normal([2], seed=seed, dtype=self.dtype)
        return tf.linalg.matvec(self._A, x) + self._L_Sigma @ w

    def sample_observation(self, x, seed):
        v = tf.random.stateless_normal([1], seed=seed, dtype=self.dtype)
        mean = self._b * x[0]
        std = tf.exp(x[1] / 2.0)
        return tf.reshape(mean + std * v[0], [1])

    # --- Deterministic (for EKF/UKF) ---
    def state_transition_mean(self, x):
        return tf.linalg.matvec(self._A, x)

    def state_transition_cov(self, x):
        return self._Sigma

    def state_jacobian(self, x):
        return self._A

    def observation_mean(self, x):
        return tf.reshape(self._b * x[0], [1])

    def observation_cov(self, x):
        """State-dependent: Var[y|x] = exp(x_2)."""
        return tf.reshape(tf.exp(x[1]), [1, 1])

    def observation_jacobian(self, x):
        """H = [b, 0] — no information about x_2 from the mean."""
        return tf.stack([[self._b, tf.constant(0.0, dtype=self.dtype)]])

    def observation_function(self, x):
        return self.observation_mean(x)

    def observation_hessian(self, x):
        return tf.zeros([1, 2, 2], dtype=self.dtype)

    # --- Log probability ---
    def log_observation_prob(self, y, x):
        var = tf.exp(x[1])
        residual = y[0] - self._b * x[0]
        return -0.5 * (tf.math.log(self._pi2 * var) + residual**2 / var)

    # --- Score-based interface (the key addition) ---
    def observation_score(self, y, x):
        """Score: nabla_x log p(y|x).

        g_1 = b * (y - b*x_1) * exp(-x_2)
        g_2 = -1/2 + (y - b*x_1)^2 * exp(-x_2) / 2
        """
        residual = y[0] - self._b * x[0]
        exp_neg_x2 = tf.exp(-x[1])
        g1 = self._b * residual * exp_neg_x2
        g2 = -0.5 + 0.5 * residual**2 * exp_neg_x2
        return tf.stack([g1, g2])

    def observation_fisher_info(self, x):
        """Fisher information: diag(b^2 exp(-x_2), 1/2).

        Always positive definite. Captures variance-channel information.
        """
        exp_neg_x2 = tf.exp(-x[1])
        return tf.linalg.diag(tf.stack([
            self._b**2 * exp_neg_x2,
            tf.constant(0.5, dtype=self.dtype)
        ]))

    # --- Batch methods ---
    def sample_initial_state_batch(self, n, seed):
        L_inf = tf.linalg.cholesky(self._Sigma_inf)
        z = tf.random.stateless_normal([n, 2], seed=seed, dtype=self.dtype)
        return z @ tf.transpose(L_inf)

    def state_transition_batch(self, particles, seed, t=None):
        w = tf.random.stateless_normal(
            tf.shape(particles), seed=seed, dtype=self.dtype
        )
        return particles @ tf.transpose(self._A) + w @ tf.transpose(self._L_Sigma)

    def state_transition_mean_batch(self, particles, t=None):
        return particles @ tf.transpose(self._A)

    def state_transition_cov_batch(self, particles):
        return self._Sigma

    def log_observation_prob_batch(self, observation, particles):
        var = tf.exp(particles[:, 1])
        residual = observation[0] - self._b * particles[:, 0]
        return -0.5 * (tf.math.log(self._pi2 * var) + residual**2 / var)

    def observation_jacobian_batch(self, particles):
        N = tf.shape(particles)[0]
        H_row = tf.stack([self._b, tf.constant(0.0, dtype=self.dtype)])
        return tf.tile(tf.reshape(H_row, [1, 1, 2]), [N, 1, 1])

    def observation_function_batch(self, particles):
        return tf.reshape(self._b * particles[:, 0], [-1, 1])

    def state_jacobian_batch(self, particles):
        N = tf.shape(particles)[0]
        return tf.tile(tf.expand_dims(self._A, 0), [N, 1, 1])

    def observation_score_batch(self, y, particles):
        """Batched score: (N, 2)."""
        residual = y[0] - self._b * particles[:, 0]
        exp_neg_x2 = tf.exp(-particles[:, 1])
        g1 = self._b * residual * exp_neg_x2
        g2 = -0.5 + 0.5 * residual**2 * exp_neg_x2
        return tf.stack([g1, g2], axis=1)

    def observation_fisher_info_batch(self, particles):
        """Batched Fisher information: (N, 2, 2), diagonal."""
        N = tf.shape(particles)[0]
        exp_neg_x2 = tf.exp(-particles[:, 1])
        fi_11 = self._b**2 * exp_neg_x2
        fi_22 = tf.fill([N], tf.constant(0.5, dtype=self.dtype))
        zeros = tf.zeros([N], dtype=self.dtype)
        # Build (N, 2, 2) diagonal matrices
        row1 = tf.stack([fi_11, zeros], axis=1)
        row2 = tf.stack([zeros, fi_22], axis=1)
        return tf.stack([row1, row2], axis=2)
```

### 6.3 Generalized Flow Parameters

New function in `src/utils/flow_params.py`. Add alongside existing functions:

```python
@tf.function
def compute_flow_params_score_batch(
    model,
    linearization_points: tf.Tensor,   # (N, sd)
    lambda_val: tf.Tensor,
    observation: tf.Tensor,             # (od,)
    P: tf.Tensor,                       # (N, sd, sd) or (sd, sd)
    eta_bar_0: tf.Tensor,               # (N, sd) or (sd,)
    state_dim: int,
    regularization: tf.Tensor = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute A(lambda) and b(lambda) using score and Fisher information.

    Generalized Daum-Huang flow for models with state-dependent noise.
    Reduces to standard flow for additive-noise models.

    Equations:
        A(lambda) = -1/2 P Lambda (I + lambda P Lambda)^{-1}
        b(lambda) = (I + 2 lambda A)[(I + lambda A) P (g + Lambda x_bar) + A eta_bar_0]

    where:
        g = nabla_x log p(y|x)  (score)
        Lambda = Fisher information (always PSD)

    Args:
        model: StateSpaceModel with observation_score_batch,
               observation_fisher_info_batch
        linearization_points: (N, state_dim)
        lambda_val: scalar pseudo-time
        observation: (obs_dim,)
        P: covariance — (N, sd, sd) per-particle or (sd, sd) global
        eta_bar_0: prior mean — (N, sd) per-particle or (sd,) global
        state_dim: state dimension
        regularization: optional regularization strength

    Returns:
        A_batch: (N, state_dim, state_dim)
        b_batch: (N, state_dim)
    """
    if regularization is None:
        regularization = tf.constant(0.0, dtype=P.dtype)

    # Broadcast P to (N, sd, sd)
    if len(P.shape) == 2:
        P_b = tf.expand_dims(P, 0)
    else:
        P_b = P

    # Regularize P
    if regularization > 0.0:
        trace_P = tf.linalg.trace(P_b)
        state_dim_f = tf.cast(state_dim, P_b.dtype)
        reg_strength = regularization * (trace_P / state_dim_f)
        I_sd = tf.eye(state_dim, dtype=P_b.dtype)
        P_b = P_b + reg_strength[..., tf.newaxis, tf.newaxis] * I_sd

    I_sd = tf.eye(state_dim, dtype=P_b.dtype)
    I_batch = tf.expand_dims(I_sd, 0)

    # --- Score and Fisher information (batched) ---
    # g: (N, sd) — the score at each particle
    g_batch = model.observation_score_batch(observation, linearization_points)

    # Lambda: (N, sd, sd) — Fisher information at each particle
    Lambda_batch = model.observation_fisher_info_batch(linearization_points)

    # --- A(lambda) = -1/2 P Lambda (I + lambda P Lambda)^{-1} ---
    P_Lambda = tf.matmul(P_b, Lambda_batch)       # (N, sd, sd)
    S_gen = I_batch + lambda_val * P_Lambda        # (N, sd, sd)

    # Solve: S_gen @ X = P_Lambda  =>  X = S_gen^{-1} P_Lambda
    # Instead compute A = -0.5 * P Lambda S_gen^{-1}
    # which is  A = -0.5 * P_Lambda @ S_gen^{-1}
    L_S = safe_cholesky(S_gen)
    # cholesky_solve(L, B) computes  L L^T X = B  =>  X = S_gen^{-1} B
    # We need P_Lambda @ S_gen^{-1} = (S_gen^{-T} @ (P_Lambda)^T)^T
    # For symmetric S_gen: S_gen^{-1} = S_gen^{-T}
    # So: (P_Lambda @ S_gen^{-1})^T = S_gen^{-1} @ P_Lambda^T
    S_inv_PLam_T = tf.linalg.cholesky_solve(
        L_S, tf.linalg.matrix_transpose(P_Lambda)
    )
    A_batch = -0.5 * tf.linalg.matrix_transpose(S_inv_PLam_T)  # (N, sd, sd)

    # --- b(lambda) = (I + 2 lambda A)[(I + lambda A) P (g + Lambda x_bar) + A eta_bar_0] ---

    # g + Lambda @ x_bar: (N, sd)
    Lambda_x = tf.einsum('nij,nj->ni', Lambda_batch, linearization_points)
    g_plus_Lambda_x = g_batch + Lambda_x

    # P @ (g + Lambda x_bar): (N, sd)
    P_g_Lx = tf.einsum('nij,nj->ni', P_b, g_plus_Lambda_x)

    # (I + lambda A): (N, sd, sd)
    I_lA = I_batch + lambda_val * A_batch

    # (I + lambda A) @ P @ (g + Lambda x_bar): (N, sd)
    inner_term1 = tf.einsum('nij,nj->ni', I_lA, P_g_Lx)

    # A @ eta_bar_0: (N, sd)
    if len(eta_bar_0.shape) == 1:
        A_eta = tf.einsum('nij,j->ni', A_batch, eta_bar_0)
    else:
        A_eta = tf.einsum('nij,nj->ni', A_batch, eta_bar_0)

    # (I + 2 lambda A) @ (term1 + term2): (N, sd)
    I_2lA = I_batch + 2 * lambda_val * A_batch
    b_batch = tf.einsum('nij,nj->ni', I_2lA, inner_term1 + A_eta)

    return A_batch, b_batch
```

### 6.4 Generalized Batched EKF Update

For per-particle covariance tracking with Fisher information, the update uses
the **information-form** Kalman update:

$$
P_{\text{update}}^{-1} = P_{\text{pred}}^{-1} + \Lambda_{\text{FI}}
$$

or equivalently via Woodbury:

$$
P_{\text{update}} = P_{\text{pred}} - P_{\text{pred}} \Lambda_{\text{FI}} (I + P_{\text{pred}} \Lambda_{\text{FI}})^{-1} P_{\text{pred}}
$$

$$
m_{\text{update}} = m_{\text{pred}} + P_{\text{update}} \cdot g
$$

New function in `src/filters/kalman/batched_ekf.py`:

```python
@tf.function
def batched_ekf_update_score(
    model,
    means: tf.Tensor,          # (N, sd)
    covs: tf.Tensor,           # (N, sd, sd)
    observation: tf.Tensor,    # (od,)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched EKF update using score and Fisher information.

    For models with state-dependent noise where H alone is insufficient.

    Covariance update (Woodbury form):
        P_upd = P - P Lambda (I + P Lambda)^{-1} P

    Mean update:
        m_upd = m + P_upd @ g

    where g = score, Lambda = Fisher information.

    Args:
        model: StateSpaceModel with observation_score_batch,
               observation_fisher_info_batch
        means: (N, state_dim) predicted means
        covs: (N, state_dim, state_dim) predicted covariances
        observation: (obs_dim,) observation vector

    Returns:
        mean_updated: (N, state_dim)
        cov_updated: (N, state_dim, state_dim)
    """
    sd = model.state_dim
    I_sd = tf.eye(sd, dtype=covs.dtype)

    # Score and Fisher info at each particle
    g_batch = model.observation_score_batch(observation, means)       # (N, sd)
    Lambda = model.observation_fisher_info_batch(means)               # (N, sd, sd)

    # P @ Lambda: (N, sd, sd)
    P_Lam = tf.matmul(covs, Lambda)

    # (I + P Lambda): (N, sd, sd)
    S = tf.expand_dims(I_sd, 0) + P_Lam

    # Woodbury: P_upd = P - P Lambda (I + P Lambda)^{-1} P
    L_S = safe_cholesky(S)
    S_inv_P = tf.linalg.cholesky_solve(L_S, covs)     # (I+P Lam)^{-1} P
    P_Lam_S_inv_P = tf.matmul(P_Lam, S_inv_P)         # P Lam (I+P Lam)^{-1} P
    cov_updated = symmetrize(covs - P_Lam_S_inv_P)

    # m_upd = m + P_upd @ g
    mean_updated = means + tf.einsum('nij,nj->ni', cov_updated, g_batch)

    return mean_updated, cov_updated
```

### 6.5 LEDH Integration

In `src/filters/particle/ledh_invertible.py`, the `update` method needs a
conditional branch:

```python
def update(self, y: tf.Tensor):
    """Update step — uses score-based flow if model has state-dependent noise."""
    R = self.model.observation_noise_cov
    eta_1 = self.eta_0.value()
    eta_bar = self.eta_bar_0.value()
    # ... (existing setup code) ...

    use_score = getattr(self.model, 'has_state_dependent_obs_noise', False)

    for j in range(self.n_lambda_steps):
        d_lambda = self.lambda_steps[j]
        lambda_val = lambda_val + d_lambda

        if use_score:
            A_batch, b_batch = compute_flow_params_score_batch(
                self.model, eta_bar, lambda_val, y,
                particle_covs_tf, eta_bar_0_tf,
                self.state_dim, regularization_tf
            )
        else:
            A_batch, b_batch = compute_flow_params_batch(
                self.model, eta_bar, lambda_val, y,
                particle_covs_tf, R, R_inv,
                eta_bar_0_tf, self.state_dim, regularization_tf
            )

        # ... (flow step, Jacobian accumulation — unchanged) ...

    # ... (weight computation — unchanged, uses log_observation_prob) ...

    # Update covariances
    if use_score:
        _, cov_updated = batched_ekf_update_score(
            self.model, self.eta_bar_0.value(),
            self.particle_covs.value(), y
        )
    else:
        _, cov_updated = batched_ekf_update(
            self.model, self.eta_bar_0.value(),
            self.particle_covs.value(), y
        )
    self.particle_covs.assign(cov_updated)
```

### 6.6 EKF and UKF Standalone

For standalone EKF/UKF (used in HMC inference), two options:

**Option A (minimal)**: Use the information-form update for the mean and
covariance, same as the batched version above. This lets EKF track $x_2$
properly.

**Option B (log-likelihood)**: For the log-likelihood computation in
`log_marginal_likelihood_tf`, replace the innovation-based formula with the
score-based one:

```python
# Standard: log p(y_t | y_{1:t-1}) = -0.5 [log|2 pi S| + nu^T S^{-1} nu]
# Score-based: use the actual log p(y|x) evaluated at the filtered mean
# as an approximation (Laplace-style)

# Or better: keep using the actual model.log_observation_prob
# evaluated at the predicted mean, which works for all models.
```

For the EKF/UKF `log_marginal_likelihood_tf`, the cleanest approach is to
compute $\log p(y_t \mid y_{1:t-1})$ using the predictive distribution
directly from the model's `log_observation_prob`, evaluated with numerical
integration or approximation. This is a separate consideration from the flow
fix.

---

## 7. Impact on Each Filter Type <a name="7-impact-on-each-filter"></a>

### 7.1 Summary Table

| Filter | Current SV behavior | After score-based fix |
|--------|--------------------|-----------------------|
| **BPF** (bootstrap) | Works correctly | No change needed |
| **LEDH** (local flow) | Dead flow for $x_2$ | Full flow for both components |
| **EDH** (global flow) | Dead flow for $x_2$ | Full flow for both components |
| **EKF** (standalone) | No $x_2$ update (H=0) | $x_2$ updated via Fisher info |
| **UKF** (standalone) | No $x_2$ update (sigma pts all map to same obs mean) | $x_2$ updated via Fisher info |
| **LEDH covariance tracking** | $x_2$ covariance never reduces | $x_2$ covariance properly shrinks |
| **Weights** | Correct (uses full likelihood) | No change needed |

### 7.2 Backward Compatibility

All changes are **fully backward compatible**:

- `has_state_dependent_obs_noise` defaults to `False`
- Default `observation_score` and `observation_fisher_info` compute
  $H^\top R^{-1}(y - h(x))$ and $H^\top R^{-1} H$ respectively
- Existing models (linear Gaussian, range-bearing, cubic sensor, 1D SV)
  continue to work identically
- The LEDH `update` method dispatches based on the model flag

### 7.3 What Stays the Same

- Weight computation (`compute_flow_weights`): unchanged, already correct
- Flow step mechanics (Euler integration, Jacobian accumulation): unchanged
- Resampling: unchanged
- Data generation: unchanged (uses `sample_observation`, not the score)
- `log_marginal_likelihood_tf` for HMC: unchanged (independent computation)

### 7.4 Implementation Order

1. Add `observation_score`, `observation_fisher_info` (+ batch versions) to
   `StateSpaceModel` with default implementations
2. Implement `StochasticVolatility2D` model with score/Fisher overrides
3. Add `compute_flow_params_score_batch` to `flow_params.py`
4. Add `batched_ekf_update_score` to `batched_ekf.py`
5. Modify `LEDHParticleFlowFilter.update` to dispatch on model flag
6. Test on linear Gaussian (verify score-based flow matches standard flow)
7. Test on 2D SV model (verify $x_2$ tracking)

---

## Appendix A: Verification — Score-Based Flow Reduces to Standard Flow

For an additive noise model $y = Hx + v$, $v \sim \mathcal{N}(0, R)$:

**Score**: $g = H^\top R^{-1}(y - Hx)$

**Fisher info**: $\Lambda_{\text{FI}} = H^\top R^{-1} H$

**Flow A**:
$$
A_{\text{score}} = -\tfrac{1}{2} P \Lambda (I + \lambda P \Lambda)^{-1}
= -\tfrac{1}{2} P H^\top R^{-1} H (I + \lambda P H^\top R^{-1} H)^{-1}
$$

Using the push-through identity $(B^\top C^{-1} B)(I + \lambda A B^\top C^{-1} B)^{-1} = B^\top(\lambda B A B^\top + C)^{-1} B$:

$$
= -\tfrac{1}{2} P H^\top (\lambda H P H^\top + R)^{-1} H = A_{\text{standard}} \quad \checkmark
$$

**Flow b** (at linearization point $\bar{x}$):

$g + \Lambda \bar{x} = H^\top R^{-1}(y - H\bar{x}) + H^\top R^{-1} H \bar{x} = H^\top R^{-1} y$

For the case where $e = h(\bar{x}) - H\bar{x} = 0$ (linear model):

$H^\top R^{-1}(z - e) = H^\top R^{-1} y = g + \Lambda \bar{x} \quad \checkmark$

The generalized formula is mathematically identical to the standard formula
for all additive noise models.

## Appendix B: Fisher Information Derivation (General Scalar Case)

For a scalar observation $y \sim \mathcal{N}(h(x), \sigma^2(x))$ with
state-dependent variance:

$$
\log p(y \mid x) = -\frac{1}{2}\log(2\pi\sigma^2(x)) - \frac{(y - h(x))^2}{2\sigma^2(x)}
$$

The score vector has components:

$$
\frac{\partial}{\partial x_i}\log p = \frac{\partial h}{\partial x_i}\frac{y - h}{\sigma^2} - \frac{1}{2\sigma^2}\frac{\partial \sigma^2}{\partial x_i} + \frac{(y-h)^2}{2\sigma^4}\frac{\partial \sigma^2}{\partial x_i}
$$

Taking the expectation of $-\nabla^2 \log p$ over $y \mid x$ (noting
$\mathbb{E}[(y-h)^2] = \sigma^2$ and $\mathbb{E}[(y-h)^4] = 3\sigma^4$):

$$
[\Lambda_{\text{FI}}]_{ij} = \frac{1}{\sigma^2}\frac{\partial h}{\partial x_i}\frac{\partial h}{\partial x_j} + \frac{1}{2\sigma^4}\frac{\partial \sigma^2}{\partial x_i}\frac{\partial \sigma^2}{\partial x_j}
$$

For the 2D SV model: $h(x) = bx_1$, $\sigma^2(x) = e^{x_2}$,
$\partial h/\partial x = (b, 0)$, $\partial \sigma^2/\partial x = (0, e^{x_2})$:

$$
\Lambda_{\text{FI}} = \frac{1}{e^{x_2}}\begin{pmatrix} b^2 \\ 0 \end{pmatrix}\begin{pmatrix} b^2 & 0\end{pmatrix}
+ \frac{1}{2 e^{2x_2}}\begin{pmatrix} 0 \\ e^{x_2}\end{pmatrix}\begin{pmatrix} 0 & e^{x_2}\end{pmatrix}
= \begin{pmatrix} b^2 e^{-x_2} & 0 \\ 0 & \frac{1}{2}\end{pmatrix}
$$

## Appendix C: Why the Negative Hessian Is Indefinite

The actual negative Hessian $\Lambda_{\text{Hess}} = -\nabla^2_x \log p(y \mid x)$
at a specific observation $y$:

$$
\Lambda_{\text{Hess}} = \begin{pmatrix}
b^2 e^{-x_2} & b(y-bx_1)e^{-x_2} \\
b(y-bx_1)e^{-x_2} & \frac{1}{2}(y-bx_1)^2 e^{-x_2}
\end{pmatrix}
$$

Its determinant:

$$
\det(\Lambda_{\text{Hess}}) = \frac{1}{2}b^2(y-bx_1)^2 e^{-2x_2} - b^2(y-bx_1)^2 e^{-2x_2} = -\frac{1}{2}b^2(y-bx_1)^2 e^{-2x_2} \leq 0
$$

Since $\text{tr}(\Lambda_{\text{Hess}}) > 0$ and $\det(\Lambda_{\text{Hess}}) < 0$, the
matrix has one positive and one negative eigenvalue — it is **indefinite**.

The Fisher information avoids this by averaging over $y$, which eliminates the
off-diagonal cross terms and replaces the observation-dependent $(y-bx_1)^2$
with its expectation $\sigma^2 = e^{x_2}$, yielding the clean diagonal form.
