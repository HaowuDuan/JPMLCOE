# OT Resampling: Remaining Differences vs FilterFlow Reference

After fixing the critical bug (log_weights applied per-row instead of per-column in
`compute_transport_matrix_from_potentials`), four secondary differences remain between
`code/src/resampling/ot_entropy.py` and the filterflow reference at
`filterflow/filterflow/resampling/differentiable/regularized_transport/`.

These are listed in order of likely impact.

---

## 1. Serial Extrapolation (Medium)

The extrapolation step (gradient stitching via the implicit function theorem) computes
one final Sinkhorn step from stop-gradiented converged potentials. The gradient of the
transport matrix w.r.t. `logw` flows *only* through this step.

### Current code (`ot_entropy.py`, lines 207-212)

```python
alpha_stop = tf.stop_gradient(alpha_final)
beta_stop  = tf.stop_gradient(beta_final)

alpha_extra = softmin(eps, C^T, log_weights + beta_stop / eps)
beta_extra  = softmin(eps, C,   uniform    + alpha_extra / eps)   # <-- uses NEW alpha_extra
```

`beta_extra` depends on the freshly computed `alpha_extra`, which itself depends on
`log_weights`. This creates a gradient path: `logw -> alpha_extra -> beta_extra -> T`.
That path does not exist in filterflow.

### FilterFlow (`sinkhorn.py`, line 46)

```python
a_y, b_x, _ = apply_one(tf.stop_gradient(converged_a_y),
                         tf.stop_gradient(converged_b_x))
```

Inside `apply_one`, both updates use the *stopped* inputs:

```python
at_y = softmin(eps, C_yx, log_alpha + stop_grad(b_x) / eps)   # b_x is stopped
bt_x = softmin(eps, C_xy, log_beta  + stop_grad(a_y) / eps)   # a_y is stopped (NOT at_y)
```

So `bt_x` has **zero** gradient w.r.t. `logw` — it only depends on `uniform` (constant)
and stopped values. The two potentials are computed *independently* from the stopped
converged values.

### Impact

The spurious `logw -> alpha -> beta` path in the current code introduces an extra
gradient term that does not correspond to the implicit function theorem derivation.
Whether this helps or hurts depends on the problem, but it deviates from the published
method.

### Fix

Replace lines 207-212 of `sinkhorn_iteration` with parallel extrapolation:

```python
alpha_stop = tf.stop_gradient(alpha_final)
beta_stop  = tf.stop_gradient(beta_final)

alpha_extra = softmin(eps, C^T, log_weights        + beta_stop  / eps)
beta_extra  = softmin(eps, C,   uniform_log_weights + alpha_stop / eps)  # use alpha_STOP
```

---

## 2. No Damping in Extrapolation (Low)

### Current code

The extrapolation returns the raw softmin outputs:

```python
return alpha_extra, beta_extra, n_iter
```

### FilterFlow

The extrapolation goes through `apply_one`, which applies the same 0.5 damping:

```python
a_y_new = 0.5 * (stop_grad(a_y) + at_y)
b_x_new = 0.5 * (stop_grad(b_x) + bt_x)
```

Since the stopped term has zero gradient, the effective gradient is scaled by 0.5:

```
d(a_y_new)/d(logw) = 0.5 * d(at_y)/d(logw)
```

### Impact

The current code produces gradients through the potentials that are ~2x larger than
filterflow. This changes the relative contribution of the potential-path gradient vs the
direct `logw` term in `transport_from_potentials`. With Adam this may not matter much
(adaptive learning rate), but it changes the gradient balance.

### Fix

Apply the same damping to the extrapolation output:

```python
alpha_extra = 0.5 * (alpha_stop + softmin(eps, C^T, log_weights + beta_stop / eps))
beta_extra  = 0.5 * (beta_stop  + softmin(eps, C,   uniform     + alpha_stop / eps))
```

---

## 3. No `stop_gradient` on Cost Matrix Second Argument (Low)

### Current code

```python
cost_matrix = compute_cost_matrix(scaled, scaled)
# Both arguments are the same live tensor — gradient flows through both
```

### FilterFlow (`sinkhorn.py`, lines 127-130)

```python
cost_xy = cost(x, tf.stop_gradient(y))
cost_yx = cost(y, tf.stop_gradient(x))
```

Even though `x == y == scaled_x`, the stop_gradient on the second argument means the
gradient of each cost matrix flows only through the *first* argument. This is a deliberate
choice from the geomloss library (Feydy et al.).

### Impact

For a symmetric cost `C(x,x)`, the gradient `dC[i,j]/dx_k` differs:

- **With stop_gradient**: `dC[i,j]/dx_k = (x_i - x_j) * delta(i,k)` (gradient only
  through first argument position `i`)
- **Without stop_gradient**: `dC[i,j]/dx_k = (x_i - x_j) * (delta(i,k) - delta(j,k))`
  (gradient through both `i` and `j`)

The second form doubles the gradient magnitude and creates antisymmetric contributions.
This only affects the gradient of the extrapolated potentials w.r.t. `particles` (not
`logw`), so it primarily changes how particle positions are adjusted, not the weight
gradient.

### Fix

In `sinkhorn_iteration`, use separate cost matrices:

```python
cost_xy = compute_cost_matrix(particles, tf.stop_gradient(particles))
cost_yx = compute_cost_matrix(particles, tf.stop_gradient(particles))  # same since symmetric
# Use cost_xy for beta updates and cost_yx (= transposed cost_xy) for alpha updates
```

Or equivalently, just stop-gradient the second argument:

```python
cost_matrix = compute_cost_matrix(scaled, tf.stop_gradient(scaled))
```

Since the cost is symmetric, `C^T == C`, so one matrix suffices — just ensure the
second argument has no gradient.

---

## 4. No Gradient Clipping (Low)

### Current code

```python
def gradient(dT):
    if clip_gradients:  # default: False
        dT = tf.clip_by_value(dT, -1.0, 1.0)
    dparticles, dlog_weights = tf.gradients(T, [particles, log_weights], dT)
    return dparticles, dlog_weights, None, None, None
```

Clipping is off by default.

### FilterFlow (`plan.py`, lines 77-78)

```python
def grad(d_transport):
    d_transport = tf.clip_by_value(d_transport, -1., 1.)
    dx, dlogw = tf.gradients(transport_matrix, [x, logw], d_transport)
```

Clipping is always on.

### Impact

The incoming gradient `dT` can have large entries when the loss is sensitive to specific
transport matrix elements. Without clipping, these large entries flow unmodified into
`tf.gradients`, potentially producing large and noisy parameter gradients.

For Adam-based MAP optimization, the adaptive learning rate partially compensates, but
extreme gradient entries can still cause instability in the gradient estimates — which
may contribute to the observed gradient norm oscillation (0.5 to 48).

For HMC, clipping destroys the Hamiltonian's symplectic structure and should NOT be used.
The current default of `clip_gradients=False` is correct for HMC.

### Recommendation

Enable clipping by default for MAP, keep it off for HMC:

```python
# In the MAP config (bpf_ot.yaml or similar):
resampling:
  method: ot_entropy
  clip_gradients: true   # for MAP/Adam
```

---

## Summary Table

| # | Issue | Severity | Affects | Fix complexity |
|---|-------|----------|---------|----------------|
| 1 | Serial extrapolation | Medium | Gradient direction (spurious path through beta) | 1 line change |
| 2 | No extrapolation damping | Low | Gradient magnitude (~2x) | 2 line change |
| 3 | No cost stop_gradient | Low | Particle position gradient | 1 line change |
| 4 | No gradient clipping | Low | Gradient stability for MAP | Config change |

All four are independent and can be applied separately. Issue 1 is the most likely to
affect optimization quality after the critical axis bug is fixed.
