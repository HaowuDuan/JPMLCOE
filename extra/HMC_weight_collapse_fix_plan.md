# Fix: HMC Weight Collapse on CUDA

## Problem

The LEDH particle filter works fine in a **single forward pass** on CUDA (ESS healthy, min 4.5/200, log-lik = -1328). But during **HMC sampling**, weights collapse. Works on MPS + TF 2.16, collapses on CUDA + TF 2.20. Weight clipping (50, 30) doesn't fix it.

## Root Cause

During HMC leapfrog integration, unconstrained parameters get pushed to extreme values. The `Exp` bijector maps these to very small or very large `sigma_V`/`sigma_W` (e.g., 0.0001 or 1e6). At these extreme values:

1. The LEDH filter produces `-inf` or `NaN` log-likelihood
2. The gradient becomes `NaN`
3. The NaN gradient **permanently corrupts** the HMC momentum vector
4. All subsequent leapfrog steps use NaN momentum → chain is broken

The code at `hmc_runner.py:79-88` has **no guard** against this:
```python
log_likelihood = self.filter_obj.log_marginal_likelihood_tf(...)
# Nothing checks if log_likelihood is finite!
return -(log_likelihood + log_prior)
```

The different behavior between MPS and CUDA is because slightly different gradient values cause different leapfrog trajectories — MPS happens to avoid the unstable regions, CUDA doesn't. It's luck, not robustness.

## Fix (2 changes)

### Change 1: Guard log-likelihood in `_negative_log_posterior`

**File:** `code/src/DF/hmc_runner.py` — `_negative_log_posterior` method (lines 79-88)

After computing `log_likelihood`, add:
```python
# Replace NaN/inf with a large but finite penalty
log_likelihood = tf.where(
    tf.math.is_finite(log_likelihood),
    log_likelihood,
    tf.constant(-1e10, dtype=log_likelihood.dtype)
)
# Floor prevents -inf gradients; tf.maximum preserves gradients above floor
log_likelihood = tf.maximum(log_likelihood, -1e10)
```

**Why this works:** When the filter collapses at extreme parameter values, `log_likelihood` is `-inf`. `tf.where` replaces it with `-1e10` (a very bad but finite value). `tf.maximum` ensures a gradient of 0 when clamped (rather than NaN), so HMC will reject the proposal without corrupting the momentum. The leapfrog step recovers and continues normally.

### Change 2: Use `slogdet` instead of `log(abs(det(...)))` for Jacobian

**File:** `code/src/filters/particle/ledh_invertible.py` — `update` method (line 252)

Replace:
```python
log_det_M = tf.math.log(tf.abs(tf.linalg.det(M_batch)))  # (N,)
```
With:
```python
sign, log_det_M = tf.linalg.slogdet(M_batch)  # (N,), (N,)
```

**Why:** `tf.linalg.det` computes the full determinant, which can overflow/underflow for matrices with large/small eigenvalues. `tf.linalg.slogdet` computes the log-determinant directly (numerically stable). The `sign` output is not needed here since we already take `abs` — `slogdet` returns the absolute log-determinant. This doesn't fix the HMC collapse directly but prevents a secondary source of `NaN` in the Jacobian accumulation.

## Verification

1. Push updated code: `make push`
2. On office, run the standalone filter test (should still work):
```bash
python -c "
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, '.')
import tensorflow as tf
import numpy as np
from src.models.kitagawa import KitagawaModel
from src.filters.particle.ledh_invertible import LEDHParticleFlowFilter
model = KitagawaModel(sigma_V=10.0, sigma_W=1.0, dtype=tf.float32)
rng = np.random.default_rng(42)
x = np.zeros(100); y = np.zeros(100)
x[0] = rng.normal(0, np.sqrt(5))
for t in range(1, 100):
    x[t] = x[t-1]/2 + 25*x[t-1]/(1+x[t-1]**2) + 8*np.cos(1.2*t) + rng.normal(0, 1)
for t in range(100):
    y[t] = x[t]**2/20 + rng.normal(0, 10)
pf = LEDHParticleFlowFilter(model, n_particles=200, n_lambda_steps=29, resampling_method='ot_entropy', resampling_config={'epsilon': 0.5}, weight_clip_range=50.0)
result = pf.filter(y.reshape(-1,1), random_seed=42)
print('ESS:', result.ess[:10])
print('Min ESS:', np.min(result.ess))
print('Log-lik:', result.log_likelihood)
"
```
3. Run HMC with small sample size:
```bash
python src/experiments/run_dpf_experiment.py dpf=kitagawa_ledh dpf.hmc.num_samples=10 dpf.hmc.num_burnin=5
```
4. Check that no HMC step produces NaN params and that acceptance rate is reasonable (>10%)
