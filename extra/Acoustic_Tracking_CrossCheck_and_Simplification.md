# Acoustic Tracking Model: Cross-Check & Simplification Guide

**Paper**: Li & Coates (2017), "Particle Filtering with Invertible Particle Flow" (arXiv:1607.08799v5), Section V-A.

**Reference MATLAB code**: `PFPF/Acoustic_Example/` (author's official implementation)

**Code files reviewed**:
- `code/src/models/acoustic_tracking_full.py` — multi-target (4 targets), pure TF
- `code/src/models/acoustic_tracking.py` — single-target, numpy + TF hybrid
- `PFPF/Acoustic_Example/Acoustic_hfunc.m` — MATLAB observation function
- `PFPF/Acoustic_Example/Acoustic_dh_dxfunc.m` — MATLAB Jacobian
- `PFPF/Acoustic_Example/Acoustic_example_initialization.m` — MATLAB setup
- `PFPF/Acoustic_Example/AcousticGaussInit.m` — MATLAB initialization

---

## 1. MATLAB Reference (Ground Truth)

From `Acoustic_example_initialization.m` and `Acoustic_hfunc.m`:

| Parameter | MATLAB Value | Source |
|-----------|-------------|--------|
| Targets | `nTarget = 4` | Line 18 |
| State per target | `dimState_per_target = 4` — [x, y, vx, vy] | Line 19 |
| Sensors | 25 (5x5 grid, 10m spacing, loaded from `sensorsXY.mat`) | Verified: matches 5x5 grid |
| F (per target) | `[1,0,1,0; 0,1,0,1; 0,0,1,0; 0,0,0,1]` | Line 61 |
| V_true (data gen) | `gammavar_real * Gamma` = `0.05 * Gamma` = `(1/20) * Gamma` | Lines 63, 66, 68 |
| V_filter (algorithms) | `[3,0,0.1,0; 0,3,0,0.1; 0.1,0,0.03,0; 0,0.1,0,0.03]` | Line 70 |
| **Observation** | **`h = Amp / (r^invPow + d0)` with `invPow = 1`** | Lines 43, 20-21 |
| **= Amp / (r + d0)** | **where `r = sqrt(dx^2 + dy^2)` (DISTANCE, not squared)** | hfunc lines 41-43 |
| Amp (Psi) | 10 | Line 42 |
| d_0 | 0.1 | Line 44 |
| sigma_w^2 | 0.01 | Line 112 |
| Initial states | `[12,6,0.001,0.001, 32,32,-0.001,-0.005, 20,13,-0.1,0.01, 15,35,0.002,0.002]` | Line 50 |
| Initial sigma | `10*[1;1;0.1;0.1]` = `[10,10,1,1]` per target | Line 96 |
| N_particles | 500 | Line 14 |
| T | 40 | Line 17 |
| Q_correction | `0.1*diag([1,1,0.01,0.01])` — added to filter Q | Line 77 |

### 1.1 MATLAB Observation Model (Definitive)

From `Acoustic_hfunc.m` lines 36-46:
```matlab
v = bsxfun(@minus, sensorsPos, permute(x,[1 3 2]));
v = v.^2;
v = v(1:nTarget,:,:) + v(nTarget+1:2*nTarget,:,:);  % dx^2 + dy^2
v = sqrt(v);                                          % r = sqrt(dx^2 + dy^2)
v = likeparams.Amp ./ (v.^likeparams.invPow + likeparams.d0);  % Amp / (r^1 + d0)
v = sum(v,1);                                         % sum over targets
```

With `invPow = 1`: **h = Amp / (r + d_0)** where r = Euclidean distance.

### 1.2 MATLAB Jacobian (Definitive)

From `Acoustic_dh_dxfunc.m` lines 13-21, comments:
```
%  h = Amp/(r+d0)
%  dh/dx1 = dh/dr * dr/dx1
%         = -Amp/(r+d0)^2 * 2(x1-s1) * 1/(2r)
%         = -Amp*(x1-s1) / [r*(r+d0)^2]
```

Code (line 46):
```matlab
v = -likeparams.Amp ./ (((v+likeparams.d0).^2) .* v);  % -Amp / ((r+d0)^2 * r)
v = v .* mv;                                             % multiply by (x-s)
```

So: **dh/dx = -Amp * (x - s_x) / (r * (r + d_0)^2)**

### 1.3 MATLAB Initial Distribution

From `AcousticGaussInit.m`:
```matlab
m0 = x0 + sigma0 .* randn(dim, 1);          % Randomly perturb mean
% reject if outside [0,0,40,40]
P0 = diag(sigma0.^2);                        % Filter covariance
xp = m0*ones(1,nParticle) + sigma0.*randn(dim,nParticle);  % Particles ~ N(m0, P0)
```

Key: the filter's initial **mean is randomly perturbed** from truth, then particles are drawn from Gaussian centered at this perturbed mean. Each trial gets a different initial mean.

---

## 2. Cross-Check: `acoustic_tracking_full.py` (Multi-Target)

### 2.1 Correct

| Item | MATLAB | Python | Status |
|------|--------|--------|--------|
| n_targets | 4 | `n_targets=4` | OK |
| state_dim | 16 | `4 * n_targets` = 16 | OK |
| Sensors | 25, 5x5 grid (from .mat) | `_build_sensor_grid()`, spacing=10m | OK (verified identical) |
| F matrix | Block-diagonal `[1,0,1,0;...]` | `_block_diag([F_single]*4)` | OK |
| V_true | `0.05 * Gamma` | `(1/20) * [[1/3,...]]` | OK |
| V_filter | `[3,0,0.1,0;...]` | Lines 96-101 | OK |
| V selection | separate structs | `use_true_process_noise` flag | OK |
| Amp/Psi, d_0 | 10, 0.1 | 10.0, 0.1 | OK |
| sigma_w | 0.1 | `measurement_noise_std=0.1` | OK |
| Initial states | `x0 = [12,6,...]` | Lines 119-124 | OK |
| Initial cov | `diag([10,10,1,1].^2)` per target | `diag([100,100,1,1]*4)` | OK |
| R matrix | `0.01 * eye(25)` | `np.eye(25) * 0.01` | OK |

### 2.2 Issues Found

#### Issue #1 (CRITICAL BUG): Observation model uses WRONG formula

**MATLAB** (`Acoustic_hfunc.m`):
```
h = Amp / (r + d_0)        where r = sqrt(dx^2 + dy^2)   (DISTANCE)
```

**Python** (`acoustic_tracking_full.py:236`):
```python
r_squared = dx**2 + dy**2
amp_sum += self.psi / (r_squared + self.d0)    # Psi / (r^2 + d0)  (SQUARED DISTANCE)
```

**These are completely different functions.** For a target at distance r = 20m:
- MATLAB: h = 10 / (20 + 0.1) = 0.497
- Python: h = 10 / (400 + 0.1) = 0.025 (20x smaller!)

This affects every downstream computation: observations, likelihoods, Jacobians, flow parameters.

**Affected files**: Both `acoustic_tracking_full.py` and `acoustic_tracking.py` use r^2 + d_0.

#### Issue #2 (CRITICAL BUG): Jacobian uses WRONG formula

Direct consequence of Issue #1.

**MATLAB** (`Acoustic_dh_dxfunc.m`):
```
dh/dx = -Amp * (x - s_x) / (r * (r + d_0)^2)
```

**Python** (`acoustic_tracking_full.py:323`):
```python
dh_dx = -self.psi * 2.0 * dx / denominator    # -2*Psi*dx / (r^2 + d0)^2
```

For h = Psi/(r + d0): dh/dx = -Psi * dx / (r * (r + d0)^2)
For h = Psi/(r^2 + d0): dh/dx = -2*Psi * dx / (r^2 + d0)^2

The Python code is self-consistent (its Jacobian matches its observation model), but both are wrong relative to the paper/MATLAB.

**Impact**: The Jacobian H drives all flow filter computations (A(lambda), b(lambda) via Eqs. 10-11). Wrong H means wrong flow parameters, wrong particle migration, wrong posterior.

#### Issue #3 (BUG): `observation_jacobian` uses `tf.constant` on tensor values

**File**: `acoustic_tracking_full.py`, line 333

```python
return tf.constant(H_list, dtype=tf.float32)  # H_list contains TF tensors
```

`tf.constant()` cannot accept symbolic tensors during `@tf.function` tracing. Will either fail at graph compilation or silently freeze values from the first call.

**Fix**: Use `tf.stack` instead.

#### Issue #4 (BUG): `n_targets` parameter is not generalizable

Lines 119-124 hardcode 4 targets worth of initial states:
```python
paper_initial_states = np.array([
    12.0, 6.0, 0.001, 0.001,      # Target 1
    32.0, 32.0, -0.001, -0.005,   # Target 2
    ...
])
```

Line 129 hardcodes `4` instead of `n_targets`:
```python
Sigma_0_np = np.diag(np.tile(single_target_cov, 4))  # Should be n_targets
```

Setting `n_targets=1` would create state_dim=4 but mu_0 would be 16D → shape mismatch crash.

#### Issue #5 (MINOR): Initial distribution differs from MATLAB

**MATLAB**: Randomly perturbs initial mean, rejects if outside area, then samples particles from Gaussian centered at perturbed mean.

**Python**: Samples particles directly from N(true_state, Sigma_0) — no mean perturbation.

#### Issue #6 (MINOR): Missing Q_correction

**MATLAB** has `Q_correction = 0.1*diag([1,1,0.01,0.01])` which appears to be a regularization term. Not present in Python code.

---

## 3. Cross-Check: `acoustic_tracking.py` (Single-Target)

### 3.1 Issues (in addition to the observation model bug shared with full model)

| # | Severity | Description |
|---|----------|-------------|
| 7 | **CRITICAL** | Same wrong observation formula: uses r^2 + d_0 instead of r + d_0 |
| 8 | **CRITICAL** | Same wrong Jacobian formula (consistent with wrong obs model) |
| 9 | MEDIUM | Default 4 sensors at corners, not 25-sensor grid |
| 10 | MEDIUM | No V_true/V_filter distinction — always uses V_true |
| 11 | MINOR | Uniform initial position, MATLAB uses Gaussian |
| 12 | LOW | No TF-native observation_jacobian |

---

## 4. Summary of All Issues

| # | Severity | File(s) | Description |
|---|----------|---------|-------------|
| 1 | **CRITICAL** | Both models | **Observation model uses r^2 + d_0 instead of r + d_0** (MATLAB `invPow=1`, so h = Amp/(r + d0)) |
| 2 | **CRITICAL** | Both models | **Jacobian formula wrong** (consequence of #1) |
| 3 | **HIGH** | `acoustic_tracking_full.py:333` | `observation_jacobian` uses `tf.constant()` on tensors |
| 4 | **MEDIUM** | `acoustic_tracking_full.py` | `n_targets` parameter hardcoded to 4 in init (mu_0, Sigma_0) |
| 5 | MINOR | `acoustic_tracking_full.py` | Initial mean not randomly perturbed |
| 6 | MINOR | `acoustic_tracking_full.py` | Missing Q_correction regularization |
| 7 | MEDIUM | `acoustic_tracking.py` | Default 4 sensors, not 25 |
| 8 | MEDIUM | `acoustic_tracking.py` | No V_true/V_filter distinction |
| 9 | MINOR | `acoustic_tracking.py` | Uniform initial, should be Gaussian |

---

## 5. Fixing the Observation Model

### 5.1 Correct observation function

```python
# WRONG (current):
r_squared = dx**2 + dy**2
amplitude = self.psi / (r_squared + self.d0)          # Psi / (r^2 + d0)

# CORRECT (MATLAB):
r_squared = dx**2 + dy**2
r = sqrt(r_squared)                                     # distance
amplitude = self.psi / (r + self.d0)                    # Psi / (r + d0)
```

### 5.2 Correct Jacobian

For h = Psi / (r + d_0) where r = sqrt((x-sx)^2 + (y-sy)^2):

```
dh/dr = -Psi / (r + d_0)^2
dr/dx = (x - sx) / r
dh/dx = -Psi * (x - sx) / (r * (r + d_0)^2)
dh/dy = -Psi * (y - sy) / (r * (r + d_0)^2)
dh/dvx = 0
dh/dvy = 0
```

```python
# WRONG (current):
denominator_squared = (r_squared + d0) ** 2
H[j, 0] = -2.0 * psi * dx / denominator_squared

# CORRECT (MATLAB):
r = tf.sqrt(r_squared)
H[j, 0] = -psi * dx / (r * (r + d0)**2)
```

**Edge case**: When r = 0 (target exactly at sensor), the Jacobian has a singularity from the 1/r term. The MATLAB code doesn't handle this explicitly (d_0 prevents the observation from blowing up, but the Jacobian's 1/r still diverges). Add a small epsilon: `r = tf.sqrt(r_squared + 1e-10)`.

---

## 6. Reduced-Complexity Model Proposal

### 6.1 Motivation

The full model (4 targets, 16D state, 25 sensors) is expensive to debug. Before fixing the observation model bug and testing all filters, validate with a simpler version.

**Proposed**: 1 target, 4D state, 25 sensors. Preserves the nonlinear observation while making results directly interpretable.

### 6.2 What Changes

| Aspect | Full Model (Paper) | Reduced Model |
|--------|-------------------|---------------|
| Targets | 4 | 1 |
| State dim | 16 | 4 |
| Sensors | 25 (5x5 grid) | 25 (5x5 grid, same) |
| Obs dim | 25 | 25 |
| F | 16x16 block-diagonal | 4x4 single block |
| Q (filter) | 16x16 block-diagonal V_filter | 4x4 V_filter |
| H (Jacobian) | 25x16 | 25x4 |
| Observation | z_s = Sum_c Psi/(r_{s,c} + d0) | z_s = Psi/(r_s + d0) |
| A(lambda) | 16x16 | 4x4 |

### 6.3 Why This is a Good Intermediate Test

1. **Still nonlinear**: h = Psi/(r + d_0) with the 1/r Jacobian singularity. EKF/UKF will still struggle near sensors.

2. **Over-determined**: 25 observations for 4 states — highly informative measurements, same as the paper.

3. **Debuggable**: Plot 2D trajectory directly. No OMAT metric or permutation matching needed.

4. **Fast**: A(lambda) is 4x4 vs 16x16. ~64x speedup for state-space matrix operations.

5. **Isolates bugs**: No multi-target summation ambiguity.

### 6.4 Implementation Plan

**Step 1**: Fix observation model (CRITICAL — both files)
- Change `Psi / (r^2 + d0)` to `Psi / (r + d0)` where `r = sqrt(dx^2 + dy^2)`
- Fix Jacobian accordingly: `-Psi * dx / (r * (r + d0)^2)`

**Step 2**: Make `acoustic_tracking_full.py` generalizable to `n_targets=1`
- Parameterize `mu_0` and `Sigma_0` by `n_targets` instead of hardcoding 4
- For n_targets=1, use a single initial state (e.g., `[20, 20, 0, 0]`)

**Step 3**: Fix `observation_jacobian` tf.constant bug
- Replace `tf.constant(H_list)` with `tf.stack`

**Step 4**: Add V_true/V_filter to `acoustic_tracking.py`
- Add `use_true_process_noise` flag

**Step 5**: Create configs for reduced model
- Use N_lambda=29, N_particles=500 (paper values)

### 6.5 Gradual Complexity Ramp

| Level | Targets | State dim | Sensors | Key test |
|-------|---------|-----------|---------|----------|
| 1 (proposed) | 1 | 4 | 25 | Basic nonlinear tracking, filter correctness |
| 2 | 1 | 4 | 4 | Sparse sensors, harder geometry |
| 3 | 2 | 8 | 25 | Multi-target, moderate dimension |
| 4 (paper) | 4 | 16 | 25 | Full paper scenario |

---

## 7. Lambda Schedule Cross-Check

**MATLAB** (`generateExponentialLambda.m`): N_lambda = 29, ratio q = 1.2, epsilon_1 = (1-q)/(1-q^N) ≈ 0.001.

**Python** (`edh_flow.py`): Same formula, but defaults to N_lambda = 100 (not 29). The invertible variants (`edh_invertible.py`, `ledh_invertible.py`) default to 29.

With N=100 and q=1.2: epsilon_1 ≈ 2.6e-9 (extremely tiny). The paper's 29 steps give epsilon_1 ≈ 0.001. Recommend using 29 for acoustic tracking experiments.

---

## References

- Li, Y. & Coates, M. (2017). "Particle Filtering with Invertible Particle Flow." IEEE TSP. arXiv:1607.08799v5.
- PFPF MATLAB code: `PFPF/Acoustic_Example/` (author's reference implementation)
- Hlinka et al. (2011). "Distributed Gaussian particle filtering using likelihood consensus." ICASSP. (Original acoustic tracking setup)
