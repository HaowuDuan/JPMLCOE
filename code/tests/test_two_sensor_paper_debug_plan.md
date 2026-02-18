# Debug Plan: Two-Sensor Bearing Paper Reproduction (arXiv:2107.04672, Section 4)

## Problem
Both linear (schedule_mu=0.0) and optimal (schedule_mu=0.2) experiments give:
- Identical MSE and tr(P) for every run
- tr(P) ≈ 2.46 instead of paper's 1535.2 (linear) and 1028.8 (optimal)

All tests go in `code/tests/test_two_sensor_paper.py`.

---

## Test 1 — Model Parameters Match Paper

**What**: Verify model constants exactly match Section 4 of the paper.

**Paper values**:
- mu_0 = [3.0, 5.0]
- Sigma_0 = diag(1000, 2)
- R = diag(0.04, 0.04)  (sigma_bearing = 0.2 rad)
- sensor_positions = [[3.5, 0.0], [-3.5, 0.0]]
- z_fixed = [0.4754, 1.1868]
- truth = [4.0, 4.0]

```python
def test_model_parameters():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    np.testing.assert_allclose(model.mu_0.numpy(), [3.0, 5.0])
    np.testing.assert_allclose(np.diag(model.Sigma_0.numpy()), [1000.0, 2.0])
    np.testing.assert_allclose(np.diag(model.R.numpy()), [0.04, 0.04])
    np.testing.assert_allclose(model.sensor_positions.numpy(),
                               [[3.5, 0.0], [-3.5, 0.0]])
```

---

## Test 2 — Observation Jacobian H at Prior Mean

**What**: Verify H = dh/dx at eta_bar_0 = [3.0, 5.0].

**Expected** (computed analytically):
For sensor 1 at (3.5, 0): dx=3-3.5=-0.5, dy=5-0=5, r^2=25.25
  h1/dx = -dy/r^2 = -5/25.25 ≈ -0.19802
  h1/dy =  dx/r^2 = -0.5/25.25 ≈ -0.01980

For sensor 2 at (-3.5, 0): dx=3-(-3.5)=6.5, dy=5, r^2=67.25
  h2/dx = -dy/r^2 = -5/67.25 ≈ -0.07435
  h2/dy =  dx/r^2 =  6.5/67.25 ≈  0.09665

```python
def test_observation_jacobian_at_prior():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    eta_bar = tf.constant([3.0, 5.0], dtype=tf.float64)
    H = model.observation_jacobian(eta_bar).numpy()
    # Row 0: sensor 1
    assert H.shape == (2, 2)
    dx1, dy1 = 3.0 - 3.5, 5.0 - 0.0  # -0.5, 5.0
    r1sq = dx1**2 + dy1**2  # 25.25
    np.testing.assert_allclose(H[0, 0], -dy1 / r1sq, rtol=1e-5)  # ≈ -0.198
    np.testing.assert_allclose(H[0, 1],  dx1 / r1sq, rtol=1e-5)  # ≈ -0.0198
    # Row 1: sensor 2
    dx2, dy2 = 3.0 - (-3.5), 5.0 - 0.0  # 6.5, 5.0
    r2sq = dx2**2 + dy2**2  # 67.25
    np.testing.assert_allclose(H[1, 0], -dy2 / r2sq, rtol=1e-5)
    np.testing.assert_allclose(H[1, 1],  dx2 / r2sq, rtol=1e-5)
```

---

## Test 3 — Flow Params A_det and b_det at Key lambda Values

**What**: Verify `compute_flow_params_global` with paper's P0, H, R at λ=0, 0.5, 1.0.

**Analytical check for A at λ=0**:
  A(0) = -0.5 * P0 @ H^T @ (0*HPH + R)^-1 @ H
       = -0.5 * P0 @ H^T @ R^-1 @ H

```python
def test_flow_params_A_at_lambda_half():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    P = tf.constant(np.diag([1000.0, 2.0]), dtype=tf.float64)
    R = model.R
    R_inv = tf.linalg.inv(R)
    eta_bar = tf.constant([3.0, 5.0], dtype=tf.float64)
    H = model.observation_jacobian(eta_bar)
    z = tf.constant([0.4754, 1.1868], dtype=tf.float64)

    lam = tf.constant(0.5, dtype=tf.float64)
    A, b = compute_flow_params_global(H, lam, z, P, R, R_inv, eta_bar, 2)

    # A must have negative eigenvalues (stable flow)
    eigvals = np.linalg.eigvals(A.numpy())
    print(f"A eigenvalues at lambda=0.5: {eigvals}")
    # RECORD these values for paper comparison

    # Check A formula: A = -0.5 * P @ H^T @ (lambda*HPH + R)^-1 @ H
    HPH = H.numpy() @ P.numpy() @ H.numpy().T
    S = 0.5 * HPH + np.diag([0.04, 0.04])
    A_expected = -0.5 * P.numpy() @ H.numpy().T @ np.linalg.inv(S) @ H.numpy()
    np.testing.assert_allclose(A.numpy(), A_expected, rtol=1e-5)
```

---

## Test 4 — Score Correction at Lambda=0.5

**What**: Verify A_stoch = A_det - (Q/2)*Sigma_inv and its eigenvalues.

This determines whether the stochastic flow is stable, unstable, or stiff.

```python
def test_score_correction_eigenvalues():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    P = tf.constant(np.diag([1000.0, 2.0]), dtype=tf.float64)
    R = model.R
    R_inv = tf.linalg.inv(R)
    eta_bar = tf.constant([3.0, 5.0], dtype=tf.float64)
    H = model.observation_jacobian(eta_bar)
    z = tf.constant([0.4754, 1.1868], dtype=tf.float64)
    Q_diag = np.array([4.0, 0.4])

    for lam_val in [0.01, 0.1, 0.5, 1.0]:
        lam = tf.constant(lam_val, dtype=tf.float64)
        A_det, _ = compute_flow_params_global(H, lam, z, P, R, R_inv, eta_bar, 2)

        # Score correction: Sigma_inv = P_inv + lambda * H^T R^-1 H
        P_inv = tf.linalg.inv(P)
        Sigma_inv = P_inv + lam * tf.transpose(H) @ R_inv @ H

        # A_stoch = A_det - (Q/2) * Sigma_inv  (row-scaled)
        Q_half = tf.constant(Q_diag / 2, dtype=tf.float64)
        A_stoch = A_det - tf.expand_dims(Q_half, 1) * Sigma_inv

        eigvals = np.linalg.eigvals(A_stoch.numpy())
        print(f"lambda={lam_val}: A_stoch eigenvalues = {eigvals}")
        # KEY: are eigenvalues positive (blow-up) or negative (collapse)?
```

---

## Test 5 — Lambda Schedule: Uniform vs Exponential

**What**: Verify uniform schedule gives exactly dλ=0.01 for n_lambda_steps=100.

```python
def test_uniform_lambda_schedule():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=10, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], lambda_schedule='uniform')

    steps = filt.lambda_steps.numpy()
    np.testing.assert_allclose(steps, np.ones(100) / 100.0, rtol=1e-10)
    np.testing.assert_allclose(steps.sum(), 1.0, rtol=1e-10)
```

---

## Test 6 — BVP Solver: Optimal Schedule Differs from Linear

**What**: Verify that with mu=0.2 the BVP produces beta*(lambda) != lambda.

```python
def test_bvp_produces_different_schedule():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt_opt = StochasticEDHFlowPaper(
        model, n_particles=10, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.2, lambda_schedule='uniform')

    # Initialize to set eta_bar_0 and predicted_cov
    rng = np.random.default_rng(0)
    filt_opt.initialize(rng)
    filt_opt.predict()

    P = filt_opt.predicted_cov
    R_inv = tf.linalg.inv(model.R)
    beta_vals, dbeta_vals = filt_opt._compute_optimal_schedule(P, R_inv)

    beta_np = beta_vals.numpy()
    lambda_np = np.cumsum(np.ones(100) / 100.0)

    # Must be different from linear
    max_diff = np.max(np.abs(beta_np - lambda_np))
    print(f"Max |beta*(lambda) - lambda| = {max_diff:.4f}")
    assert max_diff > 0.01, "BVP returned linear schedule (may have bracket failure)"
```

---

## Test 7 — StochasticEDHFlowPaper Uses Exact Prior for EKF

**What**: Verify initialize() sets EKF mean/cov to model.mu_0/Sigma_0, NOT particle empirical.

```python
def test_paper_filter_uses_exact_prior():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], lambda_schedule='uniform')

    rng = np.random.default_rng(42)
    filt.initialize(rng)

    # EKF must be at exact prior, not particle empirical
    np.testing.assert_allclose(filt.global_filter.mean.numpy(),
                               [3.0, 5.0], rtol=1e-6)
    np.testing.assert_allclose(np.diag(filt.global_filter.cov.numpy()),
                               [1000.0, 2.0], rtol=1e-6)
    # eta_bar_0 and predicted_cov must also be exact prior
    np.testing.assert_allclose(filt.eta_bar_0.numpy(), [3.0, 5.0], rtol=1e-6)
    np.testing.assert_allclose(np.diag(filt.predicted_cov.numpy()),
                               [1000.0, 2.0], rtol=1e-6)
```

---

## Test 8 — predict() Does NOT Feed Particle Mean to EKF

**What**: After predict(), eta_bar_0 must come from EKF own prediction, not particle empirical mean.

For static model (identity transition), EKF predicted mean = mu_0 = [3.0, 5.0].
Particle empirical mean ≈ [3±large, 5±small] (50 draws from N([3,5], diag(1000,2))).

```python
def test_predict_no_particle_feedback():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    filt.initialize(rng)

    particle_mean_before = filt.particles.value().numpy().mean(axis=0)
    filt.predict()
    eta_bar_after = filt.eta_bar_0.numpy()

    print(f"Particle empirical mean: {particle_mean_before}")
    print(f"eta_bar_0 after predict: {eta_bar_after}")

    # eta_bar_0 must be EKF predicted mean (≈ [3,5] for identity transition)
    # NOT the particle empirical mean (which differs significantly)
    np.testing.assert_allclose(eta_bar_after, [3.0, 5.0], atol=0.1)

    # Particle empirical mean should differ significantly from [3,5]
    # (50 draws from broad prior); this confirms it's NOT being used
    diff = np.abs(particle_mean_before - np.array([3.0, 5.0]))
    assert diff[0] > 0.5 or diff[1] > 0.1, \
        "Particle mean accidentally equals prior mean — weaken test"
```

---

## Test 9 — Particle Trajectory: Is tr(P) Growing or Collapsing?

**What**: Track particle sample covariance DURING the flow (every 10 steps) for a single run.

This tells us whether the score correction is causing collapse or the expected behavior.

```python
def test_particle_spread_during_flow():
    """Instrument the flow to print tr(P) at each step."""
    import types
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.0, lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    filt.initialize(rng)
    filt.predict()

    # Manually run the flow loop with instrumentation
    z = tf.constant([0.4754, 1.1868], dtype=tf.float64)
    P = filt.predicted_cov
    R = model.R
    R_inv = tf.linalg.inv(R)
    eta_bar_0 = filt.eta_bar_0
    H_fixed = model.observation_jacobian(eta_bar_0)
    Q_tf = tf.constant([4.0, 0.4], dtype=tf.float64)
    P_inv = tf.linalg.inv(P)
    H_T_R_inv_H = tf.transpose(H_fixed) @ R_inv @ H_fixed
    P_inv_eta = tf.linalg.matvec(P_inv, eta_bar_0)
    H_T_R_inv_y = tf.linalg.matvec(tf.transpose(H_fixed) @ R_inv, z)

    particles = filt.particles.value()
    lambda_val = tf.constant(0.0, dtype=tf.float64)

    print("\nParticle spread during flow:")
    print(f"  step   0: tr(P) = {np.trace(np.cov(particles.numpy().T)):.4f}")

    for i in range(100):
        d_lambda = filt.lambda_steps[i]
        lambda_val = lambda_val + d_lambda
        homotopy_param = lambda_val  # linear schedule

        A, b = compute_flow_params_global(
            H_fixed, homotopy_param, z, P, R, R_inv, eta_bar_0, 2)

        correction_A = P_inv + homotopy_param * H_T_R_inv_H
        correction_b = P_inv_eta + homotopy_param * H_T_R_inv_y
        A = A - tf.expand_dims(Q_tf, 1) / 2 * correction_A
        b = b + (Q_tf / 2) * correction_b

        drift = particles @ tf.transpose(A) + b
        particles = particles + d_lambda * drift

        seed = tf.constant([0, i], dtype=tf.int32)
        noise = tf.random.stateless_normal(tf.shape(particles), seed=seed, dtype=tf.float64)
        particles = particles + noise * tf.sqrt(Q_tf * d_lambda)

        if (i + 1) % 10 == 0:
            cov = np.cov(particles.numpy().T)
            print(f"  step {i+1:3d}: tr(P) = {np.trace(cov):.4f}, "
                  f"mean = {particles.numpy().mean(axis=0)}")
```

---

## Test 10 — tr(P) Matches Paper Definition

**What**: Verify `result.covs[0]` is the particle sample covariance (not EKF cov).
Confirm it uses `np.cov` denominator (N-1) vs `N`. Identify what the paper uses.

```python
def test_trP_is_particle_sample_cov():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    filt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.0, lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    z_fixed = np.array([[0.4754, 1.1868]])
    result = filt.filter(z_fixed, random_state=rng)

    # What _estimate_mean_cov computes (flow_base.py line 72): divides by N, not N-1
    particles_final = filt.particles.value().numpy()
    mean = particles_final.mean(axis=0)
    diff = particles_final - mean
    cov_biased = (diff.T @ diff) / 50  # divides by N

    cov_unbiased = np.cov(particles_final.T)  # divides by N-1

    print(f"result.covs[0] (from filter):\n{result.covs[0]}")
    print(f"tr(P) reported: {np.trace(result.covs[0]):.4f}")
    print(f"tr(biased cov, /N): {np.trace(cov_biased):.4f}")
    print(f"tr(unbiased cov, /(N-1)): {np.trace(cov_unbiased):.4f}")

    # Must match the biased version (flow_base uses /N)
    np.testing.assert_allclose(result.covs[0], cov_biased, rtol=1e-5)
```

---

## Test 11 — Linear vs Optimal Give DIFFERENT Results

**What**: With seed=0, linear (mu=0.0) and optimal (mu=0.2) must produce different tr(P).

```python
def test_linear_vs_optimal_differ():
    model = TwoSensorBearingOnlyModel(dtype=tf.float64)
    z_fixed = np.array([[0.4754, 1.1868]])

    filt_lin = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.0, lambda_schedule='uniform')
    filt_opt = StochasticEDHFlowPaper(
        model, n_particles=50, n_lambda_steps=100,
        diffusion_scale=[4.0, 0.4], schedule_mu=0.2, lambda_schedule='uniform')

    rng = np.random.default_rng(0)
    res_lin = filt_lin.filter(z_fixed, random_state=rng)

    rng = np.random.default_rng(0)
    res_opt = filt_opt.filter(z_fixed, random_state=rng)

    trP_lin = float(np.trace(res_lin.covs[0]))
    trP_opt = float(np.trace(res_opt.covs[0]))
    print(f"Linear tr(P) = {trP_lin:.2f}, Optimal tr(P) = {trP_opt:.2f}")

    # They must differ (otherwise BVP is falling back to linear)
    assert abs(trP_lin - trP_opt) > 1.0, \
        f"Linear and optimal give same tr(P) = {trP_lin:.2f}"
```

---

## How to Run

```bash
cd /Users/haowuduan/Documents/githubrepos/JPMLCOE/code
python -m pytest tests/test_two_sensor_paper.py -v -s
```

The `-s` flag is critical to see the printed diagnostic values.

## Expected Failure Points (to investigate)

Based on current symptom (both cases give identical tr(P) ≈ 2.46):

- **Test 6** (BVP): likely fails — bracket failure causes fallback to linear
- **Test 9** (particle spread): will reveal if score correction collapses particles to posterior
  - If tr(P) decreases monotonically → score correction is over-damping
  - If tr(P) increases then explodes → true stiffness behaviour (paper's result)
- **Test 11** (linear vs optimal differ): will fail because BVP is returning linear schedule
