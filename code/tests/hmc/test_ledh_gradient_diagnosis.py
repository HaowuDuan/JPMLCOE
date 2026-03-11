"""
LEDH Gradient Diagnosis Tests.

Systematic isolation of the LEDH gradient bug on 1D linear Gaussian.
BPF+HMC works; LEDH+HMC collapses. These tests pinpoint exactly why.

See code/LEDH_gradient_diagnosis.md for the full plan and decision tree.

Run:
  pytest code/tests/hmc/test_ledh_gradient_diagnosis.py -v -s 2>&1 | tee ledh_gradient_diagnosis.txt
"""

import numpy as np
import tensorflow as tf
import pytest

from conftest_hmc_diag import (
    DTYPE, N_PARTICLES, N_LAMBDA_STEPS, SEED, T,
    TRUE_OBS_NOISE_STD, INIT_OBS_NOISE_STD, SIGMA_GRID,
    lg_model_and_data, observations_tf,
    _fresh_model, _make_bpf_kwargs, _make_ledh_kwargs,
    kf_log_likelihood, pf_log_likelihood,
    compute_gradient, finite_difference_gradient,
    compute_likelihood_gradient, _make_fresh_runner,
    _make_param_specs,
)
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.parameter_handler import ParameterHandler
from src.utils.flow_params import compute_flow_params_batch
from src.utils.linalg import safe_log_abs_det, safe_inv
from src.utils.distributions import compute_flow_weights


# Fine grid for likelihood surface tests
FINE_GRID = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

# Points for gradient checks
GRAD_CHECK_POINTS = [0.5, 1.0, 1.5, 2.0]

# Resampling configs to test
RESAMPLE_CONFIGS = [
    ('systematic', True),
    ('soft', False),
    ('ot_entropy', False),
]


# ============================================================================
# Test A: Likelihood Surface Bias
# ============================================================================

class TestLikelihoodSurfaceBias:
    """Is the LEDH log-likelihood surface itself biased?"""

    @pytest.mark.parametrize("method,stop_grad", RESAMPLE_CONFIGS,
                             ids=['systematic', 'soft', 'ot_entropy'])
    def test_ledh_likelihood_peak(self, lg_model_and_data, observations_tf,
                                  method, stop_grad):
        """Test A: LEDH likelihood peaks at the same sigma as KF."""
        _, observations, _ = lg_model_and_data

        ledh_kwargs = _make_ledh_kwargs(method, stop_grad)
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        bpf_kwargs['n_particles'] = 1000

        print(f"\n  LEDH ({method}) vs KF vs BPF likelihood surface:")
        print(f"  {'sigma':>6s}  {'kf_ll':>10s}  {'bpf_ll':>10s}  {'ledh_ll':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

        kf_lls, bpf_lls, ledh_lls = [], [], []
        for sigma in FINE_GRID:
            kf_ll = kf_log_likelihood(sigma, observations)
            bpf_ll = pf_log_likelihood(BootstrapPFHMC, bpf_kwargs, sigma, observations_tf)
            ledh_ll = pf_log_likelihood(
                LEDHParticleFlowFilterHMC, ledh_kwargs, sigma, observations_tf)
            kf_lls.append(kf_ll)
            bpf_lls.append(bpf_ll)
            ledh_lls.append(ledh_ll)
            print(f"  {sigma:6.2f}  {kf_ll:10.4f}  {bpf_ll:10.4f}  {ledh_ll:10.4f}")

        kf_best = FINE_GRID[np.argmax(kf_lls)]
        bpf_best = FINE_GRID[np.argmax(bpf_lls)]
        ledh_best = FINE_GRID[np.argmax(ledh_lls)]
        print(f"\n  KF   peaks at sigma={kf_best}")
        print(f"  BPF  peaks at sigma={bpf_best}")
        print(f"  LEDH peaks at sigma={ledh_best}")
        assert ledh_best == kf_best, \
            f"LEDH ({method}) peaks at {ledh_best}, KF at {kf_best} — biased surface"


# ============================================================================
# Test B: Autodiff vs Finite Difference at Multiple Points
# ============================================================================

class TestAutodiffVsFiniteDifference:
    """Does TF autodiff match finite differences for LEDH?"""

    @pytest.mark.parametrize("method,stop_grad", RESAMPLE_CONFIGS,
                             ids=['systematic', 'soft', 'ot_entropy'])
    def test_ledh_autodiff_vs_fd_grid(self, observations_tf, method, stop_grad):
        """Test B: LEDH autodiff vs FD at multiple obs_noise_std values."""
        ledh_kwargs = _make_ledh_kwargs(method, stop_grad)

        print(f"\n  LEDH ({method}) autodiff vs FD:")
        print(f"  {'sigma':>6s}  {'autodiff':>10s}  {'FD':>10s}  {'rel_err':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

        for sigma in GRAD_CHECK_POINTS:
            _, ad_grad, _, _ = compute_gradient(
                LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf, sigma)
            fd_grad = finite_difference_gradient(
                LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf, sigma)

            ad_val = ad_grad[0]
            rel_err = abs(ad_val - fd_grad) / max(abs(fd_grad), 1e-10)
            print(f"  {sigma:6.2f}  {ad_val:+10.4f}  {fd_grad:+10.4f}  {rel_err:10.4f}")

            # At minimum, autodiff and FD must agree on sign
            if abs(fd_grad) > 1.0:
                assert np.sign(ad_val) == np.sign(fd_grad), \
                    f"LEDH {method} at sigma={sigma}: autodiff sign ({np.sign(ad_val):+.0f}) != " \
                    f"FD sign ({np.sign(fd_grad):+.0f}) — backward pass is broken"


# ============================================================================
# Test C: Likelihood-Only Gradient (No Prior)
# ============================================================================

class TestLikelihoodOnlyGradient:
    """Is the gradient bias from the filter or the prior?"""

    @pytest.mark.parametrize("method,stop_grad", RESAMPLE_CONFIGS,
                             ids=['systematic', 'soft', 'ot_entropy'])
    def test_ledh_likelihood_gradient_vs_bpf(self, observations_tf, method, stop_grad):
        """Test C: LEDH likelihood-only gradient vs BPF reference."""
        ledh_kwargs = _make_ledh_kwargs(method, stop_grad)
        bpf_kwargs = _make_bpf_kwargs('ot_entropy', stop_gradient=False)

        bpf_nll, bpf_ad, bpf_fd = compute_likelihood_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf)
        ledh_nll, ledh_ad, ledh_fd = compute_likelihood_gradient(
            LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf)

        print(f"\n  Likelihood-only gradient (no prior), sigma_init={INIT_OBS_NOISE_STD}:")
        print(f"  {'filter':>6s}  {'method':>12s}  {'-nll':>10s}  {'autodiff':>10s}  {'FD':>10s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
        print(f"  {'BPF':>6s}  {'ot_entropy':>12s}  {bpf_nll:10.4f}  {bpf_ad:+10.4f}  {bpf_fd:+10.4f}")
        print(f"  {'LEDH':>6s}  {method:>12s}  {ledh_nll:10.4f}  {ledh_ad:+10.4f}  {ledh_fd:+10.4f}")

        # LEDH and BPF should agree on gradient sign
        if abs(bpf_ad) > 0.1 and abs(ledh_ad) > 0.1:
            assert np.sign(ledh_ad) == np.sign(bpf_ad), \
                f"LEDH ({method}) lik-grad sign ({np.sign(ledh_ad)}) disagrees with BPF ({np.sign(bpf_ad)})"


# ============================================================================
# Test D: Component-wise Gradient Decomposition
# ============================================================================

class TestComponentGradient:
    """Which term in the LEDH weight formula has the wrong gradient?"""

    def test_gradient_decomposition_at_true_value(self, lg_model_and_data, observations_tf):
        """Test D: Decompose LEDH gradient into components at true parameter."""
        _, observations, _ = lg_model_and_data

        # We need to run one timestep of LEDH in eager mode and extract
        # intermediate quantities under separate GradientTapes.
        obs_np = observations
        y0 = tf.constant(obs_np[0], dtype=DTYPE)

        # Use unconstrained parameterization
        param_specs = _make_param_specs(TRUE_OBS_NOISE_STD)
        handler = ParameterHandler(param_specs, dtype=DTYPE)
        q = handler.unconstrained_init

        # Helper: build model + filter from unconstrained q, run 1 timestep,
        # return decomposed log-weight components as tensors.
        def _run_one_timestep(q_val):
            constrained = handler.constrain(q_val)
            sigma = constrained['obs_noise_std']

            model = _fresh_model(float(sigma.numpy()))
            model.obs_noise_std = sigma  # symbolic tensor

            filt = LEDHParticleFlowFilterHMC(
                model, n_particles=N_PARTICLES, n_lambda_steps=N_LAMBDA_STEPS,
                resampling_method='systematic',
                stop_gradient_resampling=True,
                weight_clip_range=50.0,
                eager_mode=True,
            )
            filt.initialize(random_seed=SEED)

            # Predict
            filt.predict(t=1)

            R = model.observation_noise_cov
            R_inv = safe_inv(R)
            regularization = tf.constant(1e-8, dtype=DTYPE)

            eta_1 = filt.eta_0.value()
            eta_bar = filt.eta_bar_0.value()
            log_theta = tf.zeros([N_PARTICLES], dtype=DTYPE)
            lambda_val = tf.constant(0.0, dtype=DTYPE)
            I_sd = tf.eye(filt.state_dim, dtype=DTYPE)

            for j in range(N_LAMBDA_STEPS):
                d_lambda = filt.lambda_steps[j]
                lambda_val = lambda_val + d_lambda

                A_batch, b_batch = compute_flow_params_batch.python_function(
                    model, eta_bar, lambda_val, y0,
                    filt.particle_covs.value(),
                    R, R_inv, filt.eta_bar_0.value(),
                    filt.state_dim, regularization,
                )

                drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
                eta_bar = eta_bar + d_lambda * drift_bar

                drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
                eta_1 = eta_1 + d_lambda * drift_1

                M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
                log_det_M = safe_log_abs_det.python_function(M_batch)
                log_theta = log_theta + log_det_M

            # Normalize Jacobians
            max_lt = tf.reduce_max(log_theta)
            log_theta_norm = log_theta - max_lt
            theta = tf.exp(log_theta_norm)

            eta_0 = filt.eta_0.value()
            particles_prev = filt.particles_prev.value()

            # Component 1: observation log-prob
            log_p_obs = model.log_observation_prob_batch(y0, eta_1)

            # Component 2: transition log-prob for flowed particles
            from src.utils.linalg import safe_cholesky
            import math
            f_prev = model.state_transition_mean_batch(particles_prev)
            Q = model.state_transition_cov_batch(particles_prev)
            L_Q = safe_cholesky(Q)

            diff_1 = eta_1 - f_prev
            y_1 = tf.linalg.triangular_solve(L_Q, tf.transpose(diff_1), lower=True)
            y_1 = tf.transpose(y_1)
            sd_f = tf.cast(filt.state_dim, DTYPE)
            log_p_eta1 = -0.5 * (
                tf.reduce_sum(y_1**2, axis=1) +
                2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Q))) +
                sd_f * tf.math.log(2.0 * tf.constant(math.pi, dtype=DTYPE))
            )

            # Component 3: log Jacobian
            log_jac = log_theta_norm  # per-particle

            # Component 4: proposal log-prob
            diff_0 = eta_0 - f_prev
            y_0 = tf.linalg.triangular_solve(L_Q, tf.transpose(diff_0), lower=True)
            y_0 = tf.transpose(y_0)
            log_p_eta0 = -0.5 * (
                tf.reduce_sum(y_0**2, axis=1) +
                2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Q))) +
                sd_f * tf.math.log(2.0 * tf.constant(math.pi, dtype=DTYPE))
            )

            # Return mean over particles for each component
            return (
                tf.reduce_mean(log_p_obs),
                tf.reduce_mean(log_p_eta1),
                tf.reduce_mean(log_jac),
                tf.reduce_mean(log_p_eta0),
            )

        # Compute gradient of each component
        components = ['log_p_obs', 'log_p_eta1', 'log_jacobian', 'log_p_eta0']
        print(f"\n  Component-wise gradient at true sigma={TRUE_OBS_NOISE_STD} (T=1, t=1):")
        print(f"  {'component':>15s}  {'value':>10s}  {'grad':>10s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*10}")

        for idx, name in enumerate(components):
            with tf.GradientTape() as tape:
                tape.watch(q)
                vals = _run_one_timestep(q)
                target = vals[idx]
            grad = tape.gradient(target, q)
            grad_val = float(grad.numpy()[0]) if grad is not None else 0.0
            print(f"  {name:>15s}  {float(target.numpy()):10.4f}  {grad_val:+10.4f}")


# ============================================================================
# Test E: Stop-Gradient Jacobian
# ============================================================================

class TestStopGradientJacobian:
    """Is the Jacobian backward pass the source of bias?"""

    def test_stop_gradient_jacobian(self, lg_model_and_data, observations_tf):
        """Test E: Run LEDH with tf.stop_gradient(theta) — does gradient fix?"""
        _, observations, _ = lg_model_and_data

        # We need a custom eager run that stops gradient on theta.
        # Compare: normal LEDH gradient vs stopped-Jacobian LEDH gradient vs BPF.
        param_specs = _make_param_specs(INIT_OBS_NOISE_STD)
        handler = ParameterHandler(param_specs, dtype=DTYPE)
        q = handler.unconstrained_init
        seed_tf = tf.constant([SEED, 0], dtype=tf.int32)

        def _ledh_ll_with_option(q_val, stop_jac):
            constrained = handler.constrain(q_val)
            sigma = constrained['obs_noise_std']

            model = _fresh_model(float(sigma.numpy()))
            model.obs_noise_std = sigma

            filt = LEDHParticleFlowFilterHMC(
                model, n_particles=N_PARTICLES, n_lambda_steps=N_LAMBDA_STEPS,
                resampling_method='systematic',
                stop_gradient_resampling=True,
                weight_clip_range=50.0,
                eager_mode=True,
            )
            filt.initialize(random_seed=SEED)

            R = model.observation_noise_cov
            R_inv = safe_inv(R)
            regularization = tf.constant(1e-8, dtype=DTYPE)
            obs_tf = tf.constant(observations, dtype=DTYPE)
            total_ll = tf.constant(0.0, dtype=DTYPE)

            for t_idx in range(min(T, 10)):  # Use 10 timesteps for speed
                if hasattr(model, 't'):
                    model.t = t_idx + 1
                filt.particles_prev.assign(filt.particles.value())
                from src.filters.kalman.batched_ekf import batched_ekf_predict, batched_ekf_update
                eta_bar_0, cov_pred = batched_ekf_predict.python_function(
                    model, filt.particles.value(), filt.particle_covs.value())
                filt.particle_covs.assign(cov_pred)
                filt.eta_bar_0.assign(eta_bar_0)

                seed_t = filt._next_seed()
                eta_0 = model.state_transition_batch(filt.particles_prev.value(), seed_t, t=t_idx+1)
                filt.eta_0.assign(eta_0)

                y = obs_tf[t_idx]
                eta_1 = filt.eta_0.value()
                eta_bar = filt.eta_bar_0.value()
                log_theta = tf.zeros([N_PARTICLES], dtype=DTYPE)
                lambda_val = tf.constant(0.0, dtype=DTYPE)
                I_sd = tf.eye(filt.state_dim, dtype=DTYPE)

                for j in range(N_LAMBDA_STEPS):
                    d_lambda = filt.lambda_steps[j]
                    lambda_val = lambda_val + d_lambda
                    A_batch, b_batch = compute_flow_params_batch.python_function(
                        model, eta_bar, lambda_val, y,
                        filt.particle_covs.value(),
                        R, R_inv, filt.eta_bar_0.value(),
                        filt.state_dim, regularization)
                    drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
                    eta_bar = eta_bar + d_lambda * drift_bar
                    drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
                    eta_1 = eta_1 + d_lambda * drift_1
                    M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch
                    log_det_M = safe_log_abs_det.python_function(M_batch)
                    log_theta = log_theta + log_det_M

                max_lt = tf.reduce_max(log_theta)
                log_theta = log_theta - max_lt
                theta = tf.exp(log_theta)

                if stop_jac:
                    theta = tf.stop_gradient(theta)

                filt.particles.assign(eta_1)
                weights_new, log_lik = compute_flow_weights.python_function(
                    eta_1=eta_1, eta_0=filt.eta_0.value(),
                    particles_prev=filt.particles_prev.value(),
                    observation=y, model=model,
                    prev_weights=filt.weights.value(),
                    jacobians=theta, clip_range=filt.weight_clip_range)
                filt.weights.assign(weights_new)
                total_ll = total_ll + log_lik

                _, cov_upd = batched_ekf_update.python_function(
                    model, filt.eta_bar_0.value(), filt.particle_covs.value(), y)
                filt.particle_covs.assign(cov_upd)

                from src.resampling.diagnosis import effective_sample_size as ess_tf
                ess = ess_tf(filt.weights.value())
                if ess < 0.5 * N_PARTICLES:
                    filt._resample_hmc()

            return total_ll

        # Normal LEDH
        with tf.GradientTape() as tape:
            tape.watch(q)
            ll_normal = _ledh_ll_with_option(q, stop_jac=False)
        grad_normal = float(tape.gradient(ll_normal, q).numpy()[0])

        # Stop-gradient Jacobian
        with tf.GradientTape() as tape:
            tape.watch(q)
            ll_stop = _ledh_ll_with_option(q, stop_jac=True)
        grad_stop = float(tape.gradient(ll_stop, q).numpy()[0])

        # BPF reference
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        _, bpf_grad, _, _ = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf)

        print(f"\n  Stop-gradient Jacobian test (sigma_init={INIT_OBS_NOISE_STD}, T=10):")
        print(f"    LEDH normal:        grad = {grad_normal:+.4f}")
        print(f"    LEDH stop_grad(θ):  grad = {grad_stop:+.4f}")
        print(f"    BPF reference:      grad = {bpf_grad[0]:+.4f}")
        print(f"    Same sign? normal={np.sign(grad_normal)}, stopped={np.sign(grad_stop)}, bpf={np.sign(bpf_grad[0])}")


# ============================================================================
# Test F: Single Timestep (T=1)
# ============================================================================

class TestSingleTimestep:
    """Does the gradient error occur per-timestep or accumulate?"""

    def test_ledh_gradient_T1(self, lg_model_and_data):
        """Test F: LEDH gradient at T=1 (no resampling) vs BPF."""
        _, observations, _ = lg_model_and_data
        obs_T1 = tf.constant(observations[:1], dtype=DTYPE)

        ledh_kwargs = _make_ledh_kwargs('systematic', True)
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)

        _, ledh_grad, _, ledh_ll = compute_gradient(
            LEDHParticleFlowFilterHMC, ledh_kwargs, obs_T1)
        _, bpf_grad, _, bpf_ll = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, obs_T1)

        ledh_fd = finite_difference_gradient(
            LEDHParticleFlowFilterHMC, ledh_kwargs, obs_T1)
        bpf_fd = finite_difference_gradient(
            BootstrapPFHMC, bpf_kwargs, obs_T1)

        print(f"\n  Single timestep (T=1) gradient check:")
        print(f"  {'filter':>6s}  {'autodiff':>10s}  {'FD':>10s}  {'rel_err':>10s}  {'log_lik':>10s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        ledh_re = abs(ledh_grad[0] - ledh_fd) / max(abs(ledh_fd), 1e-10)
        bpf_re = abs(bpf_grad[0] - bpf_fd) / max(abs(bpf_fd), 1e-10)
        print(f"  {'BPF':>6s}  {bpf_grad[0]:+10.4f}  {bpf_fd:+10.4f}  {bpf_re:10.4f}  {bpf_ll:10.4f}")
        print(f"  {'LEDH':>6s}  {ledh_grad[0]:+10.4f}  {ledh_fd:+10.4f}  {ledh_re:10.4f}  {ledh_ll:10.4f}")

        # T=1 gradient directions should match
        if abs(bpf_grad[0]) > 0.1 and abs(ledh_grad[0]) > 0.1:
            assert np.sign(ledh_grad[0]) == np.sign(bpf_grad[0]), \
                f"T=1: LEDH grad sign ({np.sign(ledh_grad[0])}) != BPF ({np.sign(bpf_grad[0])})"

    def test_ledh_gradient_scaling_with_T(self, lg_model_and_data):
        """Test F (cont): Does LEDH gradient bias grow with T?"""
        _, observations, _ = lg_model_and_data

        ledh_kwargs = _make_ledh_kwargs('systematic', True)
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)

        print(f"\n  LEDH vs BPF gradient scaling with T:")
        print(f"  {'T':>4s}  {'ledh_grad':>10s}  {'bpf_grad':>10s}  {'ratio':>10s}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")

        for t_val in [1, 5, 10, 20, 50]:
            obs_t = tf.constant(observations[:t_val], dtype=DTYPE)
            _, ledh_g, _, _ = compute_gradient(
                LEDHParticleFlowFilterHMC, ledh_kwargs, obs_t)
            _, bpf_g, _, _ = compute_gradient(
                BootstrapPFHMC, bpf_kwargs, obs_t)
            ratio = ledh_g[0] / bpf_g[0] if abs(bpf_g[0]) > 1e-10 else float('inf')
            print(f"  {t_val:4d}  {ledh_g[0]:+10.4f}  {bpf_g[0]:+10.4f}  {ratio:10.4f}")


# ============================================================================
# Test G: Eager vs Compiled
# ============================================================================

class TestEagerVsCompiled:
    """Does tf.function / tf.while_loop introduce gradient errors?"""

    def test_eager_vs_compiled_gradient(self, observations_tf):
        """Test G: Compare LEDH gradient in eager vs compiled mode."""
        # Eager
        ledh_eager = _make_ledh_kwargs('systematic', True)
        ledh_eager['eager_mode'] = True
        _, eager_grad, _, eager_ll = compute_gradient(
            LEDHParticleFlowFilterHMC, ledh_eager, observations_tf)

        # Compiled
        ledh_compiled = _make_ledh_kwargs('systematic', True)
        ledh_compiled['eager_mode'] = False
        _, comp_grad, _, comp_ll = compute_gradient(
            LEDHParticleFlowFilterHMC, ledh_compiled, observations_tf)

        # BPF reference
        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)
        _, bpf_grad, _, bpf_ll = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf)

        print(f"\n  Eager vs Compiled gradient (sigma_init={INIT_OBS_NOISE_STD}):")
        print(f"    LEDH eager:    grad={eager_grad[0]:+.4f}  ll={eager_ll:.4f}")
        print(f"    LEDH compiled: grad={comp_grad[0]:+.4f}  ll={comp_ll:.4f}")
        print(f"    BPF reference: grad={bpf_grad[0]:+.4f}  ll={bpf_ll:.4f}")

        rel_diff = abs(eager_grad[0] - comp_grad[0]) / max(abs(eager_grad[0]), 1e-10)
        print(f"    Eager/compiled rel_diff: {rel_diff:.4f}")
        assert rel_diff < 0.30, \
            f"Eager and compiled gradients differ by {rel_diff:.4f} — compilation issue"


# ============================================================================
# Test H: Flow Steps Sensitivity
# ============================================================================

class TestFlowStepsSensitivity:
    """Does gradient bias grow with the number of Euler steps?"""

    @pytest.mark.parametrize("n_steps", [3, 5, 10, 15, 29],
                             ids=['3steps', '5steps', '10steps', '15steps', '29steps'])
    def test_gradient_vs_flow_steps(self, observations_tf, n_steps):
        """Test H: LEDH gradient at true param with varying n_lambda_steps."""
        ledh_kwargs = _make_ledh_kwargs('systematic', True)
        ledh_kwargs['n_lambda_steps'] = n_steps

        bpf_kwargs = _make_bpf_kwargs('systematic', stop_gradient=True)

        _, ledh_grad, _, ledh_ll = compute_gradient(
            LEDHParticleFlowFilterHMC, ledh_kwargs, observations_tf)
        _, bpf_grad, _, bpf_ll = compute_gradient(
            BootstrapPFHMC, bpf_kwargs, observations_tf)

        print(f"\n  n_lambda_steps={n_steps}:")
        print(f"    LEDH grad={ledh_grad[0]:+.4f}  ll={ledh_ll:.4f}")
        print(f"    BPF  grad={bpf_grad[0]:+.4f}  ll={bpf_ll:.4f}")


# ============================================================================
# Test I: Float64 vs Float32
# ============================================================================

class TestFloat64VsFloat32:
    """Is float32 precision the root cause?"""

    def test_float32_vs_float64_gradient(self, lg_model_and_data):
        """Test I: Compare LEDH gradient in float32 vs float64."""
        _, observations, _ = lg_model_and_data

        results = {}
        for dtype, dtype_name in [(tf.float32, 'float32'), (tf.float64, 'float64')]:
            obs_tf = tf.constant(observations, dtype=dtype)

            # Build LEDH kwargs with appropriate dtype model
            model = LinearGaussianModel(
                F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
                obs_noise_std=INIT_OBS_NOISE_STD, dtype=dtype,
            )
            from src.DF.types import ParameterSpec
            import tensorflow_probability as tfp
            param_specs = {
                'obs_noise_std': ParameterSpec(
                    name='obs_noise_std',
                    init_value=INIT_OBS_NOISE_STD,
                    constraint='positive',
                    prior=tfp.distributions.LogNormal(
                        loc=tf.constant(0.0, dtype=dtype),
                        scale=tf.constant(1.0, dtype=dtype),
                    ),
                )
            }
            from src.DF.hmc_runner import DPFRunner
            ledh_kwargs = dict(
                n_particles=N_PARTICLES,
                n_lambda_steps=N_LAMBDA_STEPS,
                resampling_method='systematic',
                weight_clip_range=50.0,
                stop_gradient_resampling=True,
                eager_mode=False,
            )
            runner = DPFRunner(
                base_model=model,
                filter_class=LEDHParticleFlowFilterHMC,
                filter_kwargs=ledh_kwargs,
                param_specs=param_specs,
                sampler='hmc',
            )
            runner._observations_tf = obs_tf
            q = runner.param_handler.unconstrained_init
            with tf.GradientTape() as tape:
                tape.watch(q)
                nlp = runner._negative_log_posterior(q)
            grad = tape.gradient(nlp, q)
            results[dtype_name] = {
                'grad': float(grad.numpy()[0]),
                'nlp': float(nlp.numpy()),
            }

        print(f"\n  Float32 vs Float64 (sigma_init={INIT_OBS_NOISE_STD}):")
        print(f"    float32: grad={results['float32']['grad']:+.4f}  nlp={results['float32']['nlp']:.4f}")
        print(f"    float64: grad={results['float64']['grad']:+.4f}  nlp={results['float64']['nlp']:.4f}")
        print(f"    Same sign? {np.sign(results['float32']['grad']) == np.sign(results['float64']['grad'])}")


# Need this import at module level for Test I
from src.models.linear_gaussian import LinearGaussianModel
