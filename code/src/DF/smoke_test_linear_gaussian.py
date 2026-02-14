"""
Phase 1 Smoke Test: LinearGaussianModel + EKF + HMC/NUTS

Infer process_noise_std and obs_noise_std of a linear Gaussian model.
The Kalman filter (via EKF on a linear model) is optimal for this model,
so the log-likelihood surface is smooth and HMC should converge easily.

Expected outcome:
- Posterior means near true values (process_noise_std=1.0, obs_noise_std=0.5)
- Acceptance rate > 0.3
- R-hat < 1.2
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from pathlib import Path

# Ensure code/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.DF import DPFRunner, ParameterSpec
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data


def smoke_test_linear_gaussian():
    print("=" * 60)
    print("PHASE 1 SMOKE TEST: LinearGaussian + EKF + NUTS")
    print("=" * 60)

    # ---------- True model (for data generation) ----------
    # B and D encode FULL noise (no process_noise_std/obs_noise_std)
    true_model = LinearGaussianModel(
        F=[[0.9, 0.1], [0.0, 0.8]],
        B=[[1.0], [0.5]],
        H=[[1.0, 0.0]],
        D=[[0.5]],
        mu_0=[0.0, 0.0],
        Sigma_0=[[1.0, 0.0], [0.0, 1.0]],
    )
    # True Q = B@B^T = [[1.0, 0.5], [0.5, 0.25]]  (process_noise_std=1.0 equiv.)
    # True R = D@D^T = [[0.25]]                     (obs_noise_std=0.5 equiv.)

    rng = np.random.default_rng(42)
    T = 200
    initial_state, states, observations = generate_data(true_model, T=T, rng=rng)
    print(f"Generated {T} observations, state_dim=2, obs_dim=1")
    print(f"True equivalent: process_noise_std=1.0, obs_noise_std=0.5")

    # ---------- Inference model (wrong noise scales) ----------
    # B and D are unit-scale "direction" matrices.
    # process_noise_std and obs_noise_std control the actual noise scale:
    #   Q = process_noise_std^2 * B@B^T
    #   R = obs_noise_std^2 * D@D^T
    init_sigma_q = 0.5   # Wrong! (true is 1.0)
    init_sigma_r = 1.0   # Wrong! (true is 0.5)

    inference_model = LinearGaussianModel(
        F=[[0.9, 0.1], [0.0, 0.8]],
        B=[[1.0], [0.5]],      # Same direction as true model
        H=[[1.0, 0.0]],
        D=[[1.0]],             # Unit scale; actual scale via obs_noise_std
        mu_0=[0.0, 0.0],
        Sigma_0=[[1.0, 0.0], [0.0, 1.0]],
        process_noise_std=init_sigma_q,
        obs_noise_std=init_sigma_r,
    )

    print(f"\nInference model initial guesses:")
    print(f"  process_noise_std = {init_sigma_q} (true: 1.0)")
    print(f"  obs_noise_std = {init_sigma_r} (true: 0.5)")
    print(f"  Initial Q = pns^2 * B@B^T = {init_sigma_q**2} * [[1, 0.5], [0.5, 0.25]]")
    print(f"  Initial R = ons^2 * D@D^T = {init_sigma_r**2} * [[1]]")

    # ---------- Verify EKF log-likelihood works ----------
    print("\n--- Verifying EKF log_marginal_likelihood_tf ---")
    ekf = ExtendedKalmanFilter(inference_model, sample_initial_mean=False, random_seed=0)
    obs_tf = tf.constant(observations, dtype=tf.float32)
    ll = ekf.log_marginal_likelihood_tf(obs_tf)
    print(f"  Log-likelihood at initial params: {float(ll.numpy()):.2f}")

    # Verify gradient exists
    with tf.GradientTape() as tape:
        pns_tensor = tf.constant(init_sigma_q, dtype=tf.float32)
        tape.watch(pns_tensor)
        inference_model.process_noise_std = pns_tensor
        ll_grad = ekf.log_marginal_likelihood_tf(obs_tf)
    grad = tape.gradient(ll_grad, pns_tensor)
    inference_model.process_noise_std = init_sigma_q  # restore
    print(f"  Gradient d(LL)/d(pns) exists: {grad is not None}")
    if grad is not None:
        print(f"  Gradient value: {float(grad.numpy()):.4f}")

    # ---------- Parameter specs ----------
    param_specs = {
        'process_noise_std': ParameterSpec(
            name='process_noise_std',
            init_value=init_sigma_q,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32)
            )
        ),
        'obs_noise_std': ParameterSpec(
            name='obs_noise_std',
            init_value=init_sigma_r,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32)
            )
        ),
    }

    # ---------- DPF Runner ----------
    runner = DPFRunner(
        base_model=inference_model,
        filter_class=ExtendedKalmanFilter,
        filter_kwargs={'sample_initial_mean': False, 'random_seed': 0},
        param_specs=param_specs,
        sampler='hmc'
    )

    # ---------- Run ----------
    result = runner.run_inference(
        observations=observations,
        num_samples=200,
        num_burnin=100,
        step_size=0.01,
        num_leapfrog_steps=10,
        seed=42
    )

    # ---------- Validate ----------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    true_vals = {'process_noise_std': 1.0, 'obs_noise_std': 0.5}
    all_passed = True

    for name in ['process_noise_std', 'obs_noise_std']:
        true_val = true_vals[name]
        stats = result.summary[name]
        in_ci = stats['q5'] <= true_val <= stats['q95']
        print(f"\n  {name}:")
        print(f"    True:           {true_val:.4f}")
        print(f"    Posterior mean:  {stats['mean']:.4f} +/- {stats['std']:.4f}")
        print(f"    90% CI:         [{stats['q5']:.4f}, {stats['q95']:.4f}]")
        print(f"    True in 90% CI: {'YES' if in_ci else 'NO'}")

        # Basic sanity checks
        if np.isnan(stats['mean']):
            print(f"    FAIL: posterior mean is NaN!")
            all_passed = False
        if stats['mean'] <= 0:
            print(f"    FAIL: posterior mean is non-positive!")
            all_passed = False

    print(f"\nDiagnostics:")
    print(f"  Acceptance rate: {result.diagnostics['acceptance_rate']:.1%}")
    print(f"  Final step size: {result.diagnostics['final_step_size']:.6f}")
    print(f"  ESS: {result.diagnostics['ess']}")
    print(f"  R-hat: {result.diagnostics['rhat']}")

    accept = result.diagnostics['acceptance_rate']
    if accept < 0.1:
        print(f"\n  WARNING: Acceptance rate very low ({accept:.1%})")
        all_passed = False

    if all_passed:
        print("\n>>> PHASE 1 SMOKE TEST PASSED <<<")
    else:
        print("\n>>> PHASE 1 SMOKE TEST COMPLETED WITH WARNINGS <<<")

    return result


if __name__ == '__main__':
    smoke_test_linear_gaussian()
