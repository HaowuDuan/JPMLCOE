"""Leapfrog energy-error replay for RB LEDH+OT HMC at proposed tuned settings.

Validates the mass_vector + step_size choice without running the full chain.
Runs K independent leapfrog trajectories from a fixed peak with random
momenta, computes energy error `delta_H` for each, and predicts HMC
acceptance probability from `mean(min(1, exp(-delta_H)))`.

Proposed settings (for the RB 4-chain multi-chain configs):
  step_size = 0.05
  num_leapfrog_steps = 5
  mass_vector = [50.0, 50.0]

Reading:
  mean |delta_H| ~ 0.5  -> acceptance ~ 0.6-0.85  (target)
  mean |delta_H| < 0.1  -> step_size can go bigger
  mean |delta_H| > 2    -> step_size too big, will reject often

Starting point: q corresponding to constrained (sigma_range, sigma_bearing)
= (0.1, 0.113). That is truth-for-range and the seed-42 peak-for-bearing.

Fixed PF seed = [42, 0] (matches HMC production), so the target is the
deterministic surrogate. Only momentum varies across trajectories.

K=10 trajectories total. Each trajectory = num_leapfrog steps of leapfrog,
each step needs 1 gradient evaluation = 1 PF forward pass. Plus
start/end log-prob. Total ~70 filter evaluations at N=500, T=50, n_lambda=15.
Runtime: ~15-20 min on CPU.

No assertion. Prints and saves ΔH per trajectory + summary to
tests/filters/results/rb_leapfrog_energy_replay.json.

Run:
  cd code
  .venv/bin/python -m pytest tests/filters/test_rb_leapfrog_energy_replay.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

tf.config.set_visible_devices([], 'GPU')

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel


# === Experiment config ===
DTYPE = tf.float32
T = 50
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1
DATA_SEED = 42

# Filter settings match the 4-chain configs
N_PARTICLES = 500
N_LAMBDA_STEPS = 15
OT_EPSILON = 0.5

# Tuned HMC settings being validated
STEP_SIZE = 0.10
NUM_LEAPFROG = 5
MASS_VECTOR = [50.0, 50.0]

# Starting point (peak of log p_hat)
PEAK_SIGMA_RANGE = 0.1
PEAK_SIGMA_BEARING = 0.113

# Prior (matches 4-chain configs)
PRIOR_LOC = -2.05
PRIOR_SCALE = 0.5

# Number of leapfrog trajectories with random momenta
K_TRAJECTORIES = 5

# Posterior std in unconstrained q-space (from prior lever-sweep curvature);
# used to report per-step movement as a fraction of one posterior std.
POSTERIOR_STD_Q = 0.14

# Fixed PF seed (deterministic target surface)
PF_SEED = tf.constant([42, 0], dtype=tf.int32)


def _softplus_inverse(y):
    """Inverse of softplus: q = log(exp(y) - 1)."""
    return float(np.log(np.expm1(y)))


def _make_filter(sigma_r_val, sigma_b_val):
    base_model = RangeBearingModel(
        sigma_range=sigma_r_val,
        sigma_bearing=sigma_b_val,
        dtype=DTYPE,
    )
    diff_model = DifferentiableModel(
        base_model, ['sigma_range', 'sigma_bearing']
    )
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': OT_EPSILON},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


class TestRBLeapfrogEnergyReplay:
    def test_energy_error_at_tuned_settings(self):
        # Data at truth
        truth_model = RangeBearingModel(
            sigma_range=TRUE_SIGMA_RANGE,
            sigma_bearing=TRUE_SIGMA_BEARING,
            dtype=DTYPE,
        )
        rng = np.random.default_rng(DATA_SEED)
        _, _, obs = generate_data(truth_model, T=T, rng=rng)
        obs_tf = tf.constant(obs, dtype=DTYPE)

        # Build filter once at peak; we'll swap params via diff_model.update
        diff_model, filt = _make_filter(PEAK_SIGMA_RANGE, PEAK_SIGMA_BEARING)

        # Peak in unconstrained q-space (Softplus bijector)
        q_peak = tf.constant([
            _softplus_inverse(PEAK_SIGMA_RANGE),
            _softplus_inverse(PEAK_SIGMA_BEARING),
        ], dtype=DTYPE)

        # Prior in constrained sigma-space (LogNormal), same for both params
        prior = tfp.distributions.LogNormal(
            loc=tf.constant(PRIOR_LOC, dtype=DTYPE),
            scale=tf.constant(PRIOR_SCALE, dtype=DTYPE),
        )
        softplus_bij = tfp.bijectors.Softplus()

        def target_log_prob(q):
            """Unconstrained target: log p_hat + log prior + log |dsigma/dq|."""
            sigma = softplus_bij.forward(q)
            # Set model parameters via diff_model (expects dict {name: scalar tensor}).
            diff_model.update_parameters({
                'sigma_range': sigma[0],
                'sigma_bearing': sigma[1],
            })
            ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
            lp = tf.reduce_sum(prior.log_prob(sigma))
            log_jac = tf.reduce_sum(softplus_bij.forward_log_det_jacobian(q, event_ndims=0))
            return ll + lp + log_jac

        # Momentum distribution: N(0, diag(mass_vector))
        mass_tf = tf.constant(MASS_VECTOR, dtype=DTYPE)
        scale_diag = tf.sqrt(mass_tf)

        # Preconditioned HMC kernel
        def build_kernel():
            return tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=tf.constant(STEP_SIZE, dtype=DTYPE),
                num_leapfrog_steps=NUM_LEAPFROG,
                momentum_distribution=tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(2, dtype=DTYPE),
                    scale_diag=scale_diag,
                ),
            )

        print(f"\n  [RB leapfrog energy replay  "
              f"step_size={STEP_SIZE}  num_leapfrog={NUM_LEAPFROG}  "
              f"mass={MASS_VECTOR}]")
        print(f"  peak: sigma_range={PEAK_SIGMA_RANGE}, "
              f"sigma_bearing={PEAK_SIGMA_BEARING}  "
              f"(q={[float(v) for v in q_peak.numpy()]})")
        print(f"  running {K_TRAJECTORIES} leapfrog trajectories...")

        # Build kernel once; bootstrap + step repeatedly with different momentum seeds.
        kernel = build_kernel()

        per_traj = []
        for k in range(K_TRAJECTORIES):
            state = q_peak
            kr = kernel.bootstrap_results(state)
            mom_seed = tf.constant([42, 1000 + k], dtype=tf.int32)
            new_state, new_kr = kernel.one_step(state, kr, seed=mom_seed)
            # log_accept_ratio = target_log_prob(q_new) - target_log_prob(q_init)
            #                   - kinetic(p_new) + kinetic(p_init) = -delta_H
            log_accept_ratio = float(new_kr.log_accept_ratio.numpy())
            delta_H = -log_accept_ratio
            accept_prob = float(np.exp(min(log_accept_ratio, 0.0)))
            # Post-MH-accept displacement. If rejected, new_state == state so
            # displacement is 0. Per-trajectory squared displacement is already an
            # unbiased estimator of accept_prob * |q_proposed - q_peak|^2 --
            # do NOT multiply by accept_prob again (double-count, per Codex).
            disp = float(tf.norm(new_state - q_peak).numpy())
            sq_disp = disp ** 2
            is_accepted = bool(new_kr.is_accepted.numpy()) if hasattr(new_kr, 'is_accepted') else (log_accept_ratio > -1e-9 or disp > 1e-9)
            print(f"    traj {k+1:2d}:  delta_H = {delta_H:+7.4f}  "
                  f"accept_prob = {accept_prob:.4f}  "
                  f"is_accept = {int(is_accepted)}  "
                  f"|Δq| = {disp:.4f}")
            per_traj.append({
                'traj_idx': k,
                'momentum_seed': [42, 1000 + k],
                'delta_H': delta_H,
                'log_accept_ratio': log_accept_ratio,
                'accept_prob': accept_prob,
                'is_accepted': is_accepted,
                'displacement_q': disp,
                'sq_displacement_q': sq_disp,
            })

        delta_H_list = np.array([r['delta_H'] for r in per_traj])
        accept_list = np.array([r['accept_prob'] for r in per_traj])
        sq_disp_list = np.array([r['sq_displacement_q'] for r in per_traj])

        # ESJD (expected squared jumping distance): mean of observed post-accept
        # squared displacement. This is an unbiased ESJD estimator -- no extra
        # acceptance factor.
        esjd = float(sq_disp_list.mean())
        rms_move = float(np.sqrt(esjd))
        mixing_rate = rms_move / POSTERIOR_STD_Q   # per-step move as fraction of posterior std

        # Report
        print(f"\n  === RESULTS (K={K_TRAJECTORIES}) ===")
        print(f"  delta_H:         mean={delta_H_list.mean():+.4f}  "
              f"std={delta_H_list.std():.4f}  "
              f"|mean|={np.abs(delta_H_list).mean():.4f}")
        print(f"  acceptance:      mean={accept_list.mean():.4f}  "
              f"median={np.median(accept_list):.4f}  "
              f"n_accepted={sum(r['is_accepted'] for r in per_traj)}/{K_TRAJECTORIES}")
        print(f"  ESJD (mean sq move in q-space): {esjd:.6f}")
        print(f"  RMS move / posterior_std_q: {mixing_rate:.3f}  "
              f"(per-step move as fraction of one posterior std)")
        print(f"\n  VERDICT (screening-level, K={K_TRAJECTORIES} is small):")
        mean_abs = float(np.abs(delta_H_list).mean())
        mean_acc = float(accept_list.mean())
        # Screening categories only. Does NOT certify convergence or ESS.
        if mean_acc < 0.2:
            verdict = "BROKEN: acceptance too low. step_size too big OR target has cliff."
        elif rms_move < 0.02 * POSTERIOR_STD_Q:
            verdict = "BROKEN: per-step move ~0. Effective step_size too small (chain won't move)."
        elif mean_abs > 3.0:
            verdict = "BROKEN: energy error huge. step_size too big."
        else:
            verdict = (f"SCREENING PASSED: accept={mean_acc:.2f}, "
                       f"rms_move/std_q={mixing_rate:.2f}. "
                       "Settings are not obviously broken. Full chain needed to certify convergence.")
        print(f"  {verdict}")

        # Save
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / 'rb_leapfrog_energy_replay.json'
        payload = {
            'case_name': 'RB leapfrog energy replay at tuned settings',
            'model': 'range_bearing',
            'filter': 'ledh_invertible_hmc',
            'T': T,
            'data_seed': DATA_SEED,
            'n_particles': N_PARTICLES,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'ot_epsilon': OT_EPSILON,
            'step_size': STEP_SIZE,
            'num_leapfrog_steps': NUM_LEAPFROG,
            'mass_vector': MASS_VECTOR,
            'peak_sigma_range': PEAK_SIGMA_RANGE,
            'peak_sigma_bearing': PEAK_SIGMA_BEARING,
            'pf_seed': [42, 0],
            'num_trajectories': K_TRAJECTORIES,
            'per_trajectory': per_traj,
            'summary': {
                'delta_H_mean': float(delta_H_list.mean()),
                'delta_H_std': float(delta_H_list.std()),
                'delta_H_abs_mean': float(np.abs(delta_H_list).mean()),
                'delta_H_abs_median': float(np.median(np.abs(delta_H_list))),
                'accept_prob_mean': float(accept_list.mean()),
                'accept_prob_median': float(np.median(accept_list)),
                'esjd_q_space': esjd,
                'rms_move_q_space': rms_move,
                'posterior_std_q_assumed': POSTERIOR_STD_Q,
                'mixing_rate_per_step': mixing_rate,
                'verdict': verdict,
            },
        }
        with out_path.open('w') as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {out_path}")
