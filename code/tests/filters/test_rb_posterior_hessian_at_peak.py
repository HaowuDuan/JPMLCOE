"""Posterior Hessian at the RB LEDH+OT target peak, in unconstrained q-space.

Computes the 2x2 Hessian of `log p_hat(y|sigma) + log prior(sigma) +
log |dsigma/dq|` with respect to `q = softplus_inverse(sigma)` at the
target peak. Uses analytical first-order gradient via TF autodiff plus
a central finite difference of the gradient for the second derivative.
This keeps the Hessian cost to ~4 filter-with-tape evaluations instead
of the ~9 needed for a pure FD Hessian.

From the Hessian `H`:
  posterior_cov_q = -inv(H)                 (inverse of negative Hessian)
  posterior_std_q = sqrt(diag(posterior_cov_q))
  posterior_corr  = posterior_cov_q / (std_i * std_j)
  optimal_mass_diag = 1 / diag(posterior_cov_q)    (isotropic mass)
  optimal_mass_full = -H                            (full mass matrix)

Compares the derived mass matrix to the current hand-picked [80, 80].

Settings match production HMC config for RB LEDH+OT.

Fixed PF seed = [42, 0].

Cost: 4 filter-with-tape evaluations + some numpy. ~3-5 min on CPU.

Output: tests/filters/results/rb_posterior_hessian_at_peak.json.

Run:
  cd code
  .venv/bin/python -m pytest tests/filters/test_rb_posterior_hessian_at_peak.py -v -s
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


DTYPE = tf.float32
T = 50
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1
DATA_SEED = 42
PF_SEED = tf.constant([42, 0], dtype=tf.int32)

N_PARTICLES = 500
N_LAMBDA_STEPS = 15
OT_EPSILON = 0.5

# Initial peak guess in constrained space (from existing scan results);
# Newton iteration refines this until gradient norm < GRAD_TOL.
PEAK_SIGMA_RANGE_INIT = 0.1
PEAK_SIGMA_BEARING_INIT = 0.113

PRIOR_LOC = -2.05
PRIOR_SCALE = 0.5

# Finite-difference step for gradient
FD_H = 1e-3

# Newton iteration parameters
MAX_NEWTON_ITERS = 3
GRAD_TOL = 0.1      # stop when ||grad|| < this


def _softplus_inverse(y):
    return float(np.log(np.expm1(y)))


def _make_filter():
    """Build filter. Parameters will be updated per call via diff_model."""
    base_model = RangeBearingModel(
        sigma_range=PEAK_SIGMA_RANGE_INIT,
        sigma_bearing=PEAK_SIGMA_BEARING_INIT,
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


class TestRBPosteriorHessianAtPeak:
    def test_hessian(self):
        # Data at truth, same seed as HMC/MAP
        truth_model = RangeBearingModel(
            sigma_range=TRUE_SIGMA_RANGE,
            sigma_bearing=TRUE_SIGMA_BEARING,
            dtype=DTYPE,
        )
        rng = np.random.default_rng(DATA_SEED)
        _, _, obs = generate_data(truth_model, T=T, rng=rng)
        obs_tf = tf.constant(obs, dtype=DTYPE)

        diff_model, filt = _make_filter()

        prior = tfp.distributions.LogNormal(
            loc=tf.constant(PRIOR_LOC, dtype=DTYPE),
            scale=tf.constant(PRIOR_SCALE, dtype=DTYPE),
        )
        softplus_bij = tfp.bijectors.Softplus()

        def grad_at_q(q_np):
            """Return numerical gradient at q as np array."""
            q_var = tf.Variable(q_np, dtype=DTYPE)
            with tf.GradientTape() as tape:
                sigma = softplus_bij.forward(q_var)
                diff_model.update_parameters({
                    'sigma_range': sigma[0],
                    'sigma_bearing': sigma[1],
                })
                ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
                lp = tf.reduce_sum(prior.log_prob(sigma))
                log_jac = tf.reduce_sum(
                    softplus_bij.forward_log_det_jacobian(q_var, event_ndims=0)
                )
                log_post = ll + lp + log_jac
            g = tape.gradient(log_post, q_var).numpy()
            return g, float(log_post.numpy())

        q_peak = np.array([
            _softplus_inverse(PEAK_SIGMA_RANGE_INIT),
            _softplus_inverse(PEAK_SIGMA_BEARING_INIT),
        ], dtype=np.float64)

        print(f"\n  [RB posterior Hessian at peak]")
        print(f"  initial sigma = ({PEAK_SIGMA_RANGE_INIT}, {PEAK_SIGMA_BEARING_INIT})")
        print(f"  initial q     = ({q_peak[0]:.6f}, {q_peak[1]:.6f})")
        print(f"  FD step h  = {FD_H}")

        # Newton iteration: compute grad + Hessian at current estimate,
        # step q <- q - H^{-1} grad, repeat until ||grad|| < GRAD_TOL.
        H = None
        H_sym = None
        g0 = None
        logp0 = None
        for newton_iter in range(MAX_NEWTON_ITERS):
            # Gradient + log_post at current q
            g0, logp0 = grad_at_q(q_peak.astype(np.float32))
            grad_norm = float(np.linalg.norm(g0))
            print(f"\n  Newton iter {newton_iter}:")
            print(f"    q          = ({q_peak[0]:.6f}, {q_peak[1]:.6f})")
            sig_cur = [float(np.log1p(np.exp(q_peak[0]))), float(np.log1p(np.exp(q_peak[1])))]
            print(f"    sigma      = ({sig_cur[0]:.6f}, {sig_cur[1]:.6f})")
            print(f"    log_post   = {logp0:.4f}")
            print(f"    grad       = {g0}")
            print(f"    |grad|     = {grad_norm:.6f}")

            # Central FD Hessian at current q
            H = np.zeros((2, 2), dtype=np.float64)
            for j in range(2):
                q_plus = q_peak.copy(); q_plus[j] += FD_H
                q_minus = q_peak.copy(); q_minus[j] -= FD_H
                g_plus, _ = grad_at_q(q_plus.astype(np.float32))
                g_minus, _ = grad_at_q(q_minus.astype(np.float32))
                H[:, j] = (g_plus - g_minus) / (2.0 * FD_H)
            H_sym = 0.5 * (H + H.T)

            if grad_norm < GRAD_TOL:
                print(f"    converged (||grad|| < {GRAD_TOL})")
                break

            # Newton step: q_new = q - H^{-1} g
            try:
                dq = np.linalg.solve(H_sym, g0)
                q_peak = q_peak - dq
                print(f"    Newton step Δq = ({-dq[0]:+.5f}, {-dq[1]:+.5f})")
            except np.linalg.LinAlgError:
                print(f"    WARNING: Hessian singular, stopping")
                break
        else:
            print(f"\n  max Newton iterations ({MAX_NEWTON_ITERS}) reached, "
                  f"final ||grad||={np.linalg.norm(g0):.4f}")

        print(f"\n  Hessian (raw):")
        print(f"    {H[0, 0]:+12.4f}  {H[0, 1]:+12.4f}")
        print(f"    {H[1, 0]:+12.4f}  {H[1, 1]:+12.4f}")

        print(f"\n  Hessian (symmetrized):")
        print(f"    {H_sym[0, 0]:+12.4f}  {H_sym[0, 1]:+12.4f}")
        print(f"    {H_sym[1, 0]:+12.4f}  {H_sym[1, 1]:+12.4f}")

        # Posterior quantities
        eigenvals, eigenvecs = np.linalg.eigh(H_sym)
        print(f"  eigenvalues: {eigenvals}  (both should be negative at a mode)")

        # posterior covariance = -inv(H)
        try:
            post_cov = -np.linalg.inv(H_sym)
            post_std = np.sqrt(np.clip(np.diag(post_cov), 0.0, None))
            post_corr = post_cov[0, 1] / (post_std[0] * post_std[1] + 1e-30)
        except np.linalg.LinAlgError:
            post_cov = None
            post_std = None
            post_corr = None

        # Optimal mass: full = -H; diagonal isotropic = 1/diag(post_cov)
        mass_full = -H_sym
        mass_diag = 1.0 / np.clip(np.diag(post_cov), 1e-10, None) if post_cov is not None else None

        print(f"\n  Posterior std in q-space:        {post_std}")
        print(f"  Posterior correlation:           {post_corr:.4f}")
        print(f"\n  Optimal diagonal mass (isotropic per-dim): "
              f"{mass_diag}")
        print(f"  Optimal FULL mass matrix (-H_sym):")
        print(f"    {mass_full[0, 0]:+10.2f}  {mass_full[0, 1]:+10.2f}")
        print(f"    {mass_full[1, 0]:+10.2f}  {mass_full[1, 1]:+10.2f}")

        # Compare to current hand-picked [80, 80]
        current = np.array([80.0, 80.0])
        diag_diff = (mass_diag - current) / current * 100.0 if mass_diag is not None else None
        print(f"\n  Current mass_vector: [80.0, 80.0]")
        if mass_diag is not None:
            print(f"  Computed mass_diag:  [{mass_diag[0]:.2f}, {mass_diag[1]:.2f}]")
            print(f"  Relative diff from current (%): "
                  f"[{diag_diff[0]:+.1f}, {diag_diff[1]:+.1f}]")

        # Save
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / 'rb_posterior_hessian_at_peak.json'
        payload = {
            'case_name': 'RB posterior Hessian at peak',
            'model': 'range_bearing',
            'filter': 'ledh_invertible_hmc',
            'T': T,
            'data_seed': DATA_SEED,
            'pf_seed': [42, 0],
            'n_particles': N_PARTICLES,
            'n_lambda_steps': N_LAMBDA_STEPS,
            'ot_epsilon': OT_EPSILON,
            'peak_sigma_range_init': PEAK_SIGMA_RANGE_INIT,
            'peak_sigma_bearing_init': PEAK_SIGMA_BEARING_INIT,
            'peak_q_converged': q_peak.tolist(),
            'peak_sigma_converged': [float(np.log1p(np.exp(q_peak[0]))),
                                      float(np.log1p(np.exp(q_peak[1])))],
            'fd_h': FD_H,
            'grad_at_peak': g0.tolist(),
            'log_post_at_peak': logp0,
            'hessian_raw': H.tolist(),
            'hessian_sym': H_sym.tolist(),
            'eigenvalues': eigenvals.tolist(),
            'posterior_cov_q': post_cov.tolist() if post_cov is not None else None,
            'posterior_std_q': post_std.tolist() if post_std is not None else None,
            'posterior_correlation': float(post_corr) if post_corr is not None else None,
            'optimal_mass_diag_isotropic_per_dim': mass_diag.tolist() if mass_diag is not None else None,
            'optimal_mass_full': mass_full.tolist(),
            'current_mass_vector': current.tolist(),
            'relative_diff_from_current_percent': diag_diff.tolist() if diag_diff is not None else None,
        }
        with out_path.open('w') as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {out_path}")
