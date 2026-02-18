"""Unit tests for range-bearing DPF gradient and filter diagnostics.

These tests run without full HMC — they evaluate the gradient once or run
the filter forward pass. Fast enough for CI.

Debug plan (maps to the debug printouts that were in the production code):

  Step 1 — test_gradient_finite_at_true_params
      Mirrors: "grad_sigma_range=X grad_sigma_bearing=Y" at truth.
      Catches: disconnected graph, NaN at the easy point.

  Step 2 — test_ess_does_not_collapse
      Mirrors: "ESS=Y" per-timestep printout in _run_eager.
      Catches: weight collapse (ESS→1) at wrong params.

  Step 3 — test_max_log_theta_bounded
      Mirrors: "max_log_theta=Z" per-timestep printout in _run_eager.
      Catches: Jacobian accumulation overflow. If max_log_theta > 100,
      exp(max_log_theta) > 10^43 → backward pass explodes.

  Step 4 — test_gradient_magnitude_reasonable
      Mirrors: "|grad|=Y d/d_sigma_range=A d/d_sigma_bearing=B" in hmc_runner.
      Catches: gradient explosion at the starting point before HMC even starts.
      The failing Kitagawa run showed |grad|=1.9M here.

  Step 5 — test_filter_log_lik_positive_at_true_params
      Sanity check: true params must score better than wrong params.
      Catches: broken forward pass (wrong R, wrong observation model).

  Step 6 — test_per_timestep_log_lik_finite
      Mirrors: "log_lik_t=X" per-timestep printout.
      Catches: NaN propagation from a single bad timestep.

  Step 7 — test_ess_and_theta_per_timestep
      Mirrors: full per-timestep debug line combining ESS and max_log_theta.
      Catches: concurrent collapse of both diagnostics, which together
      guarantee gradient explosion.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np
import tensorflow as tf

from src.models.range_bearing import RangeBearingModel
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC


def _make_filter(model, n_lambda_steps=10, on_timestep=None):
    return LEDHParticleFlowFilterHMC(
        model,
        n_particles=100,
        n_lambda_steps=n_lambda_steps,
        weight_clip_range=50.0,
        stop_gradient_resampling=True,
        eager_mode=True,
        on_timestep=on_timestep,
    )


def _compute_log_lik_and_grad(model, observations,
                               param_names=('sigma_range', 'sigma_bearing')):
    """Compute log-likelihood and gradient w.r.t. trainable params.

    Replicates exactly what DPFRunner._negative_log_posterior does:
    set model attributes to plain tensors (tf.identity), then differentiate.
    """
    param_tensors = [
        tf.Variable(float(getattr(model, name)), dtype=tf.float32)
        for name in param_names
    ]
    filter_obj = _make_filter(model)

    with tf.GradientTape() as tape:
        tape.watch(param_tensors)
        for name, tensor in zip(param_names, param_tensors):
            setattr(model, name, tf.identity(tensor))
        seed = tf.constant([42, 0], dtype=tf.int32)
        log_lik = filter_obj.log_marginal_likelihood_tf(observations, seed=seed)

    grads = tape.gradient(log_lik, param_tensors)
    return log_lik, grads


# ─────────────────────────────────────────────────────────────────────────── #
# Step 1 — gradient connected and finite at true params
# ─────────────────────────────────────────────────────────────────────────── #

def test_gradient_finite_at_true_params(short_observations):
    """Gradient at sigma=(0.1, 0.1) is finite and at least one nonzero.

    If this fails the problem is structural — the gradient graph is broken
    regardless of parameter values.
    """
    model = RangeBearingModel(sigma_range=0.1, sigma_bearing=0.1)
    log_lik, (grad_r, grad_b) = _compute_log_lik_and_grad(model, short_observations)

    # Match original debug format: "d/d_sigma_range=X d/d_sigma_bearing=Y"
    print(f"\n  log_lik={log_lik.numpy():.4f}  "
          f"d/d_sigma_range={grad_r.numpy() if grad_r is not None else 'None':.6f}  "
          f"d/d_sigma_bearing={grad_b.numpy() if grad_b is not None else 'None':.6f}")

    assert grad_r is not None, "grad_sigma_range is None — graph disconnected"
    assert grad_b is not None, "grad_sigma_bearing is None — graph disconnected"
    assert tf.math.is_finite(grad_r), f"grad_sigma_range={grad_r.numpy()} not finite"
    assert tf.math.is_finite(grad_b), f"grad_sigma_bearing={grad_b.numpy()} not finite"
    assert tf.abs(grad_r) > 1e-10 or tf.abs(grad_b) > 1e-10, \
        "Both gradients are exactly zero at true params"


# ─────────────────────────────────────────────────────────────────────────── #
# Step 2 — ESS does not collapse at wrong params
# ─────────────────────────────────────────────────────────────────────────── #

def test_ess_does_not_collapse(short_observations, timestep_collector):
    """ESS stays above 5% of n_particles (100) at sigma=(0.3, 0.3).

    Directly mirrors the "ESS=Y" printout from _run_eager. If ESS→1 here,
    weight collapse is the root cause of gradient explosion.
    """
    model = RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)
    filter_obj = _make_filter(model, on_timestep=timestep_collector['callback'])

    seed = tf.constant([42, 0], dtype=tf.int32)
    filter_obj.log_marginal_likelihood_tf(short_observations, seed=seed)

    ess_vals = timestep_collector['ess_vals']
    assert len(ess_vals) > 0, "on_timestep was never called"

    min_ess = min(ess_vals)
    print(f"\n  ESS per timestep: {[f'{e:.1f}' for e in ess_vals]}")
    print(f"  min_ESS={min_ess:.1f}  mean_ESS={np.mean(ess_vals):.1f}")

    assert min_ess > 5.0, (
        f"ESS collapsed to {min_ess:.1f} (n_particles=100) — "
        "weight collapse confirmed as root cause of gradient explosion"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Step 3 — Jacobian accumulation bounded (max_log_theta)
# ─────────────────────────────────────────────────────────────────────────── #

def test_max_log_theta_bounded(short_observations, timestep_collector):
    """max_log_theta < 100 at every timestep.

    Directly mirrors the "max_log_theta=Z" printout from _run_eager.
    max_log_theta is the maximum log-det Jacobian across particles BEFORE
    normalization (i.e. the raw scale of the Jacobian accumulation over
    n_lambda_steps). If this exceeds ~100, exp(max_log_theta) overflows
    float32 (max ~3×10^38 = exp(87.5)), corrupting the backward pass.

    At true params expect max_log_theta O(1). At wrong params, if the flow
    pushes particles far, max_log_theta can grow large.
    """
    model = RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)
    filter_obj = _make_filter(model, on_timestep=timestep_collector['callback'])

    seed = tf.constant([42, 0], dtype=tf.int32)
    filter_obj.log_marginal_likelihood_tf(short_observations, seed=seed)

    thetas = timestep_collector['max_log_thetas']
    assert len(thetas) > 0, "on_timestep was never called"

    max_theta = max(thetas)
    print(f"\n  max_log_theta per timestep: {[f'{t:.2f}' for t in thetas]}")
    print(f"  global max_log_theta = {max_theta:.2f}")

    assert max_theta < 100.0, (
        f"max_log_theta={max_theta:.2f} — Jacobian accumulation overflow, "
        "backward pass will explode even if ESS looks okay"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Step 4 — gradient magnitude at starting point
# ─────────────────────────────────────────────────────────────────────────── #

def test_gradient_magnitude_reasonable(short_observations):
    """Initial |grad| < 1e6 at sigma=(0.3, 0.3).

    Mirrors the hmc_runner debug line:
      "[GRAD] nlp=X |grad|=Y d/d_sigma_range=A d/d_sigma_bearing=B"
    at the very first gradient evaluation (before any HMC step).

    The failing Kitagawa run showed |grad|=1.9M here. Threshold 1e6 already
    flags explosion — a healthy gradient for a 2-param problem is O(1)–O(100).
    """
    model = RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)
    log_lik, (grad_r, grad_b) = _compute_log_lik_and_grad(model, short_observations)

    nlp = -log_lik.numpy()

    if grad_r is None or grad_b is None:
        pytest.fail("gradient is None at wrong params — graph disconnected")

    grad_norm = float(tf.norm(tf.stack([grad_r, grad_b])).numpy())

    # Exactly match debug format from hmc_runner.py
    print(f"\n  nlp={nlp:.4f}  |grad|={grad_norm:.4e}  "
          f"d/d_sigma_range={float(grad_r.numpy()):.6f}  "
          f"d/d_sigma_bearing={float(grad_b.numpy()):.6f}")

    assert np.isfinite(nlp), f"nlp={nlp} — log posterior is NaN/Inf at starting point"
    assert grad_norm < 1e6, (
        f"|grad|={grad_norm:.2e} ≥ 1e6 at starting point — "
        "gradient explosion before HMC even starts (likely weight collapse or Jacobian overflow)"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Step 5 — log-lik ordering (identifiability)
# ─────────────────────────────────────────────────────────────────────────── #

def test_filter_log_lik_positive_at_true_params(short_observations):
    """True params score higher log p(y|θ) than wrong params.

    Prerequisite for everything else: if the filter can't distinguish
    true from wrong params, HMC has nothing to converge to.
    """
    model_true = RangeBearingModel(sigma_range=0.1, sigma_bearing=0.1)
    model_wrong = RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)

    filt_true = _make_filter(model_true)
    filt_wrong = _make_filter(model_wrong)

    seed = tf.constant([42, 0], dtype=tf.int32)
    ll_true = filt_true.log_marginal_likelihood_tf(short_observations, seed=seed).numpy()
    ll_wrong = filt_wrong.log_marginal_likelihood_tf(short_observations, seed=seed).numpy()

    print(f"\n  ll(true)={ll_true:.4f}  ll(wrong)={ll_wrong:.4f}  "
          f"diff={ll_true - ll_wrong:.4f}")

    assert ll_true > ll_wrong, (
        f"True params score {ll_true:.4f} ≤ wrong params {ll_wrong:.4f} — "
        "model is not identifiable or R property is broken"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Step 6 — per-timestep log-lik finite
# ─────────────────────────────────────────────────────────────────────────── #

def test_per_timestep_log_lik_finite(short_observations, timestep_collector):
    """All per-timestep log p(y_t | y_{1:t-1}) are finite at wrong params.

    Mirrors "log_lik_t=X" printout. A single NaN propagates into total_log_lik
    → NaN gradient → 0% HMC acceptance.
    """
    model = RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)
    filter_obj = _make_filter(model, on_timestep=timestep_collector['callback'])

    seed = tf.constant([42, 0], dtype=tf.int32)
    filter_obj.log_marginal_likelihood_tf(short_observations, seed=seed)

    log_liks = timestep_collector['log_liks']
    ts = timestep_collector['ts']
    assert len(log_liks) > 0, "on_timestep was never called"

    print(f"\n  log_lik_t per timestep: {[f't={t}:{ll:.2f}' for t, ll in zip(ts, log_liks)]}")

    bad = [(t, ll) for t, ll in zip(ts, log_liks) if not np.isfinite(ll)]
    assert len(bad) == 0, f"Non-finite log p(y_t|...) at timesteps: {bad}"


# ─────────────────────────────────────────────────────────────────────────── #
# Step 7 — ESS and max_log_theta jointly (full debug line)
# ─────────────────────────────────────────────────────────────────────────── #

def test_ess_and_theta_per_timestep(short_observations, timestep_collector):
    """Print the full per-timestep debug line and assert both ESS and theta.

    Mirrors the original _run_eager debug printout:
      "[t=T] log_lik_t=X cumul=Y ESS=Z max_log_theta=W"

    Gradient explosion requires BOTH conditions:
      ESS → 1 (degenerate weights) AND max_log_theta large (Jacobian overflow).
    If only one fails, there may be another root cause.
    """
    model = RangeBearingModel(sigma_range=0.3, sigma_bearing=0.3)
    filter_obj = _make_filter(model, on_timestep=timestep_collector['callback'])

    seed = tf.constant([42, 0], dtype=tf.int32)
    total_ll = filter_obj.log_marginal_likelihood_tf(short_observations, seed=seed)

    d = timestep_collector
    assert len(d['ts']) > 0, "on_timestep was never called"

    cumulative = np.cumsum(d['log_liks'])
    print("\n  [t] log_lik_t   cumul    ESS    max_log_theta")
    for t, ll, cum, ess, theta in zip(
            d['ts'], d['log_liks'], cumulative, d['ess_vals'], d['max_log_thetas']):
        print(f"  [{t:3d}] {ll:+.4f}  {cum:+.4f}  {ess:6.1f}  {theta:+.4f}")

    min_ess = min(d['ess_vals'])
    max_theta = max(d['max_log_thetas'])

    # Both conditions must hold for gradient to be well-behaved
    ess_ok = min_ess > 5.0
    theta_ok = max_theta < 100.0

    if not ess_ok and not theta_ok:
        pytest.fail(
            f"BOTH collapsed: min_ESS={min_ess:.1f} AND max_log_theta={max_theta:.2f} — "
            "classic gradient explosion signature"
        )
    elif not ess_ok:
        pytest.fail(f"ESS collapsed to {min_ess:.1f} — weight collapse")
    elif not theta_ok:
        pytest.fail(f"max_log_theta={max_theta:.2f} — Jacobian overflow")
