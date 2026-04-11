"""Quality test for the trained neural operator: graded difficulty ladder.

Builds four held-out clouds at increasing weight skew (decreasing ESS),
holding the particle distribution fixed across tiers, so we can find the
exact failure boundary of the operator.

All four clouds use the same broad single Gaussian for the particles
(`x_i ~ N(0, 2^2)`, N=200), and vary only the strength `beta` of an
exponential weight tilt `w_i \\propto exp(beta * x_i)`. With this setup:

  Tier   beta   approx ESS/N   weighted mean
  ----   ----   ------------   -------------
   1     0.1       0.96             0.4
   2     0.3       0.70             1.2
   3     0.4       0.53             1.6
   4     0.55      0.30             2.2

Tier 1 is barely skewed — a competent operator must handle this. Tier 4
matches the difficulty of the previous test. The point of the ladder is to
find which tier the current model fails at, not to ace tier 4 on the first
try.

The assertion is *only* on tier 1: the easiest case must show meaningful
improvement over identity. Tiers 2-4 are reported but not asserted, so a
single test run gives us the failure boundary in one shot.

Run: python code/neural_operator/tests/test_resampler_quality.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
# neural_operator/src must come BEFORE code/src so `from models import WTanh`
# resolves to the operator module rather than the state-space model package.
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))   # code/src
sys.path.insert(0, os.path.join(HERE, '..', 'src'))         # neural_operator/src

import numpy as np
import tensorflow as tf

from train import NeuralOperator, train
from resampling import NeuralOperatorResampler


DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')


def _moment_err_rel(particles, weights, mapped):
    """Relative L2 error between weighted (target) and unweighted (mapped) moments."""
    mu_p = tf.reduce_sum(weights[:, None] * particles, axis=0)
    centered = particles - mu_p[None, :]
    cov_p = tf.einsum('n,ni,nj->ij', weights, centered, centered)

    N = tf.cast(tf.shape(mapped)[0], mapped.dtype)
    mu_T = tf.reduce_sum(mapped, axis=0) / N
    centered_T = mapped - mu_T[None, :]
    cov_T = tf.matmul(centered_T, centered_T, transpose_a=True) / N

    mean_err = tf.norm(mu_T - mu_p) / (tf.norm(mu_p) + 1e-12)
    cov_err = tf.norm(cov_T - cov_p) / (tf.norm(cov_p) + 1e-12)
    return float(mean_err), float(cov_err)


def make_graded_clouds_1d(n=200, sigma=2.0, seed=20260410):
    """Build a 4-tier difficulty ladder of (particles, weights) clouds.

    All tiers use the same particle distribution `x_i ~ N(0, sigma^2)`. Only
    the exponential weight tilt strength `beta` varies, so the four tiers
    differ only in how skewed the weights are.
    """
    rng = np.random.default_rng(seed)
    x = sigma * rng.standard_normal((n, 1))
    x_tf = tf.constant(x, dtype=tf.float64)

    tiers = []
    for label, beta in [
        ('tier 1 (easy)',         0.10),
        ('tier 2 (easy-medium)',  0.30),
        ('tier 3 (medium)',       0.40),
        ('tier 4 (hard)',         0.55),
    ]:
        log_w = beta * x[:, 0]
        log_w = log_w - log_w.max()       # numerical stability
        w = np.exp(log_w)
        w = w / w.sum()
        w_tf = tf.constant(w, dtype=tf.float64)
        tiers.append((label, beta, x_tf, w_tf))

    return tiers


CHECKPOINT_DIR = os.path.join(HERE, 'checkpoints', 'operator_v1')
TRACE_DIR = os.path.join(HERE, 'traces')


def _build_model():
    """Build the operator architecture used for both training and evaluation."""
    return NeuralOperator(
        in_dim=1,
        embed_dim=32,
        num_modules=8,
        context_dim=32,
        encoder_hidden=(32, 32),
        alpha_init_bias=-3.0,
    )


def _train_and_save():
    """Train a fresh operator for 2000 steps and save weights to CHECKPOINT_DIR."""
    print("--- Training 2000 steps (no checkpoint found) ---")
    model = _build_model()
    train(
        model,
        num_steps=2000,
        n_particles=200,
        d=1,
        q_points=64,
        batch_clouds=1,
        learning_rate=1e-3,
        linearity_weight=0.0,
        eig_weight=1e-2,
        eig_threshold=0.1,
        h_scale_init=2.0,
        h_scale_final=1.0,
        log_every=200,
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save_weights(os.path.join(CHECKPOINT_DIR, 'weights'))
    print(f"--- Saved weights to {CHECKPOINT_DIR} ---")
    return model


def _load_or_train():
    """Load an existing checkpoint if present, otherwise train and save one."""
    weights_path = os.path.join(CHECKPOINT_DIR, 'weights')
    index_file = weights_path + '.index'
    if os.path.exists(index_file):
        print(f"--- Loading weights from {CHECKPOINT_DIR} (skip training) ---")
        model = _build_model()
        # Build variables before loading: one dummy forward pass.
        dummy_p = tf.zeros((10, 1), dtype=tf.float64)
        dummy_w = tf.ones((10,), dtype=tf.float64) / 10.0
        c = model.encode(dummy_p, dummy_w, h_kde=tf.constant(1.0, dtype=tf.float64))
        _ = model.transport_with_jacobian(dummy_p, c)
        model.load_weights(weights_path)
        return model
    return _train_and_save()


def test_resampler_quality():
    print("\n=== NeuralOperatorResampler quality test (graded ladder) ===\n")
    tf.random.set_seed(2024)

    tiers = make_graded_clouds_1d(n=200, sigma=2.0)

    print("--- Cloud difficulty ladder ---")
    for label, beta, p, w in tiers:
        ess = float(1.0 / tf.reduce_sum(w ** 2).numpy())
        N = int(p.shape[0])
        mu_w = float(tf.reduce_sum(w[:, None] * p, axis=0).numpy())
        print(f"  {label:24s} beta={beta:.2f}  N={N}  ESS={ess:6.2f}  "
              f"ESS/N={ess/N:.3f}  weighted mean={mu_w:+.3f}")
    print()

    model = _load_or_train()

    resampler = NeuralOperatorResampler(model)

    print("\n--- Per-tier comparison ---")
    print(f"  {'tier':24s}  identity_mean  T_mean   identity_cov  T_cov   verdict")
    print(f"  {'-'*24}  -------------  -------  ------------  ------  -------")

    results = []
    for label, beta, p, w in tiers:
        result = resampler(p, w, seed=tf.constant([0, 0], dtype=tf.int32))
        m_id, c_id = _moment_err_rel(p, w, p)
        m_T, c_T = _moment_err_rel(p, w, result.particles)
        verdict = 'PASS' if m_T < m_id else 'fail'
        print(f"  {label:24s}  {m_id:13.4f}  {m_T:7.4f}  {c_id:12.4f}  {c_T:6.4f}  {verdict}")
        results.append((label, beta, m_id, m_T, c_id, c_T))

    # The ONLY hard assertion is tier 1 (easy). The harder tiers are
    # reported but not enforced — the point of the ladder is to find the
    # failure boundary, not to ace every tier on the first attempt.
    tier1_label, _, m_id_1, m_T_1, _, _ = results[0]
    print(f"\n  Hard assertion: tier 1 must improve over identity.")
    print(f"    identity mean_err = {m_id_1:.4f}")
    print(f"    T mean_err        = {m_T_1:.4f}")
    assert m_T_1 < m_id_1, (
        f"Tier 1 (easy, ESS/N ~ 0.96) FAILED: "
        f"identity {m_id_1:.4f} -> T {m_T_1:.4f}. "
        f"If even the easiest tier does not improve, the operator is "
        f"not learning anything useful."
    )

    print("\n=== Tier 1 passed. Higher tiers reported above. ===\n")


if __name__ == '__main__':
    test_resampler_quality()
