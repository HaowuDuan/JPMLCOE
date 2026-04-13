"""Quality test for the trained neural operator: minimal sanity test.

Builds ONE held-out test cloud designed to be the smallest possible
non-trivial transport problem. If the operator cannot improve on identity
on this single cloud, the architecture or the training is broken and there
is no point running anything harder.

The cloud is:
  - 200 particles equally spaced on `[-A, A]` with `A = 5`
    (so the inter-particle spacing is `Δ = 10/199 ≈ 0.05`).
  - Weights `w_i ∝ exp(-(x_i − μ)² / (2 σ_w²))` with `μ = 1.0` and
    `σ_w = 20.0`. Because `σ_w >> A`, the Gaussian is nearly flat over
    the particle interval — the weights are barely skewed and the
    target distribution is barely shifted from the source. The transport
    map should be close to identity, with a small offset.

The KDE bandwidth is computed by the resampler using the resolution
policy: `h = 5 × mean_nearest_neighbor_distance ≈ 0.25`. Independent
of the weights, set only by particle spacing.

The assertion is binary: T's mean error must be less than half the
identity mean error. This tests whether the operator does *anything* in
the right direction. It is not a quality bar.

Run: pytest code/neural_operator/tests/test_resampler_quality.py -v -s
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from train import NeuralOperator, train
from resampling import NeuralOperatorResampler
from _result_utils import save_result, reset_results


DTYPE = tf.float64


def setup_module():
    reset_results(__file__)


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


def make_minimal_test_cloud(n=200, A=5.0, mu=1.0, sigma_w=20.0):
    """Build the minimal sanity test cloud.

    Args:
        n: number of particles (default 200)
        A: half-width of the particle interval (default 5 → particles on [-5, 5])
        mu: center of the Gaussian weight (default 1.0 — must be ≠ 0 so that
            the weighted mean differs from the unweighted mean and identity
            has a finite relative error to compare against)
        sigma_w: width of the Gaussian weight (default 20, much larger than A
            so the Gaussian is nearly flat over the particle interval and
            the transport is barely perturbed from identity)

    Returns:
        (particles, weights) as (n, 1) and (n,) tf.float64 tensors.
    """
    x = np.linspace(-A, A, n).reshape(-1, 1)
    log_w = -0.5 * ((x[:, 0] - mu) ** 2) / (sigma_w ** 2)
    log_w = log_w - log_w.max()  # numerical stability
    w = np.exp(log_w)
    w = w / w.sum()
    return (
        tf.constant(x, dtype=tf.float64),
        tf.constant(w, dtype=tf.float64),
    )


HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, 'results', 'checkpoints', 'operator_v2')
TRACE_DIR = os.path.join(HERE, 'results', 'traces')


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


def _train_and_save(trace_label='baseline'):
    """Train a fresh operator for 2000 steps and save weights to CHECKPOINT_DIR.

    Also writes a per-step CSV trace to TRACE_DIR/{trace_label}.csv with
    bandwidth ingredients (h, n_eff, weighted_std, unweighted_std) and the
    usual loss / grad-norm columns. Pure logging.
    """
    print("--- Training 2000 steps (no checkpoint found) ---")
    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = os.path.join(TRACE_DIR, f'{trace_label}.csv')
    print(f"--- Writing per-step trace to {trace_path} ---")
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
        trace_csv_path=trace_path,
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
    print("\n=== NeuralOperatorResampler minimal sanity test ===\n")
    tf.random.set_seed(2024)

    A = 5.0
    mu = 1.0
    sigma_w = 20.0
    p, w = make_minimal_test_cloud(n=200, A=A, mu=mu, sigma_w=sigma_w)

    # Cloud diagnostics
    N = int(p.shape[0])
    ess = float(1.0 / tf.reduce_sum(w ** 2).numpy())
    delta = float(2 * A / (N - 1))
    mu_w = float(tf.reduce_sum(w[:, None] * p, axis=0).numpy()[0])
    mu_unweighted = float(tf.reduce_mean(p).numpy())
    print("--- Test cloud ---")
    print(f"  N         = {N}")
    print(f"  interval  = [-{A}, {A}]")
    print(f"  spacing Δ = {delta:.4f}")
    print(f"  weight    = exp(-(x-{mu})²/(2·{sigma_w}²))")
    print(f"  ESS       = {ess:.2f} (ESS/N = {ess/N:.3f})")
    print(f"  unweighted mean = {mu_unweighted:+.4f}")
    print(f"  weighted mean   = {mu_w:+.4f}")
    print(f"  expected h ≈ {5*delta:.4f} (resolution rule, 5·Δ)")
    print()

    model = _load_or_train()

    resampler = NeuralOperatorResampler(model)
    result = resampler(p, w, seed=tf.constant([0, 0], dtype=tf.int32))

    m_id, c_id = _moment_err_rel(p, w, p)
    m_T, c_T = _moment_err_rel(p, w, result.particles)
    mu_T = float(tf.reduce_mean(result.particles).numpy())

    print("--- Result ---")
    print(f"  identity mean_err = {m_id:.4f}  (output mean = {mu_unweighted:+.4f})")
    print(f"  T        mean_err = {m_T:.4f}  (output mean = {mu_T:+.4f})")
    print(f"  identity cov_err  = {c_id:.4f}")
    print(f"  T        cov_err  = {c_T:.4f}")
    print()

    threshold = 0.5 * m_id
    passed = bool(m_T < threshold)

    print(f"--- Hard assertion: T mean_err < 0.5 × identity mean_err ---")
    print(f"  threshold = {threshold:.4f}")
    print(f"  T value   = {m_T:.4f}")

    save_result(__file__, {
        'case_name': 'minimal_sanity_test',
        'description': 'Equally-spaced particles, Gaussian weights, near-identity transport',
        'config': {
            'n': N,
            'A': A,
            'mu': mu,
            'sigma_w': sigma_w,
            'delta': delta,
        },
        'cloud': {
            'ess': ess,
            'ess_over_n': ess / N,
            'weighted_mean': mu_w,
            'unweighted_mean': mu_unweighted,
        },
        'metrics': {
            'identity_mean_err': m_id,
            'T_mean_err': m_T,
            'identity_cov_err': c_id,
            'T_cov_err': c_T,
            'T_output_mean': mu_T,
        },
        'threshold': threshold,
        'passed': passed,
    })

    assert passed, (
        f"Minimal sanity test FAILED: identity mean_err = {m_id:.4f}, "
        f"T mean_err = {m_T:.4f}, threshold = {threshold:.4f}. "
        f"The operator is not moving the mean toward the weighted target "
        f"on the simplest possible cloud (nearly-identity transport). "
        f"Architecture or training is broken."
    )

    print("\n=== Minimal sanity test passed ===\n")
