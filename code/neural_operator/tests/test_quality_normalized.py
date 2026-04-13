"""Diagnostic: does per-cloud normalization stabilize training on random clouds?

Same setup as test_resampler_quality.py, but each training cloud is
normalized to zero mean and unit std before entering the operator:

    x_hat = (x - mean(x)) / (std(x) + eps)

The encoder sees x_hat and the original weights. The KDE, loss, and
query points are all computed in normalized coordinates. The bandwidth
is computed from x_hat (approximately constant because all clouds now
have unit spread).

Two benefits of normalization:
1. Bandwidth h is approximately constant across clouds → stable loss
   stiffness → no gradient spikes.
2. Pre-activations Wx stay O(1) with Glorot init → tanh stays in its
   linear regime → all modules stay active (see
   issues_to_be_addressed/1_initialization.md).

At inference, the output is un-normalized:
    x_new = mean + std * T(x_hat)

This test uses a LOCAL copy of the training loop. The production
train.py is NOT modified.

Run: pytest code/neural_operator/tests/test_quality_normalized.py -v -s
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from train import NeuralOperator
from losses import ma_residual_loss
from kde import (
    compute_bandwidth_scalar,
    sample_from_uniform_kde,
)
from data import sample_random_cloud
from _result_utils import save_result, reset_results


DTYPE = tf.float64


def setup_module():
    reset_results(__file__)


def _normalize_cloud(particles):
    """Normalize particles to zero mean, unit std. Returns (x_hat, mu, s)."""
    mu = tf.reduce_mean(particles, axis=0)
    s = tf.math.reduce_std(particles) + tf.constant(1e-8, dtype=particles.dtype)
    x_hat = (particles - mu[None, :]) / s
    return x_hat, mu, s


def _unnormalize(T_hat, mu, s):
    """Map T(x_hat) back to original coordinates."""
    return mu[None, :] + s * T_hat


def _moment_err_rel(particles, weights, mapped):
    mu_p = tf.reduce_sum(weights[:, None] * particles, axis=0)
    N = tf.cast(tf.shape(mapped)[0], mapped.dtype)
    mu_T = tf.reduce_sum(mapped, axis=0) / N
    mean_err = tf.norm(mu_T - mu_p) / (tf.norm(mu_p) + 1e-12)
    return float(mean_err)


def _train_normalized(model, num_steps=2000, n_particles=200, d=1,
                       q_points=64, learning_rate=1e-3, log_every=200,
                       base_seed=0):
    """Train on random clouds with per-cloud normalization. JIT graph mode."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Warmup
    dummy_p = tf.zeros((n_particles, d), dtype=tf.float64)
    dummy_w = tf.ones((n_particles,), dtype=tf.float64) / float(n_particles)
    _c = model.encode(dummy_p, dummy_w, h_kde=tf.constant(0.25, dtype=tf.float64))
    _ = model.transport_with_jacobian(dummy_p, _c)
    optimizer.build(model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def _compiled_step(x_hat, weights, h, query_points):
        with tf.GradientTape() as tape:
            c = model.encode(x_hat, weights, h_kde=h)
            loss_ma, diag = ma_residual_loss(
                model.trunk, x_hat, weights, h, query_points, c
            )
        grads = tape.gradient(loss_ma, model.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, model.trainable_variables)]
        grads, grad_norm = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        lambda_pos = model.trunk._lambda_pos(c)
        return {
            'loss': loss_ma,
            'grad_norm': grad_norm,
            'lambda_pos': lambda_pos,
            'residual_abs_mean': diag['residual_abs_mean'],
        }

    import time
    t0 = time.perf_counter()
    history = []

    for step in range(num_steps):
        seed = tf.constant([base_seed + step, 0], dtype=tf.int32)
        seeds = tf.random.experimental.stateless_split(seed, num=2)

        # Sample a random cloud
        particles, weights = sample_random_cloud(n_particles, d, seeds[0])

        # Normalize to zero mean, unit std
        x_hat, _, _ = _normalize_cloud(particles)

        # Fixed bandwidth in normalized space. All clouds have unit std
        # after normalization, so h depends only on N.
        # Silverman for unit-variance Gaussian: h = N^(-1/5) ≈ 0.35 for N=200.
        h = tf.constant(0.35, dtype=tf.float64)

        # Query points in normalized space
        q_seed = tf.random.experimental.stateless_split(seeds[1], num=2)[0]
        query_points = sample_from_uniform_kde(x_hat, h, q_points, q_seed)

        diag = _compiled_step(x_hat, weights, h, query_points)

        row = {
            'step': step,
            'loss': float(diag['loss'].numpy()),
            'grad_norm': float(diag['grad_norm'].numpy()),
            'residual_abs_mean': float(diag['residual_abs_mean'].numpy()),
            'lambda_pos': float(diag['lambda_pos'].numpy()),
            'h': float(h.numpy()),
        }
        history.append(row)

        if step % log_every == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:5d}/{num_steps} | "
                f"loss={row['loss']:.4e} | "
                f"|grad|={row['grad_norm']:.2f} | "
                f"residual_abs={row['residual_abs_mean']:.4f} | "
                f"λ={row['lambda_pos']:.4f} | "
                f"h={row['h']:.4f} | "
                f"{elapsed:.1f}s"
            )

    return history


def make_minimal_test_cloud(n=200, A=5.0, mu=1.0, sigma_w=20.0):
    """Same held-out test cloud as test_resampler_quality.py."""
    x = np.linspace(-A, A, n).reshape(-1, 1)
    log_w = -0.5 * ((x[:, 0] - mu) ** 2) / (sigma_w ** 2)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w = w / w.sum()
    return (
        tf.constant(x, dtype=tf.float64),
        tf.constant(w, dtype=tf.float64),
    )


def test_quality_normalized():
    print("\n=== Quality test with per-cloud normalization ===\n")
    tf.random.set_seed(2024)

    model = NeuralOperator(
        in_dim=1,
        embed_dim=32,
        num_modules=8,
        context_dim=32,
        encoder_hidden=(32, 32),
        alpha_init_bias=-3.0,
    )

    print("--- Training 2000 steps on normalized random clouds ---")
    history = _train_normalized(
        model,
        num_steps=2000,
        n_particles=200,
        d=1,
        q_points=64,
        learning_rate=1e-3,
        log_every=200,
    )

    # Training stability metrics
    initial_loss = history[0]['loss']
    final_loss = history[-1]['loss']
    min_loss = min(r['loss'] for r in history)
    max_grad_late = max(r['grad_norm'] for r in history[-500:])
    final_lambda = history[-1]['lambda_pos']
    h_values = [r['h'] for r in history]
    h_min, h_max = min(h_values), max(h_values)

    print(f"\n  initial_loss = {initial_loss:.4e}")
    print(f"  final_loss   = {final_loss:.4e}")
    print(f"  min_loss     = {min_loss:.4e}")
    print(f"  max_grad (last 500) = {max_grad_late:.2f}")
    print(f"  final λ      = {final_lambda:.4f}")
    print(f"  h range      = [{h_min:.4f}, {h_max:.4f}]")

    # Evaluate on the held-out cloud (must normalize it too)
    p_raw, w = make_minimal_test_cloud(n=200, A=5.0, mu=1.0, sigma_w=20.0)
    x_hat, mu_cloud, s_cloud = _normalize_cloud(p_raw)

    h_eval = tf.constant(0.35, dtype=tf.float64)  # same fixed h as training
    c = model.encode(x_hat, w, h_kde=h_eval)
    T_hat, J_hat = model.transport_with_jacobian(x_hat, c)

    # Un-normalize
    T_raw = _unnormalize(T_hat, mu_cloud, s_cloud)

    m_id = _moment_err_rel(p_raw, w, p_raw)
    m_T = _moment_err_rel(p_raw, w, T_raw)
    mu_T = float(tf.reduce_mean(T_raw).numpy())
    mu_w = float(tf.reduce_sum(w[:, None] * p_raw, axis=0).numpy()[0])

    print(f"\n--- Held-out cloud (un-normalized) ---")
    print(f"  weighted mean   = {mu_w:+.4f}")
    print(f"  identity mean   = +0.0000")
    print(f"  T output mean   = {mu_T:+.4f}")
    print(f"  identity mean_err = {m_id:.4f}")
    print(f"  T mean_err        = {m_T:.4f}")

    # Checks
    loss_decreased = bool(final_loss < initial_loss)
    h_stable = bool(h_max / max(h_min, 1e-10) < 3.0)
    grad_stable = bool(max_grad_late < 100.0)
    quality_improved = bool(m_T < m_id)

    print(f"\n  loss decreased:     {'YES' if loss_decreased else 'NO'}")
    print(f"  h stable (<3x):    {'YES' if h_stable else 'NO'} (range {h_min:.4f}–{h_max:.4f})")
    print(f"  grads stable:       {'YES' if grad_stable else 'NO'} (max late = {max_grad_late:.2f})")
    print(f"  T beats identity:   {'YES' if quality_improved else 'NO'}")

    save_result(__file__, {
        'case_name': 'quality_normalized',
        'description': 'Random clouds with per-cloud normalization',
        'config': {
            'num_steps': 2000, 'n_particles': 200, 'd': 1,
            'learning_rate': 1e-3, 'alpha_init_bias': -3.0,
        },
        'training': {
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'min_loss': float(min_loss),
            'max_grad_late': float(max_grad_late),
            'final_lambda': float(final_lambda),
            'h_min': float(h_min),
            'h_max': float(h_max),
        },
        'eval': {
            'identity_mean_err': float(m_id),
            'T_mean_err': float(m_T),
            'T_output_mean': float(mu_T),
            'weighted_mean': float(mu_w),
        },
        'checks': {
            'loss_decreased': loss_decreased,
            'h_stable': h_stable,
            'grad_stable': grad_stable,
            'quality_improved': quality_improved,
        },
        'passed': bool(loss_decreased and h_stable and grad_stable and quality_improved),
    })

    assert loss_decreased, f"Loss did not decrease: {initial_loss:.4e} -> {final_loss:.4e}"
    assert h_stable, f"h still unstable after normalization: [{h_min:.4f}, {h_max:.4f}]"
    assert grad_stable, f"Late gradient spikes: max |grad| = {max_grad_late:.2f}"
    assert quality_improved, (
        f"T did not beat identity on held-out cloud: "
        f"identity {m_id:.4f} -> T {m_T:.4f}"
    )

    print("\n=== Quality test with normalization: PASSED ===\n")
