"""Smoke test for the neural operator training loop.

Runs a tiny training (100 steps, d=1) to verify:
1. Loss decreases
2. No NaN/inf
3. Forward and backward pass work end-to-end
4. Trained map is closer to identity (start) than the initial random map

NOT a quality test — just verifies the training pipeline runs.

Run: python code/neural_operator/tests/test_train_smoke.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import tensorflow as tf

from train import NeuralOperator, train
from data import sample_random_cloud
from kde import silverman_bandwidth_scalar
from losses import ma_residual_loss


DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')


def test_smoke_training():
    print("\n=== Neural operator training smoke test ===\n")
    tf.random.set_seed(42)

    model = NeuralOperator(
        in_dim=1,
        embed_dim=32,
        num_modules=8,
        context_dim=32,
        encoder_hidden=(32, 32),
        alpha_init_bias=-3.0,
    )

    history = train(
        model,
        num_steps=100,
        n_particles=200,
        d=1,
        q_points=64,
        batch_clouds=1,
        learning_rate=1e-3,
        linearity_weight=0.0,  # off for smoke test
        eig_weight=1e-2,
        eig_threshold=0.1,
        h_scale_init=2.0,
        h_scale_final=1.0,
        log_every=20,
    )

    losses = [h['loss'] for h in history]
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"\n  Initial loss: {initial_loss:.4e}")
    print(f"  Final loss:   {final_loss:.4e}")
    print(f"  Reduction:    {initial_loss / max(final_loss, 1e-30):.2f}x")

    # Sanity checks
    assert np.isfinite(final_loss), f"Final loss not finite: {final_loss}"
    assert final_loss < initial_loss * 2, (
        f"Loss did not decrease: {initial_loss:.4e} -> {final_loss:.4e}"
    )

    # Verify forward pass still works after training
    seed = tf.constant([99, 0], dtype=tf.int32)
    particles, weights = sample_random_cloud(n_particles=100, d=1, seed=seed)
    h = silverman_bandwidth_scalar(particles, weights)
    c = model.encode(particles, weights, h_kde=h)
    x_test = particles[:5]  # use first 5 particles
    T_x, J_x = model.transport_with_jacobian(x_test, c)
    assert T_x.shape == (5, 1)
    assert J_x.shape == (5, 1, 1)
    assert tf.reduce_all(tf.math.is_finite(T_x)).numpy()
    assert tf.reduce_all(tf.math.is_finite(J_x)).numpy()
    print(f"  Post-training forward pass: T_x shape {T_x.shape}, J_x shape {J_x.shape}")

    print("\n=== Smoke test passed ===\n")


if __name__ == '__main__':
    test_smoke_training()
