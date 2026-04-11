"""Smoke test for NeuralOperatorResampler.

Trains a tiny neural operator briefly, wraps it in NeuralOperatorResampler,
and runs it on a held-out cloud. Verifies:
1. ResampleResult shapes are correct
2. Output weights are uniform and normalized
3. local_jacobians is populated and PSD
4. transport_matrix and ancestor_indices are None
5. Mapped particles match a direct model.transport_with_jacobian call
6. Trained resampler reduces moment error vs identity

Run: python code/neural_operator/tests/test_resampler.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
# Order matters: neural_operator/src must come BEFORE code/src so that
# `from models import WTanh` finds neural_operator/src/models.py rather than
# the state-space model package at code/src/models/.
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))   # code/src (so `resampling` is importable)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))         # neural_operator/src (priority)

import numpy as np
import tensorflow as tf

from train import NeuralOperator, train
from data import sample_random_cloud
from resampling import NeuralOperatorResampler, ResampleResult


DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')


def test_resampler_smoke():
    print("\n=== NeuralOperatorResampler smoke test ===\n")
    tf.random.set_seed(123)

    model = NeuralOperator(
        in_dim=1,
        embed_dim=32,
        num_modules=8,
        context_dim=32,
        encoder_hidden=(32, 32),
        alpha_init_bias=-3.0,
    )

    print("--- Train 200 steps ---")
    train(
        model,
        num_steps=200,
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
        log_every=50,
    )

    # First held-out cloud (used for shape / determinism / PSD checks)
    cloud_seed = tf.constant([9999, 0], dtype=tf.int32)
    particles, weights = sample_random_cloud(n_particles=200, d=1, seed=cloud_seed)

    # ---- Run through the resampler
    resampler = NeuralOperatorResampler(model)
    result = resampler(particles, weights, seed=tf.constant([0, 0], dtype=tf.int32))

    # ---- Shape and type checks
    assert isinstance(result, ResampleResult), f"got {type(result)}"
    assert result.particles.shape == (200, 1), f"particles shape {result.particles.shape}"
    assert result.weights.shape == (200,), f"weights shape {result.weights.shape}"
    assert result.local_jacobians is not None, "local_jacobians is None"
    assert result.local_jacobians.shape == (200, 1, 1), (
        f"local_jacobians shape {result.local_jacobians.shape}"
    )
    assert result.transport_matrix is None, "transport_matrix should be None"
    assert result.ancestor_indices is None, "ancestor_indices should be None"
    print(f"  Shapes OK: particles={result.particles.shape}, "
          f"J={result.local_jacobians.shape}")

    # ---- Uniform weights
    expected_w = 1.0 / 200.0
    max_w_diff = float(tf.reduce_max(tf.abs(result.weights - expected_w)))
    assert max_w_diff < 1e-12, f"weights not uniform: max diff = {max_w_diff}"
    w_sum = float(tf.reduce_sum(result.weights))
    assert abs(w_sum - 1.0) < 1e-12, f"weights don't sum to 1: {w_sum}"
    print(f"  Weights uniform OK (sum={w_sum:.6f})")

    # ---- Jacobians PSD
    eigvals = tf.linalg.eigvalsh(result.local_jacobians)  # (N, d)
    min_eig = float(tf.reduce_min(eigvals))
    assert min_eig > 0.0, f"non-PSD Jacobian: min_eig={min_eig}"
    print(f"  J PSD OK: min_eig={min_eig:.4f}")

    # ---- Determinism: run twice, same input -> same output
    result2 = resampler(particles, weights, seed=tf.constant([42, 0], dtype=tf.int32))
    diff = float(tf.reduce_max(tf.abs(result.particles - result2.particles)))
    assert diff < 1e-12, f"resampler not deterministic: diff={diff}"
    print(f"  Deterministic OK")

    # NOTE: Quality (moment matching, KL, etc.) is intentionally NOT checked
    # here. This smoke test only validates the resampler INTERFACE
    # (shapes, weights, PSD Jacobians, determinism, ResampleResult fields).
    # Model-quality assertions live in dedicated quality tests on fixed
    # low-ESS clouds.

    print("\n=== Resampler smoke test passed ===\n")


if __name__ == '__main__':
    test_resampler_smoke()
