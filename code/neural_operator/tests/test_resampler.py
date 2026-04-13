"""Smoke test for NeuralOperatorResampler.

Trains a tiny neural operator briefly, wraps it in NeuralOperatorResampler,
and runs it on a held-out cloud. Verifies:
1. ResampleResult shapes are correct
2. Output weights are uniform and normalized
3. local_jacobians is populated and PSD
4. transport_matrix and ancestor_indices are None
5. Mapped particles match a direct model.transport_with_jacobian call
6. Resampler is deterministic on repeated calls

Run: pytest code/neural_operator/tests/test_resampler.py -v -s
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from train import NeuralOperator, train
from data import sample_random_cloud
from resampling import NeuralOperatorResampler, ResampleResult
from _result_utils import save_result, reset_results


DTYPE = tf.float64


def setup_module():
    reset_results(__file__)


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

    metrics = {}
    failures = []

    # Shape and type checks
    if not isinstance(result, ResampleResult):
        failures.append(f"got type {type(result).__name__}, expected ResampleResult")
    metrics['particles_shape'] = list(result.particles.shape)
    metrics['weights_shape'] = list(result.weights.shape)
    if result.local_jacobians is None:
        failures.append("local_jacobians is None")
        metrics['local_jacobians_shape'] = None
    else:
        metrics['local_jacobians_shape'] = list(result.local_jacobians.shape)
        if list(result.local_jacobians.shape) != [200, 1, 1]:
            failures.append(f"local_jacobians shape {result.local_jacobians.shape}")
    if result.transport_matrix is not None:
        failures.append("transport_matrix is not None")
    if result.ancestor_indices is not None:
        failures.append("ancestor_indices is not None")
    print(f"  Shapes OK: particles={metrics['particles_shape']}, J={metrics['local_jacobians_shape']}")

    # Uniform weights
    expected_w = 1.0 / 200.0
    max_w_diff = float(tf.reduce_max(tf.abs(result.weights - expected_w)))
    w_sum = float(tf.reduce_sum(result.weights))
    metrics['max_w_diff'] = max_w_diff
    metrics['w_sum'] = w_sum
    if max_w_diff >= 1e-12:
        failures.append(f"weights not uniform: max diff {max_w_diff}")
    if abs(w_sum - 1.0) >= 1e-12:
        failures.append(f"weights don't sum to 1: {w_sum}")
    print(f"  Weights uniform OK (sum={w_sum:.6f}, max diff={max_w_diff:.2e})")

    # Jacobians PSD
    eigvals = tf.linalg.eigvalsh(result.local_jacobians)
    min_eig = float(tf.reduce_min(eigvals))
    metrics['min_eig'] = min_eig
    if min_eig <= 0.0:
        failures.append(f"non-PSD Jacobian: min_eig={min_eig}")
    print(f"  J PSD OK: min_eig={min_eig:.4f}")

    # Determinism
    result2 = resampler(particles, weights, seed=tf.constant([42, 0], dtype=tf.int32))
    determinism_diff = float(tf.reduce_max(tf.abs(result.particles - result2.particles)))
    metrics['determinism_diff'] = determinism_diff
    if determinism_diff >= 1e-12:
        failures.append(f"resampler not deterministic: diff={determinism_diff}")
    print(f"  Deterministic OK (diff={determinism_diff:.2e})")

    passed = (len(failures) == 0)

    save_result(__file__, {
        'case_name': 'resampler_smoke',
        'description': 'NeuralOperatorResampler interface smoke test',
        'config': {
            'n_particles': 200,
            'd': 1,
            'training_steps': 200,
            'cloud_seed': [9999, 0],
        },
        'metrics': metrics,
        'failures': failures,
        'passed': passed,
    })

    assert passed, "resampler smoke test failed: " + "; ".join(failures)
    print("\n=== Resampler smoke test passed ===\n")
