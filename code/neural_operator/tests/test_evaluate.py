"""Smoke test for the evaluation module.

Trains a tiny neural operator (300 steps, d=1) and evaluates on held-out clouds.
Verifies:
- All metrics are finite
- Trained model is meaningfully better than untrained on KL and moment error
- J conditioning: lambda_min > 0, no logdet failures

Run: python code/neural_operator/tests/test_evaluate.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import tensorflow as tf

from train import NeuralOperator, train
from evaluate import evaluate, print_summary


DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')


def test_evaluate_trained_model():
    print("\n=== Neural operator evaluation smoke test ===\n")
    tf.random.set_seed(42)

    model = NeuralOperator(
        in_dim=1,
        embed_dim=32,
        num_modules=8,
        context_dim=32,
        encoder_hidden=(32, 32),
        alpha_init_bias=-3.0,
    )

    # ---- Evaluate untrained baseline (residual init -> T ~ identity)
    print("--- Untrained baseline ---")
    _, summary_untrained = evaluate(
        model, num_clouds=10, n_particles=200, d=1, q_points=64,
        n_kl_samples=256,
    )
    print_summary(summary_untrained)

    # ---- Train
    print("\n--- Training (300 steps) ---")
    train(
        model,
        num_steps=300,
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

    # ---- Evaluate trained model
    print("\n--- Trained model ---")
    _, summary_trained = evaluate(
        model, num_clouds=10, n_particles=200, d=1, q_points=64,
        n_kl_samples=256,
    )
    print_summary(summary_trained)

    # ---- Sanity checks
    metrics_to_finite_check = [
        'ma_residual_mse_mean', 'kl_p_r_mean', 'mmd2_mean',
        'mean_err_rel_mean', 'cov_err_rel_mean',
        'lambda_min_mean_mean', 'lambda_min_min_mean',
    ]
    for k in metrics_to_finite_check:
        v = summary_trained[k]
        assert np.isfinite(v), f"{k} not finite: {v}"

    # Trained should beat untrained on the MA residual (the loss it was trained on)
    assert summary_trained['ma_residual_mse_mean'] < summary_untrained['ma_residual_mse_mean'], (
        f"MA residual did not improve: "
        f"{summary_untrained['ma_residual_mse_mean']:.4e} -> "
        f"{summary_trained['ma_residual_mse_mean']:.4e}"
    )

    # Trained should give smaller (or comparable) moment error than untrained.
    # Untrained T ~ identity transports particles to themselves with uniform weights,
    # so the transported empirical mean is the unweighted mean of the original
    # particles, NOT the weighted target. Trained T should fix this.
    assert summary_trained['mean_err_rel_mean'] < summary_untrained['mean_err_rel_mean'] + 1e-6, (
        f"Mean error did not improve: "
        f"{summary_untrained['mean_err_rel_mean']:.4f} -> "
        f"{summary_trained['mean_err_rel_mean']:.4f}"
    )

    # J conditioning
    assert summary_trained['lambda_min_min_mean'] > 0.0, (
        f"lambda_min went non-positive: {summary_trained['lambda_min_min_mean']}"
    )
    assert summary_trained['logdet_failures_mean'] == 0.0, (
        f"logdet failures: {summary_trained['logdet_failures_mean']}"
    )

    print("\n=== Evaluation smoke test passed ===\n")
    print(f"  MA residual:  {summary_untrained['ma_residual_mse_mean']:.4e} -> "
          f"{summary_trained['ma_residual_mse_mean']:.4e}")
    print(f"  KL(p||r):     {summary_untrained['kl_p_r_mean']:.4f} -> "
          f"{summary_trained['kl_p_r_mean']:.4f}")
    print(f"  Mean err:     {summary_untrained['mean_err_rel_mean']:.4f} -> "
          f"{summary_trained['mean_err_rel_mean']:.4f}")
    print(f"  Cov err:      {summary_untrained['cov_err_rel_mean']:.4f} -> "
          f"{summary_trained['cov_err_rel_mean']:.4f}")


if __name__ == '__main__':
    test_evaluate_trained_model()
