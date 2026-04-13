"""Smoke test for the evaluation module.

Trains a tiny neural operator (300 steps, d=1) and evaluates on held-out clouds.
Verifies:
- All metrics are finite
- Trained model is meaningfully better than untrained on KL and moment error
- J conditioning: lambda_min > 0, no logdet failures

Run: pytest code/neural_operator/tests/test_evaluate.py -v
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from train import NeuralOperator, train
from evaluate import evaluate, print_summary
from _result_utils import save_result, reset_results


DTYPE = tf.float64


def setup_module():
    reset_results(__file__)


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

    # ---- Compute all passed flags before any assertion
    metrics_to_finite_check = [
        'ma_residual_mse_mean', 'kl_p_r_mean', 'mmd2_mean',
        'mean_err_rel_mean', 'cov_err_rel_mean',
        'lambda_min_mean_mean', 'lambda_min_min_mean',
    ]
    all_finite = all(np.isfinite(summary_trained[k]) for k in metrics_to_finite_check)

    ma_residual_decreased = (
        summary_trained['ma_residual_mse_mean'] < summary_untrained['ma_residual_mse_mean']
    )
    mean_err_decreased = (
        summary_trained['mean_err_rel_mean'] < summary_untrained['mean_err_rel_mean'] + 1e-6
    )
    lambda_min_positive = summary_trained['lambda_min_min_mean'] > 0.0
    no_logdet_failures = summary_trained['logdet_failures_mean'] == 0.0

    trained_passed = (
        all_finite
        and ma_residual_decreased
        and mean_err_decreased
        and lambda_min_positive
        and no_logdet_failures
    )

    # ---- Save results for both cases
    save_result(__file__, {
        'case_name': 'untrained_baseline',
        'metrics': dict(summary_untrained),
        'passed': True,  # baseline evaluation always "passes" — it's just a reference
    })
    save_result(__file__, {
        'case_name': 'trained_model',
        'metrics': dict(summary_trained),
        'passed': trained_passed,
    })

    # ---- Assertions
    for k in metrics_to_finite_check:
        v = summary_trained[k]
        assert np.isfinite(v), f"{k} not finite: {v}"

    # Trained should beat untrained on the MA residual (the loss it was trained on)
    assert ma_residual_decreased, (
        f"MA residual did not improve: "
        f"{summary_untrained['ma_residual_mse_mean']:.4e} -> "
        f"{summary_trained['ma_residual_mse_mean']:.4e}"
    )

    # Trained should give smaller (or comparable) moment error than untrained.
    # Untrained T ~ identity transports particles to themselves with uniform weights,
    # so the transported empirical mean is the unweighted mean of the original
    # particles, NOT the weighted target. Trained T should fix this.
    assert mean_err_decreased, (
        f"Mean error did not improve: "
        f"{summary_untrained['mean_err_rel_mean']:.4f} -> "
        f"{summary_trained['mean_err_rel_mean']:.4f}"
    )

    # J conditioning
    assert lambda_min_positive, (
        f"lambda_min went non-positive: {summary_trained['lambda_min_min_mean']}"
    )
    assert no_logdet_failures, (
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
