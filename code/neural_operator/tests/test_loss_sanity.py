"""Loss sanity tests for the Monge-Ampère residual.

Two tests:

A0 — uniform-weights identity-map test.
   Set up: 200 equally-spaced particles on [-5, 5], uniform weights w_i = 1/N,
   stub model with T(x)=x and J(x)=I. The MA residual should be exactly zero
   because (a) log det I = 0 and (b) with uniform weights the weighted KDE
   equals the uniform KDE at every query point, so log p_h(T(x)) = log q_h(x)
   and the residual log det J - (log q_h - log p_h(T)) = 0 - 0 = 0.

   If A0 fails, the loss formula has a sign error, a source/target convention
   mismatch, or a bug in slogdet / log_kde_*.

A1 — uniform-weights random-untrained-model test.
   Same particles, same uniform weights, same query points, but the model is
   a real (random init) NeuralOperator. The loss should be > 0 because a
   random map is not identity.

   If A1 fails (loss is ~0), then A0's success was trivial — the loss is
   returning zero regardless of the map.

Run: pytest code/neural_operator/tests/test_loss_sanity.py -v -s
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from losses import ma_residual_loss
from kde import (
    particle_resolution_bandwidth_scalar,
    sample_from_uniform_kde,
)
from train import NeuralOperator
from _result_utils import save_result, reset_results


DTYPE = tf.float64


def setup_module():
    reset_results(__file__)


class IdentityTrunk:
    """Stub trunk that returns T(x) = x and J(x) = I.

    Has the duck-typed interface ma_residual_loss expects:
        forward_and_jacobian(x, c) -> (T_x, J_x)

    Ignores the context argument c entirely.
    """

    def forward_and_jacobian(self, x, c):
        T_x = x
        d = tf.shape(x)[-1]
        Q = tf.shape(x)[0]
        J_x = tf.eye(d, batch_shape=[Q], dtype=x.dtype)
        return T_x, J_x


def _build_test_setup(n=200, A=5.0, n_query=64, seed=42):
    """Build the shared test inputs: particles, uniform weights, h, query points."""
    x = np.linspace(-A, A, n).reshape(-1, 1)
    particles = tf.constant(x, dtype=tf.float64)
    weights = tf.ones([n], dtype=tf.float64) / float(n)
    h = particle_resolution_bandwidth_scalar(particles, scale=5.0)
    query_seed = tf.constant([seed, 0], dtype=tf.int32)
    query_points = sample_from_uniform_kde(particles, h, n_query, query_seed)
    return particles, weights, h, query_points


def test_A0_identity_map_uniform_weights():
    print("\n--- A0: identity map + uniform weights ---")
    particles, weights, h, query_points = _build_test_setup()
    print(f"  N = {int(particles.shape[0])}")
    print(f"  weights are uniform: w_i = 1/N = {float(weights[0]):.6f}")
    print(f"  h = {float(h):.6f}")
    print(f"  Q = {int(query_points.shape[0])} query points")

    stub = IdentityTrunk()
    loss, diag = ma_residual_loss(
        stub, particles, weights, h, query_points, c=None
    )
    loss_val = float(loss.numpy())
    log_det_J_mean = float(diag['log_det_J_mean'].numpy())
    log_q_mean = float(diag['log_q_mean'].numpy())
    log_p_T_mean = float(diag['log_p_T_mean'].numpy())
    residual_mean = float(diag['residual_mean'].numpy())
    residual_abs_mean = float(diag['residual_abs_mean'].numpy())

    print(f"  loss          = {loss_val:.6e}")
    print(f"  log_det_J_mean = {log_det_J_mean:+.6e}  (expected 0)")
    print(f"  log_q_mean     = {log_q_mean:+.6e}")
    print(f"  log_p_T_mean   = {log_p_T_mean:+.6e}")
    print(f"  residual_mean  = {residual_mean:+.6e}  (expected 0)")
    print(f"  residual_abs   = {residual_abs_mean:+.6e}")

    threshold = 1e-12
    passed = bool(np.isfinite(loss_val) and loss_val < threshold)

    save_result(__file__, {
        'case_name': 'A0_identity_map_uniform_weights',
        'description': 'Identity stub map + uniform weights, loss should be 0',
        'config': {
            'n_particles': int(particles.shape[0]),
            'A': 5.0,
            'h': float(h),
            'n_query': int(query_points.shape[0]),
        },
        'metrics': {
            'loss': loss_val,
            'log_det_J_mean': log_det_J_mean,
            'log_q_mean': log_q_mean,
            'log_p_T_mean': log_p_T_mean,
            'residual_mean': residual_mean,
            'residual_abs_mean': residual_abs_mean,
        },
        'threshold': threshold,
        'passed': passed,
    })

    assert passed, (
        f"A0 FAILED: identity map with uniform weights should give loss = 0, "
        f"got loss = {loss_val:.6e}. The MA residual formula has a sign error, "
        f"a source/target convention mismatch, or a bug in slogdet / log_kde_*."
    )
    print("  A0 PASS")


def test_A1_random_model_uniform_weights():
    print("\n--- A1: random model + uniform weights ---")
    particles, weights, h, query_points = _build_test_setup()
    print(f"  N = {int(particles.shape[0])}, h = {float(h):.6f}")

    tf.random.set_seed(7)
    model = NeuralOperator(
        in_dim=1,
        embed_dim=32,
        num_modules=8,
        context_dim=32,
        encoder_hidden=(32, 32),
        alpha_init_bias=0.0,  # NOT residual init: we want a non-identity map
    )
    # Build the model variables with one forward pass.
    c = model.encode(particles, weights, h_kde=h)
    _ = model.transport_with_jacobian(particles, c)

    loss, diag = ma_residual_loss(
        model.trunk, particles, weights, h, query_points, c
    )
    loss_val = float(loss.numpy())
    log_det_J_mean = float(diag['log_det_J_mean'].numpy())
    log_q_mean = float(diag['log_q_mean'].numpy())
    log_p_T_mean = float(diag['log_p_T_mean'].numpy())
    residual_mean = float(diag['residual_mean'].numpy())
    residual_abs_mean = float(diag['residual_abs_mean'].numpy())

    print(f"  loss          = {loss_val:.6e}")
    print(f"  log_det_J_mean = {log_det_J_mean:+.6e}")
    print(f"  log_q_mean     = {log_q_mean:+.6e}")
    print(f"  log_p_T_mean   = {log_p_T_mean:+.6e}")
    print(f"  residual_mean  = {residual_mean:+.6e}")
    print(f"  residual_abs   = {residual_abs_mean:+.6e}")

    threshold = 1e-6
    passed = bool(np.isfinite(loss_val) and loss_val > threshold)

    save_result(__file__, {
        'case_name': 'A1_random_model_uniform_weights',
        'description': 'Random untrained model + uniform weights, loss should be > 0',
        'config': {
            'n_particles': int(particles.shape[0]),
            'A': 5.0,
            'h': float(h),
            'n_query': int(query_points.shape[0]),
            'alpha_init_bias': 0.0,
            'num_modules': 8,
            'embed_dim': 32,
        },
        'metrics': {
            'loss': loss_val,
            'log_det_J_mean': log_det_J_mean,
            'log_q_mean': log_q_mean,
            'log_p_T_mean': log_p_T_mean,
            'residual_mean': residual_mean,
            'residual_abs_mean': residual_abs_mean,
        },
        'threshold': threshold,
        'passed': passed,
    })

    assert passed, (
        f"A1 FAILED: random untrained model with uniform weights gave "
        f"loss = {loss_val:.6e}, which is suspiciously close to zero. "
        f"A0 success may be trivial — the loss might be returning ~0 "
        f"regardless of the map."
    )
    print("  A1 PASS")
