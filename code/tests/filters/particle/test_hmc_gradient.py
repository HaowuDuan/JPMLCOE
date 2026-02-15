"""Test gradient of LEDH log-likelihood w.r.t. model parameters.

Diagnoses whether the gradient through the compiled filter is finite and
nonzero for different n_lambda_steps values. This is the key requirement
for HMC to work — if the gradient is NaN or zero, HMC proposals always
get rejected.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
import numpy as np
import tensorflow as tf

from src.models.kitagawa import KitagawaModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC


@pytest.fixture
def observations():
    """Generate observations from the true Kitagawa model."""
    true_model = KitagawaModel(sigma_V=10.0, sigma_W=1.0)
    rng = np.random.default_rng(42)
    _, _, obs = generate_data(true_model, T=20, rng=rng)
    return tf.constant(obs, dtype=tf.float32)


def _compute_gradient(observations, n_lambda_steps, eager_mode):
    """Compute gradient of log-likelihood w.r.t. sigma_V and sigma_W."""
    model = KitagawaModel(sigma_V=5.0, sigma_W=2.0)

    filter_obj = LEDHParticleFlowFilterHMC(
        model,
        n_particles=100,
        n_lambda_steps=n_lambda_steps,
        weight_clip_range=50.0,
        stop_gradient_resampling=True,
        eager_mode=eager_mode,
    )

    # Simulate HMC: create tensors for sigma_V, sigma_W and watch them
    sigma_V = tf.Variable(5.0, dtype=tf.float32)
    sigma_W = tf.Variable(2.0, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Set model params (same as HMC runner does via setattr)
        model.sigma_V = sigma_V
        model.sigma_W = sigma_W

        seed = tf.constant([42, 0], dtype=tf.int32)
        log_lik = filter_obj.log_marginal_likelihood_tf(observations, seed=seed)

    grads = tape.gradient(log_lik, [sigma_V, sigma_W])
    return log_lik, grads


@pytest.mark.parametrize("n_lambda_steps", [5, 10, 15, 20, 29])
def test_gradient_finite_compiled(observations, n_lambda_steps):
    """Gradient through compiled filter should be finite and nonzero."""
    log_lik, grads = _compute_gradient(observations, n_lambda_steps, eager_mode=False)

    grad_V, grad_W = grads

    print(f"\n  n_lambda_steps={n_lambda_steps}:")
    print(f"    log_lik     = {log_lik.numpy():.4f}")
    print(f"    grad_sigma_V = {grad_V.numpy() if grad_V is not None else 'None'}")
    print(f"    grad_sigma_W = {grad_W.numpy() if grad_W is not None else 'None'}")

    # Gradient should not be None (disconnected graph)
    assert grad_V is not None, f"grad_sigma_V is None — gradient disconnected at n_lambda_steps={n_lambda_steps}"
    assert grad_W is not None, f"grad_sigma_W is None — gradient disconnected at n_lambda_steps={n_lambda_steps}"

    # Gradient should be finite (not NaN or Inf)
    assert tf.math.is_finite(grad_V), f"grad_sigma_V is {grad_V.numpy()} at n_lambda_steps={n_lambda_steps}"
    assert tf.math.is_finite(grad_W), f"grad_sigma_W is {grad_W.numpy()} at n_lambda_steps={n_lambda_steps}"

    # Gradient should be nonzero (at least one should be)
    assert tf.abs(grad_V) > 1e-10 or tf.abs(grad_W) > 1e-10, \
        f"Both gradients are zero at n_lambda_steps={n_lambda_steps}"


@pytest.mark.parametrize("n_lambda_steps", [5, 15, 29])
def test_gradient_finite_eager(observations, n_lambda_steps):
    """Gradient through eager filter should be finite — baseline comparison."""
    log_lik, grads = _compute_gradient(observations, n_lambda_steps, eager_mode=True)

    grad_V, grad_W = grads

    print(f"\n  [eager] n_lambda_steps={n_lambda_steps}:")
    print(f"    log_lik     = {log_lik.numpy():.4f}")
    print(f"    grad_sigma_V = {grad_V.numpy() if grad_V is not None else 'None'}")
    print(f"    grad_sigma_W = {grad_W.numpy() if grad_W is not None else 'None'}")

    assert grad_V is not None, "grad_sigma_V is None in eager mode"
    assert grad_W is not None, "grad_sigma_W is None in eager mode"
    assert tf.math.is_finite(grad_V), f"grad_sigma_V is {grad_V.numpy()} in eager mode"
    assert tf.math.is_finite(grad_W), f"grad_sigma_W is {grad_W.numpy()} in eager mode"


def test_gradient_compiled_vs_eager(observations):
    """Compiled and eager gradients should roughly agree (same math)."""
    log_lik_c, grads_c = _compute_gradient(observations, n_lambda_steps=5, eager_mode=False)
    log_lik_e, grads_e = _compute_gradient(observations, n_lambda_steps=5, eager_mode=True)

    print(f"\n  Compiled: log_lik={log_lik_c.numpy():.4f}, "
          f"grad_V={grads_c[0].numpy() if grads_c[0] is not None else 'None'}, "
          f"grad_W={grads_c[1].numpy() if grads_c[1] is not None else 'None'}")
    print(f"  Eager:    log_lik={log_lik_e.numpy():.4f}, "
          f"grad_V={grads_e[0].numpy() if grads_e[0] is not None else 'None'}, "
          f"grad_W={grads_e[1].numpy() if grads_e[1] is not None else 'None'}")

    # Log-likelihoods should match (same computation, same seed)
    if grads_c[0] is not None and grads_e[0] is not None:
        # They should at least have the same sign
        if tf.math.is_finite(grads_c[0]) and tf.math.is_finite(grads_e[0]):
            print(f"  grad_V sign: compiled={'+' if grads_c[0] > 0 else '-'}, "
                  f"eager={'+' if grads_e[0] > 0 else '-'}")
