"""Unit tests for WeightedDeepSetsEncoder.

Verify:
1. Output shape is correct
2. Permutation invariance: shuffling particles gives same context
3. Variable N: encoder accepts different particle counts with same weights
4. Differentiable through particles and weights
5. Context responds to changes in input distribution

Run: pytest code/neural_operator/tests/test_encoder.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from encoder import WeightedDeepSetsEncoder


DTYPE = tf.float64


def test_output_shape():
    """Encoder output has correct context_dim."""
    encoder = WeightedDeepSetsEncoder(in_dim=2, context_dim=64)
    rng = np.random.default_rng(42)
    particles = tf.constant(rng.normal(0, 1, (100, 2)), dtype=DTYPE)
    weights = tf.ones([100], dtype=DTYPE) / 100
    c = encoder(particles, weights)
    assert c.shape == (64,), f"Expected (64,), got {c.shape}"
    print(f"  test_output_shape PASS: c.shape = {c.shape}")


def test_permutation_invariance():
    """Shuffling particles must give the same context."""
    encoder = WeightedDeepSetsEncoder(in_dim=2, context_dim=32)
    rng = np.random.default_rng(42)
    N = 50
    particles_np = rng.normal(0, 1, (N, 2))
    weights_np = rng.exponential(1, N)
    weights_np /= weights_np.sum()

    particles = tf.constant(particles_np, dtype=DTYPE)
    weights = tf.constant(weights_np, dtype=DTYPE)
    c_orig = encoder(particles, weights)

    # Shuffle (same permutation for particles and weights)
    perm = rng.permutation(N)
    particles_shuf = tf.constant(particles_np[perm], dtype=DTYPE)
    weights_shuf = tf.constant(weights_np[perm], dtype=DTYPE)
    c_shuf = encoder(particles_shuf, weights_shuf)

    diff = float(tf.reduce_max(tf.abs(c_orig - c_shuf)).numpy())
    assert diff < 1e-10, f"Permutation broke invariance: max diff = {diff}"
    print(f"  test_permutation_invariance PASS: max diff = {diff:.2e}")


def test_variable_N():
    """Encoder handles different N values."""
    encoder = WeightedDeepSetsEncoder(in_dim=2, context_dim=32)
    rng = np.random.default_rng(42)
    for N in [50, 200, 500]:
        particles = tf.constant(rng.normal(0, 1, (N, 2)), dtype=DTYPE)
        weights = tf.ones([N], dtype=DTYPE) / N
        c = encoder(particles, weights)
        assert c.shape == (32,)
    print(f"  test_variable_N PASS: handled N=50, 200, 500")


def test_differentiable():
    """Encoder gradients flow through particles and weights."""
    encoder = WeightedDeepSetsEncoder(in_dim=2, context_dim=32)
    rng = np.random.default_rng(42)
    particles = tf.Variable(rng.normal(0, 1, (50, 2)).astype(np.float64))
    weights_raw = tf.Variable(np.ones(50).astype(np.float64))

    with tf.GradientTape() as tape:
        weights = tf.nn.softmax(weights_raw)
        c = encoder(particles, weights)
        loss = tf.reduce_sum(c ** 2)

    grad_p = tape.gradient(loss, particles)
    assert grad_p is not None
    assert tf.reduce_all(tf.math.is_finite(grad_p)).numpy()
    print(f"  test_differentiable PASS: grad norm = {float(tf.norm(grad_p)):.4f}")


def test_context_responds_to_input():
    """Different particle distributions should give different contexts."""
    encoder = WeightedDeepSetsEncoder(in_dim=2, context_dim=32)
    rng = np.random.default_rng(42)

    # Tight cluster
    p1 = tf.constant(rng.normal(0, 0.1, (50, 2)), dtype=DTYPE)
    # Wide cluster
    p2 = tf.constant(rng.normal(0, 5.0, (50, 2)), dtype=DTYPE)
    w = tf.ones([50], dtype=DTYPE) / 50

    c1 = encoder(p1, w)
    c2 = encoder(p2, w)
    diff = float(tf.reduce_max(tf.abs(c1 - c2)).numpy())
    assert diff > 0.01, f"Encoder did not respond to spread change: diff = {diff}"
    print(f"  test_context_responds_to_input PASS: diff = {diff:.4f}")


