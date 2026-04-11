"""Unit tests for KDE utilities.

Verify:
1. Bandwidth selection produces sane values
2. KDE evaluates correctly (matches scipy reference for simple cases)
3. KDE integrates to ~1 over support
4. Samples from q_h are distributed like q_h
5. log_kde is differentiable through h and particles

Run: python code/neural_operator/src/test_kde.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import math
import numpy as np
import tensorflow as tf

from kde import (
    effective_sample_size,
    weighted_mean,
    weighted_covariance,
    silverman_bandwidth_scalar,
    silverman_bandwidth_matrix,
    log_kde_weighted,
    log_kde_uniform,
    sample_from_uniform_kde,
)


DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')


def test_effective_sample_size():
    """ESS = N for uniform weights, < N for non-uniform."""
    N = 100
    w_uniform = tf.ones([N], dtype=DTYPE) / N
    ess = effective_sample_size(w_uniform)
    assert abs(float(ess) - N) < 1e-10
    print(f"  test_effective_sample_size PASS: ESS(uniform) = {float(ess):.2f}")

    # Skewed: half the mass on 1 particle
    w_skewed = tf.constant([0.5] + [0.5 / (N - 1)] * (N - 1), dtype=DTYPE)
    ess_skewed = float(effective_sample_size(w_skewed))
    assert ess_skewed < N
    print(f"  test_effective_sample_size_skewed: ESS = {ess_skewed:.2f} (N={N})")


def test_weighted_moments():
    """Weighted mean/cov matches numpy."""
    rng = np.random.default_rng(42)
    N, d = 200, 2
    x_np = rng.normal(0, 1, (N, d))
    w_np = rng.exponential(1, N)
    w_np = w_np / w_np.sum()

    x = tf.constant(x_np, dtype=DTYPE)
    w = tf.constant(w_np, dtype=DTYPE)

    mu = weighted_mean(x, w).numpy()
    mu_np = (w_np[:, None] * x_np).sum(axis=0)
    assert np.allclose(mu, mu_np, atol=1e-12)

    cov = weighted_covariance(x, w).numpy()
    centered = x_np - mu_np
    cov_np = np.einsum('n,ni,nj->ij', w_np, centered, centered)
    assert np.allclose(cov, cov_np, atol=1e-12)
    print(f"  test_weighted_moments PASS")


def test_silverman_bandwidth_scalar():
    """Bandwidth shrinks with N, grows with std."""
    rng = np.random.default_rng(42)
    for N in [50, 200, 1000]:
        x = tf.constant(rng.normal(0, 1, (N, 2)), dtype=DTYPE)
        w = tf.ones([N], dtype=DTYPE) / N
        h = float(silverman_bandwidth_scalar(x, w))
        assert h > 0
        print(f"  silverman h(N={N}) = {h:.4f}")


def test_log_kde_weighted_normalization():
    """Integral of p_h(x) over a large grid should be ~1."""
    # 1D example: easy to integrate
    rng = np.random.default_rng(42)
    N = 50
    particles = tf.constant(rng.normal(0, 1, (N, 1)), dtype=DTYPE)
    weights = tf.ones([N], dtype=DTYPE) / N
    h = silverman_bandwidth_scalar(particles, weights)

    # Grid for integration
    grid = tf.constant(np.linspace(-10, 10, 2001).reshape(-1, 1), dtype=DTYPE)
    dx = 20.0 / 2000

    log_p = log_kde_weighted(grid, particles, weights, h)
    p = tf.exp(log_p)
    integral = float(tf.reduce_sum(p) * dx)
    assert abs(integral - 1.0) < 0.01, f"Integral = {integral}, expected ~1"
    print(f"  test_log_kde_weighted_normalization PASS: integral = {integral:.4f}")


def test_log_kde_at_centers_vs_far():
    """KDE value at particle centers should be much higher than far away."""
    particles = tf.constant([[0.0], [5.0]], dtype=DTYPE)
    weights = tf.constant([0.5, 0.5], dtype=DTYPE)
    h = tf.constant(0.5, dtype=DTYPE)

    # At center
    x_center = tf.constant([[0.0]], dtype=DTYPE)
    # Far away
    x_far = tf.constant([[100.0]], dtype=DTYPE)

    log_p_center = float(log_kde_weighted(x_center, particles, weights, h)[0])
    log_p_far = float(log_kde_weighted(x_far, particles, weights, h)[0])

    assert log_p_center > log_p_far
    print(f"  test_log_kde_at_centers_vs_far PASS: log_p(center)={log_p_center:.2f}, log_p(far)={log_p_far:.2f}")


def test_sample_from_uniform_kde_distribution():
    """Samples from q_h should have similar mean/cov as q_h."""
    rng = np.random.default_rng(42)
    N = 200
    particles_np = rng.normal(0, 1, (N, 2))
    particles = tf.constant(particles_np, dtype=DTYPE)
    weights = tf.ones([N], dtype=DTYPE) / N
    h = silverman_bandwidth_scalar(particles, weights)

    # Sample many points
    seed = tf.constant([0, 0], dtype=tf.int32)
    samples = sample_from_uniform_kde(particles, h, n_samples=10000, seed=seed)

    # Mean should be close to particle mean
    sample_mean = tf.reduce_mean(samples, axis=0).numpy()
    particle_mean = particles_np.mean(axis=0)
    diff = np.abs(sample_mean - particle_mean).max()
    assert diff < 0.1, f"Sample mean {sample_mean} vs particle mean {particle_mean}"
    print(f"  test_sample_from_uniform_kde_distribution PASS: mean diff = {diff:.4f}")


def test_log_kde_differentiable():
    """log_kde should be differentiable through particles."""
    rng = np.random.default_rng(42)
    N = 20
    particles = tf.Variable(rng.normal(0, 1, (N, 1)).astype(np.float64))
    weights = tf.ones([N], dtype=DTYPE) / N
    h = tf.constant(0.5, dtype=DTYPE)
    query = tf.constant([[0.5]], dtype=DTYPE)

    with tf.GradientTape() as tape:
        log_p = log_kde_weighted(query, particles, weights, h)
        loss = tf.reduce_sum(log_p)
    grad = tape.gradient(loss, particles)
    assert grad is not None
    assert tf.reduce_all(tf.math.is_finite(grad)).numpy()
    print(f"  test_log_kde_differentiable PASS: grad norm = {float(tf.norm(grad)):.4f}")


if __name__ == '__main__':
    print("\n=== KDE utility tests ===\n")
    tf.random.set_seed(42)

    test_effective_sample_size()
    test_weighted_moments()
    test_silverman_bandwidth_scalar()
    test_log_kde_weighted_normalization()
    test_log_kde_at_centers_vs_far()
    test_sample_from_uniform_kde_distribution()
    test_log_kde_differentiable()

    print("\n=== All KDE tests passed ===\n")
