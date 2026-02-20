"""
Tests for src/utils/distributions.py — stable probability computations.

# Run and save results:
#   .venv/bin/python -m pytest tests/test_utils_distributions.py -v 2>&1 | tee tests/test_utils_distributions_results.txt
"""

import pytest
import math
import numpy as np
import tensorflow as tf

from src.utils.distributions import (
    log_gaussian_prob, log_sum_exp, normalize_log_weights,
    multivariate_normal_sample, sample_particles_cholesky,
    compute_flow_weights,
)


# ============================================================================
# log_gaussian_prob — T3.1 to T3.3
# ============================================================================

class TestLogGaussianProb:

    def test_1d_standard_normal(self):
        """T3.1 — 1D: x=0, μ=0, σ²=1 → log p = -0.5*log(2π)."""
        x = tf.constant([0.0], dtype=tf.float64)
        mean = tf.constant([0.0], dtype=tf.float64)
        cov = tf.eye(1, dtype=tf.float64)
        logp = log_gaussian_prob(x, mean, cov)
        np.testing.assert_allclose(
            logp.numpy(), -0.5 * np.log(2 * np.pi), rtol=1e-6
        )

    def test_2d_diagonal_matches_scipy(self):
        """T3.2 — 2D diagonal covariance: matches scipy.stats."""
        from scipy.stats import multivariate_normal as mvn

        x_np = np.array([1.0, 2.0])
        mean_np = np.array([0.0, 0.0])
        cov_np = np.diag([2.0, 3.0])

        expected = mvn.logpdf(x_np, mean_np, cov_np)

        x = tf.constant(x_np, dtype=tf.float64)
        mean = tf.constant(mean_np, dtype=tf.float64)
        cov = tf.constant(cov_np, dtype=tf.float64)
        result = log_gaussian_prob(x, mean, cov)

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_batch_input(self):
        """T3.3 — Batch (2, 2) input → (2,) output."""
        x_batch = tf.constant([[0.0, 0.0], [1.0, 1.0]], dtype=tf.float64)
        mean = tf.constant([0.0, 0.0], dtype=tf.float64)
        cov = tf.eye(2, dtype=tf.float64)
        logp = log_gaussian_prob(x_batch, mean, cov)
        assert logp.shape == (2,)
        # First element: x=mean → maximum log-prob
        assert logp.numpy()[0] > logp.numpy()[1]


# ============================================================================
# log_sum_exp — T3.4 to T3.5
# ============================================================================

class TestLogSumExp:

    def test_large_values_stability(self):
        """T3.4 — Numerically stable with large values (no overflow)."""
        log_vals = tf.constant([1000.0, 1001.0, 1002.0], dtype=tf.float64)
        result = log_sum_exp(log_vals)
        expected = 1002.0 + np.log(1 + np.exp(-1) + np.exp(-2))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_single_element(self):
        """T3.5 — Single element: log_sum_exp([a]) == a."""
        val = tf.constant([5.0], dtype=tf.float64)
        np.testing.assert_allclose(log_sum_exp(val).numpy(), 5.0, rtol=1e-10)


# ============================================================================
# normalize_log_weights — T3.6 to T3.8
# ============================================================================

class TestNormalizeLogWeights:

    def test_sum_to_one(self):
        """T3.6 — Output sums to 1 and all weights are non-negative."""
        log_w = tf.constant([-1.0, -2.0, -0.5, -3.0], dtype=tf.float64)
        w = normalize_log_weights(log_w)
        np.testing.assert_allclose(tf.reduce_sum(w).numpy(), 1.0, rtol=1e-6)
        assert np.all(w.numpy() >= 0)

    def test_uniform_input(self):
        """T3.7 — Uniform log-weights → uniform output."""
        log_w = tf.zeros(5, dtype=tf.float64)
        w = normalize_log_weights(log_w)
        np.testing.assert_allclose(w.numpy(), np.ones(5) / 5.0, rtol=1e-6)

    def test_clip_range(self):
        """T3.8 — Clip range prevents overflow for extreme log-weights."""
        log_w = tf.constant([1e6, 0.0, -1e6], dtype=tf.float64)
        w = normalize_log_weights(log_w, clip_range=(-30.0, 30.0))
        assert np.all(np.isfinite(w.numpy()))
        np.testing.assert_allclose(tf.reduce_sum(w).numpy(), 1.0, rtol=1e-6)


# ============================================================================
# multivariate_normal_sample — T3.9 to T3.10
# ============================================================================

class TestMultivariateNormalSample:

    def test_sample_statistics(self):
        """T3.9 — Sample mean and covariance match distribution (Monte Carlo)."""
        N = 50000
        mean_np = np.array([2.0, -1.0])
        cov_np = np.array([[3.0, 1.0], [1.0, 2.0]])

        mean_tf = tf.constant(mean_np, dtype=tf.float64)
        cov_tf = tf.constant(cov_np, dtype=tf.float64)
        seed = tf.constant([0, 0], dtype=tf.int32)

        samples = multivariate_normal_sample(mean_tf, cov_tf, N, seed)
        sample_np = samples.numpy()

        np.testing.assert_allclose(sample_np.mean(axis=0), mean_np, atol=0.05)
        np.testing.assert_allclose(np.cov(sample_np.T), cov_np, rtol=0.05)

    def test_output_shape(self):
        """T3.10 — Output shape is (n_samples, d)."""
        mean = tf.constant([0.0, 0.0], dtype=tf.float64)
        cov = tf.eye(2, dtype=tf.float64)
        seed = tf.constant([0, 0], dtype=tf.int32)
        samples = multivariate_normal_sample(mean, cov, 100, seed)
        assert samples.shape == (100, 2)


# ============================================================================
# sample_particles_cholesky — T3.11 to T3.12
# ============================================================================

class TestSampleParticlesCholesky:

    def test_output_shape(self):
        """T3.11 — Output shape is (N, d)."""
        mean = tf.constant([0.0, 0.0], dtype=tf.float64)
        cov = tf.eye(2, dtype=tf.float64)
        seed = tf.constant([0, 0], dtype=tf.int32)
        samples = sample_particles_cholesky(mean, cov, n_particles=100, state_dim=2, seed=seed)
        assert samples.shape == (100, 2)

    def test_sample_mean(self):
        """T3.12 — Sample mean approaches distribution mean (Monte Carlo)."""
        N = 50000
        mean_np = np.array([3.0, -2.0])
        cov_np = 2.0 * np.eye(2)

        mean_tf = tf.constant(mean_np, dtype=tf.float64)
        cov_tf = tf.constant(cov_np, dtype=tf.float64)
        seed = tf.constant([0, 0], dtype=tf.int32)

        samples = sample_particles_cholesky(mean_tf, cov_tf, N, 2, seed)
        np.testing.assert_allclose(samples.numpy().mean(axis=0), mean_np, atol=0.05)


# ============================================================================
# compute_flow_weights — T3.13 to T3.15
# ============================================================================

class TestComputeFlowWeights:
    """Test weight computation using LinearGaussianModel as the model."""

    @pytest.fixture
    def model_and_data(self):
        """Set up a simple 2D linear-Gaussian model and data for weight tests."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.models.linear_gaussian import LinearGaussianModel

        model = LinearGaussianModel(
            F=np.eye(2), B=0.1 * np.eye(2),
            H=np.eye(2), D=0.5 * np.eye(2),
            dtype=tf.float64,
        )
        N = 20
        seed = tf.constant([42, 0], dtype=tf.int32)

        # Previous particles: near the origin
        particles_prev = tf.random.stateless_normal(
            [N, 2], seed=seed, dtype=tf.float64
        ) * 0.5

        # Observation
        observation = tf.constant([1.0, 0.5], dtype=tf.float64)

        return model, particles_prev, observation, N

    def test_identity_flow_sums_to_one(self, model_and_data):
        """T3.13 — Identity flow (eta_1 == eta_0): weights sum to 1.

        When eta_1 == eta_0, the transition log-probs cancel in
        log(p(eta_1|x)) - log(p(eta_0|x)), leaving only p(z|eta_1)
        and prior terms.  Regardless, the output must be normalized.
        """
        model, particles_prev, observation, N = model_and_data
        # eta_0 = eta_1 = particles_prev (identity flow)
        weights = compute_flow_weights(
            particles_prev, particles_prev, particles_prev,
            observation, model,
        )
        np.testing.assert_allclose(
            tf.reduce_sum(weights).numpy(), 1.0, rtol=1e-5
        )

    def test_uniform_prior_sums_to_one(self, model_and_data):
        """T3.14 — Uniform prior weights → normalized weights sum to 1."""
        model, particles_prev, observation, N = model_and_data

        # eta_0: sampled near the prior mean
        eta_0 = particles_prev + tf.random.stateless_normal(
            [N, 2], seed=tf.constant([7, 0], dtype=tf.int32), dtype=tf.float64
        ) * 0.1
        # eta_1: shifted toward observation (mimic a small flow)
        eta_1 = eta_0 + 0.1 * (observation - eta_0)

        weights = compute_flow_weights(
            eta_1, eta_0, particles_prev, observation, model,
            prev_weights=None, jacobians=None,
        )
        np.testing.assert_allclose(
            tf.reduce_sum(weights).numpy(), 1.0, rtol=1e-6
        )
        assert np.all(weights.numpy() >= 0)

    def test_nan_inputs_finite_weights(self, model_and_data):
        """T3.15 — NaN in eta_1 → weight collapse guard returns finite weights."""
        model, particles_prev, observation, N = model_and_data

        eta_0 = particles_prev
        eta_1 = tf.identity(particles_prev)
        # Inject NaN into first particle
        eta_1_bad = tf.concat([
            tf.constant([[float('nan'), float('nan')]], dtype=tf.float64),
            eta_1[1:],
        ], axis=0)

        weights = compute_flow_weights(
            eta_1_bad, eta_0, particles_prev, observation, model,
        )
        assert np.all(np.isfinite(weights.numpy()))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
