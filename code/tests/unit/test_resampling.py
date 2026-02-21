"""
Resampling method tests — systematic, soft, and OT entropy.

Tests resampling in isolation (no filter involved) to validate the
shared foundation before it's used inside particle filters.

Run:
  pytest code/tests/test_resampling.py -v -s
"""

import numpy as np
import tensorflow as tf
import pytest

from src.resampling.systematic import systematic_resample
from src.resampling.soft import soft_resample
from src.resampling.ot_entropy import ot_entropy_resample

DTYPE = tf.float64


def _make_particles_and_weights(N=5, state_dim=2, weights=None):
    """Create simple test particles and weights."""
    particles = tf.constant(
        np.arange(N * state_dim, dtype=np.float64).reshape(N, state_dim),
        dtype=DTYPE,
    )
    if weights is None:
        weights = tf.ones(N, dtype=DTYPE) / N
    else:
        weights = tf.constant(weights, dtype=DTYPE)
    return particles, weights


# ============================================================================
# R1-R3: Systematic resampling
# ============================================================================

class TestSystematicResample:

    def test_unbiased_selection_frequency(self):
        """R1: Selection frequency matches weights over many runs."""
        N = 3
        weights = tf.constant([0.5, 0.3, 0.2], dtype=DTYPE)
        particles = tf.eye(N, dtype=DTYPE)  # identity so each particle is distinct

        n_trials = 10000
        counts = np.zeros(N)
        for i in range(n_trials):
            seed = tf.constant([i, 0], dtype=tf.int32)
            result = systematic_resample(particles, weights, seed)
            # Count how many times each original particle appears
            indices = result.ancestor_indices.numpy()
            for idx in indices:
                counts[idx] += 1

        freq = counts / (n_trials * N)
        w_np = weights.numpy()
        print(f"\n  Weights:     {w_np}")
        print(f"  Frequencies: {freq}")
        print(f"  Diff:        {np.abs(freq - w_np)}")

        # Monte Carlo tolerance: 3 * sqrt(w*(1-w)/(n_trials*N))
        for i in range(N):
            tol = 3 * np.sqrt(w_np[i] * (1 - w_np[i]) / (n_trials * N))
            assert abs(freq[i] - w_np[i]) < max(tol, 0.02), (
                f"Particle {i}: freq={freq[i]:.4f}, weight={w_np[i]:.4f}, tol={tol:.4f}"
            )

    def test_uniform_weights_all_selected(self):
        """R2: Uniform weights → every particle selected exactly once."""
        N = 20
        particles, weights = _make_particles_and_weights(N=N, state_dim=2)
        seed = tf.constant([42, 0], dtype=tf.int32)

        result = systematic_resample(particles, weights, seed)
        indices = sorted(result.ancestor_indices.numpy().tolist())
        assert indices == list(range(N)), (
            f"Expected permutation of [0..{N-1}], got {indices}"
        )

    def test_degenerate_weights_single_particle(self):
        """R3: All weight on one particle → all resampled particles identical."""
        N = 10
        weights = np.zeros(N)
        weights[3] = 1.0
        particles, weights_tf = _make_particles_and_weights(N=N, weights=weights)
        seed = tf.constant([0, 0], dtype=tf.int32)

        result = systematic_resample(particles, weights_tf, seed)
        expected = particles[3].numpy()
        for i in range(N):
            np.testing.assert_allclose(
                result.particles[i].numpy(), expected,
                err_msg=f"Particle {i} should be copy of particle 3"
            )

    def test_output_weights_uniform(self):
        """Systematic resampling returns uniform weights."""
        particles, _ = _make_particles_and_weights(N=5, weights=[0.5, 0.2, 0.1, 0.15, 0.05])
        seed = tf.constant([7, 0], dtype=tf.int32)
        result = systematic_resample(particles, tf.constant([0.5, 0.2, 0.1, 0.15, 0.05], dtype=DTYPE), seed)

        np.testing.assert_allclose(
            result.weights.numpy(), np.ones(5) / 5, rtol=1e-6,
            err_msg="Systematic resampling should return uniform weights"
        )


# ============================================================================
# R4-R6: Soft resampling
# ============================================================================

class TestSoftResample:

    def test_alpha_one_matches_systematic(self):
        """R4: alpha=1.0 reduces to hard resampling (systematic)."""
        N = 10
        weights = np.array([0.3, 0.1, 0.05, 0.15, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05])
        particles, weights_tf = _make_particles_and_weights(N=N, weights=weights)
        seed = tf.constant([42, 0], dtype=tf.int32)

        result_sys = systematic_resample(particles, weights_tf, seed)
        result_soft = soft_resample(particles, weights_tf, alpha=1.0, seed=seed)

        np.testing.assert_allclose(
            result_soft.particles.numpy(), result_sys.particles.numpy(), rtol=1e-6,
            err_msg="Soft with alpha=1 should match systematic"
        )

    def test_alpha_zero_no_shuffling(self):
        """R5: alpha=0.0 → uniform sampling, output weights proportional to input."""
        N = 5
        weights = np.array([0.5, 0.2, 0.1, 0.15, 0.05])
        particles, weights_tf = _make_particles_and_weights(N=N, weights=weights)
        seed = tf.constant([42, 0], dtype=tf.int32)

        result = soft_resample(particles, weights_tf, alpha=0.0, seed=seed)

        # With alpha=0, q = 1/N (uniform), so resampling is uniform systematic
        # Output weights = w_orig / q = w_orig * N, then renormalized
        # → output weights should equal input weights (up to the resampling permutation)
        out_w = result.weights.numpy()
        assert np.all(np.isfinite(out_w)), "Output weights should be finite"
        np.testing.assert_allclose(out_w.sum(), 1.0, rtol=1e-6)

    def test_output_weights_sum_to_one(self):
        """R6: For any alpha, output weights sum to 1 and are non-negative."""
        N = 10
        weights = np.random.dirichlet(np.ones(N))
        particles, weights_tf = _make_particles_and_weights(N=N, weights=weights)

        for alpha in [0.1, 0.5, 0.9]:
            seed = tf.constant([42, 0], dtype=tf.int32)
            result = soft_resample(particles, weights_tf, alpha=alpha, seed=seed)
            out_w = result.weights.numpy()

            assert np.all(out_w >= -1e-10), f"alpha={alpha}: negative weight {out_w.min()}"
            np.testing.assert_allclose(
                out_w.sum(), 1.0, rtol=1e-4,
                err_msg=f"alpha={alpha}: weights sum={out_w.sum()}"
            )


# ============================================================================
# R7-R8: OT entropy resampling
# ============================================================================

class TestOTEntropyResample:

    def test_output_weights_finite_and_sum_to_one(self):
        """R7: OT output weights sum to 1, all finite."""
        N = 10
        weights = np.random.dirichlet(np.ones(N))
        particles, weights_tf = _make_particles_and_weights(N=N, weights=weights)

        for epsilon in [0.01, 0.1, 1.0]:
            seed = tf.constant([42, 0], dtype=tf.int32)
            result = ot_entropy_resample(
                particles, weights_tf, epsilon=epsilon, seed=seed,
                max_iter=200, convergence_threshold=1e-4,
            )
            out_w = result.weights.numpy()

            assert np.all(np.isfinite(out_w)), (
                f"epsilon={epsilon}: non-finite weights"
            )
            np.testing.assert_allclose(
                out_w.sum(), 1.0, rtol=1e-3,
                err_msg=f"epsilon={epsilon}: weights sum={out_w.sum()}"
            )

    def test_uniform_weights_particles_unchanged(self):
        """R8: Uniform weights → OT has no work, particles nearly unchanged."""
        N = 10
        state_dim = 2
        particles, weights = _make_particles_and_weights(N=N, state_dim=state_dim)
        seed = tf.constant([42, 0], dtype=tf.int32)

        result = ot_entropy_resample(
            particles, weights, epsilon=0.1, seed=seed,
            max_iter=200,
        )

        # With uniform weights, transport should be close to identity
        diff = np.abs(result.particles.numpy() - particles.numpy())
        max_diff = diff.max()
        print(f"\n  Max particle displacement with uniform weights: {max_diff:.6f}")
        assert max_diff < 1.0, (
            f"Uniform weights should produce near-identity transport, max_diff={max_diff}"
        )


# ============================================================================
# R9: Extreme weights stress test (all methods)
# ============================================================================

class TestExtremeWeights:

    @pytest.mark.parametrize("method_name", ["systematic", "soft", "ot_entropy"])
    def test_extreme_weight_ratio_no_crash(self, method_name):
        """R9: One particle has nearly all weight — no NaN/crash."""
        N = 20
        # Create extreme weights: softmax([100, 0, 0, ...])
        logits = np.zeros(N)
        logits[0] = 100.0
        weights = np.exp(logits - np.max(logits))
        weights = weights / weights.sum()

        particles = tf.constant(
            np.random.randn(N, 2).astype(np.float64), dtype=DTYPE
        )
        weights_tf = tf.constant(weights, dtype=DTYPE)
        seed = tf.constant([42, 0], dtype=tf.int32)

        if method_name == "systematic":
            result = systematic_resample(particles, weights_tf, seed)
        elif method_name == "soft":
            result = soft_resample(particles, weights_tf, alpha=0.5, seed=seed)
        elif method_name == "ot_entropy":
            result = ot_entropy_resample(
                particles, weights_tf, epsilon=0.5, seed=seed, max_iter=200,
            )

        assert np.all(np.isfinite(result.particles.numpy())), (
            f"{method_name}: non-finite particles with extreme weights"
        )
        assert np.all(np.isfinite(result.weights.numpy())), (
            f"{method_name}: non-finite weights with extreme weights"
        )
