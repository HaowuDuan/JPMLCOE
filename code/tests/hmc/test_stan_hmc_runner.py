"""Tests for stan_hmc_runner.py — Stan-style windowed HMC.

Unit tests cover:
    - window_schedule: tiny/short/long regimes, sum-correctness
    - DualAveragingState: anchor at log(10*eps_init), update reduces error
    - DiagonalMetric.estimate_from_samples: Stan's exact shrinkage formula
    - find_reasonable_epsilon: converges on a 1D Gaussian, respects hard bracket

Integration test:
    - Full stan_warmup + sample_phase on a 2D Gaussian target (no PF / no real
      model — analytic target so it runs in ~5 seconds). Verifies posterior
      mean/std are recovered and adapted M is close to inverse posterior var.

Run:
    cd code && python -m pytest tests/hmc/test_stan_hmc_runner.py -v -s
"""

import math
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.DF.stan_hmc_runner import (
    DiagonalMetric,
    DualAveragingState,
    WarmupInit,
    default_warmup_init,
    dual_averaging_step,
    find_reasonable_epsilon,
    fresh_da_state,
    sample_phase,
    stan_warmup,
    window_schedule,
)


# ----------------------------------------------------------------------------
# 1. window_schedule
# ----------------------------------------------------------------------------

class TestWindowSchedule:
    def test_long_warmup_matches_stan_classic(self):
        """num_warmup=1000 should match Stan's documented [75, 25, 50, 100, 200, 500, 50]."""
        s = window_schedule(1000)
        assert s == [75, 25, 50, 100, 200, 500, 50], f"got {s}"

    def test_long_warmup_sums(self):
        for n in [150, 200, 500, 1000, 2000, 5000]:
            s = window_schedule(n)
            assert sum(s) == n, f"num_warmup={n}: schedule {s} sums to {sum(s)}"

    def test_short_warmup_uses_15_75_10(self):
        """20 <= num_warmup < 150 should use 15/75/10 fallback with single adapt window."""
        s = window_schedule(100)
        # 15% of 100 = 15, 10% = 10, middle = 75
        assert s == [15, 75, 10]
        assert len(s) == 3  # init + single adapt + term

    def test_short_warmup_integer_truncation(self):
        """Stan uses int truncation, not round. 50 → 7 (not 8) for init."""
        s = window_schedule(50)
        assert s == [7, 38, 5]  # int(50*0.15)=7, int(50*0.10)=5, middle=38

    def test_tiny_warmup_skips_metric(self, recwarn):
        """num_warmup < 20 returns single window with warning."""
        s = window_schedule(15)
        assert s == [15]
        assert any("skipping metric adaptation" in str(w.message) for w in recwarn)

    def test_long_threshold_is_150(self):
        """At exactly 150, Stan absolute layout fits: 75+25+50."""
        s = window_schedule(150)
        assert s == [75, 25, 50]

    def test_just_below_long_uses_short_fallback(self):
        """149 should use 15/75/10 fallback."""
        s = window_schedule(149)
        # int(149*0.15)=22, int(149*0.10)=14, middle=149-22-14=113
        assert s == [22, 113, 14]
        assert sum(s) == 149


# ----------------------------------------------------------------------------
# 2. DualAveragingState
# ----------------------------------------------------------------------------

class TestDualAveraging:
    def test_anchor_is_log_10x_eps(self):
        """Stan default: mu = log(10 * eps_init)."""
        state = fresh_da_state(eps_init=0.1, da_shrinkage_factor=10.0)
        assert math.isclose(state.mu, math.log(1.0))  # log(10 * 0.1) = log(1)
        assert math.isclose(state.log_step, math.log(0.1))
        assert state.error_sum == 0.0
        assert state.iter == 0

    def test_step_grows_when_accept_above_target(self):
        """If observed accept > target, error_sum < 0 → log_step > mu."""
        state = fresh_da_state(eps_init=0.01)
        # Repeat many high-accept steps
        for _ in range(20):
            state = dual_averaging_step(state, accept_prob=1.0, target_accept=0.75)
        # Step should have grown above eps_init
        assert math.exp(state.log_step) > 0.01

    def test_step_shrinks_when_accept_below_target(self):
        """If observed accept < target, error_sum > 0 → log_step < mu."""
        state = fresh_da_state(eps_init=0.5)
        for _ in range(20):
            state = dual_averaging_step(state, accept_prob=0.0, target_accept=0.75)
        # Step should have shrunk
        assert math.exp(state.log_step) < 0.5


# ----------------------------------------------------------------------------
# 3. DiagonalMetric
# ----------------------------------------------------------------------------

class TestDiagonalMetric:
    def test_estimate_recovers_inverse_variance(self):
        """Synthetic samples from N(0, diag([1, 4, 9])) should give M ≈ [1, 0.25, 1/9]."""
        rng = np.random.default_rng(42)
        true_var = np.array([1.0, 4.0, 9.0])
        n = 5000
        samples = rng.normal(loc=0.0, scale=np.sqrt(true_var), size=(n, 3))
        samples_tf = tf.constant(samples, dtype=tf.float32)

        metric = DiagonalMetric(tf.ones(3, dtype=tf.float32))
        adapted = metric.estimate_from_samples(samples_tf)

        expected_M = 1.0 / true_var
        actual_M = adapted.M.numpy()
        # Stan shrinkage with n=5000 toward 1e-3 is negligible; expect within 5%
        np.testing.assert_allclose(actual_M, expected_M, rtol=0.05)

    def test_low_count_falls_back_to_identity(self, recwarn):
        """n < 2 → identity M with warning."""
        samples = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        metric = DiagonalMetric(tf.ones(2, dtype=tf.float32))
        adapted = metric.estimate_from_samples(samples)
        np.testing.assert_array_equal(adapted.M.numpy(), [1.0, 1.0])
        assert any("n<2" in str(w.message) for w in recwarn)

    def test_shrinkage_pulls_toward_1e_3_at_low_count(self):
        """At n=5, var_shrunk = (5/(5+5))*var + 1e-3*(5/(5+5))*1 = 0.5*var + 5e-4."""
        # Constant samples → sample var = 0
        samples = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float32)
        samples = tf.transpose(samples)  # shape (5, 1) — n=5, dim=1
        metric = DiagonalMetric(tf.ones(1, dtype=tf.float32))
        adapted = metric.estimate_from_samples(samples)
        # var = 0, var_shrunk = 0.5 * 0 + 5e-4 = 5e-4
        # M = 1 / 5e-4 = 2000
        assert math.isclose(float(adapted.M.numpy()[0]), 2000.0, rel_tol=0.01)


# ----------------------------------------------------------------------------
# 4. find_reasonable_epsilon
# ----------------------------------------------------------------------------

def gaussian_target(q: tf.Tensor) -> tf.Tensor:
    """Standard 2D Gaussian log-density: -0.5 * |q|^2 (unnormalized)."""
    return -0.5 * tf.reduce_sum(q * q)


class TestFindReasonableEpsilon:
    def test_converges_on_2d_gaussian(self):
        """For N(0, I), reasonable eps with L=10 leapfrog is around 0.3-0.6."""
        q = tf.constant([0.0, 0.0], dtype=tf.float32)
        metric = DiagonalMetric(tf.ones(2, dtype=tf.float32))
        eps = find_reasonable_epsilon(
            gaussian_target, q, metric, num_leapfrog=10, eps_init=10.0
        )
        # Should converge into a sensible range, not stuck at extreme.
        assert 0.05 < eps < 5.0, f"eps={eps} out of expected range"

    def test_respects_hard_bracket_on_failure(self):
        """If the target throws non-finite at large eps, search should bracket below it."""

        def cliff_target(q: tf.Tensor) -> tf.Tensor:
            # Standard Gaussian, but log_p becomes NaN when ||q|| > 5
            r2 = tf.reduce_sum(q * q)
            # Use tf.where to produce nan when r2 > 25
            log_p = -0.5 * r2
            log_p = tf.where(r2 > 25.0, tf.constant(float('nan')), log_p)
            return log_p

        q = tf.constant([0.0], dtype=tf.float32)
        metric = DiagonalMetric(tf.ones(1, dtype=tf.float32))
        eps = find_reasonable_epsilon(
            cliff_target, q, metric, num_leapfrog=20, eps_init=10.0,
            max_iters=30,
        )
        # Search should not return a wildly large eps; the cliff at large eps
        # should bracket the search.
        assert eps < 10.0
        assert math.isfinite(eps)


# ----------------------------------------------------------------------------
# 5. Integration test: stan_warmup + sample_phase on 2D Gaussian
# ----------------------------------------------------------------------------

class TestStanWarmupIntegration:
    """End-to-end test on an analytic 2D Gaussian. No PF, no real model.

    Target: N(mu, diag([1, 9])). Posterior std = [1, 3]. Optimal M = [1, 1/9].
    """

    @pytest.fixture(scope="class")
    def target_and_truth(self):
        mu = tf.constant([0.5, -1.0], dtype=tf.float32)
        var = tf.constant([1.0, 9.0], dtype=tf.float32)

        def target(q: tf.Tensor) -> tf.Tensor:
            diff = q - mu
            return -0.5 * tf.reduce_sum(diff * diff / var)

        return target, mu.numpy(), var.numpy()

    def test_warmup_recovers_diagonal_M(self, target_and_truth):
        target, mu, var = target_and_truth
        q0 = tf.constant([0.0, 0.0], dtype=tf.float32)

        result = stan_warmup(
            target_log_prob_fn=target,
            q0=q0,
            num_warmup=200,
            num_leapfrog=10,
            target_accept=0.8,
        )
        # Adapted M should be close to 1/var = [1, 1/9].
        assert isinstance(result.metric, DiagonalMetric)
        actual_M = result.metric.M.numpy()
        expected_M = 1.0 / var
        # 200 warmup with two adaptation windows is enough to get within ~50%
        # of the true value on a well-behaved Gaussian.
        np.testing.assert_allclose(actual_M, expected_M, rtol=0.5)

    def test_sample_phase_recovers_posterior_mean_std(self, target_and_truth):
        target, mu, var = target_and_truth
        q0 = tf.constant([0.0, 0.0], dtype=tf.float32)

        # num_leapfrog=11 (not 10) to dodge fixed-trajectory resonance on
        # this 2D Gaussian. With L=10, eps~0.3-0.6, the trajectory length
        # tau = L*eps ~ 3-6 sits near 2π=6.28, where cos(tau)~1 makes the
        # narrow axis sticky and biases its sample mean toward the init.
        # Stan dodges this with NUTS (dynamic L); fixed-L static HMC has to
        # avoid resonant L explicitly. See Betancourt 2017 / Stan docs.
        warmup_result = stan_warmup(
            target_log_prob_fn=target, q0=q0,
            num_warmup=500, num_leapfrog=11, target_accept=0.8,
        )
        samples, diag = sample_phase(
            target, warmup_result.q, warmup_result.metric, warmup_result.eps,
            num_samples=2000, num_leapfrog=11,
        )
        s = samples.numpy()
        sample_mean = s.mean(axis=0)
        sample_std = s.std(axis=0, ddof=1)

        # Tolerances calibrated for 2000 samples on a 2D Gaussian with
        # disparate variances (var=[1,9]). The narrow axis (var=1) mixes
        # slower in MC time than the wide axis, so we set generous tolerances.
        np.testing.assert_allclose(sample_mean, mu, atol=0.5)
        np.testing.assert_allclose(sample_std, np.sqrt(var), rtol=0.30)
        # Sanity: acceptance rate not absurd
        assert 0.4 < diag["acceptance_rate"] < 1.0
        # E-BFMI should be positive and finite for a well-mixed chain
        assert math.isfinite(diag["e_bfmi"])
        assert diag["e_bfmi"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
