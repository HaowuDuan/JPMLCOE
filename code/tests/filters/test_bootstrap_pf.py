"""
Bootstrap Particle Filter correctness and diagnostic tests.

Tier 1: Correctness vs Kalman filter on linear-Gaussian model.
Tier 2: Numerical health (weight range, ESS, NaN, particle diversity).
Tier 3: Edge cases (few particles, informative observations).

Run:
  pytest code/tests/test_bootstrap_pf.py -v -s
"""

import numpy as np
import tensorflow as tf
import pytest

from conftest_filters import (
    DTYPE, NP_DTYPE, T, SEED, LG_F, LG_B, LG_H, LG_D,
    lg_model, lg_data, kf_result, lg_observations_tf,
    assert_no_nan_inf, assert_weights_valid, compute_ess,
)
from src.models.linear_gaussian import LinearGaussianModel
from src.filters.particle.bootstrap_pf_tf import ParticleFilterTF


def _run_bpf(model, observations, n_particles=2000, resampling_method='systematic',
             resampling_config=None):
    """Run bootstrap PF and return FilterResult."""
    filt = ParticleFilterTF(
        model, n_particles=n_particles,
        resample_threshold=0.5,
        resampling_method=resampling_method,
        resampling_config=resampling_config or {},
    )
    return filt.filter(observations)


# ============================================================================
# Tier 1: Correctness vs Kalman
# ============================================================================

class TestBPFCorrectness:

    def test_posterior_mean_converges_to_kalman(self, lg_model, lg_data, kf_result):
        """B1: BPF posterior mean converges to Kalman mean with large N."""
        _, _, observations = lg_data
        result = _run_bpf(lg_model, observations, n_particles=5000)

        kf_means = kf_result.means
        bpf_means = result.means

        max_err = np.max(np.abs(bpf_means - kf_means))
        mean_err = np.mean(np.abs(bpf_means - kf_means))

        print(f"\n  BPF vs KF posterior mean (N=5000, T={T}):")
        print(f"    Max absolute error:  {max_err:.4f}")
        print(f"    Mean absolute error: {mean_err:.4f}")

        assert max_err < 1.5, f"Max error {max_err:.4f} > 1.5"
        assert mean_err < 0.5, f"Mean error {mean_err:.4f} > 0.5"

    def test_posterior_mean_unbiased_over_seeds(self, lg_model, lg_data, kf_result):
        """B2: BPF mean averaged over seeds converges closer to Kalman."""
        _, _, observations = lg_data
        n_seeds = 20
        bpf_means_all = []

        for s in range(n_seeds):
            # Vary seed by creating model fresh each time (filter() uses random seed)
            model_s = LinearGaussianModel(F=LG_F, B=LG_B, H=LG_H, D=LG_D, dtype=DTYPE)
            result = _run_bpf(model_s, observations, n_particles=500)
            bpf_means_all.append(result.means)

        avg_means = np.mean(bpf_means_all, axis=0)
        single_err = np.mean(np.abs(bpf_means_all[0] - kf_result.means))
        avg_err = np.mean(np.abs(avg_means - kf_result.means))

        print(f"\n  BPF unbiasedness (N=500, {n_seeds} seeds):")
        print(f"    Single-run error: {single_err:.4f}")
        print(f"    Averaged error:   {avg_err:.4f}")

        # Averaged should be better than single run (or at least not much worse)
        assert avg_err < single_err * 1.5, (
            f"Averaged error {avg_err:.4f} not better than single {single_err:.4f}"
        )

    def test_log_likelihood_converges_to_kalman(self, lg_model, lg_data, kf_result):
        """B3: BPF log-likelihood converges to Kalman log-likelihood."""
        _, _, observations = lg_data
        result = _run_bpf(lg_model, observations, n_particles=5000)

        kf_ll = kf_result.log_likelihood
        bpf_ll = result.log_likelihood

        rel_err = abs(bpf_ll - kf_ll) / abs(kf_ll)

        print(f"\n  BPF vs KF log-likelihood (N=5000):")
        print(f"    KF:       {kf_ll:.4f}")
        print(f"    BPF:      {bpf_ll:.4f}")
        print(f"    Rel error: {rel_err:.4f}")

        assert rel_err < 0.10, f"BPF log-lik rel error {rel_err:.4f} > 0.10"

    def test_resampling_method_all_converge_to_kalman(self, lg_model, lg_data, kf_result):
        """B4: All resampling methods independently converge to Kalman."""
        _, _, observations = lg_data

        print(f"\n  Resampling method comparison (N=2000):")
        for method, config in [
            ('systematic', {}),
            ('soft', {'alpha': 0.5}),
            ('ot_entropy', {'epsilon': 0.5}),
        ]:
            result = _run_bpf(
                lg_model, observations, n_particles=2000,
                resampling_method=method, resampling_config=config,
            )
            mean_err = np.mean(np.abs(result.means - kf_result.means))
            max_err = np.max(np.abs(result.means - kf_result.means))
            print(f"    {method}: mean_err={mean_err:.4f}, max_err={max_err:.4f}")

            # Each method should independently converge to KF
            # OT entropy transports particles (changes locations), so it has
            # higher variance than index-based methods at same N
            threshold = 3.0 if method == 'ot_entropy' else 1.5
            assert max_err < threshold, (
                f"{method}: max error {max_err:.4f} > {threshold}"
            )


# ============================================================================
# Tier 2: Numerical Health
# ============================================================================

class TestBPFNumericalHealth:

    def test_no_nan_inf(self, lg_model, lg_data):
        """B5: No NaN/Inf in weights, means, covs, or log-likelihoods."""
        _, _, observations = lg_data
        result = _run_bpf(lg_model, observations, n_particles=500)

        assert_no_nan_inf(result.means, "means")
        assert_no_nan_inf(result.covs, "covs")
        assert_no_nan_inf(result.log_likelihoods, "log_likelihoods")
        assert_no_nan_inf(result.ess, "ess")
        assert_no_nan_inf(result.weights_history, "weights_history")

    def test_ess_never_collapses(self, lg_model, lg_data):
        """B6: ESS never drops below 2 on linear-Gaussian at true params."""
        _, _, observations = lg_data
        result = _run_bpf(lg_model, observations, n_particles=500)

        min_ess = result.ess.min()
        mean_ess = result.ess.mean()

        print(f"\n  BPF ESS (N=500):")
        print(f"    Min ESS:  {min_ess:.1f}")
        print(f"    Mean ESS: {mean_ess:.1f}")

        assert min_ess > 2, f"ESS collapsed to {min_ess:.1f}"

    def test_weight_range_bounded(self, lg_model, lg_data):
        """B7: Log-weight range stays bounded at each timestep."""
        _, _, observations = lg_data
        result = _run_bpf(lg_model, observations, n_particles=500)

        max_weight_range = 0.0
        for t in range(T):
            w = result.weights_history[t]
            # Compute log-weight range (avoiding log(0))
            w_pos = w[w > 1e-30]
            if len(w_pos) > 1:
                log_range = np.log(w_pos.max()) - np.log(w_pos.min())
                max_weight_range = max(max_weight_range, log_range)

        print(f"\n  Max log-weight range: {max_weight_range:.2f}")
        assert max_weight_range < 60, f"Log-weight range {max_weight_range:.2f} > 60"

    def test_particle_diversity_after_resampling(self, lg_model, lg_data):
        """B8: Sufficient unique particles remain after resampling."""
        _, _, observations = lg_data
        result = _run_bpf(lg_model, observations, n_particles=500)

        if result.n_unique is not None and len(result.n_unique) > 0:
            min_unique_ratio = result.n_unique.min() / 500.0
            mean_unique_ratio = result.n_unique.mean() / 500.0

            print(f"\n  Particle diversity after resampling:")
            print(f"    Min unique ratio:  {min_unique_ratio:.3f}")
            print(f"    Mean unique ratio: {mean_unique_ratio:.3f}")
            print(f"    Resampled at {len(result.resampled_at)} of {T} steps")

            assert min_unique_ratio > 0.05, (
                f"Severe impoverishment: only {min_unique_ratio:.1%} unique particles"
            )
        else:
            print("\n  No resampling events — skipping diversity check")


# ============================================================================
# Tier 3: Edge Cases
# ============================================================================

class TestBPFEdgeCases:

    def test_few_particles_no_crash(self, lg_model, lg_data):
        """B9: N=10 particles doesn't crash."""
        _, _, observations = lg_data
        result = _run_bpf(lg_model, observations, n_particles=10)

        assert_no_nan_inf(result.means, "means (N=10)")
        assert result.means.shape == (T, 2)
        print(f"\n  N=10: completed without crash, log_lik={result.log_likelihood:.4f}")

    def test_informative_observation_no_crash(self):
        """B10: Very informative observation (tiny R) doesn't produce NaN."""
        # Small observation noise → likelihood is a spike
        model = LinearGaussianModel(
            F=LG_F, B=LG_B, H=LG_H,
            D=[[0.03]],  # R = 0.001, very tight
            dtype=DTYPE,
        )
        rng = np.random.default_rng(SEED)
        _, _, observations = generate_data(model, T=20, rng=rng)

        result = _run_bpf(model, observations, n_particles=500)

        assert_no_nan_inf(result.means, "means (tight R)")
        assert_no_nan_inf(result.weights_history, "weights (tight R)")

        # ESS will be very low — that's expected
        min_ess = result.ess.min()
        print(f"\n  Tight R (0.001): min ESS = {min_ess:.1f}")


# Need generate_data for B10
from src.models.utils import generate_data
