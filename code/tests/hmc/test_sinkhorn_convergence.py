"""Test Sinkhorn convergence: does the transport plan satisfy marginal constraints?

Checks:
1. Did Sinkhorn reach max_iter or converge early?
2. Do the marginals of T match the target (weights) and source (uniform)?
3. How does convergence depend on epsilon, N, and max_iter?

Run: python -m pytest tests/hmc/test_sinkhorn_convergence.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.models.linear_gaussian import LinearGaussianModel
from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.resampling.ot_entropy import (
    sinkhorn_iteration, compute_cost_matrix,
    compute_transport_matrix_from_potentials,
)
from src.DF.differentiable_model import DifferentiableModel


DTYPE = tf.float64
SEED = 42


def _run_sinkhorn_and_check(particles, weights, epsilon, max_iter=100, threshold=1e-3):
    """Run Sinkhorn and report convergence diagnostics."""
    N = particles.shape[0]
    log_weights = tf.math.log(weights + tf.constant(1e-10, dtype=DTYPE))

    # Normalize particles (same as ot_entropy_resample)
    mean = tf.reduce_mean(particles, axis=0, keepdims=True)
    centered = particles - mean
    std = tf.math.reduce_std(particles)
    dimension = tf.cast(tf.shape(particles)[-1], DTYPE)
    scale_factor = std * tf.sqrt(dimension) + tf.constant(1e-8, dtype=DTYPE)
    scaled = centered / scale_factor

    cost_matrix = compute_cost_matrix(scaled, scaled)
    epsilon_tensor = tf.cast(epsilon, DTYPE)
    alpha_init = tf.zeros_like(log_weights)
    beta_init = tf.zeros_like(log_weights)

    alpha, beta, n_iter = sinkhorn_iteration(
        log_weights, cost_matrix, epsilon_tensor,
        alpha_init, beta_init,
        max_iter=max_iter, threshold=threshold,
    )

    T = compute_transport_matrix_from_potentials(
        scaled, alpha, beta, epsilon, log_weights
    )

    # Check marginals
    # T is the resampling matrix: row sums = 1 (each new particle is a convex combo)
    # col sums = N * weights (total mass transported from each old particle)
    row_sums = tf.reduce_sum(T, axis=1)  # should be 1
    col_sums = tf.reduce_sum(T, axis=0)  # should be N * weights

    target_row = tf.ones([N], dtype=DTYPE)
    target_col = tf.cast(N, DTYPE) * weights

    row_err = tf.reduce_max(tf.abs(row_sums - target_row)).numpy()
    col_err = tf.reduce_max(tf.abs(col_sums - target_col)).numpy()
    row_rel_err = (tf.reduce_max(tf.abs(row_sums - target_row) / target_row)).numpy()
    col_rel_err = (tf.reduce_max(tf.abs(col_sums - target_col) / (target_col + 1e-30))).numpy()

    n_iter_val = int(n_iter.numpy())
    hit_max = n_iter_val >= max_iter - 1

    return {
        'n_iter': n_iter_val,
        'hit_max_iter': hit_max,
        'row_max_err': row_err,
        'col_max_err': col_err,
        'row_max_rel_err': row_rel_err,
        'col_max_rel_err': col_rel_err,
        'T_min': float(tf.reduce_min(T).numpy()),
        'T_max': float(tf.reduce_max(T).numpy()),
        'T_sum': float(tf.reduce_sum(T).numpy()),
        'N': int(N),
    }


def _report(label, result):
    print(f"\n  [{label}]")
    print(f"    Iterations:       {result['n_iter']} (hit max: {result['hit_max_iter']})")
    print(f"    Row marginal err: {result['row_max_err']:.2e} (rel: {result['row_max_rel_err']:.2e})")
    print(f"    Col marginal err: {result['col_max_err']:.2e} (rel: {result['col_max_rel_err']:.2e})")
    print(f"    T range:          [{result['T_min']:.2e}, {result['T_max']:.2e}]")
    print(f"    T sum:            {result['T_sum']:.4f} (should be N={result['N']} particles)")


class TestSinkhornConvergenceUniform:
    """Baseline: uniform weights. Sinkhorn should converge trivially."""

    def test_uniform_weights(self):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        weights = tf.ones(200, dtype=DTYPE) / 200.0

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        _report("Uniform weights, N=200, eps=0.5", result)
        assert not result['hit_max_iter'], "Should converge for uniform weights"
        assert result['row_max_err'] < 2e-3
        assert result['col_max_err'] < 2e-3


class TestSinkhornConvergenceSkewed:
    """Skewed weights — harder for Sinkhorn."""

    def test_skewed_weights(self):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        # Skewed: a few particles get most weight
        raw_w = rng.exponential(1.0, 200)
        raw_w[0:5] *= 100  # 5 particles dominate
        weights = tf.constant(raw_w / raw_w.sum(), dtype=DTYPE)

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        _report("Skewed weights, N=200, eps=0.5", result)
        assert result['row_max_err'] < 1e-2, f"Row err {result['row_max_err']:.2e}"
        assert result['col_max_err'] < 1e-2, f"Col err {result['col_max_err']:.2e}"


class TestSinkhornConvergenceVsEpsilon:
    """How does convergence change with epsilon?"""

    @pytest.mark.parametrize("epsilon", [0.01, 0.1, 0.5, 1.0, 5.0])
    def test_epsilon(self, epsilon):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        raw_w = rng.exponential(1.0, 200)
        weights = tf.constant(raw_w / raw_w.sum(), dtype=DTYPE)

        result = _run_sinkhorn_and_check(particles, weights, epsilon=epsilon)
        _report(f"eps={epsilon}", result)


class TestSinkhornConvergenceVsMaxIter:
    """Does max_iter=100 suffice?"""

    @pytest.mark.parametrize("max_iter", [10, 50, 100, 200, 500])
    def test_max_iter(self, max_iter):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        raw_w = rng.exponential(1.0, 200)
        weights = tf.constant(raw_w / raw_w.sum(), dtype=DTYPE)

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5, max_iter=max_iter)
        _report(f"max_iter={max_iter}", result)


class TestSinkhornConvergenceLEDH:
    """Test with actual LEDH particles and weights from the LG model."""

    def test_lg_ledh_particles(self):
        model = LinearGaussianModel(
            F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
            obs_noise_std=1.0, dtype=DTYPE,
        )
        rng = np.random.default_rng(SEED)
        _, _, obs = generate_data(model, T=5, rng=rng)
        obs_tf = tf.constant(obs, dtype=DTYPE)

        diff_model = DifferentiableModel(model, ['obs_noise_std'])
        filt = LEDHParticleFlowFilterHMC(
            model=diff_model,
            n_particles=200,
            n_lambda_steps=15,
            resampling_method='ot_entropy',
            resampling_config={'epsilon': 0.5},
            weight_clip_range=50.0,
            stop_gradient_resampling=False,
            eager_mode=True,
            always_resample=True,
        )
        filt.initialize(random_seed=SEED)

        # Run a few timesteps to get realistic particles and weights
        for t in range(3):
            filt.predict(t=t+1)
            filt.update(obs_tf[t])

        particles = filt.particles.value()
        weights = filt.weights.value()

        print(f"\n  LEDH particles after 3 timesteps:")
        print(f"    Weight range: [{float(tf.reduce_min(weights).numpy()):.4e}, {float(tf.reduce_max(weights).numpy()):.4e}]")
        print(f"    Weight ESS/N: {float((1.0 / tf.reduce_sum(weights**2)).numpy() / 200):.4f}")

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        _report("LEDH LG particles t=3", result)
        assert result['row_max_err'] < 1e-2
        assert result['col_max_err'] < 1e-2

    def test_sv2d_ledh_particles(self):
        model = StochasticVolatility2DModel(
            a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0, dtype=DTYPE,
        )
        rng = np.random.default_rng(SEED)
        _, _, obs = generate_data(model, T=5, rng=rng)
        obs_tf = tf.constant(obs, dtype=DTYPE)

        diff_model = DifferentiableModel(model, ['sigma2'])
        filt = LEDHParticleFlowFilterHMC(
            model=diff_model,
            n_particles=200,
            n_lambda_steps=15,
            resampling_method='ot_entropy',
            resampling_config={'epsilon': 0.5},
            weight_clip_range=50.0,
            stop_gradient_resampling=False,
            eager_mode=True,
            always_resample=True,
        )
        filt.initialize(random_seed=SEED)

        for t in range(3):
            filt.predict(t=t+1)
            filt.update(obs_tf[t])

        particles = filt.particles.value()
        weights = filt.weights.value()

        print(f"\n  LEDH SV2D particles after 3 timesteps:")
        print(f"    Weight range: [{float(tf.reduce_min(weights).numpy()):.4e}, {float(tf.reduce_max(weights).numpy()):.4e}]")
        print(f"    Weight ESS/N: {float((1.0 / tf.reduce_sum(weights**2)).numpy() / 200):.4f}")

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        _report("LEDH SV2D particles t=3", result)
        assert result['row_max_err'] < 1e-2
        assert result['col_max_err'] < 1e-2
