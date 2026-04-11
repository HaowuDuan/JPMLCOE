"""Sinkhorn convergence tests: do transport plans satisfy marginal constraints?

Checks:
1. Convergence on uniform / skewed weights
2. Sensitivity to epsilon and max_iter
3. Realistic LEDH particles from LG and SV2D

Each case prints + saves to tests/hmc/results/test_sinkhorn_convergence.json.

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

from _gradient_test_utils import save_result, reset_results


DTYPE = tf.float64
SEED = 42

# Tolerances for marginal constraint violation
TOL_UNIFORM = 2e-3      # uniform weights — Sinkhorn should converge trivially
TOL_REAL = 1e-2         # skewed / real-particle weights
TOL_LOOSE = 1e-1        # for epsilon sweep where some configs should naturally fail to fully converge
MAX_ITER_DEFAULT = 100


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


def _run_sinkhorn_and_check(particles, weights, epsilon, max_iter=MAX_ITER_DEFAULT, threshold=1e-3):
    """Run Sinkhorn and report convergence diagnostics."""
    N = particles.shape[0]
    log_weights = tf.math.log(weights + tf.constant(1e-10, dtype=DTYPE))

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

    row_sums = tf.reduce_sum(T, axis=1)
    col_sums = tf.reduce_sum(T, axis=0)
    target_row = tf.ones([N], dtype=DTYPE)
    target_col = tf.cast(N, DTYPE) * weights

    row_err = float(tf.reduce_max(tf.abs(row_sums - target_row)).numpy())
    col_err = float(tf.reduce_max(tf.abs(col_sums - target_col)).numpy())
    n_iter_val = int(n_iter.numpy())

    return {
        'N': int(N),
        'epsilon': float(epsilon),
        'max_iter': int(max_iter),
        'n_iter': n_iter_val,
        'hit_max_iter': n_iter_val >= max_iter - 1,
        'row_max_err': row_err,
        'col_max_err': col_err,
        'T_min': float(tf.reduce_min(T).numpy()),
        'T_max': float(tf.reduce_max(T).numpy()),
        'T_sum': float(tf.reduce_sum(T).numpy()),
    }


def _report_and_save(case_name, result, tol):
    print(f"\n  [{case_name}]")
    print(f"    epsilon={result['epsilon']}  max_iter={result['max_iter']}  N={result['N']}")
    print(f"    n_iter={result['n_iter']} (hit_max={result['hit_max_iter']})")
    print(f"    row_max_err={result['row_max_err']:.2e}  col_max_err={result['col_max_err']:.2e}")
    print(f"    T range=[{result['T_min']:.2e}, {result['T_max']:.2e}]  T sum={result['T_sum']:.4f}")
    passed = (result['row_max_err'] <= tol) and (result['col_max_err'] <= tol)
    print(f"    tol={tol}  {'PASS' if passed else 'FAIL'}")

    save_result(__file__, {
        'case_name': case_name,
        'tolerance': float(tol),
        'passed': bool(passed),
        **result,
    })
    return passed


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

class TestSinkhornUniform:
    """Baseline: uniform weights — Sinkhorn should converge trivially."""

    def test_uniform_weights(self):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        weights = tf.ones(200, dtype=DTYPE) / 200.0

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        passed = _report_and_save("Uniform weights N=200 eps=0.5", result, TOL_UNIFORM)
        assert not result['hit_max_iter'], "Should converge for uniform weights"
        assert passed, f"Marginal error exceeds {TOL_UNIFORM}"


class TestSinkhornSkewed:
    """Skewed weights — harder for Sinkhorn but should still converge."""

    def test_skewed_weights(self):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        raw_w = rng.exponential(1.0, 200)
        raw_w[0:5] *= 100
        weights = tf.constant(raw_w / raw_w.sum(), dtype=DTYPE)

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        passed = _report_and_save("Skewed weights N=200 eps=0.5", result, TOL_REAL)
        assert passed, f"Marginal error exceeds {TOL_REAL}"


class TestSinkhornVsEpsilon:
    """Convergence at different epsilon values.
    For epsilon in [0.1, 5.0] we require marginals < TOL_LOOSE.
    Smaller epsilon may not converge in 100 iterations — we still record."""

    @pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0, 5.0])
    def test_epsilon(self, epsilon):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        raw_w = rng.exponential(1.0, 200)
        weights = tf.constant(raw_w / raw_w.sum(), dtype=DTYPE)

        result = _run_sinkhorn_and_check(particles, weights, epsilon=epsilon)
        passed = _report_and_save(f"eps={epsilon}", result, TOL_LOOSE)
        assert passed, f"Marginal error exceeds {TOL_LOOSE} at epsilon={epsilon}"


class TestSinkhornVsMaxIter:
    """Convergence vs max_iter. At max_iter=200 we require TOL_REAL convergence."""

    @pytest.mark.parametrize("max_iter,tol", [
        (10, TOL_LOOSE),
        (50, TOL_LOOSE),
        (100, TOL_LOOSE),
        (200, TOL_REAL),
        (500, TOL_REAL),
    ])
    def test_max_iter(self, max_iter, tol):
        rng = np.random.default_rng(SEED)
        particles = tf.constant(rng.normal(0, 1, (200, 1)), dtype=DTYPE)
        raw_w = rng.exponential(1.0, 200)
        weights = tf.constant(raw_w / raw_w.sum(), dtype=DTYPE)

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5, max_iter=max_iter)
        passed = _report_and_save(f"max_iter={max_iter}", result, tol)
        assert passed, f"Marginal error exceeds {tol} at max_iter={max_iter}"


class TestSinkhornRealParticles:
    """Test with realistic LEDH particles from LG and SV2D models."""

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
        for t in range(3):
            filt.predict(t=t + 1)
            filt.update(obs_tf[t])

        particles = filt.particles.value()
        weights = filt.weights.value()

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        passed = _report_and_save("LEDH LG particles t=3", result, TOL_REAL)
        assert passed, f"Marginal error exceeds {TOL_REAL}"

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
            filt.predict(t=t + 1)
            filt.update(obs_tf[t])

        particles = filt.particles.value()
        weights = filt.weights.value()

        result = _run_sinkhorn_and_check(particles, weights, epsilon=0.5)
        passed = _report_and_save("LEDH SV2D particles t=3", result, TOL_REAL)
        assert passed, f"Marginal error exceeds {TOL_REAL}"
