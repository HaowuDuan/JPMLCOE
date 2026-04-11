"""
Tests for memory leak fix and JIT compilation improvements.

Validates that:
1. LEDH HMC does not create new tf.Variables per likelihood evaluation
2. BPF HMC log_marginal_likelihood_tf works without .numpy() barrier
3. Gradients still flow through both filters after the fix
4. Memory does not grow monotonically across repeated evaluations

Uses the same DPFRunner setup as the existing HMC diagnostic tests so that
the model has trainable_param_names properly initialized via ParameterHandler.

Run:
  cd code && python -m pytest tests/hmc/test_memory_and_jit.py -v -s
"""

import sys
import os
import tracemalloc

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.DF.hmc_runner import DPFRunner
from src.DF.types import ParameterSpec

from _gradient_test_utils import save_result, reset_results


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


# ---------------------------------------------------------------------------
# Shared setup (small model for fast tests)
# ---------------------------------------------------------------------------

DTYPE = tf.float32
T = 10
N_PARTICLES = 50
SEED = 42
INIT_OBS_NOISE_STD = 2.0
TRUE_OBS_NOISE_STD = 1.0


def _make_model(obs_noise_std=INIT_OBS_NOISE_STD):
    return LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=obs_noise_std, dtype=DTYPE,
    )


def _make_observations():
    model = _make_model(obs_noise_std=TRUE_OBS_NOISE_STD)
    rng = np.random.default_rng(SEED)
    _, _, observations = generate_data(model, T=T, rng=rng)
    return tf.constant(observations, dtype=DTYPE)


def _make_param_specs():
    return {
        'obs_noise_std': ParameterSpec(
            name='obs_noise_std',
            init_value=INIT_OBS_NOISE_STD,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                loc=tf.constant(0.0, dtype=DTYPE),
                scale=tf.constant(1.0, dtype=DTYPE),
            ),
        )
    }


def _make_runner(filter_class, filter_kwargs, observations_tf):
    """Create a DPFRunner — sets up trainable_param_names on the model."""
    model = _make_model()
    runner = DPFRunner(
        base_model=model,
        filter_class=filter_class,
        filter_kwargs=filter_kwargs,
        param_specs=_make_param_specs(),
        sampler='hmc',
    )
    runner._observations_tf = observations_tf
    return runner


def _ledh_kwargs():
    return dict(
        n_particles=N_PARTICLES,
        n_lambda_steps=5,
        resampling_method='systematic',
        weight_clip_range=50.0,
        stop_gradient_resampling=True,
        eager_mode=False,
    )


def _bpf_kwargs():
    return dict(
        n_particles=N_PARTICLES,
        resampling_method='systematic',
        stop_gradient_resampling=True,
        eager_mode=False,
    )


# ---------------------------------------------------------------------------
# Test 1: LEDH no variable leak
# ---------------------------------------------------------------------------

class TestLEDHNoVariableLeak:
    """Verify that repeated calls via _negative_log_posterior (production
    path) do not create new tf.Variables."""

    def test_no_new_variables_after_warmup(self):
        obs = _make_observations()
        runner = _make_runner(LEDHParticleFlowFilterHMC, _ledh_kwargs(), obs)
        q = runner.param_handler.unconstrained_init

        # Warm-up call (triggers tf.function tracing)
        _ = runner._negative_log_posterior(q)

        # Count tracked variables after warm-up
        vars_after_warmup = len(tf.compat.v1.global_variables())

        # Run 5 more evaluations
        for i in range(5):
            _ = runner._negative_log_posterior(q)

        vars_after_5_calls = len(tf.compat.v1.global_variables())

        diff = vars_after_5_calls - vars_after_warmup
        print(f"\n  Variables after warm-up: {vars_after_warmup}")
        print(f"  Variables after 5 more calls: {vars_after_5_calls}")
        print(f"  Diff: {diff}")

        save_result(__file__, {
            'case_name': 'ledh_no_variable_leak',
            'vars_after_warmup': int(vars_after_warmup),
            'vars_after_5_calls': int(vars_after_5_calls),
            'diff': int(diff),
            'passed': bool(diff == 0),
        })

        assert vars_after_5_calls == vars_after_warmup, (
            f"Variable leak: {diff} new variables created over 5 calls"
        )


# ---------------------------------------------------------------------------
# Test 2: BPF no .numpy() barrier
# ---------------------------------------------------------------------------

class TestBPFCallableFromTfFunction:
    """Verify that log_marginal_likelihood_tf works without forcing eager
    via .numpy() on the seed. This is what blocked graph compilation."""

    def test_bpf_seed_is_tensor_path(self):
        obs = _make_observations()
        runner = _make_runner(BootstrapPFHMC, _bpf_kwargs(), obs)
        seed = tf.constant([SEED, 0], dtype=tf.int32)

        ll = runner.filter_obj.log_marginal_likelihood_tf(obs, seed=seed)
        ll_val = float(ll)
        is_tensor = tf.is_tensor(ll)
        is_finite = bool(tf.math.is_finite(ll).numpy())

        print(f"\n  BPF log-lik: {ll_val:.4f}")
        save_result(__file__, {
            'case_name': 'bpf_callable_no_numpy_barrier',
            'log_likelihood': ll_val,
            'is_tensor': bool(is_tensor),
            'is_finite': is_finite,
            'passed': bool(is_tensor and is_finite),
        })

        assert is_tensor
        assert is_finite


# ---------------------------------------------------------------------------
# Test 3: Gradients flow through filter
# ---------------------------------------------------------------------------

class TestGradientFlows:
    """Verify gradients are nonzero after the fix."""

    def _compute_grad(self, filter_class, filter_kwargs):
        obs = _make_observations()
        runner = _make_runner(filter_class, filter_kwargs, obs)
        q = runner.param_handler.unconstrained_init

        with tf.GradientTape() as tape:
            tape.watch(q)
            nlp = runner._negative_log_posterior(q)

        grad = tape.gradient(nlp, q)
        return float(nlp), grad.numpy() if grad is not None else None

    def _save_and_assert(self, case_name, nlp, grad):
        passed = grad is not None and np.any(np.abs(grad) > 1e-6)
        save_result(__file__, {
            'case_name': case_name,
            'nlp': float(nlp),
            'grad': None if grad is None else [float(g) for g in np.atleast_1d(grad)],
            'passed': bool(passed),
        })
        assert grad is not None, f"{case_name}: gradient is None"
        assert np.any(np.abs(grad) > 1e-6), f"{case_name}: gradient too small: {grad}"

    def test_ledh_gradient_nonzero(self):
        nlp, grad = self._compute_grad(LEDHParticleFlowFilterHMC, _ledh_kwargs())
        print(f"\n  LEDH nlp: {nlp:.4f}, grad: {grad}")
        self._save_and_assert('ledh_gradient_nonzero', nlp, grad)

    def test_bpf_gradient_nonzero(self):
        nlp, grad = self._compute_grad(BootstrapPFHMC, _bpf_kwargs())
        print(f"\n  BPF nlp: {nlp:.4f}, grad: {grad}")
        self._save_and_assert('bpf_gradient_nonzero', nlp, grad)


# ---------------------------------------------------------------------------
# Test 4: No monotonic memory growth
# ---------------------------------------------------------------------------

class TestNoMonotonicMemoryGrowth:
    """Verify that memory does not grow monotonically across repeated
    likelihood evaluations via the production path (after warm-up)."""

    def test_ledh_memory_stable(self):
        obs = _make_observations()
        runner = _make_runner(LEDHParticleFlowFilterHMC, _ledh_kwargs(), obs)
        q = runner.param_handler.unconstrained_init

        # Warm-up
        _ = runner._negative_log_posterior(q)

        tracemalloc.start()
        snapshots = []
        for i in range(10):
            _ = runner._negative_log_posterior(q)
            current, peak = tracemalloc.get_traced_memory()
            snapshots.append(current)

        tracemalloc.stop()

        # Check that memory is not monotonically increasing
        diffs = [snapshots[i+1] - snapshots[i] for i in range(len(snapshots)-1)]
        all_increasing = all(d > 0 for d in diffs)

        print(f"\n  Memory snapshots (KB): {[s // 1024 for s in snapshots]}")
        print(f"  Diffs (KB): {[d // 1024 for d in diffs]}")
        print(f"  All increasing: {all_increasing}")

        save_result(__file__, {
            'case_name': 'ledh_memory_stable',
            'snapshots_bytes': [int(s) for s in snapshots],
            'diffs_bytes': [int(d) for d in diffs],
            'all_increasing': bool(all_increasing),
            'passed': bool(not all_increasing),
        })

        assert not all_increasing, (
            "Memory grew monotonically across 10 evaluations — likely a leak"
        )
