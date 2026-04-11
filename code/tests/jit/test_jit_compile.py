"""JIT (XLA) compilation probe for HMC filter pipeline.

Goal: determine whether @tf.function(jit_compile=True) works on the outer
compiled filter without running a full HMC loop. If it works, measure the
speedup; if it fails, capture which op broke.

Test progression (fail-fast localization, per Codex review):
1. Standalone helper tests (log_abs_det, ot_entropy_resample) — narrowest scope
2. Filter forward-only at T=1, N=50 — bpf_systematic, ledh_systematic, bpf_ot, ledh_ot
3. Filter forward+gradient at T=1, N=50 — same four filters
4. Speedup measurement (ledh+ot, T=20, N=200) — informative
5. Gradient correctness (XLA grad ≈ graph grad) — informative

The XLA filter is built by re-wrapping the existing compiled_filter's
python_function with jit_compile=True (no production code changes).

Each case prints + saves to tests/jit/results/test_jit_compile.json.

Run: cd code && python -m pytest tests/jit/test_jit_compile.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.DF.differentiable_model import DifferentiableModel

from src.utils.linalg import graph_safe_log_abs_det_fast
from src.resampling.ot_entropy import ot_entropy_resample

from _jit_test_utils import try_compile, time_call, save_result, reset_results


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DTYPE = tf.float32   # XLA prefers float32; LG smooth enough that this is fine
N_PARTICLES_SMALL = 50
N_PARTICLES_TIMING = 200
T_SMOKE = 1
T_TIMING = 20
SEED = tf.constant([42, 0], dtype=tf.int32)
DATA_SEED = 42


@pytest.fixture(scope="module", autouse=True)
def _reset():
    reset_results(__file__)
    yield


# ----------------------------------------------------------------------------
# Helpers: model + filter construction
# ----------------------------------------------------------------------------

def _make_lg_obs(T):
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=1.0, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def _make_diff_model(ons_val=1.0):
    base = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ons_val, dtype=DTYPE,
    )
    return DifferentiableModel(base, ['obs_noise_std'])


def _make_ledh(diff_model, n_particles, resampling_method, n_lambda_steps=5):
    if resampling_method == 'ot_entropy':
        resampling_config = {'epsilon': 0.5}
    else:
        resampling_config = {}
    return LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=n_particles,
        n_lambda_steps=n_lambda_steps,
        resampling_method=resampling_method,
        resampling_config=resampling_config,
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,  # force resampler to actually run for OT tests
    )


def _make_bpf(diff_model, n_particles, resampling_method):
    if resampling_method == 'ot_entropy':
        resampling_config = {'epsilon': 0.5}
    else:
        resampling_config = {}
    return BootstrapPFHMC(
        model=diff_model,
        n_particles=n_particles,
        resampling_method=resampling_method,
        resampling_config=resampling_config,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )


# ----------------------------------------------------------------------------
# Re-wrap compiled_filter with jit_compile=True (no production changes)
# ----------------------------------------------------------------------------

def _xla_recompile(filt):
    """Replace filt._compiled_filter with a jit_compile=True version.

    Uses the existing compiled_filter's .python_function (which still has
    its closure captures) and re-decorates it with jit_compile=True.
    """
    original = filt._compiled_filter
    py_fn = original.python_function
    filt._compiled_filter = tf.function(
        py_fn, jit_compile=True, reduce_retracing=True
    )
    return filt


# ----------------------------------------------------------------------------
# Compile probe wrapper
# ----------------------------------------------------------------------------

def _probe_filter(filt, obs):
    """Try one log_marginal_likelihood_tf call with the given filter.

    Returns (succeeded, error_type, error_message).
    """
    return try_compile(filt.log_marginal_likelihood_tf, obs, seed=SEED)


def _probe_filter_grad(filt, diff_model, obs):
    """Try a gradient call. Returns (succeeded, error_type, error_message)."""
    def _go():
        var = tf.constant(1.0, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(var)
            diff_model.update_parameters({'obs_noise_std': var})
            ll = filt.log_marginal_likelihood_tf(obs, seed=SEED)
        g = tape.gradient(ll, var)
        diff_model.restore_parameters()
        return g
    return try_compile(_go)


def _save(case_name, succeeded, err_type, err_msg, **extra):
    case = {
        'case_name': case_name,
        'compile_succeeded': bool(succeeded),
        'error_type': err_type,
        'error_message': err_msg,
        **extra,
    }
    print(f"\n  [{case_name}]  {'PASS' if succeeded else 'FAIL'}")
    if not succeeded:
        print(f"    {err_type}: {err_msg[:300] if err_msg else ''}")
    save_result(__file__, case)
    return case


# ============================================================================
# Phase 1: Standalone helper tests
# ============================================================================

class TestHelperXLA:
    """Probe individual helpers with jit_compile=True. Forward and forward+grad."""

    def test_log_abs_det_fast_forward(self):
        n = 4
        M = tf.eye(3, dtype=DTYPE) + 0.1 * tf.random.normal([n, 3, 3], seed=0, dtype=DTYPE)
        fn = tf.function(
            graph_safe_log_abs_det_fast.python_function,
            jit_compile=True,
        )
        ok, et, em = try_compile(fn, M)
        _save("helper.log_abs_det_fast.forward", ok, et, em)

    def test_log_abs_det_fast_grad(self):
        n = 4
        M_init = tf.eye(3, dtype=DTYPE) + 0.1 * tf.random.normal([n, 3, 3], seed=1, dtype=DTYPE)
        var = tf.Variable(M_init)
        fn = tf.function(
            graph_safe_log_abs_det_fast.python_function,
            jit_compile=True,
        )
        def _go():
            with tf.GradientTape() as tape:
                out = fn(var)
                loss = tf.reduce_sum(out)
            return tape.gradient(loss, var)
        ok, et, em = try_compile(_go)
        _save("helper.log_abs_det_fast.gradient", ok, et, em)

    def test_ot_entropy_forward(self):
        N = N_PARTICLES_SMALL
        rng = np.random.default_rng(0)
        particles = tf.constant(rng.normal(0, 1, (N, 1)), dtype=DTYPE)
        weights = tf.ones(N, dtype=DTYPE) / N
        seed = tf.constant([0, 1], dtype=tf.int32)
        fn = tf.function(
            lambda p, w, s: ot_entropy_resample(p, w, seed=s, epsilon=0.5),
            jit_compile=True,
        )
        ok, et, em = try_compile(fn, particles, weights, seed)
        _save("helper.ot_entropy.forward", ok, et, em)

    def test_ot_entropy_grad(self):
        N = N_PARTICLES_SMALL
        rng = np.random.default_rng(1)
        particles_init = rng.normal(0, 1, (N, 1)).astype(np.float32)
        var = tf.Variable(particles_init)
        weights = tf.ones(N, dtype=DTYPE) / N
        seed = tf.constant([0, 1], dtype=tf.int32)
        fn = tf.function(
            lambda p, w, s: ot_entropy_resample(p, w, seed=s, epsilon=0.5),
            jit_compile=True,
        )
        def _go():
            with tf.GradientTape() as tape:
                result = fn(var, weights, seed)
                loss = tf.reduce_sum(result.particles)
            return tape.gradient(loss, var)
        ok, et, em = try_compile(_go)
        _save("helper.ot_entropy.gradient", ok, et, em)


# ============================================================================
# Phase 2: Filter forward-only at T=1, N=50
# ============================================================================

class TestFilterXLAForward:
    """Probe full filter forward call under XLA. T=1, N=50."""

    def _run(self, case_name, make_filt, resampling_method):
        obs = _make_lg_obs(T_SMOKE)
        diff_model = _make_diff_model()
        filt = make_filt(diff_model, N_PARTICLES_SMALL, resampling_method)
        _xla_recompile(filt)
        ok, et, em = _probe_filter(filt, obs)
        _save(case_name, ok, et, em,
              n_particles=N_PARTICLES_SMALL, T=T_SMOKE,
              filter_type=case_name.split('.')[1],
              resampling=resampling_method)

    def test_bpf_systematic(self):
        self._run("filter_forward.bpf_systematic", _make_bpf, 'systematic')

    def test_ledh_systematic(self):
        self._run("filter_forward.ledh_systematic", _make_ledh, 'systematic')

    def test_bpf_ot(self):
        self._run("filter_forward.bpf_ot", _make_bpf, 'ot_entropy')

    def test_ledh_ot(self):
        self._run("filter_forward.ledh_ot", _make_ledh, 'ot_entropy')


# ============================================================================
# Phase 3: Filter forward+gradient at T=1, N=50
# ============================================================================

class TestFilterXLAGradient:
    """Probe full filter gradient call under XLA. T=1, N=50."""

    def _run(self, case_name, make_filt, resampling_method):
        obs = _make_lg_obs(T_SMOKE)
        diff_model = _make_diff_model()
        filt = make_filt(diff_model, N_PARTICLES_SMALL, resampling_method)
        _xla_recompile(filt)
        ok, et, em = _probe_filter_grad(filt, diff_model, obs)
        _save(case_name, ok, et, em,
              n_particles=N_PARTICLES_SMALL, T=T_SMOKE,
              filter_type=case_name.split('.')[1],
              resampling=resampling_method)

    def test_bpf_systematic(self):
        self._run("filter_gradient.bpf_systematic", _make_bpf, 'systematic')

    def test_ledh_systematic(self):
        self._run("filter_gradient.ledh_systematic", _make_ledh, 'systematic')

    def test_bpf_ot(self):
        self._run("filter_gradient.bpf_ot", _make_bpf, 'ot_entropy')

    def test_ledh_ot(self):
        self._run("filter_gradient.ledh_ot", _make_ledh, 'ot_entropy')


# ============================================================================
# Phase 4: Speedup measurement (informative)
# ============================================================================

class TestXLASpeedup:
    """If LEDH+OT compiles under XLA, measure forward and forward+grad speedup."""

    def test_ledh_ot_forward_speedup(self):
        obs = _make_lg_obs(T_TIMING)
        diff_graph = _make_diff_model()
        filt_graph = _make_ledh(diff_graph, N_PARTICLES_TIMING, 'ot_entropy')

        diff_xla = _make_diff_model()
        filt_xla = _make_ledh(diff_xla, N_PARTICLES_TIMING, 'ot_entropy')
        _xla_recompile(filt_xla)

        # Probe XLA compile first
        ok, et, em = _probe_filter(filt_xla, obs)
        if not ok:
            _save("speedup.ledh_ot.forward", False, et, em,
                  n_particles=N_PARTICLES_TIMING, T=T_TIMING,
                  speedup=None)
            pytest.skip(f"XLA failed to compile: {et}")

        graph_t = time_call(lambda: filt_graph.log_marginal_likelihood_tf(obs, seed=SEED))
        xla_t = time_call(lambda: filt_xla.log_marginal_likelihood_tf(obs, seed=SEED))
        speedup = graph_t / xla_t if xla_t > 0 else float('nan')

        print(f"\n  graph={graph_t*1000:.2f}ms  xla={xla_t*1000:.2f}ms  speedup={speedup:.2f}x")
        _save("speedup.ledh_ot.forward", True, None, None,
              n_particles=N_PARTICLES_TIMING, T=T_TIMING,
              graph_time_s=graph_t, xla_time_s=xla_t, speedup=speedup)

    def test_ledh_ot_gradient_speedup(self):
        obs = _make_lg_obs(T_TIMING)
        diff_graph = _make_diff_model()
        filt_graph = _make_ledh(diff_graph, N_PARTICLES_TIMING, 'ot_entropy')

        diff_xla = _make_diff_model()
        filt_xla = _make_ledh(diff_xla, N_PARTICLES_TIMING, 'ot_entropy')
        _xla_recompile(filt_xla)

        ok, et, em = _probe_filter_grad(filt_xla, diff_xla, obs)
        if not ok:
            _save("speedup.ledh_ot.gradient", False, et, em,
                  n_particles=N_PARTICLES_TIMING, T=T_TIMING,
                  speedup=None)
            pytest.skip(f"XLA failed to compile: {et}")

        def _grad_call(filt, dm):
            var = tf.constant(1.0, dtype=DTYPE)
            with tf.GradientTape() as tape:
                tape.watch(var)
                dm.update_parameters({'obs_noise_std': var})
                ll = filt.log_marginal_likelihood_tf(obs, seed=SEED)
            g = tape.gradient(ll, var)
            dm.restore_parameters()
            return g

        graph_t = time_call(lambda: _grad_call(filt_graph, diff_graph))
        xla_t = time_call(lambda: _grad_call(filt_xla, diff_xla))
        speedup = graph_t / xla_t if xla_t > 0 else float('nan')

        print(f"\n  graph={graph_t*1000:.2f}ms  xla={xla_t*1000:.2f}ms  speedup={speedup:.2f}x")
        _save("speedup.ledh_ot.gradient", True, None, None,
              n_particles=N_PARTICLES_TIMING, T=T_TIMING,
              graph_time_s=graph_t, xla_time_s=xla_t, speedup=speedup)


# ============================================================================
# Phase 5: Gradient correctness (XLA vs graph)
# ============================================================================

class TestXLAGradientCorrectness:
    """If LEDH+OT compiles, verify XLA gradient matches graph gradient."""

    def test_ledh_ot(self):
        obs = _make_lg_obs(T_TIMING)

        diff_graph = _make_diff_model()
        filt_graph = _make_ledh(diff_graph, N_PARTICLES_TIMING, 'ot_entropy')

        diff_xla = _make_diff_model()
        filt_xla = _make_ledh(diff_xla, N_PARTICLES_TIMING, 'ot_entropy')
        _xla_recompile(filt_xla)

        def _grad(filt, dm):
            var = tf.constant(1.0, dtype=DTYPE)
            with tf.GradientTape() as tape:
                tape.watch(var)
                dm.update_parameters({'obs_noise_std': var})
                ll = filt.log_marginal_likelihood_tf(obs, seed=SEED)
            g = tape.gradient(ll, var)
            dm.restore_parameters()
            return float(ll.numpy()), float(g.numpy())

        try:
            ll_g, grad_g = _grad(filt_graph, diff_graph)
            ll_x, grad_x = _grad(filt_xla, diff_xla)
        except Exception as e:
            _save("correctness.ledh_ot.gradient", False, type(e).__name__, str(e)[:500],
                  n_particles=N_PARTICLES_TIMING, T=T_TIMING)
            pytest.skip(f"Could not run both modes: {e}")

        ll_diff = abs(ll_g - ll_x) / max(abs(ll_g), 1e-6)
        grad_diff = abs(grad_g - grad_x) / max(abs(grad_g), 1e-6)
        print(f"\n  graph: ll={ll_g:.6f} grad={grad_g:.6f}")
        print(f"  xla:   ll={ll_x:.6f} grad={grad_x:.6f}")
        print(f"  ll rel diff={ll_diff:.2e}  grad rel diff={grad_diff:.2e}")

        passed = ll_diff < 1e-3 and grad_diff < 1e-3
        _save("correctness.ledh_ot.gradient", passed, None, None,
              n_particles=N_PARTICLES_TIMING, T=T_TIMING,
              graph_log_lik=ll_g, xla_log_lik=ll_x,
              graph_grad=grad_g, xla_grad=grad_x,
              ll_relative_diff=ll_diff, grad_relative_diff=grad_diff)
