"""XLA value-and-gradient validation.

Tests the new filter.value_and_grad_tf() method, which puts the GradientTape
INSIDE a @tf.function(jit_compile=True) wrapper. This avoids the
'TensorList crossing the XLA/TF boundary' limitation that blocks the naive
approach of taking gradients externally on a jit_compile'd forward function.

Phases:
1. Compile probe at T=1, N=50 — does it compile at all?
2. Speedup measurement at T=20, N=200 — graph mode tape vs XLA value_and_grad
3. Numerical equivalence — XLA grad ≈ graph grad

Run: cd code && python -m pytest tests/jit/test_jit_value_and_grad.py -v -s
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

from _jit_test_utils import try_compile, time_call, save_result, reset_results


DTYPE = tf.float32
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
    cfg = {'epsilon': 0.5} if resampling_method == 'ot_entropy' else {}
    return LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=n_particles,
        n_lambda_steps=n_lambda_steps,
        resampling_method=resampling_method,
        resampling_config=cfg,
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )


def _make_bpf(diff_model, n_particles, resampling_method):
    cfg = {'epsilon': 0.5} if resampling_method == 'ot_entropy' else {}
    return BootstrapPFHMC(
        model=diff_model,
        n_particles=n_particles,
        resampling_method=resampling_method,
        resampling_config=cfg,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )


def _save(case_name, succeeded, err_type, err_msg, **extra):
    case = {
        'case_name': case_name,
        'compile_succeeded': bool(succeeded),
        'error_type': err_type,
        'error_message': err_msg,
        **extra,
    }
    print(f"\n  [{case_name}]  {'PASS' if succeeded else 'FAIL'}")
    if not succeeded and err_msg:
        print(f"    {err_type}: {err_msg[:300]}")
    save_result(__file__, case)


# ============================================================================
# Phase 1: Compile probe at T=1, N=50
# ============================================================================

class TestValueAndGradCompile:
    """Does filter.value_and_grad_tf() actually compile under XLA?"""

    def _run(self, case_name, make_filt, resampling_method):
        obs = _make_lg_obs(T_SMOKE)
        diff_model = _make_diff_model()
        filt = make_filt(diff_model, N_PARTICLES_SMALL, resampling_method)
        var = tf.constant(1.0, dtype=DTYPE)

        ok, et, em = try_compile(
            filt.value_and_grad_tf, obs, {'obs_noise_std': var}, SEED
        )
        _save(case_name, ok, et, em,
              n_particles=N_PARTICLES_SMALL, T=T_SMOKE,
              resampling=resampling_method)

    def test_bpf_systematic(self):
        self._run("v&g_compile.bpf_systematic", _make_bpf, 'systematic')

    def test_ledh_systematic(self):
        self._run("v&g_compile.ledh_systematic", _make_ledh, 'systematic')

    def test_bpf_ot(self):
        self._run("v&g_compile.bpf_ot", _make_bpf, 'ot_entropy')

    def test_ledh_ot(self):
        self._run("v&g_compile.ledh_ot", _make_ledh, 'ot_entropy')


# ============================================================================
# Phase 2: Speedup measurement at T=20, N=200
# ============================================================================

class TestValueAndGradSpeedup:
    """Compare XLA value_and_grad_tf vs graph-mode external GradientTape."""

    def test_ledh_ot(self):
        obs = _make_lg_obs(T_TIMING)

        # Graph mode: external tape
        diff_g = _make_diff_model()
        filt_g = _make_ledh(diff_g, N_PARTICLES_TIMING, 'ot_entropy')

        def graph_call():
            var = tf.constant(1.0, dtype=DTYPE)
            with tf.GradientTape() as tape:
                tape.watch(var)
                diff_g.update_parameters({'obs_noise_std': var})
                ll = filt_g.log_marginal_likelihood_tf(obs, seed=SEED)
            grad = tape.gradient(ll, var)
            diff_g.restore_parameters()
            return ll, grad

        # XLA mode: value_and_grad_tf
        diff_x = _make_diff_model()
        filt_x = _make_ledh(diff_x, N_PARTICLES_TIMING, 'ot_entropy')

        def xla_call():
            var = tf.constant(1.0, dtype=DTYPE)
            return filt_x.value_and_grad_tf(obs, {'obs_noise_std': var}, SEED)

        # Probe XLA compile first
        ok, et, em = try_compile(xla_call)
        if not ok:
            _save("v&g_speedup.ledh_ot", False, et, em,
                  n_particles=N_PARTICLES_TIMING, T=T_TIMING, speedup=None)
            pytest.skip(f"XLA value_and_grad failed to compile: {et}")

        graph_t = time_call(graph_call)
        xla_t = time_call(xla_call)
        speedup = graph_t / xla_t if xla_t > 0 else float('nan')

        print(f"\n  graph={graph_t*1000:.2f}ms  xla={xla_t*1000:.2f}ms  speedup={speedup:.2f}x")
        _save("v&g_speedup.ledh_ot", True, None, None,
              n_particles=N_PARTICLES_TIMING, T=T_TIMING,
              graph_time_s=graph_t, xla_time_s=xla_t, speedup=speedup)


# ============================================================================
# Phase 3: Numerical equivalence
# ============================================================================

class TestValueAndGradCorrectness:
    """XLA value_and_grad_tf should produce the same numbers as graph mode."""

    def test_ledh_ot(self):
        obs = _make_lg_obs(T_TIMING)

        diff_g = _make_diff_model()
        filt_g = _make_ledh(diff_g, N_PARTICLES_TIMING, 'ot_entropy')
        diff_x = _make_diff_model()
        filt_x = _make_ledh(diff_x, N_PARTICLES_TIMING, 'ot_entropy')

        # Graph
        var_g = tf.constant(1.0, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(var_g)
            diff_g.update_parameters({'obs_noise_std': var_g})
            ll_g = filt_g.log_marginal_likelihood_tf(obs, seed=SEED)
        grad_g = tape.gradient(ll_g, var_g)
        diff_g.restore_parameters()
        ll_g_v, grad_g_v = float(ll_g.numpy()), float(grad_g.numpy())

        # XLA
        try:
            var_x = tf.constant(1.0, dtype=DTYPE)
            ll_x, grads_x = filt_x.value_and_grad_tf(
                obs, {'obs_noise_std': var_x}, SEED
            )
            ll_x_v = float(ll_x.numpy())
            grad_x_v = float(grads_x['obs_noise_std'].numpy())
        except Exception as e:
            _save("v&g_correctness.ledh_ot", False, type(e).__name__, str(e)[:500],
                  n_particles=N_PARTICLES_TIMING, T=T_TIMING)
            pytest.skip(f"XLA path failed: {e}")

        ll_diff = abs(ll_g_v - ll_x_v) / max(abs(ll_g_v), 1e-6)
        grad_diff = abs(grad_g_v - grad_x_v) / max(abs(grad_g_v), 1e-6)
        print(f"\n  graph: ll={ll_g_v:.6f} grad={grad_g_v:.6f}")
        print(f"  xla:   ll={ll_x_v:.6f} grad={grad_x_v:.6f}")
        print(f"  ll rel diff={ll_diff:.2e}  grad rel diff={grad_diff:.2e}")

        passed = ll_diff < 1e-3 and grad_diff < 1e-3
        _save("v&g_correctness.ledh_ot", passed, None, None,
              n_particles=N_PARTICLES_TIMING, T=T_TIMING,
              graph_log_lik=ll_g_v, xla_log_lik=ll_x_v,
              graph_grad=grad_g_v, xla_grad=grad_x_v,
              ll_relative_diff=ll_diff, grad_relative_diff=grad_diff)
