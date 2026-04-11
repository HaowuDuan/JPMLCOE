"""LEDH gradient ablation: isolate which gradient path causes any autodiff/FD mismatch.

This test is INFORMATIVE — it runs each stop-gradient ablation case and saves
results, but does NOT assert on individual ratios. The point is to compare
ablations against the baseline to identify which gradient path contributes to
the mismatch (if any).

All cases on LG model, obs_noise_std=1.0, T=20, n_lambda=15, OT resampling.

Each case prints + saves to tests/hmc/results/test_ledh_ablation.json.

Run: python -m pytest tests/hmc/test_ledh_ablation.py -v -s
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
from src.DF.differentiable_model import DifferentiableModel

# The ablation filter variant lives next to this test as a utility module
from ledh_invertible_hmc_ablation import LEDHParticleFlowFilterHMCAblation

from _gradient_test_utils import save_result, reset_results, fd_grad_central


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
T = 20
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
ONS_VAL = 1.0
FD_H = 1e-4

MODEL = 'linear_gaussian'
FILTER = 'ledh_ot_ablation'


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


@pytest.fixture(scope="module")
def obs_lg():
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ONS_VAL, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def _make_filter(ons_val, **ablation_kwargs):
    base_model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ons_val, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['obs_noise_std'])
    filt = LEDHParticleFlowFilterHMCAblation(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method='ot_entropy',
        resampling_config={'epsilon': 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
        **ablation_kwargs,
    )
    return diff_model, filt


def _eval_ll(obs_tf, ons_val, **ablation_kwargs):
    diff_model, filt = _make_filter(ons_val, **ablation_kwargs)
    ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    return float(ll.numpy())


def _autodiff_grad(obs_tf, ons_val, **ablation_kwargs):
    diff_model, filt = _make_filter(ons_val, **ablation_kwargs)
    var = tf.constant(ons_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'obs_noise_std': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    if grad is None:
        return float(ll.numpy()), float('nan')
    return float(ll.numpy()), float(grad.numpy())


def _run_ablation(case_name, obs_tf, **ablation_kwargs):
    """Run one ablation case and save results. NO assertion."""
    ll, ad = _autodiff_grad(obs_tf, ONS_VAL, **ablation_kwargs)
    eval_fn = lambda x: _eval_ll(obs_tf, x, **ablation_kwargs)
    fd_med, fd_slopes = fd_grad_central(eval_fn, ONS_VAL, h=FD_H, n_radii=5)

    ratio = (ad / fd_med) if (abs(fd_med) > 1e-6 and not np.isnan(ad)) else float('nan')
    rel_err = abs(ad - fd_med) / max(abs(fd_med), 1e-6) if not np.isnan(ad) else float('nan')

    print(f"\n  [{case_name}]")
    print(f"    log_lik={ll:.6f}")
    print(f"    autodiff_grad={ad:.6f}")
    print(f"    fd_grad_median={fd_med:.6f}")
    print(f"    fd_slopes={np.array2string(fd_slopes, precision=4)}")
    print(f"    ratio (ad/fd)={ratio:.4f}")
    print(f"    relative_error={rel_err:.4f}")

    save_result(__file__, {
        'case_name': case_name,
        'model': MODEL,
        'filter': FILTER,
        'ablation_flags': {k: bool(v) for k, v in ablation_kwargs.items()},
        'param_name': 'obs_noise_std',
        'param_value': float(ONS_VAL),
        'n_particles': N_PARTICLES,
        'T': int(obs_tf.shape[0]),
        'n_lambda_steps': N_LAMBDA_STEPS,
        'fd_h': FD_H,
        'fd_slopes': [None if np.isnan(s) else float(s) for s in fd_slopes.tolist()],
        'log_likelihood': float(ll),
        'autodiff_grad': float(ad),
        'fd_grad_median': float(fd_med),
        'ratio': float(ratio),
        'relative_error': float(rel_err),
    })


# ----------------------------------------------------------------------------
# Ablation cases (informative — no assertions)
# ----------------------------------------------------------------------------

class TestAblationCases:

    def test_baseline(self, obs_lg):
        _run_ablation("BASELINE (no stop-gradient)", obs_lg)

    def test_sg_covs_in_flow(self, obs_lg):
        _run_ablation("sg(covs) in flow", obs_lg, sg_covs_in_flow=True)

    def test_sg_R_in_flow(self, obs_lg):
        _run_ablation("sg(R, R_inv) in flow", obs_lg, sg_R_in_flow=True)

    def test_sg_log_theta(self, obs_lg):
        _run_ablation("sg(log_theta)", obs_lg, sg_log_theta=True)

    def test_sg_ot_transport(self, obs_lg):
        _run_ablation("sg(OT transport) in covs", obs_lg, sg_ot_transport=True)

    def test_sg_covs_after_update(self, obs_lg):
        _run_ablation("sg(covs) after update", obs_lg, sg_covs_after_update=True)

    def test_zero_ot_particle_grad(self, obs_lg):
        _run_ablation("zero OT dparticles", obs_lg, zero_ot_particle_gradient=True)


class TestAblationVsT:
    """Sweep T to see how ratio scales with timesteps. Informative."""

    @pytest.mark.parametrize("t_val", [1, 3, 5, 10, 20])
    def test_T_sweep(self, obs_lg, t_val):
        _run_ablation(f"BASELINE T={t_val}", obs_lg[:t_val])
