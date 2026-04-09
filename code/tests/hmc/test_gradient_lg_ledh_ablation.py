"""LEDH gradient ablation: isolate which gradient path causes the 1.33 ratio.

Uses LEDHParticleFlowFilterHMCAblation (compiled path) with stop_gradient
flags at specific points. Each test cuts one gradient path.

All tests on LG model, obs_noise_std=1.0, T=20, n_lambda=15, OT resampling.

Run: python -m pytest tests/hmc/test_gradient_lg_ledh_ablation.py -v -s
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

# Import ablation variant from test directory
from ledh_invertible_hmc_ablation import LEDHParticleFlowFilterHMCAblation


DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
ONS_VAL = 1.0

M_RADII = 5
H_MAX = 0.05


@pytest.fixture(scope="module")
def obs_lg():
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=ONS_VAL, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=20, rng=rng)
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
    ad_val = grad.numpy() if grad is not None else None
    return float(ll.numpy()), ad_val


def _numerical_grad(obs_tf, ons_val, **ablation_kwargs):
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)
    for i, r in enumerate(radii):
        if ons_val - r <= 0.01:
            slopes[i] = np.nan
            continue
        f_plus = _eval_ll(obs_tf, ons_val + r, **ablation_kwargs)
        f_minus = _eval_ll(obs_tf, ons_val - r, **ablation_kwargs)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


def _report(label, obs_tf, **ablation_kwargs):
    ll, ad = _autodiff_grad(obs_tf, ONS_VAL, **ablation_kwargs)
    num, slopes = _numerical_grad(obs_tf, ONS_VAL, **ablation_kwargs)
    print(f"\n  [{label}]")
    print(f"    Forward ll:       {ll:.4f}")
    print(f"    Autodiff grad:    {ad}")
    print(f"    Numerical grad:   {num:.4f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if ad is not None and abs(num) > 1e-6:
        ratio = ad / num
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    else:
        ratio = None
    return ad, num, ratio


RATIO_TOL = 0.05  # ratio must be within 1.0 ± 0.05


def _assert_ratio(ratio, label):
    assert ratio is not None, f"{label}: ratio is None"
    assert 1.0 - RATIO_TOL < ratio < 1.0 + RATIO_TOL, (
        f"{label}: ratio={ratio:.4f}, expected 1.0 ± {RATIO_TOL}"
    )


class TestBaseline:
    """No ablation — KNOWN BUG: ratio ~1.33 due to OT dparticles bias."""
    def test_baseline(self, obs_lg):
        ad, num, ratio = _report("BASELINE", obs_lg)
        _assert_ratio(ratio, "BASELINE")


class TestAblation1:
    """Cut paths C+E: sg(covs) before flow_params."""
    def test_sg_covs_in_flow(self, obs_lg):
        ad, num, ratio = _report("sg(covs) in flow", obs_lg, sg_covs_in_flow=True)
        _assert_ratio(ratio, "sg(covs) in flow")


class TestAblation2:
    """Cut paths A+D: sg(R, R_inv) in flow."""
    def test_sg_R_in_flow(self, obs_lg):
        ad, num, ratio = _report("sg(R,R_inv) in flow", obs_lg, sg_R_in_flow=True)
        _assert_ratio(ratio, "sg(R,R_inv) in flow")


class TestAblation3:
    """Cut path D: sg(log_theta) — both theta and max_log_theta."""
    def test_sg_log_theta(self, obs_lg):
        ad, num, ratio = _report("sg(log_theta)", obs_lg, sg_log_theta=True)
        _assert_ratio(ratio, "sg(log_theta)")


class TestAblation4:
    """Cut path E (T branch): sg(transport_matrix) in covs transport."""
    def test_sg_ot_transport(self, obs_lg):
        ad, num, ratio = _report("sg(OT transport) in covs", obs_lg, sg_ot_transport=True)
        _assert_ratio(ratio, "sg(OT transport) in covs")


class TestAblation5:
    """Cut path C: sg(covs) after EKF update."""
    def test_sg_covs_after_update(self, obs_lg):
        ad, num, ratio = _report("sg(covs) after update", obs_lg, sg_covs_after_update=True)
        _assert_ratio(ratio, "sg(covs) after update")


class TestAblation6:
    """Zero dparticles in OT custom gradient (keep dlog_weights)."""
    def test_zero_ot_particle_grad(self, obs_lg):
        ad, num, ratio = _report("zero OT dparticles", obs_lg, zero_ot_particle_gradient=True)
        _assert_ratio(ratio, "zero OT dparticles")


class TestRatioVsT:
    """Measure autodiff/numerical ratio as a function of T."""

    @pytest.mark.parametrize("t_val", [1, 2, 3, 5, 10, 15, 20])
    def test_ratio_vs_T(self, obs_lg, t_val):
        obs_t = obs_lg[:t_val]
        ll, ad = _autodiff_grad(obs_t, ONS_VAL)
        num, slopes = _numerical_grad(obs_t, ONS_VAL)
        ratio = ad / num if abs(num) > 1e-6 else None
        print(f"\n  T={t_val}: ad={ad:.4f}  num={num:.4f}  ratio={ratio:.4f}" if ratio else
              f"\n  T={t_val}: ad={ad}  num={num:.4f}  ratio=N/A")
        _assert_ratio(ratio, f"T={t_val}")


class TestControlT1:
    """T=1: Only paths A/B/D. No C/E."""
    def test_T1(self, obs_lg):
        obs_t1 = obs_lg[:1]
        ll, ad = _autodiff_grad(obs_t1, ONS_VAL)
        num, slopes = _numerical_grad(obs_t1, ONS_VAL)
        ratio = ad / num if ad is not None and abs(num) > 1e-6 else None
        print(f"\n  [CONTROL T=1]")
        print(f"    Forward ll:       {ll:.4f}")
        print(f"    Autodiff grad:    {ad}")
        print(f"    Numerical grad:   {num:.4f}")
        if ratio is not None:
            print(f"    Ratio:            {ratio:.4f}")
        _assert_ratio(ratio, "T=1")


class TestControlT2NoResample:
    """T=2, no resampling: EKF covs carry (C) without OT transport (E)."""
    def test_T2_no_resample(self, obs_lg):
        obs_t2 = obs_lg[:2]
        # Need to create filter with always_resample=False
        # and high threshold so resampling never triggers
        base_model = LinearGaussianModel(
            F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
            obs_noise_std=ONS_VAL, dtype=DTYPE,
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
            always_resample=False,
        )
        var = tf.constant(ONS_VAL, dtype=DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(var)
            diff_model.update_parameters({'obs_noise_std': var})
            ll = filt.log_marginal_likelihood_tf(obs_t2, seed=PF_SEED)
        grad = tape.gradient(ll, var)
        ad = grad.numpy() if grad is not None else None
        diff_model.restore_parameters()

        # Numerical
        radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
        slopes = np.empty(M_RADII)
        for i, r in enumerate(radii):
            m_p = LinearGaussianModel(F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
                                       obs_noise_std=ONS_VAL + r, dtype=DTYPE)
            dm_p = DifferentiableModel(m_p, ['obs_noise_std'])
            f_p = LEDHParticleFlowFilterHMCAblation(
                model=dm_p, n_particles=N_PARTICLES, n_lambda_steps=N_LAMBDA_STEPS,
                resampling_method='ot_entropy', resampling_config={'epsilon': 0.5},
                weight_clip_range=50.0, stop_gradient_resampling=False,
                eager_mode=False, always_resample=False)
            ll_p = float(f_p.log_marginal_likelihood_tf(obs_t2, seed=PF_SEED).numpy())

            m_m = LinearGaussianModel(F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
                                       obs_noise_std=ONS_VAL - r, dtype=DTYPE)
            dm_m = DifferentiableModel(m_m, ['obs_noise_std'])
            f_m = LEDHParticleFlowFilterHMCAblation(
                model=dm_m, n_particles=N_PARTICLES, n_lambda_steps=N_LAMBDA_STEPS,
                resampling_method='ot_entropy', resampling_config={'epsilon': 0.5},
                weight_clip_range=50.0, stop_gradient_resampling=False,
                eager_mode=False, always_resample=False)
            ll_m = float(f_m.log_marginal_likelihood_tf(obs_t2, seed=PF_SEED).numpy())
            slopes[i] = (ll_p - ll_m) / (2.0 * r)

        num = np.median(slopes)
        ratio = ad / num if ad is not None and abs(num) > 1e-6 else None
        print(f"\n  [CONTROL T=2, no resampling]")
        print(f"    Forward ll:       {float(ll.numpy()):.4f}")
        print(f"    Autodiff grad:    {ad}")
        print(f"    Numerical grad:   {num:.4f}")
        print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
        if ratio is not None:
            print(f"    Ratio:            {ratio:.4f}")
        _assert_ratio(ratio, "T=2 no resampling")
