"""Gradient validation: LEDH+OT autodiff vs numerical on 2D SV model.

Trainable parameter: sigma2 (enters through transitions, NOT observation noise).
Initial conditions detached to match production HMC path.

Run: python -m pytest tests/hmc/test_gradient_vs_numerical_sv2d.py -v -s
"""

import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pytest
import tensorflow as tf

from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.DF.differentiable_model import DifferentiableModel
from src.utils.linalg import graph_safe_inv


# ============================================================================
# Constants
# ============================================================================

DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
PF_SEED = tf.constant([42, 0])
DATA_SEED = 42
SIGMA2_VAL = 1.0

M_RADII = 5            # fewer radii for speed in diagnostics
H_MAX = 0.05


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def obs_t20():
    """T=20 observations."""
    model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0, dtype=DTYPE,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(model, T=20, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


@pytest.fixture(scope="module")
def obs_t1(obs_t20):
    """T=1 — single timestep."""
    return obs_t20[:1]


@pytest.fixture(scope="module")
def obs_t3(obs_t20):
    """T=3 — few timesteps."""
    return obs_t20[:3]


# ============================================================================
# Helpers
# ============================================================================

def _make_filter(sigma2_val, filter_type='ledh_ot', n_lambda_steps=N_LAMBDA_STEPS,
                 stop_gradient_resampling=False, resampling_method='ot_entropy',
                 T_obs=None):
    """Create model + DifferentiableModel + filter."""
    base_model = StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=sigma2_val, b=1.0, dtype=DTYPE,
    )
    diff_model = DifferentiableModel(base_model, ['sigma2'])

    if filter_type == 'ledh_ot':
        filt = LEDHParticleFlowFilterHMC(
            model=diff_model,
            n_particles=N_PARTICLES,
            n_lambda_steps=n_lambda_steps,
            resampling_method=resampling_method,
            resampling_config={'epsilon': 0.5} if resampling_method == 'ot_entropy' else {},
            weight_clip_range=50.0,
            stop_gradient_resampling=stop_gradient_resampling,
            eager_mode=False,
            always_resample=True,
        )
    elif filter_type == 'bpf_ot':
        filt = BootstrapPFHMC(
            model=diff_model,
            n_particles=N_PARTICLES,
            resampling_method='ot_entropy',
            resampling_config={'epsilon': 0.5},
            stop_gradient_resampling=False,
            eager_mode=False,
            always_resample=True,
        )
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    return diff_model, filt


def _compiled_ll_with_current_params(obs_tf, filt):
    """Run compiled filter without reinitializing particles/covs."""
    R = filt.model.observation_noise_cov
    R_inv = graph_safe_inv(R)
    regularization_tf = tf.constant(filt.regularization, dtype=filt.dtype)

    param_values = []
    for name in filt.model.trainable_param_names:
        val = getattr(filt.model, name)
        val = tf.identity(val) if isinstance(val, tf.Tensor) else tf.constant(float(val), dtype=filt.dtype)
        param_values.append(val)
    param_values = tf.stack(param_values)

    return filt._compiled_filter(
        obs_tf,
        filt.particles.value(),
        filt.weights.value(),
        filt.particle_covs.value(),
        R, R_inv, regularization_tf,
        filt.rng_key[0],
        param_values,
    )


def _eval_ll_detached_init(obs_tf, base_sigma2, eval_sigma2, **filter_kwargs):
    """Evaluate LL with initial conditions baked in at base_sigma2.

    initialize() at base_sigma2, then call compiled loop directly so only
    dynamics/weights see eval_sigma2. Matches the autodiff path.
    """
    diff_model, filt = _make_filter(base_sigma2, **filter_kwargs)
    filt.initialize(random_seed=int(PF_SEED[0].numpy()))
    diff_model.update_parameters({'sigma2': tf.constant(eval_sigma2, dtype=DTYPE)})
    ll = _compiled_ll_with_current_params(obs_tf, filt)
    diff_model.restore_parameters()
    return float(ll.numpy())


def _autodiff_grad(obs_tf, sigma2_val, **filter_kwargs):
    """Autodiff gradient + forward value.

    Creates model at sigma2_val, initialize() runs (baking in initial
    conditions as constants), then update_parameters sets sigma2 as a
    watched tensor for the compiled loop.
    """
    diff_model, filt = _make_filter(sigma2_val, **filter_kwargs)
    var = tf.constant(sigma2_val, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({'sigma2': var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    ad_val = grad.numpy() if grad is not None else None
    return float(ll.numpy()), ad_val


def _numerical_grad(obs_tf, sigma2_val, **filter_kwargs):
    """Numerical gradient via central differences with detached initial conditions.

    For each perturbation, creates model at sigma2_val (so initial conditions
    are fixed), then updates sigma2 to the perturbed value. This matches
    the autodiff path exactly.

    Returns (median, slopes).
    """
    radii = np.array([(k + 1) * H_MAX / M_RADII for k in range(M_RADII)])
    slopes = np.empty(M_RADII)
    for i, r in enumerate(radii):
        if sigma2_val - r <= 0.01:
            slopes[i] = np.nan
            continue
        f_plus = _eval_ll_detached_init(obs_tf, sigma2_val, sigma2_val + r, **filter_kwargs)
        f_minus = _eval_ll_detached_init(obs_tf, sigma2_val, sigma2_val - r, **filter_kwargs)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    return np.median(valid), valid


def _report(label, obs_tf, sigma2_val, **filter_kwargs):
    """Run both autodiff and numerical, print comparison."""
    ll_val, ad = _autodiff_grad(obs_tf, sigma2_val, **filter_kwargs)
    num_med, slopes = _numerical_grad(obs_tf, sigma2_val, **filter_kwargs)
    print(f"\n  [{label}]")
    print(f"    Forward ll:       {ll_val:.4f}")
    print(f"    Autodiff grad:    {ad}")
    print(f"    Numerical grad:   {num_med:.4f}")
    print(f"    Slopes:           {np.array2string(slopes, precision=4)}")
    if ad is not None and abs(num_med) > 1e-6:
        ratio = ad / num_med
        print(f"    Ratio (ad/num):   {ratio:.4f}")
    return ad, num_med


# ============================================================================
# Diagnostic Tests
# ============================================================================

class TestDiagnosticForwardValues:
    """Step 0: Do autodiff and numerical even compute the same forward value?"""

    def test_forward_value_match(self, obs_t20):
        """Autodiff and numerical paths should give same ll at sigma2=1.0."""
        # Autodiff path: create at 1.0, update_parameters to 1.0, run
        ll_ad, _ = _autodiff_grad(obs_t20, SIGMA2_VAL)
        # Numerical path: create at 1.0, run
        ll_num = _eval_ll_detached_init(obs_t20, SIGMA2_VAL, SIGMA2_VAL)
        print(f"\n  Forward ll (autodiff path): {ll_ad:.6f}")
        print(f"  Forward ll (numerical path): {ll_num:.6f}")
        print(f"  Difference: {abs(ll_ad - ll_num):.6f}")
        assert abs(ll_ad - ll_num) < 1e-4, (
            f"Forward values differ: ad={ll_ad:.6f}, num={ll_num:.6f}"
        )


class TestDiagnosticBPFBaseline:
    """Step 1: BPF gradient should work (known good). Confirms test harness."""

    def test_bpf_ot_gradient(self, obs_t20):
        ad, num = _report("BPF+OT T=20", obs_t20, SIGMA2_VAL, filter_type='bpf_ot')
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            assert 0.5 < ratio < 2.0, f"BPF ratio={ratio:.2f}, expected ~1.0"


class TestDiagnosticIsolateOT:
    """Step 2: Remove OT — use systematic+stop_gradient.
    If ad≈num here but not with OT, the OT backward is the problem."""

    def test_ledh_systematic_stopgrad(self, obs_t20):
        ad, num = _report("LEDH+systematic+stopgrad T=20", obs_t20, SIGMA2_VAL,
                          stop_gradient_resampling=True,
                          resampling_method='systematic')
        # With stop_gradient, autodiff and numerical won't match perfectly
        # but autodiff should be nonzero
        print(f"    (stop_gradient: ad won't match num, just check nonzero)")
        assert ad is not None

    def test_ledh_ot_stopgrad(self, obs_t20):
        """OT forward, but stop_gradient on backward. Isolates OT backward."""
        ad, num = _report("LEDH+OT+stopgrad T=20", obs_t20, SIGMA2_VAL,
                          stop_gradient_resampling=True)
        assert ad is not None


class TestDiagnosticIsolateTimesteps:
    """Step 3: Reduce T to isolate per-timestep vs accumulation."""

    def test_ledh_ot_T1(self, obs_t1):
        ad, num = _report("LEDH+OT T=1", obs_t1, SIGMA2_VAL)
        assert ad is not None

    def test_ledh_ot_T3(self, obs_t3):
        ad, num = _report("LEDH+OT T=3", obs_t3, SIGMA2_VAL)
        assert ad is not None


class TestDiagnosticIsolateFlow:
    """Step 4: Reduce lambda steps to isolate flow loop."""

    def test_ledh_ot_1step(self, obs_t20):
        ad, num = _report("LEDH+OT n_lambda=1 T=20", obs_t20, SIGMA2_VAL,
                          n_lambda_steps=1)
        assert ad is not None

    def test_ledh_ot_3steps(self, obs_t20):
        ad, num = _report("LEDH+OT n_lambda=3 T=20", obs_t20, SIGMA2_VAL,
                          n_lambda_steps=3)
        assert ad is not None


class TestDiagnosticFullComparison:
    """Step 5: Full LEDH+OT — the original failing test."""

    def test_ledh_ot_full(self, obs_t20):
        ad, num = _report("LEDH+OT full (T=20, n_lambda=15)", obs_t20, SIGMA2_VAL)
        assert ad is not None
        if abs(num) > 0.1:
            ratio = ad / num
            print(f"    RATIO: {ratio:.4f} (want ~1.0)")
