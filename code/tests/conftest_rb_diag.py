"""
Shared fixtures and helpers for Range-Bearing HMC diagnostic tests.

Used by:
  test_rb_hmc_diagnosis.py — BPF/LEDH/EKF/UKF gradient diagnostics
"""

import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.range_bearing import RangeBearingModel
from src.models.utils import generate_data
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
from src.filters.kalman.unscented_kalman import UnscentedKalmanFilter
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.hmc_runner import DPFRunner
from src.DF.parameter_handler import ParameterHandler
from src.DF.types import ParameterSpec


# ============================================================================
# CONSTANTS
# ============================================================================

DTYPE = tf.float32
N_PARTICLES = 500
N_LAMBDA_STEPS = 15
T = 50
SEED = 42
TRUE_SIGMA_RANGE = 0.1
TRUE_SIGMA_BEARING = 0.1
INIT_SIGMA_RANGE = 0.3
INIT_SIGMA_BEARING = 0.3

SIGMA_SCAN_GRID = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def rb_model_and_data():
    """RangeBearingModel with true params, T=50 observations."""
    model = RangeBearingModel(
        sigma_range=TRUE_SIGMA_RANGE,
        sigma_bearing=TRUE_SIGMA_BEARING,
        dtype=DTYPE,
    )
    rng = np.random.default_rng(SEED)
    initial_state, states, observations = generate_data(model, T=T, rng=rng)
    return model, observations, states, initial_state


@pytest.fixture(scope="session")
def rb_observations_tf(rb_model_and_data):
    """Observations as float32 TF tensor."""
    _, observations, _, _ = rb_model_and_data
    return tf.constant(observations, dtype=DTYPE)


# ============================================================================
# HELPERS
# ============================================================================

def _fresh_rb_model(sigma_range=INIT_SIGMA_RANGE, sigma_bearing=INIT_SIGMA_BEARING):
    """Create a fresh RangeBearingModel instance."""
    return RangeBearingModel(
        sigma_range=sigma_range,
        sigma_bearing=sigma_bearing,
        dtype=DTYPE,
    )


def _make_rb_param_specs(init_sr=INIT_SIGMA_RANGE, init_sb=INIT_SIGMA_BEARING):
    """ParameterSpec for sigma_range and sigma_bearing.

    Uses LogNormal(loc=-2.3, scale=0.5) prior matching the YAML configs.
    """
    return {
        'sigma_range': ParameterSpec(
            name='sigma_range',
            init_value=init_sr,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                loc=tf.constant(-2.3, dtype=DTYPE),
                scale=tf.constant(0.5, dtype=DTYPE),
            ),
        ),
        'sigma_bearing': ParameterSpec(
            name='sigma_bearing',
            init_value=init_sb,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                loc=tf.constant(-2.3, dtype=DTYPE),
                scale=tf.constant(0.5, dtype=DTYPE),
            ),
        ),
    }


def _make_rb_bpf_kwargs(resampling_method, stop_gradient, n_particles=N_PARTICLES):
    """Filter kwargs for BootstrapPFHMC on Range-Bearing."""
    kwargs = dict(
        n_particles=n_particles,
        resampling_method=resampling_method,
        stop_gradient_resampling=stop_gradient,
        eager_mode=False,
    )
    if resampling_method == 'soft':
        kwargs['hmc_resampling_method'] = 'soft'
        kwargs['hmc_resampling_config'] = {'alpha': 0.5}
        kwargs['resampling_config'] = {'alpha': 0.5}
    elif resampling_method == 'ot_entropy':
        kwargs['hmc_resampling_method'] = 'ot_entropy'
        kwargs['hmc_resampling_config'] = {'epsilon': 0.5}
        kwargs['resampling_config'] = {'epsilon': 0.5}
    return kwargs


def _make_rb_ledh_kwargs(resampling_method, stop_gradient, n_particles=N_PARTICLES):
    """Filter kwargs for LEDHParticleFlowFilterHMC on Range-Bearing."""
    kwargs = dict(
        n_particles=n_particles,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method=resampling_method,
        weight_clip_range=50.0,
        stop_gradient_resampling=stop_gradient,
        eager_mode=False,
    )
    if resampling_method == 'soft':
        kwargs['hmc_resampling_method'] = 'soft'
        kwargs['hmc_resampling_config'] = {'alpha': 0.5}
        kwargs['resampling_config'] = {'alpha': 0.5}
    elif resampling_method == 'ot_entropy':
        kwargs['hmc_resampling_method'] = 'ot_entropy'
        kwargs['hmc_resampling_config'] = {'epsilon': 0.5}
        kwargs['resampling_config'] = {'epsilon': 0.5}
    return kwargs


def _make_fresh_rb_runner(filter_class, filter_kwargs, observations_tf,
                          init_sr=INIT_SIGMA_RANGE, init_sb=INIT_SIGMA_BEARING):
    """Create a fresh DPFRunner with a new RangeBearingModel.

    Each runner gets its own model instance to avoid @tf.function cache reuse.
    """
    model = _fresh_rb_model(init_sr, init_sb)
    param_specs = _make_rb_param_specs(init_sr, init_sb)
    runner = DPFRunner(
        base_model=model,
        filter_class=filter_class,
        filter_kwargs=filter_kwargs,
        param_specs=param_specs,
        sampler='hmc',
    )
    runner._observations_tf = observations_tf
    return runner


def ekf_log_likelihood(sigma_range, sigma_bearing, observations_tf):
    """EKF log-likelihood at given params (deterministic reference)."""
    model = _fresh_rb_model(sigma_range, sigma_bearing)
    ekf = ExtendedKalmanFilter(model, sample_initial_mean=False)
    ll = ekf.log_marginal_likelihood_tf(observations_tf)
    return float(ll.numpy())


def ukf_log_likelihood(sigma_range, sigma_bearing, observations_tf):
    """UKF log-likelihood at given params (deterministic reference)."""
    model = _fresh_rb_model(sigma_range, sigma_bearing)
    ukf = UnscentedKalmanFilter(model, alpha=1.0, beta=2.0, kappa=0.0,
                                 sample_initial_mean=False)
    ll = ukf.log_marginal_likelihood_tf(observations_tf)
    return float(ll.numpy())


def pf_log_likelihood_rb(filter_class, filter_kwargs, sigma_range, sigma_bearing,
                          observations_tf):
    """PF log-likelihood at given params via eager mode."""
    eager_kwargs = dict(filter_kwargs)
    eager_kwargs['eager_mode'] = True
    model = _fresh_rb_model(sigma_range, sigma_bearing)
    filt = filter_class(model, **eager_kwargs)
    seed = tf.constant([SEED, 0], dtype=tf.int32)
    ll = filt.log_marginal_likelihood_tf(observations_tf, seed=seed)
    return float(ll.numpy())


def compute_gradient_rb(filter_class, filter_kwargs, observations_tf,
                         init_sr=INIT_SIGMA_RANGE, init_sb=INIT_SIGMA_BEARING):
    """Autodiff gradient of NLP w.r.t. unconstrained params.

    Returns: (nlp, grad_array_2d, constrained_dict, log_lik)
    """
    runner = _make_fresh_rb_runner(filter_class, filter_kwargs, observations_tf,
                                   init_sr, init_sb)
    q = runner.param_handler.unconstrained_init
    with tf.GradientTape() as tape:
        tape.watch(q)
        nlp = runner._negative_log_posterior(q)
    grad = tape.gradient(nlp, q)
    if grad is None:
        grad = tf.zeros_like(q)
    grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))

    constrained = runner.param_handler.constrain(q)
    log_prior = runner.param_handler.log_prior(constrained)
    log_lik = float((-nlp - log_prior).numpy())
    return float(nlp.numpy()), grad.numpy().copy(), constrained, log_lik


def finite_difference_gradient_rb(filter_class, filter_kwargs, observations_tf,
                                   init_sr=INIT_SIGMA_RANGE, init_sb=INIT_SIGMA_BEARING,
                                   eps=1e-4):
    """Central FD gradient of NLP, per-component.

    Returns: 2-element numpy array [d/dq_range, d/dq_bearing].
    """
    q0 = ParameterHandler(
        _make_rb_param_specs(init_sr, init_sb), dtype=DTYPE
    ).unconstrained_init
    n_params = len(q0.numpy())
    fd_grad = np.zeros(n_params)

    for i in range(n_params):
        e_i = np.zeros(n_params, dtype=np.float32)
        e_i[i] = eps
        e_i_tf = tf.constant(e_i)

        runner_plus = _make_fresh_rb_runner(
            filter_class, filter_kwargs, observations_tf, init_sr, init_sb)
        nlp_plus = float(runner_plus._negative_log_posterior(q0 + e_i_tf).numpy())

        runner_minus = _make_fresh_rb_runner(
            filter_class, filter_kwargs, observations_tf, init_sr, init_sb)
        nlp_minus = float(runner_minus._negative_log_posterior(q0 - e_i_tf).numpy())

        fd_grad[i] = (nlp_plus - nlp_minus) / (2 * eps)

    return fd_grad


def compute_likelihood_gradient_rb(filter_class, filter_kwargs, observations_tf,
                                    init_sr=INIT_SIGMA_RANGE, init_sb=INIT_SIGMA_BEARING,
                                    eps=1e-4):
    """Autodiff and FD gradient of -log_likelihood (no prior), per-component.

    Returns: (nll, ad_grad_2d, fd_grad_2d)
    """
    seed = tf.constant([SEED, 0], dtype=tf.int32)
    q0 = ParameterHandler(
        _make_rb_param_specs(init_sr, init_sb), dtype=DTYPE
    ).unconstrained_init

    # Autodiff
    runner_ad = _make_fresh_rb_runner(
        filter_class, filter_kwargs, observations_tf, init_sr, init_sb)
    with tf.GradientTape() as tape:
        tape.watch(q0)
        constrained = runner_ad.param_handler.constrain(q0)
        runner_ad.diff_model.update_parameters(constrained)
        ll = runner_ad.filter_obj.log_marginal_likelihood_tf(
            runner_ad._observations_tf, seed=seed)
        nll = -ll
    ad_grad = tape.gradient(nll, q0)
    if ad_grad is None:
        ad_grad = tf.zeros_like(q0)
    ad_grad = tf.where(tf.math.is_finite(ad_grad), ad_grad, tf.zeros_like(ad_grad))
    nll_val = float(nll.numpy())
    ad_grad_np = ad_grad.numpy().copy()

    # FD per-component
    n_params = len(q0.numpy())
    fd_grad = np.zeros(n_params)
    for i in range(n_params):
        e_i = np.zeros(n_params, dtype=np.float32)
        e_i[i] = eps
        e_i_tf = tf.constant(e_i)

        runner_plus = _make_fresh_rb_runner(
            filter_class, filter_kwargs, observations_tf, init_sr, init_sb)
        c_plus = runner_plus.param_handler.constrain(q0 + e_i_tf)
        runner_plus.diff_model.update_parameters(c_plus)
        ll_plus = float(runner_plus.filter_obj.log_marginal_likelihood_tf(
            runner_plus._observations_tf, seed=seed).numpy())

        runner_minus = _make_fresh_rb_runner(
            filter_class, filter_kwargs, observations_tf, init_sr, init_sb)
        c_minus = runner_minus.param_handler.constrain(q0 - e_i_tf)
        runner_minus.diff_model.update_parameters(c_minus)
        ll_minus = float(runner_minus.filter_obj.log_marginal_likelihood_tf(
            runner_minus._observations_tf, seed=seed).numpy())

        fd_grad[i] = (-ll_plus - (-ll_minus)) / (2 * eps)

    return nll_val, ad_grad_np, fd_grad
