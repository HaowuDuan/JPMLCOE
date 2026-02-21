"""
Shared fixtures and helpers for HMC gradient diagnostic tests.

Used by:
  test_hmc_bpf.py  — BPF baseline
  test_hmc_ledh.py — LEDH vs BPF
"""

import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data
from src.filters.kalman.kalman import KalmanFilter
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.hmc_runner import DPFRunner
from src.DF.parameter_handler import ParameterHandler
from src.DF.types import ParameterSpec


# ============================================================================
# CONSTANTS
# ============================================================================

DTYPE = tf.float64
N_PARTICLES = 200
N_LAMBDA_STEPS = 15
T = 50
SEED = 42
TRUE_OBS_NOISE_STD = 1.0
INIT_OBS_NOISE_STD = 2.0

SIGMA_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def lg_model_and_data():
    """1D Linear Gaussian model with true obs_noise_std=1.0, T=50 observations."""
    model = LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=TRUE_OBS_NOISE_STD, dtype=DTYPE,
    )
    rng = np.random.default_rng(SEED)
    states, _, observations = generate_data(model, T=T, rng=rng)
    return model, observations, states


@pytest.fixture(scope="session")
def observations_tf(lg_model_and_data):
    """Observations as float64 TF tensor."""
    _, observations, _ = lg_model_and_data
    return tf.constant(observations, dtype=DTYPE)


# ============================================================================
# HELPERS
# ============================================================================

def _make_param_specs(init_value=INIT_OBS_NOISE_STD):
    """ParameterSpec for obs_noise_std with LogNormal(0,1) prior."""
    return {
        'obs_noise_std': ParameterSpec(
            name='obs_noise_std',
            init_value=init_value,
            constraint='positive',
            prior=tfp.distributions.LogNormal(
                loc=tf.constant(0.0, dtype=DTYPE),
                scale=tf.constant(1.0, dtype=DTYPE),
            ),
        )
    }


def _make_bpf_kwargs(resampling_method, stop_gradient):
    """Filter kwargs for BootstrapPFHMC.

    Always uses eager_mode=False (compiled path) — this is the production
    path. The compiled filter passes param_values as explicit @tf.function
    arguments, so parameter updates flow correctly.
    """
    kwargs = dict(
        n_particles=N_PARTICLES,
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


def _make_ledh_kwargs(resampling_method, stop_gradient):
    """Filter kwargs for LEDHParticleFlowFilterHMC.

    Always uses eager_mode=False (compiled path) — this is the production
    path and what we need to validate. The compiled filter passes
    param_values as explicit @tf.function arguments, so parameter updates
    flow correctly through the graph.
    """
    kwargs = dict(
        n_particles=N_PARTICLES,
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


def _fresh_model(obs_noise_std=INIT_OBS_NOISE_STD):
    """Create a fresh model instance with given obs_noise_std."""
    return LinearGaussianModel(
        F=[[0.9]], B=[[1.0]], H=[[1.0]], D=[[1.0]],
        obs_noise_std=obs_noise_std, dtype=DTYPE,
    )


def kf_log_likelihood(obs_noise_std, observations):
    """Exact KF log-likelihood at a given obs_noise_std."""
    model = _fresh_model(obs_noise_std)
    kf = KalmanFilter(
        F=model.F.numpy(), B=model.B.numpy(),
        H=model.H.numpy(), D=(obs_noise_std * model.D).numpy(),
        dtype=DTYPE,
    )
    result = kf.filter(observations)
    return result.log_likelihood


def pf_log_likelihood(filter_class, filter_kwargs, obs_noise_std, observations_tf):
    """Run a particle filter at given obs_noise_std, return log-likelihood.

    Uses eager_mode=True because this evaluates at a fixed parameter
    (no HMC, no gradient). A bare model has no trainable_param_names,
    so the compiled path's _build_compiled_filter would fail.
    """
    eager_kwargs = dict(filter_kwargs)
    eager_kwargs['eager_mode'] = True
    model = _fresh_model(obs_noise_std)
    filt = filter_class(model, **eager_kwargs)
    seed = tf.constant([SEED, 0], dtype=tf.int32)
    ll = filt.log_marginal_likelihood_tf(observations_tf, seed=seed)
    return float(ll.numpy())


def compute_gradient(filter_class, filter_kwargs, observations_tf,
                     init_obs_noise_std=INIT_OBS_NOISE_STD):
    """One gradient evaluation. Returns (nlp, grad, constrained_value, log_lik)."""
    model = _fresh_model(init_obs_noise_std)
    param_specs = _make_param_specs(init_obs_noise_std)
    runner = DPFRunner(
        base_model=model,
        filter_class=filter_class,
        filter_kwargs=filter_kwargs,
        param_specs=param_specs,
        sampler='hmc',
    )
    runner._observations_tf = observations_tf
    q = runner.param_handler.unconstrained_init
    with tf.GradientTape() as tape:
        tape.watch(q)
        nlp = runner._negative_log_posterior(q)
    grad = tape.gradient(nlp, q)
    constrained = runner.param_handler.constrain(q)
    log_prior = runner.param_handler.log_prior(constrained)
    log_lik = float((-nlp - log_prior).numpy())
    sigma_val = float(constrained['obs_noise_std'].numpy())
    return float(nlp.numpy()), grad.numpy().copy(), sigma_val, log_lik


def _make_fresh_runner(filter_class, filter_kwargs, observations_tf,
                       init_obs_noise_std=INIT_OBS_NOISE_STD):
    """Create a fresh DPFRunner (new model + filter instance).

    Each runner gets its own model instance, avoiding @tf.function cache
    reuse across evaluations with different parameter values.
    """
    model = _fresh_model(init_obs_noise_std)
    param_specs = _make_param_specs(init_obs_noise_std)
    runner = DPFRunner(
        base_model=model,
        filter_class=filter_class,
        filter_kwargs=filter_kwargs,
        param_specs=param_specs,
        sampler='hmc',
    )
    runner._observations_tf = observations_tf
    return runner


def finite_difference_gradient(filter_class, filter_kwargs, observations_tf,
                               init_obs_noise_std=INIT_OBS_NOISE_STD, eps=1e-4):
    """Central finite-difference gradient of NLL w.r.t. unconstrained param.

    Uses fresh runner for each evaluation point to avoid @tf.function cache
    returning stale results when model attributes change.
    """
    q = ParameterHandler(_make_param_specs(init_obs_noise_std), dtype=DTYPE).unconstrained_init

    runner_plus = _make_fresh_runner(filter_class, filter_kwargs, observations_tf, init_obs_noise_std)
    nlp_plus = float(runner_plus._negative_log_posterior(q + eps).numpy())

    runner_minus = _make_fresh_runner(filter_class, filter_kwargs, observations_tf, init_obs_noise_std)
    nlp_minus = float(runner_minus._negative_log_posterior(q - eps).numpy())

    fd_grad = (nlp_plus - nlp_minus) / (2 * eps)
    return fd_grad


def compute_prior_gradient(init_obs_noise_std=INIT_OBS_NOISE_STD, eps=1e-4):
    """Autodiff and FD gradient of -log_prior w.r.t. unconstrained q.

    Isolates the prior + Softplus bijector chain — no filter involved.
    Returns: (neg_log_prior_value, autodiff_grad, fd_grad)
    """
    param_specs = _make_param_specs(init_obs_noise_std)
    handler = ParameterHandler(param_specs, dtype=DTYPE)
    q = handler.unconstrained_init

    # Autodiff
    with tf.GradientTape() as tape:
        tape.watch(q)
        constrained = handler.constrain(q)
        neg_log_prior = -handler.log_prior(constrained)
    ad_grad = float(tape.gradient(neg_log_prior, q).numpy()[0])

    # Finite difference
    q_plus, q_minus = q + eps, q - eps
    nlp_plus = float(-handler.log_prior(handler.constrain(q_plus)).numpy())
    nlp_minus = float(-handler.log_prior(handler.constrain(q_minus)).numpy())
    fd_grad = (nlp_plus - nlp_minus) / (2 * eps)

    return float(neg_log_prior.numpy()), ad_grad, fd_grad


def compute_likelihood_gradient(filter_class, filter_kwargs, observations_tf,
                                init_obs_noise_std=INIT_OBS_NOISE_STD, eps=1e-4):
    """Autodiff and FD gradient of -log_likelihood w.r.t. unconstrained q.

    Isolates the filter backward pass — no prior involved.
    Uses fresh runners for FD to avoid stale results.
    Returns: (neg_log_lik_value, autodiff_grad, fd_grad)
    """
    q = ParameterHandler(_make_param_specs(init_obs_noise_std), dtype=DTYPE).unconstrained_init
    seed = tf.constant([42, 0], dtype=tf.int32)

    # Autodiff — fresh runner
    runner_ad = _make_fresh_runner(filter_class, filter_kwargs, observations_tf, init_obs_noise_std)
    with tf.GradientTape() as tape:
        tape.watch(q)
        constrained = runner_ad.param_handler.constrain(q)
        runner_ad.diff_model.update_parameters(constrained)
        log_lik = runner_ad.filter_obj.log_marginal_likelihood_tf(
            runner_ad._observations_tf, seed=seed)
        neg_log_lik = -log_lik
    ad_grad = float(tape.gradient(neg_log_lik, q).numpy()[0])
    neg_log_lik_val = float(neg_log_lik.numpy())

    # FD+ — fresh runner
    runner_plus = _make_fresh_runner(filter_class, filter_kwargs, observations_tf, init_obs_noise_std)
    constrained_plus = runner_plus.param_handler.constrain(q + eps)
    runner_plus.diff_model.update_parameters(constrained_plus)
    ll_plus = float(runner_plus.filter_obj.log_marginal_likelihood_tf(
        runner_plus._observations_tf, seed=seed).numpy())

    # FD- — fresh runner
    runner_minus = _make_fresh_runner(filter_class, filter_kwargs, observations_tf, init_obs_noise_std)
    constrained_minus = runner_minus.param_handler.constrain(q - eps)
    runner_minus.diff_model.update_parameters(constrained_minus)
    ll_minus = float(runner_minus.filter_obj.log_marginal_likelihood_tf(
        runner_minus._observations_tf, seed=seed).numpy())

    fd_grad = (-ll_plus - (-ll_minus)) / (2 * eps)

    return neg_log_lik_val, ad_grad, fd_grad
