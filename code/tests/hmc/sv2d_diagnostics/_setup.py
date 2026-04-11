"""Shared setup for SV2D HMC diagnostics.

Builds the model, filter, observations, and target_log_prob_fn deterministically.
All diagnostic tests in this folder import from here so they share the SAME setup
as the smoke config that froze (configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml).

No Hydra is used — Hydra has global state that contaminates pytest sessions.
All parameters are hard-coded here to match the smoke config exactly.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import tensorflow as tf

from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel
from src.DF.parameter_handler import ParameterHandler
from src.DF.types import ParameterSpec

import tensorflow_probability as tfp


# -----------------------------------------------------------------------------
# Configuration — must match ledh_ot_sigma2_smoke.yaml exactly
# -----------------------------------------------------------------------------
DTYPE = tf.float32          # smoke config uses float32
T = 100                     # num timesteps
DATA_SEED = 42              # observation generator seed
PF_SEED = tf.constant([42, 0], dtype=tf.int32)  # particle filter seed (fixed)

# Model params
TRUE_SIGMA2 = 1.0
INIT_SIGMA2 = 1.6
MODEL_KWARGS = dict(a1=0.95, a2=0.91, sigma1=0.5, b=1.0)

# Filter params
N_PARTICLES = 300
N_LAMBDA_STEPS = 15
EPSILON_OT = 0.5
WEIGHT_CLIP = 50.0


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_observations():
    """Generate the same observation sequence the smoke config uses."""
    true_model = StochasticVolatility2DModel(
        sigma2=TRUE_SIGMA2, dtype=DTYPE, **MODEL_KWARGS,
    )
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(true_model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def build_filter(initial_sigma2=INIT_SIGMA2):
    """Build the diff_model + filter at a given initial sigma2."""
    base_model = StochasticVolatility2DModel(
        sigma2=initial_sigma2, dtype=DTYPE, **MODEL_KWARGS,
    )
    diff_model = DifferentiableModel(base_model, ["sigma2"])
    filt = LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=N_LAMBDA_STEPS,
        resampling_method="ot_entropy",
        resampling_config={"epsilon": EPSILON_OT},
        weight_clip_range=WEIGHT_CLIP,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )
    return diff_model, filt


def build_param_handler():
    """ParameterHandler matching the smoke config's prior on sigma2."""
    spec = ParameterSpec(
        name="sigma2",
        init_value=INIT_SIGMA2,
        constraint="positive",
        prior=tfp.distributions.LogNormal(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale=tf.constant(0.5, dtype=DTYPE),
        ),
    )
    return ParameterHandler({"sigma2": spec}, dtype=DTYPE)


def build_target_log_prob_fn(diff_model, filt, observations, param_handler):
    """Build target_log_prob_fn(unconstrained_q) -> scalar log_posterior.

    Mirrors hmc_runner.DPFRunner._negative_log_posterior but inverted:
    returns +log_posterior, ready for tfp.mcmc.HamiltonianMonteCarlo.
    """
    obs_tf = observations

    def target_log_prob_fn(q_unconstrained):
        constrained = param_handler.constrain(q_unconstrained)
        diff_model.update_parameters(constrained)
        log_lik = filt.log_marginal_likelihood_tf(obs_tf, seed=PF_SEED)
        log_prior = param_handler.log_prior(constrained)
        return log_lik + log_prior

    return target_log_prob_fn


def build_all(initial_sigma2=INIT_SIGMA2):
    """One-call builder that returns everything a diagnostic needs."""
    obs = build_observations()
    diff_model, filt = build_filter(initial_sigma2=initial_sigma2)
    handler = build_param_handler()
    tlp = build_target_log_prob_fn(diff_model, filt, obs, handler)
    return {
        "observations": obs,
        "diff_model": diff_model,
        "filter": filt,
        "param_handler": handler,
        "target_log_prob_fn": tlp,
        "init_unconstrained": handler.unconstrained_init,
    }
