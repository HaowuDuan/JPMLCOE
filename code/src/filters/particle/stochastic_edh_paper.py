"""
StochasticEDHFlowPaper — paper-reproduction variant (arXiv:2107.04672, Section 4).

Uses FIXED prior mean/covariance as global linearization point η̄₀ / P,
as in the paper's static two-sensor bearing experiment (one time step,
linearization does not update from particle state).
"""
import tensorflow as tf
from .stochastic_edh import StochasticEDHFlow


class StochasticEDHFlowPaper(StochasticEDHFlow):
    """
    Paper-reproduction variant of StochasticEDHFlow.

    Overrides initialize() and predict() so that η̄₀ and P are fixed
    at model.mu_0 and model.Sigma_0 (the exact prior), never updated
    from the particle ensemble.  This matches the global linearization
    used in Dai & Daum (2021), arXiv:2107.04672, Section 4.
    """

    def initialize(self, random_state=None):
        """Sample particles from prior; EKF fixed at exact prior (not particle empirical)."""
        super().initialize(random_state)
        # Parent sets EKF to particle empirical mean/cov — override with exact prior.
        prior_mean = tf.cast(self.model.mu_0, self.dtype)
        prior_cov  = tf.cast(self.model.Sigma_0, self.dtype)
        if self.global_filter is not None:
            self.global_filter.mean_0  = prior_mean
            self.global_filter.Sigma_0 = prior_cov
            self.global_filter.mean.assign(prior_mean)
            self.global_filter.cov.assign(prior_cov)
        self.predicted_cov = prior_cov
        self.eta_bar_0     = prior_mean

    def predict(self):
        """Propagate particles; EKF advances from its own state (no particle feedback)."""
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        particles_predicted = self.model.state_transition_batch(
            self.particles.value(), seed)
        self.particles.assign(particles_predicted)
        # EKF predicts from its own mean — no particle empirical mean overwrite
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.value()
        self.eta_bar_0     = self.global_filter.mean.value()
