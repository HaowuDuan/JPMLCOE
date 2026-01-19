"""Bootstrap Particle Filter - Simple NumPy implementation."""

import numpy as np

from .particle_base import ParticleFilterBase
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel


class ParticleFilter(ParticleFilterBase):
    """
    Bootstrap Particle Filter for nonlinear/non-Gaussian state-space models.

    Simple NumPy implementation without TensorFlow complications.
    Inherits diagnostic tracking from ParticleFilterBase.
    """

    def __init__(self, model: StateSpaceModel, n_particles: int = 1000,
                 resample_threshold: float = 0.5):
        """
        Initialize the Particle Filter.

        Args:
            model: State-space model instance
            n_particles: Number of particles (default 1000)
            resample_threshold: Resample when ESS/N < threshold (default 0.5)
        """
        super().__init__(model, n_particles, resample_threshold)
        self.rng = np.random.default_rng()

    def reset(self):
        """Reset filter state."""
        super().reset()

    def _initialize(self):
        """Initialize particles from initial distribution."""
        self.particles = np.array([
            self.model.sample_initial_state(self.rng)
            for _ in range(self.n_particles)
        ])
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _predict_step(self):
        """Propagate particles through dynamics."""
        for i in range(self.n_particles):
            self.particles[i] = self.model.sample_state_transition(
                self.particles[i], self.rng
            )

    def _update_step(self, observation):
        """Update particle weights based on observation."""
        # Compute observation log-probabilities for all particles
        log_weights = np.array([
            self.model.log_observation_prob(observation, self.particles[i])
            for i in range(self.n_particles)
        ])

        # Update weights: w_new = w_old * p(y|x)
        # In log space: log(w_new) = log(w_old) + log(p(y|x))
        log_weights = np.log(self.weights + 1e-30) + log_weights

        # Compute marginal likelihood: p(y_t | y_{1:t-1}) = sum(w_{t-1} * p(y|x))
        log_likelihood = self._log_sum_exp(log_weights)
        self.log_likelihoods.append(log_likelihood)

        # Normalize weights
        self.weights = np.exp(log_weights - self._log_sum_exp(log_weights))

    def _log_sum_exp(self, log_values):
        """Numerically stable log-sum-exp."""
        max_val = np.max(log_values)
        return max_val + np.log(np.sum(np.exp(log_values - max_val)))

    def _estimate_mean_cov(self):
        """Estimate mean and covariance from weighted particles."""
        # Weighted mean
        mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)

        # Weighted covariance
        diff = self.particles - mean
        cov = np.sum(
            self.weights[:, np.newaxis, np.newaxis] *
            np.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )

        return mean, cov

    def _resample(self):
        """Systematic resampling."""
        N = self.n_particles

        # Compute cumulative sum
        cumsum = np.cumsum(self.weights)

        # Generate systematic samples
        u = (self.rng.random() + np.arange(N)) / N

        # Find indices
        indices = np.searchsorted(cumsum, u)

        # Resample particles
        self.particles = self.particles[indices]

        # Reset weights to uniform
        self.weights = np.ones(N) / N

    def filter(self, observations: np.ndarray) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Array of shape (T, obs_dim) containing observations

        Returns:
            FilterResult with filtered means, covariances, and diagnostics
        """
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = observations[:, np.newaxis]

        T = len(observations)

        # Initialize
        self._initialize()

        # Run filter
        for t in range(T):
            # Predict
            self._predict_step()

            # Update
            self._update_step(observations[t])

            # Estimate mean and covariance BEFORE resampling
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
            
            # Compute ESS
            ess = self._effective_sample_size(self.weights)

            # Track diagnostics BEFORE resampling (like TensorFlow version)
            self.ess_history.append(ess)
            self.weights_history.append(self.weights.copy())

            # Check if resampling needed
            if ess < self.resample_threshold * self.n_particles:
                self._resample()
                # Track resampling event
                self.resampled_at.append(t)
                n_unique = len(np.unique(self.particles, axis=0))
                self.n_unique_particles.append(n_unique)

        # Compute resampling rate
        resampling_rate = len(self.resampled_at) / T if T > 0 else 0.0

        return FilterResult(
            means=np.array(self.means),
            covs=np.array(self.covs),
            log_likelihood=np.sum(self.log_likelihoods),
            # Diagnostics as top-level fields (saved as separate .npy files)
            log_likelihoods=np.array(self.log_likelihoods),
            ess=np.array(self.ess_history),
            weights_history=np.array(self.weights_history),
            resampled_at=self.resampled_at,
            n_unique=np.array(self.n_unique_particles),
            # Metadata for scalars only
            metadata={
                'filter_type': 'ParticleFilter',
                'n_particles': self.n_particles,
                'resampling_rate': resampling_rate
            }
        )
