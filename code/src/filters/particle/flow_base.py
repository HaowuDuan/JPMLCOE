"""Base class for particle flow filters."""

import numpy as np
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import os
from ...core.types import FilterResult


class FlowFilterBase:
    """
    Base class for particle flow filters.

    Flow filters avoid resampling by moving particles along deterministic flows
    from prior to posterior distribution.
    """

    def __init__(self, model, n_particles: int = 1000,
                 n_lambda_steps: int = 100,
                 integration_method: str = 'euler',
                 n_threads: Optional[int] = None):
        """
        Initialize base flow filter.

        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            integration_method: 'euler' or 'rk4' for ODE integration
            n_threads: Number of threads (None = use CPU count)
        """
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.integration_method = integration_method

        # Threading setup
        if n_threads is None:
            self.n_threads = os.cpu_count() if os.cpu_count() else 1
        else:
            self.n_threads = max(1, int(n_threads))

        # State
        self.particles = None
        self.random_state = np.random.default_rng()

        # Storage
        self.means = []
        self.covs = []

        # Diagnostic tracking (for flow filters with equal weights)
        self.ess_history = []
        self.weights_history = []

    def _compute_observation_matrix(self, particles: np.ndarray) -> np.ndarray:
        """
        Evaluate observation function h(x) for all particles.

        Args:
            particles: Shape (N, state_dim)

        Returns:
            h_particles: Shape (N, obs_dim)
        """
        if self.n_threads > 1:
            def compute_h(i):
                return self.model.observation_function(particles[i])

            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                h_particles = np.array(list(executor.map(compute_h, range(self.n_particles))))
        else:
            h_particles = np.array([
                self.model.observation_function(particles[i])
                for i in range(self.n_particles)
            ])
        return h_particles

    def _estimate_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean and covariance from equally-weighted particles.

        Returns:
            mean: Shape (state_dim,)
            cov: Shape (state_dim, state_dim)
        """
        mean = np.mean(self.particles, axis=0)
        diff = self.particles - mean
        cov = (diff.T @ diff) / self.n_particles
        return mean, cov

    def predict(self):
        """Prediction step: propagate particles through state transition."""
        if self.n_threads > 1:
            # Pre-generate seeds to avoid race conditions
            seeds = self.random_state.integers(0, 2**32, size=self.n_particles)

            def propagate_particle(args):
                i, seed = args
                thread_rng = np.random.default_rng(seed)
                return self.model.sample_state_transition(self.particles[i], thread_rng)

            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                self.particles = np.array(list(executor.map(propagate_particle,
                                                           zip(range(self.n_particles), seeds))))
        else:
            self.particles = np.array([
                self.model.sample_state_transition(self.particles[i], self.random_state)
                for i in range(self.n_particles)
            ])

    def update(self, y: np.ndarray):
        """Update step - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement update()")

    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles from initial distribution."""
        if random_state is not None:
            self.random_state = random_state

        if self.n_threads > 1:
            # Pre-generate seeds
            seeds = self.random_state.integers(0, 2**32, size=self.n_particles)

            def sample_particle(args):
                i, seed = args
                thread_rng = np.random.default_rng(seed)
                return self.model.sample_initial_state(thread_rng)

            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                self.particles = np.array(list(executor.map(sample_particle, enumerate(seeds))))
        else:
            self.particles = np.array([
                self.model.sample_initial_state(self.random_state)
                for _ in range(self.n_particles)
            ])

        # Reset storage
        self.means = []
        self.covs = []
        self.ess_history = []
        self.weights_history = []

    def filter(self, observations: np.ndarray,
               random_state: Optional[np.random.Generator] = None) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Shape (T, obs_dim)
            random_state: Optional random generator

        Returns:
            FilterResult with means, covariances, and diagnostics
        """
        self.initialize(random_state)
        T = len(observations)

        for t in range(T):
            self.predict()
            self.update(observations[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)

            # Track diagnostics: flow filters maintain equal weights
            uniform_weights = np.ones(self.n_particles) / self.n_particles
            self.weights_history.append(uniform_weights.copy())
            # ESS is always N for equal weights
            self.ess_history.append(float(self.n_particles))

        return FilterResult(
            means=np.array(self.means),
            covs=np.array(self.covs),
            # NOTE: Flow filters don't compute log-likelihood by default because they use
            # deterministic flow with equal weights (1/N) throughout. The flow geometrically
            # corrects particle positions without computing observation likelihoods.
            # TODO: Could add approximate log-likelihood by evaluating p(y|x) at flowed particles
            # using logsumexp(log_probs) - log(N). See invertible versions for exact computation.
            log_likelihood=None,
            ess=np.array(self.ess_history),
            weights_history=np.array(self.weights_history),
            metadata={
                'filter_type': self.__class__.__name__,
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'integration_method': self.integration_method
            }
        )
