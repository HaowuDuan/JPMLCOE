"""Base class for particle flow filters."""

import numpy as np
import tensorflow as tf
import time
from typing import Tuple, Optional, Callable
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
        self.dtype = getattr(model, 'dtype', tf.float64)
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

        # State (TF Variable, set in initialize())
        self.particles = None
        self.random_state = np.random.default_rng()

        # Storage (lists of TF tensors, converted in filter())
        self.means = []
        self.covs = []

        # Diagnostic tracking (for flow filters with equal weights)
        self.ess_history = []
        self.weights_history = []

    def _estimate_mean_cov(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Estimate mean and covariance from equally-weighted particles.

        Returns TF tensors (no numpy conversion).

        Returns:
            mean: TF Tensor, shape (state_dim,)
            cov: TF Tensor, shape (state_dim, state_dim)
        """
        particles = self.particles.value() if isinstance(self.particles, tf.Variable) else self.particles
        mean = tf.reduce_mean(particles, axis=0)
        diff = particles - mean
        cov = tf.matmul(diff, diff, transpose_a=True) / tf.cast(self.n_particles, self.dtype)
        return mean, cov

    def _make_tf_seed(self) -> tf.Tensor:
        """Create a TF-compatible seed from the numpy random state."""
        return tf.constant(self.random_state.integers(0, 2**31, size=2), dtype=tf.int32)

    def predict(self):
        """Prediction step: propagate particles through state transition using batch method."""
        seed = self._make_tf_seed()
        particles_predicted = self.model.state_transition_batch(self.particles.value(), seed)
        self.particles.assign(particles_predicted)

    def update(self, y: tf.Tensor):
        """Update step - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement update()")

    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles from initial distribution as TF Variable."""
        if random_state is not None:
            self.random_state = random_state

        # Sample initial particles using model's batch method
        seed = self._make_tf_seed()
        particles_tf = self.model.sample_initial_state_batch(self.n_particles, seed)
        self.particles = tf.Variable(particles_tf, dtype=self.dtype)

        # Reset storage
        self.means = []
        self.covs = []
        self.ess_history = []
        self.weights_history = []

    def filter(self, observations: np.ndarray,
               random_state: Optional[np.random.Generator] = None,
               progress_callback: Optional[Callable[[int, int, float], None]] = None) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Shape (T, obs_dim)
            random_state: Optional random generator
            progress_callback: Optional callback(t, T, step_time_sec) called after each step

        Returns:
            FilterResult with means, covariances, and diagnostics
        """
        self.initialize(random_state)
        T = len(observations)

        # Pre-convert observations to TF once
        obs_tf = tf.constant(observations, dtype=self.dtype)

        for t in range(T):
            t0 = time.perf_counter()
            self.predict()
            self.update(obs_tf[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)
            self.covs.append(cov)
            if progress_callback is not None:
                progress_callback(t, T, time.perf_counter() - t0)

        # Convert accumulated TF tensors to numpy once
        means_np = tf.stack(self.means).numpy()
        covs_np = tf.stack(self.covs).numpy()

        # Flow filters maintain equal weights — ESS is always N
        ess_np = np.full(T, float(self.n_particles))
        weights_np = np.ones((T, self.n_particles)) / self.n_particles

        return FilterResult(
            means=means_np,
            covs=covs_np,
            log_likelihood=None,
            ess=ess_np,
            weights_history=weights_np,
            metadata={
                'filter_type': self.__class__.__name__,
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'integration_method': self.integration_method
            }
        )
