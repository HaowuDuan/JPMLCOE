"""Bootstrap Particle Filter - TensorFlow implementation."""

import numpy as np
from typing import Callable, Dict, Any, Optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    raise ImportError("TensorFlow is required for TF particle filter")

from .particle_base import ParticleFilterBase
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel
from ...resampling import systematic_resample, soft_resample, ot_entropy_resample


class ParticleFilterTF(ParticleFilterBase):
    """
    Bootstrap Particle Filter using TensorFlow.

    Uses @tf.function with for-loop instead of tf.scan for simplicity.
    Inherits diagnostic tracking from ParticleFilterBase.
    """

    def __init__(self, model: StateSpaceModel, n_particles: int = 1000,
                 resample_threshold: float = 0.5, track_diagnostics: bool = True,
                 resampling_method: Optional[Callable] = None,
                 resampling_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TensorFlow Particle Filter.

        Args:
            model: State-space model instance (must have TF methods)
            n_particles: Number of particles (default 1000)
            resample_threshold: Resample when ESS/N < threshold (default 0.5)
            track_diagnostics: Whether to track detailed diagnostics (default True)
                              Set to False for maximum speed
            resampling_method: Resampling function to use. Options:
                              - systematic_resample (default)
                              - soft_resample
                              - ot_entropy_resample
                              - Any custom callable with signature (particles, weights, seed, **kwargs)
            resampling_config: Dictionary of additional parameters for resampling method.
                              For soft_resample: {'alpha': 0.9}
                              For ot_entropy_resample: {'epsilon': 0.5, 'max_iter': 100, ...}
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for ParticleFilterTF")

        super().__init__(model, n_particles, resample_threshold)
        self.model = model
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.np_dtype = np.float64 if self.dtype == tf.float64 else np.float32
        self.track_diagnostics = track_diagnostics
        
        # Handle resampling method configuration
        # If resampling_method is a string, convert it to the actual function
        if isinstance(resampling_method, str):
            method_map = {
                'systematic': systematic_resample,
                'soft': soft_resample,
                'ot_entropy': ot_entropy_resample,
            }
            self.resampling_method = method_map.get(resampling_method, systematic_resample)
            self.resampling_method_name = resampling_method
        elif resampling_method is not None:
            self.resampling_method = resampling_method
            self.resampling_method_name = getattr(resampling_method, '__name__', 'custom')
        else:
            # Default to systematic
            self.resampling_method = systematic_resample
            self.resampling_method_name = 'systematic'
        
        # Convert resampling config values to Python scalars
        # This prevents them from becoming symbolic tensors in @tf.function context
        self.resampling_config = {}
        if resampling_config is not None:
            for key, value in resampling_config.items():
                # Convert numeric values to Python scalars
                if isinstance(value, (int, np.integer)):
                    self.resampling_config[key] = int(value)
                elif isinstance(value, (float, np.floating)):
                    self.resampling_config[key] = float(value)
                else:
                    self.resampling_config[key] = value

    def reset(self):
        """Reset filter state."""
        super().reset()

    def _resample(self, particles, weights, seed):
        """
        Wrapper for resampling that returns (particles, weights).

        Args:
            particles: Particle positions (N, state_dim)
            weights: Normalized weights (N,)
            seed: Random seed tensor (2,)

        Returns:
            Tuple of (resampled_particles, new_weights)
        """
        result = self.resampling_method(particles, weights, seed=seed, **self.resampling_config)
        return result.particles, result.weights

    @tf.function
    def _effective_sample_size_tf(self, weights):
        """Compute ESS in TensorFlow."""
        return 1.0 / tf.reduce_sum(weights ** 2)

    @tf.function
    def _log_sum_exp(self, log_values):
        """Numerically stable log-sum-exp."""
        return tf.reduce_logsumexp(log_values)

    @tf.function
    def filter_tf(self, observations: tf.Tensor, seed: tf.Tensor):
        """
        Run particle filter using TensorFlow.

        Args:
            observations: (T, obs_dim) observations
            seed: Random seed

        Returns:
            (means, covs, log_likelihoods, ess_history, weights_history, resampled_at, n_unique)
        """
        T = tf.shape(observations)[0]

        # Initialize
        particles = self.model.sample_initial_state_batch(self.n_particles, seed)
        weights = tf.ones(self.n_particles, dtype=self.dtype) / tf.cast(self.n_particles, self.dtype)

        # Storage for results
        means_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[self.state_dim])
        covs_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[self.state_dim, self.state_dim])
        log_liks_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[])

        # Storage for diagnostics
        ess_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[])
        weights_history_list = tf.TensorArray(dtype=self.dtype, size=T, element_shape=[self.n_particles])
        resampled_at_list = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        n_unique_list = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        resample_count = tf.constant(0, dtype=tf.int32)

        # Main loop
        for t in tf.range(T):
            # Split seed for this timestep
            seed, trans_seed, resample_seed = tf.unstack(
                tf.random.experimental.stateless_split(seed, num=3)
            )

            # PREDICT: Propagate particles
            particles = self.model.state_transition_batch(particles, trans_seed)

            # UPDATE: Compute weights
            log_obs_probs = self.model.log_observation_prob_batch(observations[t], particles)
            log_weights = tf.math.log(weights + tf.constant(1e-30, dtype=weights.dtype)) + log_obs_probs

            # Log-likelihood
            log_lik = self._log_sum_exp(log_weights)
            log_liks_list = log_liks_list.write(t, log_lik)

            # Normalize weights
            log_weights_norm = log_weights - self._log_sum_exp(log_weights)
            weights = tf.exp(log_weights_norm)

            # Compute mean and covariance BEFORE resampling
            mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
            diff = particles - mean[tf.newaxis, :]
            cov = tf.reduce_sum(
                weights[:, tf.newaxis, tf.newaxis] *
                tf.einsum('ij,ik->ijk', diff, diff),
                axis=0
            )

            means_list = means_list.write(t, mean)
            covs_list = covs_list.write(t, cov)

            # Track diagnostics - ESS and weights
            ess = self._effective_sample_size_tf(weights)
            ess_list = ess_list.write(t, ess)
            weights_history_list = weights_history_list.write(t, weights)

            # RESAMPLE if needed
            should_resample = ess < self.resample_threshold * tf.cast(self.n_particles, self.dtype)

            if should_resample:
                particles, weights = self._resample(particles, weights, resample_seed)

                # Track resampling event
                resampled_at_list = resampled_at_list.write(resample_count, t)

                # Count unique particles (approximate - TF doesn't have exact unique for 2D)
                # Use a simple heuristic: check first dimension for uniqueness as proxy
                if self.state_dim > 0:
                    unique_first_dim = tf.unique(particles[:, 0])[0]
                    n_unique = tf.shape(unique_first_dim)[0]
                else:
                    n_unique = self.n_particles
                n_unique_list = n_unique_list.write(resample_count, n_unique)
                resample_count += 1

        # Stack results
        means = means_list.stack()
        covs = covs_list.stack()
        log_liks = log_liks_list.stack()
        ess_history = ess_list.stack()
        weights_history = weights_history_list.stack()
        resampled_at = resampled_at_list.stack()
        n_unique = n_unique_list.stack()

        return means, covs, log_liks, ess_history, weights_history, resampled_at, n_unique

    def filter(self, observations: np.ndarray) -> FilterResult:
        """
        Run the filter on observations.

        Args:
            observations: Array of shape (T, obs_dim)

        Returns:
            FilterResult with means, covariances, and diagnostics
        """
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = observations[:, np.newaxis]

        T = len(observations)

        # Convert to TensorFlow
        observations_tf = tf.constant(observations, dtype=self.dtype)
        seed = tf.random.uniform([2], minval=0, maxval=2**31 - 1, dtype=tf.int32)

        # Run filter
        means_tf, covs_tf, log_liks_tf, ess_tf, weights_tf, resampled_at_tf, n_unique_tf = self.filter_tf(observations_tf, seed)

        # Convert back to NumPy
        means = means_tf.numpy()
        covs = covs_tf.numpy()
        log_likelihoods = log_liks_tf.numpy()
        ess_history = ess_tf.numpy()
        weights_history = weights_tf.numpy()
        resampled_at = resampled_at_tf.numpy().tolist()
        n_unique = n_unique_tf.numpy()

        # Compute resampling rate
        resampling_rate = len(resampled_at) / T if T > 0 else 0.0

        return FilterResult(
            means=means,
            covs=covs,
            log_likelihood=np.sum(log_likelihoods),
            # Diagnostics as top-level fields (saved as separate .npy files)
            log_likelihoods=log_likelihoods,
            ess=ess_history,
            weights_history=weights_history,
            resampled_at=resampled_at,
            n_unique=n_unique,
            # Metadata for scalars only
            metadata={
                'filter_type': 'ParticleFilterTF',
                'n_particles': self.n_particles,
                'resampling_rate': resampling_rate,
                'resampling_method': self.resampling_method_name,
                'resampling_config': self.resampling_config
            }
        )
