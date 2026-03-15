"""TensorFlow Probability Particle Filter - Cross-validation implementation."""

import numpy as np
from typing import Optional
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False

from .particle_base import ParticleFilterBase
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel
from ...utils.distributions import normalize_log_weights


class ParticleFilterTFP(ParticleFilterBase):
    """
    Particle Filter using TensorFlow Probability's implementation.
    
    This is a wrapper that adapts TFP's functional particle filter API
    to your codebase's object-oriented Filter interface. Useful for:
    - Cross-validation against custom implementations
    - Benchmarking performance
    - Verifying correctness
    
    Note: TFP's particle filter processes all observations at once (batch mode),
    so predict()/update() are not supported. Use filter() method instead.
    
    Reference:
        tfp.experimental.mcmc.particle_filter
        https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/particle_filter
    """

    def __init__(self, model: StateSpaceModel, n_particles: int = 1000,
                 resample_threshold: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize TFP Particle Filter.

        Args:
            model: State-space model (must have initial_state_mean, initial_state_cov)
            n_particles: Number of particles
            resample_threshold: Resample when ESS/N < threshold (for compatibility; TFP uses internal threshold)
            seed: Random seed for reproducibility
        """
        if not TFP_AVAILABLE:
            raise ImportError("tensorflow-probability required for ParticleFilterTFP")
        
        super().__init__(model, n_particles, resample_threshold)
        self.seed = seed
        
        # Verify model has required methods/properties
        if not hasattr(model, 'initial_state_mean') or not hasattr(model, 'initial_state_cov'):
            raise AttributeError(
                "Model must have initial_state_mean and initial_state_cov properties for TFP filter"
            )
        if not hasattr(model, 'observation_mean'):
            raise AttributeError("Model must have observation_mean() method for TFP filter")
        
    def reset(self):
        """Reset filter state."""
        super().reset()

    def _create_initial_state_prior(self):
        """
        Create TFP distribution for initial state.
        
        Uses model's initial_state_mean and initial_state_cov.
        
        Returns:
            tfd.Distribution for p(x_0)
        """
        mean = tf.constant(self.model.initial_state_mean, dtype=tf.float32)
        cov = tf.constant(self.model.initial_state_cov, dtype=tf.float32)
        
        # Use MultivariateNormalTriL for numerical stability
        return tfd.MultivariateNormalTriL(
            loc=mean,
            scale_tril=tf.linalg.cholesky(cov)
        )

    def _create_transition_fn(self):
        """
        Create transition function for TFP.
        
        Returns a function: (step, state) -> tfd.Distribution over next state
        """
        F_tf = tf.constant(self.model.F, dtype=tf.float32)
        Q_tf = tf.constant(self.model.Q, dtype=tf.float32)
        Q_chol = tf.linalg.cholesky(Q_tf)
        
        @tf.function
        def transition_fn(step, state):
            """
            Transition function: p(x_{t+1} | x_t).
            
            Args:
                step: Time step (ignored for time-homogeneous systems)
                state: Current state tensor (n_particles, state_dim)
                
            Returns:
                tfd.Distribution over next states
            """
            # For linear Gaussian: x' = F @ x + N(0, Q)
            # Mean for each particle
            if len(state.shape) == 1:
                # Single state
                mean = F_tf @ state
            else:
                # Batch of states (n_particles, state_dim)
                mean = tf.linalg.matvec(F_tf, state)
            
            # Independent MVN for each particle
            return tfd.MultivariateNormalTriL(
                loc=mean,
                scale_tril=Q_chol
            )
        
        return transition_fn

    def _create_observation_fn(self):
        """
        Create observation function for TFP.
        
        Returns a function: (step, state) -> log p(y_t | state)
        """
        R_tf = tf.constant(self.model.R, dtype=tf.float32)
        R_chol = tf.linalg.cholesky(R_tf)
        
        @tf.function
        def observation_fn(step, state):
            """
            Observation function: p(y_t | x_t).
            
            Args:
                step: Time step
                state: State tensor (n_particles, state_dim) or (state_dim,)
                
            Returns:
                tfd.Distribution over observations
            """
            # Use tf.py_function to wrap NumPy-based observation_mean
            def compute_h_numpy(x_np):
                """Compute h(x) using model's observation_mean (NumPy)."""
                h_np = self.model.observation_mean(x_np)
                return h_np.astype(np.float32)
            
            if len(state.shape) == 1:
                # Single state - wrap in py_function
                h_mean = tf.py_function(
                    compute_h_numpy,
                    [state],
                    tf.float32
                )
                h_mean.set_shape([self.model.obs_dim])
            else:
                # Batch of states - map over batch dimension
                def compute_h_single(x):
                    h = tf.py_function(
                        compute_h_numpy,
                        [x],
                        tf.float32
                    )
                    h.set_shape([self.model.obs_dim])
                    return h
                
                h_mean = tf.map_fn(
                    compute_h_single,
                    state,
                    dtype=tf.float32,
                    parallel_iterations=10
                )
            
            # Return MVN distribution
            return tfd.MultivariateNormalTriL(
                loc=h_mean,
                scale_tril=R_chol
            )
        
        return observation_fn

    def filter(self, observations: np.ndarray) -> FilterResult:
        """
        Run TFP particle filter on all observations.
        
        Args:
            observations: Observation sequence (T, obs_dim)
            
        Returns:
            FilterResult with means, covariances, log-likelihood, metadata
        """
        T = len(observations)
        
        # Convert to TF tensors
        obs_tf = tf.constant(observations, dtype=tf.float32)
        
        # Create distributions and functions
        initial_state_prior = self._create_initial_state_prior()
        transition_fn = self._create_transition_fn()
        observation_fn = self._create_observation_fn()
        
        # Run TFP particle filter
        # Note: TFP's particle_filter returns particles at each timestep
        # The state is a WeightedParticles object with .particles and .log_weights
        result = tfp.experimental.mcmc.particle_filter(
            observations=obs_tf,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            num_particles=self.n_particles,
            trace_fn=lambda state, kernel_results: {
                'particles': state.particles,
                'log_weights': state.log_weights,
                'parent_indices': kernel_results.parent_indices if hasattr(kernel_results, 'parent_indices') else None
            },
            seed=self.seed if self.seed is not None else None
        )
        
        # Extract results
        log_likelihood = float(result.log_marginal_likelihood.numpy())
        traced = result.traced_results
        
        particles_history = traced['particles'].numpy()  # (T, n_particles, state_dim)
        log_weights_history = traced['log_weights'].numpy()  # (T, n_particles)
        
        # Compute means and covariances from weighted particles
        means = np.zeros((T, self.model.state_dim))
        covs = np.zeros((T, self.model.state_dim, self.model.state_dim))
        
        for t in range(T):
            particles_t = particles_history[t]
            log_weights_t = log_weights_history[t]
            
            # Normalize weights using shared utility
            weights_t = normalize_log_weights(log_weights_t)
            
            # Weighted mean
            means[t] = np.sum(weights_t[:, np.newaxis] * particles_t, axis=0)
            
            # Weighted covariance
            diff = particles_t - means[t]
            covs[t] = np.sum(
                weights_t[:, np.newaxis, np.newaxis] * 
                np.einsum('ij,ik->ijk', diff, diff),
                axis=0
            )
            
            # Store diagnostics for compatibility
            self.log_likelihoods.append(float(np.sum(log_weights_t)))
            
            # Compute ESS
            ess = 1.0 / np.sum(weights_t ** 2)
            self.ess.append(float(ess))
        
        # Build metadata
        metadata = {
            'filter_type': 'ParticleFilterTFP',
            'n_particles': self.n_particles,
            'tfp_version': tfp.__version__ if TFP_AVAILABLE else 'N/A',
            'log_likelihoods': np.array(self.log_likelihoods),
            'ess': np.array(self.ess),
            'mean_ess': np.mean(self.ess) if self.ess else 0.0
        }
        
        return FilterResult(
            means=means,
            covs=covs,
            log_likelihood=log_likelihood,
            metadata=metadata
        )

    def predict(self):
        """Not implemented - TFP filter is batch-only."""
        raise NotImplementedError(
            "TFP particle filter operates in batch mode only. "
            "Use filter(observations) method instead of step-by-step predict/update."
        )
    
    def update(self, observation: np.ndarray):
        """Not implemented - TFP filter is batch-only."""
        raise NotImplementedError(
            "TFP particle filter operates in batch mode only. "
            "Use filter(observations) method instead of step-by-step predict/update."
        )

