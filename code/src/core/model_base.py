"""Abstract base class for state-space models (TensorFlow)."""

from abc import ABC, abstractmethod
import tensorflow as tf


class StateSpaceModel(ABC):
    """
    Abstract base class for TensorFlow state-space models.

    All models must implement:
    1. Properties: state_dim, obs_dim
    2. Sampling methods (for data generation & particle filters)
    3. Deterministic methods (for Kalman/EKF/UKF)
    4. Log-probability methods (for particle filters)
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state space."""
        pass

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of observation space."""
        pass

    # Sampling methods (for data generation & particle filters)

    @abstractmethod
    def sample_initial_state(self, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample from initial state distribution p(x_0).

        Args:
            seed: TensorFlow random seed, shape (2,), dtype int32

        Returns:
            Initial state of shape (state_dim,)
        """
        pass

    @abstractmethod
    def sample_state_transition(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample next state from transition distribution p(x_{t+1} | x_t).

        Args:
            x: Current state of shape (state_dim,)
            seed: TensorFlow random seed, shape (2,), dtype int32

        Returns:
            Next state of shape (state_dim,)
        """
        pass

    @abstractmethod
    def sample_observation(self, x: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """
        Sample observation from observation distribution p(y_t | x_t).

        Args:
            x: Current state of shape (state_dim,)
            seed: TensorFlow random seed, shape (2,), dtype int32

        Returns:
            Observation of shape (obs_dim,)
        """
        pass

    # Deterministic methods (for Kalman/EKF/UKF)

    @abstractmethod
    def state_transition_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of state transition: E[x_{t+1} | x_t]."""
        pass

    @abstractmethod
    def state_transition_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of state transition: Cov[x_{t+1} | x_t]."""
        pass

    @abstractmethod
    def state_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of state transition function: df/dx."""
        pass

    @abstractmethod
    def observation_mean(self, x: tf.Tensor) -> tf.Tensor:
        """Mean of observation: E[y_t | x_t]."""
        pass

    @abstractmethod
    def observation_cov(self, x: tf.Tensor) -> tf.Tensor:
        """Covariance of observation: Cov[y_t | x_t]."""
        pass

    @abstractmethod
    def observation_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Jacobian of observation function: dh/dx."""
        pass

    # For particle filters

    @abstractmethod
    def log_observation_prob(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Log probability of observation: log p(y_t | x_t).

        Args:
            y: Observation of shape (obs_dim,)
            x: State of shape (state_dim,) or (N, state_dim)

        Returns:
            Log probability (scalar or (N,))
        """
        pass

    # Batch methods - default implementations (override for efficiency)

    def sample_initial_state_batch(self, n: int, seed: tf.Tensor) -> tf.Tensor:
        """Sample n initial states. Default: loop."""
        seeds = tf.random.experimental.stateless_split(seed, num=n)
        return tf.stack([self.sample_initial_state(seeds[i]) for i in range(n)])

    def state_transition_batch(self, particles: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Propagate batch of particles. Default: loop."""
        N = particles.shape[0]
        seeds = tf.random.experimental.stateless_split(seed, num=N)
        return tf.stack([self.sample_state_transition(particles[i], seeds[i]) for i in range(N)])

    def state_transition_mean_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Compute transition means for batch. Default: loop."""
        return tf.stack([self.state_transition_mean(x) for x in particles])

    def state_transition_cov_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Compute transition covariance. Default: assumes constant Q."""
        return self.state_transition_cov(particles[0])

    def log_observation_prob_batch(self, observation: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
        """Compute log obs probs for batch. Default: loop."""
        return tf.stack([self.log_observation_prob(observation, x) for x in particles])

    def observation_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Compute observation Jacobian for batch of states.

        Args:
            particles: (N, state_dim)

        Returns:
            H_batch: (N, obs_dim, state_dim)

        Default: tf.map_fn over observation_jacobian. Override for efficiency.
        """
        return tf.map_fn(
            self.observation_jacobian,
            particles,
            fn_output_signature=tf.TensorSpec([self.obs_dim, self.state_dim], particles.dtype)
        )

    def observation_function_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Compute observation function h(x) for batch of states.

        Args:
            particles: (N, state_dim)

        Returns:
            h_batch: (N, obs_dim)

        Default: tf.map_fn over observation_function. Override for efficiency.
        """
        return tf.map_fn(
            self.observation_function,
            particles,
            fn_output_signature=tf.TensorSpec([self.obs_dim], particles.dtype)
        )

    def state_jacobian_batch(self, particles: tf.Tensor) -> tf.Tensor:
        """Compute state transition Jacobian for batch of states.

        Args:
            particles: (N, state_dim)

        Returns:
            F_batch: (N, state_dim, state_dim)

        Default: tf.map_fn over state_jacobian. Override for efficiency.
        """
        return tf.map_fn(
            self.state_jacobian,
            particles,
            fn_output_signature=tf.TensorSpec([self.state_dim, self.state_dim], particles.dtype)
        )

    @property
    def has_state_dependent_noise(self) -> bool:
        """Whether process noise Q depends on state. Default: False."""
        return False
