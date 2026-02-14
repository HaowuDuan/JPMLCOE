"""Protocol for filters that support differentiable log-likelihood computation."""

import tensorflow as tf
from typing import Protocol, runtime_checkable


@runtime_checkable
class DifferentiableFilter(Protocol):
    """
    Protocol for filters usable with HMC parameter inference.

    Any filter that implements this protocol can be plugged into the DPF
    framework. The key requirement: log_marginal_likelihood_tf() must run
    entirely inside TensorFlow's computation graph so tf.GradientTape
    can differentiate through it.

    Contract:
        - All operations must be TF ops (no .numpy(), no Python float)
        - Returns a tf.Tensor scalar, not a Python float
        - Model parameters are read via model methods (e.g. model.state_transition_cov())
          which pick up dynamic values set by DifferentiableModel
    """

    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor
    ) -> tf.Tensor:
        """
        Run the full filter, return total log p(y_{1:T}) as a TF scalar.

        Args:
            observations: (T, obs_dim), dtype matching model
            seed: TF random seed (2,), dtype int32

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        ...
