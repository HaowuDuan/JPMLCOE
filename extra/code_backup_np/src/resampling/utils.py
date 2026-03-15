"""Utility functions for particle filter resampling (TensorFlow)."""

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    raise ImportError("TensorFlow is required for resampling utilities")


@tf.function
def effective_sample_size(weights: tf.Tensor) -> tf.Tensor:
    """
    Compute Effective Sample Size (ESS).

    ESS = 1 / sum(w_i^2)

    where weights are normalized (sum to 1).

    Args:
        weights: Normalized weights of shape (N,) or (batch, N)

    Returns:
        ESS of shape () or (batch,)
    """
    return 1.0 / tf.reduce_sum(weights ** 2, axis=-1)


@tf.function
def normalize_log_weights(log_weights: tf.Tensor) -> tf.Tensor:
    """
    Normalize weights from log-space using log-sum-exp trick.

    Args:
        log_weights: Log weights of shape (N,) or (batch, N)

    Returns:
        Normalized weights (not in log-space) of same shape
    """
    # Log-sum-exp trick for numerical stability
    max_log = tf.reduce_max(log_weights, axis=-1, keepdims=True)
    weights_unnorm = tf.exp(log_weights - max_log)
    weights = weights_unnorm / tf.reduce_sum(weights_unnorm, axis=-1, keepdims=True)
    return weights


@tf.function
def normalize_weights(weights: tf.Tensor) -> tf.Tensor:
    """
    Normalize weights (already in regular space).

    Args:
        weights: Weights of shape (N,) or (batch, N)

    Returns:
        Normalized weights of same shape
    """
    return weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
