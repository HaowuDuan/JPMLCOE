"""
Batched EKF operations for per-particle filters.

Uses true batched linear algebra (matmul with batch dimensions) instead of
tf.map_fn, for significant speedup when processing N (mean, cov) pairs.
"""

import tensorflow as tf
from typing import Tuple
from ...utils.linalg import symmetrize


@tf.function
def batched_ekf_predict(
    model,
    means: tf.Tensor,
    covs: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched EKF prediction for N particles using true batched matmul.

    Args:
        model: StateSpaceModel with state_transition_mean_batch, state_jacobian_batch,
               state_transition_cov_batch
        means: (N, state_dim)
        covs: (N, state_dim, state_dim)

    Returns:
        mean_pred: (N, state_dim)
        cov_pred: (N, state_dim, state_dim)
    """
    # Batched predicted means: (N, sd)
    mean_pred = model.state_transition_mean_batch(means)

    # Batched Jacobians: (N, sd, sd)
    F_batch = model.state_jacobian_batch(means)

    # Process noise covariance — typically constant: (sd, sd)
    Q = model.state_transition_cov_batch(means)

    # Batched covariance prediction: F @ cov @ F^T + Q
    # F_batch @ covs: (N, sd, sd)
    F_cov = tf.matmul(F_batch, covs)
    # F_cov @ F^T: (N, sd, sd)
    F_T = tf.linalg.matrix_transpose(F_batch)
    cov_pred = tf.matmul(F_cov, F_T)

    # Add Q (broadcast if Q is (sd, sd))
    if len(Q.shape) == 2:
        cov_pred = cov_pred + tf.expand_dims(Q, 0)
    else:
        cov_pred = cov_pred + Q

    # Symmetrize for numerical stability
    cov_pred = symmetrize(cov_pred)

    return mean_pred, cov_pred


@tf.function
def batched_ekf_update(
    model,
    means: tf.Tensor,
    covs: tf.Tensor,
    observation: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched EKF update for N particles using true batched matmul.

    Args:
        model: StateSpaceModel with observation_function_batch, observation_cov,
               observation_jacobian_batch
        means: (N, state_dim) predicted means
        covs: (N, state_dim, state_dim) predicted covariances
        observation: (obs_dim,) observation vector

    Returns:
        mean_updated: (N, state_dim)
        cov_updated: (N, state_dim, state_dim)
    """
    # Batched observation predictions: (N, od)
    y_pred = model.observation_function_batch(means)

    # Batched Jacobians: (N, od, sd)
    H_batch = model.observation_jacobian_batch(means)

    # Observation noise covariance — typically constant: (od, od)
    R = model.observation_cov(means[0])

    # Innovation: (N, od)
    innovation = tf.expand_dims(observation, 0) - y_pred

    # S = H @ cov @ H^T + R: (N, od, od)
    H_cov = tf.matmul(H_batch, covs)  # (N, od, sd)
    H_T = tf.linalg.matrix_transpose(H_batch)  # (N, sd, od)
    S = tf.matmul(H_cov, H_T) + tf.expand_dims(R, 0)

    # Kalman gain: K = cov @ H^T @ S^{-1}: (N, sd, od)
    S_inv = tf.linalg.inv(S)  # (N, od, od)
    cov_HT = tf.matmul(covs, H_T)  # (N, sd, od)
    K = tf.matmul(cov_HT, S_inv)  # (N, sd, od)

    # Update mean: mean + K @ innovation: (N, sd)
    mean_updated = means + tf.einsum('nij,nj->ni', K, innovation)

    # Update covariance: (I - K @ H) @ cov: (N, sd, sd)
    I = tf.eye(model.state_dim, dtype=tf.float32)
    KH = tf.matmul(K, H_batch)  # (N, sd, sd)
    I_KH = tf.expand_dims(I, 0) - KH
    cov_updated = tf.matmul(I_KH, covs)
    cov_updated = symmetrize(cov_updated)

    return mean_updated, cov_updated
