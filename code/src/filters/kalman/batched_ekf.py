"""
Batched EKF operations for per-particle filters.

Provides single tf.function calls that process N (mean, cov) pairs at once,
avoiding retracing when used in loops (e.g., LEDH invertible with N particles).
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
    Batched EKF prediction for N particles.

    Args:
        model: StateSpaceModel with state_transition_mean, state_jacobian, state_transition_cov
        means: (N, state_dim)
        covs: (N, state_dim, state_dim)

    Returns:
        mean_pred: (N, state_dim)
        cov_pred: (N, state_dim, state_dim)
    """
    def predict_single(i):
        mean_i = means[i]
        cov_i = covs[i]
        mean_pred_i = model.state_transition_mean(mean_i)
        F = model.state_jacobian(mean_i)
        Q = model.state_transition_cov(mean_i)
        cov_pred_i = F @ cov_i @ tf.transpose(F) + Q
        cov_pred_i = symmetrize(cov_pred_i)
        return mean_pred_i, cov_pred_i

    state_dim = model.state_dim
    n = tf.shape(means)[0]

    def body(i):
        return predict_single(i)

    results = tf.map_fn(
        body,
        tf.range(n),
        fn_output_signature=(
            tf.TensorSpec([state_dim], tf.float32),
            tf.TensorSpec([state_dim, state_dim], tf.float32),
        ),
    )
    mean_pred = results[0]
    cov_pred = results[1]
    return mean_pred, cov_pred


@tf.function
def batched_ekf_update(
    model,
    means: tf.Tensor,
    covs: tf.Tensor,
    observation: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched EKF update for N particles.

    Args:
        model: StateSpaceModel with observation_mean, observation_cov, observation_jacobian
        means: (N, state_dim) predicted means
        covs: (N, state_dim, state_dim) predicted covariances
        observation: (obs_dim,) observation vector

    Returns:
        mean_updated: (N, state_dim)
        cov_updated: (N, state_dim, state_dim)
    """
    def update_single(i):
        mean_i = means[i]
        cov_i = covs[i]
        y_pred = model.observation_mean(mean_i)
        R = model.observation_cov(mean_i)
        H = model.observation_jacobian(mean_i)
        innovation = observation - y_pred
        S = H @ cov_i @ tf.transpose(H) + R
        H_norm = tf.reduce_max(tf.abs(H))

        def do_update():
            S_inv = tf.linalg.inv(S)
            K = cov_i @ tf.transpose(H) @ S_inv
            mean_updated = mean_i + tf.linalg.matvec(K, innovation)
            I = tf.eye(model.state_dim, dtype=tf.float32)
            cov_updated = (I - K @ H) @ cov_i
            return mean_updated, symmetrize(cov_updated)

        def no_update():
            return mean_i, cov_i

        mean_updated, cov_updated = tf.cond(
            H_norm > 1e-10,
            do_update,
            no_update,
        )
        return mean_updated, cov_updated

    state_dim = model.state_dim
    n = tf.shape(means)[0]

    results = tf.map_fn(
        lambda i: update_single(i),
        tf.range(n),
        fn_output_signature=(
            tf.TensorSpec([state_dim], tf.float32),
            tf.TensorSpec([state_dim, state_dim], tf.float32),
        ),
    )
    mean_updated = results[0]
    cov_updated = results[1]
    return mean_updated, cov_updated
