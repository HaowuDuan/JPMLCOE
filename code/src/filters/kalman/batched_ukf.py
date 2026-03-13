"""
Batched UKF operations for per-particle filters.

Uses batched sigma point generation and propagation for N (mean, cov) pairs.
Mirrors the API of batched_ekf.py: batched_ukf_predict and batched_ukf_update.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple
from ...utils.linalg import symmetrize, safe_cholesky, safe_inv


def compute_ukf_weights(state_dim, alpha=1e-3, beta=2.0, kappa=0.0, dtype=tf.float64):
    """Pre-compute UKF weights and lambda (called once, reused across timesteps).

    Args:
        state_dim: int, dimension of state
        alpha: spread parameter
        beta: prior knowledge parameter (2 optimal for Gaussian)
        kappa: secondary scaling parameter

    Returns:
        weights_mean: (2*state_dim+1,) tensor
        weights_cov: (2*state_dim+1,) tensor
        lambda_: float scalar
    """
    n = state_dim
    lambda_ = alpha**2 * (n + kappa) - n
    W_m_0 = lambda_ / (n + lambda_)
    W_c_0 = W_m_0 + (1 - alpha**2 + beta)
    W_i = 1.0 / (2 * (n + lambda_))

    weights_mean = tf.constant(
        [W_m_0] + [W_i] * (2 * n), dtype=dtype
    )
    weights_cov = tf.constant(
        [W_c_0] + [W_i] * (2 * n), dtype=dtype
    )
    return weights_mean, weights_cov, lambda_


@tf.function
def _batched_sigma_points(means, covs, lambda_, state_dim):
    """
    Generate sigma points for N (mean, cov) pairs.

    Args:
        means: (N, d)
        covs: (N, d, d)
        lambda_: Python float or scalar tensor
        state_dim: int d

    Returns:
        sigma_points: (N, 2d+1, d)
    """
    scale = tf.cast(state_dim + lambda_, covs.dtype)
    sqrt_cov = safe_cholesky(scale * covs)  # (N, d, d)

    # Center sigma point: the mean itself
    center = means[:, tf.newaxis, :]  # (N, 1, d)

    # Columns of sqrt_cov as offsets: transpose so axis=1 indexes columns
    # sqrt_cov is lower-triangular (N, d, d), its transpose rows = columns
    cols = tf.linalg.matrix_transpose(sqrt_cov)  # (N, d, d) — axis=1 is column index

    positive = center + cols  # (N, d, d) — d sigma points
    negative = center - cols  # (N, d, d)

    # Stack: (N, 1+d+d, d) = (N, 2d+1, d)
    sigma_points = tf.concat([center, positive, negative], axis=1)
    return sigma_points


@tf.function
def batched_ukf_predict(
    model,
    means: tf.Tensor,
    covs: tf.Tensor,
    weights_mean: tf.Tensor,
    weights_cov: tf.Tensor,
    lambda_: float,
    state_dim: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched UKF prediction for N particles.

    Args:
        model: StateSpaceModel with state_transition_mean_batch, state_transition_cov_batch
        means: (N, d)
        covs: (N, d, d)
        weights_mean: (2d+1,) pre-computed UKF mean weights
        weights_cov: (2d+1,) pre-computed UKF covariance weights
        lambda_: scalar (pre-computed)
        state_dim: int

    Returns:
        mean_pred: (N, d)
        cov_pred: (N, d, d)
    """
    N = tf.shape(means)[0]
    n_sigma = 2 * state_dim + 1

    # Generate sigma points: (N, 2d+1, d)
    sigma_pts = _batched_sigma_points(means, covs, lambda_, state_dim)

    # Reshape to (N*(2d+1), d) for a single batch model call
    flat_pts = tf.reshape(sigma_pts, [N * n_sigma, state_dim])

    # Propagate through state transition
    flat_pred = model.state_transition_mean_batch(flat_pts)  # (N*(2d+1), d)

    # Reshape back to (N, 2d+1, d)
    pred_pts = tf.reshape(flat_pred, [N, n_sigma, state_dim])

    # Weighted mean: sum_j w_m[j] * pred_pts[:, j, :]
    mean_pred = tf.einsum('j,njd->nd', weights_mean, pred_pts)  # (N, d)

    # Weighted covariance
    diff = pred_pts - mean_pred[:, tf.newaxis, :]  # (N, 2d+1, d)
    cov_pred = tf.einsum('j,nji,njk->nik', weights_cov, diff, diff)  # (N, d, d)

    # Add process noise Q
    Q = model.state_transition_cov_batch(means)
    Q = tf.cast(Q, covs.dtype)
    if len(Q.shape) == 2:
        cov_pred = cov_pred + tf.expand_dims(Q, 0)
    else:
        cov_pred = cov_pred + Q

    cov_pred = symmetrize(cov_pred)
    return mean_pred, cov_pred


@tf.function
def batched_ukf_update(
    model,
    means: tf.Tensor,
    covs: tf.Tensor,
    observation: tf.Tensor,
    weights_mean: tf.Tensor,
    weights_cov: tf.Tensor,
    lambda_: float,
    state_dim: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched UKF update for N particles.

    Args:
        model: StateSpaceModel with observation_function_batch, observation_cov
        means: (N, d) predicted means
        covs: (N, d, d) predicted covariances
        observation: (obs_dim,)
        weights_mean: (2d+1,)
        weights_cov: (2d+1,)
        lambda_: scalar
        state_dim: int

    Returns:
        mean_updated: (N, d)
        cov_updated: (N, d, d)
    """
    N = tf.shape(means)[0]
    n_sigma = 2 * state_dim + 1

    # Generate sigma points: (N, 2d+1, d)
    sigma_pts = _batched_sigma_points(means, covs, lambda_, state_dim)

    # Reshape to (N*(2d+1), d) for batch model call
    flat_pts = tf.reshape(sigma_pts, [N * n_sigma, state_dim])

    # Propagate through observation model
    flat_obs = model.observation_function_batch(flat_pts)  # (N*(2d+1), obs_dim)
    obs_dim = tf.shape(flat_obs)[1]

    # Reshape back: (N, 2d+1, obs_dim)
    obs_pts = tf.reshape(flat_obs, [N, n_sigma, obs_dim])

    # Predicted observation mean
    y_pred = tf.einsum('j,njm->nm', weights_mean, obs_pts)  # (N, obs_dim)

    # Innovation covariance S = sum_j w_c[j] * (y_j - y_pred)(y_j - y_pred)^T + R
    diff_y = obs_pts - y_pred[:, tf.newaxis, :]  # (N, 2d+1, obs_dim)
    S = tf.einsum('j,nji,njk->nik', weights_cov, diff_y, diff_y)  # (N, obs_dim, obs_dim)
    R = tf.cast(model.observation_cov(means[0]), covs.dtype)
    S = S + tf.expand_dims(R, 0)

    # Cross-covariance P_xy = sum_j w_c[j] * (x_j - x_mean)(y_j - y_pred)^T
    diff_x = sigma_pts - means[:, tf.newaxis, :]  # (N, 2d+1, d)
    P_xy = tf.einsum('j,nji,njk->nik', weights_cov, diff_x, diff_y)  # (N, d, obs_dim)

    # Kalman gain K = P_xy @ S^{-1}
    # Solve via Cholesky for numerical stability: S @ K^T = P_xy^T
    L_S = safe_cholesky(S)  # (N, obs_dim, obs_dim)
    # cholesky_solve expects (N, obs_dim, ?) on RHS
    P_xy_T = tf.linalg.matrix_transpose(P_xy)  # (N, obs_dim, d)
    K_T = tf.linalg.cholesky_solve(L_S, P_xy_T)  # (N, obs_dim, d)
    K = tf.linalg.matrix_transpose(K_T)  # (N, d, obs_dim)

    # Innovation
    innovation = tf.expand_dims(observation, 0) - y_pred  # (N, obs_dim)

    # Update mean
    mean_updated = means + tf.einsum('nij,nj->ni', K, innovation)  # (N, d)

    # Update covariance: P - K @ S @ K^T
    KS = tf.matmul(K, S)           # (N, d, obs_dim)
    K_T2 = tf.linalg.matrix_transpose(K)  # (N, obs_dim, d)
    cov_updated = covs - tf.matmul(KS, K_T2)
    cov_updated = symmetrize(cov_updated)

    return mean_updated, cov_updated
