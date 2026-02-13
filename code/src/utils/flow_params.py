"""Shared utilities for particle flow filters."""

import tensorflow as tf
from typing import Tuple
from .linalg import safe_solve


@tf.function
def compute_flow_params(
    model,
    linearization_point: tf.Tensor,
    lambda_val: tf.Tensor,
    observation: tf.Tensor,
    P: tf.Tensor,
    R: tf.Tensor,
    R_inv: tf.Tensor,
    eta_bar_0: tf.Tensor,
    state_dim: int,
    regularization: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute A(λ) and b(λ) from Equations (10) and (11).

    Used by: ALL flow filters (EDH and LEDH variants)
    - EDH (global): linearization_point = ensemble mean
    - LEDH (local): linearization_point = individual particle

    Equations (10) and (11) from Li & Coates (2017):
    - A(λ) = -1/2 * P @ H^T @ (λ*H@P@H^T + R)^(-1) @ H
    - b(λ) = (I + 2λA)[(I + λA)P@H^T@R^(-1)@(z - e) + A@η̄_0]
    - e(λ) = h(x) - H@x

    Args:
        model: StateSpaceModel instance with observation_jacobian and observation_function
        linearization_point: Point to linearize at, shape (state_dim,)
            - For EDH: ensemble mean η̄_λ
            - For LEDH: individual particle x_i
        lambda_val: Current pseudo-time λ ∈ [0,1]
        observation: Measurement z, shape (obs_dim,)
        P: Predicted covariance, shape (state_dim, state_dim)
            - For EDH: global covariance
            - For LEDH: GLOBAL covariance (not per-particle)
        R: Observation noise covariance, shape (obs_dim, obs_dim)
        R_inv: Inverse of R (precomputed), shape (obs_dim, obs_dim)
        eta_bar_0: Mean at λ=0, shape (state_dim,)
            - Always GLOBAL mean (even for LEDH)
        state_dim: State dimension
        regularization: Optional regularization for numerical stability (LEDH uses this)

    Returns:
        A: Matrix A(λ), shape (state_dim, state_dim)
        b: Vector b(λ), shape (state_dim,)
    """
    # Linearize at given point: H = ∂h/∂x |_x
    H = model.observation_jacobian(linearization_point)
    # Regularize P before computing S
    if regularization > 0.0:
        trace_P = tf.linalg.trace(P)
        state_dim_f = tf.cast(tf.shape(P)[0], P.dtype)

        # Scale regularization by average variance
        reg_strength = regularization * (trace_P / state_dim_f)
        P_reg = P + reg_strength * tf.eye(state_dim, dtype=P.dtype)
    else:
        P_reg = P



    # Compute A(λ) from Equation (10)
    # A(λ) = -1/2 * P @ H^T @ (λ*H@P@H^T + R)^(-1) @ H
    HPH = H @ P @ tf.transpose(H)
    S = lambda_val * HPH + R



    # Solve S @ S_inv_H = H using safe_solve with cholesky
    S_inv_H = safe_solve(S, H, method='cholesky')

    # Compute A(λ)
    A = -0.5 * P @ tf.transpose(H) @ S_inv_H

    # Compute e(λ) for b(λ) - Equation (11)
    # e(λ) = h(x) - H@x
    h_x = model.observation_function(linearization_point)
    e = h_x - tf.linalg.matvec(H, linearization_point)

    # Compute b(λ) from Equation (11)
    # b(λ) = (I + 2λA)[(I + λA)P@H^T@R^(-1)@(z - e) + A@η̄_0]
    I = tf.eye(state_dim, dtype=tf.float32)

    term1 = tf.linalg.matvec((I + lambda_val * A) @ P @ tf.transpose(H) @ R_inv, observation - e)
    term2 = tf.linalg.matvec(A, eta_bar_0)
    b = tf.linalg.matvec(I + 2 * lambda_val * A, term1 + term2)

    return A, b
