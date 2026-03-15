"""Shared utilities for particle flow filters."""

import tensorflow as tf
from typing import Tuple
from .linalg import safe_solve, safe_cholesky


@tf.function(jit_compile=True)
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
    regularization: tf.Tensor = None
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
    if regularization is None:
        regularization = tf.constant(0.0, dtype=P.dtype)
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
    HPH = H @ P_reg @ tf.transpose(H)
    S = lambda_val * HPH + R



    # Solve S @ S_inv_H = H using safe_solve with cholesky
    S_inv_H = safe_solve(S, H, method='cholesky')

    # Compute A(λ)
    A = -0.5 * P_reg @ tf.transpose(H) @ S_inv_H

    # Compute e(λ) for b(λ) - Equation (11)
    # e(λ) = h(x) - H@x
    h_x = model.observation_function(linearization_point)
    e = h_x - tf.linalg.matvec(H, linearization_point)

    # Compute b(λ) from Equation (11)
    # b(λ) = (I + 2λA)[(I + λA)P@H^T@R^(-1)@(z - e) + A@η̄_0]
    I = tf.eye(state_dim, dtype=P.dtype)

    term1 = tf.linalg.matvec((I + lambda_val * A) @ P_reg @ tf.transpose(H) @ R_inv, observation - e)
    term2 = tf.linalg.matvec(A, eta_bar_0)
    b = tf.linalg.matvec(I + 2 * lambda_val * A, term1 + term2)

    return A, b


@tf.function(jit_compile=True)
def compute_flow_params_global(
    H: tf.Tensor,
    lambda_val: tf.Tensor,
    observation: tf.Tensor,
    P: tf.Tensor,
    R: tf.Tensor,
    R_inv: tf.Tensor,
    eta_bar_0: tf.Tensor,
    state_dim: int,
    regularization: tf.Tensor = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute A(λ) and b(λ) for the global (non-relinearizing) EDH flow.

    Unlike compute_flow_params, H is precomputed once at η̄_0 and passed in.
    The b formula uses z directly (no nonlinear correction e):
    - A(λ) = -1/2 * P @ H^T @ (λ*H@P@H^T + R)^(-1) @ H
    - b(λ) = (I + 2λA)[(I + λA)P@H^T@R^(-1)@z + A@η̄_0]

    Args:
        H: Precomputed observation Jacobian at η̄_0, shape (obs_dim, state_dim)
        lambda_val: Current pseudo-time λ ∈ [0,1]
        observation: Measurement z, shape (obs_dim,)
        P: Predicted covariance, shape (state_dim, state_dim)
        R: Observation noise covariance, shape (obs_dim, obs_dim)
        R_inv: Inverse of R (precomputed), shape (obs_dim, obs_dim)
        eta_bar_0: Mean at λ=0, shape (state_dim,)
        state_dim: State dimension
        regularization: Optional regularization for numerical stability

    Returns:
        A: Matrix A(λ), shape (state_dim, state_dim)
        b: Vector b(λ), shape (state_dim,)
    """
    if regularization is None:
        regularization = tf.constant(0.0, dtype=P.dtype)

    # Regularize P before computing S
    if regularization > 0.0:
        trace_P = tf.linalg.trace(P)
        state_dim_f = tf.cast(tf.shape(P)[0], P.dtype)
        reg_strength = regularization * (trace_P / state_dim_f)
        P_reg = P + reg_strength * tf.eye(state_dim, dtype=P.dtype)
    else:
        P_reg = P

    # Compute A(λ) from Equation (10)
    # A(λ) = -1/2 * P @ H^T @ (λ*H@P@H^T + R)^(-1) @ H
    HPH = H @ P_reg @ tf.transpose(H)
    S = lambda_val * HPH + R

    S_inv_H = safe_solve(S, H, method='cholesky')

    A = -0.5 * P_reg @ tf.transpose(H) @ S_inv_H

    # Compute b(λ) — global formula uses z directly (no e correction)
    # b(λ) = (I + 2λA)[(I + λA)P@H^T@R^(-1)@z + A@η̄_0]
    I = tf.eye(state_dim, dtype=P.dtype)

    term1 = tf.linalg.matvec((I + lambda_val * A) @ P_reg @ tf.transpose(H) @ R_inv, observation)
    term2 = tf.linalg.matvec(A, eta_bar_0)
    b = tf.linalg.matvec(I + 2 * lambda_val * A, term1 + term2)

    return A, b


@tf.function(jit_compile=True)
def compute_flow_params_batch(
    model,
    linearization_points: tf.Tensor,
    lambda_val: tf.Tensor,
    observation: tf.Tensor,
    P: tf.Tensor,
    R: tf.Tensor,
    R_inv: tf.Tensor,
    eta_bar_0: tf.Tensor,
    state_dim: int,
    regularization: tf.Tensor = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched computation of A(λ) and b(λ) for N particles simultaneously.

    Same equations as compute_flow_params but vectorized across particles.

    Args:
        model: StateSpaceModel with observation_jacobian_batch and observation_function_batch
        linearization_points: (N, state_dim) — per-particle linearization points
        lambda_val: Current pseudo-time λ (scalar)
        observation: Measurement z, shape (obs_dim,)
        P: Predicted covariance — either (N, sd, sd) per-particle or (sd, sd) global
        R: Observation noise covariance (obs_dim, obs_dim) — shared
        R_inv: Inverse of R (obs_dim, obs_dim) — shared
        eta_bar_0: Mean at λ=0 — either (N, sd) per-particle or (sd,) global
        state_dim: State dimension
        regularization: Regularization strength

    Returns:
        A_batch: (N, state_dim, state_dim)
        b_batch: (N, state_dim)
    """
    if regularization is None:
        regularization = tf.constant(0.0, dtype=P.dtype)
    # Broadcast P to (N, sd, sd) if global (sd, sd)
    if len(P.shape) == 2:
        P_b = tf.expand_dims(P, 0)  # (1, sd, sd) — broadcasts with (N, ...)
    else:
        P_b = P  # already (N, sd, sd)

    # Regularize P
    if regularization > 0.0:
        # For batched P: trace along last two dims
        trace_P = tf.linalg.trace(P_b)  # (N,) or (1,)
        state_dim_f = tf.cast(state_dim, P_b.dtype)
        reg_strength = regularization * (trace_P / state_dim_f)  # (N,) or (1,)
        I_sd = tf.eye(state_dim, dtype=P_b.dtype)
        P_b = P_b + reg_strength[..., tf.newaxis, tf.newaxis] * I_sd

    # Batch Jacobians: (N, od, sd)
    H_batch = model.observation_jacobian_batch(linearization_points)

    # HP = H @ P: (N, od, sd)
    HP = tf.matmul(H_batch, P_b)

    # HPH = H @ P @ H^T: (N, od, od)
    H_T = tf.linalg.matrix_transpose(H_batch)  # (N, sd, od)
    HPH = tf.matmul(HP, H_T)

    # S = λ * HPH + R: (N, od, od)
    S = lambda_val * HPH + tf.expand_dims(R, 0)

    # Cholesky solve: S @ X = H → X = S^{-1} @ H: (N, od, sd)
    L_S = safe_cholesky(S)
    S_inv_H = tf.linalg.cholesky_solve(L_S, H_batch)

    # A = -0.5 * P @ H^T @ S_inv_H: (N, sd, sd)
    PH_T = tf.matmul(P_b, H_T)  # (N, sd, od)
    A_batch = -0.5 * tf.matmul(PH_T, S_inv_H)

    # e = h(x) - H @ x: (N, od)
    h_batch = model.observation_function_batch(linearization_points)  # (N, od)
    Hx = tf.einsum('nij,nj->ni', H_batch, linearization_points)  # (N, od)
    e_batch = h_batch - Hx

    # b = (I + 2λA) @ [(I + λA) @ P @ H^T @ R_inv @ (z - e) + A @ η̄_0]
    I_sd = tf.eye(state_dim, dtype=P_b.dtype)
    I_batch = tf.expand_dims(I_sd, 0)  # (1, sd, sd)

    # (I + λA): (N, sd, sd)
    I_lA = I_batch + lambda_val * A_batch

    # P @ H^T @ R_inv: (N, sd, od)
    PHT_Rinv = tf.matmul(PH_T, tf.expand_dims(R_inv, 0))

    # (z - e): (N, od)
    z_minus_e = tf.expand_dims(observation, 0) - e_batch

    # (I + λA) @ P @ H^T @ R_inv @ (z - e): (N, sd)
    inner = tf.einsum('nij,nj->ni', tf.matmul(I_lA, PHT_Rinv), z_minus_e)

    # A @ η̄_0: (N, sd)
    if len(eta_bar_0.shape) == 1:
        A_eta = tf.einsum('nij,j->ni', A_batch, eta_bar_0)
    else:
        A_eta = tf.einsum('nij,nj->ni', A_batch, eta_bar_0)

    # (I + 2λA) @ (inner + A_eta): (N, sd)
    I_2lA = I_batch + 2 * lambda_val * A_batch
    b_batch = tf.einsum('nij,nj->ni', I_2lA, inner + A_eta)

    return A_batch, b_batch


@tf.function(jit_compile=True)
def compute_flow_params_batch_global(
    H_batch: tf.Tensor,
    lambda_val: tf.Tensor,
    observation: tf.Tensor,
    P: tf.Tensor,
    R: tf.Tensor,
    R_inv: tf.Tensor,
    eta_bar_0: tf.Tensor,
    state_dim: int,
    regularization: tf.Tensor = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched computation of A(λ) and b(λ) for the global (non-relinearizing) flow.

    Unlike compute_flow_params_batch, H_batch is precomputed and passed in.
    The b formula uses z directly (no nonlinear correction e).

    Args:
        H_batch: Precomputed Jacobians, shape (N, obs_dim, state_dim)
        lambda_val: Current pseudo-time λ (scalar)
        observation: Measurement z, shape (obs_dim,)
        P: Predicted covariance — either (N, sd, sd) per-particle or (sd, sd) global
        R: Observation noise covariance (obs_dim, obs_dim) — shared
        R_inv: Inverse of R (obs_dim, obs_dim) — shared
        eta_bar_0: Mean at λ=0 — either (N, sd) per-particle or (sd,) global
        state_dim: State dimension
        regularization: Regularization strength

    Returns:
        A_batch: (N, state_dim, state_dim)
        b_batch: (N, state_dim)
    """
    if regularization is None:
        regularization = tf.constant(0.0, dtype=P.dtype)

    # Broadcast P to (N, sd, sd) if global (sd, sd)
    if len(P.shape) == 2:
        P_b = tf.expand_dims(P, 0)
    else:
        P_b = P

    # Regularize P
    if regularization > 0.0:
        trace_P = tf.linalg.trace(P_b)
        state_dim_f = tf.cast(state_dim, P_b.dtype)
        reg_strength = regularization * (trace_P / state_dim_f)
        I_sd = tf.eye(state_dim, dtype=P_b.dtype)
        P_b = P_b + reg_strength[..., tf.newaxis, tf.newaxis] * I_sd

    # HP = H @ P: (N, od, sd)
    HP = tf.matmul(H_batch, P_b)

    # HPH = H @ P @ H^T: (N, od, od)
    H_T = tf.linalg.matrix_transpose(H_batch)  # (N, sd, od)
    HPH = tf.matmul(HP, H_T)

    # S = λ * HPH + R: (N, od, od)
    S = lambda_val * HPH + tf.expand_dims(R, 0)

    # Cholesky solve: S @ X = H → X = S^{-1} @ H: (N, od, sd)
    L_S = safe_cholesky(S)
    S_inv_H = tf.linalg.cholesky_solve(L_S, H_batch)

    # A = -0.5 * P @ H^T @ S_inv_H: (N, sd, sd)
    PH_T = tf.matmul(P_b, H_T)  # (N, sd, od)
    A_batch = -0.5 * tf.matmul(PH_T, S_inv_H)

    # b = (I + 2λA) @ [(I + λA) @ P @ H^T @ R_inv @ z + A @ η̄_0]
    # Global formula: uses z directly (no e correction)
    I_sd = tf.eye(state_dim, dtype=P_b.dtype)
    I_batch = tf.expand_dims(I_sd, 0)  # (1, sd, sd)

    # (I + λA): (N, sd, sd)
    I_lA = I_batch + lambda_val * A_batch

    # P @ H^T @ R_inv: (N, sd, od)
    PHT_Rinv = tf.matmul(PH_T, tf.expand_dims(R_inv, 0))

    # z broadcast: (N, od) — tile to match batch dimension for einsum
    N = tf.shape(H_batch)[0]
    z_broadcast = tf.tile(tf.expand_dims(observation, 0), [N, 1])

    # (I + λA) @ P @ H^T @ R_inv @ z: (N, sd)
    inner = tf.einsum('nij,nj->ni', tf.matmul(I_lA, PHT_Rinv), z_broadcast)

    # A @ η̄_0: (N, sd)
    if len(eta_bar_0.shape) == 1:
        A_eta = tf.einsum('nij,j->ni', A_batch, eta_bar_0)
    else:
        A_eta = tf.einsum('nij,nj->ni', A_batch, eta_bar_0)

    # (I + 2λA) @ (inner + A_eta): (N, sd)
    I_2lA = I_batch + 2 * lambda_val * A_batch
    b_batch = tf.einsum('nij,nj->ni', I_2lA, inner + A_eta)

    return A_batch, b_batch
