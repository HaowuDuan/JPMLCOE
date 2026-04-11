"""Training losses for ConditionalConvexPotentialMap.

Two losses:
1. MA residual: enforces the Monge-Ampere PDE
   q_h(x) = p_h(T(x)) * det J_T(x)
   In log form: log q_h(x) = log p_h(T(x)) + log det J(x)
   Residual: r(x) = log det J(x) - (log q_h(x) - log p_h(T(x)))

2. Local-linearity penalty: encourages smooth Jacobian on the kernel scale
   L_lin = E[|| T(x + delta) - T(x) - J(x) delta ||^2]
   delta ~ K_h
"""

import tensorflow as tf

from kde import log_kde_uniform, log_kde_weighted


def ma_residual_loss(model, particles, weights, h, query_points, c):
    """Monge-Ampere residual loss.

    Args:
        model: ConditionalConvexPotentialMap
        particles: (N, d) particle positions
        weights: (N,) normalized weights
        h: scalar bandwidth
        query_points: (Q, d) collocation points sampled from q_h
        c: (context_dim,) context vector

    Returns:
        loss (scalar): mean squared MA residual
        diagnostics (dict): logdet, log_q, log_p stats
    """
    T_x, J_x = model.forward_and_jacobian(query_points, c)  # (Q, d), (Q, d, d)

    # log det J for each query point
    sign, logabsdet = tf.linalg.slogdet(J_x)  # (Q,), (Q,)
    log_det_J = logabsdet  # ignore sign — for PSD J it should be positive

    # log q_h(x) and log p_h(T(x))
    log_q = log_kde_uniform(query_points, particles, h)  # (Q,)
    log_p_T = log_kde_weighted(T_x, particles, weights, h)  # (Q,)

    # Residual: log det J - (log q - log p(T))
    residual = log_det_J - (log_q - log_p_T)
    loss = tf.reduce_mean(residual ** 2)

    diagnostics = {
        'log_det_J_mean': tf.reduce_mean(log_det_J),
        'log_q_mean': tf.reduce_mean(log_q),
        'log_p_T_mean': tf.reduce_mean(log_p_T),
        'residual_mean': tf.reduce_mean(residual),
        'residual_abs_mean': tf.reduce_mean(tf.abs(residual)),
    }
    return loss, diagnostics


def eigenvalue_barrier_penalty(model, query_points, c, threshold=0.1):
    """Soft barrier on the smallest eigenvalue of J(x).

    Penalizes points where lambda_min(J(x)) drops below `threshold`,
    using a one-sided quadratic hinge: relu(threshold - lambda_min)^2.

    The Jacobian is symmetric (PSD by construction in ConditionalConvexPotentialMap),
    so tf.linalg.eigvalsh is well-defined and differentiable.

    Args:
        model: ConditionalConvexPotentialMap
        query_points: (Q, d)
        c: context (context_dim,)
        threshold: minimum acceptable lambda_min; penalty kicks in below this

    Returns:
        loss (scalar), min_eig_mean (scalar diagnostic)
    """
    J_x = model.jacobian(query_points, c)  # (Q, d, d)
    eig_vals = tf.linalg.eigvalsh(J_x)  # (Q, d), ascending
    lambda_min = eig_vals[:, 0]  # smallest eigenvalue per query point
    hinge = tf.nn.relu(threshold - lambda_min)
    loss = tf.reduce_mean(hinge ** 2)
    return loss, tf.reduce_mean(lambda_min)


def local_linearity_penalty(model, query_points, c, h, n_perturbations=4, seed=None):
    """Penalty: || T(x + delta) - T(x) - J(x) delta ||^2.

    Encourages T to be locally linear (smooth Jacobian) on the kernel scale.

    Args:
        model: ConditionalConvexPotentialMap
        query_points: (Q, d)
        c: context
        h: bandwidth (used as perturbation scale)
        n_perturbations: number of random delta vectors per query point
        seed: stateless seed

    Returns:
        loss (scalar)
    """
    if seed is None:
        seed = tf.constant([0, 0], dtype=tf.int32)

    Q = tf.shape(query_points)[0]
    d = tf.shape(query_points)[-1]

    T_x, J_x = model.forward_and_jacobian(query_points, c)  # (Q, d), (Q, d, d)

    losses = []
    for k in range(n_perturbations):
        sub_seed = tf.random.experimental.stateless_split(seed, num=2)[0]
        seed = tf.random.experimental.stateless_split(seed, num=2)[1]
        delta = h * tf.random.stateless_normal(
            [Q, d], seed=sub_seed, dtype=query_points.dtype
        )
        T_x_plus_delta = model(query_points + delta, c)  # (Q, d)
        # Linear prediction: T(x) + J(x) @ delta
        J_delta = tf.einsum('qij,qj->qi', J_x, delta)
        residual = T_x_plus_delta - T_x - J_delta
        losses.append(tf.reduce_mean(tf.reduce_sum(residual ** 2, axis=-1)))

    return tf.reduce_mean(tf.stack(losses))
