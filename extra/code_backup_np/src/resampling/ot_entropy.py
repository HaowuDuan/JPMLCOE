"""
Entropy-regularized optimal transport resampling for particle filters.

Implements differentiable particle resampling using the Sinkhorn algorithm
to solve entropy-regularized optimal transport problems.

Reference:
    Corenflos et al. (2021). Differentiable Particle Filtering via
    Entropy-Regularized Optimal Transport. ICML 2021.
"""

import tensorflow as tf
from typing import Tuple


# =============================================================================
# Section 1: Utility Functions
# =============================================================================

@tf.function
def compute_cost_matrix(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Compute squared Euclidean distance cost matrix.

    Cost matrix C[i,j] = ||x_i - y_j||^2 / 2

    Args:
        x: Tensor of shape (N, D) or (B, N, D)
        y: Tensor of shape (M, D) or (B, M, D)

    Returns:
        Cost matrix of shape (N, M) or (B, N, M)
    """
    # Efficient vectorized computation: ||x-y||^2 = x^2 - 2xy + y^2
    x_sqnorm = tf.reduce_sum(x * x, axis=-1, keepdims=True)
    y_sqnorm = tf.expand_dims(tf.reduce_sum(y * y, axis=-1), -2)
    xy = tf.matmul(x, y, transpose_b=True)

    squared_dist = tf.clip_by_value(x_sqnorm - 2 * xy + y_sqnorm, 0.0, float('inf'))
    return squared_dist / 2.0


@tf.function
def softmin(epsilon: tf.Tensor, cost_matrix: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
    """
    Softmin operation for Sinkhorn algorithm.

    Computes: -epsilon * log(sum_j exp((f_j - C_ij) / epsilon))

    This is a smooth approximation of min_j(C_ij - f_j) when epsilon -> 0.
    Uses log-sum-exp trick for numerical stability.

    Args:
        epsilon: Regularization parameter, shape () or (B,)
        cost_matrix: Cost matrix C, shape (N, M) or (B, N, M)
        f: Potential vector, shape (N,) or (B, N)

    Returns:
        Softmin result, shape (M,) or (B, M)
    """
    # Track if input was 1D
    is_1d = len(f.shape) == 1
    
    # Reshape for broadcasting
    if is_1d:
        f = tf.reshape(f, (1, -1))
        cost_matrix = tf.expand_dims(cost_matrix, 0)
        epsilon = tf.reshape(epsilon, (1, 1, 1))
    else:
        batch_size = cost_matrix.shape[0]
        f = tf.reshape(f, (batch_size, 1, -1))
        epsilon = tf.reshape(epsilon, (-1, 1, 1))

    # Compute (f - C) / epsilon with numerical stability
    temp_val = f - cost_matrix / epsilon

    # Log-sum-exp trick
    log_sum_exp = tf.reduce_logsumexp(temp_val, axis=-1)

    # Result: -epsilon * log(sum(exp(...)))
    result = -tf.squeeze(epsilon, axis=[1, 2]) * log_sum_exp

    # If input was 1D, squeeze back to 1D
    if is_1d:
        result = tf.squeeze(result, axis=0)
    
    return result


@tf.function
def compute_diameter(x: tf.Tensor) -> tf.Tensor:
    """
    Compute diameter of particle cloud for epsilon-scaling initialization.

    Diameter is defined as the maximum standard deviation across dimensions.

    Args:
        x: Particles, shape (N, D) or (B, N, D)

    Returns:
        Diameter, shape () or (B,)
    """
    if len(x.shape) == 2:
        x = tf.expand_dims(x, 0)

    std_per_dim = tf.math.reduce_std(x, axis=1)  # (B, D)
    diameter = tf.reduce_max(std_per_dim, axis=-1)  # (B,)

    # Avoid zero diameter
    diameter = tf.where(diameter == 0.0, 1.0, diameter)

    if len(x.shape) == 2:
        diameter = diameter[0]

    return diameter


# =============================================================================
# Section 2: Sinkhorn Algorithm
# =============================================================================

@tf.function
def sinkhorn_iteration(
    log_weights: tf.Tensor,
    cost_matrix: tf.Tensor,
    epsilon: tf.Tensor,
    alpha_init: tf.Tensor,
    beta_init: tf.Tensor,
    max_iter: int,
    threshold: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Sinkhorn fixed-point iterations to compute dual potentials.

    Solves the entropy-regularized optimal transport problem:
        min <C, T> + epsilon * KL(T | uniform)
        s.t. T @ 1 = weights (target distribution)
             T^T @ 1 = 1/N (uniform source)

    Args:
        log_weights: Target log-weights, shape (N,) or (B, N)
        cost_matrix: Cost matrix C, shape (N, N) or (B, N, N)
        epsilon: Regularization parameter
        alpha_init: Initial alpha potential
        beta_init: Initial beta potential
        max_iter: Maximum iterations
        threshold: Convergence threshold

    Returns:
        Tuple of (alpha, beta, n_iterations):
        - alpha: Dual potential, shape (N,) or (B, N)
        - beta: Dual potential, shape (N,) or (B, N)
        - n_iterations: Number of iterations performed
    """
    N = tf.shape(cost_matrix)[-1]
    float_N = tf.cast(N, tf.float32)
    log_N = tf.math.log(float_N)

    # Uniform target for resampling
    uniform_log_weights = -log_N * tf.ones_like(log_weights)

    # Initialize potentials
    alpha = alpha_init
    beta = beta_init

    def iteration_step(i, alpha_curr, beta_curr, _):
        """Single Sinkhorn iteration with damping."""
        # Update alpha: softmin over columns
        alpha_new = softmin(epsilon, tf.transpose(cost_matrix, [0, 2, 1] if len(cost_matrix.shape) == 3 else [1, 0]),
                           log_weights + beta_curr / epsilon)

        # Update beta: softmin over rows
        beta_new = softmin(epsilon, cost_matrix, uniform_log_weights + alpha_new / epsilon)

        # Damping factor 0.5 for stability (as in filterflow)
        alpha_damped = 0.5 * (alpha_curr + alpha_new)
        beta_damped = 0.5 * (beta_curr + beta_new)

        # Compute convergence metric
        alpha_diff = tf.reduce_max(tf.abs(alpha_damped - alpha_curr))
        beta_diff = tf.reduce_max(tf.abs(beta_damped - beta_curr))
        max_diff = tf.maximum(alpha_diff, beta_diff)

        return i + 1, alpha_damped, beta_damped, max_diff

    def convergence_check(i, _, __, max_diff):
        """Check if converged or max iterations reached."""
        not_converged = max_diff > threshold
        not_max_iter = i < max_iter - 1
        return tf.logical_and(not_converged, not_max_iter)

    # Run iterations
    initial_diff = 2.0 * threshold
    n_iter, alpha_final, beta_final, _ = tf.while_loop(
        convergence_check,
        iteration_step,
        loop_vars=[tf.constant(0), alpha, beta, initial_diff]
    )

    # Final extrapolation step for better gradients (implicit function theorem)
    # Stop gradient on converged values, then do one more iteration
    alpha_stop = tf.stop_gradient(alpha_final)
    beta_stop = tf.stop_gradient(beta_final)

    alpha_extra = softmin(epsilon, tf.transpose(cost_matrix, [0, 2, 1] if len(cost_matrix.shape) == 3 else [1, 0]),
                          log_weights + beta_stop / epsilon)
    beta_extra = softmin(epsilon, cost_matrix, uniform_log_weights + alpha_extra / epsilon)

    return alpha_extra, beta_extra, n_iter


@tf.function
def sinkhorn_with_epsilon_scaling(
    log_weights: tf.Tensor,
    particles: tf.Tensor,
    epsilon: float,
    scaling: float,
    max_iter: int,
    threshold: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Sinkhorn algorithm with epsilon-scaling for better convergence.

    Starts with large epsilon (based on particle diameter), gradually
    reduces to target epsilon. This improves numerical stability and
    convergence speed.

    Args:
        log_weights: Log-weights, shape (N,) or (B, N)
        particles: Particle positions, shape (N, D) or (B, N, D)
        epsilon: Target regularization parameter
        scaling: Scaling factor (epsilon reduced by scaling^2 each iteration)
        max_iter: Maximum Sinkhorn iterations
        threshold: Convergence threshold

    Returns:
        Tuple of (alpha, beta) dual potentials
    """
    # Compute cost matrix
    cost_matrix = compute_cost_matrix(particles, particles)

    # Initialize epsilon based on particle diameter
    diameter = compute_diameter(particles)
    epsilon_init = diameter ** 2
    
    # Convert epsilon to tensor if needed (handle both scalar and tensor inputs)
    if isinstance(epsilon, tf.Tensor):
        epsilon_tensor = tf.cast(epsilon, tf.float32)
    else:
        epsilon_tensor = tf.constant(epsilon, dtype=tf.float32)

    # Initialize potentials with large epsilon
    N = tf.shape(particles)[-2]
    float_N = tf.cast(N, tf.float32)
    log_N = tf.math.log(float_N)
    uniform_log = -log_N * tf.ones_like(log_weights)

    alpha_init = softmin(epsilon_init, tf.transpose(cost_matrix, [0, 2, 1] if len(cost_matrix.shape) == 3 else [1, 0]),
                         log_weights)
    beta_init = softmin(epsilon_init, cost_matrix, uniform_log)

    # Epsilon-scaling: gradually reduce epsilon
    # Convert scaling to tensor if needed
    if isinstance(scaling, tf.Tensor):
        scaling_factor = tf.cast(scaling, tf.float32) ** 2
    else:
        scaling_factor = float(scaling) ** 2

    def scaling_step(eps_curr, alpha_curr, beta_curr):
        """One epsilon-scaling step."""
        # Reduce epsilon
        eps_new = tf.maximum(eps_curr * scaling_factor, epsilon_tensor)

        # Run Sinkhorn iterations at this epsilon
        alpha_new, beta_new, _ = sinkhorn_iteration(
            log_weights, cost_matrix, eps_new,
            alpha_curr, beta_curr,
            max_iter=max_iter // 3,  # Fewer iterations per scaling step
            threshold=threshold
        )

        return eps_new, alpha_new, beta_new

    def should_continue(eps_curr, _, __):
        """Continue while epsilon > target."""
        return eps_curr > epsilon_tensor * 1.01  # Small tolerance

    # Epsilon-scaling loop
    _, alpha_scaled, beta_scaled = tf.while_loop(
        should_continue,
        scaling_step,
        loop_vars=[epsilon_init, alpha_init, beta_init],
        maximum_iterations=10  # Typically converges in <10 scaling steps
    )

    # Final Sinkhorn iterations at target epsilon
    alpha_final, beta_final, _ = sinkhorn_iteration(
        log_weights, cost_matrix, epsilon_tensor,
        alpha_scaled, beta_scaled,
        max_iter=max_iter,
        threshold=threshold
    )

    return alpha_final, beta_final


# =============================================================================
# Section 3: Transport Matrix and Main Resampling Function
# =============================================================================

@tf.function
def compute_transport_matrix_from_potentials(
    particles: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    epsilon: float,
    log_weights: tf.Tensor
) -> tf.Tensor:
    """
    Construct transport matrix from Sinkhorn dual potentials.

    Transport matrix: T[i,j] = exp((alpha_i + beta_j - C_ij) / epsilon) * w_i

    Args:
        particles: Particle positions, shape (N, D) or (B, N, D)
        alpha: First dual potential, shape (N,) or (B, N)
        beta: Second dual potential, shape (N,) or (B, N)
        epsilon: Regularization parameter
        log_weights: Source log-weights, shape (N,) or (B, N)

    Returns:
        Transport matrix T, shape (N, N) or (B, N, N)
    """
    # Compute cost matrix
    cost_matrix = compute_cost_matrix(particles, particles)

    # Expand potentials for broadcasting: alpha + beta^T
    if len(alpha.shape) == 1:
        alpha_expanded = tf.expand_dims(alpha, 1)
        beta_expanded = tf.expand_dims(beta, 0)
    else:
        alpha_expanded = tf.expand_dims(alpha, 2)
        beta_expanded = tf.expand_dims(beta, 1)

    # Compute (alpha + beta - C) / epsilon
    log_T = (alpha_expanded + beta_expanded - cost_matrix) / epsilon

    # Normalize columns to sum to 1/N (for uniform resampling)
    N = tf.shape(particles)[-2]
    float_N = tf.cast(N, tf.float32)
    log_N = tf.math.log(float_N)

    log_T_normalized = log_T - tf.reduce_logsumexp(log_T, axis=-2, keepdims=True) + log_N

    # Apply source weights
    if len(log_weights.shape) == 1:
        log_T_weighted = log_T_normalized + tf.expand_dims(log_weights, 1)
    else:
        log_T_weighted = log_T_normalized + tf.expand_dims(log_weights, 2)

    # Exponentiate to get transport matrix
    T = tf.exp(log_T_weighted)

    return T


@tf.custom_gradient
def compute_transport_matrix_with_gradient(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    epsilon: float,
    scaling: float,
    max_iter: int,
    threshold: float
) -> tf.Tensor:
    """
    Compute transport matrix with custom gradient for stability.

    Forward pass: Standard Sinkhorn algorithm
    Backward pass: Implicit differentiation with gradient clipping

    Args:
        particles: Particle positions, shape (N, D)
        log_weights: Log-weights, shape (N,)
        epsilon: Regularization parameter
        scaling: Epsilon-scaling factor
        max_iter: Maximum Sinkhorn iterations
        threshold: Convergence threshold

    Returns:
        Transport matrix T, shape (N, N)
    """
    # Center and scale particles for numerical stability
    mean = tf.reduce_mean(particles, axis=0, keepdims=True)
    centered = particles - tf.stop_gradient(mean)

    std = tf.math.reduce_std(particles)
    dimension = tf.cast(tf.shape(particles)[-1], tf.float32)
    scale_factor = tf.stop_gradient(std * tf.sqrt(dimension) + 1e-8)
    scaled = centered / scale_factor

    # Compute Sinkhorn potentials
    alpha, beta = sinkhorn_with_epsilon_scaling(
        log_weights, scaled, epsilon, scaling, max_iter, threshold
    )

    # Construct transport matrix
    T = compute_transport_matrix_from_potentials(
        scaled, alpha, beta, epsilon, log_weights
    )

    def gradient(dT):
        """Custom gradient with clipping for stability."""
        # Clip incoming gradients
        dT_clipped = tf.clip_by_value(dT, -1.0, 1.0)

        # Compute gradients using implicit differentiation
        dparticles, dlog_weights = tf.gradients(T, [particles, log_weights], dT_clipped)

        # Return gradients (None for non-differentiable parameters)
        return dparticles, dlog_weights, None, None, None, None

    return T, gradient


@tf.function
def ot_entropy_resample(
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float,
    seed: tf.Tensor,
    max_iter: int = 100,
    convergence_threshold: float = 1e-3,
    epsilon_scaling: float = 0.9
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Entropy-regularized optimal transport resampling.

    Uses Sinkhorn algorithm to compute transport matrix that smoothly
    moves weighted particles to uniform distribution while minimizing
    transport cost + entropy regularization.

    This is a fully differentiable resampling method suitable for
    gradient-based learning through particle filters.

    Args:
        particles: Particle positions, shape (N, state_dim)
        weights: Normalized weights, shape (N,) - must sum to 1
        epsilon: Entropy regularization parameter (required).
                 Smaller = closer to hard resampling, larger = more smoothing
                 Typical values: 0.1 (low bias) to 1.0 (high smoothing)
        seed: Random seed tensor, shape (2,) for API consistency
              (not used in deterministic OT, included for interface compatibility)
        max_iter: Maximum Sinkhorn iterations (default: 100)
        convergence_threshold: Stop when potential change < threshold (default: 1e-3)
        epsilon_scaling: Epsilon reduction factor for epsilon-scaling (default: 0.9)
                        Larger = faster reduction (may be less stable)

    Returns:
        Tuple of (resampled_particles, new_weights):
        - resampled_particles: (N, state_dim) - transported particles
        - new_weights: (N,) - uniform weights (all 1/N)

    Example:
        >>> import tensorflow as tf
        >>> N = 100
        >>> particles = tf.random.normal([N, 2])
        >>> weights = tf.nn.softmax(tf.random.normal([N]))
        >>> seed = tf.constant([0, 0], dtype=tf.int32)
        >>>
        >>> resampled, new_weights = ot_entropy_resample(
        ...     particles, weights, epsilon=0.5, seed=seed
        ... )
        >>>
        >>> # Check uniform weights
        >>> tf.reduce_all(tf.abs(new_weights - 1.0/N) < 1e-6)

    Reference:
        Corenflos et al. (2021). Differentiable Particle Filtering via
        Entropy-Regularized Optimal Transport. ICML 2021.
    """
    # Convert weights to log-space for numerical stability
    log_weights = tf.math.log(weights + 1e-10)

    # Compute transport matrix with custom gradients
    T = compute_transport_matrix_with_gradient(
        particles,
        log_weights,
        epsilon,
        epsilon_scaling,
        max_iter,
        convergence_threshold
    )

    # Apply transport: resampled = T @ particles
    # T has shape (N, N), particles has shape (N, D)
    # Result: (N, D)
    resampled_particles = T @ particles

    # Return uniform weights
    N = tf.shape(particles)[0]
    N_float = tf.cast(N, tf.float32)
    uniform_weights = tf.ones(N, dtype=tf.float32) / N_float

    return resampled_particles, uniform_weights
