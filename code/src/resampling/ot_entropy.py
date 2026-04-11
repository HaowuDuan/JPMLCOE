"""
Entropy-regularized optimal transport resampling for particle filters.

Implements differentiable particle resampling using the Sinkhorn algorithm
to solve entropy-regularized optimal transport problems.

Backward pass uses implicit differentiation at the Sinkhorn fixed point,
following Eisenberger et al. (2022), "A Unified Framework for Implicit
Sinkhorn Differentiation", CVPR 2022.

Reference:
    Corenflos et al. (2021). Differentiable Particle Filtering via
    Entropy-Regularized Optimal Transport. ICML 2021.
"""

import tensorflow as tf
from typing import Tuple
from .types import ResampleResult


# =============================================================================
# Section 1: Utility Functions
# =============================================================================

@tf.function(jit_compile=True)
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

    diff = x_sqnorm - 2 * xy + y_sqnorm
    squared_dist = tf.maximum(diff, tf.zeros((), dtype=diff.dtype))
    return squared_dist / 2.0


@tf.function(jit_compile=True)
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


@tf.function(jit_compile=True)
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
    diameter = tf.where(tf.equal(diameter, tf.zeros_like(diameter)), tf.ones_like(diameter), diameter)

    if len(x.shape) == 2:
        diameter = diameter[0]

    return diameter


# =============================================================================
# Section 2: Sinkhorn Algorithm
# =============================================================================

@tf.function(jit_compile=True)
def sinkhorn_iteration(
    log_weights: tf.Tensor,
    cost_matrix: tf.Tensor,
    epsilon: tf.Tensor,
    alpha_init: tf.Tensor,
    beta_init: tf.Tensor,
    max_iter: int,
    threshold: float,
    extrapolation_damping: float = None
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
        extrapolation_damping: If None (default), extrapolation returns raw
            softmin values (serial, no damping). If a float (e.g. 0.5),
            extrapolation returns damping * stopped + (1 - damping) * new.

    Returns:
        Tuple of (alpha, beta, n_iterations):
        - alpha: Dual potential, shape (N,) or (B, N)
        - beta: Dual potential, shape (N,) or (B, N)
        - n_iterations: Number of iterations performed
    """
    dtype = cost_matrix.dtype
    threshold = tf.cast(threshold, dtype)

    N = tf.shape(cost_matrix)[-1]
    float_N = tf.cast(N, dtype)
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

        # Damping factor 0.5 for stability (as in the paper)
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
        loop_vars=[tf.constant(0), alpha, beta, initial_diff],
        maximum_iterations=max_iter,
    )

    # Final extrapolation step for better gradients (implicit function theorem)
    # Stop gradient on converged values, then do one more serial step (Gauss-Seidel).
    # Serial: beta uses alpha_extra, capturing alpha-beta coupling.
    alpha_stop = tf.stop_gradient(alpha_final)
    beta_stop = tf.stop_gradient(beta_final)

    alpha_new = softmin(epsilon, tf.transpose(cost_matrix, [0, 2, 1] if len(cost_matrix.shape) == 3 else [1, 0]),
                        log_weights + beta_stop / epsilon)

    if extrapolation_damping is not None:
        d = extrapolation_damping
        alpha_extra = d * alpha_stop + (1 - d) * alpha_new
    else:
        alpha_extra = alpha_new

    beta_new = softmin(epsilon, cost_matrix, uniform_log_weights + alpha_extra / epsilon)

    if extrapolation_damping is not None:
        beta_extra = d * beta_stop + (1 - d) * beta_new
    else:
        beta_extra = beta_new

    return alpha_extra, beta_extra, n_iter


@tf.function(jit_compile=True)
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
        epsilon_tensor = tf.cast(epsilon, particles.dtype)
    else:
        epsilon_tensor = tf.constant(epsilon, dtype=particles.dtype)

    # Initialize potentials with large epsilon
    N = tf.shape(particles)[-2]
    float_N = tf.cast(N, particles.dtype)
    log_N = tf.math.log(float_N)
    uniform_log = -log_N * tf.ones_like(log_weights)

    alpha_init = softmin(epsilon_init, tf.transpose(cost_matrix, [0, 2, 1] if len(cost_matrix.shape) == 3 else [1, 0]),
                         log_weights)
    beta_init = softmin(epsilon_init, cost_matrix, uniform_log)

    # Epsilon-scaling: gradually reduce epsilon
    # Convert scaling to tensor if needed
    if isinstance(scaling, tf.Tensor):
        scaling_factor = tf.cast(scaling, particles.dtype) ** 2
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

@tf.function(jit_compile=True)
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
    # Cast epsilon to match particle dtype
    epsilon = tf.cast(epsilon, particles.dtype)

    # Compute cost matrix
    # NOTE: filterflow uses cost(particles, tf.stop_gradient(particles)) but
    # this worsened HMC bias (MAP improved, HMC stuck at ~1.78 vs ~1.33)
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
    float_N = tf.cast(N, particles.dtype)
    log_N = tf.math.log(float_N)

    log_T_normalized = log_T - tf.reduce_logsumexp(log_T, axis=-2, keepdims=True) + log_N

    # Apply source weights per-column: T[i,j] ∝ kernel[i,j] * w_j
    # Column j corresponds to source particle j, so weight w_j multiplies column j
    if len(log_weights.shape) == 1:
        log_T_weighted = log_T_normalized + tf.expand_dims(log_weights, 0)
    else:
        log_T_weighted = log_T_normalized + tf.expand_dims(log_weights, 1)

    # Exponentiate to get transport matrix
    T = tf.exp(log_T_weighted)

    return T


def _sinkhorn_implicit_vjp(dT, T, epsilon):
    """Implicit VJP for the Sinkhorn transport matrix.

    Following Eisenberger et al. (2022), "A Unified Framework for Implicit
    Sinkhorn Differentiation", CVPR 2022.

    Given upstream gradient dL/dT, computes dL/dC (cost matrix gradient)
    and dL/da, dL/db (marginal gradients) by solving the KKT system at the
    Sinkhorn fixed point.

    Args:
        dT: Upstream gradient dL/dT, shape (N, N)
        T: Converged transport matrix, shape (N, N)
        epsilon: Regularization parameter (scalar)

    Returns:
        grad_p: Gradient w.r.t. transport plan (= dL/dC after scaling),
                shape (N, N). Chain through cost_matrix -> particles externally.
        grad_a: Gradient w.r.t. row marginal, shape (N,)
        grad_b: Gradient w.r.t. column marginal, shape (N,)
    """
    dtype = T.dtype
    eps = tf.cast(epsilon, dtype)

    # Use actual marginals from P (robust under finite convergence)
    a = tf.reduce_sum(T, axis=1)  # row sums, shape (N,)
    b = tf.reduce_sum(T, axis=0)  # col sums, shape (N,)

    N = tf.shape(T)[0]

    # R = -(dT * T) / epsilon  (Eisenberger eq.)
    R = -dT * T / eps

    # RHS: row sums and column sums of R
    rhs_a = tf.reduce_sum(R, axis=1)        # (N,)
    rhs_b = tf.reduce_sum(R, axis=0)        # (N,)

    # Gauge fix: drop last column-potential variable (matching reference code)
    # Build (N + N-1) x (N + N-1) system:
    # K = [[diag(a),  T[:, :-1]],
    #      [T[:, :-1]^T, diag(b[:-1])]]
    T_free = T[:, :-1]                       # (N, N-1)
    rhs = tf.concat([rhs_a, rhs_b[:-1]], axis=0)  # (2N-1,)

    top = tf.concat([tf.linalg.diag(a), T_free], axis=1)           # (N, 2N-1)
    bottom = tf.concat([tf.transpose(T_free), tf.linalg.diag(b[:-1])], axis=1)  # (N-1, 2N-1)
    K = tf.concat([top, bottom], axis=0)     # (2N-1, 2N-1)

    # Solve K @ [u; v] = rhs
    sol = tf.linalg.solve(K, rhs[:, tf.newaxis])[:, 0]  # (2N-1,)

    u = sol[:N]                              # (N,) — row potential adjoint
    v_free = sol[N:]                         # (N-1,) — column potential adjoint
    v = tf.concat([v_free, tf.zeros([1], dtype=dtype)], axis=0)  # (N,) gauge: v_last=0

    # Gradient w.r.t. transport plan (= gradient w.r.t. cost after -1/eps scaling)
    U = u[:, tf.newaxis] + v[tf.newaxis, :]  # (N, N)
    grad_p = R - T * U                       # (N, N)

    # Gradients w.r.t. marginals
    grad_a = -eps * u                        # (N,)
    grad_b = -eps * v                        # (N,)

    return grad_p, grad_a, grad_b


def ot_entropy_resample(
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float,
    seed: tf.Tensor,
    max_iter: int = 100,
    convergence_threshold: float = 1e-3,
    epsilon_scaling: float = 0.9,
    use_epsilon_scaling: bool = False,
    clip_gradients: bool = False,
    extrapolation_damping: float = None,
    zero_particle_gradient: bool = False,
) -> ResampleResult:
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
                        Only used when use_epsilon_scaling=True.
        use_epsilon_scaling: If True, use epsilon annealing from diameter^2 down
                            to target epsilon. If False (default), run Sinkhorn
                            directly at the target epsilon with zero-initialized
                            potentials. The default follows Corenflos et al. (2021)
                            which recommends data normalization + fixed epsilon.
        clip_gradients: If True, clip incoming gradients element-wise to [-1, 1]
                        in the backward pass. May help SGD-style optimizers but
                        destroys gradient direction needed by HMC. Default: False.
        extrapolation_damping: Damping factor for the extrapolation step.
                               If None (default), no damping — extrapolation returns
                               raw softmin values. If a float (e.g. 0.5), returns
                               damping * stopped + (1 - damping) * new.

    Returns:
        ResampleResult with transported particles, uniform weights, and transport matrix T

    Reference:
        Corenflos et al. (2021). Differentiable Particle Filtering via
        Entropy-Regularized Optimal Transport. ICML 2021.
    """
    # Convert weights to log-space for numerical stability
    log_weights = tf.math.log(weights + tf.constant(1e-10, dtype=weights.dtype))

    # Compute transport matrix with custom gradients.
    # Two paths depending on use_epsilon_scaling (Python-level branch,
    # each traces a different graph at construction time).
    if use_epsilon_scaling:
        @tf.custom_gradient
        def _compute_transport_matrix(particles, log_weights, epsilon, scaling,
                                      max_iter, threshold):
            mean = tf.reduce_mean(particles, axis=0, keepdims=True)
            # Original: stop_gradient on mean and scale_factor (biases gradient)
            # centered = particles - tf.stop_gradient(mean)
            centered = particles - mean
            std = tf.math.reduce_std(particles)
            dimension = tf.cast(tf.shape(particles)[-1], particles.dtype)
            # scale_factor = tf.stop_gradient(std * tf.sqrt(dimension) + tf.constant(1e-8, dtype=particles.dtype))
            scale_factor = std * tf.sqrt(dimension) + tf.constant(1e-8, dtype=particles.dtype)
            scaled = centered / scale_factor

            cost_matrix = compute_cost_matrix(scaled, scaled)
            alpha, beta = sinkhorn_with_epsilon_scaling(
                log_weights, scaled, epsilon, scaling, max_iter, threshold
            )
            T = compute_transport_matrix_from_potentials(
                scaled, alpha, beta, epsilon, log_weights
            )

            def gradient(dT):
                if clip_gradients:
                    dT = tf.clip_by_value(dT, tf.constant(-1.0, dtype=dT.dtype), tf.constant(1.0, dtype=dT.dtype))
                N_float = tf.cast(tf.shape(T)[0], T.dtype)
                P = T / N_float
                dP = dT * N_float
                grad_cost, _, grad_b = _sinkhorn_implicit_vjp(dP, P, epsilon)
                dparticles = tf.gradients(cost_matrix, particles, grad_cost)[0]
                dlog_weights = grad_b * tf.exp(log_weights)
                if zero_particle_gradient:
                    dparticles = tf.zeros_like(particles)
                return dparticles, dlog_weights, None, None, None, None

            return T, gradient

        T = _compute_transport_matrix(
            particles,
            log_weights,
            epsilon,
            epsilon_scaling,
            max_iter,
            convergence_threshold
        )
    else:
        @tf.custom_gradient
        def _compute_transport_matrix_fixed(particles, log_weights, epsilon,
                                            max_iter, threshold):
            mean = tf.reduce_mean(particles, axis=0, keepdims=True)
            # Original: stop_gradient on mean and scale_factor (biases gradient)
            # centered = particles - tf.stop_gradient(mean)
            centered = particles - mean
            std = tf.math.reduce_std(particles)
            dimension = tf.cast(tf.shape(particles)[-1], particles.dtype)
            # scale_factor = tf.stop_gradient(std * tf.sqrt(dimension) + tf.constant(1e-8, dtype=particles.dtype))
            scale_factor = std * tf.sqrt(dimension) + tf.constant(1e-8, dtype=particles.dtype)
            scaled = centered / scale_factor

            # Direct Sinkhorn at fixed epsilon — no annealing
            # NOTE: filterflow uses cost(scaled, tf.stop_gradient(scaled)) but
            # this worsened HMC bias (MAP improved, HMC stuck at ~1.78 vs ~1.33)
            cost_matrix = compute_cost_matrix(scaled, scaled)
            epsilon_tensor = tf.cast(epsilon, scaled.dtype)
            alpha_init = tf.zeros_like(log_weights)
            beta_init = tf.zeros_like(log_weights)
            alpha, beta, _ = sinkhorn_iteration(
                log_weights, cost_matrix, epsilon_tensor,
                alpha_init, beta_init,
                max_iter=max_iter, threshold=threshold,
                extrapolation_damping=extrapolation_damping
            )
            T = compute_transport_matrix_from_potentials(
                scaled, alpha, beta, epsilon, log_weights
            )

            def gradient(dT):
                if clip_gradients:
                    dT = tf.clip_by_value(dT, tf.constant(-1.0, dtype=dT.dtype), tf.constant(1.0, dtype=dT.dtype))
                N_float = tf.cast(tf.shape(T)[0], T.dtype)
                P = T / N_float
                dP = dT * N_float
                grad_cost, _, grad_b = _sinkhorn_implicit_vjp(dP, P, epsilon)
                dparticles = tf.gradients(cost_matrix, particles, grad_cost)[0]
                dlog_weights = grad_b * tf.exp(log_weights)
                if zero_particle_gradient:
                    dparticles = tf.zeros_like(particles)
                return dparticles, dlog_weights, None, None, None

            return T, gradient

        T = _compute_transport_matrix_fixed(
            particles,
            log_weights,
            epsilon,
            max_iter,
            convergence_threshold
        )

    # Apply transport: resampled = T @ particles
    # T has shape (N, N), particles has shape (N, D)
    # Result: (N, D)
    resampled_particles = T @ particles

    # Uniform weights
    N = tf.shape(particles)[0]
    N_float = tf.cast(N, particles.dtype)
    uniform_weights = tf.ones(N, dtype=particles.dtype) / N_float

    return ResampleResult(
        particles=resampled_particles,
        weights=uniform_weights,
        ancestor_indices=None,
        transport_matrix=T,
    )
