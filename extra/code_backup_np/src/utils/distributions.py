"""Distribution utilities for stable probability computations."""

import numpy as np
from typing import Union, Optional


def log_gaussian_prob(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute log probability under multivariate Gaussian (numerically stable).

    log p(x) = -0.5 * [n*log(2π) + log|Σ| + (x-μ)^T Σ^(-1) (x-μ)]

    Args:
        x: Data point of shape (n,)
        mean: Mean of shape (n,)
        cov: Covariance of shape (n, n)

    Returns:
        Log probability (scalar)
    """
    diff = x - mean
    n = len(x)

    # Compute log determinant
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite")

    # Compute Mahalanobis distance
    try:
        mahalanobis = diff @ np.linalg.solve(cov, diff)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        mahalanobis = diff @ np.linalg.pinv(cov) @ diff

    return -0.5 * (n * np.log(2 * np.pi) + logdet + mahalanobis)


def log_sum_exp(log_values: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Compute log(sum(exp(log_values))) stably.

    Uses the log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Args:
        log_values: Array of log values
        axis: Axis along which to sum (None for all)

    Returns:
        log(sum(exp(log_values)))
    """
    max_log = np.max(log_values, axis=axis, keepdims=True)
    return np.squeeze(max_log + np.log(np.sum(np.exp(log_values - max_log), axis=axis, keepdims=True)))


def normalize_log_weights(log_weights: np.ndarray, clip_range: Optional[tuple] = None) -> np.ndarray:
    """
    Normalize weights in log-space: w_i / sum(w_j)

    Args:
        log_weights: Log weights of shape (N,) or (batch, N)
        clip_range: Optional (min, max) tuple to clip log_weights - max before exp()
                    e.g., (-30, 30) prevents overflow/underflow

    Returns:
        Normalized weights (not in log space) of same shape
    """
    max_log = np.max(log_weights, axis=-1, keepdims=True)
    log_weights_normalized = log_weights - max_log

    if clip_range is not None:
        log_weights_normalized = np.clip(log_weights_normalized, clip_range[0], clip_range[1])

    weights_unnorm = np.exp(log_weights_normalized)
    return weights_unnorm / np.sum(weights_unnorm, axis=-1, keepdims=True)


def multivariate_normal_sample(mean: np.ndarray, cov: np.ndarray,
                               rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
    """
    Sample from multivariate Gaussian with stable Cholesky decomposition.

    Args:
        mean: Mean of shape (d,)
        cov: Covariance of shape (d, d)
        rng: Random number generator
        n_samples: Number of samples

    Returns:
        Samples of shape (n_samples, d) or (d,) if n_samples=1
    """
    from .linalg import safe_cholesky

    d = len(mean)
    L = safe_cholesky(cov)

    # Sample standard normal
    z = rng.standard_normal((n_samples, d))

    # Transform to N(mean, cov)
    samples = mean + z @ L.T

    return samples if n_samples > 1 else samples[0]


def compute_flow_weights(
    eta_1: np.ndarray,
    eta_0: np.ndarray,
    particles_prev: np.ndarray,
    observation: np.ndarray,
    model,
    prev_weights: Optional[np.ndarray] = None,
    jacobians: Optional[np.ndarray] = None,
    clip_range: tuple = (-30, 30)
) -> np.ndarray:
    """
    Compute particle weights for invertible flow filters with numerical stability.

    Optimized with vectorization when Q is state-independent (default).

    Weight formula: w_i ∝ p(z|η₁ⁱ) * p(η₁ⁱ|x_{k-1}ⁱ) * θ_i / p(η₀ⁱ|x_{k-1}ⁱ) * w_{k-1}ⁱ

    where:
    - η₁ⁱ: Flowed particle at λ=1
    - η₀ⁱ: Sampled particle at λ=0
    - θ_i: Product of Jacobian determinants (optional, for LEDH)
    - w_{k-1}ⁱ: Previous weight (optional, for sequential updates)

    Args:
        eta_1: Flowed particles at λ=1, shape (N, d)
        eta_0: Sampled particles at λ=0, shape (N, d)
        particles_prev: Particles from previous timestep, shape (N, d)
        observation: Current observation, shape (obs_dim,)
        model: StateSpaceModel with batch methods for optimization
        prev_weights: Previous weights, shape (N,). Defaults to uniform if None.
        jacobians: Product of flow Jacobian determinants θ_i, shape (N,). Defaults to ones if None.
        clip_range: Range to clip log weights before exp() to prevent overflow

    Returns:
        Normalized weights, shape (N,)
    """
    n_particles, state_dim = eta_1.shape

    if prev_weights is None:
        prev_weights = np.ones(n_particles) / n_particles

    if jacobians is None:
        jacobians = np.ones(n_particles)

    # Check if Q is state-dependent
    state_dependent_Q = getattr(model, 'has_state_dependent_noise', False)

    if not state_dependent_Q:
        # VECTORIZED PATH (10-100x faster for state-independent Q)

        # 1. Batch transition means: (N, d)
        f_prev = model.state_transition_mean_batch(particles_prev)

        # 2. Single Q matrix: (d, d)
        Q = model.state_transition_cov_batch(particles_prev)

        # 3. Batch observation log-probs: (N,)
        log_p_obs = model.log_observation_prob_batch(observation, eta_1)

        # 4. Vectorized log p(η₁ | x_{k-1}) for all particles
        diff_1 = eta_1 - f_prev  # (N, d)
        try:
            L_Q = np.linalg.cholesky(Q)  # (d, d)
            # Solve L_Q @ y = diff_1.T → y.T = diff_1 @ inv(L_Q.T)
            y_1 = np.linalg.solve(L_Q, diff_1.T).T  # (N, d)
            log_p_eta1 = -0.5 * (
                np.sum(y_1**2, axis=1) +  # (N,)
                2 * np.sum(np.log(np.diag(L_Q))) +
                state_dim * np.log(2 * np.pi)
            )
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            Q_inv = np.linalg.pinv(Q)
            mahalanobis = np.sum(diff_1 @ Q_inv * diff_1, axis=1)  # (N,)
            log_p_eta1 = -0.5 * (
                mahalanobis +
                np.log(np.linalg.det(2 * np.pi * Q) + 1e-10)
            )

        # 5. Vectorized log p(η₀ | x_{k-1}) for all particles
        diff_0 = eta_0 - f_prev  # (N, d)
        try:
            y_0 = np.linalg.solve(L_Q, diff_0.T).T  # (N, d)
            log_p_eta0 = -0.5 * (
                np.sum(y_0**2, axis=1) +
                2 * np.sum(np.log(np.diag(L_Q))) +
                state_dim * np.log(2 * np.pi)
            )
        except:
            mahalanobis = np.sum(diff_0 @ Q_inv * diff_0, axis=1)
            log_p_eta0 = -0.5 * (
                mahalanobis +
                np.log(np.linalg.det(2 * np.pi * Q) + 1e-10)
            )

        # 6. Combine all terms in log space (vectorized)
        log_weights = (
            log_p_eta1 +
            log_p_obs +
            np.log(np.maximum(jacobians, 1e-300)) -
            log_p_eta0 +
            np.log(np.maximum(prev_weights, 1e-300))
        )

    else:
        # LOOP PATH (state-dependent Q) - fallback for models with Q(x)
        log_weights = np.zeros(n_particles)

        for i in range(n_particles):
            # 1. Transition mean and covariance
            f_prev = model.state_transition_mean(particles_prev[i])
            Q = model.state_transition_cov(particles_prev[i])

            # 2. Log p(η₁ⁱ | x_{k-1}ⁱ) - flowed particle transition density
            diff_1 = eta_1[i] - f_prev
            try:
                L_Q = np.linalg.cholesky(Q)
                y_1 = np.linalg.solve(L_Q, diff_1)
                log_p_eta1 = -0.5 * (
                    np.sum(y_1**2) +
                    2 * np.sum(np.log(np.diag(L_Q))) +
                    state_dim * np.log(2 * np.pi)
                )
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                log_p_eta1 = -0.5 * (
                    diff_1.T @ np.linalg.pinv(Q) @ diff_1 +
                    np.log(np.linalg.det(2 * np.pi * Q) + 1e-10)
                )

            # 3. Log p(z_k | η₁ⁱ) - observation likelihood
            log_p_obs = model.log_observation_prob(observation, eta_1[i])

            # 4. Log p(η₀ⁱ | x_{k-1}ⁱ) - sampled particle transition density
            diff_0 = eta_0[i] - f_prev
            try:
                y_0 = np.linalg.solve(L_Q, diff_0)
                log_p_eta0 = -0.5 * (
                    np.sum(y_0**2) +
                    2 * np.sum(np.log(np.diag(L_Q))) +
                    state_dim * np.log(2 * np.pi)
                )
            except:
                log_p_eta0 = -0.5 * (
                    diff_0.T @ np.linalg.pinv(Q) @ diff_0 +
                    np.log(np.linalg.det(2 * np.pi * Q) + 1e-10)
                )

            # 5. Combine all terms in log space
            weight_eps = max(prev_weights[i], 1e-300)
            log_weights[i] = (
                log_p_eta1 +
                log_p_obs +
                np.log(jacobians[i] + 1e-300) -
                log_p_eta0 +
                np.log(weight_eps)
            )

    # Normalize weights with clipping
    weights = normalize_log_weights(log_weights, clip_range=clip_range)

    # Check for weight collapse
    if not np.all(np.isfinite(weights)):
        weights = np.ones(n_particles) / n_particles

    return weights
