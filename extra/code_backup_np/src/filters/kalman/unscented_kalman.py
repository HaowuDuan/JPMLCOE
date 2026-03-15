"""Unscented Kalman Filter for nonlinear systems."""

import numpy as np
from typing import Optional
from ...core.filter_base import Filter
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel
from ...utils.linalg import safe_cholesky, symmetrize


class UnscentedKalmanFilter(Filter):
    """
    Unscented Kalman Filter (UKF) for nonlinear state-space models.

    The UKF uses sigma points to capture the mean and covariance through
    nonlinear transformations, avoiding the need for Jacobian computation.
    This works well when:
    - Nonlinearities are moderate
    - Distributions are roughly Gaussian
    - State dimension is moderate (scales as O(n²) with n states)

    The filter can fail when:
    - Distributions are highly non-Gaussian
    - State dimension is very high (computational cost)
    - Sigma points become unstable or invalid

    Uses the standard unscented transform with parameters:
    - α: controls spread of sigma points (default 1e-3)
    - β: incorporates prior knowledge (default 2.0)
    - κ: secondary scaling parameter (default 0.0)
    """

    def __init__(self, model: StateSpaceModel, mean_0: np.ndarray = None,
                 Sigma_0: np.ndarray = None, alpha: float = 1e-3,
                 beta: float = 2.0, kappa: float = 0.0,
                 sample_initial_mean: bool = True, random_seed: Optional[int] = None):
        """
        Initialize the Unscented Kalman Filter.

        Args:
            model: State-space model instance
            mean_0: Initial mean (state_dim,). If None, defaults to zeros.
            Sigma_0: Initial covariance matrix (state_dim, state_dim).
                     If None, uses stationary variance if available, else identity.
            alpha: Spread parameter for sigma points (small positive)
            beta: Incorporates prior knowledge (2 is optimal for Gaussian)
            kappa: Secondary scaling parameter (0 for default)
            sample_initial_mean: If True (default), sample initial mean from model's initial distribution
                                (requires model to have sample_initial_state method).
                                If False, use mean_0 or zeros.
            random_seed: Random seed for sampling initial mean (only used if sample_initial_mean=True)
        """
        self.model = model
        self.state_dim = model.state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute lambda parameter
        self.lambda_ = alpha**2 * (self.state_dim + kappa) - self.state_dim

        # Weights for mean and covariance
        W_m_0 = self.lambda_ / (self.state_dim + self.lambda_)
        W_c_0 = W_m_0 + (1 - alpha**2 + beta)
        W_i = 1.0 / (2 * (self.state_dim + self.lambda_))

        self.weights_mean = np.concatenate([[W_m_0], np.full(2 * self.state_dim, W_i)])
        self.weights_cov = np.concatenate([[W_c_0], np.full(2 * self.state_dim, W_i)])

        # Store initial state
        if sample_initial_mean:
            # Sample initial mean from model's distribution
            if not hasattr(self.model, 'sample_initial_state'):
                raise ValueError("Model must have sample_initial_state method to use sample_initial_mean=True")
            rng = np.random.default_rng(random_seed)
            self.mean_0 = self.model.sample_initial_state(rng)
        elif mean_0 is None:
            # Try to use model's initial_state_mean property
            if hasattr(self.model, 'initial_state_mean'):
                self.mean_0 = self.model.initial_state_mean.copy()
            else:
                self.mean_0 = np.zeros(self.state_dim)
        else:
            self.mean_0 = mean_0.copy()

        if Sigma_0 is None:
            # Try to use model's initial_state_cov property
            if hasattr(self.model, 'initial_state_cov'):
                self.Sigma_0 = self.model.initial_state_cov.copy()
            # Otherwise use stationary covariance if available
            elif hasattr(self.model, 'stationary_var'):
                self.Sigma_0 = np.eye(self.state_dim) * self.model.stationary_var
            else:
                self.Sigma_0 = np.eye(self.state_dim)
        else:
            self.Sigma_0 = Sigma_0.copy()

        # Filter state
        self.mean = None
        self.cov = None

        # Storage
        self.log_likelihoods = []
        
        self.reset()

    def _compute_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Compute sigma points using the unscented transform.

        Args:
            mean: State mean, shape (state_dim,)
            cov: State covariance, shape (state_dim, state_dim)

        Returns:
            Array of shape (2*state_dim + 1, state_dim) containing sigma points
        """
        n = self.state_dim
        lambda_ = self.lambda_

        # Compute matrix square root: sqrt((n + λ)·P)
        # Use safe_cholesky which handles regularization
        sqrt_cov = safe_cholesky((n + lambda_) * cov)

        # Initialize sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean

        # Generate sigma points
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[:, i]
            sigma_points[i + n + 1] = mean - sqrt_cov[:, i]

        return sigma_points

    def reset(self):
        """Reset filter to initial state and clear history."""
        self.mean = self.mean_0.copy()
        self.cov = self.Sigma_0.copy()
        self.log_likelihoods = []

    def predict(self):
        """Prediction step using unscented transform."""
        # Generate sigma points
        sigma_points = self._compute_sigma_points(self.mean, self.cov)

        # Propagate through state transition (deterministic part only)
        # The noise will be added to the covariance separately
        sigma_points_pred = np.array([
            self.model.state_transition_mean(sp) for sp in sigma_points
        ])

        # Predict mean: weighted sum of propagated sigma points
        self.mean = np.sum(self.weights_mean[:, np.newaxis] * sigma_points_pred, axis=0)

        # Predict covariance: weighted sum of outer products
        diff = sigma_points_pred - self.mean
        self.cov = np.sum(
            self.weights_cov[:, np.newaxis, np.newaxis] *
            np.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )

        # Add process noise
        Q = self.model.state_transition_cov(self.mean)
        self.cov = self.cov + Q
        self.cov = symmetrize(self.cov)

    def update(self, observation: np.ndarray):
        """Update step using unscented transform."""
        # Generate sigma points
        sigma_points = self._compute_sigma_points(self.mean, self.cov)

        # Propagate through observation model
        y_sigma = np.array([
            self.model.observation_mean(sp) for sp in sigma_points
        ])

        # Predicted observation mean
        y_pred = np.sum(self.weights_mean[:, np.newaxis] * y_sigma, axis=0)

        # Innovation covariance
        diff_y = y_sigma - y_pred
        S = np.sum(
            self.weights_cov[:, np.newaxis, np.newaxis] *
            np.einsum('ij,ik->ijk', diff_y, diff_y),
            axis=0
        )

        # Add observation noise
        R = self.model.observation_cov(self.mean)
        S = S + R

        # Cross-covariance between state and observation
        diff_x = sigma_points - self.mean
        P_xy = np.sum(
            self.weights_cov[:, np.newaxis, np.newaxis] *
            np.einsum('ij,ik->ijk', diff_x, diff_y),
            axis=0
        )

        # Kalman gain
        K = P_xy @ np.linalg.inv(S)

        # Innovation
        innovation = observation - y_pred

        # Update mean and covariance
        self.mean = self.mean + K @ innovation
        self.cov = self.cov - K @ S @ K.T
        self.cov = symmetrize(self.cov)

        # Log-likelihood
        log_lik = -0.5 * (innovation.T @ np.linalg.solve(S, innovation) +
                          np.linalg.slogdet(2 * np.pi * S)[1])
        self.log_likelihoods.append(log_lik)

    def filter(self, observations: np.ndarray) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Array of shape (T, obs_dim) containing observations

        Returns:
            FilterResult with filtered means, covariances, and diagnostics
        """
        T = observations.shape[0]

        # Reset to initial state and clear history
        self.reset()

        means = []
        covs = []

        for t in range(T):
            self.predict()
            self.update(observations[t])
            means.append(self.mean.copy())
            covs.append(self.cov.copy())

        # Compute total log-likelihood
        total_log_likelihood = sum(self.log_likelihoods)

        return FilterResult(
            means=np.array(means),
            covs=np.array(covs),
            log_likelihood=total_log_likelihood,
            metadata={
                'filter_type': 'UnscentedKalmanFilter',
                'alpha': self.alpha,
                'beta': self.beta,
                'kappa': self.kappa,
                'log_likelihoods': np.array(self.log_likelihoods)
            }
        )
