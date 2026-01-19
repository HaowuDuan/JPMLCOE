"""Extended Kalman Filter for nonlinear systems."""

import numpy as np
from typing import Optional
from ...core.filter_base import Filter
from ...core.types import FilterResult
from ...core.model_base import StateSpaceModel
from ...utils.linalg import symmetrize


class ExtendedKalmanFilter(Filter):
    """
    Extended Kalman Filter (EKF) for nonlinear state-space models.

    The EKF linearizes the nonlinear functions around the current state estimate
    using first-order Taylor expansion (Jacobian matrices). This works well when
    the nonlinearity is mild, but can fail when:
    - Nonlinearities are strong
    - Jacobians are difficult to compute or inaccurate
    - State estimates are far from the true state

    The filter maintains:
    - Mean: x̂_n|n (filtered state estimate)
    - Covariance: P_n|n (filtered state covariance)
    """

    def __init__(self, model: StateSpaceModel, mean_0: np.ndarray = None, Sigma_0: np.ndarray = None,
                 sample_initial_mean: bool = True, random_seed: Optional[int] = None):
        """
        Initialize the Extended Kalman Filter.

        Args:
            model: State-space model instance
            mean_0: Initial mean (state_dim,). If None, defaults to zeros.
            Sigma_0: Initial covariance matrix (state_dim, state_dim).
                     If None, uses stationary variance if available, else identity.
            sample_initial_mean: If True (default), sample initial mean from model's initial distribution
                                (requires model to have sample_initial_state method).
                                If False, use mean_0 or zeros.
            random_seed: Random seed for sampling initial mean (only used if sample_initial_mean=True)
        """
        self.model = model
        self.state_dim = model.state_dim

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

        # Storage for log-likelihoods
        self.log_likelihoods = []

        self.reset()
        
    def reset(self):
        """Reset filter to initial state and clear history."""
        self.mean = self.mean_0.copy()
        self.cov = self.Sigma_0.copy()
        self.log_likelihoods = []

    def predict(self):
        """
        Prediction step: propagate state estimate forward.

        Uses linearization: x' ≈ f(x̂) + F·(x - x̂), where F is the Jacobian.
        """
        # Store previous mean for Jacobian evaluation
        mean_prev = self.mean.copy()

        # Predict mean: x̂_{n|n-1} = f(x̂_{n-1|n-1})
        self.mean = self.model.state_transition_mean(mean_prev)

        # Predict covariance: P_{n|n-1} = F·P_{n-1|n-1}·F^T + Q
        F = self.model.state_jacobian(mean_prev)
        Q = self.model.state_transition_cov(mean_prev)
        self.cov = F @ self.cov @ F.T + Q
        self.cov = symmetrize(self.cov)

    def update(self, observation: np.ndarray):
        """
        Update step: incorporate observation.

        Uses linearization: y ≈ h(x̂) + H·(x - x̂), where H is the Jacobian.

        Args:
            observation: Observation, shape (obs_dim,)
        """
        # Predicted observation mean
        y_pred = self.model.observation_mean(self.mean)

        # Observation covariance
        R = self.model.observation_cov(self.mean)

        # Observation Jacobian
        H = self.model.observation_jacobian(self.mean)

        # Innovation: ν = y - y_pred
        innovation = observation - y_pred

        # Innovation covariance: S = H·P·H^T + R
        S = H @ self.cov @ H.T + R

        # Handle case where H is zero (e.g., observation mean doesn't depend on state)
        # This occurs in StochasticVolatilityModel where E[y|x] = 0
        if np.allclose(H, 0):
            # No update possible - state estimate remains unchanged
            # Log-likelihood still computed
            pass
        else:
            # Kalman gain: K = P·H^T·S^{-1}
            K = self.cov @ H.T @ np.linalg.inv(S)

            # Update mean: x̂_{n|n} = x̂_{n|n-1} + K·ν
            self.mean = self.mean + K @ innovation

            # Update covariance: P_{n|n} = (I - K·H)·P_{n|n-1}
            I = np.eye(self.state_dim)
            self.cov = (I - K @ H) @ self.cov
            self.cov = symmetrize(self.cov)

        # Log-likelihood: log p(y_n | y_{1:n-1})
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
                'filter_type': 'ExtendedKalmanFilter',
                'log_likelihoods': np.array(self.log_likelihoods)
            }
        )
