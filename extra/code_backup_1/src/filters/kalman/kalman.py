"""Kalman Filter for linear-Gaussian systems."""

import numpy as np
from typing import Tuple, Union, List, Optional
from ...core.filter_base import Filter
from ...core.types import FilterResult
from ...utils.linalg import symmetrize

# TODO: Kalman filter initial condition is not clear, sample or fix? 
class KalmanFilter(Filter):
    """
    Kalman Filter for linear-Gaussian state-space models.

    Model:
        X_n = F·X_{n-1} + B·V_n,  V_n ~ N(0, I)
        Y_n = H·X_n + D·W_n,  W_n ~ N(0, I)

    where:
        - F: State transition matrix (nx, nx)
        - B: Process noise matrix (nx, nv)
        - H: Observation matrix (ny, nx)
        - D: Observation noise matrix (ny, nw)
        - Q = B·B^T: Process noise covariance
        - R = D·D^T: Observation noise covariance

    Supports both standard and Joseph-form covariance updates for numerical stability.
    """

    def __init__(
        self,
        F: Union[np.ndarray, List],
        B: Union[np.ndarray, List],
        H: Union[np.ndarray, List],
        D: Union[np.ndarray, List],
        mean_0: Optional[Union[np.ndarray, List]] = None,
        Sigma_0: Optional[Union[np.ndarray, List]] = None,
        use_joseph_form: bool = True
    ):
        """
        Initialize the Kalman Filter.

        Args:
            F: State transition matrix (nx, nx)
            B: Process noise matrix (nx, nv)
            H: Observation matrix (ny, nx)
            D: Observation noise matrix (ny, nw)
            mean_0: Initial mean (nx,). If None, defaults to zeros.
            Sigma_0: Initial covariance matrix (nx, nx). If None, defaults to identity.
            use_joseph_form: If True, use Joseph stabilized covariance update
        """
        # Convert to numpy arrays (handles both arrays and Hydra ListConfig)
        F = np.array(F, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        H = np.array(H, dtype=np.float64)
        D = np.array(D, dtype=np.float64)

        self.nx = F.shape[0]
        self.nv = B.shape[1]
        self.ny = H.shape[0]
        self.nw = D.shape[1]

        self.F = F
        self.B = B
        self.H = H
        self.D = D

        # Store initial state (convert to numpy if needed)
        if mean_0 is None:
            self.mean_0 = np.zeros(self.nx)
        else:
            self.mean_0 = np.array(mean_0, dtype=np.float64).copy()

        if Sigma_0 is None:
            self.Sigma_0 = np.eye(self.nx)
        else:
            self.Sigma_0 = np.array(Sigma_0, dtype=np.float64).copy()

        self.Q = B @ B.T
        self.R = D @ D.T

        self.use_joseph_form = use_joseph_form

        # Current state
        self.mean = None
        self.cov = None
        # History tracking
        self.history = {
            'mean_pred': [],
            'cov_pred': [],
            'mean_updated': [],
            'cov_updated': [],
            'kalman_gain': [],
            'innovations': [],
            'condition_numbers': [],
            'log_likelihoods': []
        }
        
        self.reset()

    def reset(self):
        """Reset filter to initial state and clear history."""
        self.mean = self.mean_0.copy()
        self.cov = self.Sigma_0.copy()

        # Clear history
        for key in self.history:
            self.history[key] = []

    def predict(self):
        """
        Prediction step: propagate state estimate forward.

        Returns:
            Tuple of (predicted_mean, predicted_cov)
        """
        self.mean = self.F @ self.mean
        self.cov = self.F @ self.cov @ self.F.T + self.Q
        self.cov = symmetrize(self.cov)  # Ensure symmetry

        self.history['mean_pred'].append(self.mean.copy())
        self.history['cov_pred'].append(self.cov.copy())

        return self.mean, self.cov

    def update(self, observation: np.ndarray):
        """
        Update step: incorporate observation.

        Args:
            observation: Observation vector, shape (ny,)

        Returns:
            Tuple of (updated_mean, updated_cov, kalman_gain)
        """
        # Innovation
        innovation = observation - self.H @ self.mean
        innovation_cov = self.H @ self.cov @ self.H.T + self.R

        # Kalman gain
        # TODO: the inverse of the innovation is not optimzed for stability 
        kalman_gain = self.cov @ self.H.T @ np.linalg.inv(innovation_cov)

        # Update mean
        self.mean = self.mean + kalman_gain @ innovation

        # Update covariance
        if self.use_joseph_form:
            # Joseph form: P = (I - KH)P(I - KH)^T + KRK^T
            I = np.eye(self.nx)
            I_minus_KH = I - kalman_gain @ self.H
            self.cov = (I_minus_KH @ self.cov @ I_minus_KH.T +
                       kalman_gain @ self.R @ kalman_gain.T)
        else:
            # Standard form: P = (I - KH)P
            I = np.eye(self.nx)
            self.cov = (I - kalman_gain @ self.H) @ self.cov

        self.cov = symmetrize(self.cov)  # Ensure symmetry

        # Compute log-likelihood
        log_lik = -0.5 * (innovation @ np.linalg.solve(innovation_cov, innovation) +
                         np.linalg.slogdet(2 * np.pi * innovation_cov)[1])

        # Store history
        # TODO: we can also store the conditioning number of the innovation covariance 
        self.history['mean_updated'].append(self.mean.copy())
        self.history['cov_updated'].append(self.cov.copy())
        self.history['kalman_gain'].append(kalman_gain.copy())
        self.history['innovations'].append(innovation.copy())
        self.history['condition_numbers'].append(np.linalg.cond(self.cov))
        self.history['log_likelihoods'].append(log_lik)

        return self.mean, self.cov, kalman_gain

    def filter(self, observations: np.ndarray) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Array of shape (T, ny) containing observations

        Returns:
            FilterResult with filtered means, covariances, and diagnostics
        """
        T = observations.shape[0]

        # Reset to initial state and clear history
        self.reset()

        means = []
        covariances = []

        for t in range(T):
            self.predict()
            self.update(observations[t])

            means.append(self.mean.copy())
            covariances.append(self.cov.copy())

        # Compute total log-likelihood
        total_log_likelihood = sum(self.history['log_likelihoods'])

        return FilterResult(
            means=np.array(means),
            covs=np.array(covariances),
            log_likelihood=total_log_likelihood,
            metadata={
                'filter_type': 'KalmanFilter',
                'use_joseph_form': self.use_joseph_form,
                'kalman_gains': np.array(self.history['kalman_gain']),
                'innovations': np.array(self.history['innovations']),
                'condition_numbers': np.array(self.history['condition_numbers']),
                'log_likelihoods': np.array(self.history['log_likelihoods'])
            }
        )
