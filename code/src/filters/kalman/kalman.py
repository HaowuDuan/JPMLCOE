"""Kalman Filter for linear-Gaussian systems."""

import tensorflow as tf
import numpy as np
import time
from typing import Tuple, Union, List, Optional, Callable
from ...core.filter_base import Filter
from ...core.types import FilterResult
from ...utils.linalg import symmetrize, safe_inv


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
        use_joseph_form: bool = True,
        dtype=tf.float32
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
        self.dtype = dtype
        self.np_dtype = np.float64 if dtype == tf.float64 else np.float32

        # Convert to numpy arrays first (handles both arrays and Hydra ListConfig)
        F_np = np.array(F, dtype=self.np_dtype)
        B_np = np.array(B, dtype=self.np_dtype)
        H_np = np.array(H, dtype=self.np_dtype)
        D_np = np.array(D, dtype=self.np_dtype)

        # Store dimensions
        self.nx = int(F_np.shape[0])
        self.nv = int(B_np.shape[1])
        self.ny = int(H_np.shape[0])
        self.nw = int(D_np.shape[1])

        # Convert to TensorFlow constants
        self.F = tf.constant(F_np, dtype=self.dtype)
        self.B = tf.constant(B_np, dtype=self.dtype)
        self.H = tf.constant(H_np, dtype=self.dtype)
        self.D = tf.constant(D_np, dtype=self.dtype)

        # Compute covariances
        self.Q = tf.linalg.matmul(self.B, self.B, transpose_b=True)
        self.R = tf.linalg.matmul(self.D, self.D, transpose_b=True)

        # Store initial state
        if mean_0 is None:
            mean_0_np = np.zeros(self.nx, dtype=self.np_dtype)
        else:
            mean_0_np = np.array(mean_0, dtype=self.np_dtype)

        if Sigma_0 is None:
            Sigma_0_np = np.eye(self.nx, dtype=self.np_dtype)
        else:
            Sigma_0_np = np.array(Sigma_0, dtype=self.np_dtype)

        self.mean_0 = tf.constant(mean_0_np, dtype=self.dtype)
        self.Sigma_0 = tf.constant(Sigma_0_np, dtype=self.dtype)

        self.use_joseph_form = use_joseph_form

        # Current state (will be tf.Variable for mutable state)
        self.mean = None
        self.cov = None

        # History tracking (lists of numpy arrays for compatibility)
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
        self.mean = tf.Variable(self.mean_0, dtype=self.dtype)
        self.cov = tf.Variable(self.Sigma_0, dtype=self.dtype)

        # Clear history
        for key in self.history:
            self.history[key] = []

    @tf.function
    def _predict_step(self, mean: tf.Tensor, cov: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Prediction step: propagate state estimate forward.

        Args:
            mean: Current mean estimate (nx,)
            cov: Current covariance estimate (nx, nx)

        Returns:
            Tuple of (predicted_mean, predicted_cov)
        """
        mean_pred = tf.linalg.matvec(self.F, mean)
        cov_pred = self.F @ cov @ tf.transpose(self.F) + self.Q
        cov_pred = symmetrize(cov_pred)

        return mean_pred, cov_pred

    @tf.function
    def _update_step(
        self,
        mean: tf.Tensor,
        cov: tf.Tensor,
        observation: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Update step: incorporate observation.

        Args:
            mean: Predicted mean (nx,)
            cov: Predicted covariance (nx, nx)
            observation: Observation vector (ny,)

        Returns:
            Tuple of (updated_mean, updated_cov, kalman_gain, innovation, log_likelihood)
        """
        # Innovation
        innovation = observation - tf.linalg.matvec(self.H, mean)
        innovation_cov = self.H @ cov @ tf.transpose(self.H) + self.R

        # Kalman gain
        kalman_gain = cov @ tf.transpose(self.H) @ safe_inv(innovation_cov)

        # Update mean
        mean_updated = mean + tf.linalg.matvec(kalman_gain, innovation)

        # Update covariance
        I = tf.eye(self.nx, dtype=self.dtype)
        if self.use_joseph_form:
            # Joseph form: P = (I - KH)P(I - KH)^T + KRK^T
            I_minus_KH = I - kalman_gain @ self.H
            cov_updated = (I_minus_KH @ cov @ tf.transpose(I_minus_KH) +
                          kalman_gain @ self.R @ tf.transpose(kalman_gain))
        else:
            # Standard form: P = (I - KH)P
            cov_updated = (I - kalman_gain @ self.H) @ cov

        cov_updated = symmetrize(cov_updated)

        # Compute log-likelihood
        sign, logdet = tf.linalg.slogdet(2.0 * np.pi * innovation_cov)
        innovation_col = tf.reshape(innovation, [-1, 1])
        mahalanobis = tf.reduce_sum(innovation * tf.squeeze(tf.linalg.solve(innovation_cov, innovation_col), axis=-1))
        log_lik = -0.5 * (logdet + mahalanobis)

        return mean_updated, cov_updated, kalman_gain, innovation, log_lik

    def predict(self):
        """
        Prediction step: propagate state estimate forward.

        Returns:
            Tuple of (predicted_mean, predicted_cov)
        """
        mean_pred, cov_pred = self._predict_step(self.mean.value(), self.cov.value())

        self.mean.assign(mean_pred)
        self.cov.assign(cov_pred)

        self.history['mean_pred'].append(mean_pred.numpy().copy())
        self.history['cov_pred'].append(cov_pred.numpy().copy())

        return mean_pred.numpy(), cov_pred.numpy()

    def update(self, observation: np.ndarray):
        """
        Update step: incorporate observation.

        Args:
            observation: Observation vector, shape (ny,)

        Returns:
            Tuple of (updated_mean, updated_cov, kalman_gain)
        """
        obs_tf = tf.constant(observation, dtype=self.dtype)

        mean_updated, cov_updated, kalman_gain, innovation, log_lik = self._update_step(
            self.mean.value(),
            self.cov.value(),
            obs_tf
        )

        self.mean.assign(mean_updated)
        self.cov.assign(cov_updated)

        # Store history (convert to numpy)
        self.history['mean_updated'].append(mean_updated.numpy().copy())
        self.history['cov_updated'].append(cov_updated.numpy().copy())
        self.history['kalman_gain'].append(kalman_gain.numpy().copy())
        self.history['innovations'].append(innovation.numpy().copy())
        self.history['condition_numbers'].append(float(np.linalg.cond(cov_updated.numpy())))
        self.history['log_likelihoods'].append(float(log_lik.numpy()))

        return mean_updated.numpy(), cov_updated.numpy(), kalman_gain.numpy()

    def filter(self, observations: np.ndarray,
               progress_callback: Optional[Callable[[int, int, float], None]] = None) -> FilterResult:
        """
        Run the filter on a sequence of observations.

        Args:
            observations: Array of shape (T, ny) containing observations
            progress_callback: Optional callback(t, T, step_time_sec) called after each step

        Returns:
            FilterResult with filtered means, covariances, and diagnostics
        """
        T = observations.shape[0]

        # Reset to initial state and clear history
        self.reset()

        means = []
        covariances = []

        for t in range(T):
            t0 = time.perf_counter()
            self.predict()
            self.update(observations[t])

            means.append(self.mean.numpy().copy())
            covariances.append(self.cov.numpy().copy())
            if progress_callback is not None:
                progress_callback(t, T, time.perf_counter() - t0)

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
