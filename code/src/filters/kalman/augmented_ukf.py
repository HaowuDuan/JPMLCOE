"""Augmented Unscented Kalman Filter (Algorithm 5.15, Särkkä).

For models where observation noise enters non-additively (e.g. stochastic
volatility: y = b*x_1 + exp(x_2/2)*v), the standard UKF cannot capture
the coupling between state and noise in the observation function.

The augmented UKF augments the state vector with the observation noise:
    x_aug = [x; r],  P_aug = blkdiag(P_x, R)
and generates sigma points in this joint space. Each sigma point is split
into (x, r) parts and propagated through h(x, r), capturing nonlinear
coupling between state and noise.

The model must implement:
  - observation_function_with_noise(x, r) -> y
  - observation_cov(x) -> R  (for building augmented covariance)
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple

from .unscented_kalman import UnscentedKalmanFilter
from ...core.model_base import StateSpaceModel
from ...utils.linalg import safe_cholesky, safe_inv, symmetrize


class AugmentedUnscentedKalmanFilter(UnscentedKalmanFilter):
    """
    Augmented UKF for non-additive observation noise (Alg 5.15).

    Prediction: standard UKF (process noise is additive).
    Update: augments state with observation noise r, generates sigma points
    in the joint (x, r) space, and propagates through h(x, r).
    """

    def __init__(
        self,
        model: StateSpaceModel,
        mean_0: np.ndarray = None,
        Sigma_0: np.ndarray = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        sample_initial_mean: bool = True,
        random_seed: Optional[int] = None,
    ):
        # Initialize parent (sets up prediction sigma-point weights for state_dim)
        super().__init__(
            model=model,
            mean_0=mean_0,
            Sigma_0=Sigma_0,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            sample_initial_mean=sample_initial_mean,
            random_seed=random_seed,
        )

        # Augmented dimension = state_dim + obs_dim
        self.n_aug = self.state_dim + model.obs_dim

        # Recompute lambda and weights for augmented dimension
        self.lambda_aug = alpha**2 * (self.n_aug + kappa) - self.n_aug

        W_m_0 = self.lambda_aug / (self.n_aug + self.lambda_aug)
        W_c_0 = W_m_0 + (1 - alpha**2 + beta)
        W_i = 1.0 / (2 * (self.n_aug + self.lambda_aug))

        weights_mean_np = np.concatenate([[W_m_0], np.full(2 * self.n_aug, W_i)])
        weights_cov_np = np.concatenate([[W_c_0], np.full(2 * self.n_aug, W_i)])

        self.weights_mean_aug = tf.constant(weights_mean_np, dtype=self.dtype)
        self.weights_cov_aug = tf.constant(weights_cov_np, dtype=self.dtype)

    @tf.function(reduce_retracing=True)
    def _update_step(
        self,
        mean: tf.Tensor,
        cov: tf.Tensor,
        observation: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Augmented UKF update (Alg 5.15).

        Augments state with observation noise r, generates sigma points in
        the joint (x, r) space, and propagates through h(x, r).
        No separate R is added to S — it's captured by the augmented sigma points.
        """
        n_x = self.state_dim
        n_r = self.model.obs_dim

        # Observation noise covariance (may be state-dependent)
        R = self.model.observation_cov(mean)

        # Build augmented mean: [x; 0]
        mean_aug = tf.concat([mean, tf.zeros([n_r], dtype=self.dtype)], axis=0)

        # Build augmented covariance: blkdiag(P_x, R)
        zeros_xr = tf.zeros([n_x, n_r], dtype=self.dtype)
        zeros_rx = tf.zeros([n_r, n_x], dtype=self.dtype)
        cov_aug = tf.concat([
            tf.concat([cov, zeros_xr], axis=1),
            tf.concat([zeros_rx, R], axis=1),
        ], axis=0)

        # Generate sigma points in augmented space
        n_aug = self.n_aug
        lambda_aug = self.lambda_aug
        sqrt_cov_aug = safe_cholesky((n_aug + lambda_aug) * cov_aug)

        sigma_aug = tf.TensorArray(dtype=self.dtype, size=2 * n_aug + 1)
        sigma_aug = sigma_aug.write(0, mean_aug)
        for i in tf.range(n_aug):
            sigma_aug = sigma_aug.write(i + 1, mean_aug + sqrt_cov_aug[:, i])
            sigma_aug = sigma_aug.write(i + n_aug + 1, mean_aug - sqrt_cov_aug[:, i])
        sigma_aug_pts = sigma_aug.stack()  # (2*n_aug+1, n_aug)

        # Split into state and noise parts
        sigma_x = sigma_aug_pts[:, :n_x]   # (2*n_aug+1, n_x)
        sigma_r = sigma_aug_pts[:, n_x:]   # (2*n_aug+1, n_r)

        # Propagate through h(x, r)
        def h_xr(inputs):
            x_i = inputs[:n_x]
            r_i = inputs[n_x:]
            return self.model.observation_function_with_noise(x_i, r_i)

        y_sigma = tf.map_fn(h_xr, sigma_aug_pts, dtype=self.dtype)  # (2*n_aug+1, obs_dim)

        # Predicted observation mean
        y_pred = tf.reduce_sum(
            tf.expand_dims(self.weights_mean_aug, -1) * y_sigma,
            axis=0
        )

        # Innovation covariance (no separate R — already in augmented sigma points)
        diff_y = y_sigma - y_pred
        S = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(self.weights_cov_aug, -1), -1) *
            tf.einsum('ij,ik->ijk', diff_y, diff_y),
            axis=0
        )

        # Cross-covariance between state and observation
        diff_x = sigma_x - mean
        P_xy = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(self.weights_cov_aug, -1), -1) *
            tf.einsum('ij,ik->ijk', diff_x, diff_y),
            axis=0
        )

        # Kalman gain
        K = P_xy @ safe_inv(S)

        # Innovation
        innovation = observation - y_pred

        # Update
        mean_updated = mean + tf.linalg.matvec(K, innovation)
        cov_updated = cov - K @ tf.transpose(P_xy) - P_xy @ tf.transpose(K) + K @ S @ tf.transpose(K)
        cov_updated = symmetrize(cov_updated)

        # Log-likelihood
        _, logdet = tf.linalg.slogdet(2.0 * np.pi * S)
        innovation_col = tf.reshape(innovation, [-1, 1])
        mahalanobis = tf.reduce_sum(innovation * tf.squeeze(tf.linalg.solve(S, innovation_col), axis=-1))
        log_lik = -0.5 * (logdet + mahalanobis)

        return mean_updated, cov_updated, log_lik
