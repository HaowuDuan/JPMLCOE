"""Factory function for creating Kalman filter instances."""

from .extended_kalman import ExtendedKalmanFilter
from .unscented_kalman import UnscentedKalmanFilter


def create_kalman_filter(filter_type, model, mean_0, Sigma_0, **kwargs):
    """
    Create a Kalman filter instance based on filter_type string.

    Args:
        filter_type: 'ekf' or 'ukf'
        model: StateSpaceModel instance
        mean_0: Initial mean (numpy array or tf.Tensor)
        Sigma_0: Initial covariance (numpy array or tf.Tensor)
        **kwargs: Additional kwargs (e.g., alpha, beta, kappa for UKF)

    Returns:
        Filter instance with .mean, .cov, .predict(), .update() interface
    """
    if filter_type == 'ekf':
        return ExtendedKalmanFilter(
            model, mean_0=mean_0, Sigma_0=Sigma_0,
            sample_initial_mean=False
        )
    elif filter_type == 'ukf':
        ukf_kwargs = {
            k: kwargs[k] for k in ('alpha', 'beta', 'kappa')
            if k in kwargs
        }
        return UnscentedKalmanFilter(
            model, mean_0=mean_0, Sigma_0=Sigma_0,
            sample_initial_mean=False,
            **ukf_kwargs
        )
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}. Use 'ekf' or 'ukf'.")
