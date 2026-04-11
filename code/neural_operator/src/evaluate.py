"""Evaluation metrics for the trained neural operator.

Computes on held-out particle clouds:
- MA residual: mean squared (log det J - (log q_h - log p_h(T(x))))
- KL(p_h || r_h): MC estimate where r_h is KDE of transported particles
- Moment error: relative |mean| and |cov|_F error between target and transported
- MMD^2 (Gaussian kernel) between samples from p_h and from r_h
- J conditioning: lambda_min statistics, log det J failure rate
"""

import tensorflow as tf
import numpy as np

from data import sample_random_cloud
from kde import (
    silverman_bandwidth_scalar,
    log_kde_weighted,
    log_kde_uniform,
    sample_from_uniform_kde,
    sample_from_weighted_kde,
    weighted_mean,
    weighted_covariance,
)
from losses import ma_residual_loss


def _gaussian_mmd2(x, y, h):
    """Unbiased MMD^2 estimator with isotropic Gaussian kernel of width h.

    k(a, b) = exp(-||a - b||^2 / (2 h^2))

    x: (n, d), y: (m, d). Returns scalar (may be slightly negative due to MC).
    """
    def k(a, b):
        diff = a[:, None, :] - b[None, :, :]
        sq = tf.reduce_sum(diff ** 2, axis=-1)
        return tf.exp(-0.5 * sq / (h ** 2))

    n = tf.cast(tf.shape(x)[0], x.dtype)
    m = tf.cast(tf.shape(y)[0], y.dtype)

    Kxx = k(x, x)
    Kyy = k(y, y)
    Kxy = k(x, y)

    # Unbiased: remove diagonal from Kxx, Kyy
    sum_xx = (tf.reduce_sum(Kxx) - tf.linalg.trace(Kxx)) / (n * (n - 1.0))
    sum_yy = (tf.reduce_sum(Kyy) - tf.linalg.trace(Kyy)) / (m * (m - 1.0))
    sum_xy = tf.reduce_sum(Kxy) / (n * m)
    return sum_xx + sum_yy - 2.0 * sum_xy


def evaluate_one_cloud(model, particles, weights, h, query_points, n_kl_samples,
                        kl_seed, mmd_seed):
    """Evaluate all metrics on one cloud.

    Returns a dict of scalar Python floats.
    """
    c = model.encode(particles, weights, h_kde=h)

    # ---- MA residual on the held-out query points
    loss_ma, ma_diag = ma_residual_loss(
        model.trunk, particles, weights, h, query_points, c
    )

    # ---- Apply T to the original particle locations + Jacobians
    T_xi, J_xi = model.transport_with_jacobian(particles, c)  # (N, d), (N, d, d)

    # ---- J conditioning: eigenvalues of J at each particle location
    eigvals = tf.linalg.eigvalsh(J_xi)  # (N, d), ascending
    lambda_min_per_pt = eigvals[:, 0]  # smallest per particle
    sign_J, logabsdet_J = tf.linalg.slogdet(J_xi)
    logdet_finite = tf.math.is_finite(logabsdet_J)
    sign_ok = tf.equal(sign_J, tf.constant(1.0, dtype=sign_J.dtype))
    logdet_failures = tf.reduce_sum(
        tf.cast(tf.logical_not(tf.logical_and(logdet_finite, sign_ok)), tf.int32)
    )

    # ---- Moment error: source target vs transported empirical
    mu_p = weighted_mean(particles, weights)
    cov_p = weighted_covariance(particles, weights)
    N_d = tf.cast(tf.shape(particles)[0], particles.dtype)
    mu_T = tf.reduce_sum(T_xi, axis=0) / N_d
    centered_T = T_xi - mu_T[None, :]
    cov_T = tf.matmul(centered_T, centered_T, transpose_a=True) / N_d

    mean_err = tf.norm(mu_T - mu_p) / (tf.norm(mu_p) + 1e-12)
    cov_err = tf.norm(cov_T - cov_p) / (tf.norm(cov_p) + 1e-12)

    # ---- KL(p_h || r_h): MC estimate
    # Sample y ~ p_h, evaluate log p_h(y) - log r_h(y)
    # r_h is the uniform KDE of the transported particles {T(x_i)}
    y = sample_from_weighted_kde(particles, weights, h, n_kl_samples, kl_seed)
    log_p = log_kde_weighted(y, particles, weights, h)
    log_r = log_kde_uniform(y, T_xi, h)
    kl_p_r = tf.reduce_mean(log_p - log_r)

    # ---- MMD^2 between samples from p_h and from r_h
    keys = tf.random.experimental.stateless_split(mmd_seed, num=2)
    samples_p = sample_from_weighted_kde(particles, weights, h, n_kl_samples, keys[0])
    samples_r = sample_from_uniform_kde(T_xi, h, n_kl_samples, keys[1])
    mmd2 = _gaussian_mmd2(samples_p, samples_r, h)

    return {
        'ma_residual_mse': float(loss_ma.numpy()),
        'ma_residual_abs_mean': float(ma_diag['residual_abs_mean'].numpy()),
        'kl_p_r': float(kl_p_r.numpy()),
        'mmd2': float(mmd2.numpy()),
        'mean_err_rel': float(mean_err.numpy()),
        'cov_err_rel': float(cov_err.numpy()),
        'lambda_min_mean': float(tf.reduce_mean(lambda_min_per_pt).numpy()),
        'lambda_min_min': float(tf.reduce_min(lambda_min_per_pt).numpy()),
        'lambda_min_p10': float(np.percentile(lambda_min_per_pt.numpy(), 10)),
        'logdet_failures': int(logdet_failures.numpy()),
        'h': float(h.numpy()),
    }


def evaluate(model, num_clouds, n_particles, d, q_points, n_kl_samples=512,
             h_scale=1.0, base_seed=10000):
    """Run evaluation on `num_clouds` held-out random clouds.

    Returns:
        per_cloud: list of metric dicts (one per cloud)
        summary: dict of mean/std across clouds for each metric
    """
    per_cloud = []
    for k in range(num_clouds):
        cloud_seed = tf.constant([base_seed + k, 7], dtype=tf.int32)
        sub = tf.random.experimental.stateless_split(cloud_seed, num=4)
        particles, weights = sample_random_cloud(n_particles, d, sub[0])
        h = silverman_bandwidth_scalar(particles, weights, scale=h_scale)
        query_points = sample_from_uniform_kde(particles, h, q_points, sub[1])
        metrics = evaluate_one_cloud(
            model, particles, weights, h, query_points, n_kl_samples,
            kl_seed=sub[2], mmd_seed=sub[3],
        )
        per_cloud.append(metrics)

    # Aggregate
    keys = list(per_cloud[0].keys())
    summary = {}
    for key in keys:
        vals = np.array([m[key] for m in per_cloud], dtype=np.float64)
        summary[f'{key}_mean'] = float(np.mean(vals))
        summary[f'{key}_std'] = float(np.std(vals))
    summary['num_clouds'] = num_clouds
    return per_cloud, summary


def print_summary(summary):
    """Pretty-print evaluation summary."""
    print(f"\n  Evaluated on {summary['num_clouds']} held-out clouds:\n")
    metrics = [
        ('ma_residual_mse', '.4e'),
        ('ma_residual_abs_mean', '.4f'),
        ('kl_p_r', '.4f'),
        ('mmd2', '.4e'),
        ('mean_err_rel', '.4f'),
        ('cov_err_rel', '.4f'),
        ('lambda_min_mean', '.4f'),
        ('lambda_min_min', '.4f'),
        ('lambda_min_p10', '.4f'),
        ('logdet_failures', '.2f'),
        ('h', '.4f'),
    ]
    for name, fmt in metrics:
        mean = summary[f'{name}_mean']
        std = summary[f'{name}_std']
        print(f"    {name:24s} {mean:{fmt}}  +/- {std:{fmt}}")
