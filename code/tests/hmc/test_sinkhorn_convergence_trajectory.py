"""Record Sinkhorn marginal-error trajectories for the OT report figure."""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from _gradient_test_utils import reset_results, save_result


SEED = 42
N = 200
STATE_DIM = 2
MAX_ITER = 500
THRESHOLD = 1e-3
EPSILONS = [0.01, 0.05, 0.1, 0.5]


@pytest.fixture(scope="module", autouse=True)
def _reset_results():
    reset_results(__file__)
    yield


def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)


def _softmin(epsilon, cost_matrix, f):
    return -epsilon * _logsumexp(f[None, :] - cost_matrix / epsilon, axis=-1)


def _cost_matrix(x):
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    dist = np.maximum(x_sq - 2.0 * x @ x.T + x_sq.T, 0.0)
    return 0.5 * dist


def _fixed_cloud():
    rng = np.random.default_rng(SEED)
    particles = rng.normal(0.0, 1.0, (N, STATE_DIM))
    raw_weights = rng.lognormal(mean=0.0, sigma=1.0, size=N)
    weights = raw_weights / np.sum(raw_weights)

    centered = particles - np.mean(particles, axis=0, keepdims=True)
    scale = np.std(particles) * np.sqrt(STATE_DIM) + 1e-8
    scaled = centered / scale
    return scaled.astype(np.float64), weights.astype(np.float64)


def _transport_from_potentials(cost_matrix, alpha, beta, epsilon, log_weights):
    log_n = np.log(float(N))
    log_t = (alpha[:, None] + beta[None, :] - cost_matrix) / epsilon
    log_t = log_t - _logsumexp(log_t, axis=0)[None, :] + log_n
    return np.exp(log_t + log_weights[None, :])


def _sinkhorn_trajectory(epsilon):
    particles, weights = _fixed_cloud()
    cost_matrix = _cost_matrix(particles)
    log_weights = np.log(weights + 1e-10)
    uniform_log_weights = -np.log(float(N)) * np.ones_like(log_weights)
    alpha = np.zeros(N, dtype=np.float64)
    beta = np.zeros(N, dtype=np.float64)
    row_errs = []
    col_errs = []
    converged = False

    for _ in range(MAX_ITER):
        alpha_new = _softmin(epsilon, cost_matrix.T, log_weights + beta / epsilon)
        beta_new = _softmin(epsilon, cost_matrix, uniform_log_weights + alpha_new / epsilon)
        alpha_next = 0.5 * (alpha + alpha_new)
        beta_next = 0.5 * (beta + beta_new)

        transport = _transport_from_potentials(
            cost_matrix, alpha_next, beta_next, epsilon, log_weights
        )
        row_err = np.max(np.abs(np.sum(transport, axis=1) - 1.0))
        col_err = np.max(np.abs(np.sum(transport, axis=0) - float(N) * weights))
        row_errs.append(float(row_err))
        col_errs.append(float(col_err))

        max_diff = max(np.max(np.abs(alpha_next - alpha)), np.max(np.abs(beta_next - beta)))
        alpha, beta = alpha_next, beta_next
        if max_diff <= THRESHOLD:
            converged = True
            break

    return {
        "N": N,
        "epsilon": float(epsilon),
        "max_iter": MAX_ITER,
        "row_err_trajectory": row_errs,
        "col_err_trajectory": col_errs,
        "n_iter": len(row_errs),
        "converged": bool(converged),
        "seed": SEED,
    }


@pytest.mark.parametrize("epsilon", EPSILONS, ids=[f"eps={eps}" for eps in EPSILONS])
def test_sinkhorn_convergence_trajectory(epsilon):
    result = _sinkhorn_trajectory(epsilon)
    case = {"case_name": f"eps={epsilon}", **result}
    save_result(__file__, case)

    assert result["n_iter"] == len(result["row_err_trajectory"])
    assert result["n_iter"] == len(result["col_err_trajectory"])
    assert np.all(np.isfinite(result["row_err_trajectory"]))
    assert np.all(np.isfinite(result["col_err_trajectory"]))
