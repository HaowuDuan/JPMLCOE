"""Diagnostic: what is the MA residual of the exact 1D oracle transport map?

For two 1D KDE densities p_h (uniform-weighted) and q_h (weighted), the
unique monotone transport map is given by CDF matching:

    T_oracle(x) = F_q^{-1}(F_p(x))

where F_p, F_q are the CDFs. The Jacobian is computed numerically:

    J_oracle(x) = dT_oracle/dx    (via np.gradient on a fine grid)

If the MA residual of this oracle is near zero, the loss CAN be driven to
zero by the correct map — and the model's plateau at ~0.23 is an
optimization / architecture problem, not a fundamental KDE limitation.

If the oracle residual is also ~0.23, there is a numerical floor at this
h and grid resolution, and the model is already at the best achievable.

This test also reports the density-ratio Jacobian J_ratio = p(x)/q(T(x))
as a cross-check: if the oracle construction is correct, J_oracle and
J_ratio should agree closely.

Run: pytest code/neural_operator/tests/test_oracle_residual.py -v -s
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from scipy import integrate

from losses import ma_residual_loss
from kde import log_kde_uniform, log_kde_weighted
from _result_utils import save_result, reset_results


DTYPE = tf.float64


def setup_module():
    reset_results(__file__)


def _make_fixed_cloud(n=200, A=5.0, mu=1.0, sigma_w=3.0):
    """Same cloud as test_overfit_single_cloud."""
    x = np.linspace(-A, A, n).reshape(-1, 1)
    log_w = -0.5 * ((x[:, 0] - mu) ** 2) / (sigma_w ** 2)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w = w / w.sum()
    return x, w


def _build_oracle(particles_np, weights_np, h_val, grid_n=10001, grid_pad=3.0):
    """Build the 1D oracle transport map via CDF matching.

    Returns:
        grid: (G,) numpy array
        T_grid: (G,) oracle map evaluated on the grid
        J_grid: (G,) numerical derivative dT/dx on the grid
        F_p: (G,) source CDF
        F_q: (G,) target CDF
    """
    A = float(particles_np.max())
    grid = np.linspace(-A - grid_pad, A + grid_pad, grid_n)
    dx = grid[1] - grid[0]

    # Evaluate KDEs on the grid using TF, then convert to numpy
    grid_tf = tf.constant(grid.reshape(-1, 1), dtype=tf.float64)
    particles_tf = tf.constant(particles_np, dtype=tf.float64)
    weights_tf = tf.constant(weights_np, dtype=tf.float64)
    h_tf = tf.constant(h_val, dtype=tf.float64)

    log_p = log_kde_uniform(grid_tf, particles_tf, h_tf).numpy()
    log_q = log_kde_weighted(grid_tf, particles_tf, weights_tf, h_tf).numpy()
    p_grid = np.exp(log_p)
    q_grid = np.exp(log_q)

    # CDFs via cumulative trapezoidal integration
    F_p = integrate.cumulative_trapezoid(p_grid, grid, initial=0)
    F_q = integrate.cumulative_trapezoid(q_grid, grid, initial=0)
    # Normalize to [0, 1]
    F_p = F_p / F_p[-1]
    F_q = F_q / F_q[-1]

    # Oracle map: T(x) = F_q^{-1}(F_p(x))
    # For each grid point, look up F_p(x), then invert F_q
    u = np.clip(F_p, 1e-10, 1.0 - 1e-10)

    # Remove any flat regions in F_q before interpolation
    # (ensures strict monotonicity for inversion)
    mask = np.diff(F_q, prepend=-1) > 0
    F_q_mono = F_q[mask]
    grid_mono = grid[mask]

    T_grid = np.interp(u, F_q_mono, grid_mono)

    # Numerical Jacobian: dT/dx
    J_grid = np.gradient(T_grid, grid)

    return grid, T_grid, J_grid, F_p, F_q, p_grid, q_grid


class OracleTrunk:
    """Mock trunk that returns T_oracle(x) and J_oracle(x) via interpolation.

    Uses the numerically-differentiated Jacobian, NOT the density ratio.
    """

    def __init__(self, grid, T_grid, J_grid):
        self._grid = grid
        self._T_grid = T_grid
        self._J_grid = J_grid

    def forward_and_jacobian(self, x, c):
        # x: (Q, 1) tensor -> numpy for interpolation
        x_np = x.numpy().ravel()
        T_np = np.interp(x_np, self._grid, self._T_grid)
        J_np = np.interp(x_np, self._grid, self._J_grid)
        T_x = tf.constant(T_np.reshape(-1, 1), dtype=x.dtype)
        J_x = tf.constant(J_np.reshape(-1, 1, 1), dtype=x.dtype)
        return T_x, J_x


def test_oracle_residual():
    print("\n=== Oracle 1D transport map residual test ===\n")

    particles_np, weights_np = _make_fixed_cloud(n=200, A=5.0, mu=1.0, sigma_w=3.0)
    N = particles_np.shape[0]
    delta = 10.0 / (N - 1)
    h_val = 5.0 * delta  # fixed h = 0.2513

    print(f"  Cloud: N={N}, A=5, mu=1, sigma_w=3, h={h_val:.4f}")

    # Build oracle
    grid, T_grid, J_grid, F_p, F_q, p_grid, q_grid = _build_oracle(
        particles_np, weights_np, h_val
    )

    # Sanity: CDF endpoints
    print(f"  F_p range: [{F_p[0]:.6e}, {F_p[-1]:.6f}]")
    print(f"  F_q range: [{F_q[0]:.6e}, {F_q[-1]:.6f}]")
    print(f"  T_oracle range: [{T_grid.min():.4f}, {T_grid.max():.4f}]")
    print(f"  J_oracle range: [{J_grid.min():.4f}, {J_grid.max():.4f}]")

    # Cross-check: compare numerical J to density-ratio J
    particles_tf = tf.constant(particles_np, dtype=tf.float64)
    weights_tf = tf.constant(weights_np, dtype=tf.float64)
    h_tf = tf.constant(h_val, dtype=tf.float64)

    # Use deterministic quantile-based query points
    Q = 200
    u_quantiles = np.linspace(1e-4, 1 - 1e-4, Q)
    # Remove flat regions for inversion
    mask_p = np.diff(F_p, prepend=-1) > 0
    F_p_mono = F_p[mask_p]
    grid_mono_p = grid[mask_p]
    x_q_np = np.interp(u_quantiles, F_p_mono, grid_mono_p)
    x_q_tf = tf.constant(x_q_np.reshape(-1, 1), dtype=tf.float64)

    # Density-ratio Jacobian at query points (diagnostic only)
    log_p_xq = log_kde_uniform(x_q_tf, particles_tf, h_tf).numpy()
    T_xq_np = np.interp(x_q_np, grid, T_grid)
    T_xq_tf = tf.constant(T_xq_np.reshape(-1, 1), dtype=tf.float64)
    log_q_Txq = log_kde_weighted(T_xq_tf, particles_tf, weights_tf, h_tf).numpy()
    log_J_ratio = log_p_xq - log_q_Txq

    # Numerical Jacobian at query points
    J_xq_np = np.interp(x_q_np, grid, J_grid)
    log_J_numerical = np.log(np.maximum(J_xq_np, 1e-30))

    # Mismatch between the two Jacobian estimates
    J_mismatch = np.mean(np.abs(log_J_numerical - log_J_ratio))
    print(f"\n  Cross-check: mean |log J_numerical - log J_ratio| = {J_mismatch:.6f}")

    # Evaluate MA residual with the oracle mock
    oracle = OracleTrunk(grid, T_grid, J_grid)
    loss, diag = ma_residual_loss(oracle, particles_tf, weights_tf, h_tf, x_q_tf, c=None)
    loss_val = float(loss.numpy())
    residual_abs = float(diag['residual_abs_mean'].numpy())
    residual_mean = float(diag['residual_mean'].numpy())

    print(f"\n  Oracle MA loss        = {loss_val:.6e}")
    print(f"  Oracle residual_abs   = {residual_abs:.6f}")
    print(f"  Oracle residual_mean  = {residual_mean:+.6f}")

    # Moment error of the oracle map applied to particle locations
    T_particles_np = np.interp(particles_np.ravel(), grid, T_grid)
    mu_w = float((weights_np * particles_np.ravel()).sum())
    mu_T = float(T_particles_np.mean())
    mean_err = abs(mu_T - mu_w) / (abs(mu_w) + 1e-12)
    print(f"\n  Oracle moment check:")
    print(f"    weighted mean   = {mu_w:+.4f}")
    print(f"    oracle T mean   = {mu_T:+.4f}")
    print(f"    oracle mean_err = {mean_err:.4f}")

    save_result(__file__, {
        'case_name': 'oracle_1d_residual',
        'description': 'Exact 1D CDF-matched oracle on the fixed sigma_w=3 cloud',
        'config': {
            'n': N,
            'A': 5.0,
            'mu': 1.0,
            'sigma_w': 3.0,
            'h': h_val,
            'Q': Q,
            'grid_n': len(grid),
        },
        'metrics': {
            'oracle_loss': loss_val,
            'oracle_residual_abs': residual_abs,
            'oracle_residual_mean': residual_mean,
            'J_mismatch_log_abs_mean': float(J_mismatch),
            'oracle_mean_err': float(mean_err),
            'oracle_T_mean': float(mu_T),
            'weighted_mean': float(mu_w),
            'T_range': [float(T_grid.min()), float(T_grid.max())],
            'J_range': [float(J_grid.min()), float(J_grid.max())],
        },
        'interpretation': (
            'If oracle_loss < 1e-4: MA loss CAN be zero, model plateau is optimization/architecture. '
            'If oracle_loss ~ 0.23: KDE floor at this h, model is already at the best achievable.'
        ),
        'passed': True,  # diagnostic only, no hard assertion
    })

    print(f"\n  Interpretation:")
    if loss_val < 1e-4:
        print(f"    Oracle loss is near zero ({loss_val:.2e}).")
        print(f"    -> The MA loss CAN be driven to zero.")
        print(f"    -> The model's plateau at ~0.23 is an optimization/architecture problem.")
    elif loss_val > 0.1:
        print(f"    Oracle loss is substantial ({loss_val:.2e}).")
        print(f"    -> There may be a KDE numerical floor at this h.")
        print(f"    -> The model may already be near the best achievable.")
    else:
        print(f"    Oracle loss is moderate ({loss_val:.2e}).")
        print(f"    -> Partially a KDE floor, partially model limitation.")

    print("\n=== Oracle residual test complete ===\n")
