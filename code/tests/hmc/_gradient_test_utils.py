"""Shared helpers for HMC gradient validation tests.

Provides:
- compute_autodiff_grad: autodiff gradient via tf.GradientTape + DifferentiableModel
- compute_fd_grad: central finite-difference gradient with multi-radius median
- relative_error: rel-err with absolute fallback for tiny FD values
- save_results: append a case record to a per-file JSON in tests/hmc/results/
- TOL_*: per-model tolerances (LG/RB/SV1D/SV2D)

Tolerances are based on Codex review:
  LG  <= 0.03
  RB  <= 0.08
  SV1D <= 0.10
  SV2D <= 0.15
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------------------
# Tolerances (relative error, with absolute fallback for tiny FD)
# ----------------------------------------------------------------------------

TOL_LG = 0.03
TOL_RB = 0.08
TOL_SV1D = 0.10
TOL_SV2D = 0.15

ABS_FALLBACK = 1e-4   # if |fd| < ABS_FALLBACK_FD, use abs error
ABS_FALLBACK_FD = 1e-3


# ----------------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------------

def relative_error(autodiff_val: float, fd_val: float) -> float:
    """Relative error with absolute fallback when |fd| is tiny."""
    if abs(fd_val) < ABS_FALLBACK_FD:
        return float(abs(autodiff_val - fd_val))
    return float(abs(autodiff_val - fd_val) / abs(fd_val))


def fd_grad_central(eval_fn: Callable[[float], float], param_val: float,
                    h: float = 1e-4, n_radii: int = 5) -> Tuple[float, np.ndarray]:
    """Central finite-difference gradient at param_val.

    Uses multiple radii (h, 2h, 3h, ..., n_radii*h) and returns the median.
    Returns (median_grad, all_slopes).
    """
    radii = np.array([(k + 1) * h for k in range(n_radii)])
    slopes = np.empty(n_radii)
    for i, r in enumerate(radii):
        if param_val - r <= 1e-3:
            slopes[i] = np.nan
            continue
        f_plus = eval_fn(param_val + r)
        f_minus = eval_fn(param_val - r)
        slopes[i] = (f_plus - f_minus) / (2.0 * r)
    valid = slopes[~np.isnan(slopes)]
    if len(valid) == 0:
        return float('nan'), slopes
    return float(np.median(valid)), slopes


def autodiff_grad(diff_model, filt, obs_tf, param_name: str, param_val: float,
                  dtype, seed: tf.Tensor) -> Tuple[float, float]:
    """Autodiff gradient of log-likelihood w.r.t. a single scalar parameter.

    Returns (log_lik, gradient).
    """
    var = tf.constant(param_val, dtype=dtype)
    with tf.GradientTape() as tape:
        tape.watch(var)
        diff_model.update_parameters({param_name: var})
        ll = filt.log_marginal_likelihood_tf(obs_tf, seed=seed)
    grad = tape.gradient(ll, var)
    diff_model.restore_parameters()
    if grad is None:
        return float(ll.numpy()), float('nan')
    return float(ll.numpy()), float(grad.numpy())


# ----------------------------------------------------------------------------
# Results saving
# ----------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / 'results'


def _results_path(test_file: str) -> Path:
    """Path to the JSON results file for a given test module file path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(test_file).stem
    return RESULTS_DIR / f'{name}.json'


def save_result(test_file: str, case: Dict[str, Any]) -> None:
    """Append a case record to the per-file JSON results.

    test_file: __file__ of the calling test module.
    case: dict with metadata + values + pass/fail flag.
    """
    path = _results_path(test_file)
    if path.exists():
        with path.open('r') as f:
            data = json.load(f)
    else:
        data = {
            'file': Path(test_file).name,
            'created': datetime.utcnow().isoformat() + 'Z',
            'cases': [],
        }
    case = dict(case)
    case.setdefault('timestamp', datetime.utcnow().isoformat() + 'Z')
    data['cases'].append(case)
    data['updated'] = datetime.utcnow().isoformat() + 'Z'
    with path.open('w') as f:
        json.dump(data, f, indent=2)


def reset_results(test_file: str) -> None:
    """Wipe the results file for a given test module — call once per session."""
    path = _results_path(test_file)
    if path.exists():
        path.unlink()


# ----------------------------------------------------------------------------
# Gradient case helper — combines compute + print + save + assert
# ----------------------------------------------------------------------------

def gradient_case(
    *,
    test_file: str,
    case_name: str,
    model_name: str,
    filter_name: str,
    param_name: str,
    param_val: float,
    eval_fn: Callable[[float], float],
    diff_model,
    filt,
    obs_tf,
    dtype,
    seed: tf.Tensor,
    tol: float,
    n_particles: int,
    T: int,
    n_lambda_steps: Optional[int] = None,
    fd_h: float = 1e-4,
    fd_n_radii: int = 5,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one gradient comparison case: autodiff vs FD.

    Computes both, prints them, saves to JSON, and asserts rel_err <= tol.
    Returns the case dict.
    """
    ll_val, ad_grad = autodiff_grad(diff_model, filt, obs_tf, param_name,
                                    param_val, dtype, seed)
    fd_med, fd_slopes = fd_grad_central(eval_fn, param_val, h=fd_h, n_radii=fd_n_radii)
    rel_err = relative_error(ad_grad, fd_med) if not np.isnan(fd_med) and not np.isnan(ad_grad) else float('nan')
    passed = (not np.isnan(rel_err)) and (rel_err <= tol)

    case = {
        'case_name': case_name,
        'model': model_name,
        'filter': filter_name,
        'param_name': param_name,
        'param_value': float(param_val),
        'n_particles': int(n_particles),
        'T': int(T),
        'n_lambda_steps': None if n_lambda_steps is None else int(n_lambda_steps),
        'fd_h': float(fd_h),
        'fd_n_radii': int(fd_n_radii),
        'fd_slopes': [None if np.isnan(s) else float(s) for s in fd_slopes.tolist()],
        'log_likelihood': float(ll_val),
        'autodiff_grad': float(ad_grad),
        'fd_grad_median': float(fd_med),
        'relative_error': float(rel_err),
        'tolerance': float(tol),
        'passed': bool(passed),
    }
    if extra:
        case.update(extra)

    print(f"\n  [{case_name}]")
    print(f"    model={model_name}  filter={filter_name}  param={param_name}={param_val}")
    print(f"    log_lik={ll_val:.6f}")
    print(f"    autodiff_grad={ad_grad:.6f}")
    print(f"    fd_grad_median={fd_med:.6f}")
    print(f"    fd_slopes={np.array2string(fd_slopes, precision=4)}")
    print(f"    rel_err={rel_err:.6f}  (tol={tol})  {'PASS' if passed else 'FAIL'}")

    save_result(test_file, case)

    assert passed, (
        f"{case_name}: relative error {rel_err:.4f} exceeds tolerance {tol} "
        f"(autodiff={ad_grad:.6f}, fd={fd_med:.6f})"
    )
    return case
