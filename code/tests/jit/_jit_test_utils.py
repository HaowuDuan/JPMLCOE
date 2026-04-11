"""Shared helpers for JIT compilation tests.

Provides:
- try_compile: catch XLA compile errors and return (success, error_type, message)
- time_call: warmup + timed average over N runs, forces host sync
- save_result / reset_results: per-file JSON result saving
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf


# ----------------------------------------------------------------------------
# Compile probing
# ----------------------------------------------------------------------------

def try_compile(fn: Callable, *args, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
    """Call fn(*args, **kwargs) and catch any exception.

    Returns (succeeded, error_type, error_message). Forces host sync via .numpy()
    if the result is a tensor — this triggers actual XLA compilation, not just
    graph construction.
    """
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            for r in result:
                if hasattr(r, 'numpy'):
                    _ = r.numpy()
        elif hasattr(result, 'numpy'):
            _ = result.numpy()
        return True, None, None
    except Exception as e:  # noqa: BLE001 — we want to catch everything
        return False, type(e).__name__, str(e)[:1000]


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------

def time_call(fn: Callable, *args, n_warmup: int = 2, n_runs: int = 5, **kwargs) -> float:
    """Warmup then time `n_runs` calls. Returns mean wall-clock seconds per call.

    Forces host sync by calling .numpy() on the result if possible.
    """
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, 'numpy'):
            _ = result.numpy()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if hasattr(result, 'numpy'):
            _ = result.numpy()
        elif isinstance(result, (list, tuple)):
            for r in result:
                if hasattr(r, 'numpy'):
                    _ = r.numpy()
        times.append(time.perf_counter() - t0)
    return float(sum(times) / len(times))


# ----------------------------------------------------------------------------
# Results saving
# ----------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / 'results'


def _results_path(test_file: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f'{Path(test_file).stem}.json'


def save_result(test_file: str, case: Dict[str, Any]) -> None:
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
        json.dump(data, f, indent=2, default=str)


def reset_results(test_file: str) -> None:
    path = _results_path(test_file)
    if path.exists():
        path.unlink()
