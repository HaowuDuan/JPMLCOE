"""Test result persistence helpers for code/neural_operator/tests/.

Mirrors the pattern in code/tests/hmc/_gradient_test_utils.py:
- save_result(__file__, case): append a case dict to results/<module>.json
- reset_results(__file__): wipe the per-module results file at session start

Each test module saves a JSON record per case to
code/neural_operator/tests/results/<module_stem>.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


RESULTS_DIR = Path(__file__).parent / 'results'


def _results_path(test_file: str) -> Path:
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
            'created': datetime.now(timezone.utc).isoformat(),
            'cases': [],
        }
    case = dict(case)
    case.setdefault('timestamp', datetime.now(timezone.utc).isoformat())
    data['cases'].append(case)
    data['updated'] = datetime.now(timezone.utc).isoformat()
    with path.open('w') as f:
        json.dump(data, f, indent=2)


def reset_results(test_file: str) -> None:
    """Wipe the results file for a given test module — call once per session."""
    path = _results_path(test_file)
    if path.exists():
        path.unlink()
