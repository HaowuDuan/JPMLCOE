"""Pytest fixtures and configuration for code/neural_operator/tests/.

Sets up sys.path so test files can import from:
- this directory (for `_result_utils`)
- code/neural_operator/src/ (the operator implementation)
- code/src/ (the main package, for `resampling` etc.)

Order matters: neural_operator/src must come BEFORE code/src so that
`from models import WTanh` resolves to neural_operator/src/models.py
rather than the state-space model package at code/src/models/.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Insertion order: each insert(0, ...) ends up at index 0, so the LAST
# inserted path has the highest priority. We want the priority to be:
#   1) neural_operator/src   (highest, beats code/src/models)
#   2) code/src              (so `resampling`, etc., are importable)
#   3) tests dir             (so `_result_utils` is importable)
sys.path.insert(0, HERE)                                    # tests/
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))   # code/src
sys.path.insert(0, os.path.join(HERE, '..', 'src'))         # neural_operator/src

import pytest
import tensorflow as tf

# All neural_operator tests use float64 by default.
tf.keras.backend.set_floatx('float64')


@pytest.fixture(scope='session', autouse=True)
def _set_global_seed():
    """Deterministic global seed for the whole test session."""
    tf.random.set_seed(42)
    yield
