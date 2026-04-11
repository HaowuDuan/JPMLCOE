"""Add tests/jit/ to sys.path so test files can import local helpers."""

import os
import sys

_jit_dir = os.path.dirname(os.path.abspath(__file__))
if _jit_dir not in sys.path:
    sys.path.insert(0, _jit_dir)
