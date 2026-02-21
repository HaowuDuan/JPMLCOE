"""Root conftest — centralized sys.path setup for all tests."""
import sys
import os

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = os.path.dirname(_tests_dir)

if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
