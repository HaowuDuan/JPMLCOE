"""Add tests/hmc/ to sys.path so test files can import local helpers
(_gradient_test_utils, ledh_invertible_hmc_ablation)."""

import os
import sys

_hmc_dir = os.path.dirname(os.path.abspath(__file__))
if _hmc_dir not in sys.path:
    sys.path.insert(0, _hmc_dir)
