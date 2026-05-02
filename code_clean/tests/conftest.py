"""pytest configuration shared by all tests under ``code_clean/tests/``.

Adds the package parent (``code_clean/``) to ``sys.path`` so tests can
``import jpml_tf`` without installing the package. The rebuild does not
yet ship a ``pyproject.toml`` install step; the package runs in place
from the working tree.
"""

import sys
from pathlib import Path

_PKG_PARENT = Path(__file__).resolve().parent.parent  # code_clean/
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))
