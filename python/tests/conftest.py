"""Make the sibling ``fixtures`` module importable in every pytest import mode.

``test_api.py`` does ``from fixtures import CASES``. Under pytest's default
``prepend`` import mode that works because this directory is put on
``sys.path`` -- but only then, and only if nothing named ``fixtures`` sits
earlier on the path. Pinning the insertion here makes the import independent
of the rootdir, of ``--import-mode``, and of whatever else is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
