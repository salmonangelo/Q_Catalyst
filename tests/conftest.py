"""Pytest configuration ensuring src/ is discoverable in tests."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure optional external dependencies not needed for evaluation do not block test collection
if "Bio" not in sys.modules:
    try:
        import Bio  # noqa: F401
    except ImportError:
        import unittest.mock as _mock
        sys.modules["Bio"] = _mock.MagicMock()
        sys.modules["Bio.Align"] = _mock.MagicMock()
