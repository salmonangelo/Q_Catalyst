"""Pytest configuration for pet_degradation_lab test suite."""

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = LAB_ROOT.parent

for path in [LAB_ROOT, REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

