"""Shared test setup.

The project modules import each other by bare name (``import engine``, ``from features import ...``),
the same way the scripts do when run from their own folder, so the folders go on sys.path here.
"""
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
for sub in ("grid", "ml", "app"):
    p = str(PROJECT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def project_root():
    return PROJECT


@pytest.fixture(scope="session")
def base_loads():
    """300 kW on every transformer: about 64% loading, the network's normal operating point."""
    import engine
    return {t: 300.0 for t in engine.TX}
