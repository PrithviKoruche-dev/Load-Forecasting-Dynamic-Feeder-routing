"""Shared test setup.

The project modules import each other by bare name (``import engine``, ``from features import ...``),
the same way the scripts do when run from their own folder, so the folders go on sys.path here.
"""
import faulthandler
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
for p in (PROJECT, PROJECT / "grid", PROJECT / "ml", PROJECT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load OpenDSS with pytest's crash reporter paused. On Windows the OpenDSS library raises and handles
# an internal exception while loading (code 0xE0465043, "FPC"), and faulthandler prints every native
# exception, even handled ones, as a long "Windows fatal exception" traceback. It is harmless.
_was_enabled = faulthandler.is_enabled()
if _was_enabled:
    faulthandler.disable()
try:
    import opendssdirect  # noqa: F401,E402
finally:
    if _was_enabled and sys.__stderr__ is not None:
        faulthandler.enable(file=sys.__stderr__)


@pytest.fixture(scope="session")
def project_root():
    return PROJECT


@pytest.fixture(scope="session")
def base_loads():
    """300 kW on every transformer: about 64% loading, the network's normal operating point."""
    import engine
    return {t: 300.0 for t in engine.TX}
