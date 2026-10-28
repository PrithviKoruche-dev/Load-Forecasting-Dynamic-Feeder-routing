"""Shared test setup.

The project modules import each other by bare name (``import engine``, ``from features import ...``),
the same way the scripts do when run from their own folder, so the folders go on sys.path here.
"""
import faulthandler
import os
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
for p in (PROJECT, PROJECT / "grid", PROJECT / "ml", PROJECT / "app", PROJECT / "sim", PROJECT / "city"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load OpenDSS with pytest's crash reporter paused. On Windows the OpenDSS library raises and handles
# an internal exception while loading (code 0xE0465043, "FPC"), and faulthandler prints every native
# exception, even handled ones, as a long "Windows fatal exception" traceback. It is harmless.
# The replay dashboard's tests run in a process of their own (tests/test_dashboard_process.py): OpenDSS crashes
# when it is used from another thread than the one that loaded it, and that page uses it from a worker thread,
# so in that process nothing may load OpenDSS before the page does.
DASHBOARD_PROCESS = os.environ.get("LFDR_DASHBOARD_PROCESS") == "1"
_was_enabled = faulthandler.is_enabled()
if _was_enabled and not DASHBOARD_PROCESS:
    faulthandler.disable()
try:
    if not DASHBOARD_PROCESS:
        import opendssdirect  # noqa: F401,E402
finally:
    if _was_enabled and not DASHBOARD_PROCESS and sys.__stderr__ is not None:
        faulthandler.enable(file=sys.__stderr__)


def pytest_collection_modifyitems(session, config, items):
    """tests/test_dashboard.py runs only in its own process (see DASHBOARD_PROCESS above): in any other test
    process its tests are deselected, and test_dashboard_process.py runs them in a fresh one."""
    if DASHBOARD_PROCESS:
        return
    out = [i for i in items if i.fspath.basename == "test_dashboard.py"]
    if out:
        items[:] = [i for i in items if i.fspath.basename != "test_dashboard.py"]
        config.hook.pytest_deselected(items=out)


@pytest.fixture(scope="session")
def project_root():
    return PROJECT


@pytest.fixture(scope="session")
def base_loads():
    """300 kW on every transformer: about 64% loading, the network's normal operating point."""
    import engine
    return {t: 300.0 for t in engine.TX}

