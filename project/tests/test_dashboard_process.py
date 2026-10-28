"""Runs tests/test_dashboard.py (the what-if replay page, headless) in a process of its own.

That page runs OpenDSS on a worker thread, and OpenDSS crashes the process when it is used from another thread
than the one that loaded it. Any other test loads it on the main thread, so in a shared test process the page
crashes (always when run alone, now and then in the full suite; on Linux, e.g. GitHub Actions). In its own
process the page loads OpenDSS first, on its worker thread, as it does in the real dashboard.
"""
import os
import subprocess
import sys

import pytest


def test_replay_dashboard_in_its_own_process(project_root, pytestconfig):
    if os.environ.get("LFDR_DASHBOARD_PROCESS") == "1":
        pytest.skip("already the dashboard's own process")
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_dashboard.py"]
    marks = pytestconfig.getoption("markexpr")
    if marks:
        args += ["-m", marks]
    r = subprocess.run(args, cwd=project_root, env={**os.environ, "LFDR_DASHBOARD_PROCESS": "1"},
                       capture_output=True, text=True, timeout=1800)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-40:])
    assert r.returncode == 0, f"the replay dashboard tests failed in their own process:\n{tail}"
    assert " passed" in r.stdout and " failed" not in r.stdout, tail
