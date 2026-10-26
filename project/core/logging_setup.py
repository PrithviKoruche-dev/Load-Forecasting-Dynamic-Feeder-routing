"""One logging setup for the whole project.

    from core.logging_setup import get_logger
    log = get_logger(__name__)
    log.info("rerouted %s", action)

Messages go to the console (stderr). Set LFDR_LOG_LEVEL (DEBUG, INFO, WARNING, ...) to change the
level, and LFDR_LOG_FILE (or call setup(log_file=...)) to also write a rotating log file.
"""
import logging
import logging.handlers
import os
from pathlib import Path

ROOT_NAME = "lfdr"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup(level=None, log_file=None):
    """Configure the project logger once. Safe to call repeatedly (later calls can add a log file)."""
    root = logging.getLogger(ROOT_NAME)
    level = level or os.environ.get("LFDR_LOG_LEVEL", "INFO")
    root.setLevel(level.upper() if isinstance(level, str) else level)
    root.propagate = False
    fmt = logging.Formatter(FORMAT)
    if not any(getattr(h, "_lfdr", None) == "console" for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        h._lfdr = "console"
        root.addHandler(h)
    log_file = log_file or os.environ.get("LFDR_LOG_FILE")
    if log_file:
        path = str(Path(log_file).resolve())
        if not any(getattr(h, "_lfdr", None) == path for h in root.handlers):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            h = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            h.setFormatter(fmt)
            h._lfdr = path
            root.addHandler(h)
    return root


def get_logger(name):
    setup()
    short = name.rsplit(".", 1)[-1] if name != "__main__" else "main"
    return logging.getLogger(f"{ROOT_NAME}.{short}")
