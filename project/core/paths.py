"""Project folder locations, so no module has to work out relative paths itself."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]      # the project/ folder
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"


def project_path(relative):
    """Resolve a path written relative to project/ (as in the config files)."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
