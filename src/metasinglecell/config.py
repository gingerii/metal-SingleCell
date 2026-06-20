"""Path configuration for metal-SingleCell.

All filesystem paths resolve through here, honoring the ``DATA_ROOT`` environment
variable so the same code runs on any machine. Never hardcode paths in the
library or notebooks — import from this module instead.

Layout (relative to the repo root, or to ``DATA_ROOT`` if set)::

    data/raw/         immutable inputs (downloaded datasets)
    data/processed/   derived shared objects (h5ads) — regenerable, gitignored
    data/external/    references / panels
    results/          reportable artifacts (csv/png/pdf), grouped by analysis
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (src/metasinglecell/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# DATA_ROOT lets you point data/ at an external mount; defaults to the repo.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT)).resolve()

DATA_DIR = DATA_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

RESULTS_DIR = REPO_ROOT / "results"


def results_dir(analysis: str, create: bool = True) -> Path:
    """Return ``results/<analysis>/``, creating it by default."""
    p = RESULTS_DIR / analysis
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_data_dirs() -> None:
    """Create the data/ subtree if missing (raw stays immutable once populated)."""
    for d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
