"""Unit tests for path resolution (config.py). Pure-CPU, GitHub-hosted lane."""
import importlib
import os
from pathlib import Path


def test_repo_root_is_package_parent():
    from metalsinglecell import config
    assert (config.REPO_ROOT / "src" / "metalsinglecell" / "config.py").exists()


def test_data_root_defaults_under_repo():
    from metalsinglecell import config
    assert config.DATA_DIR == config.DATA_ROOT / "data"
    assert config.PROCESSED_DIR == config.DATA_ROOT / "data" / "processed"
    assert config.EXTERNAL_DIR == config.DATA_ROOT / "data" / "external"


def test_data_root_env_override(monkeypatch, tmp_path):
    """DATA_ROOT redirects data/ to an external mount; RESULTS_DIR stays at the repo."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import metalsinglecell.config as config
    importlib.reload(config)
    try:
        assert config.DATA_ROOT == Path(tmp_path).resolve()
        assert config.PROCESSED_DIR == Path(tmp_path).resolve() / "data" / "processed"
        # results live at the repo, not under DATA_ROOT
        assert config.RESULTS_DIR == config.REPO_ROOT / "results"
    finally:
        monkeypatch.delenv("DATA_ROOT", raising=False)
        importlib.reload(config)


def test_results_dir_creates(tmp_path, monkeypatch):
    from metalsinglecell import config
    d = config.results_dir("__unit_test_analysis__", create=True)
    assert d.exists() and d.is_dir()
    assert d == config.RESULTS_DIR / "__unit_test_analysis__"
    try:
        d.rmdir()
    except OSError:
        pass


def test_ensure_data_dirs_idempotent():
    from metalsinglecell import config
    config.ensure_data_dirs()
    for d in (config.RAW_DIR, config.PROCESSED_DIR, config.EXTERNAL_DIR):
        assert d.exists()
