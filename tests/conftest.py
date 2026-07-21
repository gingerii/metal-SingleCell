"""Shared pytest fixtures + markers for the metal-SingleCell test suite.

Two lanes (see BRIEF_release_pypi_ci.md):
- CPU lane — pure-Python, no GPU/data; runs on GitHub-hosted CI. Import-contract, config,
  validation units.
- GPU/data lane — needs the Metal GPU and/or the fp64 oracle + PBMC3k data; runs on a
  self-hosted macOS-arm64 runner. Marked ``@pytest.mark.metal`` / ``@pytest.mark.data``.

Markers:
- ``metal``    — requires the Apple-Silicon GPU (MLX). Skipped if mlx import fails.
- ``data``     — requires ``data/pbmc3k_raw.h5ad`` (gitignored). Skipped if absent.
- ``realdata`` — requires large local/private datasets (atlas, Xenium). Never in PR CI.
"""
import importlib.util

import pytest


def pytest_configure(config):
    for name, desc in [
        ("metal", "requires the Apple-Silicon Metal GPU (mlx)"),
        ("data", "requires data/pbmc3k_raw.h5ad (gitignored)"),
        ("realdata", "requires large local/private datasets; never in PR CI"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")


def _has(mod):
    return importlib.util.find_spec(mod) is not None


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU/data tests when the environment can't run them."""
    from metasinglecell import config as msc_config
    have_mlx = _has("mlx")
    have_pbmc = (msc_config.REPO_ROOT / "data" / "pbmc3k_raw.h5ad").exists()
    for item in items:
        if "metal" in item.keywords and not have_mlx:
            item.add_marker(pytest.mark.skip(reason="mlx (Metal GPU) not available"))
        if "data" in item.keywords and not have_pbmc:
            item.add_marker(pytest.mark.skip(reason="data/pbmc3k_raw.h5ad not present"))
        if "realdata" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="realdata: large local dataset, run manually"))


@pytest.fixture(scope="session")
def pbmc_counts():
    """A small in-memory PBMC3k counts AnnData (float32 CSR). Requires the data file."""
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp
    from metasinglecell import config
    a = sc.read_h5ad(config.REPO_ROOT / "data" / "pbmc3k_raw.h5ad")
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    return a


@pytest.fixture(scope="session")
def cpu_oracle(tmp_path_factory):
    """Build the fp64 CPU reference snapshots once per session (blocks 01–09 compare vs these)."""
    from metasinglecell.reference import run_cpu_reference
    out = tmp_path_factory.mktemp("cpu_oracle")
    return run_cpu_reference(out_dir=out)
