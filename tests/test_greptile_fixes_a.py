"""Regression tests for the backed/streaming review findings (Greptile set A).

Each test reproduces one verified bug on a tiny synthetic zarr store (seconds, no data files):
the deferred-transform registry leaking across objects, ``materialize`` wiping its own source,
backed PCA silently centring under ``zero_center=False``, the streaming solver crashing when
``n_comps`` exceeds the selected genes, ``normalize_total``'s median target vs scanpy 1.11.5,
backed ``normalize_total(inplace=False)`` mutating state, and the README streaming example
loading ``.X`` into memory.
"""
import gc
import importlib.util
import os
import re
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from metalsinglecell import pp as msc_pp

pytestmark = pytest.mark.metal

_HAS_SCANPY = importlib.util.find_spec("scanpy") is not None
needs_scanpy = pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy (oracle extra) not installed")

_ZERO_MSG = "Some cells have zero counts"


def _store(path, X, block_rows=4, obs_names=None, var_names=None):
    """Write ``X`` as a chunked backed zarr at ``path`` and return the path string."""
    import anndata as ad
    from metalsinglecell.backed import write_backed_zarr
    a = ad.AnnData(sp.csr_matrix(np.asarray(X, dtype=np.float32)))
    if obs_names is not None:
        a.obs_names = obs_names
    if var_names is not None:
        a.var_names = var_names
    write_backed_zarr(a, str(path), block_rows=block_rows)
    return str(path)


def _backed(path):
    import anndata as ad
    import zarr
    from anndata.io import sparse_dataset
    return ad.AnnData(X=sparse_dataset(zarr.open(str(path), mode="r")["X"]))


def _poisson(n, g, lam=3.0, seed=0):
    return np.random.default_rng(seed).poisson(lam, (n, g)).astype(np.float32)


def _mostly_empty(seed=0):
    """12 cells x 6 genes with 7 empty rows: the all-cell median total is 0."""
    X = _poisson(12, 6, seed=seed)
    X[:7] = 0
    return X


def _snapshot(path):
    """Every file under a store with its bytes — to prove a store was left untouched."""
    out = {}
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, path)] = fh.read()
    return out


# ---------------------------------------------------------------- 1. registry id reuse


def test_stream_registry_entry_dies_with_its_object(tmp_path):
    p = _store(tmp_path / "s.zarr", _poisson(20, 8))
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    msc_pp.log1p(a)
    key = id(a)
    assert key in msc_pp._STREAM_TRANSFORMS
    del a
    gc.collect()
    assert key not in msc_pp._STREAM_TRANSFORMS, "deferred stages outlived their AnnData"


def test_fresh_backed_object_never_inherits_stages(tmp_path):
    """CPython reuses a dead object's id; a new object must not pick up its stages."""
    p = _store(tmp_path / "s.zarr", _poisson(20, 8))
    inherited = 0
    for _ in range(100):
        a = _backed(p)                          # reassigning frees the previous object
        if msc_pp._stream_stages(a):
            inherited += 1
        msc_pp.normalize_total(a, target_sum=1e4)
        msc_pp.log1p(a)
    del a
    gc.collect()
    assert inherited == 0, f"{inherited}/100 fresh objects started with another's stages"


# ------------------------------------------------------- 2. materialize onto its source


@pytest.mark.parametrize("where", ["same", "parent", "inside"])
def test_materialize_refuses_to_overwrite_its_source(tmp_path, where):
    src_dir = tmp_path / "stores"
    src_dir.mkdir()
    p = _store(src_dir / "src.zarr", _poisson(20, 8, lam=2.0))
    before = _snapshot(p)
    dest = {"same": p, "parent": str(src_dir), "inside": os.path.join(p, "ckpt.zarr")}[where]
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    msc_pp.log1p(a)
    with pytest.raises(ValueError, match="source"):
        msc_pp.materialize(a, dest)
    assert _snapshot(p) == before, "the source store was modified"
    assert not os.path.exists(os.path.join(p, "ckpt.zarr"))
    # the raw counts still read back
    assert _backed(p).X[:].nnz == sp.csr_matrix(_poisson(20, 8, lam=2.0)).nnz


def test_materialize_to_a_new_path_still_works(tmp_path):
    p = _store(tmp_path / "src.zarr", _poisson(20, 8))
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    msc_pp.log1p(a)
    msc_pp.materialize(a, tmp_path / "ckpt.zarr")
    assert a.X[:].nnz == sp.csr_matrix(_poisson(20, 8)).nnz
    assert msc_pp._stream_stages(a) == []


# ------------------------------------------------- 3. backed pca(zero_center=False)


def test_backed_pca_uncentred_raises_before_doing_anything(tmp_path):
    p = _store(tmp_path / "s.zarr", _poisson(60, 12), block_rows=16)
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    stages = list(msc_pp._stream_stages(a))
    with pytest.raises(NotImplementedError, match="zero_center"):
        msc_pp.pca(a, n_comps=4, zero_center=False, mask_var=None)
    assert "X_pca" not in a.obsm and "PCs" not in a.varm and "pca" not in a.uns
    assert msc_pp._stream_stages(a) == stages
