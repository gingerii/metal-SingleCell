"""Regression tests for the Greptile review findings (batch B), each checked against scanpy 1.11.5.

Every test here failed on the code before its fix.
"""

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

_HAS_SCANPY = importlib.util.find_spec("scanpy") is not None
needs_scanpy = pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy (oracle extra) not installed")


def _clusters(n_per=20, g=20, seed=0):
    """Three well-separated clusters; ``g`` <= 50 so ``.X`` is scanpy's automatic representation."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(m, 0.1, size=(n_per, g)) for m in (0, 5, 10)]).astype(np.float32)
    return ad.AnnData(X)


def _graph(adata, **kw):
    from metalsinglecell import pp as msc_pp
    a = adata.copy()
    msc_pp.neighbors(a, n_neighbors=5, **kw)
    return a.obsp["distances"].tocsr()


def _same_graph(A, B):
    A, B = A.tocsr(), B.tocsr()
    return A.shape == B.shape and abs(A - B).max() == 0


# --------------------------------------------------------------------------- neighbors: n_pcs


@pytest.mark.metal
def test_neighbors_n_pcs_zero_uses_full_X():
    """scanpy reads ``n_pcs=0`` as ``use_rep='X'``. Slicing .X to zero columns gave every
    distance 0 and a random graph."""
    a = _clusters()
    D0 = _graph(a, n_pcs=0)
    assert D0.data.max() > 0
    assert _same_graph(D0, _graph(a, use_rep="X"))


@pytest.mark.metal
def test_neighbors_n_pcs_does_not_truncate_X():
    """scanpy never slices .X by ``n_pcs`` — neither for ``use_rep='X'`` nor for the automatic
    .X choice on a panel of <= 50 variables. Only obsm representations are truncated."""
    a = _clusters()
    full = _graph(a, use_rep="X")
    assert _same_graph(_graph(a, use_rep="X", n_pcs=3), full)
    assert _same_graph(_graph(a, n_pcs=3), full)           # 20 vars -> .X is the rep


@pytest.mark.metal
def test_neighbors_n_pcs_still_truncates_obsm():
    a = _clusters()
    rng = np.random.default_rng(1)
    a.obsm["emb"] = rng.normal(size=(a.n_obs, 10)).astype(np.float32)
    a.obsm["emb3"] = a.obsm["emb"][:, :3].copy()
    assert _same_graph(_graph(a, use_rep="emb", n_pcs=3), _graph(a, use_rep="emb3"))
