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


# --------------------------------------------------------------------------- neighbors: tiny n_obs


@pytest.mark.metal
@pytest.mark.parametrize("k", [15, 40])
def test_neighbors_clamps_n_neighbors_on_tiny_objects(k):
    """scanpy clamps ``n_neighbors > n_obs`` to ``1 + int(0.5 * n_obs)`` with a warning and
    records the effective value. 10 cells at the default 15 raised ``negative axis 1 index: -1``
    and 40 raised a length mismatch."""
    from metalsinglecell import pp as msc_pp
    a = _clusters()[:10].copy()
    with pytest.warns(UserWarning, match="n_obs too small"):
        msc_pp.neighbors(a, n_neighbors=k, use_rep="X")
    assert a.uns["neighbors"]["params"]["n_neighbors"] == 6
    assert (np.diff(a.obsp["distances"].tocsr().indptr) == 5).all()   # k - 1, self excluded


@needs_scanpy
@pytest.mark.metal
def test_neighbors_clamp_matches_scanpy_params():
    import scanpy as sc
    from metalsinglecell import pp as msc_pp
    a = _clusters()[:10].copy()
    b = a.copy()
    with pytest.warns(UserWarning):
        msc_pp.neighbors(a, use_rep="X")
    sc.pp.neighbors(b, use_rep="X")
    assert a.uns["neighbors"]["params"]["n_neighbors"] == b.uns["neighbors"]["params"]["n_neighbors"]
    assert (np.diff(a.obsp["distances"].tocsr().indptr)
            == np.diff(b.obsp["distances"].tocsr().indptr)).all()


# --------------------------------------------------------------------------- pca: zero_center=False


def _offset(n=200, g=30, seed=0):
    rng = np.random.default_rng(seed)
    return (5.0 + rng.normal(0, 1, size=(n, g))).astype(np.float32)


@pytest.mark.metal
@pytest.mark.parametrize("solver", ["full", "arpack", "randomized", "covariance_eigh"])
def test_pca_uncentred_variance_ratio_is_a_ratio(solver):
    """Uncentred PCA divided S**2/(n-1), which carries the mean, by the centred total variance,
    so PC1's ratio was ~25 on offset data. sklearn's TruncatedSVD (scanpy's zero_center=False)
    uses the variance of the scores, which keeps every ratio and their sum <= 1."""
    from metalsinglecell.decomposition import pca
    x_pca, _, vr = pca(_offset(), n_comps=5, solver=solver, zero_center=False)
    assert (vr <= 1).all() and vr.sum() <= 1
    np.testing.assert_allclose(vr, np.var(x_pca.astype(np.float64), axis=0)
                               / np.var(_offset().astype(np.float64), axis=0).sum(), rtol=1e-4)


@needs_scanpy
@pytest.mark.metal
def test_pca_uncentred_variance_ratio_matches_scanpy():
    import anndata as ad
    import scanpy as sc
    from metalsinglecell import pp as msc_pp
    a, b = ad.AnnData(_offset()), ad.AnnData(_offset())
    # Pin arpack on both: scanpy's own zero_center=False default is randomized TruncatedSVD,
    # which is off by up to 0.5% on PCs 2-5 of this flat spectrum, against the exact SVD.
    msc_pp.pca(a, n_comps=5, zero_center=False, svd_solver="arpack")
    sc.pp.pca(b, n_comps=5, zero_center=False, svd_solver="arpack")
    np.testing.assert_allclose(a.uns["pca"]["variance_ratio"], b.uns["pca"]["variance_ratio"],
                               rtol=1e-4)


# --------------------------------------------------------------------------- scale: stored std


def _small_counts(n=10, g=4, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    return ad.AnnData(sp.csr_matrix(rng.poisson(2, size=(n, g)).astype(np.float32)))


@pytest.mark.metal
def test_scale_stores_the_std_it_divides_by():
    """var['std'] was the population std (ddof=0) while .X was divided by the sample std
    (ddof=1), so unscaling with the stored statistics did not give back the data."""
    from metalsinglecell import pp as msc_pp
    a = _small_counts()
    X = a.X.toarray().astype(np.float64)
    msc_pp.scale(a)
    np.testing.assert_allclose(a.var["std"].to_numpy(), X.std(axis=0, ddof=1), rtol=1e-6)
    rec = np.asarray(a.X) * a.var["std"].to_numpy() + a.var["mean"].to_numpy()
    np.testing.assert_allclose(rec, X, atol=1e-5)


@needs_scanpy
@pytest.mark.metal
def test_scale_std_matches_scanpy():
    import scanpy as sc
    from metalsinglecell import pp as msc_pp
    a, b = _small_counts(), _small_counts()
    msc_pp.scale(a)
    sc.pp.scale(b)
    np.testing.assert_allclose(a.var["std"].to_numpy(), b.var["std"].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(a.var["mean"].to_numpy(), b.var["mean"].to_numpy(), rtol=1e-6)


# --------------------------------------------------------------------------- filter: inplace=False


@pytest.mark.metal
@pytest.mark.parametrize("fn,kw", [
    ("filter_cells", {"min_genes": 3}), ("filter_cells", {"min_counts": 3}),
    ("filter_genes", {"min_cells": 3}), ("filter_genes", {"min_counts": 3}),
])
def test_filter_inplace_false_leaves_object_unchanged(fn, kw):
    """scanpy returns ``(mask, number)`` before writing anything; msc wrote n_genes/n_counts/
    n_cells into .obs/.var first, overwriting any existing column of that name."""
    import anndata as ad
    from metalsinglecell import pp as msc_pp
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1, size=(20, 10)).astype(np.float32)))
    a.obs["n_counts"] = -1.0                       # a pre-existing column must survive
    a.var["n_counts"] = -1.0
    obs, var = a.obs.copy(), a.var.copy()
    mask, number = getattr(msc_pp, fn)(a, inplace=False, **kw)
    assert a.shape == (20, 10)
    assert a.obs.equals(obs) and a.var.equals(var)
    assert mask.dtype == bool and len(number) == len(mask)


@needs_scanpy
@pytest.mark.metal
@pytest.mark.parametrize("fn,kw", [
    ("filter_cells", {"min_genes": 3}), ("filter_genes", {"min_cells": 3}),
])
def test_filter_inplace_false_returns_scanpys_mask_and_counts(fn, kw):
    import anndata as ad
    import scanpy as sc
    from metalsinglecell import pp as msc_pp
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1, size=(20, 10)).astype(np.float32))
    m1, n1 = getattr(msc_pp, fn)(ad.AnnData(X.copy()), inplace=False, **kw)
    m2, n2 = getattr(sc.pp, fn)(ad.AnnData(X.copy()), inplace=False, **kw)
    assert np.array_equal(m1, m2) and np.array_equal(n1, n2)


# --------------------------------------------------------------------------- validation.compare


@pytest.mark.parametrize("got,expected", [
    (np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(3, 2)),   # same flattened values
    (np.ones((5, 1)), np.ones(5)),                                   # strict: no squeezing
])
def test_compare_rejects_shape_mismatch_before_flattening(got, expected):
    from metalsinglecell import validation
    r = validation.compare("shape", got, expected)
    assert not r["passed"] and r["reason"] == "shape_mismatch"
    assert r["got_shape"] == got.shape and r["expected_shape"] == expected.shape


@pytest.mark.parametrize("got,expected", [
    ([np.nan], [np.inf]), ([np.inf], [-np.inf]), ([np.nan], [-np.inf]),
    ([1.0, np.inf], [1.0, np.nan]),
])
def test_compare_distinguishes_nan_and_signed_inf(got, expected):
    from metalsinglecell import validation
    r = validation.compare("nonfinite", np.array(got), np.array(expected))
    assert not r["passed"] and not r["nonfinite_masks_match"]


def test_compare_matching_nonfinite_values_still_pass():
    from metalsinglecell import validation
    a = np.array([[1.0, np.nan], [np.inf, -np.inf]])
    r = validation.compare("nonfinite_ok", a, a.copy())
    assert r["passed"] and r["nonfinite_masks_match"] and r["n_compared"] == 1
