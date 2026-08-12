"""Return types, payload shapes and validation across ``msc.gr`` (2026-08 API review).

These are the contract-level differences a per-function numeric comparison misses: what
``copy=True`` hands back, what shape ``occ`` is, whether a mislabelled row can slip out, and
which typos raise instead of computing the wrong statistic.
"""

import importlib.util

import numpy as np
import pytest

from metalsinglecell import gr as msc_gr

_HAS_SQUIDPY = importlib.util.find_spec("squidpy") is not None
needs_squidpy = pytest.mark.skipif(not _HAS_SQUIDPY, reason="squidpy (oracle extra) not installed")


def labelled(n=300, k=4, seed=0):
    """Coordinates + a categorical label + a few genes."""
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, (n, 2))
    a = ad.AnnData(rng.poisson(4, (n, 6)).astype(np.float32))
    a.var_names = [f"g{i}" for i in range(6)]
    a.obsm["spatial"] = xy
    a.obs["ct"] = pd.Categorical(rng.integers(0, k, n).astype(str))
    return a


# --------------------------------------------------------------------------- co_occurrence


@pytest.mark.metal
def test_co_occurrence_interval_counts_thresholds_not_bins():
    """squidpy's `interval` is a count of thresholds, so `occ` has interval-1 entries."""
    a = labelled()
    msc_gr.co_occurrence(a, "ct", interval=20)
    res = a.uns["ct_co_occurrence"]
    k = len(a.obs["ct"].cat.categories)
    assert len(res["interval"]) == 20
    assert res["occ"].shape == (k, k, 19)


@pytest.mark.metal
def test_co_occurrence_accepts_an_explicit_threshold_array():
    a = labelled()
    thr = np.linspace(50.0, 400.0, 8)
    msc_gr.co_occurrence(a, "ct", interval=thr)
    res = a.uns["ct_co_occurrence"]
    assert np.allclose(res["interval"], thr)
    assert res["occ"].shape[-1] == 7


@pytest.mark.metal
def test_co_occurrence_requires_a_categorical_and_returns_a_tuple_under_copy():
    a = labelled()
    a.obs["plain"] = a.obs["ct"].astype(str)
    with pytest.raises(TypeError, match="categorical"):
        msc_gr.co_occurrence(a, "plain")
    out = msc_gr.co_occurrence(a, "ct", interval=10, copy=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert "ct_co_occurrence" not in a.uns


@needs_squidpy
@pytest.mark.metal
def test_co_occurrence_matches_squidpy():
    import squidpy as sq
    a = labelled()
    o, t = a.copy(), a.copy()
    msc_gr.co_occurrence(o, "ct", interval=20)
    sq.gr.co_occurrence(t, cluster_key="ct", interval=20, show_progress_bar=False, n_jobs=1)
    ro, rt = o.uns["ct_co_occurrence"], t.uns["ct_co_occurrence"]
    assert ro["occ"].shape == rt["occ"].shape
    assert np.allclose(ro["interval"], rt["interval"], rtol=1e-5)
    assert np.allclose(ro["occ"], rt["occ"], atol=1e-2)


# --------------------------------------------------------------------------- ligrec


@pytest.mark.metal
def test_ligrec_labels_survive_a_dropped_pair():
    """Regression: pairs were filtered, then labelled positionally, so every row after a
    dropped pair carried the wrong name while holding the right numbers."""
    a = labelled()
    msc_gr.ligrec(a, "ct", [("g0", "g1"), ("NOPE", "g3"), ("g4", "g5")], n_perms=5)
    means = a.uns["ct_ligrec"]["means"]
    assert list(means.index) == [("g0", "g1"), ("g4", "g5")]

    # and the values on the surviving row are the ones for that pair
    cats = list(a.obs["ct"].cat.categories)
    X = np.asarray(a.X)
    m = a.obs["ct"].to_numpy()
    lig = X[m == cats[0], a.var_names.get_loc("g4")].mean()
    rec = X[m == cats[1], a.var_names.get_loc("g5")].mean()
    assert means.loc[("g4", "g5"), (cats[0], cats[1])] == pytest.approx(0.5 * (lig + rec), abs=1e-5)


@pytest.mark.metal
def test_ligrec_payload_is_squidpys_dataframe_layout():
    import pandas as pd
    a = labelled()
    msc_gr.ligrec(a, "ct", [("g0", "g1"), ("g2", "g3")], n_perms=5)
    res = a.uns["ct_ligrec"]
    assert set(res) == {"means", "pvalues", "metadata"}
    for k in ("means", "pvalues"):
        assert isinstance(res[k], pd.DataFrame)
        assert res[k].index.names == ["source", "target"]
        assert res[k].columns.names == ["cluster_1", "cluster_2"]
    k = len(a.obs["ct"].cat.categories)
    assert res["means"].shape == (2, k * k)


@pytest.mark.metal
def test_ligrec_copy_returns_the_dict():
    a = labelled()
    out = msc_gr.ligrec(a, "ct", [("g0", "g1")], n_perms=5, copy=True)
    assert set(out) == {"means", "pvalues", "metadata"}
    assert "ct_ligrec" not in a.uns


@pytest.mark.metal
def test_ligrec_raises_when_no_pair_is_present():
    a = labelled()
    with pytest.raises(ValueError, match="none of the interaction pairs"):
        msc_gr.ligrec(a, "ct", [("NOPE", "ALSO_NOPE")], n_perms=5)


# --------------------------------------------------------------------------- copy contracts


@pytest.mark.metal
@pytest.mark.parametrize("builder,kw", [
    (msc_gr.spatial_neighbors_knn, {"n_neighs": 6}),
    (msc_gr.spatial_neighbors_radius, {"radius": 200.0}),
    (msc_gr.spatial_neighbors_grid, {"n_neighs": 6}),
    (msc_gr.spatial_neighbors_delaunay, {}),
])
def test_builders_return_the_result_not_the_object(builder, kw):
    """squidpy returns SpatialNeighborsResult(connectivities, distances) and writes nothing."""
    a = labelled()
    out = builder(a, copy=True, **kw)
    assert len(out) == 2
    assert out.connectivities.shape == (a.n_obs, a.n_obs)
    assert out.distances.shape == (a.n_obs, a.n_obs)
    assert "spatial_connectivities" not in a.obsp
    assert "spatial_neighbors" not in a.uns


# --------------------------------------------------------------------------- validation


@pytest.mark.metal
def test_spatial_autocorr_rejects_an_unknown_mode():
    """Unvalidated, mode='Moran' computed Geary's C, scored it against Moran's analytic null,
    and filed it under uns['gearyC'] with pval_norm = 0 for every gene."""
    a = labelled()
    msc_gr.spatial_neighbors_knn(a, n_neighs=6)
    for bad in ("Moran", "moranI", "geary's c", ""):
        with pytest.raises(ValueError, match="mode must be"):
            msc_gr.spatial_autocorr(a, mode=bad)


@pytest.mark.metal
def test_spatial_autocorr_use_raw_intersects_with_var_names():
    """squidpy scores `set(adata.var_names) & set(raw.var_names)`, not all of .raw. The row
    count feeds the FDR divisor, so taking all of .raw shifts every corrected p-value."""
    import anndata as ad
    a = labelled()
    msc_gr.spatial_neighbors_knn(a, n_neighs=6)
    big = ad.AnnData(np.hstack([np.asarray(a.X), np.asarray(a.X)]).astype(np.float32))
    big.var_names = [f"g{i}" for i in range(12)]
    a.raw = big
    df = msc_gr.spatial_autocorr(a, mode="moran", use_raw=True, copy=True)
    assert len(df) == a.n_vars == 6


@pytest.mark.metal
def test_deprecated_shim_reproduces_squidpys_legacy_quirks():
    a = labelled()
    with pytest.raises(ValueError, match="percentile"):
        with pytest.warns(FutureWarning):
            msc_gr.spatial_neighbors(a, coord_type="grid", percentile=90)

    # scalar radius + delaunay is ignored upstream; a tuple is honoured
    scalar, tup, plain = a.copy(), a.copy(), a.copy()
    with pytest.warns(FutureWarning):
        msc_gr.spatial_neighbors(scalar, coord_type="generic", delaunay=True, radius=150.0)
    with pytest.warns(FutureWarning):
        msc_gr.spatial_neighbors(tup, coord_type="generic", delaunay=True, radius=(0.0, 150.0))
    with pytest.warns(FutureWarning):
        msc_gr.spatial_neighbors(plain, coord_type="generic", delaunay=True)
    nnz = lambda o: o.obsp["spatial_connectivities"].nnz      # noqa: E731
    assert nnz(scalar) == nnz(plain)
    assert nnz(tup) < nnz(plain)
    assert scalar.uns["spatial_neighbors"]["params"]["radius"] is None
