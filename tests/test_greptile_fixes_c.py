"""Regression tests for the verified Greptile findings in batch C.

1. Spatial k-NN / grid builders dropped self by column position, so a duplicate coordinate
   that tied self at distance 0 and sorted ahead of it became a self-edge, and the twin was lost.
2. ``gr.co_occurrence`` re-encoded labels in ``np.unique`` (lexical) order, so the axes of
   ``occ`` did not follow the categorical order that ``sq.pl.co_occurrence`` labels them with.
3. The vendored ``NNDescent.build`` called ``np.random.seed`` and drew from the global stream,
   resetting the caller's randomness.
"""

import importlib.util

import numpy as np
import pytest

_HAS_SQUIDPY = importlib.util.find_spec("squidpy") is not None
needs_squidpy = pytest.mark.skipif(not _HAS_SQUIDPY, reason="squidpy (oracle extra) not installed")


def _adata(coords, obs=None):
    import anndata as ad
    a = ad.AnnData(np.zeros((len(coords), 1), dtype=np.float32), obs=obs)
    a.obsm["spatial"] = np.asarray(coords, dtype=np.float64)
    return a


# Two exact duplicate pairs: (1, 3) and (6, 7). No ties at the 3rd-neighbour boundary, so the
# expected graph is unique and squidpy's answer is the only right one.
_DUP8 = np.array([[0, 0], [1, 0], [0, 1], [1, 0], [5, 5], [6, 5], [5, 6], [5, 6]], dtype=float)
_TWINS8 = {1: 3, 3: 1, 6: 7, 7: 6}


def _row_sets(M):
    M = M.tocsr()
    return [set(M.indices[M.indptr[i]:M.indptr[i + 1]].tolist()) for i in range(M.shape[0])]


def _tie_free_differences(A, B, coords):
    """Rows where A and B disagree for a reason other than an exact distance tie."""
    bad = 0
    for i, (sa, sb) in enumerate(zip(_row_sets(A), _row_sets(B))):
        if sa == sb:
            continue
        da = [np.linalg.norm(coords[i] - coords[j]) for j in sa - sb]
        db = [np.linalg.norm(coords[i] - coords[j]) for j in sb - sa]
        if not (da and db and abs(max(da) - max(db)) < 1e-9):
            bad += 1
    return bad


def _with_twins(n_base, n_dup, seed, scale=1000.0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, scale, size=(n_base, 2))
    src = rng.choice(n_base, n_dup, replace=False)
    coords = np.vstack([base, base[src]])
    twins = {int(s): n_base + t for t, s in enumerate(src)}
    twins.update({v: k for k, v in list(twins.items())})
    return coords, twins


# --------------------------------------------------------------------------- 1. duplicates


@pytest.mark.metal
@pytest.mark.parametrize("set_diag", [False, True])
def test_knn_duplicate_coordinates_keep_twins_not_self(set_diag):
    from metalsinglecell import gr as msc_gr
    a = _adata(_DUP8)
    msc_gr.spatial_neighbors_knn(a, n_neighs=3, set_diag=set_diag)
    C = a.obsp["spatial_connectivities"]
    diag = C.diagonal()
    assert np.all(diag == (1.0 if set_diag else 0.0))
    rows = _row_sets(C)
    for i, j in _TWINS8.items():
        assert j in rows[i], f"row {i} lost its duplicate {j}"
    assert all(len(r - {i}) == 3 for i, r in enumerate(rows))   # n_neighs non-self edges


@pytest.mark.metal
def test_low_level_spatial_helper_drops_self_by_identity():
    """``spatial.spatial_neighbors`` (the coords-level helper) had the same column-0 bug."""
    from metalsinglecell import spatial
    A = spatial.spatial_neighbors(_DUP8, n_neighs=3)
    assert np.all(A.diagonal() == 0.0)
    rows = _row_sets(A)
    for i, j in _TWINS8.items():
        assert j in rows[i], f"row {i} lost its duplicate {j}"
    assert all(len(r) == 3 for r in rows)


@pytest.mark.metal
@needs_squidpy
@pytest.mark.parametrize("set_diag", [False, True])
def test_knn_duplicate_coordinates_match_squidpy(set_diag):
    import squidpy as sq
    from metalsinglecell import gr as msc_gr
    a, b = _adata(_DUP8), _adata(_DUP8)
    msc_gr.spatial_neighbors_knn(a, n_neighs=3, set_diag=set_diag)
    sq.gr.spatial_neighbors(b, coord_type="generic", n_neighs=3, set_diag=set_diag)
    for key in ("spatial_connectivities", "spatial_distances"):
        A, B = a.obsp[key].tocsr(), b.obsp[key].tocsr()
        A.sort_indices(); B.sort_indices()
        assert _row_sets(A) == _row_sets(B), key
        np.testing.assert_allclose(A.toarray(), B.toarray(), atol=1e-12, err_msg=key)


@pytest.mark.metal
@needs_squidpy
def test_knn_duplicates_on_the_grid_index_path():
    """n > 4000 routes through the Metal cell-list kernel, whose sort is unstable on ties —
    there either twin could come first, so both orders must be handled."""
    import squidpy as sq
    from metalsinglecell import gr as msc_gr
    coords, twins = _with_twins(5000, 250, seed=0)
    a, b = _adata(coords), _adata(coords)
    msc_gr.spatial_neighbors_knn(a, n_neighs=6)
    sq.gr.spatial_neighbors(b, coord_type="generic", n_neighs=6)
    C = a.obsp["spatial_connectivities"]
    assert (C.diagonal() != 0).sum() == 0
    rows = _row_sets(C)
    assert all(j in rows[i] for i, j in twins.items())
    assert _tie_free_differences(C, b.obsp["spatial_connectivities"], coords) == 0


@pytest.mark.metal
@needs_squidpy
@pytest.mark.parametrize("set_diag", [False, True])
def test_grid_builder_duplicate_coordinates(set_diag):
    import squidpy as sq
    from metalsinglecell import gr as msc_gr
    # jittered hex lattice (ties broken) with every 120th spot duplicated
    side = 60
    r, c = np.divmod(np.arange(side * side), side)
    pts = np.c_[c + 0.5 * (r % 2), r * np.sqrt(3) / 2]
    pts = pts + np.random.default_rng(0).normal(0, 0.02, pts.shape)
    src = np.arange(0, len(pts), 120)
    coords = np.vstack([pts, pts[src]])
    twins = {int(s): len(pts) + t for t, s in enumerate(src)}
    twins.update({v: k for k, v in list(twins.items())})
    a, b = _adata(coords), _adata(coords)
    msc_gr.spatial_neighbors_grid(a, n_neighs=6, set_diag=set_diag)
    sq.gr.spatial_neighbors(b, coord_type="grid", n_neighs=6, set_diag=set_diag)
    C = a.obsp["spatial_connectivities"]
    assert np.all(C.diagonal() == (1.0 if set_diag else 0.0))
    rows = _row_sets(C)
    assert all(j in rows[i] for i, j in twins.items())
    assert _tie_free_differences(C, b.obsp["spatial_connectivities"], coords) == 0


# --------------------------------------------------------------------------- 2. co_occurrence


def _typed_field(cats, labels_fn, n=400, seed=0):
    import pandas as pd
    coords = np.random.default_rng(seed).uniform(0, 100, size=(n, 2))
    obs = pd.DataFrame({"ct": pd.Categorical(labels_fn(coords), categories=cats)})
    return _adata(coords, obs=obs)


def _four_bands(xy):
    x = xy[:, 0]
    return np.where(x < 25, "0", np.where(x < 50, "1", np.where(x < 75, "2", "10")))


@pytest.mark.metal
@needs_squidpy
def test_co_occurrence_follows_categorical_order():
    """Leiden-style labels: categories '0','1','2','10' in numeric order, which np.unique
    would sort to '0','1','10','2'."""
    import squidpy as sq
    from metalsinglecell import gr as msc_gr
    a = _typed_field(["0", "1", "2", "10"], _four_bands)
    occ_m, int_m = msc_gr.co_occurrence(a.copy(), "ct", interval=10, copy=True)
    occ_s, int_s = sq.gr.co_occurrence(a.copy(), "ct", interval=10, copy=True)
    np.testing.assert_allclose(int_m, int_s)
    assert occ_m.shape == occ_s.shape
    np.testing.assert_allclose(occ_m, occ_s, rtol=1e-6, atol=1e-9)
    # uns slot carries the same array
    msc_gr.co_occurrence(a, "ct", interval=10)
    np.testing.assert_allclose(a.uns["ct_co_occurrence"]["occ"], occ_s, rtol=1e-6, atol=1e-9)


@pytest.mark.metal
@needs_squidpy
def test_co_occurrence_drops_unused_categories_and_keeps_order():
    """An unused category is dropped (squidpy's dimensions); the rest keep category order.

    squidpy itself mis-indexes when an unused category sits in the middle, so the oracle is
    squidpy on the same data with the unused category removed first.
    """
    import squidpy as sq
    from metalsinglecell import gr as msc_gr

    def bands(xy):
        x = xy[:, 0]
        return np.where(x < 33, "A", np.where(x < 66, "C", "B"))

    a = _typed_field(["A", "gone", "C", "B"], bands)
    occ_m, _ = msc_gr.co_occurrence(a.copy(), "ct", interval=10, copy=True)
    ref = a.copy()
    ref.obs["ct"] = ref.obs["ct"].cat.remove_unused_categories()
    occ_s, _ = sq.gr.co_occurrence(ref, "ct", interval=10, copy=True)
    assert occ_m.shape == (3, 3, 9)
    np.testing.assert_allclose(occ_m, occ_s, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- 3. NNDescent RNG


def _small_embedding(n=600, d=8, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


@pytest.mark.metal
@pytest.mark.parametrize("random_state", [0, None])
def test_nndescent_leaves_global_rng_untouched(random_state):
    from metalsinglecell._vendor.mlx_vis.nndescent import NNDescent
    X = _small_embedding()
    np.random.seed(12345)
    expected = np.random.rand(4)
    np.random.seed(12345)
    NNDescent(k=10, random_state=random_state).build(X)
    np.testing.assert_array_equal(np.random.rand(4), expected)


@pytest.mark.metal
def test_nndescent_high_dim_projection_leaves_global_rng_untouched():
    """d > 100 takes the random-projection branch, which also draws random numbers."""
    from metalsinglecell._vendor.mlx_vis.nndescent import NNDescent
    X = _small_embedding(n=400, d=120)
    np.random.seed(7)
    expected = np.random.rand(4)
    np.random.seed(7)
    NNDescent(k=10, random_state=3).build(X)
    np.testing.assert_array_equal(np.random.rand(4), expected)


@pytest.mark.metal
@pytest.mark.parametrize("d", [8, 120])
def test_nndescent_is_deterministic_for_a_random_state(d):
    from metalsinglecell._vendor.mlx_vis.nndescent import NNDescent
    X = _small_embedding(n=500, d=d)
    np.random.seed(1)
    i1, d1 = NNDescent(k=10, random_state=5).build(X)
    np.random.seed(2)                       # a different global state must not matter
    i2, d2 = NNDescent(k=10, random_state=5).build(X)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(d1, d2)
