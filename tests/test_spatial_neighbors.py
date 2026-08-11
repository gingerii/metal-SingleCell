"""The mode-specific spatial-neighbour builders (github issue #4).

squidpy deprecated ``spatial_neighbors`` in 1.7 for removal in 1.9, in favour of
``spatial_neighbors_{knn,radius,delaunay,grid}``. These pin our four against squidpy where it
is installed, and pin the pieces that do not need it everywhere.
"""

import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from metalsinglecell import gr as msc_gr

_HAS_SQUIDPY = importlib.util.find_spec("squidpy") is not None
needs_squidpy = pytest.mark.skipif(not _HAS_SQUIDPY, reason="squidpy (oracle extra) not installed")


def lattice(side=40, jitter=0.0, seed=0):
    """A hex lattice like Visium, optionally jittered. Ties are deliberate: an exact lattice
    is the case where neighbour selection is ambiguous."""
    import anndata as ad
    r, c = np.divmod(np.arange(side * side), side)
    xy = np.c_[c * 100.0 + (r % 2) * 50.0, r * 86.6]
    if jitter:
        xy = xy + np.random.default_rng(seed).normal(0, jitter, xy.shape)
    a = ad.AnnData(np.zeros((side * side, 2), dtype=np.float32))
    a.obsm["spatial"] = xy
    return a


def _tie_free_differences(A, B, coords):
    """Count rows where A and B disagree for a reason other than an exact distance tie."""
    A, B = A.tocsr(), B.tocsr()
    bad = 0
    for i in range(A.shape[0]):
        sa = set(A.indices[A.indptr[i]:A.indptr[i + 1]])
        sb = set(B.indices[B.indptr[i]:B.indptr[i + 1]])
        if sa == sb:
            continue
        da = [np.linalg.norm(coords[i] - coords[j]) for j in sa - sb]
        db = [np.linalg.norm(coords[i] - coords[j]) for j in sb - sa]
        if not (da and db and abs(max(da) - max(db)) < 1e-9):
            bad += 1
    return bad


# --------------------------------------------------------------------------- shape / contract


@pytest.mark.metal
@pytest.mark.parametrize("builder,kw", [
    (msc_gr.spatial_neighbors_knn, {"n_neighs": 6}),
    (msc_gr.spatial_neighbors_radius, {"radius": 120.0}),
    (msc_gr.spatial_neighbors_grid, {"n_neighs": 6}),
    (msc_gr.spatial_neighbors_delaunay, {}),
])
def test_writes_the_squidpy_slots(builder, kw):
    a = lattice(20)
    builder(a, **kw)
    assert "spatial_connectivities" in a.obsp
    assert "spatial_distances" in a.obsp
    uns = a.uns["spatial_neighbors"]
    assert uns["connectivities_key"] == "spatial_connectivities"
    assert uns["distances_key"] == "spatial_distances"
    assert "transform" in uns["params"]
    assert a.obsp["spatial_distances"].diagonal().sum() == 0      # squidpy zeroes it


@pytest.mark.metal
def test_key_added_is_respected():
    a = lattice(15)
    msc_gr.spatial_neighbors_knn(a, n_neighs=4, key_added="mygraph")
    assert "mygraph_connectivities" in a.obsp
    assert "mygraph_neighbors" in a.uns


@pytest.mark.metal
def test_knn_is_directed_with_fixed_degree():
    a = lattice(20)
    msc_gr.spatial_neighbors_knn(a, n_neighs=6)
    A = a.obsp["spatial_connectivities"].tocsr()
    assert np.all(np.diff(A.indptr) == 6)                          # exactly k per row
    assert set(np.unique(A.data)) == {1.0}


@pytest.mark.metal
def test_radius_is_symmetric_and_respects_the_cutoff():
    # lattice spacing is 100, so the shells sit at ~100 and ~173: radius 200 captures both
    a = lattice(20, jitter=2.0)
    msc_gr.spatial_neighbors_radius(a, radius=200.0)
    A = a.obsp["spatial_connectivities"]
    D = a.obsp["spatial_distances"]
    assert (abs(A - A.T) > 1e-9).nnz == 0
    assert D.data.max() <= 200.0 + 1e-6

    b = lattice(20, jitter=2.0)
    msc_gr.spatial_neighbors_radius(b, radius=(130.0, 200.0))      # drops the first shell
    Db = b.obsp["spatial_distances"]
    assert Db.nnz > 0
    assert Db.data.min() >= 130.0 - 1e-6 and Db.data.max() <= 200.0 + 1e-6
    assert Db.nnz < D.nnz


@pytest.mark.metal
def test_grid_distances_are_ring_indices_not_lengths():
    """squidpy's convention, and easy to get wrong: for a grid graph the `distances` slot
    holds the ring number, so with n_rings=2 the only values are 1 and 2."""
    a = lattice(20)
    msc_gr.spatial_neighbors_grid(a, n_neighs=6, n_rings=2)
    vals = set(np.unique(a.obsp["spatial_distances"].data))
    assert vals == {1.0, 2.0}
    assert set(np.unique(a.obsp["spatial_connectivities"].data)) == {1.0}


@pytest.mark.metal
def test_grid_rings_expand_the_neighbourhood():
    prev = 0
    for rings in (1, 2, 3):
        a = lattice(20)
        msc_gr.spatial_neighbors_grid(a, n_neighs=6, n_rings=rings)
        nnz = a.obsp["spatial_connectivities"].nnz
        assert nnz > prev
        prev = nnz


@pytest.mark.metal
def test_set_diag_touches_connectivities_only():
    a = lattice(15)
    msc_gr.spatial_neighbors_knn(a, n_neighs=4, set_diag=True)
    assert np.all(a.obsp["spatial_connectivities"].diagonal() == 1.0)
    assert a.obsp["spatial_distances"].diagonal().sum() == 0


@pytest.mark.metal
def test_unknown_transform_raises():
    a = lattice(10)
    with pytest.raises(NotImplementedError, match="transform"):
        msc_gr.spatial_neighbors_knn(a, transform="banana")


# --------------------------------------------------------------------------- the old entry point


@pytest.mark.metal
def test_deprecated_spatial_neighbors_warns_and_dispatches():
    a = lattice(15)
    with pytest.warns(FutureWarning, match="deprecated"):
        msc_gr.spatial_neighbors(a, n_neighs=6)
    assert a.uns["spatial_neighbors"]["params"]["coord_type"] == "generic"


@pytest.mark.metal
def test_coord_type_actually_changes_the_graph():
    """Through 0.1.1 `coord_type` was accepted and ignored — 'grid' silently returned k-NN."""
    generic, grid = lattice(20), lattice(20)
    with pytest.warns(FutureWarning):
        msc_gr.spatial_neighbors(generic, n_neighs=6, coord_type="generic")
    with pytest.warns(FutureWarning):
        msc_gr.spatial_neighbors(grid, n_neighs=6, coord_type="grid")
    A, B = generic.obsp["spatial_connectivities"], grid.obsp["spatial_connectivities"]
    assert A.nnz != B.nnz or (abs(A - B) > 0).nnz > 0
    assert grid.uns["spatial_neighbors"]["params"]["coord_type"] == "grid"


@pytest.mark.metal
def test_deprecated_infers_grid_from_uns_spatial():
    a = lattice(15)
    a.uns["spatial"] = {}                                          # what a Visium reader writes
    with pytest.warns(FutureWarning):
        msc_gr.spatial_neighbors(a)
    assert a.uns["spatial_neighbors"]["params"]["coord_type"] == "grid"


# --------------------------------------------------------------------------- kernels / precision


@pytest.mark.metal
def test_radius_kernel_matches_sklearn_exactly():
    from sklearn.neighbors import NearestNeighbors
    from metalsinglecell.neighbors import _radius_grid
    rng = np.random.RandomState(0)
    for X, r in [(rng.uniform(0, 100, (3000, 2)).astype(np.float32), 6.0),
                 ((rng.normal(size=(3000, 2)) * [30, 3]).astype(np.float32), 4.0),
                 (rng.uniform(0, 50, (2000, 3)).astype(np.float32), 7.0)]:
        indptr, cols, dist = _radius_grid(X, r)
        ours = sp.csr_matrix((np.ones_like(dist), cols, indptr), shape=(len(X),) * 2)
        ref = NearestNeighbors(radius=r).fit(X).radius_neighbors_graph(mode="connectivity")
        ref.setdiag(0); ref.eliminate_zeros()
        assert (abs(ours - ref) > 0).nnz == 0


@pytest.mark.metal
def test_distances_survive_large_coordinates():
    """Regression: the fp32 Gram identity |a|^2+|b|^2-2a.b cancels catastrophically at Visium
    magnitudes. Two spots exactly 273 apart came back at 272.9395."""
    from metalsinglecell.neighbors import _exact_knn_rows
    base = np.array([[9237.0, 15447.0], [9510.0, 15447.0], [9237.0, 15720.0]])
    for offset in (0.0, 20_000.0, 200_000.0):                      # Visium, then Xenium-scale
        X = np.ascontiguousarray(base + offset, dtype=np.float32)
        _, d = _exact_knn_rows(X, np.arange(3), 2)
        assert abs(d[0, 1] - 273.0) < 1e-3, (offset, d[0, 1])


# --------------------------------------------------------------------------- squidpy parity


@needs_squidpy
@pytest.mark.metal
@pytest.mark.parametrize("ours,theirs,kw", [
    ("spatial_neighbors_knn", "spatial_neighbors_knn", {"n_neighs": 6}),
    ("spatial_neighbors_knn", "spatial_neighbors_knn", {"n_neighs": 6, "set_diag": True}),
    ("spatial_neighbors_knn", "spatial_neighbors_knn",
     {"n_neighs": 6, "transform": "spectral"}),
    ("spatial_neighbors_radius", "spatial_neighbors_radius", {"radius": 150.0}),
    ("spatial_neighbors_radius", "spatial_neighbors_radius", {"radius": (90.0, 150.0)}),
    ("spatial_neighbors_grid", "spatial_neighbors_grid", {"n_neighs": 6}),
    ("spatial_neighbors_grid", "spatial_neighbors_grid", {"n_neighs": 6, "n_rings": 2}),
    ("spatial_neighbors_delaunay", "spatial_neighbors_delaunay", {}),
    ("spatial_neighbors_delaunay", "spatial_neighbors_delaunay", {"radius": 150.0}),
])
def test_matches_squidpy(ours, theirs, kw):
    import squidpy as sq
    a = lattice(25, jitter=1.0)
    o, t = a.copy(), a.copy()
    getattr(msc_gr, ours)(o, **kw)
    getattr(sq.gr, theirs)(t, **kw)

    A = o.obsp["spatial_connectivities"].tocsr()
    B = t.obsp["spatial_connectivities"].tocsr()
    coords = np.asarray(a.obsm["spatial"], dtype=np.float64)
    # exact, or differing only where two candidates are exactly equidistant
    assert _tie_free_differences(A, B, coords) == 0
    assert o.uns["spatial_neighbors"]["params"] == t.uns["spatial_neighbors"]["params"]

    if (abs(A - B) > 0).nnz == 0:              # same pattern -> distances must match too
        Do = o.obsp["spatial_distances"].tocsr(); Do.sort_indices()
        Dt = t.obsp["spatial_distances"].tocsr(); Dt.sort_indices()
        assert np.allclose(Do.data, Dt.data, atol=1e-6)
