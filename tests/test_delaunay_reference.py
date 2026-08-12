"""The reference triangulator: is the output a Delaunay triangulation of the input?

Correctness here is checked four ways, and the redundancy is deliberate. Each of these
caught a bug the others missed during the build:

* every internal edge is locally Delaunay (exact predicates);
* the triangle count is ``2n - 2 - h``, which notices triangles missing at the boundary
  where the local test has nothing to look at;
* the triangles' total area equals the convex hull's, which is the only check that sees a
  mesh that is locally perfect but self-intersecting;
* the edge set matches Qhull, wherever the point set has no cocircular quads to make the
  triangulation non-unique.
"""


import numpy as np
import pytest
from scipy.spatial import ConvexHull, Delaunay

from metalsinglecell._delaunay import reference as R
from metalsinglecell._delaunay.predicates import condition_points, incircle, orient2d


# ------------------------------------------------------------------------- utilities


def area_sum(pts, tri):
    p = pts.astype(np.float64)
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum()


def boundary_size(pts):
    """Points on the hull boundary, counting ones that lie flat on an edge."""
    hull = ConvexHull(pts.astype(np.float64)).vertices
    on = {int(v) for v in hull}
    rest = np.setdiff1d(np.arange(len(pts)), hull)
    ring = list(hull) + [hull[0]]
    for u, v in zip(ring[:-1], ring[1:]):
        if len(rest) == 0:
            break
        o = orient2d(pts, np.full(len(rest), u), np.full(len(rest), v), rest)
        on.update(int(x) for x in rest[o == 0])
    return len(on)


def edges_of(tri):
    out = {}
    for t, (i, j, k) in enumerate(tri):
        for e in ((i, j), (j, k), (k, i)):
            out.setdefault(tuple(sorted((int(e[0]), int(e[1])))), []).append(t)
    return out


def assert_valid_delaunay(raw, *, expect_qhull_parity=True):
    """Every check that matters, in one place."""
    ipts, _ = condition_points(np.asarray(raw, dtype=np.float64))
    tri = R.triangulate(raw)

    # 1. orientation: no inverted or zero-area triangles
    o = orient2d(ipts, tri[:, 0], tri[:, 1], tri[:, 2])
    assert (o > 0).all(), f"{int((o <= 0).sum())} triangles not counter-clockwise"

    # 2. every internal edge locally Delaunay
    em = edges_of(tri)
    quads = []
    for e, ts in em.items():
        assert len(ts) <= 2, f"edge {e} shared by {len(ts)} triangles"
        if len(ts) == 2:
            opp = [v for v in tri[ts[1]] if v not in e]
            quads.append([*tri[ts[0]], opp[0]])
    quads = np.array(quads, dtype=np.int64)
    s = incircle(ipts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    assert (s <= 0).all(), f"{int((s > 0).sum())} edges are not locally Delaunay"

    # 3. the right number of triangles
    expect = 2 * len(ipts) - 2 - boundary_size(ipts)
    assert len(tri) == expect, f"{len(tri)} triangles, expected {expect}"

    # 4. covers the hull exactly once — catches a self-intersecting mesh, which every
    #    local check above will happily pass
    hull_area = ConvexHull(ipts.astype(np.float64)).volume
    assert abs(area_sum(ipts, tri) - hull_area) / hull_area < 1e-12

    if expect_qhull_parity:
        assert (s == 0).sum() == 0, "point set has ties; parity cannot be expected"
        qh = Delaunay(ipts.astype(np.float64)).simplices
        mine = {e for e in edges_of(tri)}
        theirs = {e for e in edges_of(qh)}
        assert mine == theirs, f"{len(mine ^ theirs)} edges differ from Qhull"
    return tri


# ------------------------------------------------------------------------ the cases


def test_random_points_match_qhull_exactly():
    rng = np.random.default_rng(0)
    assert_valid_delaunay(rng.uniform(0, 5000, (1000, 2)))


def test_clustered_points_match_qhull_exactly():
    """Wide density variation — long hull edges bridging empty space."""
    rng = np.random.default_rng(1)
    c = rng.uniform(0, 6000, (20, 2))
    assert_valid_delaunay(np.vstack([q + rng.normal(0, 90, (200, 2)) for q in c]))


def test_hex_lattice_matches_qhull_exactly():
    """A regular lattice with no cocircular quads: parity is achievable and required."""
    g = np.arange(30)
    x, y = np.meshgrid(g * 100.0, g * 86.6)
    x = x + 50.0 * (np.arange(30)[:, None] % 2)
    assert_valid_delaunay(np.column_stack([x.ravel(), y.ravel()]))


def test_real_visium_geometry_matches_qhull_exactly():
    """The layout this is actually for. Falls back to a synthetic Visium-style grid."""
    try:
        import scanpy as sc
        a = sc.read_visium("data/V1_Breast_Cancer_Block_A_Section_1")
        pts = np.asarray(a.obsm["spatial"], dtype=np.float64)
    except Exception:
        rows, cols = np.mgrid[0:50, 0:40]
        pts = np.column_stack([(cols * 2 + rows % 2).ravel() * 137.0,
                               rows.ravel() * 154.0])
    assert_valid_delaunay(pts)


def test_square_lattice_is_valid_but_not_expected_to_match_qhull():
    """Where the triangulation is genuinely non-unique, only validity is meaningful.

    A third of the edges on a square lattice are cocircular, so both diagonals of those
    cells are Delaunay and no algorithm can be required to pick Qhull's.
    """
    g = np.arange(30)
    x, y = np.meshgrid(g * 100.0, g * 100.0)
    pts = np.column_stack([x.ravel(), y.ravel()])
    tri = assert_valid_delaunay(pts, expect_qhull_parity=False)

    ipts, _ = condition_points(pts)
    quads = []
    for e, ts in edges_of(tri).items():
        if len(ts) == 2:
            opp = [v for v in tri[ts[1]] if v not in e]
            quads.append([*tri[ts[0]], opp[0]])
    quads = np.array(quads, dtype=np.int64)
    s = incircle(ipts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    assert (s == 0).mean() > 0.3, "expected the lattice to be heavily cocircular"


def test_points_lying_exactly_on_an_edge_are_inserted():
    """Collinear triples put a point exactly on an edge; a 1->3 split would be degenerate.

    Three columns one unit apart make the middle point of many triples exactly collinear
    with its neighbours, which is what a binned assay produces.
    """
    pts = np.column_stack([np.arange(300) * 7.0, (np.arange(300) % 3) * 1.0])
    assert_valid_delaunay(pts)


def test_a_dense_clump_inside_a_sparse_field():
    """Density ratio large enough that hull edges dwarf the local spacing."""
    rng = np.random.default_rng(2)
    pts = np.vstack([rng.normal(0, 3, (900, 2)), rng.uniform(-800, 800, (100, 2))])
    assert_valid_delaunay(pts)


def test_minimal_and_small_inputs():
    tri = R.triangulate(np.array([[0.0, 0], [10, 0], [0, 10]]))
    assert len(tri) == 1
    tri = R.triangulate(np.array([[0.0, 0], [10, 0], [10, 10], [0, 10]]))
    assert len(tri) == 2
    with pytest.raises(ValueError, match="at least 3"):
        R.triangulate(np.array([[0.0, 0], [1, 1]]))


def test_collinear_input_is_rejected_clearly():
    pts = np.column_stack([np.arange(20) * 3.0, np.arange(20) * 6.0])
    with pytest.raises(ValueError, match="collinear"):
        R.triangulate(pts)


def test_duplicate_points_are_rejected_clearly():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 100, (50, 2))
    pts[7] = pts[3]
    with pytest.raises(ValueError, match="duplicate"):
        R.triangulate(pts)


def test_result_is_deterministic():
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 1000, (400, 2))
    a, b = R.triangulate(pts), R.triangulate(pts)
    assert np.array_equal(a, b)


# --------------------------------------------------------------- component behaviour


def test_hilbert_order_is_a_permutation_with_spatial_locality():
    rng = np.random.default_rng(5)
    ipts, _ = condition_points(rng.uniform(0, 1000, (2000, 2)))
    order = R.hilbert_order(ipts)
    assert np.array_equal(np.sort(order), np.arange(len(ipts)))

    step = np.linalg.norm(np.diff(ipts[order].astype(float), axis=0), axis=1).mean()
    rand = np.linalg.norm(np.diff(ipts.astype(float), axis=0), axis=1).mean()
    assert step < rand / 5, f"Hilbert step {step:.0f} vs unordered {rand:.0f}"


def test_ghost_triangles_partition_the_exterior():
    """Every point outside the hull belongs to exactly one ghost, not two.

    The half-plane test — "outside this hull edge" — puts a point beyond a hull *vertex*
    in two ghosts at once. Inserting it into the wrong one folds the boundary into a
    self-intersecting mesh that passes every local check.
    """
    rng = np.random.default_rng(6)
    ipts, _ = condition_points(rng.uniform(0, 1000, (60, 2)))
    n = len(ipts)
    inf = n
    mesh, _ = R._seed(ipts, inf)

    probes = np.arange(n)
    ghosts = np.where((mesh.tri == inf).any(axis=1))[0]
    counts = np.zeros(n, dtype=int)
    for g in ghosts:
        counts += R.tri_contains(ipts, mesh, np.full(n, g), probes, inf)

    outside = counts > 0
    assert outside.any(), "no probe points fall outside the seed triangle"
    assert (counts[outside] == 1).all(), (
        f"{int((counts > 1).sum())} points claimed by more than one ghost")


def test_flipping_leaves_exact_ties_alone():
    """A tie must not flip: every choice is Delaunay, and flipping on one can cycle."""
    pts = np.array([[0.0, 0], [100, 0], [100, 100], [0, 100]])
    ipts, _ = condition_points(pts)
    tri = R.triangulate(pts)
    assert len(tri) == 2

    quads = []
    for e, ts in edges_of(tri).items():
        if len(ts) == 2:
            opp = [v for v in tri[ts[1]] if v not in e]
            quads.append([*tri[ts[0]], opp[0]])
    q = np.array(quads, dtype=np.int64)
    s = incircle(ipts, q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    assert (s == 0).all(), "the square's diagonal should be an exact tie"


def test_check_mesh_catches_a_broken_link():
    """The integrity check has to actually fail on a broken mesh."""
    ipts, _ = condition_points(np.array([[0.0, 0], [100, 0], [50, 90], [50, 30]]))
    inf = len(ipts)
    mesh, _ = R._seed(ipts, inf)
    R.check_mesh(mesh, ipts, inf)                       # clean to start with

    mesh.nbr[0, 0] = mesh.nbr[0, 1]
    with pytest.raises(AssertionError, match="non-reciprocal|different edges"):
        R.check_mesh(mesh, ipts, inf)
