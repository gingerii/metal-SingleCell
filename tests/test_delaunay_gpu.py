"""The Metal flip round must select exactly what the NumPy one selects.

Not "an equivalent set" — the same set. The predicate is exact, the visit rule and the
tie-break are the same, and the conflict resolution is the same scatter-min, so any
difference is a bug rather than a permissible variation. Testing for equality rather than
for "also produces a valid triangulation" is what makes these tests able to fail.
"""

import numpy as np
import pytest

from metalsinglecell._delaunay import reference as R
from metalsinglecell._delaunay.predicates import MAX_COORD, condition_points

pytest.importorskip("mlx.core")

from metalsinglecell._delaunay.gpu import flip_candidates      # noqa: E402


def run_to_completion(raw, compare_every_round=True):
    """Drive the algorithm by hand so both selections can be compared every round."""
    ipts, _ = condition_points(np.asarray(raw, dtype=np.float64))
    order = R.hilbert_order(ipts)
    pts = ipts[order]
    n = len(pts)
    inf = n
    mesh, seed = R._seed(pts, inf)
    loc = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    active[list(seed)] = False
    R._walk(pts, mesh, np.where(active)[0], loc, inf)

    compared = 0
    while active.any():
        idx = np.where(active)[0]
        es = R._edge_slot(pts, mesh, loc[idx], idx, inf)
        interior = idx[es < 0]
        if len(interior):
            st, sp, _ = R._choose_one_per_triangle(pts, mesh, interior, loc, inf)
            R._split(pts, mesh, st, sp, loc, active, inf)
        else:
            oe = idx[es >= 0]
            st, sp, pos = R._choose_one_per_triangle(pts, mesh, oe, loc, inf)
            R._edge_split(pts, mesh, st, es[es >= 0][pos], sp, loc, active, inf)
        while True:
            if compare_every_round:
                cpu = np.sort(R.select_flips(pts, mesh, inf))
                gpu = np.sort(flip_candidates(pts, mesh.tri, mesh.nbr, inf))
                assert np.array_equal(cpu, gpu), (
                    f"round {compared}: {len(np.setxor1d(cpu, gpu))} half-edges differ")
                compared += 1
            if R._flip_round(pts, mesh, loc, active, inf) == 0:
                break
    return compared


def test_selection_matches_numpy_every_round_on_random_points():
    rng = np.random.default_rng(0)
    compared = run_to_completion(rng.uniform(0, 5000, (1500, 2)))
    assert compared > 50, f"only {compared} selections compared"


def test_selection_matches_numpy_on_a_tie_heavy_lattice():
    """The lattice is where the ghost and tie branches of the kernel get exercised."""
    g = np.arange(25)
    x, y = np.meshgrid(g * 100.0, g * 100.0)
    compared = run_to_completion(np.column_stack([x.ravel(), y.ravel()]))
    assert compared > 20


def test_selection_matches_numpy_on_clustered_points():
    """Long hull edges here push in-circle operands onto the 128-bit branch."""
    rng = np.random.default_rng(1)
    c = rng.uniform(0, 6000, (15, 2))
    run_to_completion(np.vstack([q + rng.normal(0, 90, (150, 2)) for q in c]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_end_to_end_backends_agree_exactly(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 3000, (800, 2))
    assert np.array_equal(R.triangulate(pts, backend="cpu"),
                          R.triangulate(pts, backend="gpu"))


def test_gpu_backend_produces_a_valid_triangulation_on_visium_like_geometry():
    rows, cols = np.mgrid[0:40, 0:30]
    pts = np.column_stack([(cols * 2 + rows % 2).ravel() * 137.0, rows.ravel() * 154.0])
    ipts, _ = condition_points(pts)
    tri = R.triangulate(pts, backend="gpu")

    from scipy.spatial import ConvexHull
    p = ipts.astype(np.float64)
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum()
    hull = ConvexHull(p).volume
    assert abs(area - hull) / hull < 1e-12


def test_ghost_and_tie_branches_are_actually_reached():
    """Guard against the kernel's interesting branches being dead in these tests."""
    g = np.arange(12)
    x, y = np.meshgrid(g * 100.0, g * 100.0)
    ipts, _ = condition_points(np.column_stack([x.ravel(), y.ravel()]))
    inf = len(ipts)
    mesh, _ = R._seed(ipts, inf)
    assert ((mesh.tri == inf).any(axis=1)).sum() == 3, "seed should have three ghosts"
    # the seed's ghost-ghost edges must be scanned, not skipped
    assert np.array_equal(np.sort(R.select_flips(ipts, mesh, inf)),
                          np.sort(flip_candidates(ipts, mesh.tri, mesh.nbr, inf)))


def test_rejects_unconditioned_coordinates():
    ipts = np.array([[MAX_COORD * 4, 0], [0, 1], [1, 0]], dtype=np.int64)
    inf = 3
    mesh, _ = R._seed(np.array([[0, 0], [10, 0], [0, 10]], dtype=np.int64), inf)
    with pytest.raises(ValueError, match="condition_points"):
        flip_candidates(ipts, mesh.tri, mesh.nbr, inf)


def test_empty_mesh_is_handled():
    assert len(flip_candidates(np.zeros((0, 2), dtype=np.int64),
                               np.zeros((0, 3), dtype=np.int64),
                               np.zeros((0, 3), dtype=np.int64), 0)) == 0
