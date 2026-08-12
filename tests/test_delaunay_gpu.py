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

from metalsinglecell._delaunay.gpu import (                    # noqa: E402
    flip_candidates, flip_round as gpu_flip_round,
)


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
                # and the applied mesh, not just the selection
                gt, gn, gk, _ = gpu_flip_round(pts, mesh.tri, mesh.nbr, inf)
                compared += 1
            k = R._flip_round(pts, mesh, loc, active, inf)
            if compare_every_round:
                assert np.array_equal(gt, mesh.tri), f"round {compared}: vertices differ"
                assert np.array_equal(gn, mesh.nbr), f"round {compared}: adjacency differs"
                assert gk == k, f"round {compared}: {gk} flips vs {k}"
            if k == 0:
                break
    return compared


def test_selection_matches_numpy_every_round_on_random_points():
    rng = np.random.default_rng(0)
    compared = run_to_completion(rng.uniform(0, 5000, (1500, 2)))
    assert compared > 20, f"only {compared} selections compared"


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


def test_end_to_end_backends_agree_on_a_tie_heavy_lattice():
    """Where the answer is not unique, "both valid" is not good enough.

    A square lattice admits many Delaunay triangulations, so the backends can drift apart
    while both stay correct — and they did, until the GPU path re-homed points the same
    way the CPU path does rather than by walking. Equality here is what pins that down.
    """
    g = np.arange(40)
    x, y = np.meshgrid(g * 8.0, g * 8.0)               # Visium HD 8um bin spacing
    pts = np.column_stack([x.ravel(), y.ravel()])
    assert np.array_equal(R.triangulate(pts, backend="cpu"),
                          R.triangulate(pts, backend="gpu"))


def test_end_to_end_backends_agree_on_clustered_geometry():
    rng = np.random.default_rng(7)
    c = rng.uniform(0, 6000, (12, 2))
    pts = np.vstack([q + rng.normal(0, 90, (120, 2)) for q in c])
    assert np.array_equal(R.triangulate(pts, backend="cpu"),
                          R.triangulate(pts, backend="gpu"))


def test_flip_apply_reports_the_pairs_it_flipped():
    """The caller re-homes points from the flipped pairs, so the pairing has to be right."""
    rng = np.random.default_rng(8)
    ipts, _ = condition_points(rng.uniform(0, 2000, (300, 2)))
    inf = len(ipts)
    mesh, seed = R._seed(ipts, inf)
    loc = np.zeros(len(ipts), dtype=np.int64)
    active = np.ones(len(ipts), dtype=bool)
    active[list(seed)] = False
    R._walk(ipts, mesh, np.where(active)[0], loc, inf)
    st, sp, _ = R._choose_one_per_triangle(ipts, mesh, np.where(active)[0], loc, inf)
    R._split(ipts, mesh, st, sp, loc, active, inf)

    _, _, k, (ft, fu) = gpu_flip_round(ipts, mesh.tri, mesh.nbr, inf)
    assert k > 0, "this configuration should need flips"
    assert len(ft) == len(fu) == k
    assert len(np.intersect1d(ft, fu)) == 0, "a triangle is in two flips"
    assert len(np.unique(np.concatenate([ft, fu]))) == 2 * k, "duplicate triangles"
    # each reported pair really is adjacent
    assert (mesh.nbr[ft].reshape(len(ft), 3) // 3 == fu[:, None]).any(axis=1).all()


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
    mesh, poly = R._seed(ipts, inf)
    ghosts = ((mesh.tri == inf).any(axis=1)).sum()
    assert ghosts == len(poly) >= 3, "one ghost per edge of the seed hull"
    # the ghost ring's ghost-ghost edges must be scanned, not skipped
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
