"""The exact predicates underneath the GPU Delaunay build.

Every test here checks a *sign*, never a magnitude, and the references are independent of
the formula under test wherever that is possible: the in-circle reference builds the
circumcircle in exact rationals and compares squared radii, which shares no algebra with
the determinant being tested.

Two of these tests exist specifically to not be vacuous. ``test_wide_operands_*`` asserts
that the unguarded 64-bit path would give the *wrong* answer on its inputs before checking
that the guarded one gives the right one — otherwise the fallback could be dead code and
the test would still pass.
"""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from metalsinglecell._delaunay import predicates as P


# --------------------------------------------------------------- independent references


def ref_incircle(pts, a, b, c, d):
    """Sign of "d is inside the circumcircle of abc", via an exact rational circumcentre.

    Shares no algebra with the determinant in :func:`predicates.incircle`: it solves for
    the centre from the two perpendicular-bisector equations and compares exact squared
    distances. Assumes ``abc`` counter-clockwise, matching the predicate's contract.
    """
    (ax, ay), (bx, by), (cx, cy) = (tuple(map(int, pts[i])) for i in (a, b, c))
    dx, dy = map(int, pts[d])
    # 2*[(bx-ax) (by-ay); (cx-ax) (cy-ay)] @ centre = [|b|^2-|a|^2; |c|^2-|a|^2]
    m00, m01 = 2 * (bx - ax), 2 * (by - ay)
    m10, m11 = 2 * (cx - ax), 2 * (cy - ay)
    det = m00 * m11 - m01 * m10
    if det == 0:
        raise ValueError("degenerate triangle has no circumcentre")
    r0 = (bx * bx + by * by) - (ax * ax + ay * ay)
    r1 = (cx * cx + cy * cy) - (ax * ax + ay * ay)
    ox = Fraction(r0 * m11 - m01 * r1, det)
    oy = Fraction(m00 * r1 - r0 * m10, det)
    rad2 = (ox - ax) ** 2 + (oy - ay) ** 2
    dist2 = (ox - dx) ** 2 + (oy - dy) ** 2
    return (dist2 < rad2) - (dist2 > rad2)      # inside -> +1, matches the determinant


def ref_orient(pts, a, b, c):
    (ax, ay), (bx, by), (cx, cy) = (tuple(map(int, pts[i])) for i in (a, b, c))
    det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (det > 0) - (det < 0)


def ccw(pts, quads):
    """Reorder each quad's first three indices counter-clockwise."""
    out = np.array(quads, dtype=np.int64, copy=True)
    s = P.orient2d(pts, out[:, 0], out[:, 1], out[:, 2])
    flip = s < 0
    out[flip, 0], out[flip, 2] = out[flip, 2], out[flip, 0].copy()
    return out


# ------------------------------------------------------------------------- correctness


def test_incircle_matches_rational_reference_on_random_points():
    rng = np.random.default_rng(0)
    pts = rng.integers(0, 4000, size=(400, 2)).astype(np.int64)
    pts = np.unique(pts, axis=0)
    quads = rng.integers(0, len(pts), size=(3000, 4))
    quads = quads[[len(set(q)) == 4 for q in quads]]
    quads = ccw(pts, quads)
    keep = P.orient2d(pts, quads[:, 0], quads[:, 1], quads[:, 2]) != 0
    quads = quads[keep]
    assert len(quads) > 1000

    got = P.incircle(pts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    want = np.array([ref_incircle(pts, *q) for q in quads], dtype=np.int8)
    assert np.array_equal(got, want)
    assert set(np.unique(got)) >= {-1, 1}       # both outcomes actually occur


def test_incircle_is_zero_on_exactly_cocircular_points():
    """Integer points on a circle of radius 25 — the degenerate case, not a near-miss."""
    circle = np.array([[25, 0], [24, 7], [20, 15], [15, 20], [0, 25], [-15, 20],
                       [-25, 0], [0, -25], [7, -24], [-20, -15]], dtype=np.int64) + 1000
    quads = np.array([[0, 1, 2, 3], [0, 2, 4, 6], [1, 3, 5, 7], [2, 4, 6, 8],
                      [0, 4, 8, 9]], dtype=np.int64)
    quads = ccw(circle, quads)
    got = P.incircle(circle, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    assert np.array_equal(got, np.zeros(len(quads), dtype=np.int8))


def test_incircle_is_zero_on_a_square_lattice_cell():
    """The 32.8%-cocircular case that makes lattice input the hard one."""
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int64)
    q = ccw(sq, np.array([[0, 1, 2, 3]]))
    assert P.incircle(sq, q[:, 0], q[:, 1], q[:, 2], q[:, 3])[0] == 0


def test_incircle_detects_a_one_unit_perturbation_off_the_circle():
    """A tie must be a true tie: moving the query point one lattice unit breaks it."""
    circle = np.array([[25, 0], [0, 25], [-25, 0], [24, 7]], dtype=np.int64) + 1000
    q = ccw(circle, np.array([[0, 1, 2, 3]]))
    assert P.incircle(circle, q[:, 0], q[:, 1], q[:, 2], q[:, 3])[0] == 0
    inward = circle.copy()
    inward[3] = [1024 - 1, 1007]                          # one unit toward the centre
    assert P.incircle(inward, q[:, 0], q[:, 1], q[:, 2], q[:, 3])[0] == 1
    outward = circle.copy()
    outward[3] = [1024 + 1, 1007]
    assert P.incircle(outward, q[:, 0], q[:, 1], q[:, 2], q[:, 3])[0] == -1


def test_orient2d_signs_and_collinearity():
    pts = np.array([[0, 0], [10, 0], [0, 10], [20, 0], [5, 0]], dtype=np.int64)
    assert P.orient2d(pts, [0], [1], [2])[0] == 1
    assert P.orient2d(pts, [0], [2], [1])[0] == -1
    assert P.orient2d(pts, [0], [1], [3])[0] == 0            # collinear
    assert P.orient2d(pts, [0], [4], [3])[0] == 0            # collinear, interior point

    rng = np.random.default_rng(1)
    rp = rng.integers(0, 10000, size=(200, 2)).astype(np.int64)
    tri = rng.integers(0, len(rp), size=(2000, 3))
    got = P.orient2d(rp, tri[:, 0], tri[:, 1], tri[:, 2])
    want = np.array([ref_orient(rp, *t) for t in tri], dtype=np.int8)
    assert np.array_equal(got, want)


# ------------------------------------------------- the guard, and that it is not dead


def test_wide_operands_would_overflow_but_the_guard_catches_them():
    """Beyond SAFE_ABS the naive int64 determinant is wrong; the predicate still is not.

    The first assertion is the point of the test: it fails if the inputs are not actually
    wide enough to overflow, which would make the fallback dead code.
    """
    m = P.SAFE_ABS * 64                                   # 2**20, degree-4 -> ~2**82
    pts = np.array([[m, 0], [0, m], [-m, 0], [1, 1]], dtype=np.int64)
    a, b, c, d = [0], [1], [2], [3]

    ax, ay = pts[0] - pts[3]
    bx, by = pts[1] - pts[3]
    cx, cy = pts[2] - pts[3]
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    with np.errstate(over="ignore"):
        naive = np.int64(ax) * (np.int64(by) * cc - np.int64(bb) * cy) \
            - np.int64(ay) * (np.int64(bx) * cc - np.int64(bb) * cx) \
            + np.int64(aa) * (np.int64(bx) * cy - np.int64(by) * cx)
    truth = ref_incircle(pts, 0, 1, 2, 3)
    assert np.sign(int(naive)) != truth, "inputs are not wide enough to overflow int64"

    assert P.incircle(pts, a, b, c, d)[0] == truth


def test_wide_and_narrow_operands_mix_in_one_call():
    """The fallback is applied per row, not per call."""
    m = P.SAFE_ABS * 64
    pts = np.array([[m, 0], [0, m], [-m, 0], [1, 1],
                    [1000, 0], [0, 1000], [-1000, 0], [3, 5]], dtype=np.int64)
    quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    quads = ccw(pts, quads)
    got = P.incircle(pts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    want = np.array([ref_incircle(pts, *q) for q in quads], dtype=np.int8)
    assert np.array_equal(got, want)


# ------------------------------------------------------------------------ conditioning


def test_condition_points_is_lossless_at_assay_resolution():
    rng = np.random.default_rng(2)
    pts = rng.normal(0, 1, size=(2000, 2)) * 500 + 20000     # Xenium-like microns
    ipts, info = P.condition_points(pts)

    assert ipts.dtype == np.int64
    assert info["n_distinct"] == len(pts), "conditioning collapsed distinct points"
    assert np.log2(info["scale"]) == int(np.log2(info["scale"])), "scale is a power of two"
    assert info["max_shift"] < info["spacing"] / 100, "snap moved a point >1% of a spacing"


def test_condition_points_keeps_a_visium_style_lattice_exact():
    """Integer lattice input should snap to an exact rescaling, with no shape change."""
    x, y = np.meshgrid(np.arange(40) * 137, np.arange(40) * 154)
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    ipts, info = P.condition_points(pts)
    assert info["n_distinct"] == len(pts)
    assert info["max_shift"] == 0.0
    # a rescaled lattice is still a lattice: differences remain exact multiples
    dx = np.unique(np.diff(np.unique(ipts[:, 0])))
    assert len(dx) == 1


def test_conditioned_coordinates_stay_inside_the_safe_window_for_local_triangles():
    """The scale is only useful if real neighbouring points stay under SAFE_ABS apart."""
    rng = np.random.default_rng(3)
    for n in (1000, 50000):
        pts = rng.uniform(0, 1, size=(n, 2)) * 10000
        ipts, info = P.condition_points(pts)
        from scipy.spatial import cKDTree
        d, idx = cKDTree(ipts.astype(np.float64)).query(ipts.astype(np.float64), k=8)
        assert d[:, 1:].max() < P.SAFE_ABS, (
            f"n={n}: 8th-neighbour span {d[:, 1:].max():.0f} exceeds SAFE_ABS")


# -------------------------------------------------------------------------------- GPU


def test_gpu_incircle_matches_the_cpu_predicate_exactly():
    mx = pytest.importorskip("mlx.core")
    del mx
    rng = np.random.default_rng(4)
    pts = np.unique(rng.integers(0, 4000, size=(500, 2)).astype(np.int64), axis=0)
    quads = rng.integers(0, len(pts), size=(5000, 4))
    quads = ccw(pts, quads[[len(set(q)) == 4 for q in quads]])

    cpu = P.incircle(pts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    gpu = P.incircle_gpu(pts, quads)
    assert np.array_equal(cpu, gpu)


def test_gpu_incircle_takes_the_128_bit_path_for_wide_operands():
    """The wide branch must be exact, and must actually be the branch under test.

    ``m`` is chosen at the very top of the conditioned range, where the determinant needs
    ~123 bits — so a 64-bit evaluation is not merely imprecise, it wraps.
    """
    pytest.importorskip("mlx.core")
    m = P.MAX_COORD // 2
    pts = np.array([[m, 0], [0, m], [-m, 0], [1, 1],
                    [1000, 0], [0, 1000], [-1000, 0], [3, 5]], dtype=np.int64)
    quads = ccw(pts, np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64))

    widest = np.abs(np.stack([pts[quads[:, k]] - pts[quads[:, 3]]
                              for k in (0, 1, 2)])).max(axis=(0, 2))
    assert widest[0] > P.SAFE_ABS, "first quad does not exercise the 128-bit path"
    assert widest[1] <= P.SAFE_ABS, "second quad does not exercise the 64-bit path"

    want = np.array([ref_incircle(pts, *q) for q in quads], dtype=np.int8)
    got = P.incircle_gpu(pts, quads)
    assert np.array_equal(got, want)
    assert set(np.unique(got)) <= {-1, 0, 1}, "sentinel leaked into the result"


def test_gpu_128_bit_path_resolves_a_tie_at_full_coordinate_range():
    """At MAX_COORD, one lattice unit still decides the sign — nothing is being rounded."""
    pytest.importorskip("mlx.core")
    m = P.MAX_COORD // 2
    base = np.array([[m, 0], [0, m], [-m, 0], [0, -m]], dtype=np.int64)
    q = ccw(base, np.array([[0, 1, 2, 3]]))
    assert P.incircle_gpu(base, q)[0] == 0                       # exactly cocircular

    for delta, expect in ((-1, 1), (1, -1)):                     # inward / outward
        pts = base.copy()
        pts[3] = [0, -(m + delta)]
        assert P.incircle_gpu(pts, q)[0] == expect
        assert P.incircle(pts, q[:, 0], q[:, 1], q[:, 2], q[:, 3])[0] == expect


def test_gpu_128_bit_path_fuzzed_against_python_integers():
    """The 128-bit determinant, over the whole conditioned range, against arbitrary precision.

    Handwritten cases only probe where the author thought to look. This throws 4000 random
    wide quads at it — including deliberately near-degenerate ones, built by placing the
    fourth point a few units off the circumcircle through the other three — and demands
    agreement with Python integers on every single sign.
    """
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(7)
    m = P.MAX_COORD // 2

    pts = rng.integers(-m, m, size=(4000, 2)).astype(np.int64)
    # half the quads near-degenerate: a point nudged onto a circle through three others
    for i in range(0, 2000, 4):
        centre = rng.integers(-m // 2, m // 2, size=2)
        r = int(rng.integers(1000, m // 2))
        for j, ang in enumerate(rng.uniform(0, 2 * np.pi, 4)):
            off = int(rng.integers(-2, 3))               # -2..2 units off the circle
            pts[i + j] = centre + np.rint([(r + off) * np.cos(ang),
                                           (r + off) * np.sin(ang)]).astype(np.int64)
    pts = np.clip(pts, -m, m)

    quads = rng.integers(0, len(pts), size=(6000, 4))
    quads = np.array([q for q in quads if len(set(q)) == 4], dtype=np.int64)
    quads = np.vstack([quads, np.arange(2000).reshape(-1, 4)])
    quads = ccw(pts, quads)
    quads = quads[P.orient2d(pts, quads[:, 0], quads[:, 1], quads[:, 2]) != 0]
    assert len(quads) > 4000

    widest = np.abs(np.stack([pts[quads[:, k]] - pts[quads[:, 3]]
                              for k in (0, 1, 2)])).max(axis=(0, 2))
    wide = widest > P.SAFE_ABS
    assert wide.mean() > 0.9, "fuzz did not exercise the 128-bit path"

    want = P._incircle_bigint(pts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    got = P.incircle_gpu(pts, quads)
    bad = np.where(got != want)[0]
    assert len(bad) == 0, f"{len(bad)} sign disagreements, first at quad {quads[bad[:1]]}"

    # Exact ties are not reachable by this construction — rounding onto a circle of
    # radius ~2**29 never lands on one — so the zero case is covered by the constructed
    # cocircular tests instead, and by test_fp32_cannot_do_this_job.


def naive_f32_det(pts, quads):
    """The in-circle determinant as an fp32 kernel would compute it.

    Not ``np.linalg.det`` on a float32 matrix — that pivots, which is far more accurate
    than any kernel, and measuring it instead reports a misleading 0% error rate.
    """
    f32 = np.float32
    d = pts[quads[:, 3]]
    ax, ay = f32(pts[quads[:, 0], 0] - d[:, 0]), f32(pts[quads[:, 0], 1] - d[:, 1])
    bx, by = f32(pts[quads[:, 1], 0] - d[:, 0]), f32(pts[quads[:, 1], 1] - d[:, 1])
    cx, cy = f32(pts[quads[:, 2], 0] - d[:, 0]), f32(pts[quads[:, 2], 1] - d[:, 1])
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    det = ax * (by * cc - bb * cy) - ay * (bx * cc - bb * cx) + aa * (bx * cy - by * cx)
    assert det.dtype == np.float32
    return det


def test_fp32_cannot_recognise_a_tie():
    """Why the kernel is integer and not float — the alternative Metal offers cannot do it.

    Metal has no fp64, so a float kernel means fp32, whose 24-bit mantissa cannot even
    represent a conditioned coordinate above 2**24. What matters is not accuracy in
    general — on generic point clouds the naive fp32 expansion got 0 of ~700k signs wrong
    — but whether it can return *exactly zero* when four points are exactly cocircular.
    It cannot: below, it reports a confident sign on most of them. A flipping loop that
    cannot see a tie can flip back and forth across it forever, and ties are 32.8% of the
    edges on a square lattice.
    """
    # exact lattice points on a circle of radius 456675375 (= 1105 * 413281, and 1105**2
    # has many representations as a sum of two squares), scaled to the top of the range
    pts = np.array([[0, -456675375], [-274005225, -365340300],
                    [-274005225, 365340300], [127869105, -438408360]], dtype=np.int64)
    assert np.abs(pts).max() > 2 ** 24, "coordinates must exceed the fp32 integer range"
    assert np.abs(pts).max() <= P.MAX_COORD

    q = ccw(pts, np.array([[0, 1, 2, 3]]))
    assert P.incircle(pts, q[:, 0], q[:, 1], q[:, 2], q[:, 3])[0] == 0
    assert P.incircle_gpu(pts, q)[0] == 0
    assert np.sign(naive_f32_det(pts, q))[0] != 0

    # not a cherry-picked quad: most cocircular sets on this circle defeat fp32
    ring = np.array([[0, -456675375], [-274005225, -365340300], [-274005225, 365340300],
                     [127869105, -438408360], [456675375, 0], [365340300, 274005225],
                     [-438408360, 127869105], [-127869105, 438408360]], dtype=np.int64)
    quads = np.array(list(combinations(range(8), 4)))
    quads = ccw(ring, quads)
    quads = quads[P.orient2d(ring, quads[:, 0], quads[:, 1], quads[:, 2]) != 0]
    exact = P.incircle(ring, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    assert (exact == 0).all(), "test points are not all cocircular"
    missed = (np.sign(naive_f32_det(ring, quads)) != 0).mean()
    assert missed > 0.5, f"fp32 recognised {1 - missed:.0%} of these ties"
    assert np.array_equal(P.incircle_gpu(ring, quads), exact)


def test_gpu_incircle_rejects_unconditioned_coordinates():
    pytest.importorskip("mlx.core")
    pts = np.array([[P.MAX_COORD * 4, 0], [0, 1], [1, 0], [2, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="condition_points"):
        P.incircle_gpu(pts, np.array([[0, 1, 2, 3]]))


def test_gpu_incircle_handles_an_empty_batch():
    pytest.importorskip("mlx.core")
    pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int64)
    assert len(P.incircle_gpu(pts, np.zeros((0, 4), dtype=np.int64))) == 0


# ----------------------------------------------------------- against a real triangulation


def flip_quads(tri):
    """One ``[a, b, c, opposite]`` row per internal edge of a triangulation."""
    edges = {}
    for t, (i, j, k) in enumerate(tri):
        for e in ((i, j), (j, k), (k, i)):
            edges.setdefault(tuple(sorted(e)), []).append(t)
    out = []
    for e, ts in edges.items():
        if len(ts) != 2:
            continue
        opp = [v for v in tri[ts[1]] if v not in e]
        out.append([*tri[ts[0]], opp[0]])
    return np.array(out, dtype=np.int64)


def test_qhull_triangulation_is_certified_delaunay_by_these_predicates():
    """End-to-end: the predicates must agree that Qhull's own output is Delaunay.

    This is the acceptance criterion the GPU triangulation will be held to, run against a
    triangulation we know is correct — if it failed here, the predicates would be wrong.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 1, size=(3000, 2)) * 8000
    ipts, _ = P.condition_points(pts)
    tri = Delaunay(ipts.astype(np.float64)).simplices

    tri = ccw(ipts, np.column_stack([tri, tri[:, 0]]))[:, :3]
    assert (P.orient2d(ipts, tri[:, 0], tri[:, 1], tri[:, 2]) > 0).all()

    quads = flip_quads(tri)
    assert len(quads) > 8000
    s = P.incircle(ipts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    assert (s <= 0).all(), f"{(s > 0).sum()} edges are not locally Delaunay"
    assert np.array_equal(s, P.incircle_gpu(ipts, quads))


def test_conditioning_moves_only_a_tiny_fraction_of_near_ties():
    """What conditioning costs, measured — and it is not zero.

    Snapping perturbs points by up to ~0.1% of a spacing, which is enough to tip quads
    that were within ~1e-13 of cocircular in the float input. Those edges flip, so a
    triangulation of the conditioned points is *not* edge-identical to Qhull's on the
    original floats. It is a valid Delaunay triangulation of the points we were given to
    the resolution the assay can distinguish, which is the contract we can actually keep.
    This test pins the rate so a regression in the scale rule shows up as a number.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 1, size=(3000, 2)) * 8000
    ipts, info = P.condition_points(pts)

    tri = ccw(ipts, np.column_stack([Delaunay(pts).simplices,
                                     Delaunay(pts).simplices[:, 0]]))[:, :3]
    quads = flip_quads(tri)
    s = P.incircle(ipts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    tipped = int((s > 0).sum())
    assert tipped / len(quads) < 0.005, f"{tipped}/{len(quads)} edges tipped by the snap"

    # and every one of them was a near-tie in the original geometry, not a real error
    for i in np.where(s > 0)[0]:
        a, b, c, d = quads[i]
        m = np.array([[*(pts[j] - pts[d]), ((pts[j] - pts[d]) ** 2).sum()]
                      for j in (a, b, c)])
        assert abs(np.linalg.det(m)) / np.abs(m).max() ** 4 < 1e-11
    assert info["max_shift"] < info["spacing"] / 100
