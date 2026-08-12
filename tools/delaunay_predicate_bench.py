"""Does the exact-integer predicate design hold up on real data, and is it fast enough?

Three questions, all of which have to be answered before writing the triangulation on top
of these predicates:

1. **Does conditioning stay lossless?** Distinct points must stay distinct and the snap
   must move nothing by a biologically meaningful amount.
2. **How often do operands exceed SAFE_ABS?** Every one of those costs a host round trip
   into Python integers. If the rate is non-trivial on real geometry, the 64-bit path is
   the wrong design and a 128-bit MSL fallback has to be written.
3. **Is the GPU predicate actually faster than the CPU one?** A predicate that loses to
   NumPy is not worth putting in a kernel — the lesson from the first spatial-neighbours
   cut, where scipy plumbing, not arithmetic, was the whole cost.

and a fourth that decides whether the exact machinery is needed at all: **would fp32 do?**
Metal has no fp64, so fp32 is the only float alternative. The answer is subtle enough to
be worth measuring rather than asserting — see the ``f32 bad`` column and the tie test at
the bottom.

Run: ``python tools/delaunay_predicate_bench.py``
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.spatial import Delaunay

from metalsinglecell._delaunay import predicates as P

warnings.simplefilter("ignore")


def flip_quads(tri):
    edges = {}
    for t, (i, j, k) in enumerate(tri):
        for e in ((i, j), (j, k), (k, i)):
            edges.setdefault(tuple(sorted(e)), []).append(t)
    out = []
    for e, ts in edges.items():
        if len(ts) == 2:
            opp = [v for v in tri[ts[1]] if v not in e]
            out.append([*tri[ts[0]], opp[0]])
    return np.array(out, dtype=np.int64)


def ccw(pts, quads):
    out = np.array(quads, dtype=np.int64, copy=True)
    flip = P.orient2d(pts, out[:, 0], out[:, 1], out[:, 2]) < 0
    out[flip, 0], out[flip, 2] = out[flip, 2], out[flip, 0].copy()
    return out


def point_sets():
    rng = np.random.default_rng(0)
    yield "random 5k", rng.uniform(0, 1, (5000, 2)) * 5000
    yield "random 100k", rng.uniform(0, 1, (100000, 2)) * 20000

    g = np.arange(80)
    x, y = np.meshgrid(g * 100.0, g * 100.0)
    yield "square lattice 6.4k", np.column_stack([x.ravel(), y.ravel()])

    g = np.arange(80)
    x, y = np.meshgrid(g * 100.0, g * 86.6)
    x = x + (np.arange(80)[None, :] * 0 + 50.0) * (np.arange(80)[:, None] % 2)
    yield "hex lattice 6.4k", np.column_stack([x.ravel(), y.ravel()])

    # Xenium-like: clustered, float microns, wide dynamic range
    centres = rng.uniform(0, 6000, (60, 2))
    pts = np.vstack([c + rng.normal(0, 90, (2000, 2)) for c in centres])
    yield "clustered 120k", pts

    try:
        import scanpy as sc
        a = sc.read_visium("data/V1_Breast_Cancer_Block_A_Section_1")
        yield "real Visium 3.8k", np.asarray(a.obsm["spatial"], dtype=np.float64)
    except Exception as exc:                                  # data not mounted
        print(f"  (skipping real Visium: {exc})")


def naive_f32_sign(pts, q):
    """The determinant as an fp32 kernel would evaluate it — naive expansion, no pivoting.

    ``np.linalg.det`` on a float32 matrix is not this: LAPACK pivots, which is far more
    accurate than a kernel can be, and measuring it reports a misleadingly clean 0%.
    """
    f32 = np.float32
    d = pts[q[:, 3]]
    ax, ay = f32(pts[q[:, 0], 0] - d[:, 0]), f32(pts[q[:, 0], 1] - d[:, 1])
    bx, by = f32(pts[q[:, 1], 0] - d[:, 0]), f32(pts[q[:, 1], 1] - d[:, 1])
    cx, cy = f32(pts[q[:, 2], 0] - d[:, 0]), f32(pts[q[:, 2], 1] - d[:, 1])
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    det = ax * (by * cc - bb * cy) - ay * (bx * cc - bb * cx) + aa * (bx * cy - by * cx)
    return np.sign(det).astype(np.int8)


print(f"{'point set':<22} {'n':>8} {'scale':>8} {'shift/spc':>10} {'dupes':>6} "
      f"{'>SAFE':>8} {'cocirc':>8} {'f32 bad':>8} {'cpu ms':>8} {'gpu ms':>8} {'speedup':>8}")
print("-" * 118)

for name, pts in point_sets():
    n = len(pts)
    ipts, info = P.condition_points(pts)
    dupes = n - info["n_distinct"]
    rel_shift = info["max_shift"] / info["spacing"] if info["spacing"] else 0.0

    tri = Delaunay(ipts.astype(np.float64)).simplices
    quads = ccw(ipts, flip_quads(tri))

    # how wide do the operands actually get on a real triangulation?
    d = ipts[quads[:, 3]]
    widest = np.abs(np.stack([ipts[quads[:, k]] - d for k in (0, 1, 2)])).max(axis=(0, 2))
    over = int((widest > P.SAFE_ABS).sum())

    t0 = time.perf_counter()
    cpu = P.incircle(ipts, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])
    t_cpu = (time.perf_counter() - t0) * 1e3

    P.incircle_gpu(ipts, quads[:16])                          # warm the kernel cache
    t0 = time.perf_counter()
    gpu = P.incircle_gpu(ipts, quads)
    t_gpu = (time.perf_counter() - t0) * 1e3

    assert np.array_equal(cpu, gpu), f"{name}: GPU and CPU predicates disagree"
    assert (cpu <= 0).all(), f"{name}: Qhull output judged non-Delaunay — predicate bug"

    f32_bad = int((naive_f32_sign(ipts, quads) != cpu).sum())

    print(f"{name:<22} {n:>8} {info['scale']:>8.4g} {rel_shift:>10.2e} {dupes:>6} "
          f"{over:>8} {int((cpu == 0).sum()):>8} {f32_bad:>8} {t_cpu:>8.1f} "
          f"{t_gpu:>8.1f} {t_cpu / t_gpu:>7.2f}x")

# Would fp32 have been good enough? On generic geometry, apparently yes — which is why
# this has to be tested on ties specifically, the case a flipping loop actually lives in.
ring = np.array([[0, -456675375], [-274005225, -365340300], [-274005225, 365340300],
                 [127869105, -438408360], [456675375, 0], [365340300, 274005225],
                 [-438408360, 127869105], [-127869105, 438408360]], dtype=np.int64)
from itertools import combinations                                       # noqa: E402

tq = ccw(ring, np.array(list(combinations(range(8), 4))))
tq = tq[P.orient2d(ring, tq[:, 0], tq[:, 1], tq[:, 2]) != 0]
exact = P.incircle(ring, tq[:, 0], tq[:, 1], tq[:, 2], tq[:, 3])
assert (exact == 0).all()
missed = int((naive_f32_sign(ring, tq) != 0).sum())
print(f"\nexactly cocircular quads (lattice points on one circle): {len(tq)}")
print(f"  fp32 claims a definite inside/outside on {missed} of them "
      f"({100 * missed / len(tq):.0f}%) — it cannot see a tie")
print(f"  the exact predicate returns 0 on all {len(tq)}")

print("""
Columns
  shift/spc  largest snap displacement, as a fraction of the median point spacing
  dupes      distinct input points that collided on the lattice (must be 0)
  >SAFE      quads too wide for a 64-bit determinant; they take the 128-bit branch
  cocirc     exactly cocircular quads — genuine ties, where either diagonal is Delaunay
  f32 bad    signs a naive fp32 kernel would get wrong on this point set
""")
