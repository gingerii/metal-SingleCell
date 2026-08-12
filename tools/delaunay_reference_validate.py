"""Validate the reference triangulator against Qhull and the exact oracle.

Four independent checks, because no one of them is sufficient — this is the lesson of the
build, not a precaution:

* **Locally Delaunay** (`tools/delaunay_oracle.py`, exact integers). Necessary, and the
  obvious check, but it only inspects edges shared by two triangles.
* **Triangle count** ``2n - 2 - h``. Catches missing triangles that the local test cannot
  see because the edges it would have to look at are on the boundary.
* **Area equals the convex hull's.** The one that actually caught the worst bug: a mesh
  can be locally consistent, fully linked, with every triangle counter-clockwise, and
  still self-intersect — covering part of the plane twice and part not at all. Nothing
  local notices. The area does.
* **Edge parity with Qhull**, where the point set has no cocircular quads. Where it does
  (square lattices), parity is impossible in principle and only validity is required.

Run: ``python tools/delaunay_reference_validate.py``
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

sys.path.insert(0, str(Path(__file__).parent))
from delaunay_oracle import check_delaunay, edge_set        # noqa: E402

from metalsinglecell._delaunay import reference as R        # noqa: E402
from metalsinglecell._delaunay.predicates import condition_points, orient2d  # noqa: E402

warnings.simplefilter("ignore")


def triangle_area_sum(pts, tri):
    p = pts.astype(np.float64)
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum()


def boundary_points(pts):
    """Every point on the convex hull boundary, including ones collinear on an edge.

    ``ConvexHull.vertices`` omits collinear points, and the triangle count depends on the
    full boundary set, so counting corners would make a correct result look wrong.
    """
    hull = ConvexHull(pts.astype(np.float64)).vertices
    on = set(int(v) for v in hull)
    ring = list(hull) + [hull[0]]
    rest = np.setdiff1d(np.arange(len(pts)), hull)
    for u, v in zip(ring[:-1], ring[1:]):
        if len(rest) == 0:
            break
        flat = rest[orient2d(pts, np.full(len(rest), u), np.full(len(rest), v), rest) == 0]
        on.update(int(x) for x in flat)
    return on


def point_sets():
    rng = np.random.default_rng(0)
    yield "random 1k", rng.uniform(0, 5000, (1000, 2))
    yield "random 10k", rng.uniform(0, 40000, (10000, 2))

    g = np.arange(40)
    x, y = np.meshgrid(g * 100.0, g * 100.0)
    yield "square lattice 1.6k", np.column_stack([x.ravel(), y.ravel()])

    xh, yh = np.meshgrid(g * 100.0, g * 86.6)
    xh = xh + 50.0 * (np.arange(40)[:, None] % 2)
    yield "hex lattice 1.6k", np.column_stack([xh.ravel(), yh.ravel()])

    c = rng.uniform(0, 6000, (20, 2))
    yield "clustered 4k", np.vstack([q + rng.normal(0, 90, (200, 2)) for q in c])

    yield "near-collinear 500", np.column_stack([np.arange(500) * 7.0,
                                                 (np.arange(500) % 3) * 1.0])
    yield "one dense clump", np.vstack([rng.normal(0, 3, (900, 2)),
                                        rng.uniform(-800, 800, (100, 2))])
    try:
        import scanpy as sc
        a = sc.read_visium("data/V1_Breast_Cancer_Block_A_Section_1")
        yield "real Visium 3.8k", np.asarray(a.obsm["spatial"], dtype=np.float64)
    except Exception as exc:
        print(f"  (skipping real Visium: {exc})")


print(f"{'point set':<22} {'n':>7} {'tris':>7} {'expect':>7} {'delaunay':>9} "
      f"{'area err':>10} {'ties':>7} {'jaccard':>8} {'rounds':>7} {'sec':>7}")
print("-" * 108)

failures = 0
for name, raw in point_sets():
    ipts, _ = condition_points(raw)
    t0 = time.perf_counter()
    tri, info = R.triangulate(raw, return_info=True)
    el = time.perf_counter() - t0

    res = check_delaunay(ipts, tri)
    expect = 2 * len(raw) - 2 - len(boundary_points(ipts))
    hull_area = ConvexHull(ipts.astype(np.float64)).volume
    area_err = abs(triangle_area_sum(ipts, tri) - hull_area) / hull_area

    qh = Delaunay(ipts.astype(np.float64)).simplices
    eo, eq = edge_set(tri), edge_set(qh)
    jac = len(eo & eq) / len(eo | eq)

    ok = (res["is_delaunay"] and len(tri) == expect and area_err < 1e-12
          and (jac == 1.0 or res["cocircular_edges"] > 0))
    failures += not ok
    print(f"{name:<22} {len(raw):>7} {len(tri):>7} {expect:>7} "
          f"{str(res['is_delaunay']):>9} {area_err:>10.2e} "
          f"{res['cocircular_edges']:>7} {jac:>8.4f} {info['rounds']:>7} {el:>7.2f}"
          + ("" if ok else "   <-- FAIL"))

print(f"\n{'all checks passed' if not failures else f'{failures} FAILURES'}")
print("""
Columns
  expect     2n - 2 - h, with h counting every point on the hull boundary
  area err   |sum of triangle areas - convex hull area| / hull area; non-zero means the
             mesh has a hole or overlaps itself, neither of which the local test sees
  ties       exactly cocircular edges; where these exist, Qhull parity is not achievable
  jaccard    edge-set agreement with Qhull on the conditioned points
""")
sys.exit(1 if failures else 0)
