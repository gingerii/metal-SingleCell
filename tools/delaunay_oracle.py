"""A correctness oracle for Delaunay, and the question that comes before writing any kernel.

Two jobs:

1. Decide whether a triangulation is Delaunay, independently of Qhull. Visium coordinates are
   integers, so the incircle determinant can be evaluated in exact Python integer arithmetic —
   no epsilon, no floating point, no argument about tolerance.

2. Answer the question that decides the whole project: **on a lattice the Delaunay
   triangulation is not unique.** Four cocircular points can be split along either diagonal and
   both are Delaunay. If most Visium edges are choice-dependent, then NO correct GPU Delaunay
   can reproduce Qhull's edge set, and "parity with squidpy" has to mean "a valid Delaunay
   triangulation", not "the same graph". Better to know that now than after writing a kernel.
"""
import warnings
from itertools import combinations

import numpy as np

warnings.simplefilter("ignore")


def incircle_exact(a, b, c, d):
    """Sign of the incircle determinant, in exact integer arithmetic.

    > 0  d strictly inside the circumcircle of (a, b, c)   [for counter-clockwise abc]
    = 0  the four points are exactly cocircular            [the degenerate case]
    < 0  d strictly outside
    """
    ax, ay = int(a[0]) - int(d[0]), int(a[1]) - int(d[1])
    bx, by = int(b[0]) - int(d[0]), int(b[1]) - int(d[1])
    cx, cy = int(c[0]) - int(d[0]), int(c[1]) - int(d[1])
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    det = (ax * (by * cc - bb * cy)
           - ay * (bx * cc - bb * cx)
           + aa * (bx * cy - by * cx))
    return (det > 0) - (det < 0)


def orient_exact(a, b, c):
    return int((int(b[0]) - int(a[0])) * (int(c[1]) - int(a[1]))
               - (int(b[1]) - int(a[1])) * (int(c[0]) - int(a[0])))


def check_delaunay(points, simplices):
    """Is this a valid Delaunay triangulation? Returns a dict of findings.

    Uses the local Delaunay criterion: a triangulation is globally Delaunay iff every internal
    edge is locally Delaunay, i.e. the vertex opposite the edge in one triangle is not strictly
    inside the other triangle's circumcircle.
    """
    pts = np.asarray(points)
    edge_map = {}
    degenerate_tris = 0
    for t, tri in enumerate(simplices):
        i, j, k = tri
        if orient_exact(pts[i], pts[j], pts[k]) == 0:
            degenerate_tris += 1
        for e in combinations(sorted((int(i), int(j), int(k))), 2):
            edge_map.setdefault(e, []).append(t)

    violations, cocircular = 0, 0
    for e, tris in edge_map.items():
        if len(tris) != 2:
            continue                                  # hull edge
        t1, t2 = tris
        opp1 = [v for v in simplices[t1] if v not in e]
        opp2 = [v for v in simplices[t2] if v not in e]
        if not opp1 or not opp2:
            continue
        a, b, c = simplices[t1]
        if orient_exact(pts[a], pts[b], pts[c]) < 0:
            a, c = c, a                               # normalise to counter-clockwise
        s = incircle_exact(pts[a], pts[b], pts[c], pts[opp2[0]])
        if s > 0:
            violations += 1
        elif s == 0:
            cocircular += 1

    return {"n_triangles": len(simplices), "n_edges": len(edge_map),
            "degenerate_triangles": degenerate_tris,
            "not_locally_delaunay": violations,
            "cocircular_edges": cocircular,
            "is_delaunay": violations == 0 and degenerate_tris == 0}


def edge_set(simplices):
    out = set()
    for tri in simplices:
        for e in combinations(sorted(int(v) for v in tri), 2):
            out.add(e)
    return out


if __name__ == "__main__":
    import scanpy as sc
    from scipy.spatial import Delaunay

    a = sc.read_visium("data/V1_Breast_Cancer_Block_A_Section_1")
    a.var_names_make_unique()
    pts = np.asarray(a.obsm["spatial"])
    print(f"Visium: {len(pts)} spots, integer coords = {pts.dtype}")

    tri = Delaunay(pts.astype(np.float64))
    res = check_delaunay(pts, tri.simplices)
    print("\nQhull's own output, judged by the exact oracle:")
    for k, v in res.items():
        print(f"  {k:24s} {v}")

    # How much of the edge set is a free choice among cocircular points?
    print(f"\n  edges whose flip is a TIE (cocircular, either diagonal valid): "
          f"{res['cocircular_edges']} of {res['n_edges']} "
          f"({100 * res['cocircular_edges'] / res['n_edges']:.1f}%)")

    # Does a tiny perturbation change Qhull's answer? If so the edge set is not canonical.
    rng = np.random.default_rng(0)
    e0 = edge_set(tri.simplices)
    for eps in (1e-9, 1e-6, 1e-3):
        pj = pts.astype(np.float64) + rng.normal(0, eps, pts.shape)
        e1 = edge_set(Delaunay(pj).simplices)
        jac = len(e0 & e1) / len(e0 | e1)
        print(f"  jitter {eps:>7}: edge Jaccard vs unperturbed Qhull = {jac:.4f} "
              f"({len(e0 ^ e1)} edges differ)")
