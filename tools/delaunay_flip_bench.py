"""Is the Metal flip round worth having, and where does it overtake NumPy?

The flipping phase is the triangulator's hot loop, so this measures the two steps that
were ported — the exact in-circle scan over every half-edge, and the atomic conflict
resolution — against the NumPy versions they replace, on meshes far larger than the
reference could build in reasonable time.

Those meshes come from Qhull, converted into our ghost-closed adjacency structure. That
conversion is only a test fixture: it lets the scan be timed at 400k points without
waiting for the reference to insert them one batch at a time.

Run: ``python tools/delaunay_flip_bench.py``
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.spatial import Delaunay

from metalsinglecell._delaunay import reference as R
from metalsinglecell._delaunay.gpu import flip_candidates
from metalsinglecell._delaunay.predicates import condition_points, orient2d

warnings.simplefilter("ignore")


def mesh_from_qhull(ipts):
    """Qhull's triangulation, as one of our ghost-closed meshes.

    Qhull gives neighbours but leaves the hull open (``-1``); we close it with the ring of
    ghost triangles the algorithm expects, so the scan sees exactly the structure it would
    during a real run — including the ghost-ghost edges, which take the orientation branch
    rather than the in-circle one.
    """
    d = Delaunay(ipts.astype(np.float64))
    tri = d.simplices.astype(np.int64).copy()
    nb = d.neighbors.astype(np.int64).copy()

    # our meshes are counter-clockwise; swapping two vertices swaps the two neighbour
    # slots that face them
    flip = orient2d(ipts, tri[:, 0], tri[:, 1], tri[:, 2]) < 0
    tri[flip, 0], tri[flip, 2] = tri[flip, 2], tri[flip, 0].copy()
    nb[flip, 0], nb[flip, 2] = nb[flip, 2], nb[flip, 0].copy()

    m = len(tri)
    inf = len(ipts)
    hull = np.argwhere(nb < 0)                       # (triangle, slot) pairs
    ghosts = np.zeros((len(hull), 3), dtype=np.int64)
    by_first = {}
    for g, (t, j) in enumerate(hull):
        a, b = tri[t, (j + 1) % 3], tri[t, (j + 2) % 3]
        ghosts[g] = [b, a, inf]                      # ghost across hull edge (a, b)
        by_first[int(b)] = m + g

    all_tri = np.vstack([tri, ghosts])
    nbr = np.full((len(all_tri), 3), -1, dtype=np.int64)
    nbr[:m] = nb * 3                                 # slot filled in below
    for t in range(m):
        for j in range(3):
            u = nb[t, j]
            if u >= 0:
                nbr[t, j] = u * 3 + int(np.where(nb[u] == t)[0][0])
    for g, (t, j) in enumerate(hull):
        gi = m + g
        b, a = ghosts[g, 0], ghosts[g, 1]
        nbr[t, j] = gi * 3 + 2                       # finite side faces the ghost
        nbr[gi, 2] = t * 3 + j
        nbr[gi, 0] = by_first[int(a)] * 3 + 1        # ghost ring: edge (a, INF)
    for g in range(len(hull)):
        gi = m + g
        code = nbr[gi, 0]
        nbr[code // 3, code % 3] = gi * 3 + 0
    return R._Mesh(all_tri, nbr), inf


print(f"{'n points':>9} {'triangles':>10} {'half-edges':>11} {'cpu ms':>9} {'gpu ms':>9} "
      f"{'speedup':>8} {'agree':>7}")
print("-" * 70)

rng = np.random.default_rng(0)
for n in (1_000, 5_000, 20_000, 100_000, 400_000):
    pts = rng.uniform(0, 1, (n, 2)) * (50 * np.sqrt(n))
    ipts, _ = condition_points(pts)
    mesh, inf = mesh_from_qhull(ipts)
    R.check_mesh(mesh, ipts, inf, "qhull import")

    flip_candidates(ipts, mesh.tri, mesh.nbr, inf)                 # warm the kernel cache
    t0 = time.perf_counter()
    cpu = R.select_flips(ipts, mesh, inf)
    tc = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    gpu = flip_candidates(ipts, mesh.tri, mesh.nbr, inf)
    tg = (time.perf_counter() - t0) * 1e3

    agree = np.array_equal(np.sort(cpu), np.sort(gpu))
    print(f"{n:>9} {mesh.n_tri:>10} {3 * mesh.n_tri:>11} {tc:>9.1f} {tg:>9.1f} "
          f"{tc / tg:>7.2f}x {str(agree):>7}")

print("""
A Qhull mesh is already Delaunay, so both paths correctly select zero flips — what is
being timed is the scan over every half-edge, which is the per-round cost the real loop
pays dozens of times. 'agree' checks the two selections match, which they must.
""")
