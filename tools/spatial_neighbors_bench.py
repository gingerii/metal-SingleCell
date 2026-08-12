"""Speed of the four spatial-neighbour builders vs squidpy, across sizes.

Coordinates only — the graph does not depend on expression — so synthetic point clouds at a
realistic spatial density are a fair stand-in for larger sections than the Visium slide.
Warm-up + best-of-N, as the rest of the benchmark table.
"""
import time
import warnings
import numpy as np
import anndata as ad

warnings.simplefilter("ignore")
import squidpy as sq
import metalsinglecell as msc


def make(n, seed=0):
    """A jittered hex lattice — Visium-like spacing, so radius/grid see realistic degrees."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    r, c = np.divmod(np.arange(side * side), side)
    x = c * 100.0 + (r % 2) * 50.0
    y = r * 86.6
    xy = np.c_[x, y][:n] + rng.normal(0, 1.5, size=(n, 2))
    a = ad.AnnData(np.zeros((n, 2), dtype=np.float32))
    a.obsm["spatial"] = xy.astype(np.float64)
    return a


def timed(fn, reps=3, warmup=True):
    if warmup:
        fn()
    best = np.inf
    for _ in range(reps):
        t = time.time(); fn(); best = min(best, time.time() - t)
    return best


CASES = [
    ("knn(6)", lambda a: msc.gr.spatial_neighbors_knn(a, n_neighs=6),
     lambda a: sq.gr.spatial_neighbors_knn(a, n_neighs=6)),
    ("radius(150)", lambda a: msc.gr.spatial_neighbors_radius(a, radius=150.0),
     lambda a: sq.gr.spatial_neighbors_radius(a, radius=150.0)),
    ("grid(rings=2)", lambda a: msc.gr.spatial_neighbors_grid(a, n_neighs=6, n_rings=2),
     lambda a: sq.gr.spatial_neighbors_grid(a, n_neighs=6, n_rings=2)),
    ("delaunay", lambda a: msc.gr.spatial_neighbors_delaunay(a),
     lambda a: sq.gr.spatial_neighbors_delaunay(a)),
]

print(f"{'n':>8}  {'builder':<15} {'squidpy':>9} {'ours':>9} {'speedup':>9}")
for n in (10_000, 50_000, 200_000, 500_000):
    a = make(n)
    for tag, ours, theirs in CASES:
        try:
            ts = timed(lambda: theirs(a.copy()), reps=2)
            to = timed(lambda: ours(a.copy()), reps=2)
            print(f"{n:>8}  {tag:<15} {ts:>8.2f}s {to:>8.2f}s {ts/to:>8.1f}x")
        except Exception as e:
            print(f"{n:>8}  {tag:<15} ERROR {type(e).__name__}: {str(e)[:50]}")
    print()
