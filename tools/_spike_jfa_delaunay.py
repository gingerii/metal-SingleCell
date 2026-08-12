"""Spike: jump-flooding digital Voronoi on Metal, and how far its dual is from Delaunay.

This is the go/no-go measurement for a Metal gDel2D. The pipeline is
    jump flood -> discrete Voronoi labels -> dualise -> candidate triangles -> REPAIR
and the whole question is how big "repair" is. If the dual already recovers ~all of Qhull's
triangles, the repair is a local fixup and the port is tractable. If it recovers 70%, the
repair *is* the algorithm and this is a much bigger build than it looks.

Measures: JFA wall time on the GPU, and the recall/precision of the dualised triangle set
against Qhull's, at several grid resolutions.
"""
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import mlx.core as mx
from scipy.spatial import Delaunay


def jump_flood(points, G, verbose=False):
    """Discrete Voronoi labels on a G x G grid, by jump flooding (Rong & Tan).

    Each pass, a pixel considers the seed held by the neighbour `step` away in each of 8
    directions plus itself, and keeps whichever seed is nearest. Halving `step` from G/2 to 1
    gives log2(G) passes, each a handful of gathers — the shape of thing a GPU likes.
    """
    pts = np.asarray(points, dtype=np.float64)
    lo, hi = pts.min(0), pts.max(0)
    span = (hi - lo).max()
    # map into [0, G-1] preserving aspect, half-pixel inset so seeds land inside
    sxy = (pts - lo) / span * (G - 1)
    sx = mx.array(sxy[:, 0].astype(np.float32))
    sy = mx.array(sxy[:, 1].astype(np.float32))

    px = np.clip(np.round(sxy).astype(np.int64), 0, G - 1)
    lab0 = np.full((G, G), -1, dtype=np.int32)
    lab0[px[:, 1], px[:, 0]] = np.arange(len(pts))      # last writer wins on collisions
    L = mx.array(lab0)

    gy, gx = mx.meshgrid(mx.arange(G, dtype=mx.float32), mx.arange(G, dtype=mx.float32),
                         indexing="ij")
    BIG = mx.array(np.float32(1e18))

    def dist2(lbl):
        ok = lbl >= 0
        safe = mx.maximum(lbl, 0)
        dx = mx.take(sx, safe) - gx
        dy = mx.take(sy, safe) - gy
        return mx.where(ok, dx * dx + dy * dy, BIG)

    step = 1
    while step < G:
        step *= 2
    step //= 2
    while step >= 1:
        best_l, best_d = L, dist2(L)
        for oy in (-step, 0, step):
            for ox in (-step, 0, step):
                if ox == 0 and oy == 0:
                    continue
                cand = mx.roll(L, shift=(oy, ox), axis=(0, 1))
                # roll wraps; blank the wrapped band so seeds don't teleport across the border
                if oy > 0:
                    cand[:oy, :] = -1
                elif oy < 0:
                    cand[oy:, :] = -1
                if ox > 0:
                    cand[:, :ox] = -1
                elif ox < 0:
                    cand[:, ox:] = -1
                d = dist2(cand)
                take = d < best_d
                best_l = mx.where(take, cand, best_l)
                best_d = mx.where(take, d, best_d)
        L = best_l
        mx.eval(L)
        step //= 2
    return L


def dualise(L):
    """Candidate triangles: every 2x2 pixel block holding 3+ distinct Voronoi labels."""
    a = np.asarray(L)
    q = np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]], axis=-1).reshape(-1, 4)
    q = q[(q >= 0).all(1)]
    q.sort(axis=1)
    # distinct labels per block
    d01, d12, d23 = q[:, 0] != q[:, 1], q[:, 1] != q[:, 2], q[:, 2] != q[:, 3]
    ndist = 1 + d01.astype(np.int8) + d12.astype(np.int8) + d23.astype(np.int8)
    tris = set()
    three = q[ndist == 3]
    for row in three:
        u = np.unique(row)
        if len(u) == 3:
            tris.add(tuple(u))
    four = q[ndist == 4]
    for row in four:                                    # a 4-corner block: both diagonals
        for drop in range(4):
            tris.add(tuple(np.delete(row, drop)))
    return tris


def qhull_tris(points):
    return {tuple(sorted(int(v) for v in t)) for t in Delaunay(np.asarray(points,
                                                                         dtype=np.float64)).simplices}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for tag, pts in [
        ("random 5k", rng.uniform(0, 10_000, (5_000, 2))),
        ("random 20k", rng.uniform(0, 10_000, (20_000, 2))),
    ]:
        ref = qhull_tris(pts)
        print(f"\n{tag}: {len(pts)} points, Qhull gives {len(ref)} triangles")
        for G in (512, 1024, 2048, 4096):
            t = time.time(); L = jump_flood(pts, G); mx.eval(L); t_jfa = time.time() - t
            t = time.time(); cand = dualise(L); t_dual = time.time() - t
            hit = len(cand & ref)
            print(f"   G={G:>5}: JFA {t_jfa:6.2f}s  dual {t_dual:5.2f}s  "
                  f"candidates {len(cand):>7}  recall {hit/len(ref):6.2%}  "
                  f"precision {hit/max(len(cand),1):6.2%}")
