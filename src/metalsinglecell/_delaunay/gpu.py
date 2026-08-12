"""Metal kernels for the flipping phase — the triangulator's hot loop.

Flipping dominates: on real Xenium and Visium geometry the reference spends most of its
time here, running the same two steps over and over until no edge fails the in-circle
test. Both are a perfect fit for the GPU and neither needs the host:

* **Scan.** One thread per half-edge, evaluating an exact in-circle predicate. Embarrassingly
  parallel and arithmetic-heavy, which is the good case.
* **Select.** Two flips conflict when they share a triangle, so each candidate claims both
  of its triangles and proceeds only if it still owns them. The reference does this with
  ``np.minimum.at``; here it is ``atomic_fetch_min_explicit`` on the output buffer, which
  is the same operation and is what the scatter-min in the reference was written to mirror.

The claim token is the **half-edge index** rather than a compacted candidate id. That
avoids a stream compaction — which would need a prefix sum and a host round trip to find
the surviving indices — and it keeps the tie-break identical to the reference's, because
the lowest half-edge index wins in both.

Applying the flips (rewriting vertices and adjacency) is still on the host. It is scatter
work rather than arithmetic, and it needs the old-to-new half-edge remap described in
:mod:`.reference`, so it is the more delicate port and comes next.
"""

from __future__ import annotations

import numpy as np

from .predicates import MAX_COORD, i128_header

#: Sentinel for "no triangle claimed": larger than any half-edge index we can produce.
_UNCLAIMED = np.iinfo(np.int32).max


_SCAN_SOURCE = """
    uint h = thread_position_in_grid.x;
    if (h >= (uint)nhalf[0]) return;
    flag[h] = 0;

    int t = (int)h / 3;
    int code = nbr[h];
    int u = code / 3, j = code % 3;
    if (u <= t) return;                     // visit each edge once, from the lower triangle

    int INF = inf[0];
    int s = tri[u * 3 + j];
    int v0 = tri[t * 3 + 0], v1 = tri[t * 3 + 1], v2 = tri[t * 3 + 2];

    char sg;
    if (s == INF) {
        sg = -1;                            // the vertex at infinity is outside every
                                            // finite circumcircle
    } else if (v0 == INF || v1 == INF || v2 == INF) {
        // a ghost's circumcircle is the half-plane outside its hull edge, so the
        // predicate degrades to an orientation. Rotate the vertex at infinity last.
        int a, b;
        if (v0 == INF)      { a = v1; b = v2; }
        else if (v1 == INF) { a = v2; b = v0; }
        else                { a = v0; b = v1; }
        sg = orient_sign((long)px[a], (long)py[a], (long)px[b], (long)py[b],
                         (long)px[s], (long)py[s]);
    } else {
        sg = incircle_sign((long)px[v0], (long)py[v0], (long)px[v1], (long)py[v1],
                           (long)px[v2], (long)py[v2], (long)px[s],  (long)py[s]);
    }
    // strictly greater than zero: an exact tie is left alone, because both diagonals are
    // Delaunay and flipping on ties is how a flip loop starts cycling
    flag[h] = (sg > 0) ? 1 : 0;
"""


_CLAIM_SOURCE = """
    uint h = thread_position_in_grid.x;
    if (h >= (uint)nhalf[0]) return;
    if (flag[h] == 0) return;
    int t = (int)h / 3;
    int u = nbr[h] / 3;
    device atomic_int* o = (device atomic_int*)owner;
    atomic_fetch_min_explicit(&o[t], (int)h, memory_order_relaxed);
    atomic_fetch_min_explicit(&o[u], (int)h, memory_order_relaxed);
"""


def _kernels():
    import mlx.core as mx

    scan = mx.fast.metal_kernel(
        name="delaunay_flip_scan",
        input_names=["px", "py", "tri", "nbr", "inf", "nhalf"],
        output_names=["flag"],
        header=i128_header(),
        source=_SCAN_SOURCE,
    )
    claim = mx.fast.metal_kernel(
        name="delaunay_flip_claim",
        input_names=["flag", "nbr", "nhalf"],
        output_names=["owner"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_CLAIM_SOURCE,
    )
    return scan, claim


def flip_candidates(ipts, tri, nbr, inf):
    """Half-edges to flip this round: non-Delaunay, and free of conflicts.

    Returns the accepted half-edge indices ``h = t * 3 + slot``, sorted. Identical to what
    :func:`.reference._flip_round` selects — the tie-break, the ``t < u`` visit rule and
    the leave-ties-alone rule all match, so this is a drop-in for that selection.
    """
    import mlx.core as mx

    ipts = np.asarray(ipts, dtype=np.int64)
    tri = np.ascontiguousarray(np.asarray(tri, dtype=np.int32))
    nbr = np.ascontiguousarray(np.asarray(nbr, dtype=np.int32))
    m = len(tri)
    nhalf = 3 * m
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    if np.abs(ipts).max(initial=0) > MAX_COORD:
        raise ValueError(
            "coordinates exceed MAX_COORD; pass them through condition_points() first")

    scan, claim = _kernels()
    px, py = mx.array(ipts[:, 0]), mx.array(ipts[:, 1])
    tri_a, nbr_a = mx.array(tri.reshape(-1)), mx.array(nbr.reshape(-1))
    inf_a = mx.array([inf], dtype=mx.int32)
    nh = mx.array([nhalf], dtype=mx.int32)
    tg = min(256, nhalf)

    (flag,) = scan(
        inputs=[px, py, tri_a, nbr_a, inf_a, nh],
        grid=(nhalf, 1, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(nhalf,)], output_dtypes=[mx.int8],
    )
    (owner,) = claim(
        inputs=[flag, nbr_a, nh],
        grid=(nhalf, 1, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(m,)], output_dtypes=[mx.int32],
        init_value=_UNCLAIMED,
    )

    flag_np = np.array(flag).astype(bool)
    owner_np = np.array(owner)
    h = np.where(flag_np)[0]
    if len(h) == 0:
        return np.zeros(0, dtype=np.int64)
    t = h // 3
    u = nbr.reshape(-1)[h] // 3            # h indexes half-edges, not triangles
    keep = (owner_np[t] == h) & (owner_np[u] == h)
    return h[keep].astype(np.int64)
