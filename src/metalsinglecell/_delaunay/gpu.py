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


#: Applying the flips is three more dispatches, and the shape of them is the whole trick.
#: The obvious version has each accepted flip write its two triangles *and* fix up the
#: four neighbours around them, which is what the NumPy code does — but on a GPU those
#: fix-ups race, because a neighbour may itself be flipping and rewriting the same row.
#:
#: So nothing pushes. Each triangle is given its ``role`` (the half-edge of the flip it
#: belongs to, or -1), and from that alone it can derive its own new vertices, its own new
#: links, and where each of its old edges has moved to. Neighbours then *pull* through
#: that ``remap`` instead of being written to. Every array is fully covered by exactly one
#: writer per element, so there is no race and no ordering requirement between threads.
_ROLE_SOURCE = """
    uint h = thread_position_in_grid.x;
    if (h >= (uint)nhalf[0]) return;
    if (accept[h] == 0) return;
    int t = (int)h / 3;
    int u = nbr[h] / 3;
    role[t] = (int)h;          // conflict resolution guarantees one writer per triangle
    role[u] = (int)h;
"""


_TRI_REMAP_SOURCE = """
    uint t = thread_position_in_grid.x;
    if (t >= (uint)ntri[0]) return;
    int h = role[t];

    if (h < 0) {                                        // untouched: identity
        for (int k = 0; k < 3; ++k) {
            tri_out[t * 3 + k] = tri[t * 3 + k];
            remap[t * 3 + k] = (int)t * 3 + k;
        }
        return;
    }

    int ht = h / 3, hs = h % 3;
    int code = nbr[h];
    int hu = code / 3, hj = code % 3;

    int r = tri[ht * 3 + hs];
    int p = tri[ht * 3 + (hs + 1) % 3];
    int q = tri[ht * 3 + (hs + 2) % 3];
    int s = tri[hu * 3 + hj];

    if ((int)t == ht) {                                 // (r, p, q) -> (r, p, s)
        tri_out[t * 3 + 0] = r; tri_out[t * 3 + 1] = p; tri_out[t * 3 + 2] = s;
        remap[ht * 3 + (hs + 1) % 3] = hu * 3 + 1;      // edge (q, r) moves to u'
        remap[ht * 3 + (hs + 2) % 3] = ht * 3 + 2;      // edge (r, p) stays on t'
        remap[h] = h;                                   // destroyed; never read
    } else {                                            // (s, q, p) -> (r, s, q)
        tri_out[t * 3 + 0] = r; tri_out[t * 3 + 1] = s; tri_out[t * 3 + 2] = q;
        remap[hu * 3 + (hj + 1) % 3] = ht * 3 + 0;      // edge (p, s) moves to t'
        remap[hu * 3 + (hj + 2) % 3] = hu * 3 + 0;      // edge (s, q) stays on u'
        remap[code] = code;                             // destroyed; never read
    }
"""


_LINK_SOURCE = """
    uint t = thread_position_in_grid.x;
    if (t >= (uint)ntri[0]) return;
    int h = role[t];

    if (h < 0) {
        // pull: an edge shared with a flipped triangle has moved, and remap says where
        for (int k = 0; k < 3; ++k) nbr_out[t * 3 + k] = remap[nbr[t * 3 + k]];
        return;
    }

    int ht = h / 3, hs = h % 3;
    int code = nbr[h];
    int hu = code / 3, hj = code % 3;

    int A = remap[nbr[ht * 3 + (hs + 1) % 3]];          // across (q, r)
    int B = remap[nbr[ht * 3 + (hs + 2) % 3]];          // across (r, p)
    int C = remap[nbr[hu * 3 + (hj + 1) % 3]];          // across (p, s)
    int D = remap[nbr[hu * 3 + (hj + 2) % 3]];          // across (s, q)

    if ((int)t == ht) {                                 // t' = (r, p, s)
        nbr_out[t * 3 + 0] = C;
        nbr_out[t * 3 + 1] = hu * 3 + 2;                // the new edge (r, s)
        nbr_out[t * 3 + 2] = B;
    } else {                                            // u' = (r, s, q)
        nbr_out[t * 3 + 0] = D;
        nbr_out[t * 3 + 1] = A;
        nbr_out[t * 3 + 2] = ht * 3 + 1;
    }
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
    role = mx.fast.metal_kernel(
        name="delaunay_flip_role",
        input_names=["accept", "nbr", "nhalf"],
        output_names=["role"],
        source=_ROLE_SOURCE,
    )
    tri_remap = mx.fast.metal_kernel(
        name="delaunay_flip_tri_remap",
        input_names=["tri", "nbr", "role", "ntri"],
        output_names=["tri_out", "remap"],
        source=_TRI_REMAP_SOURCE,
    )
    link = mx.fast.metal_kernel(
        name="delaunay_flip_link",
        input_names=["nbr", "role", "remap", "ntri"],
        output_names=["nbr_out"],
        source=_LINK_SOURCE,
    )
    return scan, claim, role, tri_remap, link


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

    scan, claim = _kernels()[:2]
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


def flip_round(ipts, tri, nbr, inf):
    """One complete flip round on the GPU: scan, select, and apply.

    Returns ``(tri_new, nbr_new, n_flips, (t, u))`` where ``t`` and ``u`` are the flipped
    pairs, which the host needs in order to re-home the points that were in them.
    Reporting the pairs rather than just the set of changed triangles lets the caller run
    the same re-homing as the CPU path, so the two backends produce identical meshes even
    on tie-heavy input where the triangulation is not unique.
    """
    import mlx.core as mx

    ipts = np.asarray(ipts, dtype=np.int64)
    tri = np.ascontiguousarray(np.asarray(tri, dtype=np.int32))
    nbr = np.ascontiguousarray(np.asarray(nbr, dtype=np.int32))
    m = len(tri)
    nhalf = 3 * m
    if m == 0:
        return tri.astype(np.int64), nbr.astype(np.int64), 0, np.zeros(0, dtype=np.int64)
    if np.abs(ipts).max(initial=0) > MAX_COORD:
        raise ValueError(
            "coordinates exceed MAX_COORD; pass them through condition_points() first")

    scan, claim, role_k, tri_remap_k, link_k = _kernels()
    px, py = mx.array(ipts[:, 0]), mx.array(ipts[:, 1])
    tri_a, nbr_a = mx.array(tri.reshape(-1)), mx.array(nbr.reshape(-1))
    inf_a = mx.array([inf], dtype=mx.int32)
    nh = mx.array([nhalf], dtype=mx.int32)
    nt = mx.array([m], dtype=mx.int32)
    tg_h, tg_t = min(256, nhalf), min(256, m)

    (flag,) = scan(
        inputs=[px, py, tri_a, nbr_a, inf_a, nh],
        grid=(nhalf, 1, 1), threadgroup=(tg_h, 1, 1),
        output_shapes=[(nhalf,)], output_dtypes=[mx.int8],
    )
    (owner,) = claim(
        inputs=[flag, nbr_a, nh],
        grid=(nhalf, 1, 1), threadgroup=(tg_h, 1, 1),
        output_shapes=[(m,)], output_dtypes=[mx.int32],
        init_value=_UNCLAIMED,
    )

    # accept = candidate AND still owns both of its triangles. Done with MLX ops rather
    # than on the host so the mesh never leaves the device inside the flip loop.
    half = mx.arange(nhalf, dtype=mx.int32)
    t_idx = half // 3
    u_idx = nbr_a // 3
    accept = ((flag != 0)
              & (owner[t_idx] == half)
              & (owner[u_idx] == half)).astype(mx.int8)

    (role,) = role_k(
        inputs=[accept, nbr_a, nh],
        grid=(nhalf, 1, 1), threadgroup=(tg_h, 1, 1),
        output_shapes=[(m,)], output_dtypes=[mx.int32], init_value=-1,
    )
    tri_out, remap = tri_remap_k(
        inputs=[tri_a, nbr_a, role, nt],
        grid=(m, 1, 1), threadgroup=(tg_t, 1, 1),
        output_shapes=[(nhalf,), (nhalf,)], output_dtypes=[mx.int32, mx.int32],
    )
    (nbr_out,) = link_k(
        inputs=[nbr_a, role, remap, nt],
        grid=(m, 1, 1), threadgroup=(tg_t, 1, 1),
        output_shapes=[(nhalf,)], output_dtypes=[mx.int32],
    )

    role_np = np.array(role)
    accepted = np.unique(role_np[role_np >= 0])
    pair_t = (accepted // 3).astype(np.int64)
    pair_u = (nbr.reshape(-1)[accepted] // 3).astype(np.int64)
    return (np.array(tri_out).reshape(m, 3).astype(np.int64),
            np.array(nbr_out).reshape(m, 3).astype(np.int64),
            len(accepted), (pair_t, pair_u))
