"""EXPERIMENTAL hybrid Louvain: GPU computes all move gains, CPU applies them.

Workaround #1 for the two M3 limits that sank the fused single-kernel (no grid-wide
barrier, no float atomics). Instead of coloring + per-color kernel launches, ONE GPU
dispatch computes, for every vertex, its best target community and the gain (the
expensive O(degree^2) part — fully parallel, reads a Σtot snapshot, mutates nothing).
The CPU then applies moves greedily in gain order with exact, sequentially-updated
Σtot — the serialization point, so no grid barrier is needed; Σtot stays exact on the
host, so no float atomics are needed. Apple unified memory makes the per-pass GPU↔CPU
handoff cheap.

Quality is validated by modularity, like the colored Louvain. See graph-clustering skill.
"""

from __future__ import annotations

import numpy as np

from .csr_graph import Graph
from .louvain import _DEGREE_CAP, _contract_dense
from .primitives import modularity


# One dispatch, ALL vertices: for each v compute the best target community and the
# pieces the host needs to re-validate the move against the *current* Σtot — the
# edge weight to the best community (wc_best) and to v's own community (wc_curr).
# No coloring, no Σtot mutation: this is a pure read-only gain scan.
_GAIN_KERNEL_SOURCE = """
    uint v = thread_position_in_grid.x;
    uint start = indptr[v];
    uint end = indptr[v + 1];
    int curr = comm[v];
    best_comm[v] = curr; wc_best[v] = 0.0f; wc_curr[v] = 0.0f; gain[v] = 0.0f;
    if (end - start > (uint)cap[0]) { return; }     // high-degree -> host
    float kv = k[v];
    float inv = 1.0f / twom[0];
    float best_g = -1e30f; int best_c = curr; float wcb = 0.0f;
    float stay_g = 0.0f; float wcc = 0.0f;
    for (uint a = start; a < end; ++a) {
        int na = indices[a];
        if (na == (int)v) { continue; }
        int ca = comm[na];
        bool first = true;
        for (uint b = start; b < a; ++b) {
            int nb = indices[b];
            if (nb != (int)v && comm[nb] == ca) { first = false; break; }
        }
        if (!first) { continue; }
        float wc = 0.0f;
        for (uint b = start; b < end; ++b) {
            int nb = indices[b];
            if (nb != (int)v && comm[nb] == ca) { wc += weights[b]; }
        }
        float sig = sigtot[ca] - (ca == curr ? kv : 0.0f);
        float g = wc - res[0] * sig * kv * inv;
        if (ca == curr) { stay_g = g; wcc = wc; }
        if (g > best_g || (g == best_g && ca < best_c)) { best_g = g; best_c = ca; wcb = wc; }
    }
    best_comm[v] = best_c; wc_best[v] = wcb; wc_curr[v] = wcc;
    gain[v] = best_g - stay_g;          // snapshot improvement of moving vs staying
"""


def _gain_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="louvain_hybrid_gain",
        input_names=["indptr", "indices", "weights", "comm", "k", "sigtot",
                     "res", "twom", "cap"],
        output_names=["best_comm", "gain", "wc_best", "wc_curr"],
        source=_GAIN_KERNEL_SOURCE,
    )


def _local_moving_hybrid(graph: Graph, resolution: float, twom: float, seed: int = 0,
                         max_passes: int = 100, init_comm=None):
    """GPU-gain / CPU-apply local moving. Returns numpy community labels."""
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    k_np = np.asarray(k).astype(np.float64)
    comm = (np.arange(n, dtype=np.int64) if init_comm is None
            else np.asarray(init_comm).astype(np.int64))
    kernel = _gain_kernel()
    res_a = mx.array([resolution], dtype=mx.float32)
    twom_a = mx.array([twom], dtype=mx.float32)
    cap_a = mx.array([_DEGREE_CAP], dtype=mx.int32)
    inv = 1.0 / twom

    # high-degree vertices: gain kernel skips them (returns stay); host moves them too
    deg = np.diff(np.asarray(graph.indptr).astype(np.int64))
    large_ids = np.flatnonzero(deg > _DEGREE_CAP)

    for _ in range(max_passes):
        sigtot = mx.zeros((n,), dtype=mx.float32).at[mx.array(comm.astype(np.int32))].add(k)
        best_comm, gain, wc_best, wc_curr = kernel(
            inputs=[graph.indptr, graph.indices, graph.weights,
                    mx.array(comm.astype(np.int32)), k, sigtot, res_a, twom_a, cap_a],
            grid=(n, 1, 1),
            threadgroup=(min(256, n), 1, 1),
            output_shapes=[(n,), (n,), (n,), (n,)],
            output_dtypes=[mx.int32, mx.float32, mx.float32, mx.float32],
        )
        mx.eval(best_comm, gain, wc_best, wc_curr)
        bc = np.asarray(best_comm).astype(np.int64)
        gn = np.asarray(gain).astype(np.float64)
        wb = np.asarray(wc_best).astype(np.float64)
        wcur = np.asarray(wc_curr).astype(np.float64)

        # CPU: exact Σtot, apply candidate moves in descending snapshot-gain order,
        # re-validating each against the *current* Σtot (wc terms from the snapshot).
        sig_np = np.bincount(comm, weights=k_np, minlength=n).astype(np.float64)
        cand = np.flatnonzero((bc != comm) & (gn > 1e-9))
        order = cand[np.argsort(-gn[cand])]
        moved = 0
        for v in order:
            curr = comm[v]; best = bc[v]
            if best == curr:
                continue
            kv = k_np[v]
            g_move = wb[v] - resolution * sig_np[best] * kv * inv
            g_stay = wcur[v] - resolution * (sig_np[curr] - kv) * kv * inv
            if g_move > g_stay + 1e-9:
                sig_np[curr] -= kv; sig_np[best] += kv
                comm[v] = best; moved += 1

        if large_ids.size:
            _apply_high_degree(graph, comm, k_np, sig_np, large_ids, resolution, twom)
        if moved == 0:
            break

    return comm


def _apply_high_degree(graph, comm, k_np, sig_np, large_ids, resolution, twom):
    indptr = np.asarray(graph.indptr).astype(np.int64)
    indices = np.asarray(graph.indices)
    weights = np.asarray(graph.weights).astype(np.float64)
    inv = 1.0 / twom
    for v in large_ids:
        s, e = int(indptr[v]), int(indptr[v + 1])
        nbr = indices[s:e]; keep = nbr != v
        nbr, w = nbr[keep], weights[s:e][keep]
        if nbr.size == 0:
            continue
        wc = np.bincount(comm[nbr], weights=w)
        c = np.flatnonzero(wc > 0)
        kv = k_np[v]; curr = int(comm[v])
        sig = sig_np[c] - np.where(c == curr, kv, 0.0)
        g = wc[c] - resolution * sig * kv * inv
        stay = g[c == curr]; stay_g = float(stay[0]) if stay.size else 0.0
        best = int(c[np.argmax(g)])
        if g[np.argmax(g)] > stay_g + 1e-9 and best != curr:
            sig_np[curr] -= kv; sig_np[best] += kv; comm[v] = best


def louvain_hybrid(graph: Graph, resolution: float = 1.0, random_state: int = 0,
                   max_levels: int = 20, tol: float = 1e-9) -> np.ndarray:
    """Multilevel hybrid Louvain (GPU gains, CPU apply). Dense integer labels."""
    import mlx.core as mx

    twom = graph.total_weight()
    g = graph
    orig2super = np.arange(graph.n, dtype=np.int64)
    q_prev = -1.0
    for level in range(max_levels):
        comm = _local_moving_hybrid(g, resolution, twom, seed=random_state + 100 * level)
        _, comm_dense = np.unique(comm, return_inverse=True)
        comm_dense = comm_dense.astype(np.int64)
        C = int(comm_dense.max()) + 1
        q = modularity(g, mx.array(comm_dense.astype(np.int32)), resolution)
        if C == g.n or q <= q_prev + tol:
            break
        orig2super = comm_dense[orig2super]
        g = _contract_dense(g, comm_dense, C)
        q_prev = q
    _, labels = np.unique(orig2super, return_inverse=True)
    return labels.astype(np.int64)
