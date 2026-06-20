"""Parallel Leiden on the Metal GPU = Louvain local-moving + refinement.

Leiden (Traag et al. 2019) fixes Louvain's main flaw — communities that are
internally badly connected — by adding a refinement phase: each Louvain community
is split into well-connected sub-communities, the graph is aggregated on the
*refined* partition (more, smaller super-vertices), and the next level's local-
moving is *initialized from the Louvain communities* (so refined pieces can be
reassembled across the coarser graph). This both raises modularity and guarantees
well-connected communities.

Built on the Louvain machinery in ``louvain.py`` (coloring, per-vertex move kernel,
contraction). The refinement uses a P-restricted move kernel: a vertex may only
join a sub-community reachable through a neighbor in the *same* Louvain community,
which keeps refined communities pure (each ⊆ one Louvain community) and well-
connected, while the modularity gain still uses full degrees.
"""

from __future__ import annotations

import numpy as np

from .csr_graph import Graph
from .louvain import _contract_dense, _local_moving, color_graph
from .primitives import modularity


# Refinement move kernel: like the Louvain move kernel but a vertex only considers
# neighbors in its own Louvain community (part[na] == part[v]). Gains use full
# degrees/Σtot, so refinement optimizes the same modularity, restricted to moves
# within each Louvain community.
_REFINE_KERNEL_SOURCE = """
    uint v = thread_position_in_grid.x;
    if (color[v] != active[0]) { target[v] = comm[v]; return; }
    int pv = part[v];
    uint start = indptr[v];
    uint end = indptr[v + 1];
    int curr = comm[v];
    float kv = k[v];
    float inv = 1.0f / twom[0];
    float best_gain = -1e30f;
    int best_comm = curr;
    float stay_gain = 0.0f;
    for (uint a = start; a < end; ++a) {
        int na = indices[a];
        if (na == (int)v || part[na] != pv) { continue; }   // self / cross-community
        int ca = comm[na];
        bool first = true;
        for (uint b = start; b < a; ++b) {
            int nb = indices[b];
            if (nb != (int)v && part[nb] == pv && comm[nb] == ca) { first = false; break; }
        }
        if (!first) { continue; }
        float wc = 0.0f;
        for (uint b = start; b < end; ++b) {
            int nb = indices[b];
            if (nb != (int)v && part[nb] == pv && comm[nb] == ca) { wc += weights[b]; }
        }
        float sig = sigtot[ca] - (ca == curr ? kv : 0.0f);
        float gain = wc - res[0] * sig * kv * inv;
        if (ca == curr) { stay_gain = gain; }
        if (gain > best_gain || (gain == best_gain && ca < best_comm)) {
            best_gain = gain; best_comm = ca;
        }
    }
    target[v] = (best_gain > stay_gain + 1e-9f) ? best_comm : curr;
"""


def _refine_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="leiden_refine_move",
        input_names=["indptr", "indices", "weights", "comm", "part", "k", "sigtot",
                     "color", "active", "res", "twom"],
        output_names=["target"],
        source=_REFINE_KERNEL_SOURCE,
    )


def _refine(graph: Graph, part: np.ndarray, resolution: float, twom: float,
            seed: int = 0, max_passes: int = 50):
    """Split each Louvain community (``part``) into well-connected sub-communities.

    Returns dense refined labels (numpy). Each refined community is a subset of one
    Louvain community.
    """
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32)             # refinement starts from singletons
    part_mx = mx.array(part.astype(np.int32))
    kernel = _refine_kernel()
    res_a = mx.array([resolution], dtype=mx.float32)
    twom_a = mx.array([twom], dtype=mx.float32)

    # NB: refinement re-colors every pass (no recolor_every shuffle trick here).
    # The fixed-coloring + shuffled-order optimization used in _local_moving makes
    # within-community refinement fail to converge -> max_passes every level (a
    # ~100x slowdown at some sizes). Per-pass recoloring keeps refinement stable.
    for p in range(max_passes):
        color, n_colors = color_graph(graph, seed=seed + p)
        comm_before = comm
        for c in range(n_colors):
            sigtot = mx.zeros((n,), dtype=mx.float32).at[comm].add(k)
            (comm,) = kernel(
                inputs=[graph.indptr, graph.indices, graph.weights, comm, part_mx, k,
                        sigtot, color, mx.array([c], dtype=mx.int32), res_a, twom_a],
                grid=(n, 1, 1),
                threadgroup=(min(256, n), 1, 1),
                output_shapes=[(n,)],
                output_dtypes=[mx.int32],
            )
        mx.eval(comm)
        if bool(mx.all(comm == comm_before).item()):
            break

    _, dense = np.unique(np.asarray(comm), return_inverse=True)
    return dense.astype(np.int64)


def leiden(graph: Graph, resolution: float = 1.0, random_state: int = 0,
           n_iterations: int = 2, max_levels: int = 20) -> np.ndarray:
    """Parallel Leiden. Returns dense integer labels per vertex.

    ``n_iterations`` repeats the whole multilevel procedure, each time re-starting
    local-moving from the previous result (as in leidenalg/scanpy).
    """
    import mlx.core as mx

    twom = graph.total_weight()
    labels = None
    for it in range(max(1, n_iterations)):
        labels = _leiden_pass(graph, resolution, twom, random_state + 17 * it, init=labels)
    return labels


def _leiden_pass(g0: Graph, resolution: float, twom: float, seed: int,
                 init: np.ndarray | None, max_levels: int = 20) -> np.ndarray:
    """One full multilevel Leiden run (move -> refine -> aggregate, repeat)."""
    import mlx.core as mx

    g = g0
    orig2node = np.arange(g0.n, dtype=np.int64)
    # P: community label per node of the current (aggregate) graph.
    P = (init.copy() if init is not None else np.arange(g0.n, dtype=np.int64))

    for level in range(max_levels):
        # 1. Local moving, initialized from P (singletons on level 0 of a fresh run).
        Pmx = _local_moving(g, resolution, twom, seed=seed + level,
                            init_comm=mx.array(P.astype(np.int32)))
        _, P = np.unique(np.asarray(Pmx), return_inverse=True)
        P = P.astype(np.int64)
        if P.max() + 1 == g.n:                       # each node its own community: done
            break

        # 2. Refinement: split each Louvain community into well-connected pieces.
        R = _refine(g, P, resolution, twom, seed=seed + level)
        nR = int(R.max()) + 1
        if nR == g.n:                                # refinement can't aggregate further
            break

        # 3. Aggregate on the REFINED partition; lift P onto the refined super-vertices.
        P_agg = np.zeros(nR, dtype=np.int64)
        P_agg[R] = P                                 # each refined community ⊆ one Louvain community
        orig2node = R[orig2node]
        g = _contract_dense(g, R, nR)
        P = P_agg

    _, final = np.unique(P[orig2node], return_inverse=True)
    return final.astype(np.int64)
