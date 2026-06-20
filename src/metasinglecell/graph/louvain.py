"""Parallel multilevel Louvain on the Metal GPU (MLX).

Local-moving is parallelized via **graph coloring**: vertices of one color form an
independent set (no two adjacent), so they can move simultaneously while each sees
up-to-date neighbor communities — recovering near-sequential quality without the
fragmentation of fully-synchronous moves. Colors are processed in sequence; after
a vertex moves, later colors see it. Levels contract and repeat until modularity
stops improving.

Validated by modularity (close to igraph) rather than label parity — parallel
Louvain is a different, stochastic optimizer. See the `graph-clustering` skill.
"""

from __future__ import annotations

import numpy as np

from .csr_graph import Graph
from .primitives import _segments, modularity, segment_head, segment_sum


# Per-vertex best-move kernel: one thread per vertex walks its own neighbor list
# (CSR row) and, with an O(degree^2) scan over distinct neighbor communities,
# picks the community maximizing the modularity gain. No sort, no shared memory —
# replaces the global edge-sort that made the colored version slow. Only vertices
# of the active color compute; others keep their community (independent set ->
# simultaneous moves don't interfere, and each sees up-to-date neighbors).
_MOVE_KERNEL_SOURCE = """
    uint v = thread_position_in_grid.x;
    if (color[v] != active[0]) { target[v] = comm[v]; return; }
    uint start = indptr[v];
    uint end = indptr[v + 1];
    int curr = comm[v];
    float kv = k[v];
    float inv = 1.0f / twom[0];
    float best_gain = -1e30f;
    int best_comm = curr;
    float stay_gain = 0.0f;            // isolated baseline if no same-community edge
    for (uint a = start; a < end; ++a) {
        int na = indices[a];
        if (na == (int)v) { continue; }        // skip self-loop
        int ca = comm[na];
        bool first = true;                     // process each community once
        for (uint b = start; b < a; ++b) {
            int nb = indices[b];
            if (nb != (int)v && comm[nb] == ca) { first = false; break; }
        }
        if (!first) { continue; }
        float wc = 0.0f;                       // total edge weight v -> community ca
        for (uint b = start; b < end; ++b) {
            int nb = indices[b];
            if (nb != (int)v && comm[nb] == ca) { wc += weights[b]; }
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


def _move_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="louvain_best_move",
        input_names=["indptr", "indices", "weights", "comm", "k", "sigtot",
                     "color", "active", "res", "twom"],
        output_names=["target"],
        source=_MOVE_KERNEL_SOURCE,
    )


def color_graph(graph: Graph, seed: int = 0, max_colors: int = 2000):
    """Greedy Luby graph coloring on the GPU. Returns ``(color, n_colors)``.

    Each round, vertices whose random priority exceeds all *uncolored* neighbors'
    form an independent set and take the next color.
    """
    import mlx.core as mx

    n = graph.n
    src, dst = graph.edge_src, graph.indices
    color = mx.full((n,), -1, dtype=mx.int32)
    key = mx.random.key(seed)
    r = 0
    while bool(mx.any(color < 0).item()) and r < max_colors:
        key, sub = mx.random.split(key)
        prio = mx.random.uniform(shape=(n,), key=sub)
        unc = color < 0
        both = unc[src] & unc[dst]
        contrib = mx.where(both, prio[dst], mx.full(dst.shape, -mx.inf))
        max_nb = mx.full((n,), -mx.inf).at[src].maximum(contrib)
        selected = unc & (prio > max_nb)
        color = mx.where(selected, mx.array(r, dtype=mx.int32), color)
        r += 1
    return color, r


def _local_moving(graph: Graph, resolution: float, twom: float, seed: int = 0,
                  max_passes: int = 100):
    """Colored local-moving from singletons (per-vertex kernel, no sort).

    Returns MLX community labels.
    """
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32)
    kernel = _move_kernel()
    res_a = mx.array([resolution], dtype=mx.float32)
    twom_a = mx.array([twom], dtype=mx.float32)

    for p in range(max_passes):
        # Re-color each pass (fresh random independent sets) — important for
        # escaping poor optima on fuzzy graphs; cheap relative to the moves.
        color, n_colors = color_graph(graph, seed=seed + p)
        comm_before = comm
        for c in range(n_colors):
            sigtot = mx.zeros((n,), dtype=mx.float32).at[comm].add(k)
            (comm,) = kernel(
                inputs=[graph.indptr, graph.indices, graph.weights, comm, k, sigtot,
                        color, mx.array([c], dtype=mx.int32), res_a, twom_a],
                grid=(n, 1, 1),
                threadgroup=(min(256, n), 1, 1),
                output_shapes=[(n,)],
                output_dtypes=[mx.int32],
            )
        mx.eval(comm)                                  # one sync per pass, not per color
        if bool(mx.all(comm == comm_before).item()):
            break

    return comm


def louvain(graph: Graph, resolution: float = 1.0, random_state: int = 0,
            max_levels: int = 20, tol: float = 1e-9) -> np.ndarray:
    """Multilevel parallel Louvain. Returns dense integer labels per vertex."""
    import mlx.core as mx

    twom = graph.total_weight()
    g = graph
    orig2super = np.arange(graph.n, dtype=np.int64)
    q_prev = -1.0

    for level in range(max_levels):
        comm = _local_moving(g, resolution, twom, seed=random_state + 100 * level)
        _, comm_dense = np.unique(np.asarray(comm), return_inverse=True)
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


def _contract_dense(graph: Graph, comm_dense: np.ndarray, C: int) -> Graph:
    """Contract with already-dense (0..C-1) labels."""
    import mlx.core as mx

    comm = mx.array(comm_dense.astype(np.int32))
    csrc = comm[graph.edge_src]
    cdst = comm[graph.indices]
    key = csrc.astype(mx.int64) * C + cdst.astype(mx.int64)
    order = mx.argsort(key)
    seg_id, S, starts = _segments(key[order])
    w = segment_sum(seg_id, S, graph.weights[order])
    new_src = segment_head(seg_id, S, starts, csrc[order])
    new_dst = segment_head(seg_id, S, starts, cdst[order])
    return Graph.from_coo(np.asarray(new_src), np.asarray(new_dst), np.asarray(w), C)
