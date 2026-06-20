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
from .primitives import (_segments, modularity, neighbor_community_weights,
                         segment_head, segment_sum)

_INT32_MAX = 2_147_483_647


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
    """Colored local-moving from singletons. Returns MLX community labels."""
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32)

    for p in range(max_passes):
        color, n_colors = color_graph(graph, seed=seed + p)
        color_np = np.asarray(color)
        changed = False

        for c in range(n_colors):
            sigtot = mx.zeros((n,), dtype=mx.float32).at[comm].add(k)
            src, com, w = neighbor_community_weights(graph, comm)
            curr = comm[src]
            kv = k[src]
            sig_adj = sigtot[com] - mx.where(com == curr, kv, mx.zeros_like(kv))
            score = w - resolution * sig_adj * kv / twom

            seg_id, S, starts = _segments(src.astype(mx.int64))
            seg_max = mx.full((S,), -mx.inf, dtype=mx.float32).at[seg_id].maximum(score)
            is_best = score >= seg_max[seg_id] - 1e-9
            cand = mx.where(is_best, com, mx.full(com.shape, _INT32_MAX, dtype=mx.int32))
            best_com = mx.full((S,), _INT32_MAX, dtype=mx.int32).at[seg_id].minimum(cand)
            seg_src = segment_head(seg_id, S, starts, src)
            stay = mx.zeros((S,), dtype=mx.float32).at[seg_id].add(
                mx.where(com == curr, score, mx.zeros_like(score)))
            moved = seg_max > stay + 1e-9
            new_for_seg = mx.where(moved, best_com, comm[seg_src])

            target = (comm + 0).at[seg_src].add(new_for_seg - comm[seg_src])
            # apply only this color's vertices (independent set -> no interference)
            apply = mx.array(color_np == c)
            new_comm = mx.where(apply, target, comm)
            mx.eval(new_comm)
            if not bool(mx.all(new_comm == comm).item()):
                changed = True
            comm = new_comm

        if not changed:
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
