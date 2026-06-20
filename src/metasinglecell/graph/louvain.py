"""Parallel multilevel Louvain on the Metal GPU (MLX).

Each level runs synchronous (Jacobi) local-moving — every vertex picks, in
parallel, the neighbor community giving the best modularity gain — then the graph
is contracted and the level repeats until modularity stops improving.

Two devices keep the synchronous scheme correct/convergent:
* a vectorized 2-cycle **swap-breaker** (without it, a graph of singletons just
  swaps labels forever instead of merging), and
* a **modularity-monotonicity guard** (a pass that doesn't raise Q ends the level),
  which also catches the rare longer oscillation.

Validated by modularity (>= igraph) rather than label parity — parallel Louvain is
a different, stochastic optimizer. See the `graph-clustering` skill.
"""

from __future__ import annotations

import numpy as np

from .csr_graph import Graph
from .primitives import (_segments, modularity, neighbor_community_weights,
                         segment_head, segment_sum)

_INT32_MAX = 2_147_483_647


def _local_moving(graph: Graph, resolution: float, twom: float,
                  max_passes: int = 100, tol: float = 1e-9):
    """Optimize communities on `graph` starting from singletons. Returns MLX comm."""
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32)
    q_prev = modularity(graph, comm, resolution)
    idx = mx.arange(n, dtype=mx.int32)

    for _ in range(max_passes):
        sigtot = mx.zeros((n,), dtype=mx.float32).at[comm].add(k)

        # per (vertex, adjacent-community) modularity-gain score (vertex isolated first)
        src, com, w = neighbor_community_weights(graph, comm)
        curr = comm[src]
        kv = k[src]
        sig_adj = sigtot[com] - mx.where(com == curr, kv, mx.zeros_like(kv))
        score = w - resolution * sig_adj * kv / twom

        # segmented argmax over each vertex's candidate communities (ties -> lowest id)
        seg_id, S, starts = _segments(src.astype(mx.int64))
        seg_max = mx.full((S,), -mx.inf, dtype=mx.float32).at[seg_id].maximum(score)
        is_best = score >= seg_max[seg_id] - 1e-12
        cand = mx.where(is_best, com, mx.full(com.shape, _INT32_MAX, dtype=mx.int32))
        best_com = mx.full((S,), _INT32_MAX, dtype=mx.int32).at[seg_id].minimum(cand)
        seg_src = segment_head(seg_id, S, starts, src)

        # stay vs move: staying score is the (vertex, own-community) pair, else 0
        stay = mx.zeros((S,), dtype=mx.float32).at[seg_id].add(
            mx.where(com == curr, score, mx.zeros_like(score)))
        moved = seg_max > stay + tol
        new_for_seg = mx.where(moved, best_com, comm[seg_src])

        target = (comm + 0).at[seg_src].add(new_for_seg - comm[seg_src])

        # break 2-cycles (mutual targeting): cancel the higher-id vertex's move
        tt = target[target]
        is_swap = (tt == idx) & (target != idx)
        target = mx.where(is_swap & (idx > target), idx, target)
        mx.eval(target)

        if mx.all(target == comm).item():
            break
        q_new = modularity(graph, target, resolution)
        if q_new <= q_prev + tol:   # monotonicity guard: no gain -> stop
            break
        comm, q_prev = target, q_new

    return comm


def louvain(graph: Graph, resolution: float = 1.0, random_state: int = 0,
            max_levels: int = 20, tol: float = 1e-9) -> np.ndarray:
    """Multilevel parallel Louvain. Returns dense integer labels per vertex."""
    import mlx.core as mx

    twom = graph.total_weight()
    g = graph
    orig2super = np.arange(graph.n, dtype=np.int64)  # original vertex -> current super-node
    q_prev = -1.0

    for _ in range(max_levels):
        comm = _local_moving(g, resolution, twom)
        comm_np = np.asarray(comm)
        _, comm_dense = np.unique(comm_np, return_inverse=True)  # 0..C-1
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
    """Contract using already-dense (0..C-1) labels."""
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
