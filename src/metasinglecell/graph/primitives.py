"""GPU graph primitives: sort-based segment reductions and the operations parallel
Louvain/Leiden are built from.

All run on the Metal GPU via MLX. The segment reduction (which MLX lacks natively)
is sort-by-key + cumsum-derived segment ids + scatter-add — the same pattern that
sorted 20M keys in <0.1s in feasibility testing.
"""

from __future__ import annotations

import numpy as np

from .csr_graph import Graph


def _segments(keys_sorted):
    """Segment bookkeeping for sorted keys: ``(seg_id, n_segments, starts)``.

    int64 keys are fine here (compared/sorted only); we never *scatter* them, since
    GPU scatter doesn't support int64 — callers reconstruct representatives from
    int32 arrays via :func:`segment_head`.
    """
    import mlx.core as mx

    starts = mx.concatenate([mx.array([True]), keys_sorted[1:] != keys_sorted[:-1]])
    seg_id = mx.cumsum(starts.astype(mx.int32)) - 1
    return seg_id, int(seg_id[-1].item()) + 1, starts


def segment_sum(seg_id, n_seg, vals):
    """Sum ``vals`` within each segment (scatter-add)."""
    import mlx.core as mx

    return mx.zeros((n_seg,), dtype=vals.dtype).at[seg_id].add(vals)


def segment_head(seg_id, n_seg, starts, arr_i32):
    """Representative (head) value per segment from an int32 array."""
    import mlx.core as mx

    return mx.zeros((n_seg,), dtype=mx.int32).at[seg_id].add(
        mx.where(starts, arr_i32, mx.zeros_like(arr_i32)))


def neighbor_community_weights(graph: Graph, comm):
    """Per (vertex, adjacent-community) summed edge weight.

    Returns ``(src, community, weight)`` MLX arrays: for each vertex, the total
    edge weight to each community it touches. Core of Louvain local-moving.
    """
    import mlx.core as mx

    C = int(comm.max().item()) + 1
    cdst = comm[graph.indices]                       # int32 community of each dst
    key = graph.edge_src.astype(mx.int64) * C + cdst.astype(mx.int64)
    order = mx.argsort(key)
    seg_id, S, starts = _segments(key[order])
    w = segment_sum(seg_id, S, graph.weights[order])
    src = segment_head(seg_id, S, starts, graph.edge_src[order])
    com = segment_head(seg_id, S, starts, cdst[order])
    return src, com, w


def contract(graph: Graph, comm) -> Graph:
    """Contract communities into super-vertices (Louvain aggregation phase).

    ``comm`` must be dense labels 0..C-1. Inter-community edge weights are summed;
    intra-community edges become self-loops. Returns a new, smaller ``Graph``.
    """
    import mlx.core as mx

    C = int(comm.max().item()) + 1
    csrc = comm[graph.edge_src]                      # int32 community ids
    cdst = comm[graph.indices]
    key = csrc.astype(mx.int64) * C + cdst.astype(mx.int64)
    order = mx.argsort(key)
    seg_id, S, starts = _segments(key[order])
    w = segment_sum(seg_id, S, graph.weights[order])
    new_src = segment_head(seg_id, S, starts, csrc[order])
    new_dst = segment_head(seg_id, S, starts, cdst[order])
    return Graph.from_coo(np.asarray(new_src), np.asarray(new_dst), np.asarray(w), C)


def modularity(graph: Graph, comm, resolution: float = 1.0) -> float:
    """Modularity Q of a partition (igraph convention; final reduction in fp64).

    Q = sum_c [ Sigma_in_c / 2m - gamma (Sigma_tot_c / 2m)^2 ], with weights stored
    in both directions (so 2m = total directed weight).
    """
    import mlx.core as mx

    twom = graph.total_weight()
    C = int(comm.max().item()) + 1
    tot = mx.zeros((C,), dtype=mx.float32).at[comm].add(graph.degrees())
    same = (comm[graph.edge_src] == comm[graph.indices]).astype(mx.float32)
    inw = mx.zeros((C,), dtype=mx.float32).at[comm[graph.edge_src]].add(graph.weights * same)
    mx.eval(tot, inw)

    tot = np.asarray(tot, dtype=np.float64)
    inw = np.asarray(inw, dtype=np.float64)
    return float(np.sum(inw / twom - resolution * (tot / twom) ** 2))
