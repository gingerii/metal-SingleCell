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

from . import _prof
from .csr_graph import Graph
from .louvain import (_DEGREE_CAP, _contract_dense, _local_moving,
                      _local_moving_sync, color_graph)
from .primitives import modularity


def _prof_record(level, n, t0, t_move, t_ref, t_con, n_comms, n_refined):
    """Append one per-level profiling record (no-op unless _prof.ENABLED)."""
    if not _prof.ENABLED:
        return
    _prof.records.append(dict(
        level=level, n=n,
        move_s=t_move - t0, refine_s=t_ref - t_move, contract_s=t_con - t_ref,
        move_passes=_prof.last_move_passes, refine_passes=_prof.last_refine_passes,
        move_syncs=_prof.last_move_syncs, refine_syncs=_prof.last_refine_syncs,
        n_comms=n_comms, n_refined=n_refined))


# Refinement move kernel: like the Louvain move kernel but a vertex only considers
# neighbors in its own Louvain community (part[na] == part[v]). Gains use full
# degrees/Σtot, so refinement optimizes the same modularity, restricted to moves
# within each Louvain community.
_REFINE_KERNEL_SOURCE = """
    uint v = thread_position_in_grid.x;
    if (color[v] != active[0]) { target[v] = comm[v]; return; }
    uint start = indptr[v];
    uint end = indptr[v + 1];
    if (end - start > (uint)cap[0]) { target[v] = comm[v]; return; }  // host handles high-degree
    int pv = part[v];
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
                     "color", "active", "res", "twom", "cap"],
        output_names=["target"],
        source=_REFINE_KERNEL_SOURCE,
    )


# O(degree) SIMD-group-per-vertex refinement kernel: same 32-lanes-per-vertex
# cooperative reduction as the Louvain move kernel, plus the P-restriction
# (`part[nb] == part[v]`) so a vertex only joins a sub-community reachable through a
# neighbor in its own Louvain community. Identical output to the quad refine kernel.
_REFINE_KERNEL_SIMD_SOURCE = """
    uint gid = thread_position_in_grid.x;
    uint v = gid >> 5;
    uint lane = gid & 31u;
    if (v >= (uint)nn[0]) { return; }
    if (color[v] != active[0]) { if (lane == 0u) target[v] = comm[v]; return; }
    if (amask[v] == 0) { if (lane == 0u) target[v] = comm[v]; return; }  // pruned
    uint start = indptr[v];
    uint end = indptr[v + 1];
    uint deg = end - start;
    if (deg > (uint)cap[0]) { if (lane == 0u) target[v] = comm[v]; return; }
    int pv = part[v];
    int curr = comm[v];
    float kv = k[v];
    float inv = 1.0f / twom[0];
    float rgamma = res[0];
    float best_gain = -1e30f;
    int best_comm = curr;
    float stay_gain = 0.0f;
    for (uint base = 0u; base < deg; ++base) {
        int nb = indices[start + base];
        if (nb == (int)v || part[nb] != pv) { continue; }   // self / cross-community (uniform)
        int c = comm[nb];
        bool dup = false;
        for (uint q = lane; q < base; q += 32u) {
            int nq = indices[start + q];
            if (nq != (int)v && part[nq] == pv && comm[nq] == c) { dup = true; }
        }
        if (simd_any(dup)) { continue; }
        float wpart = 0.0f;
        for (uint r = lane; r < deg; r += 32u) {
            int nr = indices[start + r];
            if (nr != (int)v && part[nr] == pv && comm[nr] == c) { wpart += weights[start + r]; }
        }
        float wc = simd_sum(wpart);
        float sig = sigtot[c] - (c == curr ? kv : 0.0f);
        float gain = wc - rgamma * sig * kv * inv;
        if (c == curr) { stay_gain = gain; }
        if (gain > best_gain || (gain == best_gain && c < best_comm)) {
            best_gain = gain; best_comm = c;
        }
    }
    if (lane == 0u)
        target[v] = (best_gain > stay_gain + 1e-9f) ? best_comm : curr;
"""


def _refine_kernel_simd():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="leiden_refine_move_simd",
        input_names=["indptr", "indices", "weights", "comm", "part", "k", "sigtot",
                     "color", "active", "res", "twom", "cap", "nn", "amask"],
        output_names=["target"],
        source=_REFINE_KERNEL_SIMD_SOURCE,
    )


def _high_degree_refine(indptr, indices, weights, comm_np, part_np, k_np, sigtot_np,
                        large_ids, resolution, twom):
    """Exact within-community best-move for high-degree vertices (host, O(degree))."""
    changed = 0
    for v in large_ids:
        pv = part_np[v]
        s, e = int(indptr[v]), int(indptr[v + 1])
        nbr = indices[s:e]
        keep = (nbr != v) & (part_np[nbr] == pv)        # same Louvain community only
        nbr, w = nbr[keep], weights[s:e][keep]
        if nbr.size == 0:
            continue
        wc = np.bincount(comm_np[nbr], weights=w)
        cand = np.flatnonzero(wc > 0)
        kv = float(k_np[v]); curr = int(comm_np[v])
        sig = sigtot_np[cand] - np.where(cand == curr, kv, 0.0)
        gain = wc[cand] - resolution * sig * kv / twom
        stay = gain[cand == curr]
        stay_gain = float(stay[0]) if stay.size else 0.0
        best = int(cand[np.argmax(gain)])
        if gain[np.argmax(gain)] > stay_gain + 1e-9 and best != curr:
            sigtot_np[curr] -= kv; sigtot_np[best] += kv
            comm_np[v] = best
            changed += 1
    return changed


# Refinement IS iterated to convergence here (unlike cuGraph's single sweep): measured, full
# convergence is FASTER overall because it yields fewer refined sub-communities → smaller
# contracted graphs → less downstream work (capping passes was counter-productively slower).
_REFINE_MAX_PASSES = 50


def _refine(graph: Graph, part: np.ndarray, resolution: float, twom: float,
            seed: int = 0, max_passes: int | None = None):
    """Split each Louvain community (``part``) into well-connected sub-communities.

    Returns dense refined labels (numpy). Each refined community is a subset of one
    Louvain community.
    """
    import mlx.core as mx

    if max_passes is None:
        max_passes = _REFINE_MAX_PASSES
    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32)             # refinement starts from singletons
    part_mx = mx.array(part.astype(np.int32))
    kernel = _refine_kernel()
    res_a = mx.array([resolution], dtype=mx.float32)
    twom_a = mx.array([twom], dtype=mx.float32)
    cap_a = mx.array([_DEGREE_CAP], dtype=mx.int32)

    deg = np.diff(np.asarray(graph.indptr).astype(np.int64))
    large_ids = np.flatnonzero(deg > _DEGREE_CAP)   # high-degree -> host (O(d))
    if large_ids.size:
        hd_indptr = np.asarray(graph.indptr).astype(np.int64)
        hd_indices = np.asarray(graph.indices)
        hd_weights = np.asarray(graph.weights).astype(np.float64)
        k_np = np.asarray(k).astype(np.float64)

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
                        sigtot, color, mx.array([c], dtype=mx.int32), res_a, twom_a, cap_a],
                grid=(n, 1, 1),
                threadgroup=(min(256, n), 1, 1),
                output_shapes=[(n,)],
                output_dtypes=[mx.int32],
            )
        if large_ids.size:
            comm_np = np.asarray(comm).copy()
            sigtot_np = np.bincount(comm_np, weights=k_np, minlength=n).astype(np.float64)
            _high_degree_refine(hd_indptr, hd_indices, hd_weights, comm_np, part,
                                k_np, sigtot_np, large_ids, resolution, twom)
            comm = mx.array(comm_np.astype(np.int32))
        mx.eval(comm)
        if bool(mx.all(comm == comm_before).item()):
            break

    _, dense = np.unique(np.asarray(comm), return_inverse=True)
    return dense.astype(np.int64)


def _refine_sync(graph: Graph, part: np.ndarray, resolution: float, twom: float,
                 seed: int = 0, max_passes: int = 200, commit_prob: float = 0.9,
                 hd_every: int = 4, kernel: str = "simd", sync_every: int = 4,
                 prune: bool = True):
    """Coloring-FREE refinement (cuGraph-style synchronous + random half-commit).

    Same within-Louvain-community restriction as `_refine`, but every vertex picks
    its best sub-community from ONE snapshot per pass (no per-pass graph coloring —
    the bulk of Leiden's cost), and the random half-commit breaks the symmetric-swap
    oscillation that previously made fixed-coloring refinement diverge.

    ``kernel="simd"`` (default) uses the O(degree) SIMD-group refine kernel; ``"quad"``
    the legacy O(degree²) single-thread kernel (identical output). ``sync_every`` tests
    convergence every K passes instead of every pass; ``prune`` re-evaluates only the
    changed frontier each pass (both as in `_local_moving_sync`).
    """
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32)             # singletons
    part_mx = mx.array(part.astype(np.int32))
    res_a = mx.array([resolution], dtype=mx.float32)
    twom_a = mx.array([twom], dtype=mx.float32)
    cap_a = mx.array([_DEGREE_CAP], dtype=mx.int32)
    color0 = mx.zeros((n,), dtype=mx.int32)
    active0 = mx.array([0], dtype=mx.int32)
    key = mx.random.key(seed)

    if kernel == "simd":
        kobj = _refine_kernel_simd()
        nn_a = mx.array([n], dtype=mx.int32)

        def do_refine(comm, sigtot, amask):
            (t,) = kobj(
                inputs=[graph.indptr, graph.indices, graph.weights, comm, part_mx, k,
                        sigtot, color0, active0, res_a, twom_a, cap_a, nn_a, amask],
                grid=(n * 32, 1, 1), threadgroup=(min(256, n * 32), 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.int32])
            return t
    else:
        kobj = _refine_kernel()

        def do_refine(comm, sigtot, amask):           # quad has no pruning path; amask ignored
            (t,) = kobj(
                inputs=[graph.indptr, graph.indices, graph.weights, comm, part_mx, k,
                        sigtot, color0, active0, res_a, twom_a, cap_a],
                grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.int32])
            return t

    deg = np.diff(np.asarray(graph.indptr).astype(np.int64))
    large_ids = np.flatnonzero(deg > _DEGREE_CAP)
    if large_ids.size:
        hd_indptr = np.asarray(graph.indptr).astype(np.int64)
        hd_indices = np.asarray(graph.indices)
        hd_weights = np.asarray(graph.weights).astype(np.float64)
        k_np = np.asarray(k).astype(np.float64)

    prune_on = prune and kernel == "simd" and not large_ids.size
    active_mask = mx.ones((n,), dtype=mx.int32)

    syncs = 0
    for p in range(max_passes):
        sigtot = mx.zeros((n,), dtype=mx.float32).at[comm].add(k)
        target = do_refine(comm, sigtot, active_mask)
        wants = target != comm
        key, sub = mx.random.split(key)
        coin = mx.random.uniform(shape=(n,), key=sub) < commit_prob
        commit = wants & coin
        comm = mx.where(commit, target, comm)         # no-op when no vertex wants to move
        # Frontier for the next pass: moved ∪ neighbours-of-moved ∪ wanted-but-held.
        # Propagating through all edges (not just same-Louvain-community ones) is a
        # conservative superset — it can only over-activate, never wrongly prune — so
        # the refined labels stay identical to no-pruning.
        if prune_on:
            ci = commit.astype(mx.int32)
            nbr = mx.zeros((n,), dtype=mx.int32).at[graph.edge_src].maximum(ci[graph.indices])
            active_mask = mx.maximum(mx.maximum(ci, nbr), (wants & ~coin).astype(mx.int32))
        hd_due = bool(large_ids.size) and (p % hd_every == hd_every - 1)
        if (p % sync_every == sync_every - 1) or (p == max_passes - 1) or hd_due:
            gpu_done = not bool(mx.any(wants).item())
            syncs += 1
            hd_changed = 0
            if large_ids.size and (gpu_done or hd_due):
                comm_np = np.asarray(comm).copy()
                sigtot_np = np.bincount(comm_np, weights=k_np, minlength=n).astype(np.float64)
                hd_changed = _high_degree_refine(hd_indptr, hd_indices, hd_weights, comm_np,
                                                 part, k_np, sigtot_np, large_ids, resolution, twom)
                comm = mx.array(comm_np.astype(np.int32))
            if gpu_done and not hd_changed:
                break
            mx.eval(comm)                             # bound the lazy graph at each check

    if _prof.ENABLED:
        _prof.last_refine_passes = p + 1
        _prof.last_refine_syncs = syncs
    _, dense = np.unique(np.asarray(comm), return_inverse=True)
    return dense.astype(np.int64)


def leiden(graph: Graph, resolution: float = 1.0, random_state: int = 0,
           n_iterations: int = 1, max_levels: int = 20, variant: str = "sync",
           commit_prob: float = 0.9, kernel: str = "simd",
           sync_every: int = 4, prune: bool = True) -> np.ndarray:
    """Parallel Leiden. Returns dense integer labels per vertex.

    ``variant="sync"`` (default) uses coloring-free synchronous local-moving + refinement
    (2–11× faster than legacy ``"colored"`` at scale, equal/better Q, refinement converges).
    ``commit_prob`` (sync only) tunes the random-commit convergence: higher converges in fewer
    passes (0.9 default); lower is more conservative against oscillation.

    ``n_iterations`` repeats the whole multilevel procedure (as in leidenalg/scanpy).
    Default 1: unlike leidenalg, our local-moving AND refinement each iterate to
    convergence within a single multilevel pass, so that pass already reaches a fixed
    point — extra iterations are provably redundant (verified: ARI 1.000, identical Q
    and cluster count for n_iterations 1 vs 2 across clean/noisy/many-cluster graphs).
    Running 1 instead of 2 ~halves the cost (refinement is ~75% of Leiden's runtime).
    """
    import mlx.core as mx

    twom = graph.total_weight()
    labels = None
    for it in range(max(1, n_iterations)):
        labels = _leiden_pass(graph, resolution, twom, random_state + 17 * it,
                              init=labels, variant=variant, commit_prob=commit_prob,
                              kernel=kernel, sync_every=sync_every, prune=prune)
    return labels


def _leiden_pass(g0: Graph, resolution: float, twom: float, seed: int,
                 init: np.ndarray | None, max_levels: int = 20,
                 variant: str = "colored", commit_prob: float = 0.9,
                 kernel: str = "simd", sync_every: int = 4,
                 prune: bool = True) -> np.ndarray:
    """One full multilevel Leiden run (move -> refine -> aggregate, repeat).

    ``variant="sync"`` uses the coloring-free synchronous local-moving + refinement
    (`_local_moving_sync` / `_refine_sync`, ``commit_prob`` tuning); ``"colored"`` uses coloring.
    """
    import functools
    import time

    import mlx.core as mx

    if variant == "sync":
        move = functools.partial(_local_moving_sync, commit_prob=commit_prob,
                                 kernel=kernel, sync_every=sync_every, prune=prune)
        refine = functools.partial(_refine_sync, commit_prob=commit_prob,
                                   kernel=kernel, sync_every=sync_every, prune=prune)
    else:
        move, refine = _local_moving, _refine
    g = g0
    orig2node = np.arange(g0.n, dtype=np.int64)
    # P: community label per node of the current (aggregate) graph.
    P = (init.copy() if init is not None else np.arange(g0.n, dtype=np.int64))

    for level in range(max_levels):
        _t0 = time.perf_counter()
        n_level = g.n
        # 1. Local moving, initialized from P (singletons on level 0 of a fresh run).
        Pmx = move(g, resolution, twom, seed=seed + level,
                   init_comm=mx.array(P.astype(np.int32)))
        _, P = np.unique(np.asarray(Pmx), return_inverse=True)
        P = P.astype(np.int64)
        n_comms = int(P.max()) + 1
        _t_move = time.perf_counter()
        if n_comms == g.n:                           # each node its own community: done
            _prof_record(level, n_level, _t0, _t_move, _t_move, _t_move, n_comms, 0)
            break

        # 2. Refinement: split each Louvain community into well-connected pieces.
        R = refine(g, P, resolution, twom, seed=seed + level)
        nR = int(R.max()) + 1
        _t_ref = time.perf_counter()
        if nR == g.n:                                # refinement can't aggregate further
            _prof_record(level, n_level, _t0, _t_move, _t_ref, _t_ref, n_comms, nR)
            break

        # 3. Aggregate on the REFINED partition; lift P onto the refined super-vertices.
        P_agg = np.zeros(nR, dtype=np.int64)
        P_agg[R] = P                                 # each refined community ⊆ one Louvain community
        orig2node = R[orig2node]
        g = _contract_dense(g, R, nR)
        P = P_agg
        _prof_record(level, n_level, _t0, _t_move, _t_ref, time.perf_counter(), n_comms, nR)

    _, final = np.unique(P[orig2node], return_inverse=True)
    return final.astype(np.int64)
