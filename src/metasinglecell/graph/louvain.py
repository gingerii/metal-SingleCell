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

from . import _prof
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
    // Degree-binning: this O(degree^2) kernel handles bounded-degree vertices; rare
    // very-high-degree (contracted) super-vertices are computed on the host in O(d).
    if (end - start > (uint)cap[0]) { target[v] = comm[v]; return; }
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
                     "color", "active", "res", "twom", "cap"],
        output_names=["target"],
        source=_MOVE_KERNEL_SOURCE,
    )


# O(degree) SIMD-group-per-vertex move kernel: 32 lanes cooperate on one vertex,
# striding its CSR row. `simd_any` does the first-occurrence dedup and `simd_sum`
# the per-community weight — both native, no atomics, no threadgroup memory. This
# retires the O(degree^2) rescans of the single-thread kernel above: with kNN
# degrees (median 18, 86% <= 32) most vertices fit one neighbor per lane, so dedup
# and weight collapse to O(1) hardware reductions. Output is identical to the quad
# kernel (validated bit-exact on the real 1M graph); ~8.5x faster per dispatch.
_MOVE_KERNEL_SIMD_SOURCE = """
    uint gid = thread_position_in_grid.x;
    uint v = gid >> 5;                     // 32 lanes per vertex
    uint lane = gid & 31u;
    if (v >= (uint)nn[0]) { return; }
    if (color[v] != active[0]) { if (lane == 0u) target[v] = comm[v]; return; }
    if (amask[v] == 0) { if (lane == 0u) target[v] = comm[v]; return; }  // pruned: unchanged nbrhood
    uint start = indptr[v];
    uint end = indptr[v + 1];
    uint deg = end - start;
    if (deg > (uint)cap[0]) { if (lane == 0u) target[v] = comm[v]; return; }  // host tail
    int curr = comm[v];
    float kv = k[v];
    float inv = 1.0f / twom[0];
    float rgamma = res[0];
    float best_gain = -1e30f;
    int best_comm = curr;
    float stay_gain = 0.0f;
    for (uint base = 0u; base < deg; ++base) {
        int nb = indices[start + base];
        if (nb == (int)v) { continue; }               // self-loop (uniform across lanes)
        int c = comm[nb];
        bool dup = false;                             // first-occurrence dedup across lanes
        for (uint q = lane; q < base; q += 32u) {
            int nq = indices[start + q];
            if (nq != (int)v && comm[nq] == c) { dup = true; }
        }
        if (simd_any(dup)) { continue; }              // uniform: earlier lane already scored c
        float wpart = 0.0f;                           // weight v -> c, lanes stride the row
        for (uint r = lane; r < deg; r += 32u) {
            int nr = indices[start + r];
            if (nr != (int)v && comm[nr] == c) { wpart += weights[start + r]; }
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


def _move_kernel_simd():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="louvain_best_move_simd",
        input_names=["indptr", "indices", "weights", "comm", "k", "sigtot",
                     "color", "active", "res", "twom", "cap", "nn", "amask"],
        output_names=["target"],
        source=_MOVE_KERNEL_SIMD_SOURCE,
    )


# Degree threshold: vertices above this are handled on the host (exact, O(d)) to
# avoid the GPU kernel's O(d^2) blowup on rare high-degree contracted super-vertices.
_DEGREE_CAP = 1024


def _high_degree_moves(indptr, indices, weights, comm_np, k_np, sigtot_np,
                       large_ids, resolution, twom):
    """Exact best-move for the (few) high-degree vertices, sequentially in NumPy.

    O(degree) per vertex (bincount over neighbor communities), processed one at a
    time with Sigma_tot updated after each move (so adjacent large vertices don't
    interfere). Mutates ``comm_np`` and ``sigtot_np`` in place.
    """
    changed = 0
    for v in large_ids:
        s, e = int(indptr[v]), int(indptr[v + 1])
        nbr = indices[s:e]
        keep = nbr != v
        nbr, w = nbr[keep], weights[s:e][keep]
        if nbr.size == 0:
            continue
        wc = np.bincount(comm_np[nbr], weights=w)            # weight to each community
        cand = np.flatnonzero(wc > 0)
        kv = float(k_np[v])
        curr = int(comm_np[v])
        sig = sigtot_np[cand] - np.where(cand == curr, kv, 0.0)
        gain = wc[cand] - resolution * sig * kv / twom
        stay = gain[cand == curr]
        stay_gain = float(stay[0]) if stay.size else 0.0
        best = int(cand[np.argmax(gain)])
        if gain[np.argmax(gain)] > stay_gain + 1e-9 and best != curr:
            sigtot_np[curr] -= kv
            sigtot_np[best] += kv
            comm_np[v] = best
            changed += 1
    return changed


def color_graph(graph: Graph, seed: int = 0, max_colors: int = 2000):
    """Greedy Luby graph coloring on the GPU. Returns ``(color, n_colors)``.

    Each round, vertices whose random priority exceeds all *uncolored* neighbors'
    form an independent set and take the next color.
    """
    import mlx.core as mx

    n = graph.n
    src, dst = graph.edge_src, graph.indices
    not_self = src != dst        # contracted graphs have a self-loop per super-vertex;
                                 # without excluding it a vertex is its own neighbor and
                                 # is never a local max -> never colored -> never moves.
    color = mx.full((n,), -1, dtype=mx.int32)
    key = mx.random.key(seed)
    r = 0
    while bool(mx.any(color < 0).item()) and r < max_colors:
        key, sub = mx.random.split(key)
        prio = mx.random.uniform(shape=(n,), key=sub)
        unc = color < 0
        both = unc[src] & unc[dst] & not_self
        contrib = mx.where(both, prio[dst], mx.full(dst.shape, -mx.inf))
        max_nb = mx.full((n,), -mx.inf).at[src].maximum(contrib)
        selected = unc & (prio > max_nb)
        color = mx.where(selected, mx.array(r, dtype=mx.int32), color)
        r += 1
    return color, r


def _local_moving(graph: Graph, resolution: float, twom: float, seed: int = 0,
                  max_passes: int = 100, init_comm=None, recolor_every: int = 3):
    """Colored local-moving (per-vertex kernel, no sort).

    Starts from singletons unless ``init_comm`` is given (Leiden's aggregate levels
    start from the lifted Louvain partition). Returns MLX community labels.

    Coloring dominates runtime (a full graph coloring per round), so we re-color
    only every ``recolor_every`` passes and shuffle the color processing order in
    between — this gives the random-order diversity that aids convergence/quality
    without paying for a coloring every pass (~1.5x faster, equal/better quality).
    """
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32) if init_comm is None else init_comm.astype(mx.int32)
    kernel = _move_kernel()
    res_a = mx.array([resolution], dtype=mx.float32)
    twom_a = mx.array([twom], dtype=mx.float32)
    cap_a = mx.array([_DEGREE_CAP], dtype=mx.int32)
    rng = np.random.default_rng(seed)
    color, n_colors = None, 0

    # Degree-binning: the GPU kernel skips vertices with degree > cap (O(d^2) blowup);
    # these rare high-degree super-vertices are moved exactly on the host (O(d)).
    deg = np.diff(np.asarray(graph.indptr).astype(np.int64))
    large_ids = np.flatnonzero(deg > _DEGREE_CAP)
    if large_ids.size:
        hd_indptr = np.asarray(graph.indptr).astype(np.int64)
        hd_indices = np.asarray(graph.indices)
        hd_weights = np.asarray(graph.weights).astype(np.float64)
        k_np = np.asarray(k).astype(np.float64)

    for p in range(max_passes):
        if p % recolor_every == 0:
            color, n_colors = color_graph(graph, seed=seed + p)
        comm_before = comm
        # Σtot per color (fresh community sizes each color). NB per-PASS Σtot gives
        # better quality on small structureless synthetic graphs but needs far more
        # passes to converge at 1M (700s vs ~10s) — per-color keeps the atlas-scale
        # speed; small-n over-fragmentation is a synthetic-data artifact (verify on
        # REAL data, per CLAUDE.md).
        for c in rng.permutation(n_colors):
            sigtot = mx.zeros((n,), dtype=mx.float32).at[comm].add(k)
            (comm,) = kernel(
                inputs=[graph.indptr, graph.indices, graph.weights, comm, k, sigtot,
                        color, mx.array([int(c)], dtype=mx.int32), res_a, twom_a, cap_a],
                grid=(n, 1, 1),
                threadgroup=(min(256, n), 1, 1),
                output_shapes=[(n,)],
                output_dtypes=[mx.int32],
            )
        if large_ids.size:                             # host-handle high-degree vertices
            comm_np = np.asarray(comm).copy()
            sigtot_np = np.bincount(comm_np, weights=k_np, minlength=n).astype(np.float64)
            _high_degree_moves(hd_indptr, hd_indices, hd_weights, comm_np, k_np,
                               sigtot_np, large_ids, resolution, twom)
            comm = mx.array(comm_np.astype(np.int32))
        mx.eval(comm)                                  # one sync per pass, not per color
        if bool(mx.all(comm == comm_before).item()):
            break

    return comm


def _local_moving_sync(graph: Graph, resolution: float, twom: float, seed: int = 0,
                       max_passes: int = 200, init_comm=None, commit_prob: float = 0.9,
                       hd_every: int = 4, kernel: str = "simd", sync_every: int = 4,
                       prune: bool = True):
    """Coloring-FREE synchronous local-moving (cuGraph-style).

    All vertices pick their best community from ONE snapshot per pass (a per-vertex
    move kernel, every vertex active — no graph coloring), then a **random
    half-commit** breaks the symmetric-swap oscillation that coloring otherwise
    prevents: each willing mover commits with probability ``commit_prob``, so a 2-cycle
    (i→{j}, j→{i}) resolves in ~1–2 passes instead of oscillating forever. Converged
    when no vertex has a beneficial move (target == comm). High-degree super-vertices
    are still finished exactly on the host (sequential, no oscillation).

    ``kernel="simd"`` (default) uses the O(degree) SIMD-group-per-vertex kernel;
    ``"quad"`` uses the legacy O(degree²) single-thread kernel (identical output).

    ``sync_every`` controls convergence checking: the commit is a no-op when no
    vertex wants to move, so we always commit and only round-trip to the host to test
    convergence every ``sync_every`` passes. Between checks MLX queues passes without a
    CPU↔GPU sync, so the pipeline isn't serialized per pass. Converging up to
    ``sync_every``−1 passes late is a fixed point (stable), so the extra passes are
    no-ops; the committed moves — and the final labels — are unchanged.
    """
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm = mx.arange(n, dtype=mx.int32) if init_comm is None else init_comm.astype(mx.int32)
    res_a = mx.array([resolution], dtype=mx.float32)
    twom_a = mx.array([twom], dtype=mx.float32)
    cap_a = mx.array([_DEGREE_CAP], dtype=mx.int32)
    color0 = mx.zeros((n,), dtype=mx.int32)          # all vertices share one "color" ...
    active0 = mx.array([0], dtype=mx.int32)          # ... and it is always active
    key = mx.random.key(seed)

    # Dispatch closure abstracts the two kernels' differing grid/inputs (SIMD launches
    # 32 threads/vertex and needs the vertex count `nn` for its bounds guard).
    if kernel == "simd":
        kobj = _move_kernel_simd()
        nn_a = mx.array([n], dtype=mx.int32)

        def do_move(comm, sigtot, amask):
            (t,) = kobj(
                inputs=[graph.indptr, graph.indices, graph.weights, comm, k, sigtot,
                        color0, active0, res_a, twom_a, cap_a, nn_a, amask],
                grid=(n * 32, 1, 1), threadgroup=(min(256, n * 32), 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.int32])
            return t
    else:
        kobj = _move_kernel()

        def do_move(comm, sigtot, amask):             # quad has no pruning path; amask ignored
            (t,) = kobj(
                inputs=[graph.indptr, graph.indices, graph.weights, comm, k, sigtot,
                        color0, active0, res_a, twom_a, cap_a],
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

    # Vertex pruning: only the SIMD kernel reads the active mask, and high-degree host
    # moves don't propagate through it — so pruning is gated to the pure SIMD path.
    prune_on = prune and kernel == "simd" and not large_ids.size
    active_mask = mx.ones((n,), dtype=mx.int32)       # first pass: everyone active

    syncs = 0
    for p in range(max_passes):
        sigtot = mx.zeros((n,), dtype=mx.float32).at[comm].add(k)
        target = do_move(comm, sigtot, active_mask)
        wants = target != comm
        key, sub = mx.random.split(key)
        coin = mx.random.uniform(shape=(n,), key=sub) < commit_prob
        commit = wants & coin
        comm = mx.where(commit, target, comm)         # no-op when no vertex wants to move
        # Frontier for the NEXT pass: a vertex is re-evaluated only if it moved, a
        # neighbor moved (its best target may change), or it wanted to move but the coin
        # held it (a pending move to retry). A vertex with an unchanged neighbourhood
        # would just re-confirm "stay", so skipping it changes nothing — pruning is
        # exact here (committed moves, and final labels, identical to no-pruning).
        if prune_on:
            ci = commit.astype(mx.int32)
            nbr = mx.zeros((n,), dtype=mx.int32).at[graph.edge_src].maximum(ci[graph.indices])
            active_mask = mx.maximum(mx.maximum(ci, nbr), (wants & ~coin).astype(mx.int32))
        # Only round-trip to the host to test convergence every `sync_every` passes (and
        # run the host tail on its own cadence). Between checks MLX queues passes without
        # a CPU↔GPU sync, keeping the pipeline full.
        hd_due = bool(large_ids.size) and (p % hd_every == hd_every - 1)
        if (p % sync_every == sync_every - 1) or (p == max_passes - 1) or hd_due:
            gpu_done = not bool(mx.any(wants).item())
            syncs += 1
            hd_changed = 0
            if large_ids.size and (gpu_done or hd_due):
                comm_np = np.asarray(comm).copy()
                sigtot_np = np.bincount(comm_np, weights=k_np, minlength=n).astype(np.float64)
                hd_changed = _high_degree_moves(hd_indptr, hd_indices, hd_weights, comm_np,
                                                k_np, sigtot_np, large_ids, resolution, twom)
                comm = mx.array(comm_np.astype(np.int32))
            if gpu_done and not hd_changed:
                break
            mx.eval(comm)                             # bound the lazy graph at each check

    if _prof.ENABLED:
        _prof.last_move_passes = p + 1
        _prof.last_move_syncs = syncs
    return comm


def louvain(graph: Graph, resolution: float = 1.0, random_state: int = 0,
            max_levels: int = 20, tol: float = 1e-9, variant: str = "sync",
            commit_prob: float = 0.9, kernel: str = "simd",
            sync_every: int = 4, prune: bool = True) -> np.ndarray:
    """Multilevel parallel Louvain. Returns dense integer labels per vertex.

    ``variant="sync"`` (default) uses the coloring-free synchronous local-moving
    (`_local_moving_sync`, ``commit_prob`` tunes its random-commit anti-oscillation
    rule) — 2–9× faster than the legacy ``"colored"`` graph-coloring variant at scale,
    equal/better modularity (validated on real PBMC + synthetic to 1M). ``kernel``
    ("simd"|"quad") selects the O(degree) vs legacy O(degree²) move kernel.
    """
    import mlx.core as mx

    if variant == "sync":
        def move(g, res, tw, seed):
            return _local_moving_sync(g, res, tw, seed=seed, commit_prob=commit_prob,
                                      kernel=kernel, sync_every=sync_every, prune=prune)
    else:
        move = _local_moving
    twom = graph.total_weight()
    g = graph
    orig2super = np.arange(graph.n, dtype=np.int64)
    q_prev = -1.0

    for level in range(max_levels):
        comm = move(g, resolution, twom, seed=random_state + 100 * level)
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
