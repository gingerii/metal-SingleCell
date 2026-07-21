"""EXPERIMENTAL raw-Metal fused Louvain: cuGraph-style single-dispatch local moving.

Workaround #2 for the two M3 limits, by SYNTHESIZING the missing primitives in raw
Metal Shading Language (via MLX ``metal_kernel`` with a ``header``):

* **grid-wide barrier** (Metal has none across threadgroups) — a sense-reversing
  barrier over a device ``atomic_uint`` counter, launched with a small, co-resident
  number of threadgroups (validated: works to ≥24 TGs on M3). Lets one dispatch run
  many synchronized phases.
* **float atomic-add** (Metal has no native float atomics) — a compare-and-swap loop
  on a ``device atomic_uint`` reinterpreted as float. Lets Σtot be maintained
  PER-COLOR inside the kernel (the freshness that fuzzy real graphs need for quality,
  which the per-pass-snapshot hybrid/fused variants could not provide).

One dispatch does: seed comm → loop passes [ in-kernel Luby coloring → for each color:
two-phase (compute targets from current Σtot; barrier; apply moves + atomic Σtot
updates) ] → converge. Σtot device buffer holds float bits (zero-init = 0.0f).

NB device outputs are zero-initialized by MLX (verified) — that bootstraps the atomic
counters and Σtot. Threadgroup count G is kept small for guaranteed co-residency
(else the spin-barrier deadlocks until the GPU watchdog fires).

⚠️ EXPERIMENT OUTCOME — NOT PRODUCTION-VIABLE (kept for the record). Deep dive (round 2)
RE-ROOTED the failure — it is NOT a deadlock and occupancy control would NOT help:
1. The spin-barrier works; G=8 COMPLETES (~254ms). Earlier "hangs" were just the multilevel
   non-convergence on contracted graphs, not the barrier.
2. **Metal atomics are RELAXED-ONLY** — no acquire/release/seq_cst in MSL (the system Metal
   compiler rejects them). So no lock-free cross-threadgroup happens-before is constructible.
3. With relaxed-only, correct cross-TG sharing needs ALL shared arrays accessed as relaxed
   ATOMICS (Apple's coherent cache; plain device stores stay per-core → garbage Q=0.05).
   Making comm/color/sel/tgt atomic raised G=8 single-level to Q=0.62 — but:
   (a) atomic accesses in the O(deg^2) hot loop are SLOWER than the production path
       (2.0s vs 1.5s); (b) the local-neighbor-snapshot speedup (599ms) needs a degree cap
       that drops hub vertices → fragments (Q=0.40 / 337 comms); (c) residual relaxed-ordering
       staleness means G>1 never matches G=1/current EXACTLY (62 vs ~15 comms).
CONCLUSION: the blocker is Metal's relaxed-only atomic memory model (an MSL language limit a
raw Metal extension also has), NOT co-residency. Occupancy control addresses deadlock, which
is not the problem. See RESULTS_clustering_workarounds.md. Use production `louvain`/`leiden`.
"""

from __future__ import annotations

import numpy as np

from .csr_graph import Graph
from .louvain import _DEGREE_CAP, _contract_dense
from .primitives import modularity

_G = 4          # threadgroups (co-resident; conservative)
_T = 256        # threads per threadgroup
_MAX_ROUNDS = 64

_HEADER = r"""
inline void aaf(device atomic_uint* a, float v) {           // atomic float add (CAS)
    uint old = atomic_load_explicit(a, memory_order_relaxed);
    uint des;
    do { float f = as_type<float>(old) + v; des = as_type<uint>(f); }
    while (!atomic_compare_exchange_weak_explicit(a, &old, des,
            memory_order_relaxed, memory_order_relaxed));
}
inline float ldf(device atomic_uint* a) {
    return as_type<float>(atomic_load_explicit(a, memory_order_relaxed));
}
// Metal atomics are RELAXED-ONLY (no acquire/release/seq_cst in MSL). Cross-threadgroup
// visibility therefore relies on Apple's COHERENT device memory: relaxed-atomic accesses
// go through the coherent cache, whereas plain device stores can stay per-core (invisible
// to other TGs). So all SHARED mutable arrays (comm/color/sel/tgt) are accessed via these
// relaxed-atomic load/store helpers; combined with the mem_device fences in the barrier,
// that is the strongest cross-TG ordering Metal allows.
inline int  ldi(device int* a, uint i) {
    return atomic_load_explicit((device atomic_int*)(a + i), memory_order_relaxed);
}
inline void sti(device int* a, uint i, int v) {
    atomic_store_explicit((device atomic_int*)(a + i), v, memory_order_relaxed);
}
inline void gbar(device atomic_uint* cnt, device atomic_uint* sen,
                 uint G, uint lid, thread uint& ls) {        // grid-wide barrier
    threadgroup_barrier(mem_flags::mem_device);              // flush this TG's writes
    if (lid == 0) {
        uint s = 1u - ls;
        uint old = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed);
        if (old == G - 1u) {
            atomic_store_explicit(cnt, 0u, memory_order_relaxed);
            atomic_store_explicit(sen, s, memory_order_relaxed);
        } else {
            while (atomic_load_explicit(sen, memory_order_relaxed) != s) { }
        }
        ls = s;
    }
    threadgroup_barrier(mem_flags::mem_device);              // refresh before reading
}
"""

_SRC = r"""
    uint lid = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.x * threads_per_threadgroup.x + lid;
    uint n = (uint)dims[0];
    uint G = (uint)dims[1];
    uint max_passes = (uint)dims[2];
    uint max_rounds = (uint)dims[3];
    uint seed = (uint)dims[4];
    uint stride = G * threads_per_threadgroup.x;
    float inv = 1.0f / twom[0];
    float rs = res[0];
    uint capv = (uint)cap[0];

    device atomic_uint* cnt = (device atomic_uint*)bar;
    device atomic_uint* sen = (device atomic_uint*)(bar + 1);
    device atomic_uint* mv  = (device atomic_uint*)movecount;
    device atomic_uint* sigA = (device atomic_uint*)sig;
    thread uint ls = 0u;

    for (uint v = gid; v < n; v += stride) { sti(comm, v, (int)comm_in[v]); }
    gbar(cnt, sen, G, lid, ls);

    for (uint pass = 0; pass < max_passes; ++pass) {
        // ---- build Σtot fresh from comm (float bits, atomic add) ----
        for (uint v = gid; v < n; v += stride) {
            atomic_store_explicit((device atomic_uint*)(sig + v), as_type<uint>(0.0f),
                                  memory_order_relaxed);
            sti(color, v, -1);
        }
        gbar(cnt, sen, G, lid, ls);
        for (uint v = gid; v < n; v += stride) { aaf(&sigA[ldi(comm, v)], k[v]); }
        gbar(cnt, sen, G, lid, ls);

        // ---- fresh Luby coloring (fixed #rounds; uncolored stay -1 = skipped) ----
        for (uint r = 0; r < max_rounds; ++r) {
            for (uint v = gid; v < n; v += stride) {
                if (ldi(color, v) >= 0) { continue; }
                uint hv = (v * 2654435761u) ^ (r * 40503u) ^ seed; hv ^= hv >> 15;
                int pv = (int)(hv & 0x7fffffu);
                uint s = indptr[v]; uint e = indptr[v + 1];
                bool is_max = true;
                for (uint a = s; a < e; ++a) {
                    int na = indices[a];
                    if (na == (int)v || ldi(color, (uint)na) >= 0) { continue; }
                    uint hn = ((uint)na * 2654435761u) ^ (r * 40503u) ^ seed; hn ^= hn >> 15;
                    int pn = (int)(hn & 0x7fffffu);
                    if (pn > pv || (pn == pv && na < (int)v)) { is_max = false; break; }
                }
                sti(sel, v, is_max ? 1 : 0);
            }
            gbar(cnt, sen, G, lid, ls);
            for (uint v = gid; v < n; v += stride) {
                if (ldi(color, v) < 0 && ldi(sel, v) == 1) { sti(color, v, (int)r); }
            }
            gbar(cnt, sen, G, lid, ls);
        }

        if (lid == 0 && threadgroup_position_in_grid.x == 0) {
            atomic_store_explicit(mv, 0u, memory_order_relaxed);
        }
        gbar(cnt, sen, G, lid, ls);

        // ---- colored moves: per color, two-phase (compute target | apply) ----
        for (uint c = 0; c < max_rounds; ++c) {
            for (uint v = gid; v < n; v += stride) {
                int cur0 = ldi(comm, v);
                sti(tgt, v, cur0);
                if (ldi(color, v) != (int)c) { continue; }
                uint s = indptr[v]; uint e = indptr[v + 1];
                uint deg = e - s;
                if (deg > 64u) { continue; }   // local-snapshot cap (degree>64 -> skipped)
                // Snapshot neighbor communities ONCE (O(deg) atomic loads) into a
                // thread-local array, then dedup + weight-sum on the local copy (no
                // atomics) — turns the O(deg^2) hot loop from atomic-bound to register-fast.
                int lc[64];
                for (uint a = 0; a < deg; ++a) { lc[a] = ldi(comm, (uint)indices[s + a]); }
                int curr = cur0;
                float kv = k[v];
                float best_g = -1e30f; int best_c = curr; float stay_g = 0.0f;
                for (uint a = 0; a < deg; ++a) {
                    int na = indices[s + a];
                    if (na == (int)v) { continue; }
                    int ca = lc[a];
                    bool first = true;
                    for (uint b = 0; b < a; ++b) {
                        if (indices[s + b] != (int)v && lc[b] == ca) { first = false; break; }
                    }
                    if (!first) { continue; }
                    float wc = 0.0f;
                    for (uint b = 0; b < deg; ++b) {
                        if (indices[s + b] != (int)v && lc[b] == ca) { wc += weights[s + b]; }
                    }
                    float sg = ldf(&sigA[ca]) - (ca == curr ? kv : 0.0f);
                    float g = wc - rs * sg * kv * inv;
                    if (ca == curr) { stay_g = g; }
                    if (g > best_g || (g == best_g && ca < best_c)) { best_g = g; best_c = ca; }
                }
                sti(tgt, v, (best_g > stay_g + 1e-9f) ? best_c : curr);
            }
            gbar(cnt, sen, G, lid, ls);
            for (uint v = gid; v < n; v += stride) {
                if (ldi(color, v) != (int)c) { continue; }
                int oldc = ldi(comm, v); int newc = ldi(tgt, v);
                if (newc != oldc) {
                    float kv = k[v];
                    aaf(&sigA[oldc], -kv);
                    aaf(&sigA[newc], kv);
                    sti(comm, v, newc);
                    atomic_fetch_add_explicit(mv, 1u, memory_order_relaxed);
                }
            }
            gbar(cnt, sen, G, lid, ls);
        }

        if (atomic_load_explicit(mv, memory_order_relaxed) == 0u) { break; }
    }
"""


def _fused_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="louvain_fused_raw",
        input_names=["indptr", "indices", "weights", "comm_in", "k", "res", "twom", "cap", "dims"],
        output_names=["comm", "sig", "color", "sel", "tgt", "bar", "movecount"],
        source=_SRC,
        header=_HEADER,
    )


def _local_moving_fused_raw(graph: Graph, resolution: float, twom: float, seed: int = 0,
                            max_passes: int = 100, init_comm=None):
    import mlx.core as mx

    n = graph.n
    k = graph.degrees()
    comm_in = (mx.arange(n, dtype=mx.int32) if init_comm is None
               else mx.array(np.asarray(init_comm).astype(np.int32)))
    kernel = _fused_kernel()
    comm, *_ = kernel(
        inputs=[graph.indptr, graph.indices, graph.weights, comm_in, k,
                mx.array([resolution], dtype=mx.float32), mx.array([twom], dtype=mx.float32),
                mx.array([_DEGREE_CAP], dtype=mx.int32),
                mx.array([n, _G, max_passes, _MAX_ROUNDS, seed], dtype=mx.uint32)],
        grid=(_G * _T, 1, 1),
        threadgroup=(_T, 1, 1),
        output_shapes=[(n,), (n,), (n,), (n,), (n,), (2,), (1,)],
        output_dtypes=[mx.int32, mx.uint32, mx.int32, mx.int32, mx.int32, mx.uint32, mx.uint32],
    )
    mx.eval(comm)
    return np.asarray(comm).astype(np.int64)


def louvain_fused_raw(graph: Graph, resolution: float = 1.0, random_state: int = 0,
                      max_levels: int = 20, tol: float = 1e-9) -> np.ndarray:
    """Multilevel Louvain using the raw-Metal fused single-dispatch local moving."""
    import mlx.core as mx

    twom = graph.total_weight()
    g = graph
    orig2super = np.arange(graph.n, dtype=np.int64)
    q_prev = -1.0
    for level in range(max_levels):
        comm = _local_moving_fused_raw(g, resolution, twom, seed=random_state + 100 * level)
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
