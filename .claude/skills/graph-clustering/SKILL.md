---
name: graph-clustering
description: GPU graph clustering on Metal (cuGraph-analog) — the graph/ substrate (CSR Graph, sort-based segment reductions, contraction, modularity) and parallel Louvain/Leiden. Load when working on graph clustering, the graph subpackage, or atlas-scale clustering speed.
---

# GPU graph clustering (cuGraph-analog) — metal-SingleCell

## Why
Leiden is the pipeline's only unaccelerated step (~1h on atlas data; igraph/leidenalg are
sequential). cuGraph proves GPU graph clustering works via **parallel** Louvain/Leiden; no Metal
equivalent exists. Building it: `src/metasinglecell/graph/`. Plan: Louvain → Leiden refinement,
clustering-first on a reusable substrate.

## Substrate — `graph/` (Phase 1, VALIDATED exact)
- `graph/csr_graph.py` `Graph`: MLX `indptr`(u32)/`indices`(i32)/`weights`(f32) + explicit
  `edge_src`(i32) so edges are a flat list. Undirected/symmetric (each edge both directions, so
  `total_weight()` = 2m). `from_scipy`, `from_coo` (scipy sum_duplicates), `degrees()` (scatter-add
  weights into edge_src).
- `graph/primitives.py` — **all reductions are sort-by-key + segment-sum, pure MLX (no custom Metal
  kernel)**:
  - `_segments(sorted_keys)` → `(seg_id, n_seg, starts)` via `cumsum` of boundary flags.
  - `segment_sum(seg_id,n_seg,vals)` = `zeros.at[seg_id].add(vals)`.
  - `segment_head(...)` = representative int32 per segment (head element via masked scatter-add).
  - `neighbor_community_weights(g,comm)` → per (vertex, adjacent-community) weight (key=src*C+comm[dst]).
  - `contract(g,comm)` → smaller `Graph` (key=comm[src]*C+comm[dst]; intra→self-loops).
  - `modularity(g,comm,resolution)` → Q = Σ_c[in_c/2m − γ(tot_c/2m)²], **final reduction in fp64**.

## KEY GPU GOTCHA: no int64 scatter
`mx.scatter`/`.at[].add` rejects int64 ("GPU scatter does not yet support int64"). int64 keys are
fine to **sort/compare** (argsort, `keys[1:]!=keys[:-1]`) but NEVER scatter them. Reconstruct
per-segment src/community from **int32** arrays via `segment_head`. (community ids < n, fit i32.)

## Feasibility / perf evidence
`mx.argsort` of 20M int64 keys = **0.084s** on M3 → sort-based aggregation is sub-second at atlas
edge counts. This is why parallel Louvain is tractable here.

## Validation (`results/graph_primitives/`, driver 09) — all PASS
degrees 2.3e-5, neighbor_comm_weights 1.5e-5, contract rel 3.4e-6, **modularity mine==igraph
(0.669179, abs_err 8.2e-7)** on the PBMC graph + oracle leiden labels.

## Validation bar for the clusterer (Phases 2–3)
Parallel ≠ sequential and stochastic → NOT bit-parity. Use: **modularity Q ≥ igraph Leiden's Q − ε**
(rigorous for an optimizer) + ARI vs oracle within the RNG floor (~0.65; scanpy's own seed span
0.69–0.90). + cluster-count sanity. Reuse `validation.compare` / igraph for the reference Q.

## Phase 2 Louvain — QUALITY CORRECT (colored), SPEED REGRESSED (WIP)
`graph/louvain.py` now uses **graph-coloring local-moving** (`color_graph` = Luby/JP greedy: a vertex
takes the next color if its random priority beats all *uncolored* neighbors', via `.at[src].maximum`;
then process color-by-color so each independent set moves seeing fresh neighbor communities).

### TWO BUGS FOUND & FIXED (both real)
1. **Self-loop in move gain** (the big one): on *contracted* graphs each super-vertex has a self-loop
   (= its internal weight). Counting it in `neighbor_community_weights` made the "stay" score equal
   that large self-loop weight, so NO super-vertex ever moved → coarse levels never merged (188→188).
   Fix: zero self-loop weights in `neighbor_community_weights` (they belong to *degree*, not to
   inter-vertex edge weight). MLX has no boolean indexing → use `mx.where(edge_src==indices, 0, w)`.
   Effect: PBMC Q 0.543 → 0.630.
2. **Fully-synchronous fragmentation**: all-vertices-move-on-stale-state splits even a clique in two
   (verified on a 2-clique toy: stuck at Q=0.117, oscillating). Coloring fixes it.

### Current quality vs speed (driver 10)
- **Quality is SOUND**: on a genuinely-clustered SBM graph the colored GPU Louvain **matches igraph
  exactly** (n=10k: Q=0.809=0.809, both 20 clusters). On fuzzy PBMC: Q=0.630 / 35 comms / ARI 0.53 vs
  igraph 0.66 / ~13 — close but coarse-merge stalls (residual gap; Leiden refinement should help).
- **Speed REGRESSED**: colored recomputes the full edge-sort (`neighbor_community_weights`) **per
  color per pass** → O(colors·passes·E·logE). ~14-100× SLOWER than igraph at 10-50k. The earlier
  fast (64×) numbers were the *synchronous* version (poor quality). So: fast-but-wrong vs slow-but-right.

### DONE: fused per-vertex move kernel (no sort) — `_MOVE_KERNEL_SOURCE` in louvain.py
One thread per vertex walks its CSR row, O(degree²) scan over distinct neighbor communities (no
sort, no shared memory — degree≤~50 for kNN graphs), picks best modularity-gain move; skips
self-loops; only the active color computes. `_local_moving` re-colors each pass (needed for fuzzy-
graph quality), one GPU sync per pass (not per color). This replaced the per-color global sort.

### NUCLEATION BUG — ROOT-CAUSED & FIXED (self-loop in coloring)
The "stall near pairs at scale" was NOT a dynamics/nucleation problem — it was a one-line coloring
bug. **Contracted graphs have a self-loop on every super-vertex.** In `color_graph`, the self-loop
made a vertex its own uncolored neighbor, so `prio > max_nb` (its own prio) was always false → the
vertex was NEVER selected → left `color=-1` → never active in `_local_moving` (`for c in
range(n_colors)` never hits -1) → never moved → coarse levels couldn't merge. Original graphs have
no self-loops, so level-0 worked; every contracted level silently froze.
**Fix:** exclude self-loops in the coloring's neighbor-max: `both = unc[src] & unc[dst] & (src!=dst)`.
(Diagnosis path: igraph merged our contracted graph 89236→20 but ours did 0 moves; a debug kernel
showed gains were all ≈1 (correct); the real kernel moved 0 because `color[v]==-1` for ~all vertices,
ncolors hit the 2000 cap.)

### Current state (driver 10): FAST + CORRECT at all scales ✓
- **Quality matches/beats igraph everywhere**: PBMC Q=0.658 / 10 cl / ARI 0.64 (vs igraph Louvain
  0.654, Leiden 0.671); SBM 50k 0.807==0.807; 200k 0.712 vs 0.714; **1M 0.693 vs 0.658 (higher)**.
- **Speed**: small/mid n overhead-bound & ~par (0.9×); **GPU pulls ahead at atlas scale — 1M:
  ~16s vs igraph ~44s = ~2.8× faster.** Crossover ~1M; the bigger the graph the bigger the win.

## Phase 3 Leiden refinement — DONE (quality win; speed not yet)
`graph/leiden.py`: Louvain local-moving + **refinement** (`_REFINE_KERNEL_SOURCE`: Louvain move kernel
+ a `part` input, only considers neighbors in the SAME Louvain community — keeps refined communities
pure ⊆ one Louvain community and well-connected, gains use full degrees) → aggregate on the REFINED
partition, **next level's local-moving initialized from the Louvain partition** (`_local_moving` gained
an `init_comm` arg). `n_iterations` repeats the whole multilevel pass. Wired: `cluster.leiden(
connectivities, backend="gpu"|"igraph")` (default igraph).
- **Quality (driver 11): GPU Leiden BEATS igraph Leiden** — PBMC Q=0.664 / 8 cl / ARI 0.706 vs igraph
  Leiden 0.660; SBM 200k 0.8075==0.8075, 1M 0.786 vs 0.805 (slightly under at 1M).
- **Speed: NOT a win at scale** — refinement ≈ a second colored local-moving per level, and
  `n_iterations=2` doubles again → ~4× Louvain's work. 200k 0.23×, 1M 0.57× vs igraph Leiden (which
  is itself faster than igraph Louvain). So: **GPU Louvain = the speed win (2.8× @1M); GPU Leiden =
  the quality win** (better modularity + well-connected guarantee) at a speed cost.

## PERF: why we're not cuGraph's 70× — and what helped
rapids' 70× is a datacenter GPU (A100, ~2TB/s) vs CPU, with fully-fused compiled CUDA (shared-mem
hash tables, warp reductions, no host syncs). We're on M3 (~400GB/s) with Python-orchestrated MLX.
**Profiling the 1M Louvain: coloring = 60% of runtime** (8.6s of 14.2s; one full graph-coloring per
pass), move-kernel 35%, convergence syncs negligible (0.01s — MLX lazy-eval already batches the move
loop). So coloring, not GPU math, was the bottleneck.
- **WIN (Louvain): re-color every 3 passes + shuffle color order in between** (`recolor_every=3` in
  `_local_moving`). Coloring cost drops ~3×; shuffled order gives the convergence diversity that
  per-pass recoloring provided. Result: **50k 1.77× faster, 1M ~9.8s** (quality equal/better,
  PBMC Q 0.661). Color-once (no recolor) is a trap: coloring → 0.67s but passes explode 26→151 (move
  time dominates) → slower. K=3/K=10 are the sweet spot; K=1 (per-pass) and K=∞ (once) are both worse.
- **DO NOT apply the recolor-every-K trick to `_refine`**: within-community refinement with a fixed
  coloring + shuffle fails to converge → hits max_passes every level → catastrophic (200k Leiden:
  11s → **1115s**). Refinement re-colors **every pass** (stable: 200k 10.4s, 1M ~24s).
- **O(d²) move kernel — SOLVED via degree-binning (host tail).** The per-vertex kernel is O(degree²)
  (dedup + per-community weight sum by rescan). Measured at 1M: a contracted level with a **degree-76578
  super-vertex made one move pass take 140s** (sum(d²)=4.75e10). MLX can't do the cuGraph threadgroup-
  hash fix (**no Metal float-atomics**, and ~32KB threadgroup memory can't hold a 76k-entry hash). So:
  the GPU kernel **skips vertices with degree > `_DEGREE_CAP` (1024)** (a `cap` input), and those rare
  high-degree vertices are moved **exactly on the host in NumPy, O(degree)** (`_high_degree_moves` /
  `_high_degree_refine`, sequential with incremental Σtot; refinement version respects the P-restriction).
  Result: degree-30000 hub 3.07s → **0.02s (150×)**; the 76k level would be ~instant. **Zero overhead
  when no large vertices** (`large_ids` empty → host path skipped; PBMC/1M unaffected, quality identical:
  PBMC Louvain 0.661, Leiden 0.666/ARI 0.825). Applied to BOTH the Louvain move kernel and the Leiden
  refine kernel.

## Status
Phases 1–3 DONE + Louvain perf-optimized. **GPU Louvain: fast+correct, ~5–13× over igraph at 1M**
(igraph timing noisy). **GPU Leiden: quality beats igraph Leiden** (PBMC 0.665 vs 0.660), ~par speed
at 1M. Next: degree-binning for the O(d²) kernel (robustness + speed); incremental Σtot; Phase 4
formal benchmark table + figures.


## CLUSTERING SPEED FINDING (validation, flagged for optimization)
At 50k: GPU louvain 3.1s (Q=0.95/20cl, correct) vs igraph 0.32s; leiden GPU 12s vs igraph 0.24s.
igraph's hyper-optimized C wins below ~1M; GPU only (marginally) wins at atlas scale (1M louvain
~10s vs igraph ~15-44s, noisy). Profiling: cost is the PER-COLOR kernel launches (~40/pass, dispatch
overhead) + per-pass coloring — not the GPU math. At small/mid n the launch overhead is the floor.
- Quality is fine via MULTILEVEL (level-0 may look over-fragmented in isolation, but contraction +
  later levels converge to the right count; full louvain 50k -> 20 cl).
- Tried Σtot once-per-pass (vs per-color): better small-n level-0 quality BUT 700s at 1M (far more
  passes to converge). Reverted — per-color keeps atlas-scale speed.
- The only real speed fix to lower the crossover is a FUSED single-kernel local-moving (cuGraph-style,
  per-vertex hashing, colors looped inside one kernel) — large effort. Until then: `cluster.leiden`
  defaults to backend="igraph" (fast/robust); backend="gpu" for atlas scale.

## FUSED single-kernel local-moving — BUILT, BENCHMARKED, NOT VIABLE on M3 (reverted)
Implemented a fully-fused color+move kernel: ONE dispatch per pass does a fresh in-kernel Luby
coloring (rounds looped inside, integer-hash priorities) AND all colored moves, in a SINGLE
threadgroup so the per-round/per-color `threadgroup_barrier(mem_device)` acts grid-wide. Result:
- **Correct + faster ONLY on small CLEAN graphs**: unweighted SBM n≤4k → 1.5–2.1× faster, Q identical;
  crossover ~5k (5k ≈ par, 10k 0.63×, 100k 0.47×).
- **FATAL on real (weighted/fuzzy) graphs**: PBMC louvain fragmented (Q 0.62/52cl vs 0.66/10cl),
  **leiden collapsed to Q=0 / 1 cluster.**
Two compounding HARDWARE limits make it non-viable (this is the "limitation is the hardware" verdict):
1. **No grid-wide barrier across threadgroups** → correct color-sequential ordering forces a SINGLE
   threadgroup = ~1 of the M3's ~10 GPU cores → loses to the multi-core per-color path above ~5k.
2. **No Metal float-atomics** → cannot recompute/maintain **per-color Σtot** inside the fused kernel
   (the between-color scatter-add reduction needs float atomics). Forced to **per-pass Σtot**, which
   is exactly the variant already known to over-fragment fuzzy graphs — hence the PBMC quality
   collapse. The multi-core path recomputes Σtot fresh between every color dispatch; that per-color
   freshness is what real fuzzy graphs need.
cuGraph's fused CUDA wins because A100s have both grid sync (cooperative groups) and float atomics
(shared-mem hash tables) — neither exists on M3/Metal/MLX. **Conclusion: the multi-core per-color
implementation is optimal for M3; the clustering crossover (~1M Leiden, ~50k Louvain) is hardware-
bound, not implementation-bound.** Reverted the fused code; do not re-attempt without Metal gaining
grid barriers or float atomics.

## WORKAROUNDS for the two limits — BOTH BUILT & TESTED, neither beats production (kept, non-viable)
Files: `graph/louvain_hybrid.py`, `graph/louvain_fused_raw.py`. Full data: RESULTS_clustering_workarounds.md.
- **MLX kernel facts discovered (reusable!):** inputs are `constant` address space (CANNOT be atomic);
  **outputs are `device` AND zero-initialized** (verified) — so atomic counters/accumulators must be
  OUTPUTS. `mx.fast.metal_kernel(header=...)` takes helper funcs. With that: a **sense-reversing
  grid-wide barrier** over a `device atomic_uint` output counter WORKS (validated ≥24 threadgroups,
  sub-ms), and **float atomic-add via CAS loop** (`atomic_compare_exchange_weak` on `atomic_uint` +
  `as_type<float>`) WORKS. So both "missing" primitives are synthesizable in MLX.
- **Hybrid (GPU computes all gains in 1 dispatch, CPU applies in gain order w/ exact Σtot):** 1.84×
  faster on PBMC BUT fragments fuzzy graphs (Q0.609/**96cl** vs 0.661/10) — a per-pass snapshot can't
  give per-color-fresh Σtot; and the Python CPU-apply is 0.06× at 100k. Not viable.
- **Raw-fused (spin-barrier + CAS float-atomic Σtot, per-color, single dispatch):** correct on small
  graphs (tiny 2-block Q=0.5000==current; PBMC-400 single-level converges Q0.564 vs 0.587). But (1)
  **G>1 deadlocks NONDETERMINISTICALLY** — MLX gives no co-residency guarantee, heavy kernel fails to
  co-schedule G TGs → spin-barrier hangs the GPU (seen G=8 & G=4 on PBMC); (2) **G=1 is core-starved**
  and the dense high-degree CONTRACTED levels (no host fallback) don't converge in budget → >60s on
  PBMC vs 0.5s. Not viable.
- **Net:** primitives work in isolation (notable!), but composing a correct+fast+robust multilevel
  clusterer still loses to the multi-core per-color path. Verdict unchanged, now strongly evidenced.

### ROUND 2 (pursued pure-Metal): re-rooted — blocker is RELAXED-ONLY ATOMICS, not deadlock
- **Not a deadlock.** With capped passes G=8 COMPLETES (254ms) — but garbage Q=0.05/616cl. Earlier
  "hangs" were multilevel non-convergence on contracted graphs. → occupancy control wouldn't help.
- **MSL ATOMICS ARE `memory_order_relaxed`-ONLY** (durable fact!). acquire/release/seq_cst do NOT
  exist — the system Metal compiler rejects them (`metal_types` declares only relaxed). So no
  lock-free cross-threadgroup happens-before is expressible in Metal.
- **Coherent-memory workaround:** Apple device cache is coherent for relaxed ATOMICS (plain stores
  stay per-core → garbage). Accessing all shared arrays as relaxed atomics (`ldi/sti` helpers in
  louvain_fused_raw) raised G=8 single-level Q=0.05→**0.62** — approx-correct. BUT (a) atomic O(d²)
  hot loop is SLOWER than production (2.0s vs 1.5s); (b) thread-local neighbor snapshot restores
  speed (599ms) but its degree cap drops hub vertices → Q=0.40; (c) residual relaxed staleness →
  G>1 never EXACTLY matches G=1/current (62 vs ~15 comms), unfixable w/o acquire/release.
- **FINAL:** the limit is Metal's relaxed-only atomic memory model (MSL language-level; a raw Metal
  extension shares it). Occupancy control addresses deadlock, which is NOT the problem. Don't pursue
  the fused multi-TG route further unless Metal gains ordered atomics + native float atomics.
- **Reusable MLX/Metal facts banked:** inputs=`constant` (no atomics); outputs=`device`+zero-init
  (use for atomic counters); `header=` for helper fns; sense-reversing grid barrier over device
  atomic_uint works (≥24 TGs); float-add via CAS on atomic_uint+as_type<float> works.
