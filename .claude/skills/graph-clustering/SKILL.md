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

## Status
Phases 1–3 DONE. Louvain: fast+correct at atlas scale. Leiden: best quality, wired as a drop-in,
but slower than igraph Leiden at scale (refinement cost). **Next (optimization, not correctness):**
speed up Leiden — refinement is the bottleneck; ideas: skip the 2nd iteration when it adds little,
fuse refine into the move kernel, or only refine the final levels. Phase 4: formal 1M+ benchmark
table + figures.
