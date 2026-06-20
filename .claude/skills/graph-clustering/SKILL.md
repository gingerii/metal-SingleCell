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

### Current state (driver 10): CORRECT on small/well-structured, OPEN BUG at scale
- **Quality correct where structure is clear/small**: SBM n=10k Q=0.809==igraph (20 cl exactly);
  n=50k 0.787 vs 0.807; PBMC 0.630 / 35 cl / ARI 0.53 vs igraph 0.656/~13.
- **Speed**: small n is overhead-bound (GPU 0.05–0.4× igraph — per-pass coloring syncs + dispatch).
- **OPEN BUG (the crux)**: on LARGE + WEAKLY-structured graphs (e.g. SBM 200k, 20 blocks → sparse
  blocks) colored local-moving **stalls near pairs**, Q collapses to ~0.04. A one-off 200k/40-block
  run worked (1.5× faster, Q 0.779>igraph) — so it's structure/scale-dependent, NOT a fixed-n bug.
  Root cause (hypothesis): the colored greedy fails to *nucleate* large communities when within-
  community connectivity is sparse — sequential igraph snowballs, parallel colored doesn't. Math is
  scale-invariant (equilibrium penalty identical at 200k vs 500k), so it's a dynamics/nucleation
  issue, not fp32 or the modularity formula. NOT yet root-caused.

### Next ideas to try for the nucleation/scale bug
- Leiden-style refinement (Phase 3) — may help but won't fix nucleation alone.
- Better move acceptance: allow moving toward the *max-edge-weight* neighbor's community first
  (coalescence bias), or a few sequential "seed" sweeps before colored parallel sweeps.
- Verify the multilevel actually contracts+continues at scale (suspect level loop stops after ~1
  level when local-moving returns only pairs — check whether coarse levels re-merge).

## Status
Phase 1 substrate DONE. Phase 2: per-vertex kernel makes it fast & correct on small/well-structured
graphs; a scale+weak-structure nucleation bug is the open crux before atlas-scale is trustworthy.
Phases 3 (Leiden refinement) + 4 (1M benchmark) follow once nucleation is fixed.
