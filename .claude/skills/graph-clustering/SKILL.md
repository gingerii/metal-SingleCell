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

## Status / next
- Phase 1 substrate: DONE. Next: `graph/louvain.py` (multilevel parallel local-moving + contract),
  then `graph/leiden.py` (refinement), wire `cluster.leiden(backend="gpu")`, atlas benchmark to 1M.
- Oscillation risk in synchronous local-moving: tie-break lowest community id / half-step; guard with
  modularity-monotonicity.
