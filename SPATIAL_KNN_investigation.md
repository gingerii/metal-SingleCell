# Speeding up `gr.spatial_neighbors` — the low-dim kNN problem, and options beyond brute force

## The problem (confirmed against our code)

`gr.spatial_neighbors` builds a kNN on 2-D/3-D spatial coordinates, and it deliberately bypasses
the high-dim `_knn` dispatcher (brute → IVF → NNDescent). The docstring in `spatial.py:20-27`
already says why, and the user confirmed it empirically: NNDescent recall on 2-D coords is ~6% —
it's a high-dimensional-embedding method, degenerate in low dimensions (the RP-tree splits and
neighbor-descent heuristics that work in ~50-D PCA space carry no signal in 2-D). So
`spatial_neighbors` calls the exact GPU brute-force `_knn_gpu`, which is O(n²) in cells — fine at
60k, a wall at 2M.

The question is not "make the ANN work in 2-D" (it can't) — it's "what's the right accelerated
exact (or high-recall) kNN for low-dim point data." That's a different, well-studied problem with a
clean answer: a spatial index (uniform grid / KD-tree / quadtree), O(n log n) or O(n) build + O(k)
query.

## What the reference does (the bar to beat)

- **squidpy** (`gr/neighbors.py`): the generic-coordinate kNN builder is
  `sklearn.neighbors.NearestNeighbors(metric="euclidean")`. sklearn auto-selects a KD-tree or
  ball-tree for low-dim data — an O(n log n) spatial index, single-threaded C. That is exactly why
  our brute O(n²) loses at scale here while winning nowhere: we're racing an index with a quadratic
  scan.
- squidpy also offers a **Delaunay** builder (18 refs) as an alternative connectivity; not a kNN,
  but worth knowing it's the other spatial-graph option users reach for.
- **GPU precedent exists:** RAPIDS **cuSpatial** ships GPU point-indexing / nearest-points APIs for
  precisely this low-dim regime (its public API exposes `nearest_points` and pairwise point-distance
  primitives; cuSpatial is also documented elsewhere as offering quadtree-based spatial indexing,
  though I did not verify the quadtree internals from the source tree here). The takeaway that
  matters: a GPU spatial index for low-dim point data is a real, shipped pattern, not research
  vapor. (CUDA-only, so not directly portable to Metal, but the algorithm is.)

## Options, ranked by payoff ÷ effort

### Option 1 — Uniform grid / cell-list (RECOMMENDED first)

The classic molecular-dynamics / graphics neighbor-search structure, and the best fit for
MLX + Metal:

- Bin points into a uniform grid of cell size ≈ the typical kNN radius (spatial data is roughly
  uniform in density, so a uniform grid is near-optimal — this is why it beats a KD-tree here).
- Each query point only examines its own cell + the 8 (2-D) / 26 (3-D) neighbors — O(1) candidates
  per point instead of O(n).
- Brute-force the distances within that tiny candidate set (reuses our existing GPU distance +
  `_topk_rows` register top-k kernel — the k≤32 path is perfect for `n_neighs=6`).

**Why it's the right first move:** build is one argsort by cell id — and we already have a fast GPU
argsort (`argsort_20M_int64` = 0.084s, from the Leiden work). Query is embarrassingly parallel and
maps cleanly onto a Metal kernel (one threadgroup per grid cell / per query). No tree, no recursion,
no host-side pointer chasing. Exact results (not approximate) when the cell size ≥ the k-th neighbor
distance; a small guard (expand the ring if fewer than k candidates found) keeps it exact in sparse
regions.

**Expected:** O(n) build + O(n·c) query (c = candidates per cell), should turn the 2M brute wall
into seconds. This is the single highest-value item.

### Option 2 — Fixed-radius (`radius`) query, not top-k

squidpy's builder also supports a radius mode (`NearestNeighbors(radius=r)`), and for spatial graphs
a fixed-radius neighborhood is often what users actually want (Visium hex, Xenium physical µm).
Radius query is even friendlier to the grid: cell size = r means candidates are exactly the 3×3 (or
3×3×3) block, no top-k at all. Worth exposing `radius=` alongside `n_neighs=` and routing both
through the grid. Low extra effort once Option 1's grid exists.

### Option 3 — GPU KD-tree

The general-purpose answer, but a poorer fit than the grid: KD-tree build is recursive and GPU
KD-tree traversal is stack/branch-heavy (bad for SIMD, exactly the launch-latency regime that hurt
us elsewhere). Only worth it if point density is highly non-uniform (where a uniform grid degrades
to near-brute in dense cells). Spatial transcriptomics is mostly uniform-density, so grid > KD-tree
here. Keep as a fallback for pathological density, not the primary.

### Option 4 — Keep brute force, but tile it (cheap stopgap)

If a spatial index is more than we want to build now: the current brute force likely materializes a
big distance block. A tiled brute force (block the n×n distance matrix, top-k per tile, merge)
removes the memory wall and modestly improves cache behavior, but stays O(n²) in FLOPs — it buys
headroom at 2M, not a complexity win. Fine as an interim; not the destination.

## Recommendation

Build **Option 1** (uniform grid + within-cell brute via the existing top-k kernel), expose
**Option 2** (radius mode) on the same grid, and keep KD-tree (Option 3) only as a
non-uniform-density fallback. This reuses two primitives we already have and trust — GPU argsort
(build) and the `_topk_rows` register kernel (query) — so it's assembly of proven parts, not a
from-scratch kernel. Validate exactness against the current brute-force output (identical kNN indices
when cell size ≥ k-th distance) and benchmark on the real multi-platform spatial data (Visium →
MERFISH → Xenium 63k → Xenium 253k → 2M) from the benchmark-extend brief — the same datasets, so the
spatial-kNN speedup lands directly in the spatial table.

## Note on scope

This makes `spatial_neighbors` fast, and it also unblocks any spatial `gr` function that builds its
own neighborhood graph (`co_occurrence`, `autocorr` all consume a spatial graph). It does not touch
the high-dim `_knn` dispatcher (neighbors/bbknn) — that path is correct as-is; this is a separate,
low-dim-only index.
