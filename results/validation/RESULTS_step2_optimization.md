# Step 2 — attacking the implementation-bound functions

The benchmark (RESULTS_v_benchmark.md) flagged three functions slower than their CPU reference:
harmonize, bbknn, and leiden's refinement. This is the result of attacking them.

## harmonize — REAL BUG FIXED (~2× faster, quality better than harmonypy)
`max_iter_clustering` was **200**; harmony/harmonypy use **~20**. The block-stochastic R
updates never reach `tol_clustering=1e-5`, so it ran all 200 inner iters every harmony round
(10×200×~20 blocks ≈ 40,000 block-passes). Fixed to 20.

| n | before | after | harmonypy | quality (mixing, higher=better) |
|---|--------|-------|-----------|----------------------------------|
| PBMC | 9.5 s | **1.3 s** | 0.1 s | ours 0.52 vs harmonypy ~0.50 |
| 50k | 14.4 s (0.35×) | **8.0 s (0.58×)** | 4.6 s | ours 0.49 vs harmonypy 0.40 |
| 100k | 46.9 s (0.13×) | **22 s (0.28×)** | 6.2 s | ours 0.50 vs harmonypy 0.43 |

Still CPU-favored on speed, but our **mixing quality is better than harmonypy at every size**.
The residual gap is a workload limit: harmony is small-matrix iterative work (K×N soft assign,
per-block penalty updates, per-cluster ridge) where the 16-core CPU + tuned numpy beats GPU
dispatch overhead — same class as clustering, not a remaining code bug.

## bbknn — kNN-workload-bound (no GPU win available)
Profiling (100k): per-batch brute-force kNN = 5.7 s of 7 s (the rest is the umap fuzzy graph).
The brute-force is O(n·|batch|) per batch. Tried routing large batches through approximate kNN
(pynndescent) to remove the quadratic cost — reverted: it returned out-of-range indices (crash),
and even working it just reuses the **same approximate CPU method the bbknn package uses**, so it
reaches parity at best, never a GPU win. Kept the GPU brute-force (correct; query-row tiled → no
OOM). Verdict: bbknn is kNN-workload-bound on M3, like the regular `neighbors` — the package's
approximate CPU kNN is competitive and the GPU doesn't beat it.

## leiden refinement — hardware-bound
GPU leiden loses at every size (0.08× even at 1M) because its refinement phase ≈ a second colored
local-moving per level (×n_iterations), and Metal cannot run cuGraph-style fused clustering
(relaxed-only atomics, no grid barrier — proven earlier). igraph leiden is the right default.

## Pushing the WINNERS higher (transfer + fp16)

### CSR.from_scipy transfer — BIG, broad win
At scale the host→device transfer dominated sparse ops: **10.06s** of a 12.5s normalize at
1M×22k (2B nnz), vs only 2.4s of actual kernel compute. Most was **redundant `astype` copies**
of arrays that already had the right dtype (data float32, indices int32 — `astype` always
copies). Fix: skip the cast when dtype matches, don't re-CSR an already-CSR input, sort only if
needed (on a copy). **from_scipy 10.06s → 3.35s**; normalize+log1p+transfer @1M **0.82× → 1.8×**
vs scanpy. This speeds **every** sparse op (hvg/pca/pearson/scrublet all call `from_scipy`).
Correctness unchanged (roundtrip exact; handles int64/unsorted input without mutating caller).

### fp16 — narrow lever (kNN distance only)
fp16 matmul on M3 MLX helps **only** large-output tall-thin shapes, and **only if the result
stays fp16** (an fp32 cast-back of a block×n matrix costs back the gain); small-K outputs
(kmeans assign, K≈15) get nothing, and square matmuls (PCA Gram) are slightly slower in fp16.
Applied where it pays: `_knn_gpu` computes the whole distance + argpartition in fp16 → **1.27×**
on the brute path (also feeds IVF buckets / bbknn), recall **0.961** vs exact; the k selected
distances are recomputed in fp32 so returned values stay full precision. kmeans fp16 reverted
(no win); kept the xsq hoist. **fp16 is not a broad win on M3.**

## Honest conclusion
Of the three implementation-bound functions, **exactly one had a real, fixable bug** (harmonize's
200-iter, now 2× faster + better quality). The others are **workload/hardware-bound on M3** —
iterative / graph / kNN work where the multi-core CPU (numpy/igraph/cKDTree) wins and the
Python-orchestrated GPU does not. This matches the project-wide finding: the M3 GPU's genuine
domain is the **parallel-arithmetic ops** (normalize/HVG/PCA/Pearson/rank-genes/umap/scrublet —
already 3–49×); the iterative/graph/kNN ops are CPU-favored at the scales they're used.
