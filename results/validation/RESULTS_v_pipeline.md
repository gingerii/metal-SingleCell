# Validation — core pipeline (GPU vs CPU), first pass

Synthetic sparse counts (2000 genes, ~7% density), M3, best-of-N (GPU warmed up).
Accuracy vs scanpy/sklearn on the same data.

## Speedups (CPU walltime / GPU walltime)
| op | 10K | 50K | 100K | accuracy |
|----|-----|-----|------|----------|
| normalize+log1p | 9.2× | 26.7× | 26.5× | exact (r=1.0) |
| hvg_seurat | 1.8× | 1.7× | 1.5× | overlap ~1.0 |
| pca_randomized | 1.6× | 1.5× | 1.4× | seeded-equal |
| knn (brute-force) | 1.6× | **0.54×** | **0.09×** | exact |

## Findings / optimization targets
1. **normalize+log1p**: genuine, scaling GPU win (the sparse elementwise kernels). Done.
2. **knn brute-force — CRITICAL**: O(n²) goes negative at ≥50k and is 11× *slower* than
   sklearn's KD-tree at 100k; OOMs / infeasible at 1M–2M. This is the #1 naive implementation.
   Fix: approximate NN (NN-descent / HNSW) — the production standard. (Tiling fixed the OOM but
   not the O(n²) scaling.)
3. **hvg_seurat / pca_randomized**: only ~1.4–1.8×. Implementation-bound (HVG host binning + CSC
   build; PCA QR-on-CPU + host round-trips). Secondary optimization targets.

## KNN resolution — a hardware/workload finding
Investigated a GPU NN-descent (added reverse neighbors, tiled distances). It reached recall
~0.7–0.9 but stayed **5–15× slower than sklearn/pynndescent** on M3. Root cause: d≈50 PCA
embeddings are low-dim + the workload is irregular/memory-bound — exactly where CPU KD-trees and
pynndescent excel and the M3 GPU has **no advantage** (its strength is dense regular compute). This
is a legitimate "the hardware doesn't favor GPU here" result; **not every op GPU-accelerates on M3**.

**Decision** (consistent with "optimize until hardware-limited"): `neighbors()` uses exact GPU
brute-force for small n (fast + exact there) and **pynndescent — scanpy's own default — for n>30k**
(fast, scales to millions, matches scanpy exactly). The GPU NN-descent stays as `_knn_descent` for
the record. KNN is therefore parity-with-scanpy (CPU), not a GPU win.

Next: revisit hvg/pca (~1.4×) optimization; then scale the sweep to 1M/2M (now feasible — no O(n²)).
