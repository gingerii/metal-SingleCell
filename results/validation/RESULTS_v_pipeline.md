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

Next: replace brute-force KNN with a GPU approximate-NN, then revisit hvg/pca.
