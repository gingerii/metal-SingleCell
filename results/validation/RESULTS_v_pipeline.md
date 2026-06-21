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

## Optimizations applied (hvg, pca)
Profiling pinpointed host bottlenecks; both fixed, accuracy preserved:
- **hvg**: `gene_moments` was rebuilding a CSC on the host (0.037s of 0.055s). Replaced with a pure
  GPU **scatter-add over the gene index** (~22× faster, identical values). HVG speedup **1.4→13.6×**
  at 100k (overlap ~1.0).
- **pca_randomized**: the power-iteration **QR ran on CPU** (MLX has no GPU QR) — 1 QR = 0.055s × 15
  ≈ 0.83s of 1.2s. Replaced with **Gram-matrix orthonormalization** `Q(QᵀQ)^{-1/2}` (tall work as GPU
  matmuls + a tiny size×size eigh). PCA **1.4→2.6×**, subspace overlap vs sklearn = **1.0000** (exact).

## Post-optimization speedups
| op | 10K | 50K | 100K | accuracy |
|----|-----|-----|------|----------|
| normalize+log1p | 3.5× | 10× | 26× | exact |
| hvg_seurat | 2.9× | 9.5× | 13.6× | overlap ~1.0 |
| pca_randomized | 2.5× | 2.8× | 2.6× | exact (overlap 1.0) |
| knn | (pynndescent, parity w/ scanpy — M3 GPU doesn't win this workload) | | | |

Next: scale the sweep to 1M/2M (now feasible — no O(n²)); revisit clustering/umap at scale.
