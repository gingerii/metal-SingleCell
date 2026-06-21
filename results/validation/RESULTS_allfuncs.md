# Full per-function speedup table (GPU/ours vs CPU reference)

M3, best-of-N (GPU warmed up), 50k cells unless noted (8k for O(n²) spatial/tsne). Accuracy vs the
CPU reference. CPU baselines: scanpy / sklearn / igraph / umap-learn; squidpy/harmonypy/fa2/bbknn
unavailable → NumPy reference or GPU-only.

## Strong GPU wins (parallel arithmetic — scatter reductions, elementwise, dense; scale with n)
| function | speedup @50k | accuracy | note |
|----------|-------------|----------|------|
| spatial_autocorr (Moran) | **125×** | matches | scatter-SpMM vs numpy einsum (8k) |
| rank_genes_groups | **48×** | t-stat r~.99 | group scatter-add |
| normalize_total+log1p | **15×** (→56× @2M) | exact | sparse elementwise kernels |
| highly_variable_genes | **8.7×** (→50× @2M) | overlap ~1.0 | scatter gene_moments |
| normalize_pearson_residuals | **7.3×** | exact | dense analytic |
| calculate_qc_metrics | 3.2× | exact | scatter reductions |
| pca_randomized | 2.9× | exact (overlap 1.0) | Gram-orthonormalization |
| filter_cells / filter_genes | 2.8× | exact | qc reductions |
| neighbors | 2.7× | scanpy-parity | uses pynndescent |
| umap | 1.8× | structure-preserv | GPU force layout |

## ~Par (≈1×)
| function | speedup | note |
|----------|---------|------|
| kmeans | 1.06× | MLX Lloyd ≈ sklearn |
| score_genes | 1.02× | gene-set means |

## CPU-favored on M3 (the honest hardware/workload limit — irregular / small / hyper-optimized CPU)
| function | speedup | why |
|----------|---------|-----|
| scale | 0.66× | densify-bound; scanpy numba in-place ≈ as fast |
| tsne | 0.42× | our exact O(n²) vs sklearn Barnes-Hut |
| regress_out | 0.36× | tiny problem (k=2 covariates) — host↔GPU overhead dominates |
| knn (brute-force) | <0.1× | O(n²); → use pynndescent (CPU) |
| **louvain / leiden** | **scale-dependent** | @50k 0.10×/0.01× (igraph C is 0.6s/0.2s); **@1M GPU wins** (~2–13×). Crossover ~0.5–1M. |

## GPU-only (no CPU package installed; accuracy validated separately)
bbknn 4.0s · scrublet 1.8s · harmony_integrate 9.7s · diffmap 1.5s · draw_graph 1.1s ·
co_occurrence 2.1s (O(n²)) · calculate_niche 0.007s · ligrec 0.017s

## The validation verdict (consistent across every op)
**The M3 GPU wins where there is parallel arithmetic to amortize Python/host orchestration** —
sparse scatter-reductions and elementwise kernels (normalize/hvg/qc/rank_genes/Moran: 3–125×, and
they *grow* with n: 50–56× at 2M). It does **not** win for workloads that are irregular, tiny, or
already served by hyper-optimized CPU C code — KNN (→pynndescent), graph clustering below ~1M
(→igraph), exact t-SNE (→Barnes-Hut), and trivially-cheap densify ops (scale/regress_out). For those
the limit is genuinely the hardware/workload, not our code, and the right engineering is **hybrid**:
`neighbors()` already falls back to pynndescent; `cluster.leiden(backend=...)` exposes igraph vs gpu.
Annoy was evaluated for KNN and is *slower* than pynndescent — kept pynndescent.
