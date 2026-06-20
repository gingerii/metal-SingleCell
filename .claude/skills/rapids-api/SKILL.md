---
name: rapids-api
description: The rapids-singlecell API we mirror (scanpy-GPU pp/tl + squidpy-GPU gr), our build status per function, module layout, and batched roadmap. Load when implementing/locating any pp/tl/gr function.
---

# rapids-singlecell API parity — target & status

Goal: implement on Metal/MLX every GPU function rapids-singlecell exposes. API ref:
https://rapids-singlecell.readthedocs.io/en/latest/api/ (scanpy_gpu.html, squidpy_gpu.html).
rsc functions take an AnnData and mutate in place; ours are compute functions on arrays/CSR (an
AnnData wrapper layer can sit on top later). Validation deferred to project end (user's call).

## Module layout (ours)
- `sparse.py` (CSR primitives), `preprocess.py` (pp), `decomposition.py` (pca),
  `neighbors.py`, `embedding.py` (umap), `cluster.py` (leiden), `graph/` (louvain/leiden GPU).
- NEW: `tools.py` (tl), `spatial.py` (gr/squidpy), `integration.py` (harmony/bbknn/scrublet).

## pp (preprocessing)
| fn | status |
|----|--------|
| calculate_qc_metrics | ✅ CSR.qc_metrics (per-cell); TODO top-level pp incl per-gene + pct_counts |
| filter_cells / filter_genes | ⬜ Batch 1 |
| flag_gene_family | ⬜ Batch 1 |
| filter_highly_variable | ⬜ Batch 1 |
| normalize_total / log1p | ✅ CSR methods |
| highly_variable_genes | ✅ seurat; ⬜ cell_ranger / seurat_v3 / pearson_residuals |
| scale | ✅ |
| pca | ✅ (full/arpack/randomized) |
| regress_out | ⬜ (per-gene OLS on covariates, GPU lstsq) |
| normalize_pearson_residuals | ⬜ (analytic Pearson residuals of NB) |
| harmony_integrate | ⬜ Batch 4 (iterative; port harmony-pytorch logic) |
| scrublet / scrublet_simulate_doublets | ⬜ Batch 4 (sim doublets + kNN density) |
| neighbors | ✅ |
| bbknn | ⬜ Batch 4 (batch-balanced kNN) |

## tl (tools)
| fn | status |
|----|--------|
| umap | ✅ | louvain/leiden | ✅ |
| kmeans | ⬜ Batch 2 (MLX matmul Lloyd) |
| score_genes / score_genes_cell_cycle | ⬜ Batch 2 (gene-set mean − control-bin mean) |
| embedding_density | ⬜ Batch 2 (gaussian KDE per group) |
| rank_genes_groups | ⬜ Batch 3 (t-test/wilcoxon/logreg markers) |
| tsne | ⬜ Batch 3 (fft/BH — hard) |
| diffmap | ⬜ Batch 3 (eigh of transition matrix) |
| draw_graph | ⬜ Batch 3 (ForceAtlas2 — like umap layout) |

## gr (squidpy spatial) — high value for the user's Xenium work
| fn | status |
|----|--------|
| spatial_autocorr | ⬜ Batch 2 (Moran's I / Geary's C via sparse spatial weights; permutation p) |
| co_occurrence | ⬜ Batch 3 (distance-binned label co-occurrence) |
| ligrec | ⬜ Batch 4 (ligand-receptor permutation test) |
| calculate_niche | ⬜ Batch 4 (neighborhood composition clustering) |

## Build batches (tractable → hard)
1. ✅ DONE pp completions: filter_cells/genes, flag_gene_family, filter_highly_variable, calculate_qc_metrics
   (+ `CSR.gene_counts`). All match scanpy exactly on PBMC.
2. ✅ DONE `tools.py`: kmeans (MLX Lloyd, ARI 0.835 vs sklearn), score_genes(+cell_cycle, embedding_density);
   `spatial.py`: spatial_neighbors + spatial_autocorr (Moran's I / Geary's C via scatter-SpMM + permutation p —
   validated: smooth gene Moran 0.97/Geary 0.03, noise ~0/~1). NB MLX has no GPU sparse matmul → W·X done as
   scatter-add over edges (`_spmm_scatter`).
3. ⬜ tl: rank_genes_groups, diffmap, draw_graph, tsne; gr: co_occurrence; HVG flavors; pearson_residuals; regress_out.
4. ⬜ pp: harmony_integrate, scrublet, bbknn; gr: ligrec, calculate_niche.

## Conventions
- Compute on MLX where GPU helps; lazy-import mlx. Match scanpy/rsc signatures + defaults.
- Reuse: `sparse.CSR`, `graph/` substrate (segment ops!), `neighbors`, `validation` harness.
