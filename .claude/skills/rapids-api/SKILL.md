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
| highly_variable_genes | ✅ seurat / cell_ranger (log-norm) / seurat_v3 (raw counts). cell_ranger 0.62 overlap vs scanpy (binning/MAD deltas); seurat_v3 uses statsmodels lowess (skmisc unavailable) so loess fit approximate — verify vs scanpy at validation. ⬜ pearson_residuals flavor |
| scale | ✅ |
| pca | ✅ (full/arpack/randomized) |
| regress_out | ✅ preprocess.py (OLS residuals; corr 1.0 vs scanpy) |
| normalize_pearson_residuals | ✅ preprocess.py (corr 1.0, max_diff 7e-7 vs scanpy) |
| harmony_integrate | 🟡 integration.py `harmonize` (port of harmony-pytorch: cosine soft-kmeans + O/E diversity penalty + per-cluster ridge correction; matmuls on MLX, ridge solves numpy fp64). WORKS — synthetic 2-batch/2-bio: batch_sep 6.0→2.17, bio_sep 8.0→7.94 (biology preserved). ⚠ batch mixing only partial (0→0.08) — likely needs block-STOCHASTIC clustering updates (I used full-batch) and/or tuning; verify vs harmonypy at validation. |
| scrublet / scrublet_simulate_doublets | ⬜ Batch 4 (sim doublets + kNN density) |
| neighbors | ✅ |
| bbknn | ✅ neighbors.py (per-batch GPU KNN combined + UMAP fuzzy graph). Validated: forces 50/50 cross-batch neighbors (balanced) while keeping same-bio 100% |

## tl (tools)
| fn | status |
|----|--------|
| umap | ✅ | louvain/leiden | ✅ |
| kmeans | ✅ tools.py (ARI 0.835 vs sklearn) |
| score_genes / score_genes_cell_cycle | ✅ tools.py (corr 0.74 vs scanpy; control-sampling differs) |
| embedding_density | ✅ tools.py (gaussian KDE) |
| rank_genes_groups | ✅ tools.py (t-test; t-stat corr 0.993 vs scanpy; ⚠ marker-overlap 0.54, scanpy overestim_var/p-tiebreak not matched) |
| tsne | ⬜ Batch 3+ (fft/BH — hard) |
| diffmap | ✅ tools.py (eigsh of symmetric transition; eigvals 1.0→0.97 on PBMC) |
| draw_graph | ⬜ Batch 3+ (ForceAtlas2 — like umap layout) |

## gr (squidpy spatial) — high value for the user's Xenium work
| fn | status |
|----|--------|
| spatial_neighbors | ✅ spatial.py (KNN on coords) |
| spatial_autocorr | ✅ spatial.py (Moran's I / Geary's C, scatter-SpMM + permutation p) |
| co_occurrence | ✅ spatial.py (distance-binned cluster co-occurrence ratio; GPU pairwise + one-hot; O(n²)) |
| ligrec | ⬜ Batch 4 (ligand-receptor permutation test) |
| calculate_niche | ⬜ Batch 4 (neighborhood composition clustering) |

## Build batches (tractable → hard)
1. ✅ DONE pp completions: filter_cells/genes, flag_gene_family, filter_highly_variable, calculate_qc_metrics
   (+ `CSR.gene_counts`). All match scanpy exactly on PBMC.
2. ✅ DONE `tools.py`: kmeans (MLX Lloyd, ARI 0.835 vs sklearn), score_genes(+cell_cycle, embedding_density);
   `spatial.py`: spatial_neighbors + spatial_autocorr (Moran's I / Geary's C via scatter-SpMM + permutation p —
   validated: smooth gene Moran 0.97/Geary 0.03, noise ~0/~1). NB MLX has no GPU sparse matmul → W·X done as
   scatter-add over edges (`_spmm_scatter`).
2b. ✅ rank_genes_groups (t-test, in tools.py).
3. 🔨 IN PROGRESS — pp: regress_out, normalize_pearson_residuals; tl: diffmap; gr: co_occurrence.
   Then: draw_graph, tsne, HVG flavors (cell_ranger/seurat_v3/pearson).
4. ⬜ pp: harmony_integrate, scrublet, bbknn; gr: ligrec, calculate_niche.

## WHERE WE ARE (checkpoint)
~19/32 rsc functions built (all committed/pushed). Modules: sparse, preprocess, decomposition,
neighbors, embedding, cluster, graph/, tools, spatial. Validation DEFERRED to project end (user's
call) — but each function spot-checked vs scanpy/sklearn at build time (notes above). Parity gaps to
revisit at validation: score_genes control sampling, rank_genes_groups ranking, (and any fp32 deltas).
Batch 3 DONE. Harmony (`integration.py`) done (works; mixing partial — see note). ~24/32 functions.
Remaining: HVG flavors (cell_ranger/seurat_v3/pearson), bbknn, scrublet(+sim), tsne, draw_graph,
ligrec, calculate_niche. Harmony follow-up: block-stochastic clustering + harmonypy parity check.

## Conventions
- Compute on MLX where GPU helps; lazy-import mlx. Match scanpy/rsc signatures + defaults.
- Reuse: `sparse.CSR`, `graph/` substrate (segment ops!), `neighbors`, `validation` harness.
