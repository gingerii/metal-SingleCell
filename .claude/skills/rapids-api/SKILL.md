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
| calculate_qc_metrics | ✅ preprocess.py (per-cell + per-gene; exact vs scanpy) |
| filter_cells / filter_genes | ✅ preprocess.py (exact vs scanpy) |
| flag_gene_family | ✅ preprocess.py (13 MT- genes, exact) |
| filter_highly_variable | ✅ preprocess.py |
| normalize_total / log1p | ✅ CSR methods |
| highly_variable_genes | ✅ ALL flavors: seurat / cell_ranger (log-norm) / seurat_v3 (raw) / pearson_residuals (raw, EXACT 2000/2000 vs scanpy). cell_ranger 0.62 overlap (binning deltas); seurat_v3 lowess approximate (skmisc unavailable). |
| scale | ✅ |
| pca | ✅ (full/arpack/randomized; dense OR sparse.CSR input). CSR → sparse-aware randomized PCA: implicit mean-centering (zero_center) via Metal CSR×dense SpMM kernel (`sparse.spmm`/`CSR.spmm`/`spmm_t`), NO densify. Subspace overlap 1.0 vs sklearn, 2× faster than dense @104k, runs at full 2M where dense OOMs (~32GB). Scaling stays dense on purpose (z-score destroys sparsity). |
| regress_out | ✅ preprocess.py (OLS residuals; corr 1.0 vs scanpy) |
| normalize_pearson_residuals | ✅ preprocess.py (corr 1.0, max_diff 7e-7 vs scanpy) |
| harmony_integrate | ✅ integration.py `harmonize` (harmony-pytorch port: cosine soft-kmeans + O/E diversity penalty + ridge correction). BLOCK-STOCHASTIC clustering -> batch_sep 6.0→0.55 (removed), mixing 0→0.42 (~ideal 0.5), bio_sep 8.0→7.92 (preserved). Verify vs harmonypy at validation. |
| scrublet / scrublet_simulate_doublets | ✅ preprocess.py (sim doublets + combined PCA/kNN; doublet-score AUC 0.96 on injected) |
| neighbors | ✅ neighbors.py 3-way KNN: brute<30k (exact) / **GPU IVF 30-250k** (kmeans buckets, ~2-3.5x vs pynndescent/scanpy, recall ~0.9) / pynndescent >250k. GPU brute-force O(n²) & naive NN-descent both lose; IVF bucketing is the mid-range win. |
| bbknn | ✅ neighbors.py (per-batch GPU KNN combined + UMAP fuzzy graph). Validated: forces 50/50 cross-batch neighbors (balanced) while keeping same-bio 100% |

## tl (tools)
| fn | status |
|----|--------|
| umap | ✅ | louvain/leiden | ✅ |
| kmeans | ✅ tools.py (ARI 0.835 vs sklearn) |
| score_genes / score_genes_cell_cycle | ✅ tools.py (corr 0.74 vs scanpy; control-sampling differs) |
| embedding_density | ✅ tools.py (gaussian KDE; O(n²) eval — SAME as scanpy's gaussian_kde, so matches the reference; not a scale gap vs scanpy) |
| rank_genes_groups | ✅ tools.py — ALL 4 methods: **t-test** (corr 0.9998), **t-test_overestim_var** (0.9996; scanpy's ns_rest=ns_group var hack via scipy ttest_ind_from_stats), **wilcoxon** (corr 1.0000; rank-sum z, optional tie_correct), **logreg** (sklearn coef). REAL-DATA top-25 marker overlap **1.000** vs scanpy for all four on PBMC |
| tsne | ✅ tools.py — SCALE-DISPATCHED: exact GPU t-SNE for n≤30k (O(n²), preservation matches sklearn); sklearn Barnes-Hut (O(n log n)) fallback above (verified 50k no OOM, 46s). `exact_max_n` arg controls the threshold |
| diffmap | ✅ tools.py (eigsh of symmetric transition; eigvals 1.0→0.97 on PBMC) |
| draw_graph | ✅ tools.py (FA2-style force layout on MLX; cluster-preservation 1.0) |

## gr (squidpy spatial) — high value for the user's Xenium work
| fn | status |
|----|--------|
| spatial_neighbors | ✅ spatial.py (KNN on coords); REAL Visium edge Jaccard 0.974 vs squidpy |
| spatial_autocorr | ✅ spatial.py (Moran's I / Geary's C, scatter-SpMM + permutation p); REAL Visium **exact** vs squidpy (Moran corr 1.0 Δ6e-8; Geary corr 1.0 Δ3e-7) |
| co_occurrence | ✅ spatial.py (CUMULATIVE d≤thr + squidpy conditional norm; `interval=` accepts squidpy thresholds; GPU pairwise + one-hot; O(n²)); REAL Visium **corr 1.0** Δ7e-4 vs squidpy |
| ligrec | ✅ spatial.py (CellPhoneDB-style L-R: cluster-mean scatter + permutation p); REAL Visium mean-scores **exact** Δ1e-7 |
| calculate_niche | ✅ spatial.py (neighborhood composition via scatter-SpMM + kmeans); REAL Visium composition **exact** Δ0.0 vs scipy |

**REAL-spatial validation (v_realspatial.py, squidpy 1.8.2 on Visium V1 Breast Cancer 3,798 spots):**
all gr functions match squidpy. Two fixes surfaced ONLY by real-data parity: (1) Geary's C was the
symmetric identity `2(Σdx²−Σx·Wx)` — only exact for symmetric W (gave 0.12 error on real conn);
rewrote to sum `Σwᵢⱼ(xᵢ−xⱼ)²` directly over edges → exact. (2) co_occurrence was disjoint bins
(corr 0.60); squidpy is cumulative `d≤thr` with `P(i|c,within r)/P(i)` → rewrote to match → corr 1.0.

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

## REAL-DATA validation (done — PBMC + Visium + 1.3M neurons + 2M Xenium)
Drivers in `validation_notebooks/`: `v_realdata.py` (PBMC3k), `v_realspatial.py` (Visium, squidpy),
`v_realatlas.py` (10x 1.3M neurons), `v_realxenium.py` (user's 2M Xenium object, READ-ONLY).
- **pp/tl on real data**: normalize+log1p exact (Δ~1e-6), HVG overlap 1.000, PCA subspace 0.96–0.98,
  neighbors agreement 0.80–0.98 (IVF recall is dataset-dependent), leiden cluster COUNTS match igraph
  (ARI 0.59–0.85, RNG floor). Holds across PBMC, neuron atlas, and Xenium.
- **gr on real spatial (Visium + Xenium, vs squidpy 1.8.2)**: Moran's I & Geary's C **exact** (corr
  1.0), co_occurrence **exact** (corr 1.0), calculate_niche composition exact (Δ0), ligrec means exact,
  spatial_neighbors Jaccard 0.97 (Visium grid) / 0.76 (irregular Xenium coords).
- **2M-cell scale**: real Xenium normalize+log1p+HVG on 2,035,266 × 5,101 in **3.9s, no OOM**.
- **HARDWARE boundary**: full 1.3M × 27,998 neuron counts (~16GB, ~1.5B nnz) OOMs the 24GB M3 even
  sparse; the 28k-gene full atlas is a true hardware limit (runs on 40–80GB GPU). The dense
  `scale`→`pca` path caps ~500k–1M on-laptop, BUT the sparse-aware PCA (see pca row) now runs at
  full 2M on the lognorm — the prior implementation limit on this path is LIFTED.
- **Two bugs only real data exposed**: Geary's-C symmetric-identity (wrong on asymmetric W) → edge-sum;
  co_occurrence disjoint-bins → squidpy cumulative `d≤thr`. Both now exact. See `RESULTS_v_real*.md`.

## REMAINING-FUNCTION real-data validation (v_remaining.py) — ALL 12 PASS
tools/integration/pp/neighbors that were build-only spot-checks, now vs canonical refs on real
PBMC (HVG-restricted downstream, so it scales): kmeans ARI 0.895 (sklearn); score_genes corr 0.983;
score_genes_cell_cycle phase 0.876; rank_genes_groups top-25 overlap 1.000; tsne 0.229≈sklearn 0.242;
draw_graph 0.118≈scanpy-fr 0.101; diffmap eigval 0.996/subspace 0.991; embedding_density 1.000;
normalize_pearson_residuals 1.0000 (analytic); scrublet injected-doublet AUC 0.972 (>scanpy 0.796);
harmonize batch-mixing 0.05→0.52 = harmonypy 0.50; bbknn opp-batch 0.58 = bbknn pkg 0.57. Refs
installed in env: harmonypy, bbknn, scrublet, squidpy. See RESULTS_v_remaining.md.

## LARGER-DATA validation (v_remaining_scale.py, 100k real neurons) + 2 scale bugs FIXED
Scalable remaining functions confirmed at 100k (HVG-restricted): kmeans, score_genes (1.000),
rank_genes_groups (top-25 1.000), normalize_pearson_residuals (1.0000), diffmap, harmonize
(mixing 0.01→0.50), bbknn (0.56); scrublet at combined ~75k (AUC 0.931). exact-tsne & KDE
embedding_density are O(n²) subsample-only.
- **BUG FIXED — bbknn GPU OOM**: built full n×|batch| distance matrix (100k×50k≈20GB) → tile query
  rows (~256M-entry cap). `neighbors.bbknn`.
- **BUG FIXED — scrublet GPU OOM**: `.toarray()` densified combined ~3n×20k (≈24GB) → HVG-restrict
  + sparse PCA (canonical). `preprocess.scrublet`. (Machine is **51.5GB RAM**, not 24GB as earlier
  docs said.) Known limit: scrublet brute-force `_knn_gpu` heavy past combined~100k (future: route
  through tiled/pynndescent neighbors). These two OOMs (+harness not freeing 8GB dense intermediates)
  drove memory to exhaustion and plausibly caused a dev-machine hard reboot during validation.

## WHERE WE ARE (checkpoint)
**ALL ~32 rsc pp/tl/gr functions implemented & pushed + REAL-DATA VALIDATED** (core pipeline, spatial
gr, atlas, Xenium, AND the remaining tools/integration/pp via v_remaining.py). Modules: sparse,
preprocess, decomposition, neighbors, embedding, cluster, graph/, tools, spatial, integration.
PARITY DELTAS — progress:
- ✅ HVG flavors ALL EXACT vs scanpy on PBMC (overlap 1.000 each): seurat (was 1.0), **cell_ranger
  fixed 0.617→1.000** (scanpy applies expm1 + log(disp)/log1p(mean) ONLY for seurat; cell_ranger
  uses mean/var of the lognorm values directly via `col_moments`, no log), **seurat_v3 fixed →1.000**
  (now uses `skmisc.loess` span=0.3 degree=2 like scanpy; statsmodels lowess was a degree-1 approx).
- ✅ **score_genes EXACT (corr 1.0000, multiple seeds)** — matched scanpy's binning
  (`rankdata(method='min')//n_items`, n_items=round(n/(n_bins-1))) and per-UNIQUE-bin sampling
  (was per-gene + normalized-rank bins → 0.95-0.98). Controls are expression-matched within bins,
  so the score is RNG-invariant. score_genes_cell_cycle inherits the fix.
- ✅ rank_genes_groups: top-25 marker overlap **1.000** on PBMC & 100k real data (the overestim_var/
  p-tiebreak delta only reorders lower-ranked genes — closed for practical use).
- harmony: block-stochastic clustering DONE; verified vs harmonypy (mixing 0.52 vs 0.50). 
- fp32 deltas: none material found on real data.
- t-SNE: exact GPU O(n²) for n≤30k, sklearn Barnes-Hut fallback above (scale-dispatched, DONE).
Next: optional AnnData wrapper layer (rsc.pp/tl/gr namespaces) + end-of-project validation suite.

## Conventions
- Compute on MLX where GPU helps; lazy-import mlx. Match scanpy/rsc signatures + defaults.
- Reuse: `sparse.CSR`, `graph/` substrate (segment ops!), `neighbors`, `validation` harness.
