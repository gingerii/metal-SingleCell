# Real-data validation — remaining functions (tools / integration / pp / neighbors)

Brings the functions only spot-checked at build time to the same real-data bar as the core
pipeline. Canonical pattern: compute HVGs, then restrict downstream to them (keeps everything
in memory at any scale — the atlas-OOM concern only affected full-gene-set dense ops). Real
PBMC3k; each function vs its canonical reference (scanpy / sklearn / harmonypy / bbknn /
scrublet). Stochastic methods validated by structure/agreement, as with umap/leiden.

| function | reference | metric | result |
|----------|-----------|--------|--------|
| kmeans | sklearn KMeans | ARI | **0.895** ✓ |
| score_genes | scanpy tl.score_genes | score corr | **0.983** ✓ |
| score_genes_cell_cycle | scanpy | phase agreement | **0.876** ✓ |
| rank_genes_groups (t-test) | scanpy tl.rank_genes_groups | top-25 marker overlap | **1.000** ✓ |
| tsne | sklearn TSNE | nbr-preservation | 0.229 vs 0.242 ✓ |
| draw_graph | scanpy tl.draw_graph (fr) | nbr-preservation | 0.118 vs 0.101 ✓ |
| diffmap | scanpy tl.diffmap | eigval corr / comp subspace | **0.996 / 0.991** ✓ |
| embedding_density | scanpy tl.embedding_density | density corr | **1.000** ✓ |
| normalize_pearson_residuals | scanpy experimental | residual corr | **1.0000** ✓ |
| scrublet | injected-doublet AUC | AUC | **0.972** (scanpy 0.796) ✓ |
| harmonize | harmonypy | batch mixing (iLISI-like) | 0.05→**0.52** (harmonypy 0.50) ✓ |
| bbknn | bbknn package | graph opp-batch frac | **0.58** (bbknn pkg 0.57) ✓ |

## Notes
- **Exact/near-exact** (deterministic): normalize_pearson_residuals (analytic, 1.0000),
  embedding_density (1.000), diffmap (eigval 0.996, subspace 0.991), rank_genes_groups
  (perfect top-25 overlap), score_genes (0.983).
- **Stochastic, validated by behaviour**: tsne & draw_graph match their references on
  neighbor-preservation; harmonize recovers batch mixing to harmonypy's level (0.52 vs 0.50,
  from a separated 0.05); bbknn produces the same batch-balanced graph as the bbknn package.
- **scrublet**: score-correlation to scanpy is weak (each draws its own random doublet
  simulation), so the meaningful test is detection — **injected-doublet AUC 0.972**, exceeding
  scanpy's own 0.796 on the same injected set.
- **HVG-restriction** is what makes all of this run at scale: downstream operates on n×2000,
  not n×n_genes, so there is no atlas-scale OOM for these functions.

Driver: `validation_notebooks/v_remaining.py`. References: scanpy, scikit-learn, harmonypy,
bbknn, scrublet.
