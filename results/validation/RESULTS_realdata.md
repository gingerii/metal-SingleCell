# Real-data validation (PBMC3k) — CLAUDE.md pin

Synthetic data is structureless (worst case for clustering/KNN/UMAP), so every claim is re-checked
on real data. PBMC3k (2700 real cells); the IVF-KNN scale path on a real-structure embedding tiled
to 100k.

| function | real-data result | verdict |
|----------|------------------|---------|
| highly_variable_genes | overlap **1.000** vs scanpy | exact |
| regress_out | corr **1.0000** vs scanpy | exact |
| pca_randomized | subspace overlap **1.0000** vs sklearn | exact |
| neighbors | overlap **1.000** vs exact (brute path at n=2700) | exact |
| leiden (GPU) | ARI **0.852** vs igraph (10 vs 9 cl) | good (within RNG floor) |
| IVF-KNN (real structure, 100k) | **recall 0.983**, 1.18× vs pynndescent | high recall on real data |
| umap | preservation **0.113** vs umap-learn 0.137 | small inherent gap (see note) |

## Notes
- **IVF-KNN recall is 0.98 on real structure** (vs ~0.5–0.86 on random synthetic) — confirms the
  random-synthetic recall was pessimistic; IVF is reliable on real data.
- **umap** is ~15% below umap-learn on neighbor-preservation, and this is **inherent to our GPU
  implementation** (both the all-edges and the scheduled variants give ~0.11–0.12 — the gap is not
  from the all-edges speed change). all-edges is kept (much faster, equal quality to our scheduled
  variant). The embedding captures global structure; the gap is in fine 15-NN preservation. Acceptable
  for visualization; flagged if tighter fidelity is needed.
- Flagged-function optimizations all hold on real data: regress_out exact, IVF-KNN high-recall+faster.

Driver: `validation_notebooks/v_realdata.py`.
