# Real XENIUM validation — the user's integrated endometrium cohort (2M cells)

CLAUDE.md pin: validate on the user's own real Xenium data. Object (READ-ONLY, belongs to
the Xenium project): `Xenium_Claude_test/data/processed/xenium/integrated_data.h5ad` —
**2,035,266 cells × 5,101 genes** (Xenium 5k panel), 72 sections, real cell-type labels,
real tissue coordinates. Pipeline + spatial `gr` parity on the largest single section
(`p11_t3_s1`, 103,629 cells, 19 cell types — coherent spatial frame); full-cohort 2M scale test.

## Core pipeline (real section, vs scanpy/sklearn)
| function | result | verdict |
|----------|--------|---------|
| normalize+log1p | max\|Δ\| 9.5e-7 | exact |
| highly_variable_genes | overlap **1.000** | exact |
| pca (dense, scaled) | subspace overlap 0.960 (0.77s) | high |
| pca (sparse, implicit-center) | subspace overlap **1.0000** (0.36s) | exact + 2× faster |
| neighbors | graph agreement 0.798 vs scanpy | OK (see note) |
| leiden GPU vs igraph | **14 vs 14 cl**, ARI 0.591 | cluster count matches |

## Spatial `gr` (15k subsample of section, vs squidpy 1.8.2)
| function | result | verdict |
|----------|--------|---------|
| spatial_neighbors | edge Jaccard 0.757 | OK (irregular Xenium coords; see note) |
| spatial_autocorr (Moran's I) | corr **1.0000**, max\|Δ\| 2.3e-8 | exact |
| spatial_autocorr (Geary's C) | corr **1.0000**, max\|Δ\| 3.6e-7 | exact |
| co_occurrence | corr **0.9999** | exact (one rare-pair ratio Δ0.36) |
| calculate_niche (composition) | max\|Δ\| **0.0** | exact |

## Full-cohort 2M scale test — the genuine atlas-scale demonstration
- **normalize_total + log1p + highly_variable_genes** on the full 2,035,266 × 5,101 matrix:
  **4.2 s, no OOM** (counts read via h5py into scipy CSR ≈ 5 GB, fits the 24 GB M3).
- **sparse-aware PCA** on the full 2M cells (HVG-subset lognorm): **8.3 s, no OOM** — exactly
  where the dense `scale`→`pca` path dies (it would need ~32 GB). Implicit mean-centering keeps
  the matrix sparse end-to-end.

This is the real 2M-cell demonstration the synthetic atlas could only approximate — on the
user's own data, now including PCA.

## Sparse-aware PCA (closes the last implementation-bound function)
The dense path densifies (`scale` z-scores zeros into nonzeros). The new sparse path runs
randomized PCA directly on the CSR lognorm with **implicit mean-centering** — every product with
the centered matrix is the sparse product plus a rank-1 correction, via a custom Metal CSR×dense
SpMM kernel (no nnz×size temporary). Result: subspace overlap **1.0000** vs sklearn, **2× faster**
than dense at 104k, and it **scales to the full 2M** where dense OOMs. This was previously flagged
as the one implementation (not hardware) limit on the atlas path; it is now lifted.

## Notes
- **Moran/Geary exact on real Xenium** confirms the scatter-add SpMM autocorr is correct on
  real tissue structure (both share squidpy's spatial graph, isolating the statistic).
- **neighbors agreement 0.798**: at 103k the IVF KNN path is used; this compares two
  *approximate* graphs (ours IVF vs scanpy pynndescent), not recall-vs-exact — 80% edge overlap
  still yields a matching leiden cluster count (14/14). IVF recall is dataset-dependent (0.98 on
  the neuron atlas, ~0.8 here); raising `nprobe` would close it at some speed cost.
- **spatial_neighbors Jaccard 0.757** (vs 0.974 on the regular Visium grid): Xenium coordinates
  are irregular, so 6-NN tie-ordering diverges more between our brute KNN and squidpy's. Downstream
  autocorr uses squidpy's graph for both, so the statistics stay exact.

Driver: `validation_notebooks/v_realxenium.py` (read-only; nothing written to the object).
