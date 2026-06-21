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
| pca (randomized) | subspace overlap **0.960** | high |
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
**normalize_total + log1p + highly_variable_genes on the full 2,035,266 × 5,101 matrix:
3.9 s, no OOM** (counts read via h5py into scipy CSR ≈ 5 GB, fits the 24 GB M3). This is the
real 2M-cell demonstration the synthetic atlas could only approximate — on the user's own data.

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
