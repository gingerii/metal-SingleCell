# Real SPATIAL-data validation — squidpy parity on real Visium

CLAUDE.md pin: the `gr` spatial functions had only been checked on synthetic spatial
patterns. Here they are validated against **squidpy 1.8.2** on a **real** spatial dataset
(10x Visium V1 Breast Cancer Block A Section 1, 3,798 spots with real tissue coordinates;
200 HVGs; 7 leiden clusters for label-based functions).

| function | real-Visium result vs squidpy | verdict |
|----------|-------------------------------|---------|
| spatial_neighbors | edge Jaccard **0.974** | near-identical (tiny KNN-symmetrization edge diffs) |
| spatial_autocorr (Moran's I) | corr **1.0000**, max\|Δ\| 5.9e-8 | exact |
| spatial_autocorr (Geary's C) | corr **1.0000**, max\|Δ\| 2.5e-7 | exact |
| co_occurrence | corr **1.0000**, max\|Δ\| 6.7e-4 | exact (fp32 distance vs squidpy fp64) |
| calculate_niche (composition) | max\|Δ\| **0.0** | exact |
| ligrec (cluster-mean scores) | max\|Δ\| 1.2e-7 | exact |

## Fixes this round (surfaced only by real-data parity)
- **Geary's C** was computed via the symmetric identity `2(Σ dᵢxᵢ² − Σ xᵢ(Wx)ᵢ)`, which is
  only exact for a perfectly symmetric weight matrix; on squidpy's real connectivity it gave
  corr 0.994, max\|Δ\| 0.12. Rewrote it to sum `Σ wᵢⱼ(xᵢ−xⱼ)²` **directly over the edge
  list** — exact for any graph. Now matches squidpy to 2.5e-7.
- **co_occurrence** had used *disjoint* distance bins (corr 0.60 vs squidpy). squidpy's
  definition is **cumulative** (`d ≤ threshold`) with conditional normalization
  `P(i | c, within r)/P(i)`. Rewrote to cumulative counting + squidpy's exact formula, and
  added an `interval=` arg to accept squidpy's threshold array. Now corr 1.0000.

## Notes
- Moran's I, niche composition, and ligrec means were already exact on real data — confirms
  the scatter-add SpMM and scatter-add cluster-means are correct on real tissue structure.
- All checks share one graph (squidpy's `spatial_connectivities`) so the autocorr comparisons
  are apples-to-apples math, isolating the statistic from graph-construction differences.

Driver: `validation_notebooks/v_realspatial.py`. Requires `squidpy` in the env.
