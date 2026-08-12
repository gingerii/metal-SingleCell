# API parity with scanpy / squidpy

`metalsinglecell` is meant to be a drop-in for `scanpy.pp` / `scanpy.tl` / `squidpy.gr`, so a
pipeline works by swapping the namespace. This page records where that holds and where it
does not, audited against **scanpy 1.11.5** and **squidpy 1.8.2**.

Re-run the audit with `python tools/api_audit.py` after any signature change.

## Unsupported arguments raise

No function accepts `**kwargs`, and every scanpy argument we do not implement raises
`TypeError` rather than being silently accepted:

```python
msc.pp.highly_variable_genes(adata, batch_key="sample")
# TypeError: highly_variable_genes() got an unexpected keyword argument 'batch_key'
```

This page used to claim that property meant "a missing feature can never quietly change your
result". That was too strong, and a full audit in August 2026 found the counterexamples. Three
kinds of difference are *not* argument-shaped, so no `TypeError` can catch them:

1. **An argument we accept but never read.** `rank_genes_groups(reference=)`,
   `louvain(random_state=)`, `draw_graph(layout=)` and `gr.spatial_neighbors(coord_type=)` were
   all validated, recorded into `uns`, and ignored. Every one is fixed, and each now has a
   regression test; the class is what to watch for.
2. **A default that differs.** See *Defaults that differ* below.
3. **A key we do not write**, or write under another name. A pipeline that reads it gets a
   `KeyError` at best and a stale value at worst.

The completeness gaps in *Arguments we do not implement* really are only gaps — they raise.

## Deprecated upstream

| ours | status |
|---|---|
| `pp.pca(use_highly_variable=)` | deprecated in scanpy 1.10 → use `mask_var`. Still accepted, emits `FutureWarning`. Passing both raises. |
| `gr.spatial_neighbors()` | deprecated in squidpy 1.7, **removed in 1.9** → use the four mode-specific builders below. Still accepted, emits `FutureWarning`, and now dispatches for real. |

Nothing else in the public API mirrors a function or argument the reference has retired. The
audit checks both: a deprecated *parameter*, and a deprecated *function* that keeps its name
and signature (which is how `spatial_neighbors` went unnoticed until a user reported it).

### Spatial neighbour graphs

`gr.spatial_neighbors_knn`, `_radius`, `_delaunay` and `_grid` mirror squidpy's replacements.
Validated against squidpy 1.8.2 on real Visium across 15 option combinations: 11 match
element-wise, and the other 4 differ where two candidates are exactly equidistant. On a lattice
those ties are unavoidable — every Visium spot has six equidistant neighbours — and squidpy's
pick falls out of its tree traversal rather than a rule. Ours is deterministic: lowest index
wins.

Two further differences are **not** tie-driven, and this page previously said all of them were:

- **`percentile`** — squidpy's `setdiag(0)` inserts `n` explicit zeros before
  `np.percentile(dst.data, p)` sees them, so its threshold is computed over a longer array
  (132.386 against our 133.137 on one fixture). Ours is the defensible number; the edge sets
  agree at Jaccard ~0.98.
- **`transform="spectral"`** — we normalise by the **column** degree, matching squidpy's
  `axis=0`. Through 0.1.2 we used the row degree, which agrees on a symmetric graph and not on
  the directed k-NN graph. One deliberate difference remains: squidpy emits `inf` at a
  zero-in-degree node where we leave the row at zero.

Speed against squidpy on a jittered hex lattice (M3 Max, warm-up + best-of-N):

| n | knn(6) | radius(150) | grid(rings=2) | delaunay |
|---|---|---|---|---|
| 10k | 1.7× | 9.8× | 1.3× | 0.9× |
| 50k | 1.9× | 13.2× | 1.4× | 1.1× |
| 200k | 2.1× | 13.6× | 1.4× | 1.0× |
| 500k | 2.0× | 13.9× | 1.3× | 1.0× |

**Delaunay is not accelerated yet.** The triangulation runs on Qhull — the same library
squidpy uses — so it lands at parity; only the edge lengths and radius pruning around it are
ours. Qhull is 79-82% of that builder's runtime (8.6 s of 10.9 s at 2M points), so it is the
thing worth attacking next.

A GPU formulation exists and is well studied: GPU-DT and gDel2D build a digital Voronoi
diagram by jump flooding, dualise it, and repair the result by parallel flipping, reporting
~10x over Triangle and ~6x over CGAL. Two things keep it out of this release. Jump flooding
alone does **not** give a valid triangulation — a digital Voronoi region can be disconnected,
so its dual can contain duplicated and intersecting triangles — and the flipping repair that
fixes that is the bulk of the algorithm. It also needs adaptive-exact orientation/incircle
predicates, which bite harder here than in graphics: a Visium slide is a regular lattice, so
cocircular points are the common case, not a corner case. Getting that subtly wrong would
produce a plausible-looking but invalid graph, which is the failure mode this project has
already shipped twice.

All four take `library_key=`: the name of a categorical `obs` column identifying the section
each observation belongs to. The graph is then built per section and combined block-diagonally,
so no edge crosses between slides -- sections of a multi-slide object share a coordinate frame
only by accident. squidpy runs the whole tail (percentile prune, `set_diag`, transform) inside
that per-library loop and so do we, which matters most for `_grid`: its 1.3x-median cutoff has
to come from each section's own spacing. On a fixture pairing a 100-spaced and a 250-spaced
section, a pooled median puts the threshold between the two and the fine section keeps every
k-NN edge (mean degree 6.00 against a correct 3.80).

**`copy=True` returns the computed object, not the `AnnData`** — squidpy's contract, and
now ours for all of them: `SpatialNeighborsResult(connectivities, distances)` from the four
builders and the deprecated shim, a `(occ, interval)` tuple from `co_occurrence`, a
`{means, pvalues, metadata}` dict from `ligrec`, and a `DataFrame` from `spatial_autocorr`.
Nothing is written to the object in that mode.

`spatial_neighbors_from_builder` is **not implemented**: it takes a squidpy `GraphBuilder`
instance, and squidpy is an optional (`oracle`) dependency here, not a runtime one. The only
other arguments the four builders lack are `elements_to_coordinate_systems` and `table_key`,
which address a `SpatialData` object rather than an `AnnData`; `n_jobs` is a CPU-threading knob
with no meaning on the GPU path.

`pp.pca(mask_var=)` follows scanpy's three-state default exactly, which is easy to get wrong:

```python
msc.pp.pca(adata)                       # -> highly_variable if that column exists
msc.pp.pca(adata, mask_var=None)        # -> ALL variables (not the same thing!)
msc.pp.pca(adata, mask_var="my_col")    # -> adata.var["my_col"]
msc.pp.pca(adata, mask_var=bool_array)
```

## Defaults that differ

These are the silent ones — same call, different result — so they are listed even where our
choice is deliberate.

| function | ours | reference | effect |
|---|---|---|---|
| `tl.leiden` **flavor** | igraph | `leidenalg` | this, not `n_iterations`, is what makes our labels differ. Decomposed on one graph: scanpy with `n_iterations=2` but its own flavor is **ARI 1.000** against its default, while `flavor="igraph"` is **0.687**. Ours against scanpy's default is 0.749, and against `scanpy(flavor="igraph")` it is 0.795 — inside that backend's own seed-to-seed spread (0.700). Modularity is equal or better (0.6028 vs 0.5934). |
| `tl.leiden(n_iterations=)` | `2` | `-1` | scanpy iterates to convergence. Within our backend, `2` vs `-1` gives ARI 1.000 at resolution 0.5 and 0.991 at 1.0 — so this contributes almost nothing next to the flavor. |
| `gr.calculate_niche(random_state=)` | `0` | `42` | labels differ run-to-run only. |
| `pp.scrublet` **embedding** | joint observed+simulated log1p PCA | observed-only z-scored fit | deliberate. Ours measured a higher injected-doublet AUC (0.972 against 0.796), so it is kept. The neighbour count, score smoothing and call threshold now follow the reference. |
| `.X` dtype | `float32` | `float64` | our CSR kernels are fp32 throughout. Consistent, and stated here rather than discovered. |

Fixed in 0.1.3: `gr.spatial_autocorr` returned only the statistic and `pval_sim`, and that
`pval_sim` counted one tail — the clustered one. A gene with significant *negative* spatial
autocorrelation therefore scored p ≈ 1 where squidpy scores p ≈ 0.001, under the same column
name. squidpy folds the permutation count to the smaller tail; we now do too. The frame also
carries the rest of squidpy's columns (`pval_norm`, `var_norm`, and with `n_perms` also
`pval_z_sim`, `var_sim`, plus a `*_fdr_bh` per p-value), `n_perms` now defaults to `None` as
squidpy's does rather than `100`, and `copy=True` returns the `DataFrame` rather than the
`AnnData`. Benjamini–Hochberg is implemented in-package rather than pulled from statsmodels,
and is pinned against it to 1e-12.

### The August 2026 audit

A six-way review against scanpy 1.11.5 / squidpy 1.8.2 compared every public function's
signature, the keys it writes, and its values. Everything it found is fixed in 0.1.3; the
findings are worth keeping visible because they show which *kinds* of difference this project
keeps producing.

Corrupt or crashing:

- `gr.spatial_neighbors_radius(percentile=)` returned a graph whose distances sat on the wrong
  edges — the connectivity and distance matrices shared one `indices` buffer and the two
  `eliminate_zeros()` calls scrambled each other. 33904 of 41926 entries misplaced.
- `pp.regress_out` aborted the interpreter (SIGABRT) on a rank-deficient design, which a
  covariate of all zeros produces — `pct_counts_mt` on a panel with no MT- genes.
- `copy=True` on a backed `.X` raised `'_CSRDataset' object has no attribute 'copy'`, and the
  streaming transform prefix in `uns` made the object unwritable to h5ad.

Silently wrong numbers:

- `tl.rank_genes_groups` reported a difference of log-means as `logfoldchanges` where scanpy
  reports `log2` of the expression ratio — correlation 0.173, and
  `rank_genes_groups_df(log2fc_min=1.0)` returned 17 genes against scanpy's 196.
- `pp.neighbors` built the graph in gene space when no `X_pca` was present (neighbour overlap
  0.188), and `pp.pca` used a randomized solver where scanpy uses arpack.
- `pp.calculate_qc_metrics` chose its output slot by array length, so on a **square** object
  the per-gene metrics landed in `.obs` and overwrote the per-cell ones.
- `tl.embedding_density` ran the KDE over every component of the basis; on a 30-D PCA basis the
  result was a numerically-zero map that still plots.
- `gr.spatial_autocorr(mode=)` was unvalidated, so a capitalisation typo computed Geary's C
  against Moran's null and filed it under `uns['gearyC']`.
- `gr.co_occurrence` used a different distance grid and returned one bin too many;
  `gr.ligrec` mislabelled rows after a dropped interaction; `pp.scrublet`'s threshold was a
  quantile that made `expected_doublet_rate` a hard quota.

Contract breaks: missing `uns` params on six functions (`sc.tl.ingest` and `sc.pl.draw_graph`
raised on our objects), lexicographic instead of natsorted cluster categories, `copy=True`
returning the wrong type in `gr`, dense `.X` silently converted to CSR, and `igraph`/`sklearn`
undeclared while sitting on default code paths.

Fixed in 0.1.2: `gr.spatial_neighbors(coord_type=)` was accepted and **ignored** — every call
returned a generic k-NN graph, so `coord_type='grid'` silently produced the wrong graph on
Visium. It now dispatches, and infers `'grid'` from `uns['spatial']` as squidpy does.

Fixed in 0.1.1: `pp.calculate_qc_metrics(percent_top=)` defaulted to `None`, silently producing
none of scanpy's `pct_counts_in_top_*` columns; it is now scanpy's `(50, 100, 200, 500)`. On a
backed `.X` those columns cannot be streamed, so the **default** is skipped there rather than
raising; explicitly passing `percent_top` on a backed object still raises.

Where ours is a concrete value and the reference is `None` — `tl.rank_genes_groups(method=
't-test')`, `tl.louvain(resolution=1.0)` — the reference resolves `None` to the same thing.
Those are not drift.

That blanket claim used to cover `pp.pca(n_comps=)` and `pp.pca(svd_solver=)` as well, and it
was wrong for both. scanpy resolves `svd_solver=None` to **arpack**, not to a randomized
solver: on pbmc3k the two agreed to `|corr| >= 0.99` only through PC15 and reached `|corr| =
0.012` by PC50. And `n_comps=None` resolves to `min_dim - 1` on a panel with fewer than 51
features, not to 50. Both now follow scanpy. `tl.tsne(use_rep=)` likewise defaults to `None`
and resolves through scanpy's rule rather than assuming `X_pca`.

## Arguments we do not implement

117 in total. The ones most likely to be reached for:

| function | missing |
|---|---|
| `pp.highly_variable_genes` | `batch_key`, `subset`, `span`, `check_values` |
| `pp.neighbors` | `metric`, `metric_kwds`, `method`, `knn`, `transformer` (we are Euclidean-only) |
| `tl.rank_genes_groups` | `groups`, `n_genes`, `corr_method`, `pts`, `mask_var`, `rankby_abs` |
| `tl.leiden` / `tl.louvain` | `flavor`, `use_weights`, `restrict_to`, `neighbors_key`, `obsp`, `partition_type` |
| `pp.scale` | `mask_obs`, `obsm` |
| `pp.log1p` | `base`, `obsm` |
| `pp.scrublet` | tuning surface (`batch_key`, `threshold`, `n_prin_comps`, …). The **embedding** also differs deliberately — see below. |
| `gr.ligrec` | `complex_policy`, `threshold`, `corr_method`, `use_raw`, `gene_symbols` |
| `gr.calculate_niche` | most of squidpy 1.8's expanded surface |

`tl.umap` follows scanpy's argument *names* (`init_pos`, `alpha`, `gamma`,
`negative_sample_rate`, `maxiter`, `a`, `b`, `neighbors_key`, `key_added`); `n_epochs` is kept
as an alias of `maxiter`. `method='rapids'` is not offered — this *is* the GPU path.

## Ours only

`pp.pca(svd_solver=)` takes our solver names; `tl.leiden(backend=|variant=|commit_prob=)` select
the Metal parallel Leiden; `tl.tsne(exact_max_n=)` sets the exact/Barnes-Hut crossover;
`embedding.umap(max_node_step=)` is the layout trust region. `pp.bbknn`, `pp.harmony_integrate`,
`pp.normalize_pearson_residuals`, `tl.kmeans`, `pp.materialize` and `pp.write_obsm` have no
direct `sc.pp`/`sc.tl` counterpart under that name (some mirror `rapids_singlecell` or
`scanpy.external`).
