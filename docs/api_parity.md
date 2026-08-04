# API parity with scanpy / squidpy

`metalsinglecell` is meant to be a drop-in for `scanpy.pp` / `scanpy.tl` / `squidpy.gr`, so a
pipeline works by swapping the namespace. This page records where that holds and where it
does not, audited against **scanpy 1.11.5** and **squidpy 1.8.2**.

Re-run the audit with `python tools/api_audit.py` after any signature change.

## Unsupported arguments raise

No function accepts `**kwargs`. Every scanpy argument we do not implement raises `TypeError`
rather than being silently ignored — so a missing feature can never quietly change your
result:

```python
msc.pp.highly_variable_genes(adata, batch_key="sample")
# TypeError: highly_variable_genes() got an unexpected keyword argument 'batch_key'
```

That is the important safety property. The gaps below are *completeness* gaps, not
correctness hazards.

## Deprecated upstream

| ours | status |
|---|---|
| `pp.pca(use_highly_variable=)` | deprecated in scanpy 1.10 → use `mask_var`. Still accepted, emits `FutureWarning`. Passing both raises. |

Nothing else in the public API uses an argument scanpy has deprecated or removed.

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
| `gr.spatial_autocorr(n_perms=)` | `100` | `None` | we compute permutation p-values by default; squidpy computes none. Extra columns, ~100× the work on a default call. |
| `gr.spatial_neighbors(coord_type=)` | `'generic'` | `None` | squidpy infers `'grid'` for Visium from `uns['spatial']`; we always treat coordinates as generic, giving a different graph on grid platforms. |
| `gr.spatial_neighbors(n_neighs=)` | `6` | `None` | squidpy's effective default depends on `coord_type`. |
| `tl.leiden(n_iterations=)` | `2` | `-1` | scanpy iterates to convergence. Measured ARI between the two on the same graph: **0.958** — same cluster count, slightly different labels. |
| `gr.calculate_niche(random_state=)` | `0` | `42` | labels differ run-to-run only. |

Fixed in 0.1.1: `pp.calculate_qc_metrics(percent_top=)` defaulted to `None`, silently producing
none of scanpy's `pct_counts_in_top_*` columns; it is now scanpy's `(50, 100, 200, 500)`. On a
backed `.X` those columns cannot be streamed, so the **default** is skipped there rather than
raising; explicitly passing `percent_top` on a backed object still raises.

Where ours is a concrete value and the reference is `None` — `pp.pca(n_comps=50)`,
`tl.rank_genes_groups(method='t-test')`, `tl.tsne(use_rep='X_pca')`, `tl.louvain(resolution=1.0)`
— the reference resolves `None` to the same thing. Those are not drift.

## Arguments we do not implement

129 in total. The ones most likely to be reached for:

| function | missing |
|---|---|
| `pp.highly_variable_genes` | `batch_key`, `subset`, `span`, `check_values` |
| `pp.neighbors` | `metric`, `metric_kwds`, `method`, `knn`, `transformer` (we are Euclidean-only) |
| `tl.rank_genes_groups` | `groups`, `n_genes`, `corr_method`, `pts`, `mask_var`, `rankby_abs` |
| `tl.leiden` / `tl.louvain` | `flavor`, `use_weights`, `restrict_to`, `neighbors_key`, `obsp`, `partition_type` |
| `pp.scale` | `mask_obs`, `obsm` |
| `pp.log1p` | `base`, `obsm` |
| `pp.scrublet` | most of the tuning surface (`batch_key`, `threshold`, `n_prin_comps`, …) |
| `gr.spatial_neighbors` | `radius`, `delaunay`, `n_rings`, `percentile`, `set_diag`, `library_key` |
| `gr.spatial_autocorr` | `transformation`, `two_tailed`, `corr_method`, `attr`, `use_raw` |
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
