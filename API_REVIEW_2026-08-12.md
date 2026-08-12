# Full API review vs scanpy 1.11.5 / squidpy 1.8.2 — 2026-08-12

Six parallel audits over `pp`, `tl` and `gr`, each required to back every claim with a measured
comparison against the installed reference. Findings marked **[v]** were re-verified independently
afterwards, outside the agent that raised them.

The review was commissioned before releasing 0.1.3. It found enough that 0.1.3 should be a
correctness release, and that two of the bugs are already published.

Scope note: this covers *behaviour against the reference*. It is not a security or performance
review.

---

## Tier 0 — already published in 0.1.2, introduced by PR #5

Both are in `_finalize`/`_transform_adj`, the shared tail added when the four spatial builders
landed. Neither is on a default code path, which is the only reason the blast radius is small.

### 0.1 `spatial_neighbors_radius(percentile=…)` returns a corrupted graph **[v]**
`src/metalsinglecell/gr.py:80-84`

Both matrices are built from the same `cols` array. scipy stores `indices` by reference when the
dtype already matches, so `np.shares_memory(adj.indices, dst.indices)` is `True`; the two
`eliminate_zeros()` calls then compact that single buffer in place and each scrambles the other
matrix.

Measured, 400-point cloud, `radius=150, percentile=80`: **33,904 of 41,926** stored distances do
not match the pair they are indexed under, max error **610** on a radius-150 graph; the result is
asymmetric and carries 141 self-loops. Without `percentile`: 0 mismatches.

`knn` escapes by accident (its `cols` is `int64`, forcing a downcast copy); `delaunay` builds
through COO. `grid` never passes a percentile. The bug is latent for any builder whose index array
is already `int32`.

Missed because `tools/spatial_neighbors_parity.py` covers `knn percentile=90` and
`delaunay percentile=95` but no radius case, and `tests/test_spatial_neighbors.py` has no
percentile case at all.

### 0.2 `transform="spectral"` normalises by the wrong axis **[v]**
`src/metalsinglecell/gr.py:39-44`

We use row degree; squidpy uses column degree (`gr/neighbors.py:521`, `axis=0`). Identical on a
symmetric graph, different on the directed k-NN graph — the default builder. Ours reproduces
row-degree normalisation to `0.00e+00`, squidpy reproduces column-degree to `2.98e-08`; max
difference on the connectivity values **0.187**.

squidpy emits `inf` at zero-in-degree nodes (18 on a 300-point cloud) where our `where=deg>0`
guard emits 0. Ours is the more defensible number but it is a different matrix.

Missed because the spectral test gates its value assertion behind
`if (abs(A - B) > 0).nnz == 0` — it compares values only when they already agree.

---

## Tier 1 — crashes and unrecoverable states

### 1.1 `regress_out` aborts the interpreter on a singular design matrix **[v]**
`src/metalsinglecell/preprocess.py:123`

`mx.linalg.solve` on a rank-deficient `DᵀD` raises a C++ exception that is not catchable from
Python: exit **-6**, `sgetrf_ failed with code 3`. It kills the kernel. scanpy tests the
determinant and falls back to a per-column GLM.

The realistic trigger is a covariate that is identically zero — `pct_counts_mt` on a panel with no
MT- genes, which is normal for targeted panels like Xenium, this project's main use case.

### 1.2 `copy=True` crashes on any backed `.X`
`pp.py:155,180,203,252,321,453` — `normalize_total`, `log1p`, `highly_variable_genes`, `scale`,
`pca`, `calculate_qc_metrics`

Each does `adata.copy()` before consulting `_backed_reader`:
`AttributeError: '_CSRDataset' object has no attribute 'copy'`. `filter_cells` and `regress_out`
get this right via `_reject_backed`, so the fix pattern already exists in the file.

### 1.3 `uns['_stream_transforms']` leaks and makes the object unwritable
`pp.py:73-76`

A public `uns` key holding tuples of numpy arrays, cleared only by `pp.materialize`. After a
streaming `normalize_total → log1p → scale`, `adata.write_h5ad()` raises
`ValueError: setting an array element with a sequence`.

---

## Tier 2 — silent wrong results

Ordered by how likely a user is to hit them.

### 2.1 `rank_genes_groups` `logfoldchanges` uses the wrong formula **[v]**
`tools.py:178,199`

Ours is `mean_g − mean_r`, a difference of means of log1p values in natural-log units. scanpy is
`log2((expm1(mean_g) + 1e-9) / (expm1(mean_r) + 1e-9))`. Both formulas confirmed exactly.

Pearson r **0.173**, max difference **25.4**, median magnitude ratio **48×**.
`sc.get.rank_genes_groups_df(pval_cutoff=0.05, log2fc_min=1.0)` returns **196 genes on scanpy's
output and 17 on ours**; at `log2fc_min=2.0`, **98 vs 2**. `sc.tl.filter_rank_genes_groups` keeps
702 vs 80.

Wrong in every published release.

### 2.2 `rank_genes_groups(reference=)` accepted, recorded, never read **[v]**
`tl.py:135,166` / `tools.py:112` — found independently by two agents

Scores are bit-identical to `reference="rest"`; we emit 8 group fields where scanpy emits 7 (it
drops the reference group); `params` records the reference we did not use; an invalid group name
does not raise where scanpy raises `ValueError`.

Same failure mode as the `coord_type` bug fixed in 0.1.2 — accepted, stored in a `uns` params
dict, never read.

### 2.3 `pp.neighbors` builds the graph in gene space **[v]**
`pp.py:502-504`

With no `X_pca` present, scanpy's `_choose_representation` computes a 50-component PCA when
`n_vars > 50`. We fall back to raw `.X`.

Measured on 2700 × 2000 HVGs: scanpy builds from a `(2700, 50)` PCA, we use all 2000 genes —
neighbor-set overlap **0.188**. On a *sparse* `.X` it does not even fall through, it raises
`ValueError: setting an array element with a sequence`. So `msc.pp.neighbors(adata)` straight after
`msc.pp.log1p` either crashes or silently builds the wrong graph.

### 2.4 `pp.pca` default solver diverges from scanpy's **[v]**
`pp.py:435`

scanpy resolves `svd_solver=None` to `arpack` for both dense and sparse. Ours defaults to
`randomized`, and on sparse `.X` every exact solver raises
`ValueError: CSR input supports solver='randomized' only` — there is no way to reproduce scanpy's
default result on a sparse matrix.

Measured on pbmc3k 2700 × 2000, `n_comps=50`: first PC with `|corr| < 0.99` is **PC16**, min
`|corr|` over 50 components **0.012**. Downstream kNN overlap 0.674 at `n_pcs=50`, 0.939 at
`n_pcs=15`.

Not our kernel: sklearn's `randomized_svd` at the same settings breaks at the same component, and
raising `n_iter`/`n_oversamples` restores `|corr| = 0.998`. It is the default choice.

`docs/api_parity.md` explicitly claims this is *not* drift. That claim is wrong.

### 2.5 `calculate_qc_metrics` writes per-gene metrics into `.obs` on square objects **[v]**
`pp.py:341`

The output slot is chosen by array length, not by which axis the metric belongs to. On a square
object every per-gene array has `len == n_obs` and lands in `.obs`, overwriting the per-cell
values. Measured 600 × 600: our `.var` is **empty** where scanpy writes six columns, and
`obs["total_counts"]` differs by **202**. Non-square control matches exactly.

Reachable: 2000 cells subset to 2000 HVGs.

### 2.6 `tl.embedding_density` runs the KDE over every component
`tl.py:207` / `tools.py:364-382`

scanpy uses exactly two components and guards for it. On a 30-D PCA basis our output is
numerically all-zero (max **1.04e-11**) and plots as a blank map; on diffmap it anticorrelates
(**r = −0.314**) because scanpy skips the trivial eigenvector. On a 2-D UMAP basis we match to
4.3e-11.

Compounding: the `obs` key ignores `groupby` (so two calls overwrite each other) and no
`uns[…_params]` is written, so `sc.pl.embedding_density` rejects our output with either key.

### 2.7 Parameters accepted and ignored
Same class as 2.2, grouped because the fix is the same shape.

| parameter | our line | effect |
|---|---|---|
| `tl.louvain(random_state=)` | `tl.py:55-78` | read only on the `gpu` backend; the default `igraph` path is unseeded — three identical calls give three answers, ARI 0.789 |
| `tl.draw_graph(layout=)` | `tl.py:127-132` | names the output key only; every layout runs the same SGD, and `layout="banana"` is accepted |
| `tl.tsne(use_rep=)` | `tl.py:112` | silently falls back to `.X` when the key is absent; scanpy raises |
| backed `pp.pca(zero_center=, svd_solver=, random_state=)` | `pp.py:461-474` | none is forwarded to the streaming path, yet `zero_center` is recorded in `uns`; the in-memory path honours all three |
| `tl.leiden(variant=, commit_prob=)` | `cluster.py:15-33` | ours-only args, ignored on the default backend (documented, but contradicts the parity page's headline claim) |

### 2.8 `gr.spatial_autocorr(mode=)` is unvalidated — **in the unreleased 0.1.3 code**
`spatial.py:368-376`, `gr.py:447-463`

Any string other than `"moran"` computes Geary's C but evaluates it against **Moran's** analytic
null and writes it to `uns['gearyC']`. Measured: `mode="Moran"` (a capitalisation typo) yields
`pval_norm = 0.0` for every gene, sorted the wrong way. squidpy raises. We validate `transform`,
`corr_method` and `attr` but not `mode`.

Cheap to fix and should be fixed before 0.1.3 ships.

### 2.9 `co_occurrence` uses a different distance grid and returns a different shape
`gr.py:467-473`, `spatial.py:220-222`

Off-by-one: squidpy's `interval` is a count of *thresholds* (`interval-1` bins); we produce
`interval` bins. Measured `interval=20` on IMC: our `occ` is `(11,11,20)` vs squidpy's
`(11,11,19)`, so `sq.pl.co_occurrence` cannot read it.

Different range: squidpy derives min/max from a coordinate-sum heuristic and halves the max; we
use true min-nonzero and true max pairwise distance, so our grid extends ~2× further and starts
~2000× lower.

The kernel is correct — fed squidpy's own `interval` array, our `occ` matches at max abs 1.03e-3,
corr 1.000000.

### 2.10 `ligrec` mislabels rows when an interaction gene is missing
`spatial.py:93-95,117`

Pairs are filtered, then labelled positionally rather than by the surviving indices. With
`[("g0","g1"), ("NOT_A_GENE","g3"), ("g4","g5")]`, row 1 holds the correct values for
`("g4","g5")` but is reported as `("NOT_A_GENE","g3")`.

### 2.11 `pp.scrublet` is a different algorithm
`preprocess.py:191-211`

Doublet-score Pearson **0.417** against scanpy, `predicted_doublet` Jaccard **0.19** (6 vs 25
calls). Four verified divergences: k (14 vs scanpy's `round(k·(1+n_sim/n_obs))` = 42); threshold
(a fixed quantile of the *observed* scores, not `threshold_minimum` on the *simulated* histogram —
which makes `expected_doublet_rate` a hard quota by construction); the embedding (log1p + joint
observed/simulated PCA vs scanpy's z-scored observed-only fit); and the smoothing constant.

The docstring calls the threshold an "Otsu-like split of the score distribution"; it is a fixed
quantile. That comment is misleading and should change with the code.

### 2.12 `regress_out` fits categorical covariates as an ordinal slope
`pp.py:272`

scanpy detects `CategoricalDtype` and regresses on per-gene means within each category. We coerce
to float: string categories raise, numeric categories silently fit a slope. Measured max abs
difference 0.257; scanpy forces per-group residual means to zero, ours are not.

### 2.13 Smaller silent divergences

- `use_raw` default (`tl.py:142`, `tl.py:181`): scanpy uses `.raw` when present, we always use
  `.X`, and we reject `use_raw=` so the user cannot opt back in. On the standard tutorial object
  this runs t-tests on z-scaled negative data; top-10 marker overlap 0.6.
- `gr.spatial_autocorr(use_raw=True)` (`gr.py:407-412`): scanpy intersects `.raw` with
  `adata.var_names`; we take all of `.raw`. 12 rows vs 6 — and since FDR divides by the row count,
  every corrected p-value differs.
- Categorical categories are lexicographic, not natsorted (`tl.py:50,77,219`, `gr.py:499`).
  Identical below 10 clusters, scrambled at or above: `['0','1','10','11',…,'2',…]`. `.cat.codes`
  no longer matches the label, so colour maps and positional group access misalign.
- Deprecated shim: `delaunay=True` with a *scalar* `radius` (`gr.py:382-383`). squidpy documents
  that the legacy entry point ignores it; we prune. Jaccard 0.869. Tuple radius matches.
- Nonzero counting uses `> 0` where scanpy uses `!= 0` (`sparse.py:40,300`). Only bites on
  non-count input, and propagates into `filter_cells`/`filter_genes`.
- `normalize_pearson_residuals` does not validate `theta`; `theta <= 0` returns non-finite values
  where scanpy raises.
- `harmony_integrate` deviates on three defaults (`ridge_lambda`, tolerance, block size) and
  rejects a multi-column `key`. Output diverges materially from harmonypy — though on the tested
  fixture ours corrects *better*, so this is a documentation gap rather than a clear defect.
- `pp.bbknn`: writes `n_neighbors=3` where the reference writes 9, skips the reference's default
  trim (max row degree 158 vs 90), ignores `n_pcs`, and includes self-edges — inconsistent with
  our own `pp.neighbors`, which excludes them.

---

## Tier 3 — missing keys and broken contracts

Each breaks a drop-in swap without being numerically wrong.

| function | reference writes, we do not |
|---|---|
| `pp.pca` | `uns['pca']['variance']`; `params['layer']` |
| `pp.neighbors` | `params['metric']`, `params['random_state']` — `sc.tl.ingest` raises `KeyError: 'metric'` |
| `pp.scale` | `var['mean']`, `var['std']` |
| `pp.filter_cells` | `obs['n_genes']` / `obs['n_counts']` |
| `pp.filter_genes` | `var['n_cells']` / `var['n_counts']` |
| `pp.normalize_pearson_residuals` | `uns['pearson_residuals_normalization']` |
| `pp.calculate_qc_metrics(qc_vars=)` | `obs['log1p_total_counts_<v>']`; and we write an extra `var['log1p_n_cells_by_counts']` scanpy does not |
| `pp.scrublet` | `uns['scrublet']` missing `doublet_parents`, `doublet_scores_sim`, `parameters` — `sc.pl.scrublet_score_distribution` needs them |
| `tl.umap` / `tl.tsne` / `tl.louvain` / `tl.draw_graph` | their `uns[...]` params blocks — `sc.pl.draw_graph` raises `KeyError: 'draw_graph'` |
| `tl.leiden` | `params['random_state']` |
| `tl.rank_genes_groups` | `params['layer']`, `params['corr_method']` |
| `gr.ligrec` | squidpy returns MultiIndex DataFrames + `metadata`; we return raw ndarrays — `sq.pl.ligrec` cannot consume ours |

Contract-level:

- **`copy=True` return type.** Fixed for `spatial_autocorr` in 0.1.3; the same mismatch remains in
  six siblings — the four builders and the deprecated shim return `AnnData` where squidpy returns
  `SpatialNeighborsResult`; `co_occurrence` should return a tuple, `ligrec` a dict.
- **`calculate_qc_metrics` default is inverted.** scanpy defaults to `inplace=False`, returning two
  DataFrames and writing nothing. We always write and return `None`, and reject `inplace=`, so both
  spellings of the scanpy idiom break.
- **`neighbors_key` is plumbed into `tl.umap` only.** `_conn` already resolves it. With both a
  default and a redirected graph present, `msc.tl.leiden` silently uses the default with no way to
  redirect — measured 11 clusters vs scanpy's 9 on the redirected graph.
- **`pp.neighbors` has no `key_added`,** so we cannot create a redirected graph in the first place.
- **Dense `.X` is silently converted to CSR** by `normalize_total` and `log1p` (`pp.py:22-33`).
  Changes `adata.X`'s type behind the user's back. The fp32 downcast is a defensible GPU-wide
  policy but is undocumented.
- **`calculate_niche` has no overlapping call form with squidpy 1.8.2** — no `cluster_key`,
  `n_niches`, `connectivity_key`, `key_added` or `copy` upstream; `flavor` is a required
  positional. We raise rather than mislead, but the parity page understates this badly.

---

## Tier 4 — dependencies, tooling, documentation

### 4.1 Undeclared dependencies on default code paths
Core deps are numpy, scipy, anndata, mlx. Not declared anywhere, yet reachable:

| path | import | reachability |
|---|---|---|
| `tl.leiden` / `tl.louvain` default backend | `igraph` | **every default call** |
| `gr.spatial_neighbors_*(transform="cosine")` | `sklearn` | documented squidpy-parity option on all five builders |
| `tl.rank_genes_groups(method="logreg")` | `sklearn` | one of scanpy's four documented methods |
| `pp.highly_variable_genes(flavor="seurat_v3")` | `skmisc`, `statsmodels` | accepted by the wrapper |
| backed/streaming path, `pp.materialize` | `zarr` | headline out-of-core feature |

`igraph`, `sklearn` and `zarr` are effectively runtime dependencies. `skmisc` and `statsmodels`
are in no extra at all.

### 4.2 `tools/api_audit.py` cannot see most of this
- `REF_NOISE` suppresses `copy`, `inplace`, `key_added` and 7 others as cosmetic. `inplace` and
  `key_added` are functional; `copy` hides a **live upstream deprecation** on
  `filter_cells`/`filter_genes` (scanpy emits `"copy is deprecated, use inplace instead"` — observed
  in this run). The "116 missing arguments" figure undercounts by 17.
- It pairs `msc.pp ↔ sc.pp` only, so `normalize_pearson_residuals` has **never been audited** — its
  counterpart is `sc.experimental.pp`.
- It is signature-only, so no missing key, wrong value, or ignored parameter in this document was
  visible to it.

### 4.3 Claims in `docs/api_parity.md` that are false
1. *"Every scanpy argument we do not implement raises `TypeError` … a missing feature can never
   quietly change your result."* Contradicted by 2.2, 2.7, and the `percentile`-on-grid drop.
2. *"Where ours is a concrete value and the reference is `None` … the reference resolves `None` to
   the same thing. Those are not drift."* False for `pp.pca(svd_solver=)` (2.4) and for `n_comps`
   on panels with fewer than 51 features (scanpy uses `min_dim - 1`).
3. *"[builders] differ only where two candidates are exactly equidistant."* False for `percentile`
   (squidpy's threshold includes the explicit zeros its `setdiag(0)` inserts) and for `spectral`
   (0.2).
4. The `copy=True` return-type fix is described as if `spatial_autocorr` were the only case.
5. `pp.scrublet`'s row says "most of the tuning surface missing" — the scores and calls differ.
6. `pp.bbknn`, `pp.harmony_integrate` and `gr.co_occurrence` have no deviation rows at all.

---

## What is clean — recorded so the negatives are on the record

- **No function in the public API mirrors a scanpy/squidpy function that is deprecated or removed**,
  beyond the two we already shim deliberately (`pp.pca(use_highly_variable=)`,
  `gr.spatial_neighbors`). The `use_highly_variable` shim was verified correct. One live upstream
  deprecation we do expose: `copy` on `filter_cells`/`filter_genes`.
- **`copy=` mechanics.** All 34 functions taking `copy` were fingerprinted per-slot: none mutates
  the input under `copy=True`, all return `None` under `copy=False`.
- **The lazy-import contract holds.** After importing the package and touching all three
  namespaces, only `numpy` is in `sys.modules` from the watched set.
- **No `**kwargs` anywhere** in the public API.
- **Numerical agreement is excellent wherever the algorithm matches**: `normalize_total`, `log1p`,
  `scale` and `regress_out` at r = 1.0000000000; HVG seurat/seurat_v3 selection identical;
  `calculate_qc_metrics` shared columns to 7.6e-08; `rank_genes_groups` t-test/wilcoxon scores
  exact in fp32 with top-100 overlap 1.00; `score_genes` bit-identical; PCA head components
  `|corr| ≥ 0.99995`; kNN recall 0.995 on a shared embedding; spatial graph builders element-wise
  identical outside the two bugs above.
- **The 0.1.3 `spatial_autocorr` rewrite holds up** — column set and sort order identical across
  all 16 option combinations, `attr` modes correct, permutation null matching squidpy's
  construction, constant genes `NaN` in both. One correction to what was claimed when it landed:
  the fp32 graph dtype is the *larger* of **two** sources of the analytic difference — our W·X
  reduction also runs in fp32, contributing ~9.4e-8 on the statistic itself.

---

## Outcome

**Everything above is fixed in 0.1.3**, in one release rather than the three originally
proposed — the user's call, on the grounds that the package is new enough that a staged
rollout buys nothing. 192 tests pass, up from 116; every finding in Tiers 0-2 has a regression
test named after the failure it guards.

Two deviations are kept deliberately and are now documented rather than silently present:
the `scrublet` embedding (ours measured a higher injected-doublet AUC, 0.972 against 0.796)
and the spectral transform's zero-degree guard (squidpy emits `inf`; we leave the row at zero).

## Proposed sequencing (superseded — all shipped in 0.1.3)

**0.1.3 — correctness release.** Tier 0 (both are published), Tier 1 (crashes), and 2.8 (cheap,
and it is in unreleased code). Plus the issue #6 fix already on the branch.

**0.1.4 — parity release.** The rest of Tier 2, led by `logfoldchanges` and `reference=`, then
`pp.neighbors`/`pp.pca` defaults. These change results for existing users, so they need release
notes that say so plainly.

**0.1.5 — contract release.** Tier 3 keys and return types, `neighbors_key` plumbing,
`calculate_qc_metrics(inplace=)`.

**Alongside:** declare the real dependencies (4.1), rewrite `tools/api_audit.py` to diff written
keys as well as signatures (4.2), and correct the six false claims in `docs/api_parity.md` (4.3).

**On yanking 0.1.2:** not recommended. Both Tier 0 bugs sit behind non-default arguments
(`percentile=`, `transform="spectral"`), 0.1.2 fixes two silent-wrong-result bugs that are worse
and unavoidable, and the versions it would fall back to carry the whole of Tier 2 as well. A fast
0.1.3 with an explicit note is the better trade.

**On disclosure:** `logfoldchanges` has been wrong in every published release and is a number
people put in figures and tables. It deserves saying out loud in the release notes rather than
appearing as a bullet in a changelog.
