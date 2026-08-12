# POSTFIX_VALIDATION — re-verifying the 7 review-fix commits

**Verdict: all 7 fix commits validated.** 39/39 checks pass across the regression baseline, the new
default/raise/edge-case paths the parity suite can't see, and the two at-scale checks (fp64 moments at
1.2M cells, Leiden on the real 986k-neuron graph). One item needs a **product decision, not a fix** — the
user-facing Leiden `n_iterations` default (Step 5); the banked benchmark numbers are **not** invalidated.

Test seeds: `tests/test_postfix_fixes.py` (asserting, CI-ready — CPU + Metal) and
`results/code_review/postfix_scale.py` (the ≥1M / 986k scale runs). Machine-readable results in
`postfix_validation.csv`.

## What the parity suite covered vs what these checks added
The `validation_notebooks/` parity scripts pass **explicit args** everywhere
(`highly_variable_genes(csr, n_top_genes=2000)`, `scale(csr, max_value=10)`) and were the basis for the
green Step-1 regression — but they never touch the changed **defaults**, the new **raise-paths**, the
**BLOCKER** (`n_neighbors>32`), or **fp32-at-scale**. Steps 2–6 add exactly those. A green parity run
alone does **not** mean "fixes validated"; the additions below are why we can say it.

## Step 1 — Regression baseline (necessary, not sufficient) — 9/9 PASS
Re-ran the full parity + out-of-core suites on the post-fix code, capturing each PASS/FAIL verdict:
`01_qc, 02_normalize_log1p, 03_hvg, 04_scale, 05_pca, 09_graph_primitives, 11_leiden_gpu,
v_outofcore (streaming), v_outofcore_m2` — **all PASS**. Confirms the fp64-moments change (`5d163d5`)
did not disturb the small-scale HVG/scale/PCA snapshots (HVG still 2000/2000; means actually tightened).

## Step 2 — New drop-in defaults now match scanpy (fix `b48b61e`) — 5/5 PASS
- `highly_variable_genes(adata)` default (`n_top_genes=None` → cutoff mode) vs `sc.pp.highly_variable_genes`:
  **Jaccard = 1.000** (gene-for-gene).
- `scale(adata)` default (`max_value=None`, no clip) vs `sc.pp.scale`: max|Δ| < 1e-4.
- `calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=[50], log1p=True)`: all scanpy slot names present
  (`total_counts`, `n_genes_by_counts`, `pct_counts_mt`, `log1p_total_counts`, `pct_counts_in_top_50_genes`,
  var-side `total_counts`/`mean_counts`/…); `pct_counts_mt` matches scanpy (max|Δ| < 1e-3).
- `gene_total_counts→total_counts` rename applied **identically on both the in-core and streaming QC paths**.

## Step 3 — fp64 moments AT SCALE (fix `5d163d5`, the point of the fix) — 3/3 PASS
Synthetic 1.2M cells × 200 genes, 11.7M nnz, lognorm-like values:
- Shipped fp64 `gene_moments` var vs an **independent robust fp64 two-pass reference**: max rel-err **1.47e-12**
  (mean: **0**).
- The reconstructed **pre-fix fp32 scatter + naive-var** path vs the same reference: max rel-err **4.19e-4**
  — i.e. the fix removes a ~4e-4 relative error in per-gene variance (**~2.8e8× tighter**). This error feeds
  dispersion ranking → HVG selection, invisible at the 2.7k PBMC the suite tests. (Note: here it stems from
  fp32 accumulation order + naive cancellation across ~12M scatter-adds, not hard 2²⁴ overflow — the error
  grows further with higher counts / more cells.)

## Step 4 — BLOCKER kNN top-k for n_neighbors>32 (fix `c1370d9`) — 3/3 PASS
`_knn_gpu` vs brute-force reference for **k = 15, 33, 50**: set-recall 0.998 / 0.998 / 0.998, correct shapes.
k=15 uses the fast 32-slot kernel; k=33/50 take the `mx.argpartition` fallback (both correct). Regression-
guarded in `tests/test_postfix_fixes.py::test_knn_topk_all_k_match_bruteforce` so the OOB write can't return
silently.

## Step 5 — disputes + the Leiden n_iterations decision (fixes `0256f5e`, `ae2a42e`)
- **`pca(zero_center=False)`** (disputed BLOCKER→MAJOR): dense path decomposes **uncentered** (subspace vs
  uncentered SVD ≥ 0.99, vs centered < 0.95); sparse-CSR path **raises** the documented error; default
  `zero_center=True` unchanged. PASS.
- **backed-`.X` rejection** (`ae2a42e`): `filter_cells, regress_out, normalize_pearson_residuals, scrublet,
  rank_genes_groups, score_genes` **all raise a clear `NotImplementedError`** on backed `.X`. PASS (6/6).
- **Leiden `n_iterations`** (`0256f5e`) — measured on the real **986,434-vertex / 22.8M-edge** graph, best-of-3:

  | setting | wall | Q | clusters | note |
  |---|---:|---:|---:|---|
  | **n_iter=1** | **2.73 s** | **0.8504** | **32** | reproduces the banked 2.83 s / 0.8504 / 32 exactly |
  | n_iter=2 | 5.00 s | 0.8586 | 32 | 1.83× slower, ΔQ +0.008 (= igraph parity, 0.8588) |

  **The banked benchmark numbers are NOT invalidated.** The benchmark calls the low-level
  `graph.leiden.leiden` (default `n_iterations=1`), which the fix did **not** change — only the scanpy-facing
  `cluster.leiden`/`tl.leiden` wrapper default changed (was silently clamped to 1, now honors the caller,
  default **2** for scanpy parity). Confirmed: `cluster.leiden` default = 2, `graph.leiden.leiden` default = 1.
  → **Product decision (made by the user):** keep the user-facing default at **n_iter=2** (scanpy parity,
  igraph-parity quality) — no code change; the low-level `graph.leiden.leiden` stays at `n_iter=1`. The
  banked benchmark numbers are unaffected either way, because the benchmark calls the low-level function.
  **Recommendation (not yet decided):** the paper reports **both operating points** (the Pareto frontier):
  quality-matched n_iter=2 = ~2.6× at Q 0.8586 (= igraph 0.8588), and n_iter=1 = **4.77× at Q 0.8504**
  (~1% under igraph). The banked benchmark stands as an n_iter=1 figure.

## Step 6 — MINOR edge cases + packaging (fixes `d61c5ee`, `e54f002`) — 3/3 PASS
- `bbknn` with a 2-cell batch (< `neighbors_within_batch=3`): no argpartition OOB, valid graph.
- `umap` with 5 coincident (duplicate) points: **no NaN** in the embedding.
- `experimental/` move: `metasinglecell.graph.louvain_fused_raw` / `louvain_hybrid` are **no longer importable**
  as package submodules (excluded from the wheel).

## Nothing found wrong or incomplete
No fix failed re-validation; no silent fixes were made in this pass. The only action item is the Step-5
product decision on the user-facing Leiden default + paper framing.
