---
name: sparse-kernels
description: Stage-1 Metal sparse substrate — MLX custom-kernel API, CSR layout convention, QC-reduction kernel (validated exact), and the parity harness pattern. Load when writing/validating GPU sparse kernels.
---

# Sparse kernels (Stage 1) — metal-SingleCell

## MLX custom-kernel API (verified, mlx 0.31.2)
- Default device is `Device(gpu, 0)`; `mx.fast.metal_kernel` works on this M3.
- `mx.fast.metal_kernel(name, input_names, output_names, source)` — `source` is the **kernel body
  only** (MLX wraps the signature). Inputs/outputs are addressed as bare buffers `x[i]`.
- Call: `kernel(inputs=[...], grid=(N,1,1), threadgroup=(min(256,N),1,1), output_shapes=[...],
  output_dtypes=[...])`. `grid` is the **total thread count** (CUDA grid×block), dispatched with
  non-uniform threadgroups, so `grid=(N,1,1)` gives exactly N threads → `thread_position_in_grid.x`
  is always `< N`, no bounds guard needed. Always `mx.eval(...)` the outputs.

## CSR convention (`metasinglecell.sparse.CSR`)
- `data` float32, `indices` int32 (gene/column), `indptr` **uint32** (row pointers, len n_rows+1).
- `CSR.from_scipy` calls `sort_indices()`. Counts are small ints → fp32 row-sums are exact (fp32 is
  exact to 2^24; per-cell totals are ~10^3–10^4).

## QC-metrics kernel (`csr_qc_rowwise`) — VALIDATED EXACT
- One thread per row; single pass over `[indptr[r], indptr[r+1])` accumulating sum + nonzero count.
- Outputs per-cell `total_counts` (float32) and `n_genes_by_counts` (int32).
- Parity vs fp64 oracle (`results/qc_metrics/parity_report.csv`): **exact match, max_abs_err=0,
  r=1.0** on PBMC3k (2700 cells × 13714 genes, nnz=2.28M).

## normalize_total + log1p kernels — VALIDATED (fp32-exact)
- `CSR.normalize_total(target_sum=1e4)` (`csr_normalize_total`): one thread/row, two passes
  (sum, then scale by `target_sum/total`); `exclude_highly_expressed=False`. Scalar passed as a
  1-elem mlx array (`tsum[0]`), not f-string-baked.
- `CSR.log1p()` (`csr_log1p`): elementwise `log(1+x)` over nnz, one thread/nnz.
- Both return a new CSR via `_with_data` (shares indices/indptr). `CSR.toarray()`/`nnz` added for
  validation. Parity vs fp64 oracle (`results/normalize_log1p/`): **pass at fp32**, not bit-exact —
  normalized max_rel_err 1.1e-7, lognorm 1.9e-7, r=1.0. Use rtol≈1e-4 for these (fp32-vs-fp64).

## gene_moments + highly_variable_genes — VALIDATED (fp32, selection exact)
- `CSR.gene_moments()` → per-gene mean & ddof=1 variance of `expm1(data)`. Builds CSC
  (`csr.tocsc()`) so each thread owns one gene's column slice — **no atomics**. Two-pass
  (mean, then squared deviations) for fp32 stability; implicit zeros folded in as
  `(n_cells - nnz_col) * mean^2`.
- **MSL gotcha: no `expm1`** (compile error "use of undeclared identifier"). Use `exp(x) - 1.0f`
  (lognorm values aren't near 0, so no cancellation); likewise `log(1.0f + x)` for log1p.
- Host seurat binning in `preprocess.highly_variable_genes` (float64): expm1→mean/var→
  log-dispersion→`pd.cut` 20 equal-width mean bins→per-bin z-score (ddof=1 std; single-gene bins
  set avg=0/dev=avg per scanpy `_postprocess_dispersions_seurat`)→top-n by dispersions_norm.
- Parity vs fp64 oracle (`results/hvg/`): means max_abs_err 1.9e-6, dispersions_norm 6.1e-6,
  **highly_variable flag EXACT, 2000/2000 gene-selection overlap**, r=1.0. rtol 1e-4 (means)/1e-3 (disp_norm).

## Harness handles non-finite (`metasinglecell.validation`)
- `compare()` masks positions non-finite in either array for error metrics, and only passes if the
  non-finite *masks* match (`nonfinite_masks_match`); `exact_match` uses `equal_nan=True`.

## Validation pattern (`metasinglecell.validation`)
- `load_snapshot(name)` reads `data/processed/reference/<name>.npy` (oracle ground truth).
- `compare(name, got, expected, rtol, atol)` → record with max_abs/rel err, rmse, pearson_r,
  exact_match, allclose-pass. `write_report(records, analysis)` → `results/<analysis>/parity_report.csv`.
- Drivers live in `validation_notebooks/NN_<topic>_parity.py`; rebuild the exact matrix the oracle
  used (e.g. from snapshot `00_counts`) so the comparison is closed-loop.

## scale (sparse→dense boundary) — VALIDATED (fp32)
- `preprocess.scale(csr, max_value=10, zero_center=True)`: densifies (zero-centering forces it),
  per-gene z-score with **ddof=1** var (std==0→1), then clip to `[-max_value, max_value]` (lower
  bound only when zero_center). Pure dense **MLX** ops (mean/var/where/min/max) — no custom kernel;
  this is the densification boundary. Returns dense float32.
- scanpy `clip_array`: upper clip always, lower clip only if zero_center (confirmed from source).
- Parity vs fp64 oracle (`results/scale/`): max_abs_err 2.2e-6, rmse 5.4e-8, r=1.0.

## Front-end status: QC → normalize → log1p → HVG → scale → PCA ALL VALIDATED.

## PCA — `decomposition.pca(X, n_comps, solver=, random_state=)` — 3 solvers, VALIDATED
- Mean-center on GPU (MLX fp32); returns `(X_pca [cells×k] f32, components [k×genes] f32,
  variance_ratio f64)`. Signs fixed by sklearn `svd_flip` (u_based). variance_ratio uses
  `Xc.var(ddof=1).sum()` as total variance (matches sklearn for all solvers). NB oracle
  `06_pca_components` is genes×k → compare `components.T`.
- `solver="full"`: `np.linalg.svd` (LAPACK gesdd, fp64). `solver="arpack"`: `scipy.svds` Lanczos
  (fp64, what the oracle used) — reorder ascending→descending. Both match oracle **exactly**
  (min|r|=1.0, subspace=1.0, vr rel-err 5e-9).
- `solver="randomized"`: range finder (QR-normalized power iters, `n_iter=7`, `n_oversamples=10`) —
  **matmuls `Xc@Q`/`Xcᵀ@Q` on GPU (fp32)**, QR on CPU (MLX QR is CPU-only), then **fp64 SVD of the
  small projected `B=Qᵀ@Xc` on LAPACK**. This is the architecture's showcase: heavy work on Metal,
  stable core on Accelerate.
- **MLX hard limits (confirmed):** fp64 arrays silently downcast to fp32; `mx.linalg.svd` is
  **CPU-stream only** (GPU raises "not yet supported"); `qr`/`eigh` work but pass `stream=mx.cpu`.
  ⇒ the fp64 numerical anchor must be NumPy/SciPy (Accelerate), never MLX.

## FINDING: randomized SVD under-resolves trailing PCs on scaled single-cell data
- Our randomized matches sklearn `randomized_svd` exactly (min|r|=1.0) — implementation is faithful.
- But randomized (default oversampling/iters) reaches only **subspace overlap 0.76 vs the exact top-50**
  on the scaled HVG matrix — its spectrum decays slowly (many comparable singular values), so trailing
  PCs aren't resolved. Top PCs (drive neighbors/clustering) are fine; trailing ones rotate within
  near-degenerate blocks. Validate randomized against sklearn-randomized, NOT against arpack/exact;
  report subspace-vs-exact as a quality metric. To tighten: raise `n_oversamples`/`n_iter`.

## Validation harness additions
- `validation.compare_signed_columns(name, got, expected, min_abs_corr)` — per-column |Pearson r|,
  sign-invariant (for PCA/embeddings). `validation.subspace_overlap(a, b)` — `||QaᵀQb||_F²/k` ∈ [0,1].

## Next primitives
1. neighbors (KNN on the 50-dim embedding) → `07_*`; then leiden/umap (dense, post-PCA).
