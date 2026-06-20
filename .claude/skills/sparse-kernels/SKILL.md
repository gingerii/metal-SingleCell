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

## Validation pattern (`metasinglecell.validation`)
- `load_snapshot(name)` reads `data/processed/reference/<name>.npy` (oracle ground truth).
- `compare(name, got, expected, rtol, atol)` → record with max_abs/rel err, rmse, pearson_r,
  exact_match, allclose-pass. `write_report(records, analysis)` → `results/<analysis>/parity_report.csv`.
- Drivers live in `validation_notebooks/NN_<topic>_parity.py`; rebuild the exact matrix the oracle
  used (e.g. from snapshot `00_counts`) so the comparison is closed-loop.

## Next sparse primitives (priority order, all validatable vs oracle)
1. ~~`normalize_total` + `log1p`~~ — DONE (fp32-exact).
2. Per-gene reductions (mean/variance) for HVG — needs column scatter/atomics (vs `04_*`).
3. SpMM for PCA input; SVD core stays on Accelerate/LAPACK (fp64).
