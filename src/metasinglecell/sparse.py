"""Sparse primitives on Apple Silicon (Metal via MLX).

Stage 1 of the project: the GPU sparse substrate that the scanpy front-end needs.
This module holds a minimal CSR container backed by MLX arrays and hand-written
Metal kernels for the reductions Apple's frameworks don't provide on the GPU.

First kernel: per-cell QC reductions (``total_counts`` and ``n_genes_by_counts``),
i.e. per-row sum and per-row nonzero-count over a (cells x genes) CSR matrix —
a segmented reduction keyed by the row pointer.

Design notes
------------
* Apple GPUs are fp32-only; counts are small integers so fp32 row-sums are exact
  for realistic magnitudes (per-cell totals are thousands; fp32 is exact to 2^24).
* mlx is imported lazily so the package stays importable without a Metal backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid importing scipy/mlx at module import time
    import scipy.sparse as _sp


# One thread per row. Each thread walks its CSR row [indptr[r], indptr[r+1]) and
# accumulates the sum and the nonzero count in a single pass.
_QC_KERNEL_SOURCE = """
    uint row = thread_position_in_grid.x;
    uint start = indptr[row];
    uint end = indptr[row + 1];
    float s = 0.0f;
    uint nz = 0u;
    for (uint j = start; j < end; ++j) {
        float v = data[j];
        s += v;
        if (v > 0.0f) { nz += 1u; }
    }
    row_sum[row] = s;
    row_nnz[row] = (int)nz;
"""


def _qc_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="csr_qc_rowwise",
        input_names=["data", "indptr"],
        output_names=["row_sum", "row_nnz"],
        source=_QC_KERNEL_SOURCE,
    )


# Per-row normalization: scale every entry of row r by target_sum / row_total[r]
# (scanpy normalize_total, exclude_highly_expressed=False). One thread per row;
# first pass sums the row, second pass writes the scaled values.
_NORMALIZE_KERNEL_SOURCE = """
    uint row = thread_position_in_grid.x;
    uint start = indptr[row];
    uint end = indptr[row + 1];
    float total = 0.0f;
    for (uint j = start; j < end; ++j) { total += data[j]; }
    float scale = (total > 0.0f) ? (tsum[0] / total) : 0.0f;
    for (uint j = start; j < end; ++j) { out[j] = data[j] * scale; }
"""

# Elementwise log1p over the nnz values (natural log, matching scanpy log1p).
_LOG1P_KERNEL_SOURCE = """
    uint i = thread_position_in_grid.x;
    out[i] = log(1.0f + data[i]);
"""


def _normalize_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="csr_normalize_total",
        input_names=["data", "indptr", "tsum"],
        output_names=["out"],
        source=_NORMALIZE_KERNEL_SOURCE,
    )


def _log1p_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="csr_log1p",
        input_names=["data"],
        output_names=["out"],
        source=_LOG1P_KERNEL_SOURCE,
    )


# Per-gene (per-column) mean and variance of expm1(data) — the reduction scanpy's
# highly_variable_genes (seurat flavor) needs. Computed from a CSC layout so each
# thread owns one gene's contiguous column slice (no atomics). Two-pass (mean,
# then squared deviations) for fp32 stability; implicit zeros are folded in via
# (n_cells - nnz_col) * mean^2. ddof=1 to match scanpy's R (unbiased) convention.
# MSL has no expm1, so expm1(x) is computed as exp(x)-1 (lognorm values are not
# near zero here, so cancellation is not a concern).
_GENE_MOMENTS_KERNEL_SOURCE = """
    uint g = thread_position_in_grid.x;
    uint start = colptr[g];
    uint end = colptr[g + 1];
    uint n = ncells[0];
    float s = 0.0f;
    for (uint j = start; j < end; ++j) { s += exp(data[j]) - 1.0f; }
    float mean = s / (float)n;
    float ss = 0.0f;
    for (uint j = start; j < end; ++j) { float d = (exp(data[j]) - 1.0f) - mean; ss += d * d; }
    uint nnz_col = end - start;
    ss += (float)(n - nnz_col) * mean * mean;
    gene_mean[g] = mean;
    gene_var[g] = (n > 1u) ? (ss / (float)(n - 1u)) : 0.0f;
"""


def _gene_moments_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="csc_gene_moments",
        input_names=["data", "colptr", "ncells"],
        output_names=["gene_mean", "gene_var"],
        source=_GENE_MOMENTS_KERNEL_SOURCE,
    )


@dataclass
class CSR:
    """Compressed-sparse-row matrix backed by MLX arrays (on the Metal device).

    Attributes mirror scipy.sparse.csr_matrix: ``data`` (float32), ``indices``
    (int32 column indices), ``indptr`` (uint32 row pointers, length n_rows+1).
    """

    data: "object"      # mlx.core.array, float32
    indices: "object"   # mlx.core.array, int32
    indptr: "object"    # mlx.core.array, uint32
    shape: tuple[int, int]

    @classmethod
    def from_scipy(cls, mat: "_sp.spmatrix") -> "CSR":
        import mlx.core as mx
        import scipy.sparse as sp

        csr = sp.csr_matrix(mat)
        csr.sort_indices()
        return cls(
            data=mx.array(csr.data.astype(np.float32)),
            indices=mx.array(csr.indices.astype(np.int32)),
            indptr=mx.array(csr.indptr.astype(np.uint32)),
            shape=tuple(csr.shape),
        )

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def _with_data(self, new_data) -> "CSR":
        """Return a copy sharing indices/indptr/shape but with new values."""
        return CSR(data=new_data, indices=self.indices, indptr=self.indptr, shape=self.shape)

    def toarray(self) -> np.ndarray:
        """Densify to a numpy array (host-side; for validation/inspection only)."""
        import scipy.sparse as sp

        csr = sp.csr_matrix(
            (np.asarray(self.data), np.asarray(self.indices), np.asarray(self.indptr)),
            shape=self.shape,
        )
        return csr.toarray()

    def normalize_total(self, target_sum: float = 1e4) -> "CSR":
        """Scale each cell's counts to sum to ``target_sum`` (scanpy normalize_total).

        Returns a new CSR with the same sparsity pattern and rescaled values,
        computed on the GPU.
        """
        import mlx.core as mx

        kernel = _normalize_kernel()
        n = self.n_rows
        (out,) = kernel(
            inputs=[self.data, self.indptr, mx.array([target_sum], dtype=mx.float32)],
            grid=(n, 1, 1),
            threadgroup=(min(256, n), 1, 1),
            output_shapes=[(self.nnz,)],
            output_dtypes=[mx.float32],
        )
        mx.eval(out)
        return self._with_data(out)

    def log1p(self) -> "CSR":
        """Elementwise ``log(1 + x)`` over the nonzero values (scanpy log1p)."""
        import mlx.core as mx

        kernel = _log1p_kernel()
        nnz = self.nnz
        (out,) = kernel(
            inputs=[self.data],
            grid=(nnz, 1, 1),
            threadgroup=(min(256, nnz), 1, 1),
            output_shapes=[(nnz,)],
            output_dtypes=[mx.float32],
        )
        mx.eval(out)
        return self._with_data(out)

    def gene_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-gene mean and (ddof=1) variance of ``expm1(data)`` over all cells.

        This is the heavy reduction behind ``highly_variable_genes`` (seurat
        flavor): scanpy undoes the log (``expm1``) then takes per-gene mean/var
        including implicit zeros. Runs on the GPU via a CSC column reduction.
        Returns ``(gene_mean, gene_var)`` as numpy arrays of length n_genes.
        """
        import mlx.core as mx
        import scipy.sparse as sp

        csr = sp.csr_matrix(
            (np.asarray(self.data), np.asarray(self.indices), np.asarray(self.indptr)),
            shape=self.shape,
        )
        csc = csr.tocsc()
        data = mx.array(csc.data.astype(np.float32))
        colptr = mx.array(csc.indptr.astype(np.uint32))
        n_genes = self.shape[1]
        gene_mean, gene_var = _gene_moments_kernel()(
            inputs=[data, colptr, mx.array([self.shape[0]], dtype=mx.uint32)],
            grid=(n_genes, 1, 1),
            threadgroup=(min(256, n_genes), 1, 1),
            output_shapes=[(n_genes,), (n_genes,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        mx.eval(gene_mean, gene_var)
        return np.asarray(gene_mean), np.asarray(gene_var)

    def qc_metrics(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-row (per-cell) total counts and number of nonzero genes.

        Returns ``(total_counts, n_genes_by_counts)`` as numpy arrays, computed
        on the GPU with the Metal segmented-reduction kernel.
        """
        import mlx.core as mx

        kernel = _qc_kernel()
        n = self.n_rows
        tg = min(256, n)
        row_sum, row_nnz = kernel(
            inputs=[self.data, self.indptr],
            grid=(n, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.int32],
        )
        mx.eval(row_sum, row_nnz)
        return np.asarray(row_sum), np.asarray(row_nnz)
