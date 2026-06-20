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
