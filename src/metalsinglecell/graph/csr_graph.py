"""CSR graph container backed by MLX arrays (on the Metal device).

Undirected, weighted, symmetric (each edge stored in both directions) — the form
produced by ``neighbors`` connectivities. Carries an explicit per-edge source
array so edges can be processed as a flat list for sort-based reductions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    indptr: "object"    # mlx uint32, len n+1
    indices: "object"   # mlx int32, len E  (destination per edge)
    weights: "object"   # mlx float32, len E
    edge_src: "object"  # mlx int32, len E  (source per edge)
    n: int

    @classmethod
    def from_scipy(cls, mat) -> "Graph":
        import mlx.core as mx
        import scipy.sparse as sp

        csr = sp.csr_matrix(mat)
        csr.sort_indices()
        n = csr.shape[0]
        edge_src = np.repeat(np.arange(n, dtype=np.int32), np.diff(csr.indptr))
        return cls(
            indptr=mx.array(csr.indptr.astype(np.uint32)),
            indices=mx.array(csr.indices.astype(np.int32)),
            weights=mx.array(csr.data.astype(np.float32)),
            edge_src=mx.array(edge_src),
            n=n,
        )

    @classmethod
    def from_coo(cls, src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int) -> "Graph":
        """Build from a (possibly redundant) edge list; duplicates are summed."""
        import scipy.sparse as sp

        m = sp.csr_matrix((np.asarray(w), (np.asarray(src), np.asarray(dst))), shape=(n, n))
        m.sum_duplicates()
        return cls.from_scipy(m)

    @property
    def n_edges(self) -> int:
        return int(self.weights.size)

    def degrees(self):
        """Weighted degree per vertex (row sums), as an MLX array."""
        import mlx.core as mx

        return mx.zeros((self.n,), dtype=mx.float32).at[self.edge_src].add(self.weights)

    def total_weight(self) -> float:
        """Sum of all (directed) edge weights == 2m for an undirected graph."""
        import mlx.core as mx

        return float(mx.sum(self.weights).item())
