"""Nearest-neighbor graph on the PCA embedding (scanpy ``sc.pp.neighbors``).

The expensive part — all-pairs distances + k-NN selection on the 50-dim embedding
— runs on the Metal GPU (MLX): squared Euclidean distances via a single matmul,
then a top-k selection. The fuzzy connectivity graph is built with UMAP's
``fuzzy_simplicial_set`` (the same routine scanpy calls), so graph semantics match.

Brute-force k-NN is O(n^2) in memory/time; it's a genuine GPU win up to tens of
thousands of cells, but for very large cohorts an approximate method (pynndescent)
is asymptotically better. See the benchmark for the crossover.
"""

from __future__ import annotations

import numpy as np


def _knn_gpu(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN (incl. self) via brute-force squared-Euclidean on the GPU.

    Returns ``(knn_indices, knn_dists)`` of shape (n, k), self first.
    """
    import mlx.core as mx

    Xg = mx.array(X.astype(np.float32))
    sq = mx.sum(Xg * Xg, axis=1)
    # D2[i,j] = |x_i|^2 + |x_j|^2 - 2 x_i·x_j  (squared distance, n x n)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Xg @ Xg.T)
    D2 = mx.maximum(D2, 0.0)
    idx = mx.argpartition(D2, kth=k, axis=1)[:, :k]  # k smallest (unordered)
    mx.eval(idx, D2)

    knn_indices = np.asarray(idx)
    n = X.shape[0]
    d2 = np.asarray(D2)[np.arange(n)[:, None], knn_indices]
    order = np.argsort(d2, axis=1)  # sort the k by distance, self (0) first
    knn_indices = np.take_along_axis(knn_indices, order, axis=1)
    knn_dists = np.sqrt(np.take_along_axis(d2, order, axis=1))
    return knn_indices, knn_dists


def bbknn(X_pca: np.ndarray, batch, neighbors_within_batch: int = 3,
          random_state: int = 0):
    """Batch-balanced kNN (rapids-singlecell/scanpy ``pp.bbknn``).

    For each cell, find ``neighbors_within_batch`` nearest neighbors *within each
    batch* (GPU brute-force per batch), then build one fuzzy connectivity graph
    over the combined neighbor set — this balances neighbors across batches so the
    graph mixes them. Returns ``(distances, connectivities)`` scipy CSR matrices.
    """
    import mlx.core as mx
    import scipy.sparse as sp
    from umap.umap_ import fuzzy_simplicial_set

    X = np.asarray(X_pca, dtype=np.float32)
    n = X.shape[0]
    batch = np.asarray(batch)
    cats = np.unique(batch)
    k = neighbors_within_batch

    Xg = mx.array(X)
    xsq = mx.sum(Xg * Xg, axis=1)
    idx_parts, dist_parts = [], []
    for b in cats:
        bidx = np.flatnonzero(batch == b)
        Xb = Xg[mx.array(bidx.astype(np.int32))]
        D2 = mx.maximum(xsq[:, None] + mx.sum(Xb * Xb, axis=1)[None, :]
                        - 2.0 * (Xg @ Xb.T), 0.0)
        kk = min(k, bidx.size)
        loc = mx.argpartition(D2, kth=kk, axis=1)[:, :kk]
        mx.eval(loc, D2)
        loc = np.asarray(loc)
        d = np.take_along_axis(np.asarray(D2), loc, axis=1)
        idx_parts.append(bidx[loc])                      # map local -> global
        dist_parts.append(np.sqrt(np.maximum(d, 0.0)))

    knn_indices = np.concatenate(idx_parts, axis=1)
    knn_dists = np.concatenate(dist_parts, axis=1)
    order = np.argsort(knn_dists, axis=1)                # sort combined neighbors by distance
    knn_indices = np.take_along_axis(knn_indices, order, axis=1)
    knn_dists = np.take_along_axis(knn_dists, order, axis=1)

    n_neighbors = knn_indices.shape[1]
    conn = fuzzy_simplicial_set(
        sp.coo_matrix((n, 1)), n_neighbors, random_state, metric="euclidean",
        knn_indices=knn_indices, knn_dists=knn_dists,
        set_op_mix_ratio=1.0, local_connectivity=1.0)
    if isinstance(conn, tuple):
        conn = conn[0]
    rows = np.repeat(np.arange(n), n_neighbors)
    distances = sp.csr_matrix((knn_dists.ravel(), (rows, knn_indices.ravel())), shape=(n, n))
    return distances, conn.tocsr()


def neighbors(X_pca: np.ndarray, n_neighbors: int = 15, random_state: int = 0):
    """Compute distance + connectivity graphs from a PCA embedding.

    Returns ``(distances, connectivities)`` as scipy CSR matrices, matching
    scanpy's ``obsp['distances']`` (k-1 neighbors/row) and ``obsp['connectivities']``
    (symmetric fuzzy graph).
    """
    import scipy.sparse as sp
    from umap.umap_ import fuzzy_simplicial_set

    n = X_pca.shape[0]
    knn_indices, knn_dists = _knn_gpu(X_pca, n_neighbors)

    # UMAP fuzzy simplicial set — same call scanpy uses for method="umap".
    conn = fuzzy_simplicial_set(
        sp.coo_matrix((n, 1)),
        n_neighbors,
        random_state,
        metric="euclidean",
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    if isinstance(conn, tuple):  # newer umap returns (graph, sigmas, rhos)
        conn = conn[0]

    # Distances graph: k-1 nearest neighbors per row (exclude self), like scanpy.
    rows = np.repeat(np.arange(n), n_neighbors - 1)
    cols = knn_indices[:, 1:].ravel()
    vals = knn_dists[:, 1:].ravel()
    distances = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return distances, conn.tocsr()
