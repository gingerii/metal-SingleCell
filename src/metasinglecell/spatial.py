"""squidpy-GPU ``gr`` spatial functions on Metal/MLX.

``spatial_autocorr`` (Moran's I / Geary's C) is the first: the spatial-weights
product W·X is computed as a scatter-add SpMM over the edge list (MLX has no GPU
sparse matmul), so per-gene autocorrelation and its permutation null both run on
the GPU. Highly relevant to Xenium spatial analysis.
"""

from __future__ import annotations

import numpy as np


def spatial_neighbors(coords, n_neighs: int = 6):
    """Spatial connectivity graph from coordinates (squidpy ``gr.spatial_neighbors``).

    KNN on the spatial coordinates (GPU brute-force), symmetrized, returned as a
    scipy CSR adjacency (binary weights).
    """
    import scipy.sparse as sp

    from .neighbors import _knn_gpu

    knn_idx, _ = _knn_gpu(np.asarray(coords, dtype=np.float32), n_neighs + 1)
    n = coords.shape[0]
    rows = np.repeat(np.arange(n), n_neighs)
    cols = knn_idx[:, 1:].ravel()                      # exclude self
    A = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)             # symmetric, binary
    return A.tocsr()


def _spmm_scatter(src, dst, w, Xc):
    """W·X as scatter-add over edges: out[i] += w_ij · X[j]  (on the GPU)."""
    import mlx.core as mx

    return mx.zeros_like(Xc).at[src].add(w[:, None] * Xc[dst])


def spatial_autocorr(X, connectivity, mode: str = "moran", n_perms: int = 100,
                     random_state: int = 0) -> dict:
    """Per-gene spatial autocorrelation (squidpy ``gr.spatial_autocorr``).

    ``X`` is (cells × genes); ``connectivity`` a symmetric scipy sparse spatial
    graph. ``mode`` is ``"moran"`` (Moran's I) or ``"geary"`` (Geary's C). Returns
    the statistic per gene and a one-sided permutation p-value (shuffling cells).
    """
    import mlx.core as mx
    import scipy.sparse as sp

    coo = sp.coo_matrix(connectivity)
    src = mx.array(coo.row.astype(np.int32))
    dst = mx.array(coo.col.astype(np.int32))
    w = mx.array(coo.data.astype(np.float32))
    W_sum = float(coo.data.sum())
    n = X.shape[0]
    deg = mx.array(np.asarray(connectivity.sum(axis=1)).ravel().astype(np.float32))

    Xg = mx.array(np.asarray(X, dtype=np.float32))

    def stat(Xmat):
        mean = mx.mean(Xmat, axis=0, keepdims=True)
        Xc = Xmat - mean
        denom = mx.sum(Xc * Xc, axis=0)
        if mode == "moran":
            num = mx.sum(Xc * _spmm_scatter(src, dst, w, Xc), axis=0)
            return (n / W_sum) * num / denom
        # Geary's C: sum_ij w_ij (x_i - x_j)^2 = 2(sum_i d_i x_i^2 - sum_i x_i (Wx)_i)
        num = 2.0 * (mx.sum(deg[:, None] * Xmat * Xmat, axis=0)
                     - mx.sum(Xmat * _spmm_scatter(src, dst, w, Xmat), axis=0))
        return (n - 1) / (2.0 * W_sum) * num / denom

    obs = stat(Xg)
    mx.eval(obs)
    obs_np = np.asarray(obs)

    # permutation null: shuffle cells, recompute. Moran high => clustered;
    # Geary low => clustered, so the one-sided tail flips by mode.
    rng = np.random.default_rng(random_state)
    Xnp = np.asarray(X, dtype=np.float32)
    count = np.zeros_like(obs_np)
    for _ in range(n_perms):
        perm = rng.permutation(n)
        s = np.asarray(stat(mx.array(Xnp[perm])))
        count += (s >= obs_np) if mode == "moran" else (s <= obs_np)
    pval = (count + 1.0) / (n_perms + 1.0)

    return {mode: obs_np, "pval": pval}
