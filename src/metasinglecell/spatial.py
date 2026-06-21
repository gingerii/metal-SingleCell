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


def calculate_niche(connectivity, labels, n_niches: int = 10,
                    random_state: int = 0) -> dict:
    """Spatial niches from neighborhood composition (squidpy ``gr.calculate_niche``).

    For each cell, the cell-type composition of its spatial neighborhood (W·onehot,
    row-normalized, via scatter-SpMM on the GPU), then cluster those composition
    vectors (k-means) into ``n_niches`` niches. Returns niche labels + compositions.
    """
    import mlx.core as mx
    import scipy.sparse as sp

    from .tools import kmeans

    labels = np.asarray(labels)
    cats = np.unique(labels)
    code = np.searchsorted(cats, labels).astype(np.int32)
    onehot = mx.array((code[:, None] == np.arange(len(cats))[None, :]).astype(np.float32))

    coo = sp.coo_matrix(connectivity)
    comp = _spmm_scatter(mx.array(coo.row.astype(np.int32)),
                         mx.array(coo.col.astype(np.int32)),
                         mx.array(coo.data.astype(np.float32)), onehot)
    comp = comp / (mx.sum(comp, axis=1, keepdims=True) + 1e-12)
    mx.eval(comp)
    comp = np.asarray(comp)
    niches = kmeans(comp, n_clusters=n_niches, random_state=random_state)
    return {"niche": niches, "composition": comp, "categories": cats}


def ligrec(X, labels, lr_pairs, var_names, n_perms: int = 100,
           random_state: int = 0) -> dict:
    """Ligand-receptor interaction permutation test (squidpy ``gr.ligrec``).

    CellPhoneDB-style: for each L-R pair and ordered cluster pair (A, B), the score
    is the mean of (ligand mean in A, receptor mean in B). A permutation null
    (shuffling cluster labels) gives one-sided p-values. Cluster means are computed
    on the GPU by scatter-add over the label. ``lr_pairs`` is a list of
    (ligand_gene, receptor_gene) symbol pairs.
    """
    import mlx.core as mx

    Xd = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    Xg = mx.array(Xd)
    n, n_genes = Xg.shape
    labels = np.asarray(labels)
    cats = np.unique(labels)
    K = len(cats)
    name_to_idx = {g: i for i, g in enumerate(np.asarray(var_names).astype(str))}
    pairs = [(name_to_idx[l], name_to_idx[r]) for l, r in lr_pairs
             if l in name_to_idx and r in name_to_idx]
    lig = np.array([p[0] for p in pairs]); rec = np.array([p[1] for p in pairs])

    def cluster_means(code):
        gsum = mx.zeros((K, n_genes), dtype=mx.float32).at[mx.array(code)].add(Xg)
        ng = np.bincount(code, minlength=K).astype(np.float32)
        return np.asarray(gsum) / np.maximum(ng[:, None], 1.0)

    def score(means):                                   # (n_pairs, K_A, K_B)
        L = means[:, lig].T                             # n_pairs × K (ligand in A)
        R = means[:, rec].T                             # n_pairs × K (receptor in B)
        return 0.5 * (L[:, :, None] + R[:, None, :])

    code = np.searchsorted(cats, labels).astype(np.int32)
    obs = score(cluster_means(code))

    rng = np.random.default_rng(random_state)
    count = np.zeros_like(obs)
    for _ in range(n_perms):
        count += score(cluster_means(rng.permutation(code))) >= obs
    pval = (count + 1.0) / (n_perms + 1.0)

    return {"means": obs, "pvalues": pval, "categories": cats,
            "lr_pairs": [(lr_pairs[i]) for i in range(len(pairs))]}


def co_occurrence(coords, labels, n_intervals: int = 50, max_dist=None) -> dict:
    """Distance-binned cluster co-occurrence ratio (squidpy ``gr.co_occurrence``).

    For each distance interval, the conditional probability P(type j | type i at
    distance) divided by the unconditional P(type j) — values > 1 mean enrichment.
    Pairwise distances are computed on the GPU; counts per (i, j, bin) via one-hot
    matmuls. O(n²) memory, so suited to moderate n (tile for very large sections).

    Returns ``occ`` of shape (K, K, n_intervals) and the interval edges.
    """
    import mlx.core as mx

    X = mx.array(np.asarray(coords, dtype=np.float32))
    labels = np.asarray(labels)
    cats = np.unique(labels)
    code = np.searchsorted(cats, labels).astype(np.int32)
    K = len(cats)
    n = X.shape[0]

    sq = mx.sum(X * X, axis=1)
    D2 = mx.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    D = mx.sqrt(D2)
    mx.eval(D)
    Dnp = np.asarray(D)
    if max_dist is None:
        max_dist = float(np.percentile(Dnp[Dnp > 0], 95))
    edges = np.linspace(0, max_dist, n_intervals + 1)

    onehot = mx.array((code[:, None] == np.arange(K)[None, :]).astype(np.float32))  # n×K
    p_uncond = np.bincount(code, minlength=K) / n

    occ = np.zeros((K, K, n_intervals))
    for b in range(n_intervals):
        mask = mx.array(((Dnp >= edges[b]) & (Dnp < edges[b + 1])).astype(np.float32))
        # pair counts per (cat_i, cat_j): onehot.T @ mask @ onehot
        cnt = np.asarray(onehot.T @ (mask @ onehot))                # K×K
        row = cnt.sum(axis=1, keepdims=True)
        cond = cnt / np.maximum(row, 1e-12)                         # P(j | i, bin)
        occ[:, :, b] = cond / np.maximum(p_uncond[None, :], 1e-12)
    return {"occ": occ, "interval": edges, "categories": cats}


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
