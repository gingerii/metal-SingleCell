"""scanpy/rapids-singlecell ``tl`` tools on Metal/MLX.

GPU-accelerated where it pays: k-means (Lloyd via matmul), gene-set scoring
(subset row-means), embedding density (KDE). Mirrors scanpy signatures/defaults.
"""

from __future__ import annotations

import numpy as np


def kmeans(X, n_clusters: int = 8, max_iter: int = 300, tol: float = 1e-4,
           random_state: int = 0) -> np.ndarray:
    """Lloyd's k-means on the GPU (MLX). Returns integer labels per row.

    Distances use ``|x|^2 + |c|^2 - 2 x·c^T`` (one matmul/iter); k-means++ init.
    """
    import mlx.core as mx

    Xg = mx.array(np.asarray(X, dtype=np.float32))
    n, d = Xg.shape
    rng = np.random.default_rng(random_state)

    # k-means++ initialization (host picks seeds; cheap).
    Xnp = np.asarray(X, dtype=np.float32)
    centers = [int(rng.integers(n))]
    d2 = np.sum((Xnp - Xnp[centers[0]]) ** 2, axis=1)
    for _ in range(1, n_clusters):
        probs = d2 / d2.sum()
        nxt = int(rng.choice(n, p=probs))
        centers.append(nxt)
        d2 = np.minimum(d2, np.sum((Xnp - Xnp[nxt]) ** 2, axis=1))
    C = mx.array(Xnp[centers])

    labels = mx.zeros((n,), dtype=mx.int32)
    for _ in range(max_iter):
        # assign: argmin over centers of squared distance
        xsq = mx.sum(Xg * Xg, axis=1, keepdims=True)
        csq = mx.sum(C * C, axis=1)
        dist = xsq + csq[None, :] - 2.0 * (Xg @ C.T)
        new_labels = mx.argmin(dist, axis=1).astype(mx.int32)
        # update: mean of assigned points (one-hot matmul)
        onehot = (new_labels[:, None] == mx.arange(n_clusters)[None, :]).astype(mx.float32)
        counts = mx.sum(onehot, axis=0, keepdims=True).T
        newC = (onehot.T @ Xg) / mx.maximum(counts, 1.0)
        shift = mx.sum((newC - C) ** 2).item()
        C = newC
        mx.eval(C, new_labels)
        moved = mx.any(new_labels != labels).item()
        labels = new_labels
        if not moved or shift < tol:
            break
    return np.asarray(labels)


def _subset_row_mean(X, cols: np.ndarray) -> np.ndarray:
    """Per-row mean over a subset of columns (cells × selected genes)."""
    import mlx.core as mx

    sub = np.asarray(X[:, cols].todense() if hasattr(X[:, cols], "todense")
                     else X[:, cols], dtype=np.float32)
    return np.asarray(mx.mean(mx.array(sub), axis=1))


def score_genes(X, gene_list, var_names, ctrl_size: int = 50, n_bins: int = 25,
                random_state: int = 0) -> np.ndarray:
    """Gene-set score per cell (scanpy ``tl.score_genes``).

    Mean expression of ``gene_list`` minus the mean of a control set matched on
    average expression (binned). ``X`` should be log-normalized.
    """
    import scipy.sparse as sp

    var_names = np.asarray(var_names).astype(str)
    name_to_idx = {g: i for i, g in enumerate(var_names)}
    gene_idx = np.array([name_to_idx[g] for g in gene_list if g in name_to_idx])

    Xc = sp.csc_matrix(X)
    gene_mean = np.asarray(Xc.mean(axis=0)).ravel()         # per-gene avg expression
    # bin genes by ranked mean (scanpy: cut the ranked means into n_bins)
    order = np.argsort(gene_mean)
    ranks = np.empty_like(order); ranks[order] = np.arange(len(order))
    bins = (ranks / len(order) * n_bins).astype(int).clip(0, n_bins - 1)

    rng = np.random.default_rng(random_state)
    control = set()
    for gi in gene_idx:
        b = bins[gi]
        pool = np.flatnonzero(bins == b)
        take = min(ctrl_size, pool.size)
        control.update(rng.choice(pool, take, replace=False).tolist())
    control = np.array(sorted(control - set(gene_idx.tolist())))

    return _subset_row_mean(X, gene_idx) - _subset_row_mean(X, control)


def score_genes_cell_cycle(X, s_genes, g2m_genes, var_names, **kwargs) -> dict:
    """S/G2M scores + phase call per cell (scanpy ``tl.score_genes_cell_cycle``)."""
    s = score_genes(X, s_genes, var_names, **kwargs)
    g2m = score_genes(X, g2m_genes, var_names, **kwargs)
    phase = np.where((s < 0) & (g2m < 0), "G1",
                     np.where(s >= g2m, "S", "G2M"))
    return {"S_score": s, "G2M_score": g2m, "phase": phase}


def rank_genes_groups(X, groups, var_names=None, method: str = "t-test",
                      reference: str = "rest") -> dict:
    """Rank marker genes per group (scanpy ``tl.rank_genes_groups``).

    Welch's t-test of each group vs the rest, per gene. Group-wise per-gene
    mean/variance are computed on the GPU by scatter-add over the group label.
    Returns, per group, gene names/indices sorted by descending score, with
    scores (t-statistic) and two-sided p-values. ``X`` is log-normalized.
    """
    import mlx.core as mx
    from scipy import stats

    if method != "t-test":
        raise NotImplementedError("only method='t-test' implemented so far")

    Xg = mx.array(np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32))
    n, n_genes = Xg.shape
    groups = np.asarray(groups)
    cats = np.unique(groups)
    code = np.searchsorted(cats, groups).astype(np.int32)
    G = len(cats)

    gsum = mx.zeros((G, n_genes), dtype=mx.float32).at[mx.array(code)].add(Xg)
    gsq = mx.zeros((G, n_genes), dtype=mx.float32).at[mx.array(code)].add(Xg * Xg)
    ng = np.bincount(code, minlength=G).astype(np.float64)
    tot_sum = np.asarray(mx.sum(Xg, axis=0)); tot_sq = np.asarray(mx.sum(Xg * Xg, axis=0))
    gsum = np.asarray(gsum, dtype=np.float64); gsq = np.asarray(gsq, dtype=np.float64)
    var_names = np.arange(n_genes) if var_names is None else np.asarray(var_names)

    out = {}
    for gi, cat in enumerate(cats):
        n_g = ng[gi]; n_r = n - n_g
        mean_g = gsum[gi] / n_g
        var_g = (gsq[gi] / n_g - mean_g ** 2) * n_g / max(n_g - 1, 1)
        mean_r = (tot_sum - gsum[gi]) / n_r
        var_r = ((tot_sq - gsq[gi]) / n_r - mean_r ** 2) * n_r / max(n_r - 1, 1)
        se = np.sqrt(var_g / n_g + var_r / n_r) + 1e-12
        t = (mean_g - mean_r) / se
        # Welch–Satterthwaite df
        df = se ** 4 / ((var_g / n_g) ** 2 / max(n_g - 1, 1) + (var_r / n_r) ** 2 / max(n_r - 1, 1) + 1e-30)
        pval = 2 * stats.t.sf(np.abs(t), df)
        order = np.argsort(-t)
        out[str(cat)] = {
            "names": np.asarray(var_names)[order],
            "scores": t[order],
            "pvals": pval[order],
            "logfoldchanges": (mean_g - mean_r)[order],
        }
    return out


def _perplexity_affinities(D2, perplexity, tol=1e-5, max_iter=50):
    """Row affinities P_{j|i} calibrated to the target perplexity (host binary search)."""
    n = D2.shape[0]
    P = np.zeros((n, n))
    logU = np.log(perplexity)
    for i in range(n):
        lo, hi, beta = -np.inf, np.inf, 1.0
        Di = np.delete(D2[i], i)
        for _ in range(max_iter):
            Pi = np.exp(-Di * beta)
            sumP = Pi.sum() + 1e-12
            H = np.log(sumP) + beta * (Di * Pi).sum() / sumP
            diff = H - logU
            if abs(diff) < tol:
                break
            if diff > 0:
                lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta; beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        Pi = np.exp(-Di * beta); Pi /= Pi.sum() + 1e-12
        P[i, np.arange(n) != i] = Pi
    return P


def tsne(X, n_components: int = 2, perplexity: float = 30.0, n_iter: int = 1000,
         learning_rate: float = 200.0, early_exaggeration: float = 12.0,
         random_state: int = 0) -> np.ndarray:
    """Exact t-SNE on the GPU (scanpy/rapids ``tl.tsne``).

    Perplexity-calibrated high-dim affinities (host binary search) then KL-minimizing
    gradient descent in low-dim, with the O(n²) gradient as MLX matmuls. Exact (not
    Barnes-Hut), so suited to moderate n; subsample very large datasets.
    """
    import mlx.core as mx

    Xn = np.asarray(X, dtype=np.float64)
    n = Xn.shape[0]
    sq = np.sum(Xn * Xn, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Xn @ Xn.T, 0.0)
    P = _perplexity_affinities(D2, perplexity)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    rng = np.random.default_rng(random_state)
    Y = mx.array((1e-4 * rng.standard_normal((n, n_components))).astype(np.float32))
    Pg = mx.array((P * early_exaggeration).astype(np.float32))
    vel = mx.zeros_like(Y)
    eye = mx.array(np.eye(n, dtype=np.float32))

    for it in range(n_iter):
        if it == 250:
            Pg = mx.array(P.astype(np.float32))               # stop early exaggeration
        ysq = mx.sum(Y * Y, axis=1)
        d2 = mx.maximum(ysq[:, None] + ysq[None, :] - 2.0 * (Y @ Y.T), 0.0)
        num = 1.0 / (1.0 + d2)
        num = num * (1.0 - eye)                               # zero diagonal
        Q = mx.maximum(num / mx.sum(num), 1e-12)
        L = (Pg - Q) * num
        grad = 4.0 * ((mx.sum(L, axis=1)[:, None] * Y) - (L @ Y))
        momentum = 0.5 if it < 250 else 0.8
        vel = momentum * vel - learning_rate * grad
        Y = Y + vel
        Y = Y - mx.mean(Y, axis=0)
        mx.eval(Y)
    return np.asarray(Y)


def draw_graph(connectivities, n_iter: int = 500, random_state: int = 0) -> np.ndarray:
    """Force-directed 2-D graph layout (scanpy ``tl.draw_graph``, ForceAtlas2-style).

    Attractive forces along graph edges + repulsive forces to random negative
    samples, optimized by SGD on the GPU (MLX), like our UMAP layout. Returns a
    (cells × 2) embedding.
    """
    import mlx.core as mx
    import scipy.sparse as sp

    coo = sp.coo_matrix(connectivities).tocoo()
    n = connectivities.shape[0]
    rng = np.random.default_rng(random_state)
    pos = mx.array(rng.normal(scale=1.0, size=(n, 2)).astype(np.float32))
    head = mx.array(coo.row.astype(np.int32))
    tail = mx.array(coo.col.astype(np.int32))
    w = mx.array(coo.data.astype(np.float32))
    n_edges = coo.row.size

    alpha0 = 1.0
    for it in range(n_iter):
        alpha = alpha0 * (1.0 - it / n_iter)
        # attractive: pull connected nodes together (FA2: F ∝ weight·dist)
        diff = pos[head] - pos[tail]
        grad = mx.clip(w[:, None] * diff, -4.0, 4.0) * alpha
        pos = pos.at[head].add(-grad)
        pos = pos.at[tail].add(grad)
        # repulsive: push apart random pairs (F ∝ 1/dist)
        ridx = mx.array(rng.integers(0, n, n_edges).astype(np.int32))
        rdiff = pos[head] - pos[ridx]
        d2 = mx.sum(rdiff * rdiff, axis=1, keepdims=True) + 1e-3
        rgrad = mx.clip(rdiff / d2, -4.0, 4.0) * alpha
        pos = pos.at[head].add(rgrad)
        mx.eval(pos)
    return np.asarray(pos)


def diffmap(connectivities, n_comps: int = 15) -> dict:
    """Diffusion map from a neighbor connectivity graph (scanpy ``tl.diffmap``).

    Eigendecompose the symmetric normalized transition matrix
    ``T = D^{-1/2} W D^{-1/2}`` (ARPACK, fp64 — sparse, no dense GPU eig), then map
    back to the diffusion components ``D^{-1/2} · eigvecs``. Returns eigenvalues
    (descending) and ``X_diffmap``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    W = sp.csr_matrix(connectivities).astype(np.float64)
    d = np.asarray(W.sum(axis=1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(d + 1e-12))
    T = dinv @ W @ dinv
    vals, vecs = eigsh(T, k=min(n_comps, W.shape[0] - 1), which="LM")
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    x_diffmap = np.asarray(dinv @ vecs)
    return {"eigenvalues": vals, "X_diffmap": x_diffmap}


def embedding_density(embedding, groups=None) -> np.ndarray:
    """Per-cell gaussian-KDE density in an embedding (scanpy ``tl.embedding_density``).

    Returns density scaled to [0, 1] within each group (or globally if ``groups``
    is None). Uses scipy's gaussian_kde (low-dim embedding, cheap).
    """
    from scipy.stats import gaussian_kde

    emb = np.asarray(embedding, dtype=np.float64)
    dens = np.zeros(emb.shape[0])
    if groups is None:
        groups = np.zeros(emb.shape[0], dtype=int)
    for g in np.unique(groups):
        m = groups == g
        kde = gaussian_kde(emb[m].T)
        d = kde(emb[m].T)
        d = (d - d.min()) / (d.max() - d.min() + 1e-12)
        dens[m] = d
    return dens
