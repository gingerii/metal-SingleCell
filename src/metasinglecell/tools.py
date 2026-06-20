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
