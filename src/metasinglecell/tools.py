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
    xsq = mx.sum(Xg * Xg, axis=1, keepdims=True)          # constant (Xg fixed) — hoist out
    for _ in range(max_iter):
        # assign: argmin over centers of squared distance. (fp16 matmul gives no win here:
        # the n×d @ d×K output is small-K and memory-bound, and an fp32 cast-back would
        # cost what it saved — fp16 only helps the large-output kNN distance, see _knn_gpu.)
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
    from scipy.stats import rankdata

    var_names = np.asarray(var_names).astype(str)
    name_to_idx = {g: i for i, g in enumerate(var_names)}
    gene_idx = np.array([name_to_idx[g] for g in gene_list if g in name_to_idx])

    Xc = sp.csc_matrix(X)
    obs_avg = np.asarray(Xc.mean(axis=0)).ravel()           # per-gene avg expression
    # Match scanpy's binning exactly: integer bins of rank(method='min') // n_items,
    # with n_items = round(n_genes / (n_bins - 1)). Then sample ctrl_size control genes
    # from EACH UNIQUE bin the gene set occupies (not per gene) and union them — the
    # leftover RNG choice is expression-matched within a bin, so the score is ~invariant.
    n = obs_avg.size
    n_items = max(int(np.round(n / (n_bins - 1))), 1)
    obs_cut = rankdata(obs_avg, method="min") // n_items
    rng = np.random.RandomState(random_state)
    gene_set = set(gene_idx.tolist())
    control = set()
    for cut in np.unique(obs_cut[gene_idx]):
        pool = np.flatnonzero(obs_cut == cut)               # all genes in this bin
        if ctrl_size < pool.size:
            pool = rng.choice(pool, ctrl_size, replace=False)
        control.update(pool.tolist())
    control = np.array(sorted(control - gene_set))          # ctrl_as_ref: drop gene set after

    return _subset_row_mean(X, gene_idx) - _subset_row_mean(X, control)


def score_genes_cell_cycle(X, s_genes, g2m_genes, var_names, **kwargs) -> dict:
    """S/G2M scores + phase call per cell (scanpy ``tl.score_genes_cell_cycle``)."""
    s = score_genes(X, s_genes, var_names, **kwargs)
    g2m = score_genes(X, g2m_genes, var_names, **kwargs)
    phase = np.where((s < 0) & (g2m < 0), "G1",
                     np.where(s >= g2m, "S", "G2M"))
    return {"S_score": s, "G2M_score": g2m, "phase": phase}


def rank_genes_groups(X, groups, var_names=None, method: str = "t-test",
                      reference: str = "rest", tie_correct: bool = False, **kwds) -> dict:
    """Rank marker genes per group (scanpy ``tl.rank_genes_groups``), group vs rest.

    ``method`` ∈ {``"t-test"``, ``"t-test_overestim_var"``, ``"wilcoxon"``, ``"logreg"``}.
    Group-wise per-gene mean/variance are computed on the GPU by scatter-add over the
    group label; the per-method statistic matches scanpy exactly. Returns, per group,
    gene names/indices sorted by descending score, with scores and (where defined)
    two-sided p-values and log-fold-changes. ``X`` is log-normalized.
    """
    import mlx.core as mx
    from scipy import stats

    valid = {"t-test", "t-test_overestim_var", "wilcoxon", "logreg"}
    if method not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}")

    Xd = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    n, n_genes = Xd.shape
    groups = np.asarray(groups)
    cats = np.unique(groups)
    code = np.searchsorted(cats, groups).astype(np.int32)
    G = len(cats)
    var_names = np.arange(n_genes) if var_names is None else np.asarray(var_names)
    out = {}

    # ---- logistic regression: coefficients are the per-group scores (no p-values) ----
    if method == "logreg":
        from sklearn.linear_model import LogisticRegression

        if G < 2:
            raise ValueError("Cannot perform logistic regression on a single cluster.")
        clf = LogisticRegression(**kwds); clf.fit(Xd, code)
        coef = clf.coef_; existing = np.unique(code)            # (n_classes, n_genes)
        for gi, cat in enumerate(cats):
            scores = coef[0] if G <= 2 else coef[np.argmax(existing == gi)]
            order = np.argsort(-scores)
            out[str(cat)] = {"names": var_names[order], "scores": scores[order],
                             "pvals": None, "logfoldchanges": None}
        return out

    # GPU group sums for mean/variance (ddof=1), shared by the t-tests and lfc.
    Xg = mx.array(Xd)
    code_mx = mx.array(code)
    gsum = np.asarray(mx.zeros((G, n_genes), mx.float32).at[code_mx].add(Xg), np.float64)
    gsq = np.asarray(mx.zeros((G, n_genes), mx.float32).at[code_mx].add(Xg * Xg), np.float64)
    ng = np.bincount(code, minlength=G).astype(np.float64)
    tot_sum = np.asarray(mx.sum(Xg, axis=0), np.float64)
    tot_sq = np.asarray(mx.sum(Xg * Xg, axis=0), np.float64)

    # ---- Wilcoxon rank-sum: rank cells per gene, sum group ranks -> normal z ----
    if method == "wilcoxon":
        ranks = stats.rankdata(Xd, axis=0)                      # n × genes, average ties
        rsum = np.zeros((G, n_genes), np.float64)
        np.add.at(rsum, code, ranks)                            # exact fp64 rank sums
        tc = _tiecorrect_cols(ranks) if tie_correct else np.ones(n_genes)
        for gi, cat in enumerate(cats):
            ng_, mg_ = ng[gi], n - ng[gi]
            std = np.sqrt(tc * ng_ * mg_ * (n + 1) / 12.0) + 1e-12
            z = (rsum[gi] - ng_ * (n + 1) / 2.0) / std
            z[np.isnan(z)] = 0.0
            pval = 2 * stats.norm.sf(np.abs(z))
            lfc = gsum[gi] / ng[gi] - (tot_sum - gsum[gi]) / (n - ng[gi])
            order = np.argsort(-z)
            out[str(cat)] = {"names": var_names[order], "scores": z[order],
                             "pvals": pval[order], "logfoldchanges": lfc[order]}
        return out

    # ---- t-test / t-test_overestim_var (Welch via scipy, exact scanpy match) ----
    for gi, cat in enumerate(cats):
        n_g = ng[gi]; n_other = n - n_g
        mean_g = gsum[gi] / n_g
        var_g = (gsq[gi] / n_g - mean_g ** 2) * n_g / max(n_g - 1, 1)
        mean_r = (tot_sum - gsum[gi]) / n_other
        var_r = ((tot_sq - gsq[gi]) / n_other - mean_r ** 2) * n_other / max(n_other - 1, 1)
        n_rest = n_g if method == "t-test_overestim_var" else n_other   # scanpy's var hack
        with np.errstate(invalid="ignore"):
            t, pval = stats.ttest_ind_from_stats(
                mean_g, np.sqrt(var_g), int(n_g), mean_r, np.sqrt(var_r), n_rest,
                equal_var=False)
        t = np.nan_to_num(t, nan=0.0); pval = np.nan_to_num(pval, nan=1.0)
        order = np.argsort(-t)
        out[str(cat)] = {"names": var_names[order], "scores": t[order],
                         "pvals": pval[order], "logfoldchanges": (mean_g - mean_r)[order]}
    return out


def _tiecorrect_cols(ranks: np.ndarray) -> np.ndarray:
    """Per-column tie-correction factor (scipy.stats.tiecorrect, vectorized over genes)."""
    n = ranks.shape[0]
    if n < 2:
        return np.ones(ranks.shape[1])
    out = np.ones(ranks.shape[1])
    rs = np.sort(ranks, axis=0)
    for j in range(ranks.shape[1]):
        _, cnt = np.unique(rs[:, j], return_counts=True)
        out[j] = 1.0 - (cnt ** 3 - cnt).sum() / (n ** 3 - n)
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
         random_state: int = 0, exact_max_n: int = 30_000) -> np.ndarray:
    """t-SNE (scanpy/rapids ``tl.tsne``), scale-dispatched.

    * ``n ≤ exact_max_n``: EXACT t-SNE on the GPU — perplexity-calibrated affinities
      then KL-minimizing gradient descent with the O(n²) gradient as MLX matmuls (fast
      and exact at this scale).
    * larger: the exact O(n²) affinity/Q matrices don't fit, so fall back to sklearn's
      **Barnes-Hut** t-SNE (O(n log n), the scanpy/sklearn default) — same dispatch
      philosophy as the kNN path (GPU where it wins, optimized CPU where O(n²) can't fit).
    """
    import mlx.core as mx

    n = X.shape[0]
    if n > exact_max_n:
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_components, perplexity=perplexity,
                    learning_rate=learning_rate, early_exaggeration=early_exaggeration,
                    init="pca", random_state=random_state).fit_transform(
                        np.asarray(X, dtype=np.float32))

    Xn = np.asarray(X, dtype=np.float64)
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
