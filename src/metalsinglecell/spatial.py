"""squidpy-GPU ``gr`` spatial functions on Metal/MLX.

``spatial_autocorr`` (Moran's I / Geary's C) is the first: the spatial-weights
product W·X is computed as a scatter-add SpMM over the edge list (MLX has no GPU
sparse matmul), so per-gene autocorrelation and its permutation null both run on
the GPU. Highly relevant to Xenium spatial analysis.
"""

from __future__ import annotations

import numpy as np


def spatial_neighbors(coords, n_neighs: int = 6, symmetric: bool = False):
    """Spatial connectivity graph from coordinates (squidpy ``gr.spatial_neighbors``).

    Exact kNN on the spatial coordinates, returned as a scipy CSR adjacency (binary
    weights). By default this matches squidpy's generic-coordinate convention exactly:
    the **directed** k-NN, ``n_neighs`` edges per row, self excluded (squidpy's
    ``spatial_connectivities`` is likewise directed, ``nnz/n = n_neighs``). Pass
    ``symmetric=True`` for the undirected union ``(A + Aᵀ)``.

    NB: this uses ``_knn_grid`` — a uniform-grid (cell-list) spatial index — NOT the
    high-dim ``_knn`` dispatcher's NNDescent path. NNDescent is a high-dimensional-
    embedding method and fails on 2-D spatial coordinates (measured recall ~6% vs
    exact). The grid index is the right O(n) accelerator for low-dim point data: each
    cell holds ~a few·k points, so a query scans only its 3×3 (2-D) / 3×3×3 (3-D) cell
    block. Results are exact (grid completeness check + fp32 brute fallback) and scale
    past the O(n²) brute wall. See SPATIAL_KNN_investigation.md.
    """
    import scipy.sparse as sp

    from .neighbors import _knn_grid

    knn_idx, _ = _knn_grid(np.asarray(coords, dtype=np.float32), n_neighs + 1)
    n = coords.shape[0]
    rows = np.repeat(np.arange(n), n_neighs)
    cols = knn_idx[:, 1:].ravel()                      # exclude self (self is column 0)
    A = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(n, n))
    if symmetric:
        A = ((A + A.T) > 0).astype(np.float32)         # undirected union
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
    # Keep the surviving pair alongside its indices. Labelling the rows positionally instead
    # (lr_pairs[0..n_kept]) silently mislabels every row after the first dropped pair.
    kept = [(l, r) for l, r in lr_pairs if l in name_to_idx and r in name_to_idx]
    pairs = [(name_to_idx[l], name_to_idx[r]) for l, r in kept]
    if not pairs:
        raise ValueError("none of the interaction pairs are present in var_names")
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

    return {"means": obs, "pvalues": pval, "categories": cats, "lr_pairs": kept}


# Fused co-occurrence histogram: one thread per ordered pair (i,j). Each pair finds the
# FIRST distance bin it falls under (binary search on ascending thr2 == numpy searchsorted
# 'left') and atomic-increments hist[code_i, code_j, bin]. A cumulative sum over bins (host)
# then yields the cumulative within-radius counts. Replaces the old 50×(n×n mask + 2 n×K
# matmuls) loop with a single pass + a tiny atomic histogram.
_COOCCUR_KERNEL_SOURCE = """
    uint p = thread_position_in_grid.x;
    uint M = (uint)rows[0];                           // query rows in this tile
    uint N = (uint)n[0];
    if (p >= M * N) return;
    uint i = (uint)roff[0] + p / N;                   // global query index
    uint j = p % N;                                   // global reference index
    if (i == j) return;                               // exclude self-pairs
    float d2 = D2[p];                                 // D2 tile is M×N (row-major)
    uint L = (uint)nthr[0];
    uint lo = 0u, hi = L;                             // first r with thr2[r] >= d2
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (thr2[mid] >= d2) hi = mid; else lo = mid + 1u;
    }
    if (lo >= L) return;                              // beyond the largest threshold
    uint idx = ((uint)code[i] * (uint)K[0] + (uint)code[j]) * L + lo;
    device atomic_uint* h = (device atomic_uint*)hist;
    atomic_fetch_add_explicit(h + idx, 1u, memory_order_relaxed);
"""


def _cooccur_hist(X, sq, code, thr2, K, n, L, tile):
    """Per-bin co-occurrence histogram (K, K, L), TILED over query rows.

    Never materializes the full n×n distance matrix: each row-block computes its
    (rows×n) squared-distance tile via matmul, the fused kernel bins + atomic-counts it,
    and the tiny K×K×L histograms are summed across tiles. Avoids the int32 grid overflow
    of a flat n² grid and the O(n²) memory wall, so it scales to large sections.
    """
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name="cooccur_hist",
        input_names=["D2", "code", "thr2", "rows", "roff", "n", "K", "nthr"],
        output_names=["hist"],
        source=_COOCCUR_KERNEL_SOURCE,
    )
    code_a = mx.array(code.astype(np.int32))
    thr2_a = mx.array(thr2.astype(np.float32))
    K_a, n_a, L_a = (mx.array([v], dtype=mx.int32) for v in (K, n, L))
    total = np.zeros(K * K * L, dtype=np.float64)
    for r0 in range(0, n, tile):
        m = min(tile, n - r0)
        D2t = mx.maximum(sq[r0:r0 + m, None] + sq[None, :] - 2.0 * (X[r0:r0 + m] @ X.T), 0.0)
        (hist,) = kernel(
            inputs=[D2t.reshape(-1), code_a, thr2_a,
                    mx.array([m], dtype=mx.int32), mx.array([r0], dtype=mx.int32),
                    n_a, K_a, L_a],
            grid=(m * n, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(K * K * L,)], output_dtypes=[mx.uint32], init_value=0,
        )
        mx.eval(hist)
        total += np.asarray(hist)
    return total.reshape(K, K, L)


def _cooccur_thresholds(spatial):
    """squidpy's ``_find_min_max`` heuristic for the distance grid.

    Deliberately not the true min/max pairwise distance: squidpy anchors on the two points
    with the smallest coordinate sums, and halves the span to the largest. Using true extremes
    instead puts the grid ~2x wider and starts it ~2000x lower on real data, so the
    co-occurrence curves are sampled at entirely different radii.
    """
    coord_sum = spatial.sum(axis=1)
    lo_a, lo_b = np.argpartition(coord_sum, 2)[:2]
    hi = int(np.argmax(coord_sum))
    thres_max = float(np.linalg.norm(spatial[lo_a] - spatial[hi])) / 2.0
    thres_min = float(np.linalg.norm(spatial[lo_a] - spatial[lo_b]))
    return thres_min, thres_max


def co_occurrence(coords, labels, n_intervals: int = 50, interval=None,
                  max_dist=None) -> dict:
    """Cumulative cluster co-occurrence ratio (squidpy ``gr.co_occurrence``).

    Matches squidpy's definition exactly: for each distance threshold the pairs
    *within* that radius are counted (cumulative, ``d <= threshold``), and the
    enrichment of type i around type c is ``P(i | c, within r) / P(i)``. Distances are
    computed on the GPU and binned by a fused atomic-histogram kernel, TILED over query
    rows so the full n×n distance matrix is never materialized (scales to large sections).

    ``interval`` may be an explicit ascending array of distance thresholds (as
    squidpy stores in ``uns[...]['interval']``); otherwise ``n_intervals`` evenly spaced
    thresholds are built from squidpy's coordinate-sum heuristic. Returns ``occ`` of shape
    (K, K, len(thresholds) - 1) and the thresholds themselves.

    ``n_intervals`` counts *thresholds*, not bins, which is squidpy's convention and the
    reason its ``occ`` has one fewer entry along the last axis than you might expect.
    """
    import mlx.core as mx

    X = mx.array(np.asarray(coords, dtype=np.float32))
    labels = np.asarray(labels)
    cats = np.unique(labels)
    code = np.searchsorted(cats, labels).astype(np.int32)
    K = len(cats)
    n = code.shape[0]
    sq = mx.sum(X * X, axis=1)
    tile = max(1, 60_000_000 // n)                                  # ~60M-elem distance tiles

    if interval is not None:
        thr = np.sort(np.asarray(interval, dtype=np.float64))
    else:
        lo, hi = _cooccur_thresholds(np.asarray(coords, dtype=np.float64))
        if max_dist is not None:
            hi = float(max_dist)
        thr = np.linspace(lo, hi, num=n_intervals)                  # THRESHOLDS, not bins
    if len(thr) <= 1:
        raise ValueError(f"expected interval to be of length >= 2, found {len(thr)}")
    thr2 = thr[1:] ** 2                                             # squidpy: skip first
    L = len(thr2)

    # Fused TILED histogram kernel: one pass over the n² pairs (vs the old 50×(n×n mask + two
    # n×K matmuls)). Each pair finds the FIRST threshold bin it falls under (binary search on
    # ascending thr2) and atomic-increments hist[code_i, code_j, bin]; cumulative sum over bins
    # then gives the cumulative counts squidpy needs. Tiled over rows → scales past the n×n wall.
    hist = _cooccur_hist(X, sq, code, thr2, K, n, L, tile)         # (K, K, L) per-bin counts
    counts = np.cumsum(hist.astype(np.float64), axis=2)            # cumulative within radius

    # squidpy conditional normalization (counts symmetric in the ordered sum)
    row_sums = counts.sum(axis=0)                                   # (K, L)
    totals = row_sums.sum(axis=0)                                   # (L,)
    occ = np.zeros((K, K, L))
    for r in range(L):
        probs = row_sums[:, r] / np.maximum(totals[r], 1e-12)       # marginal P(type)
        for c in range(K):
            if row_sums[c, r] != 0.0:
                occ[:, c, r] = (counts[c, :, r] / row_sums[c, r]) / np.maximum(probs, 1e-12)
    return {"occ": occ, "interval": thr, "categories": cats}


def _spmm_scatter(src, dst, w, Xc):
    """W·X as scatter-add over edges: out[i] += w_ij · X[j]  (on the GPU)."""
    import mlx.core as mx

    return mx.zeros_like(Xc).at[src].add(w[:, None] * Xc[dst])


def row_normalize(g):
    """squidpy's ``transformation=True``: L1-normalise every row of the connectivity.

    Scale-invariant for a regular graph — Moran's I divides by ``S0`` — so it changes nothing
    on a synthetic lattice, and does change the statistic wherever degrees vary (Visium
    boundary spots, Delaunay, k-NN after a percentile prune).
    """
    import scipy.sparse as sp
    g = sp.csr_matrix(g, copy=True).astype(np.float64)
    s = np.asarray(abs(g).sum(axis=1)).ravel()
    inv = np.divide(1.0, s, out=np.zeros_like(s), where=s != 0)
    return sp.diags(inv) @ g


def graph_moments(g):
    """``s0, s1, s2`` — pysal's weight-matrix moments, for the analytic variance."""
    s0 = float(g.sum())
    t = g.transpose() + g
    s1 = float(t.multiply(t).sum()) / 2.0
    row = np.asarray(g.sum(axis=1)).ravel()
    col = np.asarray(g.sum(axis=0)).ravel()
    s2 = float(((row + col) ** 2).sum())
    return s0, s1, s2


def analytic_pval(score, g, mode: str = "moran", two_tailed: bool = False):
    """p-value and variance of the statistic under the normality assumption.

    Moran's I and Geary's C have *different* sampling variances (Cliff & Ord 1981); squidpy
    reused Moran's for both until 1.8.2, which left the Geary p-value miscalibrated
    (scverse/squidpy#1183). These are the corrected forms, matching pysal/esda.
    """
    from scipy import stats
    s0, s1, s2 = graph_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    if mode == "geary":
        var = ((2 * s1 + s2) * (n - 1) - 4 * s02) / (2 * (n + 1) * s02)
        expected = 1.0
    else:
        var = (n * n * s1 - n * s2 + 3 * s02) / ((n - 1) * (n + 1) * s02) - (1.0 / (n - 1)) ** 2
        expected = -1.0 / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (score - expected) / np.sqrt(var)
        p = np.where(z > 0, 1.0 - stats.norm.cdf(z), stats.norm.cdf(z))
    return (p * 2.0 if two_tailed else p), var


def permutation_pvals(score, sims):
    """``pval_sim``, ``pval_z_sim`` and ``var_sim`` from the permutation null.

    ``pval_sim`` folds the count to the *smaller* tail, so it detects dispersion as well as
    clustering. A one-sided count toward clustering reports p ~ 1 for a checkerboard gene
    where this reports p ~ 1/(n_perms+1).
    """
    from scipy import stats
    n_perms = sims.shape[0]
    large = (sims >= score).sum(axis=0)
    large = np.where((n_perms - large) < large, n_perms - large, large)
    p_sim = (large + 1.0) / (n_perms + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (score - sims.mean(axis=0)) / sims.std(axis=0)
        p_z_sim = np.where(z > 0, 1.0 - stats.norm.cdf(z), stats.norm.cdf(z))
    return p_sim, p_z_sim, sims.var(axis=0)


def multiple_testing(pvals, method: str = "fdr_bh"):
    """Multiple-testing correction, matching ``statsmodels.stats.multitest.multipletests``.

    Implemented here rather than imported: statsmodels is not a dependency of this package
    (numpy/scipy/anndata/mlx are), and Benjamini-Hochberg is a sort and a running minimum.
    Methods beyond these two raise instead of silently going uncorrected.
    """
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    if method != "fdr_bh":
        raise NotImplementedError(f"corr_method={method!r} is not implemented "
                                  f"(expected 'fdr_bh', 'bonferroni' or None)")
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # step-up: an adjusted p-value can never exceed one for a larger raw p-value
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return out


def spatial_autocorr(X, connectivity, mode: str = "moran", n_perms: int | None = None,
                     random_state: int = 0) -> dict:
    """Per-gene spatial autocorrelation (squidpy ``gr.spatial_autocorr``).

    ``X`` is (cells × genes); ``connectivity`` a scipy sparse spatial graph, already
    row-normalised by the caller if that is wanted. ``mode`` is ``"moran"`` (Moran's I) or
    ``"geary"`` (Geary's C). Returns the observed statistic and, when ``n_perms`` is given,
    the whole ``(n_perms, genes)`` null — the caller turns that into p-values.

    The null permutes the graph's **rows**, as squidpy does, rather than the expression
    values. On the GPU that is much the cheaper end: re-indexing the edge list costs one
    ``take`` over ``n`` integers per draw, where re-uploading a permuted ``X`` costs a full
    host-to-device copy of the matrix.
    """
    import mlx.core as mx
    import scipy.sparse as sp

    coo = sp.coo_matrix(connectivity)
    src0 = mx.array(coo.row.astype(np.int32))
    dst = mx.array(coo.col.astype(np.int32))
    w = mx.array(coo.data.astype(np.float32))
    S0 = float(coo.data.sum())
    n = X.shape[0]

    Xg = mx.array(np.asarray(X, dtype=np.float32))
    Xc = Xg - mx.mean(Xg, axis=0, keepdims=True)
    denom = mx.sum(Xc * Xc, axis=0)

    def stat(src):
        if mode == "moran":
            num = mx.sum(Xc * _spmm_scatter(src, dst, w, Xc), axis=0)
            return (n / S0) * num / denom
        # Geary's C: sum_ij w_ij (x_i - x_j)^2, summed directly over the edge list
        # (exact for any graph, no symmetry assumption).
        de = Xc[src] - Xc[dst]
        num = mx.sum(w[:, None] * de * de, axis=0)
        return (n - 1) / (2.0 * S0) * num / denom

    obs = stat(src0)
    mx.eval(obs)
    obs_np = np.asarray(obs)
    if not n_perms:
        return {mode: obs_np, "sims": None}

    rng = np.random.default_rng(random_state)
    sims = np.empty((n_perms, obs_np.shape[0]), dtype=np.float64)
    for i in range(n_perms):
        perm = mx.array(rng.permutation(n).astype(np.int32))
        sims[i] = np.asarray(stat(mx.take(perm, src0)))
    return {mode: obs_np, "sims": sims}
