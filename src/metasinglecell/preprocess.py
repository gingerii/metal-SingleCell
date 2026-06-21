"""scanpy drop-in front-end functions built on the Metal sparse substrate.

Currently: ``highly_variable_genes`` (seurat flavor). The heavy per-gene
mean/variance reduction runs on the GPU (``CSR.gene_moments``); the cheap binning
+ normalization is done on the host in float64, mirroring scanpy's exact algorithm
(expm1 -> mean/var -> log-dispersion -> equal-width mean bins -> per-bin z-score)
so results match ``sc.pp.highly_variable_genes(flavor="seurat")``.
"""

from __future__ import annotations

import numpy as np

from .sparse import CSR


def calculate_qc_metrics(X) -> dict:
    """Per-cell and per-gene QC metrics (scanpy ``pp.calculate_qc_metrics``).

    ``X`` is a (cells x genes) scipy sparse / array of counts. Returns a dict with
    per-cell ``total_counts``/``n_genes_by_counts`` and per-gene ``total_counts``/
    ``n_cells_by_counts``/``mean_counts``/``pct_dropout_by_counts``. GPU reductions.
    """
    import scipy.sparse as sp

    csr = CSR.from_scipy(sp.csr_matrix(X))
    n_cells = csr.shape[0]
    cell_total, cell_ngenes = csr.qc_metrics()
    gene_total, gene_ncells = csr.gene_counts()
    return {
        "total_counts": cell_total,
        "n_genes_by_counts": cell_ngenes.astype(np.int64),
        "gene_total_counts": gene_total,
        "n_cells_by_counts": gene_ncells.astype(np.int64),
        "mean_counts": gene_total / n_cells,
        "pct_dropout_by_counts": 100.0 * (1.0 - gene_ncells / n_cells),
    }


def filter_cells(X, min_counts=None, max_counts=None, min_genes=None,
                 max_genes=None) -> np.ndarray:
    """Boolean cell mask passing the given thresholds (scanpy ``pp.filter_cells``).

    Exactly one of the count/gene bounds is typically used at a time, mirroring
    scanpy; here any combination is ANDed.
    """
    import scipy.sparse as sp

    total, ngenes = CSR.from_scipy(sp.csr_matrix(X)).qc_metrics()
    keep = np.ones(X.shape[0], dtype=bool)
    if min_counts is not None:
        keep &= total >= min_counts
    if max_counts is not None:
        keep &= total <= max_counts
    if min_genes is not None:
        keep &= ngenes >= min_genes
    if max_genes is not None:
        keep &= ngenes <= max_genes
    return keep


def filter_genes(X, min_counts=None, max_counts=None, min_cells=None,
                 max_cells=None) -> np.ndarray:
    """Boolean gene mask passing the given thresholds (scanpy ``pp.filter_genes``)."""
    import scipy.sparse as sp

    total, ncells = CSR.from_scipy(sp.csr_matrix(X)).gene_counts()
    keep = np.ones(X.shape[1], dtype=bool)
    if min_counts is not None:
        keep &= total >= min_counts
    if max_counts is not None:
        keep &= total <= max_counts
    if min_cells is not None:
        keep &= ncells >= min_cells
    if max_cells is not None:
        keep &= ncells <= max_cells
    return keep


def flag_gene_family(var_names, startswith=None, endswith=None,
                     contains=None) -> np.ndarray:
    """Boolean mask flagging a gene family by name (scanpy ``pp.flag_gene_family``).

    e.g. mitochondrial genes via ``startswith="MT-"``. ``var_names`` is an array
    of gene symbols.
    """
    names = np.asarray(var_names).astype(str)
    if startswith is not None:
        return np.char.startswith(names, startswith)
    if endswith is not None:
        return np.char.endswith(names, endswith)
    if contains is not None:
        return np.char.find(names, contains) >= 0
    raise ValueError("provide one of startswith / endswith / contains")


def filter_highly_variable(hvg_df):
    """Indices of highly-variable genes (scanpy ``pp.filter_highly_variable``).

    ``hvg_df`` is the DataFrame from :func:`highly_variable_genes`; returns the
    boolean mask of its ``highly_variable`` column.
    """
    return hvg_df["highly_variable"].to_numpy().astype(bool)


def regress_out(X, covariates) -> np.ndarray:
    """Regress per-gene expression on covariates, return residuals (scanpy ``pp.regress_out``).

    All genes share the design matrix [intercept | covariates], so we fit one OLS
    (fp64 LAPACK for the small k×k solve) and form residuals ``X - D·beta`` on the
    GPU. ``covariates`` is (cells,) or (cells × k). ``X`` is (cells × genes).
    """
    import mlx.core as mx

    Xd = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov[:, None]
    D = np.column_stack([np.ones(Xd.shape[0]), cov])           # intercept + covariates
    beta, *_ = np.linalg.lstsq(D, Xd, rcond=None)              # (k × genes), fp64
    resid = mx.array(Xd.astype(np.float32)) - mx.array(D.astype(np.float32)) @ mx.array(beta.astype(np.float32))
    mx.eval(resid)
    return np.asarray(resid)


def normalize_pearson_residuals(X, theta: float = 100.0, clip: float | None = None) -> np.ndarray:
    """Analytic Pearson residuals of a NB model (scanpy ``pp.normalize_pearson_residuals``).

    ``mu = rowsum·colsum / total``; ``residual = (X - mu) / sqrt(mu + mu²/theta)``,
    clipped to ``[-clip, clip]`` (default ``sqrt(n_cells)``). Returns a dense array.
    """
    import mlx.core as mx

    Xd = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    Xg = mx.array(Xd)
    row = mx.sum(Xg, axis=1, keepdims=True)
    col = mx.sum(Xg, axis=0, keepdims=True)
    total = mx.sum(Xg)
    mu = (row @ col) / total
    resid = (Xg - mu) / mx.sqrt(mu + mu * mu / theta)
    c = float(np.sqrt(Xd.shape[0])) if clip is None else clip
    resid = mx.clip(resid, -c, c)
    mx.eval(resid)
    return np.asarray(resid)


def scrublet_simulate_doublets(counts, sim_doublet_ratio: float = 2.0,
                               random_state: int = 0):
    """Simulate doublets by summing random pairs of cells (scanpy ``pp.scrublet_simulate_doublets``)."""
    import scipy.sparse as sp

    C = sp.csr_matrix(counts)
    n = C.shape[0]
    n_sim = int(sim_doublet_ratio * n)
    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, n, size=(n_sim, 2))
    sim = C[pairs[:, 0]] + C[pairs[:, 1]]               # combined transcript counts
    return sim, pairs


def scrublet(counts, sim_doublet_ratio: float = 2.0, n_neighbors: int | None = None,
             expected_doublet_rate: float = 0.05, n_pcs: int = 30,
             random_state: int = 0) -> dict:
    """Doublet detection (scanpy/rapids-singlecell ``pp.scrublet``).

    Simulate doublets, embed real+simulated together (normalize → log1p → PCA),
    then score each real cell by how many of its nearest neighbors are simulated
    doublets (adjusted for the simulation ratio and expected rate). Returns
    ``doublet_scores`` per real cell and a boolean ``predicted_doublets``.
    """
    import scipy.sparse as sp

    from .decomposition import pca
    from .neighbors import _knn_gpu
    from .sparse import CSR

    C = sp.csr_matrix(counts)
    n = C.shape[0]
    sim, _ = scrublet_simulate_doublets(C, sim_doublet_ratio, random_state)
    combined = sp.vstack([C, sim]).tocsr()
    if n_neighbors is None:
        n_neighbors = int(round(0.5 * np.sqrt(n)))

    # normalize -> log1p -> scale-free PCA on the combined matrix
    lognorm = CSR.from_scipy(combined).normalize_total(1e4).log1p().toarray()
    X_pca, _, _ = pca(lognorm, n_comps=min(n_pcs, combined.shape[1] - 1),
                      solver="randomized", random_state=random_state)

    knn_idx, _ = _knn_gpu(X_pca.astype(np.float32), n_neighbors + 1)
    knn_idx = knn_idx[:n, 1:]                           # real cells, exclude self
    is_sim = knn_idx >= n                               # neighbor is a simulated doublet
    frac_sim = is_sim.mean(axis=1)

    # Bayesian-style adjustment for the simulation ratio rho (Scrublet eq.)
    rho = expected_doublet_rate
    r = sim_doublet_ratio
    q = frac_sim
    doublet_scores = (q * rho / r) / (q * rho / r + (1 - q) * (1 - rho) + 1e-12)
    # threshold: Otsu-like split of the score distribution
    thr = float(np.quantile(doublet_scores, 1 - rho))
    return {"doublet_scores": doublet_scores,
            "predicted_doublets": doublet_scores > thr, "threshold": thr}


def scale(csr, max_value: float | None = 10.0, zero_center: bool = True) -> np.ndarray:
    """Z-score each gene then clip (scanpy ``sc.pp.scale``).

    Per-gene: subtract mean, divide by std (ddof=1; std==0 -> 1), then clip to
    ``[-max_value, max_value]`` (lower bound only when ``zero_center``). Zero-
    centering densifies, so this runs as dense MLX ops on the GPU and returns a
    dense float32 numpy array (cells x genes).
    """
    import mlx.core as mx

    x = mx.array(csr.toarray().astype(np.float32))
    n = x.shape[0]
    mean = mx.mean(x, axis=0)
    var = mx.sum((x - mean) ** 2, axis=0) / (n - 1)  # ddof=1, R convention
    std = mx.sqrt(var)
    std = mx.where(std == 0, mx.array(1.0, dtype=std.dtype), std)

    if zero_center:
        x = x - mean
    x = x / std

    if max_value is not None:
        upper = mx.minimum(x, mx.array(max_value, dtype=x.dtype))
        x = mx.maximum(upper, mx.array(-max_value, dtype=x.dtype)) if zero_center else upper

    mx.eval(x)
    return np.asarray(x)


def highly_variable_genes(csr, n_top_genes: int = 2000, n_bins: int = 20,
                          flavor: str = "seurat") -> "object":
    """Highly variable genes (scanpy ``pp.highly_variable_genes``).

    ``flavor`` ∈ {"seurat", "cell_ranger", "seurat_v3"}. seurat/cell_ranger expect
    **log-normalized** data; seurat_v3 expects **raw counts**. Returns a pandas
    DataFrame with per-gene metrics and a ``highly_variable`` flag.
    """
    if flavor == "seurat_v3":
        return _hvg_seurat_v3(csr, n_top_genes)
    if flavor in ("seurat", "cell_ranger"):
        return _hvg_dispersion(csr, n_top_genes, n_bins, flavor)
    raise ValueError(f"unknown flavor {flavor!r}")


def _hvg_seurat_v3(csr, n_top_genes: int):
    """seurat_v3 HVG on raw counts: rank genes by clipped normalized variance."""
    import mlx.core as mx
    import pandas as pd
    from statsmodels.nonparametric.smoothers_lowess import lowess

    col = csr.indices
    data = csr.data
    N, ng = csr.shape
    total = np.asarray(mx.zeros((ng,), dtype=mx.float32).at[col].add(data), dtype=np.float64)
    sumsq = np.asarray(mx.zeros((ng,), dtype=mx.float32).at[col].add(data * data), dtype=np.float64)
    mean = total / N
    var = (sumsq - N * mean ** 2) / (N - 1)

    not_const = var > 0
    # loess of log10(var) ~ log10(mean) (skmisc unavailable -> statsmodels lowess, frac 0.3)
    x = np.log10(mean[not_const]); y = np.log10(var[not_const])
    fit = lowess(y, x, frac=0.3, return_sorted=True)
    estimat = np.interp(np.log10(np.where(mean > 0, mean, 1e-12)), fit[:, 0], fit[:, 1])
    reg_std = np.sqrt(10 ** estimat)

    clip_val = reg_std * np.sqrt(N) + mean
    clipped = mx.minimum(data, mx.array(clip_val[col].astype(np.float32)))
    sc = np.asarray(mx.zeros((ng,), dtype=mx.float32).at[col].add(clipped), dtype=np.float64)
    scsq = np.asarray(mx.zeros((ng,), dtype=mx.float32).at[col].add(clipped * clipped), dtype=np.float64)
    norm_var = (scsq - 2 * mean * sc + N * mean ** 2) / ((N - 1) * np.maximum(reg_std ** 2, 1e-12))
    norm_var[~not_const] = 0.0

    order = np.argsort(-norm_var)
    hv = np.zeros(ng, dtype=bool); hv[order[:n_top_genes]] = True
    df = pd.DataFrame({"means": mean, "variances": var,
                       "variances_norm": norm_var, "highly_variable": hv})
    df["highly_variable_rank"] = np.argsort(np.argsort(-norm_var)).astype(float)
    return df


def _hvg_dispersion(csr, n_top_genes: int, n_bins: int, flavor: str):
    """seurat / cell_ranger dispersion-based HVG on log-normalized data."""
    import pandas as pd

    # GPU: per-gene mean/var of expm1(lognorm). Host math in float64 for parity.
    mean, var = csr.gene_moments()
    mean = mean.astype(np.float64)
    var = var.astype(np.float64)

    mean[mean == 0] = 1e-12
    dispersion = var / mean
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    mean = np.log1p(mean)  # seurat: logarithmized mean

    df = pd.DataFrame({"means": mean, "dispersions": dispersion})

    if flavor == "cell_ranger":
        # percentile mean-bins; per-bin median + MAD-based dispersion z-score.
        bins = np.r_[-np.inf, np.percentile(df["means"], np.arange(10, 105, 5)), np.inf]
        df["mean_bin"] = pd.cut(df["means"], bins=bins)
        grouped = df.groupby("mean_bin", observed=True)["dispersions"]
        avg = grouped.median()
        dev = grouped.apply(lambda g: np.median(np.abs(g - np.median(g)))) * 1.4826
        stats = pd.DataFrame({"avg": avg, "dev": dev})
    else:  # seurat: equal-width bins; per-bin mean/std (ddof=1)
        df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
        stats = df.groupby("mean_bin", observed=True)["dispersions"].agg(avg="mean", dev="std")
        # bins with a single gene have NaN std -> treat as mean 0, dev = avg.
        one_gene = stats["dev"].isnull()
        stats.loc[one_gene, "dev"] = stats.loc[one_gene, "avg"]
        stats.loc[one_gene, "avg"] = 0

    per_gene = stats.loc[df["mean_bin"]].set_index(df.index)
    df["dispersions_norm"] = (df["dispersions"] - per_gene["avg"]) / per_gene["dev"]

    # Select the top n_top_genes by normalized dispersion (NaNs -> -inf).
    dn = df["dispersions_norm"].to_numpy()
    finite = dn[~np.isnan(dn)]
    n = min(n_top_genes, finite.size)
    cutoff = np.sort(finite)[::-1][n - 1]
    df["highly_variable"] = np.nan_to_num(dn, nan=-np.inf) >= cutoff

    df.drop(columns="mean_bin", inplace=True)
    return df
