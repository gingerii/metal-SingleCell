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


def highly_variable_genes(
    csr,
    n_top_genes: int = 2000,
    n_bins: int = 20,
) -> "object":
    """Seurat-flavor HVG on a log-normalized :class:`~metasinglecell.sparse.CSR`.

    Returns a pandas DataFrame with columns ``means``, ``dispersions``,
    ``dispersions_norm`` and ``highly_variable`` (one row per gene), matching
    scanpy's ``flavor="seurat"`` output. ``csr`` must hold log1p-normalized data.
    """
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

    # Equal-width mean bins; per-bin dispersion mean/std (ddof=1) -> z-score.
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    stats = df.groupby("mean_bin", observed=True)["dispersions"].agg(avg="mean", dev="std")

    # scanpy: bins with a single gene have NaN std -> treat as mean 0, dev = avg.
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
