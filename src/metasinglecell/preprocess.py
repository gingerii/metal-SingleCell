"""scanpy drop-in front-end functions built on the Metal sparse substrate.

Currently: ``highly_variable_genes`` (seurat flavor). The heavy per-gene
mean/variance reduction runs on the GPU (``CSR.gene_moments``); the cheap binning
+ normalization is done on the host in float64, mirroring scanpy's exact algorithm
(expm1 -> mean/var -> log-dispersion -> equal-width mean bins -> per-bin z-score)
so results match ``sc.pp.highly_variable_genes(flavor="seurat")``.
"""

from __future__ import annotations

import numpy as np


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
