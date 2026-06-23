"""AnnData ``pp`` namespace — drop-in mirror of ``scanpy.pp`` / ``rapids_singlecell.pp``.

Each function takes an ``AnnData``, runs the GPU compute on the right matrix, and writes
results back to the **same slots scanpy uses** (``adata.X``, ``adata.var``, ``adata.obsm``,
``adata.obsp``, ``adata.uns``, ``adata.obs``), with scanpy's ``copy`` semantics: mutate in
place and return ``None``, or return a modified copy when ``copy=True``. So existing scanpy
pipelines work by swapping ``sc.pp`` → ``msc.pp``.
"""

from __future__ import annotations

import numpy as np

from . import preprocess as _pp
from .sparse import CSR


def _csr(adata, layer=None):
    import scipy.sparse as sp
    X = adata.layers[layer] if layer is not None else adata.X
    return CSR.from_scipy(sp.csr_matrix(X))


def normalize_total(adata, target_sum: float | None = None, layer=None, copy: bool = False):
    """Normalize counts per cell (``sc.pp.normalize_total``). ``target_sum=None`` → median."""
    adata = adata.copy() if copy else adata
    import scipy.sparse as sp
    X = sp.csr_matrix(adata.layers[layer] if layer is not None else adata.X)
    ts = float(target_sum) if target_sum is not None else float(np.median(np.asarray(X.sum(1)).ravel()))
    out = CSR.from_scipy(X).normalize_total(ts).to_scipy()
    if layer is not None:
        adata.layers[layer] = out
    else:
        adata.X = out
    return adata if copy else None


def log1p(adata, layer=None, copy: bool = False):
    """``log(1 + x)`` (``sc.pp.log1p``); records ``adata.uns['log1p']``."""
    adata = adata.copy() if copy else adata
    out = _csr(adata, layer).log1p().to_scipy()
    if layer is not None:
        adata.layers[layer] = out
    else:
        adata.X = out
    adata.uns["log1p"] = {"base": None}
    return adata if copy else None


def highly_variable_genes(adata, n_top_genes: int = 2000, n_bins: int = 20,
                          flavor: str = "seurat", layer=None, copy: bool = False):
    """Highly variable genes (``sc.pp.highly_variable_genes``); writes ``adata.var`` columns."""
    adata = adata.copy() if copy else adata
    df = _pp.highly_variable_genes(_csr(adata, layer), n_top_genes=n_top_genes,
                                   n_bins=n_bins, flavor=flavor)
    for col in df.columns:
        adata.var[col] = df[col].to_numpy()
    adata.uns["hvg"] = {"flavor": flavor}
    return adata if copy else None


def filter_cells(adata, min_counts=None, max_counts=None, min_genes=None,
                 max_genes=None, copy: bool = False):
    """Filter cells (``sc.pp.filter_cells``); subsets ``adata`` in place."""
    adata = adata.copy() if copy else adata
    keep = _pp.filter_cells(_csr(adata), min_counts=min_counts, max_counts=max_counts,
                            min_genes=min_genes, max_genes=max_genes)
    adata._inplace_subset_obs(keep)
    return adata if copy else None


def filter_genes(adata, min_counts=None, max_counts=None, min_cells=None,
                 max_cells=None, copy: bool = False):
    """Filter genes (``sc.pp.filter_genes``); subsets ``adata`` in place."""
    adata = adata.copy() if copy else adata
    keep = _pp.filter_genes(_csr(adata), min_counts=min_counts, max_counts=max_counts,
                            min_cells=min_cells, max_cells=max_cells)
    adata._inplace_subset_var(keep)
    return adata if copy else None


def scale(adata, max_value: float | None = 10.0, zero_center: bool = True,
          layer=None, copy: bool = False):
    """Z-score genes then clip (``sc.pp.scale``). Densifies (zero-centering breaks sparsity)."""
    adata = adata.copy() if copy else adata
    out = _pp.scale(_csr(adata, layer), max_value=max_value, zero_center=zero_center)
    if layer is not None:
        adata.layers[layer] = out
    else:
        adata.X = out
    return adata if copy else None


def regress_out(adata, keys, copy: bool = False):
    """Regress out covariates in ``adata.obs[keys]`` (``sc.pp.regress_out``)."""
    adata = adata.copy() if copy else adata
    keys = [keys] if isinstance(keys, str) else list(keys)
    cov = np.column_stack([np.asarray(adata.obs[k], dtype=np.float32) for k in keys])
    adata.X = _pp.regress_out(adata.X, cov)
    return adata if copy else None


def normalize_pearson_residuals(adata, theta: float = 100.0, clip: float | None = None,
                                copy: bool = False):
    """Analytic Pearson residuals (``sc.experimental.pp.normalize_pearson_residuals``)."""
    adata = adata.copy() if copy else adata
    import scipy.sparse as sp
    adata.X = _pp.normalize_pearson_residuals(sp.csr_matrix(adata.X), theta=theta, clip=clip)
    return adata if copy else None


def scrublet(adata, sim_doublet_ratio: float = 2.0, expected_doublet_rate: float = 0.05,
             n_neighbors: int | None = None, n_pcs: int = 30, random_state: int = 0,
             copy: bool = False):
    """Doublet detection (``sc.pp.scrublet``); writes ``obs['doublet_score']``/``['predicted_doublet']``."""
    adata = adata.copy() if copy else adata
    import scipy.sparse as sp
    res = _pp.scrublet(sp.csr_matrix(adata.X), sim_doublet_ratio=sim_doublet_ratio,
                       n_neighbors=n_neighbors, expected_doublet_rate=expected_doublet_rate,
                       n_pcs=n_pcs, random_state=random_state)
    adata.obs["doublet_score"] = res["doublet_scores"]
    adata.obs["predicted_doublet"] = res["predicted_doublets"]
    adata.uns["scrublet"] = {"threshold": res["threshold"]}
    return adata if copy else None


def calculate_qc_metrics(adata, copy: bool = False):
    """Per-cell/per-gene QC metrics (``sc.pp.calculate_qc_metrics``)."""
    adata = adata.copy() if copy else adata
    m = _pp.calculate_qc_metrics(_csr(adata))
    for k, v in m.items():
        (adata.obs if len(v) == adata.n_obs else adata.var)[k] = np.asarray(v)
    return adata if copy else None


def pca(adata, n_comps: int = 50, layer=None, use_highly_variable: bool | None = None,
        zero_center: bool = True, svd_solver: str = "randomized", random_state: int = 0,
        copy: bool = False):
    """PCA (``sc.pp.pca``); writes ``obsm['X_pca']``, ``varm['PCs']``, ``uns['pca']``.

    Sparse input → the sparse-aware randomized PCA (no densify). ``use_highly_variable``
    restricts to ``adata.var['highly_variable']`` (default True if that column exists).
    """
    import scipy.sparse as sp

    from .decomposition import pca as _pca
    adata = adata.copy() if copy else adata
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    X = adata.layers[layer] if layer is not None else adata.X
    mask = adata.var["highly_variable"].to_numpy() if use_highly_variable else np.ones(adata.n_vars, bool)
    Xsub = X[:, mask]
    inp = CSR.from_scipy(sp.csr_matrix(Xsub).astype(np.float32)) if sp.issparse(Xsub) and zero_center \
        else np.asarray(Xsub.todense() if sp.issparse(Xsub) else Xsub, dtype=np.float32)
    X_pca, comps, vr = _pca(inp, n_comps=n_comps, solver=svd_solver, random_state=random_state)
    adata.obsm["X_pca"] = np.asarray(X_pca)
    pcs = np.zeros((adata.n_vars, n_comps), dtype=np.float32)
    pcs[mask] = np.asarray(comps).T
    adata.varm["PCs"] = pcs
    adata.uns["pca"] = {"variance_ratio": np.asarray(vr), "use_highly_variable": bool(use_highly_variable)}
    return adata if copy else None


def neighbors(adata, n_neighbors: int = 15, use_rep: str = "X_pca", random_state: int = 0,
              copy: bool = False):
    """kNN graph (``sc.pp.neighbors``); writes ``obsp['distances']``/``['connectivities']``, ``uns['neighbors']``."""
    from .neighbors import neighbors as _nb
    adata = adata.copy() if copy else adata
    rep = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    dist, conn = _nb(np.asarray(rep, dtype=np.float32), n_neighbors=n_neighbors,
                     random_state=random_state)
    adata.obsp["distances"] = dist
    adata.obsp["connectivities"] = conn
    adata.uns["neighbors"] = {"connectivities_key": "connectivities",
                              "distances_key": "distances",
                              "params": {"n_neighbors": n_neighbors, "method": "umap", "use_rep": use_rep}}
    return adata if copy else None


def harmony_integrate(adata, key, basis: str = "X_pca", adjusted_basis: str = "X_pca_harmony",
                      random_state: int = 0, copy: bool = False):
    """Harmony batch integration (``sc.external.pp.harmony_integrate``); writes ``obsm[adjusted_basis]``."""
    from .integration import harmonize
    adata = adata.copy() if copy else adata
    batch = adata.obs[key].to_numpy()
    adata.obsm[adjusted_basis] = np.asarray(harmonize(adata.obsm[basis], batch, random_state=random_state))
    return adata if copy else None


def bbknn(adata, batch_key, use_rep: str = "X_pca", neighbors_within_batch: int = 3,
          random_state: int = 0, copy: bool = False):
    """Batch-balanced kNN (``sc.external.pp.bbknn``); writes ``obsp`` + ``uns['neighbors']``."""
    from .neighbors import bbknn as _bbknn
    adata = adata.copy() if copy else adata
    dist, conn = _bbknn(np.asarray(adata.obsm[use_rep], dtype=np.float32),
                        adata.obs[batch_key].to_numpy(),
                        neighbors_within_batch=neighbors_within_batch, random_state=random_state)
    adata.obsp["distances"] = dist
    adata.obsp["connectivities"] = conn
    adata.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                              "params": {"n_neighbors": neighbors_within_batch, "method": "umap"}}
    return adata if copy else None
