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
    """Our GPU CSR — for funcs that take a CSR (normalize/log1p/hvg/scale)."""
    import scipy.sparse as sp
    X = adata.layers[layer] if layer is not None else adata.X
    return CSR.from_scipy(sp.csr_matrix(X))


def _sci(adata, layer=None):
    """A scipy CSR — for funcs that take raw scipy (filter/qc/regress)."""
    import scipy.sparse as sp
    X = adata.layers[layer] if layer is not None else adata.X
    return sp.csr_matrix(X)


def _backed_reader(adata, layer=None):
    """A ZarrRowReader iff ``adata.X`` is an on-disk backed CSR, else ``None``.

    This is the sole branch point: when it returns a reader the wrapper takes the
    out-of-core streaming path; when ``None`` the existing in-core path runs unchanged.
    Streaming operates on ``.X`` only (backed layers are not supported this milestone).
    """
    if layer is not None:
        return None
    try:
        import anndata.abc
    except Exception:
        return None
    if isinstance(adata.X, anndata.abc.CSRDataset):
        from .backed import open_backed
        return open_backed(adata.X)
    return None


# The backed store holds raw counts only (no intermediate write-back this milestone), so
# streaming normalize_total/log1p/scale record a DEFERRED transform prefix in
# adata.uns["_stream_transforms"] that the terminal consumers (HVG/PCA) re-apply per block.
def _record_transform(adata, *stage):
    t = list(adata.uns.get("_stream_transforms", []))
    t.append(tuple(stage))
    adata.uns["_stream_transforms"] = t


def _build_transform(adata):
    from .backed import BlockTransform
    return BlockTransform(list(adata.uns.get("_stream_transforms", [])))


def _build_pca_transform(adata, mask):
    """Deferred prefix for streaming PCA: insert ``hvg_subset`` before ``scale`` and subset
    the (column-independent) per-gene scale params to the mask, so only the HVG columns
    densify. ``mask=None`` keeps the full gene set (no-HVG / full-panel PCA)."""
    from .backed import BlockTransform
    stages, subset_done = [], False
    for st in adata.uns.get("_stream_transforms", []):
        if st[0] == "scale":
            mean, std, mx_, zc = st[1]
            if mask is not None:
                stages.append(("hvg_subset", mask)); mean, std = mean[mask], std[mask]
                subset_done = True
            stages.append(("scale", (mean, std, mx_, zc)))
        else:
            stages.append(st)
    if mask is not None and not subset_done:       # no scale stage: subset before covariance
        stages.append(("hvg_subset", mask))
    return BlockTransform(stages)


def materialize(adata, path, block_rows: int | None = None):
    """Checkpoint the deferred normalize→log1p transform to a new backed zarr (write-back).

    Streams the raw backed ``.X`` through the recorded ``normalize_total``/``log1p`` prefix,
    writes the post-log1p (still-sparse) matrix to ``path`` once, then **rebinds** ``adata.X``
    to that store and clears the deferred prefix. Subsequent ``scale``/``highly_variable_genes``/
    ``pca`` therefore read the already-transformed matrix instead of re-deriving normalize→log1p
    from raw on every pass (opt-in — the default streaming path stays fully deferred). Output
    values are identical, so downstream results are unchanged.

    Must be called at the **log1p boundary**: the recorded transform may contain only
    ``normalize_total``/``log1p`` (no ``scale``/``hvg_subset`` — those densify or reshape and
    belong to the deferred consumers). Raises otherwise.
    """
    import anndata
    import zarr
    from anndata.io import sparse_dataset

    from .backed import open_backed, write_transformed_zarr

    reader = _backed_reader(adata)
    if reader is None:
        raise ValueError("materialize requires a backed (on-disk CSR) adata.X")
    stages = list(adata.uns.get("_stream_transforms", []))
    allowed = {"normalize_total", "log1p"}
    bad = [s[0] for s in stages if s[0] not in allowed]
    if bad:
        raise ValueError(f"materialize is defined at the log1p boundary; the deferred prefix may "
                         f"only hold {sorted(allowed)}, got {[s[0] for s in stages]}. Checkpoint "
                         f"before scale / HVG-subset.")
    tf = _build_transform(adata)
    write_transformed_zarr(reader, tf, path, obs=adata.obs.copy(), var=adata.var.copy(),
                           block_rows=block_rows)
    adata.X = sparse_dataset(zarr.open(str(path), mode="r")["X"])   # rebind to the checkpoint
    adata.uns["_stream_transforms"] = []                            # prefix now baked in → identity
    return adata


def write_obsm(adata, key: str, path):
    """Persist an ``obsm`` array (e.g. ``X_pca``) to a ``.npy`` on disk so the in-memory-fitting
    downstream (neighbors/UMAP/clustering) can start from it with no recompute."""
    np.save(str(path), np.asarray(adata.obsm[key]))
    return str(path)


def normalize_total(adata, target_sum: float | None = None, layer=None,
                    exclude_highly_expressed: bool = False, copy: bool = False):
    """Normalize counts per cell (``sc.pp.normalize_total``). ``target_sum=None`` → median."""
    if exclude_highly_expressed:
        raise NotImplementedError("normalize_total(exclude_highly_expressed=True) needs a "
                                  "second global pass; not supported (scoped out).")
    adata = adata.copy() if copy else adata
    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: record a deferred transform
        if target_sum is not None:
            ts = float(target_sum)
        elif "total_counts" in adata.obs:        # reuse per-cell totals from a prior QC pass
            ts = float(np.median(adata.obs["total_counts"].to_numpy()))
        else:                                    # else one lightweight pass for row sums
            from .backed import stream_qc
            ts = float(np.median(stream_qc(reader)["total_counts"]))
        _record_transform(adata, "normalize_total", ts)
        return adata if copy else None
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
    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: record a deferred transform
        _record_transform(adata, "log1p")
        adata.uns["log1p"] = {"base": None}
        return adata if copy else None
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
    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: stream per-gene moments
        if flavor not in ("seurat", "cell_ranger"):
            raise NotImplementedError(f"streaming HVG supports seurat/cell_ranger, not {flavor!r}")
        from .backed import stream_gene_moments
        mean, var = stream_gene_moments(reader, _build_transform(adata), flavor)
        df = _pp._hvg_dispersion_from_moments(mean, var, n_top_genes, n_bins, flavor)
    else:
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
    keep = _pp.filter_cells(_sci(adata), min_counts=min_counts, max_counts=max_counts,
                            min_genes=min_genes, max_genes=max_genes)
    adata._inplace_subset_obs(keep)
    return adata if copy else None


def filter_genes(adata, min_counts=None, max_counts=None, min_cells=None,
                 max_cells=None, copy: bool = False):
    """Filter genes (``sc.pp.filter_genes``); subsets ``adata`` in place."""
    adata = adata.copy() if copy else adata
    keep = _pp.filter_genes(_sci(adata), min_counts=min_counts, max_counts=max_counts,
                            min_cells=min_cells, max_cells=max_cells)
    adata._inplace_subset_var(keep)
    return adata if copy else None


def scale(adata, max_value: float | None = 10.0, zero_center: bool = True,
          layer=None, copy: bool = False):
    """Z-score genes then clip (``sc.pp.scale``). Densifies (zero-centering breaks sparsity)."""
    adata = adata.copy() if copy else adata
    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: pass-1 stats, defer the apply
        from .backed import stream_scale_stats
        mean, std = stream_scale_stats(reader, _build_transform(adata))
        _record_transform(adata, "scale", (mean, std, max_value, zero_center))
        return adata if copy else None
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
    reader = _backed_reader(adata)
    if reader is not None:                       # out-of-core: stream row-blocks
        from .backed import stream_qc
        m = stream_qc(reader)
    else:
        m = _pp.calculate_qc_metrics(_sci(adata))
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

    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: fused streaming covariance-eigh
        from .decomposition import pca_covariance_eigh_streaming
        hvg_mask = adata.var["highly_variable"].to_numpy() if use_highly_variable else None
        H = int(hvg_mask.sum()) if hvg_mask is not None else adata.n_vars
        tf = _build_pca_transform(adata, hvg_mask)
        X_pca, comps, vr = pca_covariance_eigh_streaming(reader, tf, H, n_comps=n_comps)
        adata.obsm["X_pca"] = np.asarray(X_pca)
        pcs = np.zeros((adata.n_vars, n_comps), dtype=np.float32)
        pcs[hvg_mask if hvg_mask is not None else slice(None)] = np.asarray(comps).T
        adata.varm["PCs"] = pcs
        adata.uns["pca"] = {"variance_ratio": np.asarray(vr),
                            "use_highly_variable": bool(use_highly_variable)}
        return adata if copy else None

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
