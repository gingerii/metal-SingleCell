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

# "argument not supplied", distinct from an explicit None — scanpy's `_empty` sentinel.
# `pca(mask_var=None)` means ALL variables; omitting it defaults to highly_variable.
_EMPTY = type("_Empty", (), {"__repr__": lambda self: "_empty", "__bool__": lambda self: False})()


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


def _reject_backed(adata, fn_name, layer=None):
    """Raise for a backed (on-disk) ``.X`` on a wrapper with no streaming path.

    Better a clear error than the silent failure mode: ``sp.csr_matrix(adata.X)`` on a
    backed ``CSRDataset`` fully densifies (OOM at scale), and a wrapper that reads ``.X``
    directly would compute on raw counts, ignoring any deferred normalize→log1p prefix.
    """
    if _backed_reader(adata, layer) is not None:
        raise NotImplementedError(
            f"{fn_name} does not support a backed (on-disk) AnnData.X. Load into memory "
            "(`adata = adata.to_memory()`), or use the streaming-capable steps "
            "(calculate_qc_metrics / normalize_total / log1p / highly_variable_genes / "
            "scale / pca) which take the out-of-core path automatically.")


def _copy_or_reject(adata, copy, fn_name, layer=None):
    """Honour ``copy=`` without tripping over a backed ``.X``.

    ``AnnData.copy()`` reaches into the on-disk store and fails with
    ``AttributeError: '_CSRDataset' object has no attribute 'copy'`` — an unhelpful error for
    something that cannot work anyway, since duplicating a backed matrix in memory defeats the
    point of it being backed. Say so instead.
    """
    if copy and _backed_reader(adata, layer) is not None:
        raise NotImplementedError(
            f"{fn_name}(copy=True) is not supported on a backed (on-disk) adata.X — the store "
            "cannot be duplicated in memory. Use copy=False (the streaming path writes in "
            "place), or load first with `adata = adata.to_memory()`.")
    return adata.copy() if copy else adata


# The backed store holds raw counts only (no intermediate write-back this milestone), so
# streaming normalize_total/log1p/scale record a DEFERRED transform prefix that the terminal
# consumers (HVG/PCA) re-apply per block. It lives in a module-level registry rather than in
# `uns`: the scale stage carries per-gene arrays of differing shapes, and anndata cannot write
# that -- parking it in `uns` made the object silently unwritable to h5ad.
_STREAM_TRANSFORMS: dict[int, list] = {}


def _stream_stages(adata):
    return _STREAM_TRANSFORMS.get(id(adata), [])


def _record_transform(adata, *stage):
    _STREAM_TRANSFORMS.setdefault(id(adata), []).append(tuple(stage))


def _build_transform(adata):
    from .backed import BlockTransform
    return BlockTransform(list(_stream_stages(adata)))


def _build_pca_transform(adata, mask):
    """Deferred prefix for streaming PCA: insert ``hvg_subset`` before ``scale`` and subset
    the (column-independent) per-gene scale params to the mask, so only the HVG columns
    densify. ``mask=None`` keeps the full gene set (no-HVG / full-panel PCA)."""
    from .backed import BlockTransform
    stages, subset_done = [], False
    for st in _stream_stages(adata):
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
    stages = list(_stream_stages(adata))
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
    _STREAM_TRANSFORMS.pop(id(adata), None)                         # prefix now baked in → identity
    return adata


def write_obsm(adata, key: str, path):
    """Persist an ``obsm`` array (e.g. ``X_pca``) to a ``.npy`` on disk so the in-memory-fitting
    downstream (neighbors/UMAP/clustering) can start from it with no recompute."""
    path = str(path)
    np.save(path, np.asarray(adata.obsm[key]))
    return path if path.endswith(".npy") else path + ".npy"    # np.save appends the suffix


def normalize_total(adata, target_sum: float | None = None, layer=None,
                    exclude_highly_expressed: bool = False, copy: bool = False):
    """Normalize counts per cell (``sc.pp.normalize_total``). ``target_sum=None`` → median."""
    if exclude_highly_expressed:
        raise NotImplementedError("normalize_total(exclude_highly_expressed=True) needs a "
                                  "second global pass; not supported (scoped out).")
    adata = _copy_or_reject(adata, copy, "normalize_total")
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
    adata = _copy_or_reject(adata, copy, "log1p")
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


def highly_variable_genes(adata, n_top_genes=None, n_bins: int = 20, flavor: str = "seurat",
                          min_mean: float = 0.0125, max_mean: float = 3.0, min_disp: float = 0.5,
                          max_disp: float = np.inf, layer=None, copy: bool = False):
    """Highly variable genes (``sc.pp.highly_variable_genes``); writes ``adata.var`` columns.

    ``n_top_genes=None`` (scanpy's default) selects seurat/cell_ranger genes by the
    ``min_mean``/``max_mean``/``min_disp``/``max_disp`` cutoffs; an integer takes the top-N.
    """
    adata = _copy_or_reject(adata, copy, "highly_variable_genes")
    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: stream per-gene moments
        if flavor not in ("seurat", "cell_ranger"):
            raise NotImplementedError(f"streaming HVG supports seurat/cell_ranger, not {flavor!r}")
        from .backed import stream_gene_moments
        mean, var = stream_gene_moments(reader, _build_transform(adata), flavor)
        df = _pp._hvg_dispersion_from_moments(mean, var, n_top_genes, n_bins, flavor,
                                              min_mean=min_mean, max_mean=max_mean,
                                              min_disp=min_disp, max_disp=max_disp)
    else:
        df = _pp.highly_variable_genes(_csr(adata, layer), n_top_genes=n_top_genes, n_bins=n_bins,
                                       flavor=flavor, min_mean=min_mean, max_mean=max_mean,
                                       min_disp=min_disp, max_disp=max_disp)
    for col in df.columns:
        adata.var[col] = df[col].to_numpy()
    adata.uns["hvg"] = {"flavor": flavor}
    return adata if copy else None


def filter_cells(adata, min_counts=None, max_counts=None, min_genes=None,
                 max_genes=None, copy: bool = False):
    """Filter cells (``sc.pp.filter_cells``); subsets ``adata`` in place."""
    _reject_backed(adata, "filter_cells")
    adata = adata.copy() if copy else adata
    X = _sci(adata)
    keep = _pp.filter_cells(X, min_counts=min_counts, max_counts=max_counts,
                            min_genes=min_genes, max_genes=max_genes)
    # scanpy records the quantity it filtered on, so the threshold stays inspectable after the
    # subset: obs['n_genes'] for the gene thresholds, obs['n_counts'] for the count ones.
    if min_genes is not None or max_genes is not None:
        adata.obs["n_genes"] = _pp.nonzero_per_row(X)
    if min_counts is not None or max_counts is not None:
        adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel()
    adata._inplace_subset_obs(keep)
    return adata if copy else None


def filter_genes(adata, min_counts=None, max_counts=None, min_cells=None,
                 max_cells=None, copy: bool = False):
    """Filter genes (``sc.pp.filter_genes``); subsets ``adata`` in place."""
    _reject_backed(adata, "filter_genes")
    adata = adata.copy() if copy else adata
    X = _sci(adata)
    keep = _pp.filter_genes(X, min_counts=min_counts, max_counts=max_counts,
                            min_cells=min_cells, max_cells=max_cells)
    if min_cells is not None or max_cells is not None:
        adata.var["n_cells"] = _pp.nonzero_per_col(X)
    if min_counts is not None or max_counts is not None:
        adata.var["n_counts"] = np.asarray(X.sum(axis=0)).ravel()
    adata._inplace_subset_var(keep)
    return adata if copy else None


def scale(adata, max_value: float | None = None, zero_center: bool = True,
          layer=None, copy: bool = False):
    """Z-score genes then clip (``sc.pp.scale``). Densifies (zero-centering breaks sparsity).

    ``max_value`` defaults to ``None`` (no clip), matching scanpy/rapids-singlecell — pass a
    value (e.g. 10) to clip z-scores, as the atlas/streaming demos do explicitly.
    """
    adata = _copy_or_reject(adata, copy, "scale")
    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: pass-1 stats, defer the apply
        from .backed import stream_scale_stats
        mean, std = stream_scale_stats(reader, _build_transform(adata))
        _record_transform(adata, "scale", (mean, std, max_value, zero_center))
        adata.var["mean"], adata.var["std"] = mean, std
        return adata if copy else None
    import scipy.sparse as sp
    X = sp.csr_matrix(adata.layers[layer] if layer is not None else adata.X)
    # scanpy records the fitted per-gene statistics; they are what lets a second object be put
    # on the same scale later, and they are cheap next to the scaling itself.
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq_mean = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    std[std == 0] = 1.0
    out = _pp.scale(_csr(adata, layer), max_value=max_value, zero_center=zero_center)
    if layer is not None:
        adata.layers[layer] = out
    else:
        adata.X = out
    adata.var["mean"], adata.var["std"] = mean, std
    return adata if copy else None


def _design_matrix(adata, keys):
    """Covariate columns, expanding categoricals the way scanpy does.

    scanpy detects a ``CategoricalDtype`` and regresses on **group indicators**, so each
    category gets its own level and the residual mean within every group is zero. Coercing the
    column to float instead fits a single ordinal slope through the category codes — a
    different model that runs without complaint on numeric categories (measured max residual
    difference 0.257) and raises `could not convert string to float` on string ones.
    """
    import pandas as pd
    cols = []
    for k in keys:
        s = adata.obs[k]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
            s = s.astype("category")
            # drop the first level: the intercept already carries it, and keeping all of them
            # makes the design rank-deficient
            d = pd.get_dummies(s, drop_first=True).to_numpy(dtype=np.float32)
            if d.size:
                cols.append(d)
        else:
            cols.append(np.asarray(s, dtype=np.float32)[:, None])
    if not cols:
        raise ValueError(f"no usable covariates in {keys!r}")
    return np.column_stack(cols)


def regress_out(adata, keys, copy: bool = False):
    """Regress out covariates in ``adata.obs[keys]`` (``sc.pp.regress_out``).

    Categorical covariates are expanded into group indicators, as scanpy does.
    """
    _reject_backed(adata, "regress_out")
    adata = adata.copy() if copy else adata
    keys = [keys] if isinstance(keys, str) else list(keys)
    adata.X = _pp.regress_out(adata.X, _design_matrix(adata, keys))
    return adata if copy else None


def normalize_pearson_residuals(adata, theta: float = 100.0, clip: float | None = None,
                                copy: bool = False):
    """Analytic Pearson residuals (``sc.experimental.pp.normalize_pearson_residuals``)."""
    _reject_backed(adata, "normalize_pearson_residuals")
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    if clip is not None and clip < 0:
        raise ValueError("Pearson residuals require clip >= 0")
    adata = adata.copy() if copy else adata
    import scipy.sparse as sp
    adata.X = _pp.normalize_pearson_residuals(sp.csr_matrix(adata.X), theta=theta, clip=clip)
    adata.uns["pearson_residuals_normalization"] = {
        "theta": theta, "clip": clip, "computed_on": "adata.X"}
    return adata if copy else None


def scrublet(adata, sim_doublet_ratio: float = 2.0, expected_doublet_rate: float = 0.05,
             n_neighbors: int | None = None, n_pcs: int = 30, random_state: int = 0,
             copy: bool = False):
    """Doublet detection (``sc.pp.scrublet``); writes ``obs['doublet_score']``/``['predicted_doublet']``."""
    _reject_backed(adata, "scrublet")
    adata = adata.copy() if copy else adata
    import scipy.sparse as sp
    res = _pp.scrublet(sp.csr_matrix(adata.X), sim_doublet_ratio=sim_doublet_ratio,
                       n_neighbors=n_neighbors, expected_doublet_rate=expected_doublet_rate,
                       n_pcs=n_pcs, random_state=random_state)
    adata.obs["doublet_score"] = res["doublet_scores"]
    adata.obs["predicted_doublet"] = res["predicted_doublets"]
    adata.uns["scrublet"] = {"threshold": res["threshold"]}
    return adata if copy else None


_PERCENT_TOP_DEFAULT = (50, 100, 200, 500)   # scanpy's default
_QC_VAR_RENAME = {"gene_total_counts": "total_counts"}   # per-gene total → scanpy's var slot name
# Which axis each metric belongs to. Dispatching on len(v) instead looks equivalent and is not:
# on a SQUARE object (2000 cells subset to 2000 HVGs is enough) every per-gene array also has
# len == n_obs, so all six land in .obs and gene_total_counts overwrites the per-cell total.
_QC_PER_CELL = ("total_counts", "n_genes_by_counts")
_QC_PER_GENE = ("gene_total_counts", "n_cells_by_counts", "mean_counts",
                "pct_dropout_by_counts")


def calculate_qc_metrics(adata, qc_vars=(), percent_top=_EMPTY, log1p: bool = True,
                         layer=None, inplace: bool = False, copy: bool = False):
    """Per-cell/per-gene QC metrics (``sc.pp.calculate_qc_metrics``).

    ``qc_vars`` (e.g. ``['mt']``) adds ``total_counts_<v>``/``pct_counts_<v>`` for each boolean
    ``adata.var[v]`` gene set; ``log1p`` adds ``log1p_*`` columns; ``percent_top`` adds
    ``pct_counts_in_top_N_genes`` per N. The per-gene total lands in ``var['total_counts']``
    (scanpy's name), matching ``sc.pp.calculate_qc_metrics``.

    ``inplace`` follows scanpy and defaults to **False**: the metrics come back as an
    ``(obs_df, var_df)`` pair and the object is left untouched. Pass ``inplace=True`` to write
    them into ``adata.obs``/``adata.var``.

    ``percent_top`` defaults to scanpy's ``(50, 100, 200, 500)``; it used to default to
    ``None``, which silently produced none of those columns for anyone swapping ``sc.pp`` →
    ``msc.pp``. As in scanpy, an N larger than ``n_vars`` is an error — pass a smaller tuple
    (or ``None``) for a panel with fewer genes than that.
    """
    import pandas as pd
    adata = _copy_or_reject(adata, copy, "calculate_qc_metrics")
    # Distinguish "the user asked for these" from "this is just the default": the streaming
    # path cannot produce them, and a default must not turn a working backed call into an error.
    asked_for_top = percent_top is not _EMPTY
    if not asked_for_top:
        percent_top = _PERCENT_TOP_DEFAULT
    qc_vars = [qc_vars] if isinstance(qc_vars, str) else list(qc_vars)

    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: stream row-blocks (base metrics)
        if qc_vars or (percent_top and asked_for_top):
            raise NotImplementedError(
                "qc_vars/percent_top are not supported on a backed .X (they need a per-cell "
                "gene-subset densify); load into memory or request base QC metrics only.")
        percent_top = None                       # default only — skip rather than fail
        from .backed import stream_qc
        m = stream_qc(reader)
    else:
        m = _pp.calculate_qc_metrics(_sci(adata, layer))

    # Accumulate rather than write as we go: scanpy's default leaves the object alone, so the
    # columns cannot be parked in adata.obs/.var until we know inplace is set.
    obs_out, var_out = {}, {}
    for k, v in m.items():
        if k in _QC_PER_CELL:
            obs_out[_QC_VAR_RENAME.get(k, k)] = np.asarray(v)
        elif k in _QC_PER_GENE:
            var_out[_QC_VAR_RENAME.get(k, k)] = np.asarray(v)
        else:
            raise AssertionError(f"unclassified QC metric {k!r}")

    if qc_vars or percent_top:
        import scipy.sparse as sp
        X = sp.csr_matrix(adata.layers[layer] if layer is not None else adata.X)
        total = np.asarray(obs_out["total_counts"], dtype=np.float64)
        for v in qc_vars:
            mask = np.asarray(adata.var[v]).astype(bool)
            sub = np.asarray(X[:, mask].sum(1)).ravel().astype(np.float64)
            obs_out[f"total_counts_{v}"] = sub
            with np.errstate(invalid="ignore", divide="ignore"):
                obs_out[f"pct_counts_{v}"] = 100.0 * sub / total   # NaN on an empty cell, as scanpy
        tops = sorted(percent_top or [])
        if tops and max(tops) > adata.n_vars:    # scanpy raises IndexError here; say why
            raise IndexError(
                f"percent_top={tuple(tops)} asks for more genes than this object has "
                f"({adata.n_vars}). Pass a smaller tuple, or percent_top=None to skip these "
                f"columns."
            )
        for n_top, vals in zip(tops, _percent_top(X, tops)):
            obs_out[f"pct_counts_in_top_{n_top}_genes"] = vals
    if log1p:
        # scanpy log1p-transforms exactly these: the two per-cell totals plus one per qc_var in
        # obs, and total/mean counts in var. It does NOT transform n_cells_by_counts.
        for base in ("total_counts", "n_genes_by_counts", *(f"total_counts_{v}" for v in qc_vars)):
            if base in obs_out:
                obs_out[f"log1p_{base}"] = np.log1p(np.asarray(obs_out[base], np.float64))
        for base in ("total_counts", "mean_counts"):
            if base in var_out:
                var_out[f"log1p_{base}"] = np.log1p(np.asarray(var_out[base], np.float64))

    if not inplace:
        return (pd.DataFrame(obs_out, index=adata.obs_names),
                pd.DataFrame(var_out, index=adata.var_names))
    for k, v in obs_out.items():
        adata.obs[k] = v
    for k, v in var_out.items():
        adata.var[k] = v
    return adata if copy else None


def _percent_top(X, ns):
    """Per-cell cumulative fraction (%) of counts in the top-N expressed genes, for each N in ``ns``."""
    if not ns:
        return []
    total = np.asarray(X.sum(1)).ravel().astype(np.float64)
    out = [np.zeros(X.shape[0]) for _ in ns]
    indptr, data = X.indptr, X.data
    for i in range(X.shape[0]):
        row = data[indptr[i]:indptr[i + 1]]
        t = total[i]
        if row.size == 0 or t <= 0:
            continue
        cs = np.cumsum(np.sort(row)[::-1])
        for j, n in enumerate(ns):
            out[j][i] = 100.0 * cs[min(n, row.size) - 1] / t
    return out


def _resolve_mask_var(adata, mask_var, use_highly_variable):
    """scanpy's ``_handle_mask_var``: unify ``mask_var`` with deprecated ``use_highly_variable``.

    Returns ``(stored, mask)`` — what to record in ``uns['pca']['params']['mask_var']``, and the
    boolean array to select with (``None`` meaning "all variables").

    The three-state default is the subtle part and is easy to get wrong: **omitting**
    ``mask_var`` selects ``highly_variable`` when that column exists, whereas passing
    ``mask_var=None`` explicitly means *all* variables. Hence the ``_EMPTY`` sentinel rather
    than ``None`` as the default.
    """
    import warnings

    if use_highly_variable is not None:
        hint = ('use_highly_variable=True can be called through mask_var="highly_variable". '
                "use_highly_variable=False can be called through mask_var=None")
        warnings.warn(
            f"Argument `use_highly_variable` is deprecated, consider using the mask "
            f"argument. {hint}", FutureWarning, stacklevel=3,
        )
        if mask_var is not _EMPTY:
            raise ValueError(f"These arguments are incompatible. {hint}")
        mask_var = "highly_variable" if use_highly_variable else None

    if mask_var is _EMPTY and "highly_variable" in adata.var.columns:
        mask_var = "highly_variable"
    if mask_var is _EMPTY or mask_var is None:
        return None, None

    if isinstance(mask_var, str):
        if mask_var not in adata.var.columns:
            raise ValueError(f"Did not find `adata.var[{mask_var!r}]`.")
        return mask_var, adata.var[mask_var].to_numpy().astype(bool)

    arr = np.asarray(mask_var)
    if arr.dtype != bool or arr.shape != (adata.n_vars,):
        raise ValueError(
            f"The mask must be a boolean array of length n_vars ({adata.n_vars}), a column "
            f"name in adata.var, or None; got shape {arr.shape} of dtype {arr.dtype}."
        )
    return arr, arr


def _explained_variance(x_pca, zero_center=True):
    """scanpy's ``uns['pca']['variance']`` — the eigenvalues, not the ratio.

    Equal to ``S**2 / (n - 1)``, which is exactly the column variance of the scores, so it can
    be recovered without threading another return value out of every solver. Anything that
    reconstructs loadings or draws an elbow on absolute variance reads this key.
    """
    return np.var(np.asarray(x_pca), axis=0, ddof=1 if zero_center else 0).astype(np.float64)


def _resolve_svd_solver(svd_solver, adata, layer, zero_center, n_comps, n_sel):
    """scanpy resolves ``svd_solver=None`` to ``arpack``, dense or sparse.

    We defaulted to ``randomized``, which is a different answer rather than a slower one: on
    pbmc3k the two agree to |corr| >= 0.99 only through PC15 and reach |corr| = 0.012 by PC50.
    The exact solvers need a dense matrix, so a sparse zero-centred input still takes the
    randomized path -- but with enough power iterations to converge (measured |corr| 0.998
    against arpack at n_iter=15/n_oversamples=40, against 0.012 at the sklearn defaults).
    """
    import scipy.sparse as sp
    explicit = svd_solver not in (None, "auto")
    if not explicit:
        X = adata.layers[layer] if layer is not None else adata.X
        svd_solver = "randomized" if (sp.issparse(X) and zero_center) else "arpack"
    # ARPACK needs k < min(n_obs, n_features); a narrow panel or a small HVG mask can ask for
    # as many components as it has columns. The dense exact solver gives the same answer, and
    # a matrix that small costs nothing to decompose outright.
    if svd_solver == "arpack" and n_comps >= min(adata.n_obs, n_sel):
        return "full"
    return svd_solver


def pca(adata, n_comps: int | None = None, layer=None, use_highly_variable: bool | None = None,
        zero_center: bool = True, svd_solver: str | None = None, random_state: int = 0,
        copy: bool = False, *, mask_var=_EMPTY):
    """PCA (``sc.pp.pca``); writes ``obsm['X_pca']``, ``varm['PCs']``, ``uns['pca']``.

    Sparse input → the sparse-aware randomized PCA (no densify).

    Args:
        mask_var: which variables to run on — a boolean array of length ``n_vars``, the name
            of a boolean column in ``adata.var``, or ``None`` for all variables. **Omitting**
            it uses ``adata.var['highly_variable']`` when that column exists; that is not the
            same as passing ``None``, which forces all variables.
        use_highly_variable: deprecated, as in scanpy ≥1.10 — use ``mask_var`` instead.
            ``True`` is ``mask_var="highly_variable"``, ``False`` is ``mask_var=None``.
            Passing both raises.
    """
    import scipy.sparse as sp

    from .decomposition import pca as _pca
    adata = _copy_or_reject(adata, copy, "pca")
    mask_param, mask = _resolve_mask_var(adata, mask_var, use_highly_variable)
    n_sel = int(mask.sum()) if mask is not None else adata.n_vars
    if n_comps is None:
        # scanpy: settings.N_PCS, or min_dim - 1 on a panel narrower than that. Hard-coding 50
        # returned one component too many for any object with fewer than 51 features.
        min_dim = min(adata.n_obs, n_sel)
        n_comps = min_dim - 1 if min_dim <= _N_PCS_DEFAULT else _N_PCS_DEFAULT
    svd_solver = _resolve_svd_solver(svd_solver, adata, layer, zero_center,
                                     n_comps, n_sel)
    params = {"zero_center": bool(zero_center),
              "use_highly_variable": mask_param is not None,
              "mask_var": mask_param}
    if layer is not None:
        params["layer"] = layer

    reader = _backed_reader(adata, layer)
    if reader is not None:                       # out-of-core: fused streaming covariance-eigh
        from .decomposition import pca_covariance_eigh_streaming
        H = int(mask.sum()) if mask is not None else adata.n_vars
        tf = _build_pca_transform(adata, mask)
        X_pca, comps, vr = pca_covariance_eigh_streaming(reader, tf, H, n_comps=n_comps)
        adata.obsm["X_pca"] = np.asarray(X_pca)
        # n_comps may have been clamped to the rank of the selection — size from the
        # components actually returned, not from what was asked for.
        pcs = np.zeros((adata.n_vars, np.asarray(comps).shape[0]), dtype=np.float32)
        pcs[mask if mask is not None else slice(None)] = np.asarray(comps).T
        adata.varm["PCs"] = pcs
        adata.uns["pca"] = {"params": params, "variance_ratio": np.asarray(vr),
                            "variance": _explained_variance(X_pca, zero_center)}
        return adata if copy else None

    X = adata.layers[layer] if layer is not None else adata.X
    sel = mask if mask is not None else np.ones(adata.n_vars, bool)
    Xsub = X[:, sel]
    inp = CSR.from_scipy(sp.csr_matrix(Xsub).astype(np.float32)) if sp.issparse(Xsub) and zero_center \
        else np.asarray(Xsub.todense() if sp.issparse(Xsub) else Xsub, dtype=np.float32)
    # A sparse zero-centred input cannot take an exact solver, so give the randomized one
    # enough power iterations to land on the same subspace instead of a nearby one.
    extra = ({"n_iter": 15, "n_oversamples": 40}
             if svd_solver == "randomized" and isinstance(inp, CSR) else {})
    X_pca, comps, vr = _pca(inp, n_comps=n_comps, solver=svd_solver, random_state=random_state,
                            zero_center=zero_center, **extra)
    adata.obsm["X_pca"] = np.asarray(X_pca)
    pcs = np.zeros((adata.n_vars, np.asarray(comps).shape[0]), dtype=np.float32)
    pcs[sel] = np.asarray(comps).T
    adata.varm["PCs"] = pcs
    adata.uns["pca"] = {"params": params, "variance_ratio": np.asarray(vr),
                        "variance": _explained_variance(X_pca, zero_center)}
    return adata if copy else None


_N_PCS_DEFAULT = 50            # scanpy's settings.N_PCS


def _choose_representation(adata, use_rep, n_pcs, random_state):
    """scanpy's ``_choose_representation``, including the PCA it computes for you.

    The part that is easy to miss: with ``use_rep=None`` and no ``X_pca`` present, scanpy does
    **not** fall back to raw ``.X`` unless the object is narrow — it runs a 50-component PCA
    first. Falling back to ``.X`` builds the graph in gene space, which on 2000 HVGs gave a
    neighbour-set overlap of 0.188 against scanpy, and on a sparse ``.X`` did not even fail
    cleanly (``np.asarray(csr, dtype=...)`` raises about a ragged sequence).
    """
    if use_rep is None and n_pcs == 0:
        return np.asarray(_dense(adata.X), dtype=np.float32), None
    if use_rep is None:
        if adata.n_vars > _N_PCS_DEFAULT:
            if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < (
                    n_pcs or _N_PCS_DEFAULT):
                pca(adata, n_comps=n_pcs or _N_PCS_DEFAULT, random_state=random_state)
            return np.asarray(adata.obsm["X_pca"], dtype=np.float32), "X_pca"
        return np.asarray(_dense(adata.X), dtype=np.float32), None
    if use_rep == "X":
        return np.asarray(_dense(adata.X), dtype=np.float32), "X"
    if use_rep not in adata.obsm:
        raise ValueError(f"Did not find {use_rep} in .obsm.keys(). "
                         f"You need to compute it first.")
    return np.asarray(adata.obsm[use_rep], dtype=np.float32), use_rep


def _dense(X):
    import scipy.sparse as sp
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def neighbors(adata, n_neighbors: int = 15, n_pcs: int | None = None, *, use_rep: str | None = None,
              metric: str = "euclidean", key_added: str | None = None,
              random_state: int = 0, copy: bool = False):
    """kNN graph (``sc.pp.neighbors``); writes ``obsp['distances']``/``['connectivities']``, ``uns['neighbors']``.

    Signature mirrors scanpy: ``n_pcs`` is positional after ``n_neighbors`` and ``use_rep`` is
    keyword-only, so ``sc.pp.neighbors(adata, 15, 40)`` truncates the representation to 40 PCs
    (previously that 40 bound ``use_rep`` and silently ran on raw ``.X``).

    With ``use_rep=None`` the representation follows scanpy's rule: ``X_pca`` if present,
    otherwise a freshly computed 50-component PCA when the object has more than 50 variables,
    otherwise ``.X``. ``key_added`` redirects all three output slots, and the consumers in
    ``msc.tl`` resolve it through ``neighbors_key=``.
    """
    from .neighbors import neighbors as _nb
    if metric != "euclidean":
        raise NotImplementedError(f"metric={metric!r} is not implemented (Euclidean only)")
    adata = adata.copy() if copy else adata
    rep, rep_key = _choose_representation(adata, use_rep, n_pcs, random_state)
    if n_pcs is not None:                            # scanpy truncates the rep to the first n_pcs
        rep = rep[:, :n_pcs]
    dist, conn = _nb(rep, n_neighbors=n_neighbors, random_state=random_state)

    ck = "connectivities" if key_added is None else f"{key_added}_connectivities"
    dk = "distances" if key_added is None else f"{key_added}_distances"
    adata.obsp[dk] = dist
    adata.obsp[ck] = conn
    params = {"n_neighbors": n_neighbors, "method": "umap",
              "random_state": random_state, "metric": metric}
    if use_rep is not None:            # scanpy records only what the caller asked for
        params["use_rep"] = use_rep
    if n_pcs is not None:
        params["n_pcs"] = n_pcs
    adata.uns["neighbors" if key_added is None else key_added] = {
        "connectivities_key": ck, "distances_key": dk, "params": params}
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
