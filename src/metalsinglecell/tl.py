"""AnnData ``tl`` namespace — drop-in mirror of ``scanpy.tl`` / ``rapids_singlecell.tl``.

AnnData in; results written to the slots scanpy uses (``obs``, ``obsm``, ``uns``) with
``copy`` semantics. Graph-based tools read ``adata.obsp['connectivities']`` (from ``pp.neighbors``).
"""

from __future__ import annotations

import numpy as np

from . import tools as _tl


def _conn(adata, neighbors_key: str = "neighbors"):
    """The connectivity graph, resolved through ``uns[neighbors_key]`` as scanpy does."""
    key = "connectivities"
    if neighbors_key in adata.uns:
        key = adata.uns[neighbors_key].get("connectivities_key", key)
    if key not in adata.obsp:
        raise ValueError(f"run msc.pp.neighbors first (no adata.obsp[{key!r}])")
    return adata.obsp[key]


def _benjamini_hochberg(pvals):
    """BH-adjusted p-values (scanpy's ``method='benjamini-hochberg'``), order-preserving."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(adj, 0.0, 1.0)


def _categorical(labels):
    """Cluster labels as a Categorical with scanpy's category ORDER.

    scanpy natsorts the categories; bare ``pd.Categorical`` sorts the strings, which agrees
    below 10 clusters and then diverges: ['0','1','10','11',...,'2',...]. Once it does,
    ``.cat.codes`` no longer equals the integer label, so palettes, legend order and anything
    indexing by code silently misalign.
    """
    import pandas as pd
    vals = np.asarray([str(x) for x in labels])
    return pd.Categorical(values=vals, categories=sorted(np.unique(vals), key=_natkey))


def _natkey(s):
    """Natural-order sort key — digit runs compare numerically. Avoids a `natsort` dependency
    for the one place we need it."""
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def leiden(adata, resolution: float = 1.0, key_added: str = "leiden", random_state: int = 0,
           n_iterations: int = 2, backend: str = "igraph", variant: str = "sync",
           commit_prob: float = 0.9, neighbors_key: str = "neighbors", copy: bool = False):
    """Leiden clustering (``sc.tl.leiden``); writes ``adata.obs[key_added]`` (categorical).

    ``backend="gpu"`` runs the Metal parallel Leiden; ``variant`` ("sync"|"colored") and
    ``commit_prob`` tune its convergence (GPU path only; ignored for the igraph backend).
    """
    import pandas as pd

    from .cluster import leiden as _leiden
    adata = adata.copy() if copy else adata
    lab = _leiden(_conn(adata, neighbors_key=neighbors_key), resolution=resolution,
                  random_state=random_state, n_iterations=n_iterations, backend=backend,
                  variant=variant, commit_prob=commit_prob)
    adata.obs[key_added] = _categorical(lab)
    adata.uns[key_added] = {"params": {"resolution": resolution, "random_state": random_state,
                                       "n_iterations": n_iterations}}
    return adata if copy else None


def louvain(adata, resolution: float = 1.0, key_added: str = "louvain", random_state: int = 0,
            backend: str = "igraph", variant: str = "sync", commit_prob: float = 0.9,
            neighbors_key: str = "neighbors", copy: bool = False):
    """Louvain clustering (``sc.tl.louvain``); writes ``adata.obs[key_added]`` (categorical).

    ``backend="gpu"`` runs the Metal parallel Louvain; ``variant`` ("sync"|"colored") and
    ``commit_prob`` tune its convergence (GPU path only).
    """
    import pandas as pd
    adata = adata.copy() if copy else adata
    if backend == "gpu":
        from .graph import Graph
        from .graph.louvain import louvain as _gpu
        lab = _gpu(Graph.from_scipy(_conn(adata, neighbors_key=neighbors_key)),
                   resolution=resolution, random_state=random_state,
                   variant=variant, commit_prob=commit_prob)
    else:
        import igraph as ig
        from .cluster import _seeded_igraph
        coo = _conn(adata, neighbors_key=neighbors_key).tocoo(); up = coo.row < coo.col
        g = ig.Graph(n=adata.n_obs, edges=list(zip(coo.row[up].tolist(), coo.col[up].tolist())),
                     edge_attrs={"weight": coo.data[up]})
        # random_state was declared and only ever read on the gpu branch, so the default path
        # was unseeded: three identical calls gave three answers (ARI 0.789).
        with _seeded_igraph(random_state):
            vc = g.community_multilevel(weights="weight", resolution=resolution)
        lab = np.array(vc.membership)
    adata.obs[key_added] = _categorical(lab)
    adata.uns[key_added] = {"params": {"resolution": resolution, "random_state": random_state}}
    return adata if copy else None


def umap(adata, min_dist: float = 0.5, spread: float = 1.0, n_components: int = 2,
         n_epochs: int | None = None, random_state: int = 0,
         init_pos: str | np.ndarray = "spectral", copy: bool = False, *,
         maxiter: int | None = None, alpha: float = 1.0, gamma: float = 1.0,
         negative_sample_rate: int = 5, a: float | None = None, b: float | None = None,
         key_added: str | None = None, neighbors_key: str = "neighbors"):
    """UMAP embedding (``sc.tl.umap``); writes ``adata.obsm['X_umap']``.

    Argument names follow ``scanpy.tl.umap`` so the namespaces stay swappable:
    ``init_pos`` (``"spectral"``, ``"random"``, or an ``(n_obs, n_components)`` array),
    ``alpha`` (learning rate), ``gamma`` (repulsion strength), ``maxiter`` (scanpy's name for
    the epoch count). ``n_epochs`` is kept as an alias of ``maxiter``; passing both raises.
    ``a``/``b`` override the curve fitted from ``min_dist``/``spread``.
    """
    from .embedding import umap as _umap
    adata = adata.copy() if copy else adata
    if maxiter is not None and n_epochs is not None and maxiter != n_epochs:
        raise ValueError("pass either `maxiter` (scanpy's name) or `n_epochs`, not both")
    Y = _umap(_conn(adata, neighbors_key=neighbors_key), n_components=n_components,
              n_epochs=maxiter if maxiter is not None else n_epochs,
              min_dist=min_dist, spread=spread, random_state=random_state, init=init_pos,
              learning_rate=alpha, gamma=gamma, negative_sample_rate=negative_sample_rate,
              a=a, b=b)
    if a is None or b is None:
        from ._vendor.mlx_vis.umap import UMAP as _MlxUMAP
        a, b = _MlxUMAP._find_ab_params(spread, min_dist)
    key = key_added or "umap"
    adata.obsm["X_umap" if key_added is None else key_added] = Y
    params = {"a": float(a), "b": float(b)}
    if random_state != 0:
        params["random_state"] = random_state
    adata.uns[key] = {"params": params}
    return adata if copy else None


def tsne(adata, use_rep: str | None = None, perplexity: float = 30.0, n_components: int = 2,
         early_exaggeration: float = 12.0, learning_rate: float = 1000.0,
         random_state: int = 0, key_added: str | None = None, copy: bool = False):
    """t-SNE (``sc.tl.tsne``); writes ``adata.obsm['X_tsne']``.

    ``use_rep`` follows scanpy's resolution rule and **raises** for a key that is not there.
    It used to fall back to ``.X`` silently, so a typo returned an embedding of the wrong
    thing. ``learning_rate`` defaults to scanpy's 1000, not the 200 hard-coded previously.
    """
    from .pp import _choose_representation
    adata = adata.copy() if copy else adata
    rep, _ = _choose_representation(adata, use_rep, None, random_state)
    key = key_added or "tsne"
    adata.obsm["X_tsne" if key_added is None else key_added] = _tl.tsne(
        np.asarray(rep, dtype=np.float32), n_components=n_components, perplexity=perplexity,
        learning_rate=learning_rate, random_state=random_state)
    adata.uns[key] = {"params": {"perplexity": perplexity,
                                 "early_exaggeration": early_exaggeration,
                                 "learning_rate": learning_rate, "metric": "euclidean"}}
    return adata if copy else None


def diffmap(adata, n_comps: int = 15, neighbors_key: str = "neighbors",
            random_state: int = 0, copy: bool = False):
    """Diffusion map (``sc.tl.diffmap``); writes ``obsm['X_diffmap']`` + ``uns['diffmap_evals']``."""
    if n_comps <= 2:
        raise ValueError("Provide any value greater than 2 for `n_comps`.")
    adata = adata.copy() if copy else adata
    res = _tl.diffmap(_conn(adata, neighbors_key=neighbors_key), n_comps=n_comps,
                      random_state=random_state)
    adata.obsm["X_diffmap"] = np.asarray(res["X_diffmap"], dtype=np.float32)
    adata.uns["diffmap_evals"] = np.asarray(res["eigenvalues"], dtype=np.float32)
    return adata if copy else None


_IGRAPH_LAYOUTS = {"fr": "fr", "drl": "drl", "kk": "kk", "grid_fr": "grid_fr",
                   "lgl": "lgl", "rt": "rt", "rt_circular": "rt_circular"}


def draw_graph(adata, layout: str = "fa", n_iter: int = 500, random_state: int = 0,
               neighbors_key: str = "neighbors", key_added_ext: str | None = None,
               copy: bool = False):
    """Force-directed layout (``sc.tl.draw_graph``); writes ``obsm[f'X_draw_graph_{layout}']``.

    ``layout`` used to name the output key and nothing else — every layout ran the same
    ForceAtlas2-style SGD, and an unknown name was accepted. ``"fa"`` is our GPU SGD; the
    igraph layouts scanpy offers are dispatched to igraph.
    """
    adata = adata.copy() if copy else adata
    conn = _conn(adata, neighbors_key=neighbors_key)
    if layout == "fa":
        pos = _tl.draw_graph(conn, n_iter=n_iter, random_state=random_state)
    elif layout in _IGRAPH_LAYOUTS:
        import igraph as ig
        from .cluster import _seeded_igraph
        coo = conn.tocoo(); up = coo.row < coo.col
        g = ig.Graph(n=adata.n_obs, edges=list(zip(coo.row[up].tolist(), coo.col[up].tolist())),
                     edge_attrs={"weight": coo.data[up]})
        with _seeded_igraph(random_state):
            pos = np.asarray(g.layout(_IGRAPH_LAYOUTS[layout]).coords, dtype=np.float32)
    else:
        raise ValueError(f"Provide a valid layout, one of "
                         f"{['fa', *sorted(_IGRAPH_LAYOUTS)]}, not {layout!r}.")
    adata.obsm[f"X_draw_graph_{key_added_ext or layout}"] = pos
    adata.uns["draw_graph"] = {"params": {"layout": layout, "random_state": random_state}}
    return adata if copy else None


def _expression_source(adata, layer, use_raw):
    """scanpy's ``_check_use_raw``: ``.raw`` when it exists and the caller did not say otherwise.

    Getting this wrong is quiet and consequential. On the canonical tutorial object ``.X`` is
    z-scaled to 1838 HVGs while ``.raw`` holds log-normalised counts for 13714 genes, so always
    reading ``.X`` ran the t-tests on scaled, negative-valued data over a seventh of the genes —
    top-10 marker overlap 0.6 against scanpy.
    """
    if use_raw is None:
        use_raw = adata.raw is not None and layer is None
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None")
        if layer is not None:
            raise ValueError("cannot use both use_raw=True and layer=")
        return adata.raw.X, np.asarray(adata.raw.var_names), True
    X = adata.layers[layer] if layer is not None else adata.X
    return X, np.asarray(adata.var_names), False


def rank_genes_groups(adata, groupby, method: str = "t-test", reference: str = "rest",
                      key_added: str = "rank_genes_groups", layer=None,
                      use_raw: bool | None = None, copy: bool = False):
    """Marker genes per group (``sc.tl.rank_genes_groups``); writes scanpy-format ``adata.uns[key_added]``.

    ``use_raw`` follows scanpy: ``.raw`` is used when it exists unless you pass ``False`` or a
    ``layer``. ``reference`` names either ``"rest"`` or one of the groups; naming a group
    compares against it and drops it from the output, as scanpy does.
    """
    import scipy.sparse as sp
    from .pp import _reject_backed
    _reject_backed(adata, "rank_genes_groups", layer)   # densifies .X — no streaming path
    adata = adata.copy() if copy else adata
    src, var_names, used_raw = _expression_source(adata, layer, use_raw)
    X = np.asarray(src.todense() if sp.issparse(src) else src, dtype=np.float32)
    groups = adata.obs[groupby].to_numpy()
    base = (adata.uns.get("log1p") or {}).get("base")
    rg = _tl.rank_genes_groups(X, groups, var_names=var_names, method=method,
                               reference=reference, log1p_base=base)
    cats = [str(c) for c in (adata.obs[groupby].cat.categories
                             if hasattr(adata.obs[groupby], "cat")
                             else sorted(np.unique(groups)))]
    cats = [c for c in cats if c in rg]
    ng = len(rg[cats[0]]["names"])

    for c in cats:                                   # BH-adjusted p-values (scanpy always emits these)
        if rg[c].get("pvals") is not None:
            rg[c]["pvals_adj"] = _benjamini_hochberg(rg[c]["pvals"])

    def recarray(field, dtype):
        if rg[cats[0]].get(field) is None:
            return None
        a = np.empty(ng, dtype=[(c, dtype) for c in cats])
        for c in cats:
            a[c] = rg[c][field]
        return a

    names_dt = f"<U{max(len(str(x)) for x in var_names)}"
    uns = {"params": {"groupby": groupby, "reference": reference, "method": method,
                      "use_raw": used_raw, "layer": layer,
                      "corr_method": "benjamini-hochberg"},
           "names": recarray("names", names_dt), "scores": recarray("scores", "f4"),
           "pvals": recarray("pvals", "f8"), "pvals_adj": recarray("pvals_adj", "f8"),
           "logfoldchanges": recarray("logfoldchanges", "f4")}
    adata.uns[key_added] = {k: v for k, v in uns.items() if v is not None}
    return adata if copy else None


def score_genes(adata, gene_list, score_name: str = "score", ctrl_size: int = 50,
                n_bins: int = 25, random_state: int = 0, copy: bool = False):
    """Gene-set score (``sc.tl.score_genes``); writes ``adata.obs[score_name]``."""
    import scipy.sparse as sp
    from .pp import _reject_backed
    _reject_backed(adata, "score_genes")
    adata = adata.copy() if copy else adata
    X = np.asarray(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=np.float32)
    adata.obs[score_name] = _tl.score_genes(X, list(gene_list), adata.var_names.to_numpy(),
                                            ctrl_size=ctrl_size, n_bins=n_bins, random_state=random_state)
    return adata if copy else None


def score_genes_cell_cycle(adata, s_genes, g2m_genes, random_state: int = 0, copy: bool = False):
    """S/G2M scores + phase (``sc.tl.score_genes_cell_cycle``); writes ``obs['S_score'/'G2M_score'/'phase']``."""
    import scipy.sparse as sp
    from .pp import _reject_backed
    _reject_backed(adata, "score_genes_cell_cycle")
    adata = adata.copy() if copy else adata
    X = np.asarray(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=np.float32)
    res = _tl.score_genes_cell_cycle(X, list(s_genes), list(g2m_genes),
                                     adata.var_names.to_numpy(), random_state=random_state)
    adata.obs["S_score"] = res["S_score"]
    adata.obs["G2M_score"] = res["G2M_score"]
    import pandas as pd
    adata.obs["phase"] = pd.Categorical(res["phase"])
    return adata if copy else None


def embedding_density(adata, basis: str = "umap", groupby=None, key_added=None,
                      components=None, copy: bool = False):
    """Per-cell density in an embedding (``sc.tl.embedding_density``).

    Writes ``obs[f'{basis}_density_{groupby}']`` (or ``obs[f'{basis}_density']`` with no
    ``groupby``) plus ``uns[f'{key}_params']`` — scanpy's plotter requires both and reads the
    params to label the axes.

    The KDE runs over **two** components, as scanpy's does. Handing the estimator every column
    of a wide basis is not a slower version of the same thing: on a 30-dimensional PCA basis
    the output collapsed to a numerically-zero range (max 1.04e-11) that still plots as a valid
    density, and on diffmap it anticorrelated with scanpy's because the trivial first
    eigenvector was included.
    """
    import pandas as pd
    adata = adata.copy() if copy else adata
    basis = basis.lower()
    if basis == "fa":
        basis = "draw_graph_fa"
    rep_key = f"X_{basis}"
    if rep_key not in adata.obsm:
        raise KeyError(f"no adata.obsm[{rep_key!r}] — compute the embedding first")

    comps = [1, 2] if components is None else [int(c) for c in
                                               (components.split(",") if isinstance(components, str)
                                                else np.ravel([components]))]
    if len(comps) != 2:
        raise ValueError("Please specify exactly 2 components, or `None`.")
    if basis == "diffmap":
        comps = [c + 1 for c in comps]           # scanpy skips the trivial first eigenvector
    X = np.asarray(adata.obsm[rep_key])[:, [c - 1 for c in comps]]

    groups = None
    if groupby is not None:
        if not isinstance(adata.obs[groupby].dtype, pd.CategoricalDtype):
            raise ValueError(f"Could not find categorical adata.obs[{groupby!r}]")
        groups = adata.obs[groupby].to_numpy()

    key = key_added or (f"{basis}_density_{groupby}" if groupby is not None
                        else f"{basis}_density")
    adata.obs[key] = _tl.embedding_density(X, groups=groups)
    adata.uns[f"{key}_params"] = {"covariate": groupby,
                                  "components": [c - 1 for c in comps]}
    return adata if copy else None


def kmeans(adata, n_clusters: int = 8, use_rep: str = "X_pca", key_added: str = "kmeans",
           random_state: int = 0, copy: bool = False):
    """k-means on an embedding; writes ``adata.obs[key_added]`` (categorical)."""
    import pandas as pd
    adata = adata.copy() if copy else adata
    lab = _tl.kmeans(np.asarray(adata.obsm[use_rep], dtype=np.float32),
                     n_clusters=n_clusters, random_state=random_state)
    adata.obs[key_added] = _categorical(lab)
    return adata if copy else None
