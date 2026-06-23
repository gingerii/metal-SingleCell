"""AnnData ``tl`` namespace — drop-in mirror of ``scanpy.tl`` / ``rapids_singlecell.tl``.

AnnData in; results written to the slots scanpy uses (``obs``, ``obsm``, ``uns``) with
``copy`` semantics. Graph-based tools read ``adata.obsp['connectivities']`` (from ``pp.neighbors``).
"""

from __future__ import annotations

import numpy as np

from . import tools as _tl


def _conn(adata):
    if "connectivities" not in adata.obsp:
        raise ValueError("run msc.pp.neighbors first (no adata.obsp['connectivities'])")
    return adata.obsp["connectivities"]


def leiden(adata, resolution: float = 1.0, key_added: str = "leiden", random_state: int = 0,
           n_iterations: int = 2, backend: str = "igraph", variant: str = "sync",
           commit_prob: float = 0.9, copy: bool = False):
    """Leiden clustering (``sc.tl.leiden``); writes ``adata.obs[key_added]`` (categorical).

    ``backend="gpu"`` runs the Metal parallel Leiden; ``variant`` ("sync"|"colored") and
    ``commit_prob`` tune its convergence (GPU path only; ignored for the igraph backend).
    """
    import pandas as pd

    from .cluster import leiden as _leiden
    adata = adata.copy() if copy else adata
    lab = _leiden(_conn(adata), resolution=resolution, random_state=random_state,
                  n_iterations=n_iterations, backend=backend,
                  variant=variant, commit_prob=commit_prob)
    adata.obs[key_added] = pd.Categorical([str(x) for x in lab])
    adata.uns[key_added] = {"params": {"resolution": resolution, "n_iterations": n_iterations}}
    return adata if copy else None


def louvain(adata, resolution: float = 1.0, key_added: str = "louvain", random_state: int = 0,
            backend: str = "igraph", variant: str = "sync", commit_prob: float = 0.9,
            copy: bool = False):
    """Louvain clustering (``sc.tl.louvain``); writes ``adata.obs[key_added]`` (categorical).

    ``backend="gpu"`` runs the Metal parallel Louvain; ``variant`` ("sync"|"colored") and
    ``commit_prob`` tune its convergence (GPU path only).
    """
    import pandas as pd
    adata = adata.copy() if copy else adata
    if backend == "gpu":
        from .graph import Graph
        from .graph.louvain import louvain as _gpu
        lab = _gpu(Graph.from_scipy(_conn(adata)), resolution=resolution, random_state=random_state,
                   variant=variant, commit_prob=commit_prob)
    else:
        import igraph as ig
        coo = _conn(adata).tocoo(); up = coo.row < coo.col
        g = ig.Graph(n=adata.n_obs, edges=list(zip(coo.row[up].tolist(), coo.col[up].tolist())),
                     edge_attrs={"weight": coo.data[up]})
        vc = g.community_multilevel(weights="weight", resolution=resolution)
        lab = np.array(vc.membership)
    adata.obs[key_added] = pd.Categorical([str(x) for x in lab])
    return adata if copy else None


def umap(adata, min_dist: float = 0.5, spread: float = 1.0, n_components: int = 2,
         n_epochs: int | None = None, random_state: int = 0, copy: bool = False):
    """UMAP embedding (``sc.tl.umap``); writes ``adata.obsm['X_umap']``."""
    from .embedding import umap as _umap
    adata = adata.copy() if copy else adata
    adata.obsm["X_umap"] = _umap(_conn(adata), n_components=n_components, n_epochs=n_epochs,
                                 min_dist=min_dist, spread=spread, random_state=random_state)
    return adata if copy else None


def tsne(adata, use_rep: str = "X_pca", perplexity: float = 30.0, n_components: int = 2,
         random_state: int = 0, copy: bool = False):
    """t-SNE (``sc.tl.tsne``); writes ``adata.obsm['X_tsne']``."""
    adata = adata.copy() if copy else adata
    rep = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    adata.obsm["X_tsne"] = _tl.tsne(np.asarray(rep, dtype=np.float32), n_components=n_components,
                                    perplexity=perplexity, random_state=random_state)
    return adata if copy else None


def diffmap(adata, n_comps: int = 15, copy: bool = False):
    """Diffusion map (``sc.tl.diffmap``); writes ``obsm['X_diffmap']`` + ``uns['diffmap_evals']``."""
    adata = adata.copy() if copy else adata
    res = _tl.diffmap(_conn(adata), n_comps=n_comps)
    adata.obsm["X_diffmap"] = np.asarray(res["X_diffmap"])
    adata.uns["diffmap_evals"] = np.asarray(res["eigenvalues"])
    return adata if copy else None


def draw_graph(adata, layout: str = "fa", n_iter: int = 500, random_state: int = 0, copy: bool = False):
    """Force-directed layout (``sc.tl.draw_graph``); writes ``obsm[f'X_draw_graph_{layout}']``."""
    adata = adata.copy() if copy else adata
    adata.obsm[f"X_draw_graph_{layout}"] = _tl.draw_graph(_conn(adata), n_iter=n_iter,
                                                          random_state=random_state)
    return adata if copy else None


def rank_genes_groups(adata, groupby, method: str = "t-test", reference: str = "rest",
                      key_added: str = "rank_genes_groups", layer=None, copy: bool = False):
    """Marker genes per group (``sc.tl.rank_genes_groups``); writes scanpy-format ``adata.uns[key_added]``."""
    import scipy.sparse as sp
    adata = adata.copy() if copy else adata
    X = adata.layers[layer] if layer is not None else adata.X
    X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32)
    groups = adata.obs[groupby].to_numpy()
    rg = _tl.rank_genes_groups(X, groups, var_names=adata.var_names.to_numpy(),
                               method=method, reference=reference)
    cats = [str(c) for c in (adata.obs[groupby].cat.categories
                             if hasattr(adata.obs[groupby], "cat")
                             else sorted(np.unique(groups)))]
    cats = [c for c in cats if c in rg]
    ng = len(rg[cats[0]]["names"])

    def recarray(field, dtype):
        if rg[cats[0]].get(field) is None:
            return None
        a = np.empty(ng, dtype=[(c, dtype) for c in cats])
        for c in cats:
            a[c] = rg[c][field]
        return a

    names_dt = f"<U{max(len(str(x)) for x in adata.var_names)}"
    uns = {"params": {"groupby": groupby, "reference": reference, "method": method, "use_raw": False},
           "names": recarray("names", names_dt), "scores": recarray("scores", "f4"),
           "pvals": recarray("pvals", "f8"), "logfoldchanges": recarray("logfoldchanges", "f4")}
    adata.uns[key_added] = {k: v for k, v in uns.items() if v is not None}
    return adata if copy else None


def score_genes(adata, gene_list, score_name: str = "score", ctrl_size: int = 50,
                n_bins: int = 25, random_state: int = 0, copy: bool = False):
    """Gene-set score (``sc.tl.score_genes``); writes ``adata.obs[score_name]``."""
    import scipy.sparse as sp
    adata = adata.copy() if copy else adata
    X = np.asarray(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=np.float32)
    adata.obs[score_name] = _tl.score_genes(X, list(gene_list), adata.var_names.to_numpy(),
                                            ctrl_size=ctrl_size, n_bins=n_bins, random_state=random_state)
    return adata if copy else None


def score_genes_cell_cycle(adata, s_genes, g2m_genes, random_state: int = 0, copy: bool = False):
    """S/G2M scores + phase (``sc.tl.score_genes_cell_cycle``); writes ``obs['S_score'/'G2M_score'/'phase']``."""
    import scipy.sparse as sp
    adata = adata.copy() if copy else adata
    X = np.asarray(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=np.float32)
    res = _tl.score_genes_cell_cycle(X, list(s_genes), list(g2m_genes),
                                     adata.var_names.to_numpy(), random_state=random_state)
    adata.obs["S_score"] = res["S_score"]
    adata.obs["G2M_score"] = res["G2M_score"]
    import pandas as pd
    adata.obs["phase"] = pd.Categorical(res["phase"])
    return adata if copy else None


def embedding_density(adata, basis: str = "umap", groupby=None, key_added=None, copy: bool = False):
    """Per-cell density in an embedding (``sc.tl.embedding_density``); writes ``obs[f'{basis}_density']``."""
    adata = adata.copy() if copy else adata
    groups = adata.obs[groupby].to_numpy() if groupby is not None else None
    dens = _tl.embedding_density(np.asarray(adata.obsm[f"X_{basis}"]), groups=groups)
    adata.obs[key_added or f"{basis}_density"] = dens
    return adata if copy else None


def kmeans(adata, n_clusters: int = 8, use_rep: str = "X_pca", key_added: str = "kmeans",
           random_state: int = 0, copy: bool = False):
    """k-means on an embedding; writes ``adata.obs[key_added]`` (categorical)."""
    import pandas as pd
    adata = adata.copy() if copy else adata
    lab = _tl.kmeans(np.asarray(adata.obsm[use_rep], dtype=np.float32),
                     n_clusters=n_clusters, random_state=random_state)
    adata.obs[key_added] = pd.Categorical([str(x) for x in lab])
    return adata if copy else None
