"""AnnData ``gr`` namespace — drop-in mirror of ``squidpy.gr``.

Spatial graph functions: read ``adata.obsm['spatial']`` / ``adata.obs[cluster_key]`` /
``adata.obsp['spatial_connectivities']`` and write results to the slots squidpy uses
(``obsp['spatial_connectivities']``, ``uns['moranI']``/``['gearyC']``,
``uns[f'{cluster_key}_co_occurrence']``, ``uns[f'{cluster_key}_ligrec']``, ``obs['niche']``).
So ``sq.gr`` pipelines work by swapping ``sq.gr`` → ``msc.gr``.
"""

from __future__ import annotations

import numpy as np

from . import spatial as _gr


def spatial_neighbors(adata, n_neighs: int = 6, coord_type: str = "generic", copy: bool = False):
    """Spatial graph from coordinates (``sq.gr.spatial_neighbors``); writes ``obsp['spatial_*']``."""
    adata = adata.copy() if copy else adata
    A = _gr.spatial_neighbors(np.asarray(adata.obsm["spatial"], dtype=np.float32), n_neighs=n_neighs)
    adata.obsp["spatial_connectivities"] = A
    adata.obsp["spatial_distances"] = A  # binary graph; squidpy keeps a parallel distances key
    adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_connectivities",
                                      "distances_key": "spatial_distances",
                                      "params": {"n_neighbors": n_neighs, "coord_type": coord_type}}
    return adata if copy else None


def spatial_autocorr(adata, mode: str = "moran", genes=None, n_perms: int | None = 100,
                     connectivity_key: str = "spatial_connectivities", layer=None,
                     seed: int = 0, copy: bool = False):
    """Moran's I / Geary's C per gene (``sq.gr.spatial_autocorr``); writes ``uns['moranI']``/``['gearyC']``."""
    import pandas as pd
    import scipy.sparse as sp
    adata = adata.copy() if copy else adata
    if genes is None:
        genes = (adata.var_names[adata.var["highly_variable"].to_numpy()]
                 if "highly_variable" in adata.var else adata.var_names).tolist()
    gi = [adata.var_names.get_loc(g) for g in genes]
    X = adata.layers[layer] if layer is not None else adata.X
    Xg = np.asarray(X[:, gi].todense() if sp.issparse(X) else X[:, gi], dtype=np.float32)
    out = _gr.spatial_autocorr(Xg, adata.obsp[connectivity_key], mode=mode,
                               n_perms=n_perms or 0, random_state=seed)
    stat = "I" if mode == "moran" else "C"
    df = pd.DataFrame({stat: out[mode], "pval_sim": out["pval"]}, index=genes).sort_values(stat, ascending=False)
    adata.uns["moranI" if mode == "moran" else "gearyC"] = df
    return adata if copy else None


def co_occurrence(adata, cluster_key, interval: int = 50, copy: bool = False):
    """Cluster co-occurrence vs distance (``sq.gr.co_occurrence``); writes ``uns[f'{cluster_key}_co_occurrence']``."""
    adata = adata.copy() if copy else adata
    res = _gr.co_occurrence(np.asarray(adata.obsm["spatial"], dtype=np.float32),
                            adata.obs[cluster_key].to_numpy(), n_intervals=interval)
    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": res["occ"], "interval": res["interval"]}
    return adata if copy else None


def ligrec(adata, cluster_key, interactions, n_perms: int = 100, seed: int = 0,
           key_added: str | None = None, copy: bool = False):
    """Ligand-receptor permutation test (``sq.gr.ligrec``); writes ``uns[key_added]`` (means/pvalues).

    ``interactions`` is a list of ``(ligand, receptor)`` gene-symbol pairs.
    """
    adata = adata.copy() if copy else adata
    res = _gr.ligrec(adata.X, adata.obs[cluster_key].to_numpy(), list(interactions),
                     adata.var_names.to_numpy(), n_perms=n_perms, random_state=seed)
    adata.uns[key_added or f"{cluster_key}_ligrec"] = {
        "means": res["means"], "pvalues": res["pvalues"],
        "categories": res["categories"], "interactions": res["lr_pairs"]}
    return adata if copy else None


def calculate_niche(adata, cluster_key, n_niches: int = 10,
                    connectivity_key: str = "spatial_connectivities",
                    key_added: str = "niche", random_state: int = 0, copy: bool = False):
    """Spatial niches from neighborhood composition (``sq.gr.calculate_niche``); writes ``obs[key_added]``."""
    import pandas as pd
    adata = adata.copy() if copy else adata
    res = _gr.calculate_niche(adata.obsp[connectivity_key], adata.obs[cluster_key].to_numpy(),
                              n_niches=n_niches, random_state=random_state)
    adata.obs[key_added] = pd.Categorical([str(x) for x in res["niche"]])
    return adata if copy else None
