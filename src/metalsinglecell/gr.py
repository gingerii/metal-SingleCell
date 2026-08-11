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


def _coords(adata, spatial_key="spatial"):
    """Working (fp32, for the GPU) and exact (fp64) views of the coordinates.

    Distances are reported from the fp64 view. Visium/Xenium coordinates run to ~10^4-10^5, so
    an fp32 round trip is visible in the fourth decimal of an edge length — small, but enough
    to fail an element-wise comparison against a CPU reference for no good reason.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"no adata.obsm[{spatial_key!r}] with spatial coordinates")
    exact = np.ascontiguousarray(adata.obsm[spatial_key], dtype=np.float64)
    return np.ascontiguousarray(exact, dtype=np.float32), exact


def _edge_lengths(exact, rows, cols):
    diff = exact[rows] - exact[cols]
    return np.sqrt((diff * diff).sum(1))


def _transform_adj(adj, transform):
    """squidpy's ``TransformPostprocessor``: spectral ``D^-1/2 A D^-1/2`` or row/col cosine."""
    import scipy.sparse as sp
    if transform in (None, "none"):
        return adj
    if transform == "spectral":
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
        D = sp.diags(inv)
        return (D @ adj @ D).tocsr()
    if transform == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        return sp.csr_matrix(cosine_similarity(adj, dense_output=False))
    raise NotImplementedError(f"transform={transform!r} is not implemented "
                              f"(expected None, 'spectral' or 'cosine')")


def _csr_rows(M):
    return np.repeat(np.arange(M.shape[0], dtype=np.int64), np.diff(M.indptr))


def _drop_diag(M):
    """Remove the diagonal from a CSR. ``scipy``'s ``setdiag(0)`` needs a LIL round trip, which
    costs 0.85 s on a 500k graph against 0.02 s for masking the existing non-zeros."""
    import scipy.sparse as sp
    keep = M.indices != _csr_rows(M)
    counts = np.bincount(_csr_rows(M)[keep], minlength=M.shape[0])
    indptr = np.zeros(M.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return sp.csr_matrix((M.data[keep], M.indices[keep], indptr), shape=M.shape)


def _set_diag(M):
    """Put 1.0 on the diagonal of a 0/1 CSR (``maximum`` with I: 0.007 s vs 1.5 s via LIL)."""
    import scipy.sparse as sp
    return M.maximum(sp.identity(M.shape[0], format="csr", dtype=M.dtype)).tocsr()


def _write_graph(adata, adj, dst, params, key_added, transform, set_diag, percentile, copy):
    """Shared tail of every builder: percentile prune → set_diag → transform → write slots."""
    import scipy.sparse as sp
    adj, dst = adj.tocsr(), dst.tocsr()
    if percentile is not None and dst.nnz:                    # squidpy prunes on the DISTANCES
        keep = dst.data <= np.percentile(dst.data, percentile)
        adj = sp.csr_matrix((adj.data * keep, adj.indices, adj.indptr), shape=adj.shape)
        dst = sp.csr_matrix((dst.data * keep, dst.indices, dst.indptr), shape=dst.shape)
        adj.eliminate_zeros(); dst.eliminate_zeros()
    if set_diag:
        adj = _set_diag(adj)
    dst = _drop_diag(dst)
    dst.eliminate_zeros()
    adj = _transform_adj(adj, transform)

    ck, dk = f"{key_added}_connectivities", f"{key_added}_distances"
    adata.obsp[ck] = adj
    adata.obsp[dk] = dst
    adata.uns[f"{key_added}_neighbors"] = {
        "connectivities_key": ck, "distances_key": dk,
        "params": {**params, "transform": transform},
    }
    return adata if copy else None


_TIE_PAD = 4        # extra candidates fetched so ties resolve by index, not by arrival order


def _knn_edges(coords, n_neighs):
    """Directed k-NN edge list on the GPU grid index (self excluded), as sklearn returns.

    Over-fetches ``_TIE_PAD`` candidates and keeps the ``n_neighs`` smallest by
    ``(distance, index)``. On a lattice, exact ties are common — every Visium spot has six
    equidistant neighbours and the second shell ties in pairs — and without the extra
    candidates the winner is whichever the kernel happened to reach first. Padding makes the
    choice deterministic and reproducible (measured to converge by 4: pad 4 and pad 8 select
    the same set). It does NOT reproduce sklearn's pick on a tie, whose order falls out of its
    tree traversal; on Visium that leaves ~0.5% of rows choosing a different — equally
    correct — equidistant neighbour.
    """
    from .neighbors import _knn_grid
    n = coords.shape[0]
    want = min(n, n_neighs + 1 + _TIE_PAD)
    idx, dist = _knn_grid(coords, want)                       # self-inclusive, self first
    cand, cd = idx[:, 1:], dist[:, 1:]
    order = np.lexsort((cand, cd), axis=1)
    keep = np.take_along_axis(cand, order, axis=1)[:, :n_neighs]
    kept_d = np.take_along_axis(cd, order, axis=1)[:, :n_neighs]
    rows = np.repeat(np.arange(n, dtype=np.int64), n_neighs)
    return rows, keep.ravel().astype(np.int64), kept_d.ravel().astype(np.float32)


def spatial_neighbors_knn(adata, *, n_neighs: int = 6, spatial_key: str = "spatial",
                          percentile: float | None = None, transform: str | None = None,
                          set_diag: bool = False, key_added: str = "spatial",
                          copy: bool = False):
    """k-NN spatial graph (``sq.gr.spatial_neighbors_knn``), on the Metal grid index.

    Directed: ``n_neighs`` edges per observation, self excluded, exactly as squidpy's
    sklearn-backed builder. Connectivities are binary, distances Euclidean.
    """
    import scipy.sparse as sp
    adata = adata.copy() if copy else adata
    coords, exact = _coords(adata, spatial_key)
    n = coords.shape[0]
    rows, cols, _ = _knn_edges(coords, n_neighs)
    dist = _edge_lengths(exact, rows, cols)
    shape, indptr = (n, n), np.arange(n + 1, dtype=np.int64) * n_neighs
    adj = sp.csr_matrix((np.ones(dist.size, np.float32), cols, indptr), shape=shape)
    dst = sp.csr_matrix((dist, cols, indptr), shape=shape)
    return _write_graph(adata, adj, dst,
                        {"coord_type": "generic", "n_neighbors": n_neighs},
                        key_added, transform, set_diag, percentile, copy)


def spatial_neighbors_radius(adata, *, radius, spatial_key: str = "spatial",
                             percentile: float | None = None, transform: str | None = None,
                             set_diag: bool = False, key_added: str = "spatial",
                             copy: bool = False):
    """Fixed-radius spatial graph (``sq.gr.spatial_neighbors_radius``), on a Metal kernel.

    ``radius`` is a maximum distance, or a ``(min, max)`` interval. Every pair within it is
    connected, so the graph is symmetric and degrees vary with local density.
    """
    import scipy.sparse as sp
    from .neighbors import _radius_grid
    adata = adata.copy() if copy else adata
    coords, exact = _coords(adata, spatial_key)
    n = coords.shape[0]
    lo, hi = (0.0, float(radius)) if np.isscalar(radius) else (float(min(radius)),
                                                               float(max(radius)))
    indptr, cols, _ = _radius_grid(coords, hi, lo)
    dist = _edge_lengths(exact, np.repeat(np.arange(n), np.diff(indptr)), cols)
    adj = sp.csr_matrix((np.ones(dist.size, np.float32), cols, indptr), shape=(n, n))
    dst = sp.csr_matrix((dist, cols, indptr), shape=(n, n))
    return _write_graph(adata, adj, dst,
                        {"coord_type": "generic", "radius": radius},
                        key_added, transform, set_diag, percentile, copy)


def spatial_neighbors_grid(adata, *, n_neighs: int = 6, n_rings: int = 1,
                           delaunay: bool = False, spatial_key: str = "spatial",
                           transform: str | None = None, set_diag: bool = False,
                           key_added: str = "spatial", copy: bool = False):
    """Lattice spatial graph (``sq.gr.spatial_neighbors_grid``) — the Visium/Stereo-seq mode.

    k-NN on the lattice, then squidpy's median-distance correction (drop edges longer than
    ``1.3 × median``), which is what leaves boundary spots with fewer than ``n_neighs``
    neighbours. With ``n_rings > 1`` the graph is expanded by repeated multiplication and the
    **distances hold the ring index, not a length** — squidpy's convention, kept here.
    """
    import scipy.sparse as sp
    adata = adata.copy() if copy else adata
    coords, _exact = _coords(adata, spatial_key)
    n = coords.shape[0]

    if delaunay:
        adj = _delaunay_adjacency(coords)
    else:
        rows, cols, dist = _knn_edges(coords, n_neighs)
        keep = dist < np.median(dist) * 1.3                    # squidpy's lattice correction
        adj = sp.csr_matrix((np.ones(int(keep.sum()), np.float32),
                             (rows[keep], cols[keep])), shape=(n, n))

    if n_rings > 1:
        base = _set_diag(adj)
        res = walk = base
        for i in range(n_rings - 1):
            walk = (walk @ base).tocsr()
            # drop entries already reached in an earlier ring; masking beats `walk[res.nonzero()]
            # = 0` through LIL by ~20x (0.045s vs 0.90s at 500k)
            walk = (walk - walk.multiply(res.astype(bool))).tocsr()
            walk.eliminate_zeros()
            walk.data[:] = i + 2.0
            res = (res + walk).tocsr()
        adj = _set_diag(res) if set_diag else _drop_diag(res)
        adj.eliminate_zeros()
        dst = adj.copy()
        adj = adj.copy(); adj.data[:] = 1.0
    else:
        if set_diag:
            adj = _set_diag(adj)
        dst = adj.copy()

    return _write_graph(adata, adj, dst,
                        {"coord_type": "grid", "n_neighbors": n_neighs, "n_rings": n_rings,
                         "delaunay": delaunay},
                        key_added, transform, set_diag, None, copy)


def _delaunay_adjacency(coords):
    """Symmetric adjacency of the Delaunay triangulation (Qhull; see the note in the wrapper)."""
    import scipy.sparse as sp
    from scipy.spatial import Delaunay
    tri = Delaunay(np.asarray(coords, dtype=np.float64))
    indptr, indices = tri.vertex_neighbor_vertices
    n = coords.shape[0]
    return sp.csr_matrix((np.ones_like(indices, dtype=np.float32), indices, indptr), shape=(n, n))


def spatial_neighbors_delaunay(adata, *, radius=None, spatial_key: str = "spatial",
                               percentile: float | None = None, transform: str | None = None,
                               set_diag: bool = False, key_added: str = "spatial",
                               copy: bool = False):
    """Delaunay spatial graph (``sq.gr.spatial_neighbors_delaunay``).

    ``radius`` prunes the triangulation afterwards — a scalar ``r`` means ``(0, r)``, a tuple
    is an interval, ``None`` keeps every edge. The triangulation itself is unchanged by it.

    Unlike the other three modes the triangulation itself is **not** GPU-accelerated here: it
    runs on Qhull (``scipy.spatial.Delaunay``), the same library squidpy uses, and only the
    edge lengths and radius pruning around it are ours. Expect parity, not a speedup.

    That is a "not yet", not a "cannot". GPU Delaunay is a solved problem in the literature —
    GPU-DT and gDel2D build a digital Voronoi diagram by jump flooding, dualise it, then
    repair the result by parallel flipping, and report ~10x over Triangle and ~6x over CGAL.
    Two things make it a project rather than a patch. The jump-flooding stage alone does not
    yield a valid triangulation (a digital Voronoi region can be disconnected, so its dual can
    contain duplicated and intersecting triangles); the flipping repair that fixes this is the
    bulk of the algorithm. And it needs adaptive-exact orientation/incircle predicates, which
    matter more here than in graphics: a Visium slide is a regular lattice, so cocircular
    points are the common case rather than a corner case.
    """
    import scipy.sparse as sp
    adata = adata.copy() if copy else adata
    coords, exact = _coords(adata, spatial_key)
    n = coords.shape[0]
    adj = _delaunay_adjacency(exact).tocoo()
    dist = _edge_lengths(exact, adj.row, adj.col)
    stored_radius = radius
    if radius is not None:
        lo, hi = ((0.0, float(radius)) if np.isscalar(radius)
                  else (float(min(radius)), float(max(radius))))
        stored_radius = (lo, hi)              # squidpy normalises a scalar to an interval
        keep = (dist >= lo) & (dist <= hi)
        adj_row, adj_col, dist = adj.row[keep], adj.col[keep], dist[keep]
    else:
        adj_row, adj_col = adj.row, adj.col
    shape = (n, n)
    A = sp.csr_matrix((np.ones(dist.size, np.float32), (adj_row, adj_col)), shape=shape)
    D = sp.csr_matrix((dist, (adj_row, adj_col)), shape=shape)
    return _write_graph(adata, A, D,
                        {"coord_type": "generic", "radius": stored_radius},
                        key_added, transform, set_diag, percentile, copy)


def spatial_neighbors(adata, n_neighs: int = 6, coord_type: str | None = None,
                      n_rings: int = 1, delaunay: bool = False, radius=None,
                      spatial_key: str = "spatial", transform: str | None = None,
                      set_diag: bool = False, percentile: float | None = None,
                      key_added: str = "spatial", copy: bool = False):
    """Deprecated (``sq.gr.spatial_neighbors``) — dispatches to the mode-specific builders.

    squidpy deprecated this in 1.7 and removes it in 1.9; use ``spatial_neighbors_knn`` /
    ``_radius`` / ``_delaunay`` / ``_grid`` directly.

    Until 0.1.1 this accepted ``coord_type`` and ignored it, always returning a generic k-NN
    graph — so ``coord_type='grid'`` silently produced the wrong graph on Visium. It now
    dispatches for real, and, like squidpy, infers ``'grid'`` when ``uns['spatial']`` is
    present and ``'generic'`` otherwise.
    """
    import warnings
    warnings.warn(
        "gr.spatial_neighbors is deprecated (squidpy removes it in 1.9). Use "
        "spatial_neighbors_knn / spatial_neighbors_radius / spatial_neighbors_delaunay / "
        "spatial_neighbors_grid.", FutureWarning, stacklevel=2)
    if coord_type is None:
        coord_type = "grid" if "spatial" in adata.uns else "generic"
    common = dict(spatial_key=spatial_key, transform=transform, set_diag=set_diag,
                  key_added=key_added, copy=copy)
    if coord_type == "grid":
        return spatial_neighbors_grid(adata, n_neighs=n_neighs, n_rings=n_rings,
                                      delaunay=delaunay, **common)
    if coord_type != "generic":
        raise ValueError(f"coord_type must be 'grid', 'generic' or None, got {coord_type!r}")
    if delaunay:
        return spatial_neighbors_delaunay(adata, radius=radius, percentile=percentile, **common)
    if radius is not None:
        return spatial_neighbors_radius(adata, radius=radius, percentile=percentile, **common)
    return spatial_neighbors_knn(adata, n_neighs=n_neighs, percentile=percentile, **common)


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
