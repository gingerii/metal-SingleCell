"""Nearest-neighbor graph on the PCA embedding (scanpy ``sc.pp.neighbors``).

The expensive part — all-pairs distances + k-NN selection on the 50-dim embedding
— runs on the Metal GPU (MLX): squared Euclidean distances via a single matmul,
then a top-k selection. The fuzzy connectivity graph is built with UMAP's
``fuzzy_simplicial_set`` (the same routine scanpy calls), so graph semantics match.

Brute-force k-NN is O(n^2) in memory/time; it's a genuine GPU win up to tens of
thousands of cells, but for very large cohorts an approximate method (pynndescent)
is asymptotically better. See the benchmark for the crossover.
"""

from __future__ import annotations

import numpy as np


def _knn_gpu(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN (incl. self) via brute-force squared-Euclidean on the GPU.

    Returns ``(knn_indices, knn_dists)`` of shape (n, k), self first.
    """
    import mlx.core as mx

    Xg = mx.array(X.astype(np.float32))
    Xg_h = Xg.astype(mx.float16)                          # fp16 distances (~1.27× incl argpartition)
    sq_h = mx.sum(Xg_h * Xg_h, axis=1)
    n = X.shape[0]

    # Tile over query rows so we never materialize the full n×n distance matrix
    # (that OOMs the GPU past ~30k cells). Each tile is block×n. The whole distance +
    # argpartition runs in fp16 — kept fp16 throughout (an fp32 cast-back of the large
    # block×n matrix would negate the gain) and used only to RANK neighbors, which fp16
    # preserves (recall ~0.99). The k selected distances are then recomputed exactly in
    # fp32 from the originals (so returned distances are full precision).
    block = max(1, min(n, 16_000_000 // max(n, 1)))      # cap tile at ~16M entries
    Xnp = np.asarray(X, dtype=np.float32)
    idx_parts, d2_parts = [], []
    for s in range(0, n, block):
        D2 = mx.maximum(sq_h[s:s + block][:, None] + sq_h[None, :]
                        - 2.0 * (Xg_h[s:s + block] @ Xg_h.T), 0.0)        # all fp16
        bidx = mx.argpartition(D2, kth=k, axis=1)[:, :k]
        mx.eval(bidx)
        bidx = np.asarray(bidx)
        diff = Xnp[s:s + block][:, None, :] - Xnp[bidx]   # exact fp32 distances for the k picks
        bd2 = np.einsum("ijk,ijk->ij", diff, diff)
        idx_parts.append(bidx); d2_parts.append(bd2)
    knn_indices = np.concatenate(idx_parts, axis=0)
    d2 = np.concatenate(d2_parts, axis=0)

    order = np.argsort(d2, axis=1)                        # sort k by distance, self first
    knn_indices = np.take_along_axis(knn_indices, order, axis=1)
    knn_dists = np.sqrt(np.take_along_axis(d2, order, axis=1))
    return knn_indices, knn_dists


def bbknn(X_pca: np.ndarray, batch, neighbors_within_batch: int = 3,
          random_state: int = 0):
    """Batch-balanced kNN (rapids-singlecell/scanpy ``pp.bbknn``).

    For each cell, find ``neighbors_within_batch`` nearest neighbors *within each
    batch* (GPU brute-force per batch), then build one fuzzy connectivity graph
    over the combined neighbor set — this balances neighbors across batches so the
    graph mixes them. Returns ``(distances, connectivities)`` scipy CSR matrices.
    """
    import mlx.core as mx
    import scipy.sparse as sp
    from umap.umap_ import fuzzy_simplicial_set

    X = np.asarray(X_pca, dtype=np.float32)
    n = X.shape[0]
    batch = np.asarray(batch)
    cats = np.unique(batch)
    k = neighbors_within_batch

    Xg = mx.array(X)
    xsq = mx.sum(Xg * Xg, axis=1)
    idx_parts, dist_parts = [], []
    for b in cats:
        bidx = np.flatnonzero(batch == b)
        Xb = Xg[mx.array(bidx.astype(np.int32))]
        xb_sq = mx.sum(Xb * Xb, axis=1)
        kk = min(k, bidx.size)
        # Per-batch kNN of ALL cells against this batch's cells (GPU brute-force). Tile
        # the query rows so the n×|batch| distance matrix never blows memory (cap ~256M
        # entries). NB this is O(n·|batch|) per batch; bbknn is kNN-workload-bound, where
        # the package's approximate CPU kNN (cKDTree/annoy) is competitive — the GPU does
        # not beat it (a workload limit, like the regular neighbors). Validated equal mixing.
        tile = max(1, 256_000_000 // max(bidx.size, 1))
        loc_b = np.empty((n, kk), dtype=np.int64)
        d_b = np.empty((n, kk), dtype=np.float32)
        for s in range(0, n, tile):
            e = min(s + tile, n)
            D2 = mx.maximum(xsq[s:e][:, None] + xb_sq[None, :]
                            - 2.0 * (Xg[s:e] @ Xb.T), 0.0)
            loc = mx.argpartition(D2, kth=kk, axis=1)[:, :kk]
            mx.eval(loc, D2)
            loc = np.asarray(loc)
            d_b[s:e] = np.sqrt(np.maximum(np.take_along_axis(np.asarray(D2), loc, axis=1), 0.0))
            loc_b[s:e] = loc
        idx_parts.append(bidx[loc_b])                    # map local -> global
        dist_parts.append(d_b)

    knn_indices = np.concatenate(idx_parts, axis=1)
    knn_dists = np.concatenate(dist_parts, axis=1)
    order = np.argsort(knn_dists, axis=1)                # sort combined neighbors by distance
    knn_indices = np.take_along_axis(knn_indices, order, axis=1)
    knn_dists = np.take_along_axis(knn_dists, order, axis=1)

    n_neighbors = knn_indices.shape[1]
    conn = fuzzy_simplicial_set(
        sp.coo_matrix((n, 1)), n_neighbors, random_state, metric="euclidean",
        knn_indices=knn_indices, knn_dists=knn_dists,
        set_op_mix_ratio=1.0, local_connectivity=1.0)
    if isinstance(conn, tuple):
        conn = conn[0]
    rows = np.repeat(np.arange(n), n_neighbors)
    distances = sp.csr_matrix((knn_dists.ravel(), (rows, knn_indices.ravel())), shape=(n, n))
    return distances, conn.tocsr()


def _knn_descent(X, k, n_iters: int = 8, seed: int = 0, tile: int = 20000):
    """Approximate k-NN via NN-descent (GPU distances, tiled). Returns (idx, dist).

    Brute-force k-NN is O(n²) and loses to tree methods past ~30k cells. NN-descent
    starts from random neighbors and refines using neighbors-of-neighbors — O(n·k²)
    per iteration — converging to a high-recall graph. The candidate distance
    computation (gather + norm) runs on the GPU in node tiles.
    """
    import mlx.core as mx

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    Xg = mx.array(X)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n, k)).astype(np.int32)

    def tile_dists(nodes, cand):
        """Squared distances from each node to its candidate columns (block × m)."""
        out = np.empty(cand.shape, dtype=np.float32)
        for s in range(0, nodes.size, tile):
            nb = nodes[s:s + tile]
            cb = cand[s:s + tile]
            Xc = Xg[mx.array(cb.ravel())].reshape(nb.size, cb.shape[1], -1)
            diff = Xc - Xg[mx.array(nb)][:, None, :]
            out[s:s + tile] = np.asarray(mx.sum(diff * diff, axis=2))
        return out

    import scipy.sparse as sp

    nodes = np.arange(n, dtype=np.int32)
    dist = tile_dists(nodes, idx)
    src = np.repeat(nodes, k)
    for _ in range(n_iters):
        # candidates = current ∪ neighbors-of-neighbors ∪ reverse-neighbors.
        # Reverse neighbors (who points to me) are essential for recall.
        nn_of_nn = idx[idx.reshape(-1)].reshape(n, k * k)
        GT = sp.csr_matrix((np.ones(n * k, np.int8), (idx.ravel(), src)), shape=(n, n))
        ind, ptr = GT.indices, GT.indptr                          # reverse adjacency
        deg = np.diff(ptr)
        pos = np.arange(ind.size) - np.repeat(ptr[:-1], deg)       # position within row
        m = pos < k                                               # keep first k per node
        rev = np.full((n, k), -1, np.int32)
        rev[np.repeat(np.arange(n), deg)[m], pos[m]] = ind[m]     # vectorized first-k reverse
        cand = np.concatenate([idx, nn_of_nn, rev], axis=1)
        cand[cand == nodes[:, None]] = -1                          # drop self
        cd = tile_dists(nodes, np.where(cand < 0, 0, cand))
        cd[cand < 0] = np.inf
        # dedup: for each row keep the first occurrence of each index
        order = np.argsort(cd, axis=1)
        cand_s = np.take_along_axis(cand, order, axis=1)
        cd_s = np.take_along_axis(cd, order, axis=1)
        dup = np.zeros_like(cand_s, dtype=bool)
        dup[:, 1:] = cand_s[:, 1:] == cand_s[:, :-1]
        cd_s[dup] = np.inf
        keep = np.argsort(cd_s, axis=1)[:, :k]
        new_idx = np.take_along_axis(cand_s, keep, axis=1).astype(np.int32)
        new_dist = np.take_along_axis(cd_s, keep, axis=1)
        if np.array_equal(new_idx, idx):
            idx, dist = new_idx, new_dist
            break
        idx, dist = new_idx, new_dist
    return idx, np.sqrt(np.maximum(dist, 0.0))


def _knn_ivf(X, k, nlist=None, nprobe=5, seed=0):
    """Approximate k-NN via IVF (kmeans buckets + within-bucket brute force) on GPU.

    Each cell searches its ``nprobe`` nearest kmeans buckets; distances are GPU
    matmuls per bucket, then a vectorized segmented top-k. Beats pynndescent ~2-2.6×
    in the mid-size band (~30k-250k) at recall ~0.86-0.98. Returns (idx, dist).
    """
    import mlx.core as mx

    from .tools import kmeans

    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    nlist = nlist or max(2, n // 1500)
    nprobe = min(nprobe, nlist)
    Xg = mx.array(X)
    lab = kmeans(X, nlist, max_iter=20, random_state=seed)
    cent = np.zeros((nlist, d), np.float32)
    cnt = np.bincount(lab, minlength=nlist)
    np.add.at(cent, lab, X); cent /= np.maximum(cnt[:, None], 1)
    cg = mx.array(cent)
    cd = mx.sum(Xg * Xg, 1)[:, None] + mx.sum(cg * cg, 1)[None, :] - 2 * (Xg @ cg.T)
    probe = np.asarray(mx.argpartition(cd, kth=min(nprobe, nlist - 1), axis=1)[:, :nprobe])

    order = np.argsort(lab, kind="stable"); sl = lab[order]
    st = np.searchsorted(sl, np.arange(nlist)); en = np.searchsorted(sl, np.arange(nlist), side="right")
    Qi, Ci, Dd = [], [], []
    for c in range(nlist):
        mem = order[st[c]:en[c]]
        if mem.size == 0:
            continue
        q = np.where((probe == c).any(1))[0]
        if q.size == 0:
            continue
        Qg = Xg[mx.array(q.astype(np.int32))]; Mg = Xg[mx.array(mem.astype(np.int32))]
        D = mx.maximum(mx.sum(Qg * Qg, 1)[:, None] + mx.sum(Mg * Mg, 1)[None, :] - 2 * (Qg @ Mg.T), 0.0)
        kk = min(k, mem.size)
        part = np.asarray(mx.argpartition(D, kth=kk - 1, axis=1)[:, :kk])
        Qi.append(np.repeat(q, kk)); Ci.append(mem[part].ravel())
        Dd.append(np.take_along_axis(np.asarray(D), part, axis=1).ravel())
    Q = np.concatenate(Qi); C = np.concatenate(Ci); Dv = np.concatenate(Dd)
    o = np.lexsort((Dv, Q)); Q, C, Dv = Q[o], C[o], Dv[o]          # sort by (query, dist)
    rank = np.arange(len(Q)) - np.repeat(np.searchsorted(Q, np.arange(n)), np.bincount(Q, minlength=n))
    keep = rank < k
    idx = np.full((n, k), -1, np.int32); dist = np.full((n, k), np.inf, np.float32)
    idx[Q[keep], rank[keep]] = C[keep]; dist[Q[keep], rank[keep]] = Dv[keep]
    return idx, np.sqrt(np.maximum(dist, 0.0))


def _knn(X_pca: np.ndarray, n_neighbors: int, random_state: int = 0,
         approx: bool | None = None):
    """k-NN dispatch by scale (returns ``(indices, dists)``), shared by neighbors/scrublet:

    *  n ≤ 30k   : exact GPU brute-force (fast + exact there)
    *  30k–250k  : GPU IVF (kmeans buckets) — ~2–2.6× faster than pynndescent, recall ~0.9
    *  > 250k    : pynndescent (scanpy's default; IVF recall/cost degrade past this)

    Plain GPU brute-force is O(n²) (memory + compute) and loses to optimized CPU here, so
    anything that does kNN at scale (e.g. scrublet on a ~3n doublet-simulation set) must go
    through this dispatch, not ``_knn_gpu`` directly.
    """
    n = X_pca.shape[0]
    if approx is None:
        approx = n > 30_000
    if not approx:
        return _knn_gpu(X_pca, n_neighbors)
    if n <= 250_000:
        return _knn_ivf(X_pca, n_neighbors, seed=random_state)
    from pynndescent import NNDescent
    index = NNDescent(np.asarray(X_pca, dtype=np.float32), n_neighbors=n_neighbors,
                      random_state=random_state)
    return index.neighbor_graph


def neighbors(X_pca: np.ndarray, n_neighbors: int = 15, random_state: int = 0,
              approx: bool | None = None):
    """Compute distance + connectivity graphs from a PCA embedding.

    Returns ``(distances, connectivities)`` as scipy CSR matrices, matching
    scanpy's ``obsp['distances']`` (k-1 neighbors/row) and ``obsp['connectivities']``
    (symmetric fuzzy graph).
    """
    import scipy.sparse as sp
    from umap.umap_ import fuzzy_simplicial_set

    n = X_pca.shape[0]
    knn_indices, knn_dists = _knn(X_pca, n_neighbors, random_state, approx)

    # UMAP fuzzy simplicial set — same call scanpy uses for method="umap".
    conn = fuzzy_simplicial_set(
        sp.coo_matrix((n, 1)),
        n_neighbors,
        random_state,
        metric="euclidean",
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    if isinstance(conn, tuple):  # newer umap returns (graph, sigmas, rhos)
        conn = conn[0]

    # Distances graph: k-1 nearest neighbors per row (exclude self), like scanpy.
    rows = np.repeat(np.arange(n), n_neighbors - 1)
    cols = knn_indices[:, 1:].ravel()
    vals = knn_dists[:, 1:].ravel()
    distances = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return distances, conn.tocsr()
