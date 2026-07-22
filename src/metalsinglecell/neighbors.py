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


# Custom top-k-per-row selection kernel. MLX's general ``argpartition`` over wide rows is
# the kNN bottleneck (measured ~5.7× the distance compute: 25k → distance 39ms vs
# distance+argpartition 222ms) because it partitions the full n-wide row. For k≪n we don't
# need a full partition: one thread per row scans its distances once and keeps the k
# smallest in registers (k≤32 → a tiny register array), then sorts those k — O(n·k), one
# pass, no general sort. Reads the distance matrix once (memory-bound). This is the brute
# core that the IVF buckets and bbknn batches also use.
_TOPK_KERNEL_SOURCE = """
    uint row = thread_position_in_grid.x;
    uint n = (uint)dims[0];
    uint K = (uint)dims[1];
    float vals[32];
    int inds[32];
    for (uint i = 0; i < K; ++i) { vals[i] = 1e30f; inds[i] = -1; }
    uint maxp = 0;                                   // position of the current largest of the K
    uint base = row * n;
    for (uint j = 0; j < n; ++j) {
        float d = (float)data[base + j];
        if (d < vals[maxp]) {
            vals[maxp] = d; inds[maxp] = (int)j;
            float m = vals[0]; uint mp = 0;          // rescan for the new largest
            for (uint i = 1; i < K; ++i) { if (vals[i] > m) { m = vals[i]; mp = i; } }
            maxp = mp;
        }
    }
    for (uint i = 1; i < K; ++i) {                   // insertion-sort the K ascending
        float v = vals[i]; int id = inds[i]; int t = (int)i - 1;
        while (t >= 0 && vals[t] > v) { vals[t + 1] = vals[t]; inds[t + 1] = inds[t]; --t; }
        vals[t + 1] = v; inds[t + 1] = id;
    }
    for (uint i = 0; i < K; ++i) { out_idx[row * K + i] = inds[i]; }
"""


_TOPK_KERNEL_MAX_K = 32          # register arrays in _TOPK_KERNEL_SOURCE are float vals[32]/int inds[32]


def _topk_rows(D2, k):
    """Indices of the k smallest entries per row of ``D2`` (m×n MLX array).

    For ``k <= 32`` uses a one-pass register top-k kernel; for larger ``k`` (a legal
    ``n_neighbors`` — atlases routinely use 30–50) the kernel's fixed 32-slot register
    arrays would overflow, so we fall back to ``mx.argpartition`` (correct for any ``k``).
    Callers re-sort the selected ``k`` by exact fp32 distance, so unordered output is fine.
    """
    import mlx.core as mx

    m, ncols = D2.shape
    if k > _TOPK_KERNEL_MAX_K:                    # kernel register arrays are capped at 32 — fall back
        kth = min(int(k), ncols - 1)
        return mx.argpartition(D2, kth=kth, axis=1)[:, :k]

    kernel = mx.fast.metal_kernel(
        name="topk_rows", input_names=["data", "dims"], output_names=["out_idx"],
        source=_TOPK_KERNEL_SOURCE,
    )
    (out,) = kernel(
        inputs=[D2, mx.array([ncols, k], dtype=mx.uint32)],
        grid=(m, 1, 1), threadgroup=(min(256, m), 1, 1),
        output_shapes=[(m, k)], output_dtypes=[mx.int32],
    )
    return out


def _knn_gpu(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Approximate k-NN (incl. self) via brute-force squared-Euclidean on the GPU.

    Neighbors are RANKED on fp16 distances (rescaled to avoid overflow), so recall is
    ~0.99, not exact — the k selected distances are then recomputed in fp32 for exact
    returned values. Returns ``(knn_indices, knn_dists)`` of shape (n, k), self first.
    """
    import math

    import mlx.core as mx

    Xg = mx.array(X.astype(np.float32))
    n = X.shape[0]
    # fp16 distances are used only to RANK neighbors (selected distances are recomputed in
    # fp32 below), but a large-magnitude embedding (e.g. Pearson-residual PCA) overflows fp16
    # (max ~6.5e4) → inf → garbage top-k. Rescale by f so the worst-case squared distance
    # (~4·max‖x‖²) stays well inside fp16; ranking is scale-invariant, so this is exact.
    sq32 = mx.sum(Xg * Xg, axis=1)
    maxnorm2 = float(mx.max(sq32).item()) if n else 0.0
    f = min(1.0, math.sqrt(8000.0 / (4.0 * maxnorm2 + 1e-9))) if maxnorm2 > 0 else 1.0
    Xg_h = (Xg * f).astype(mx.float16)                    # scaled fp16 for ranking
    sq_h = mx.sum(Xg_h * Xg_h, axis=1)

    # Tile over query rows so we never materialize the full n×n distance matrix
    # (that OOMs the GPU past ~30k cells). Each tile is block×n. The whole distance +
    # argpartition runs in fp16 — kept fp16 throughout (an fp32 cast-back of the large
    # block×n matrix would negate the gain) and used only to RANK neighbors, which fp16
    # preserves (recall ~0.99). The k selected distances are then recomputed exactly in
    # fp32 from the originals (so returned distances are full precision).
    # _knn_gpu is the brute path (n≲30k); the full fp16 n×n matrix fits in one tile there
    # (30k² ≈ 1.8 GB), so a generous cap avoids per-tile dispatch overhead (~40 tiny tiles
    # at the old 16M cap cost ~190ms). Larger n still tiles to stay within memory.
    block = max(1, min(n, 1_000_000_000 // max(n, 1)))   # cap tile at ~1B entries (2GB fp16)
    idx_parts, d2_parts = [], []
    for s in range(0, n, block):
        D2 = mx.maximum(sq_h[s:s + block][:, None] + sq_h[None, :]
                        - 2.0 * (Xg_h[s:s + block] @ Xg_h.T), 0.0)        # all fp16
        bidx = _topk_rows(D2, k)                          # custom kernel (vs slow argpartition)
        # recompute the selected distances EXACTLY in fp32 from the originals (not the scaled
        # fp16 D2) — exact returned distances, immune to the fp16 rescale above.
        diff = Xg[s:s + block][:, None, :] - Xg[bidx]     # block×k×D, fp32
        bd2 = mx.sum(diff * diff, axis=2)
        mx.eval(bidx, bd2)                                # transfer only the small block×k results
        idx_parts.append(np.asarray(bidx)); d2_parts.append(np.asarray(bd2).astype(np.float32))
    knn_indices = np.concatenate(idx_parts, axis=0)
    d2 = np.concatenate(d2_parts, axis=0)

    order = np.argsort(d2, axis=1)                        # sort k by distance, self first
    knn_indices = np.take_along_axis(knn_indices, order, axis=1)
    knn_dists = np.sqrt(np.take_along_axis(d2, order, axis=1))
    return knn_indices, knn_dists


# Uniform-grid (cell-list) exact k-NN for LOW-dimensional point data — the right accelerator for
# spatial coordinates, where the high-dim ANN path (NNDescent) is degenerate (~6% recall on 2-D) and
# brute-force is O(n²). Points are binned into a uniform grid sized so each cell holds ~a few·k points;
# each query then only scans its own cell + the surrounding ring (3×3 in 2-D / 3×3×3 in 3-D) — O(1)
# candidates per query, O(n) total — via the kernel below (one thread per query, register top-k like
# _topk_rows). EXACT: after searching Chebyshev radius 1, any unexamined point is ≥ cell_size away, so a
# query whose k-th neighbour distance ≤ cell_size is provably complete; the rare failures (sparse edges,
# too-small a block) are recomputed by exact brute force. See SPATIAL_KNN_investigation.md.
_GRID_KNN_SOURCE = """
    uint q = thread_position_in_grid.x;
    int n  = iparams[0];
    int d  = iparams[1];
    int K  = iparams[2];
    int gx = iparams[3];
    int gy = iparams[4];
    int gz = iparams[5];
    float ox = fparams[0];
    float oy = fparams[1];
    float oz = fparams[2];
    float cs = fparams[3];

    float qx = coords[q * d + 0];
    float qy = coords[q * d + 1];
    float qz = (d == 3) ? coords[q * d + 2] : 0.0f;
    int cx = (int)floor((qx - ox) / cs); if (cx < 0) cx = 0; if (cx >= gx) cx = gx - 1;
    int cy = (int)floor((qy - oy) / cs); if (cy < 0) cy = 0; if (cy >= gy) cy = gy - 1;
    int cz = (d == 3) ? (int)floor((qz - oz) / cs) : 0;
    if (cz < 0) cz = 0; if (cz >= gz) cz = gz - 1;

    float vals[32]; int inds[32];
    for (int i = 0; i < K; ++i) { vals[i] = 1e30f; inds[i] = -1; }
    uint maxp = 0;                                     // position of the current largest of the K

    int dzlo = (d == 3) ? -1 : 0;
    int dzhi = (d == 3) ?  1 : 0;
    for (int dz = dzlo; dz <= dzhi; ++dz) {
        int nz = cz + dz; if (nz < 0 || nz >= gz) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy; if (ny < 0 || ny >= gy) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx; if (nx < 0 || nx >= gx) continue;
                int cell = (nz * gy + ny) * gx + nx;
                uint s = cell_start[cell];
                uint e = cell_start[cell + 1];
                for (uint p = s; p < e; ++p) {
                    int pt = sorted_pts[p];
                    float ddx = coords[pt * d + 0] - qx;
                    float ddy = coords[pt * d + 1] - qy;
                    float dist2 = ddx * ddx + ddy * ddy;
                    if (d == 3) { float ddz = coords[pt * d + 2] - qz; dist2 += ddz * ddz; }
                    if (dist2 < vals[maxp]) {
                        vals[maxp] = dist2; inds[maxp] = pt;
                        float m = vals[0]; uint mp = 0;    // rescan for the new largest
                        for (int i = 1; i < K; ++i) { if (vals[i] > m) { m = vals[i]; mp = i; } }
                        maxp = mp;
                    }
                }
            }
        }
    }
    for (uint i = 0; i < (uint)K; ++i) { out_idx[q * K + i] = inds[i]; out_d2[q * K + i] = vals[i]; }
"""


def _knn_grid(coords: np.ndarray, k: int, min_n: int = 4_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact self-inclusive k-NN on low-dim (2-D/3-D) point data via a uniform-grid cell list.

    Drop-in for ``_knn_gpu`` on spatial coordinates: same ``(knn_indices, knn_dists)`` contract
    (shape ``(n, k)``, self first, sorted by distance) but O(n) instead of O(n²). Below ``min_n`` or
    for ``d > 3`` / ``k > 32`` / degenerate geometry it defers to the exact fp32 brute path. Provably
    exact: grid results whose k-th distance ≤ cell_size are complete, the rest recomputed by brute.

    NB the fallback is the **fp32-exact** brute (``_exact_knn_rows``), NOT ``_knn_gpu``: the latter ranks
    on fp16, which corrupts on raw spatial coordinates (µm/bin values in the thousands exceed fp16's
    2048 integer-exact limit → wrong neighbours on Visium/Stereo-seq lattices). Spatial kNN is fp32
    throughout.
    """
    import mlx.core as mx

    X = np.asarray(coords, dtype=np.float32)
    n, d = X.shape
    if n <= min_n or d not in (2, 3) or k > _TOPK_KERNEL_MAX_K or k >= n:
        return _exact_knn_rows(X, np.arange(n), k)

    # ---- choose a cell size targeting ~a few·k points per cell (uniform-ish spatial density) ----
    mins = X.min(0); maxs = X.max(0)
    extent = (maxs - mins).astype(np.float64)
    nz = extent > 0
    if not nz.any():                                   # all points coincide → brute
        return _exact_knn_rows(X, np.arange(n), k)
    eff_d = int(nz.sum())
    density = n / float(np.prod(extent[nz]))           # points per unit (area/volume) over occupied axes
    target = max(2 * k, 8)                              # ~candidates guaranteeing the 3×3 block holds k
    cell = (target / density) ** (1.0 / eff_d)
    # cap the dense cell_start table (gx·gy·gz+1) at ~4n by coarsening if the bbox is sparse/elongated
    for _ in range(8):
        g = np.where(nz, np.ceil(extent / cell).astype(np.int64) + 1, 1)
        ncells = int(np.prod(g))
        if ncells <= 4 * n + 16:
            break
        cell *= (ncells / (4.0 * n + 16.0)) ** (1.0 / eff_d)
    gx, gy, gz = (int(g[0]), int(g[1]), int(g[2]) if d == 3 else 1)
    origin = mins.astype(np.float32)

    # ---- bin points → cell ids → sorted point list + CSR-style cell offsets ----
    ci = np.floor((X - origin) / cell).astype(np.int64)
    ci = np.clip(ci, 0, g[:d] - 1)
    if d == 3:
        cell_id = (ci[:, 2] * gy + ci[:, 1]) * gx + ci[:, 0]
    else:
        cell_id = ci[:, 1] * gx + ci[:, 0]
    order = np.argsort(cell_id, kind="stable").astype(np.int32)
    counts = np.bincount(cell_id, minlength=ncells)
    cell_start = np.empty(ncells + 1, dtype=np.uint32)
    cell_start[0] = 0
    cell_start[1:] = np.cumsum(counts)

    # ---- query kernel: one thread per point, scan own cell + ring, register top-k ----
    kernel = mx.fast.metal_kernel(
        name="grid_knn",
        input_names=["coords", "sorted_pts", "cell_start", "iparams", "fparams"],
        output_names=["out_idx", "out_d2"],
        source=_GRID_KNN_SOURCE,
    )
    iparams = mx.array([n, d, k, gx, gy, gz], dtype=mx.int32)
    fparams = mx.array([float(origin[0]), float(origin[1]),
                        float(origin[2]) if d == 3 else 0.0, float(cell)], dtype=mx.float32)
    out_idx, out_d2 = kernel(
        inputs=[mx.array(X), mx.array(order), mx.array(cell_start), iparams, fparams],
        grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1),
        output_shapes=[(n, k), (n, k)], output_dtypes=[mx.int32, mx.float32],
    )
    mx.eval(out_idx, out_d2)
    idx = np.asarray(out_idx)
    d2 = np.asarray(out_d2)

    order_k = np.argsort(d2, axis=1)                   # sort the k by distance, self first
    idx = np.take_along_axis(idx, order_k, axis=1)
    d2 = np.take_along_axis(d2, order_k, axis=1)
    dist = np.sqrt(np.maximum(d2, 0.0))

    # ---- exactness: a query is complete iff it found k candidates AND its k-th ≤ cell_size ----
    complete = (idx[:, -1] >= 0) & (dist[:, -1] <= float(cell))
    bad = np.flatnonzero(~complete)
    if bad.size:
        bi, bd = _exact_knn_rows(X, bad, k)            # exact brute for the rare failures
        idx[bad] = bi
        dist[bad] = bd
    return idx.astype(np.int64), dist.astype(np.float32)


def _exact_knn_rows(coords: np.ndarray, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN (self first) for a subset of query ``rows`` against all points, tiled on the GPU."""
    import mlx.core as mx

    X = np.asarray(coords, dtype=np.float32)
    n = X.shape[0]
    Xg = mx.array(X)
    sq = mx.sum(Xg * Xg, axis=1)
    tile = max(1, 256_000_000 // max(n, 1))
    oi = np.empty((rows.size, k), np.int64)
    od = np.empty((rows.size, k), np.float32)
    for s in range(0, rows.size, tile):
        rb = rows[s:s + tile]
        Qg = Xg[mx.array(rb.astype(np.int32))]
        D2 = mx.maximum(mx.sum(Qg * Qg, axis=1)[:, None] + sq[None, :] - 2.0 * (Qg @ Xg.T), 0.0)
        loc = _topk_rows(D2, k)
        Dv = mx.take_along_axis(D2, loc, axis=1)
        mx.eval(loc, Dv)
        loc = np.asarray(loc); Dv = np.asarray(Dv)
        o = np.argsort(Dv, axis=1)
        oi[s:s + tile] = np.take_along_axis(loc, o, axis=1)
        od[s:s + tile] = np.sqrt(np.maximum(np.take_along_axis(Dv, o, axis=1), 0.0))
    return oi, od


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
            loc = _topk_rows(D2, kk)                      # fast register top-k (kk≪32) vs argpartition
            Dvg = mx.take_along_axis(D2, loc, axis=1)     # gather selected dists on GPU
            mx.eval(loc, Dvg)                             # transfer only the small block×kk results
            d_b[s:e] = np.sqrt(np.maximum(np.asarray(Dvg), 0.0))
            loc_b[s:e] = np.asarray(loc)
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
        part = _topk_rows(D, kk)                          # custom top-k (vs argpartition)
        Dvg = mx.take_along_axis(D, part, axis=1)         # gather dists on GPU (avoid full D transfer)
        mx.eval(part, Dvg); part = np.asarray(part)
        Qi.append(np.repeat(q, kk)); Ci.append(mem[part].ravel())
        Dd.append(np.asarray(Dvg).ravel())
    Q = np.concatenate(Qi); C = np.concatenate(Ci); Dv = np.concatenate(Dd)
    o = np.lexsort((Dv, Q)); Q, C, Dv = Q[o], C[o], Dv[o]          # sort by (query, dist)
    rank = np.arange(len(Q)) - np.repeat(np.searchsorted(Q, np.arange(n)), np.bincount(Q, minlength=n))
    keep = rank < k
    idx = np.full((n, k), -1, np.int32); dist = np.full((n, k), np.inf, np.float32)
    idx[Q[keep], rank[keep]] = C[keep]; dist[Q[keep], rank[keep]] = Dv[keep]
    return idx, np.sqrt(np.maximum(dist, 0.0))


def _knn_nndescent(X, k, over: int = 2, random_state: int = 0):
    """Approximate self-kNN via the vendored mlx-vis NNDescent (pure-MLX, on the GPU).

    Two call-site adaptations (see results/neighbors_optimization/RESULTS_mlxvis_integration.md):
    over-build the graph to degree ``over*k`` and keep the best ``k`` (lifts recall from ~0.86 at
    raw k to ~0.92–0.98 across 50k–1M), and **prepend self** (NNDescent excludes it) to match the
    self-inclusive scanpy convention. Distances are euclidean (verified), matching pynndescent.
    Returns ``(indices, dists)`` shape ``(n, k)``, self first.
    """
    from ._vendor.mlx_vis.nndescent import NNDescent

    Xf = np.asarray(X, dtype=np.float32)
    n = Xf.shape[0]
    idx, d = NNDescent(k=min(over * k, n - 1), random_state=random_state).build(Xf)
    out_i = np.empty((n, k), np.int64); out_d = np.empty((n, k), np.float32)
    out_i[:, 0] = np.arange(n); out_d[:, 0] = 0.0             # self first (dist 0)
    m = min(k - 1, idx.shape[1])
    out_i[:, 1:1 + m] = idx[:, :m]; out_d[:, 1:1 + m] = d[:, :m]
    return out_i, out_d


def _knn(X_pca: np.ndarray, n_neighbors: int, random_state: int = 0,
         approx: bool | None = None, backend: str = "nndescent"):
    """k-NN dispatch by scale (returns ``(indices, dists)``), shared by neighbors/scrublet:

    *  n ≤ 30k                : exact GPU brute-force (fast + exact there)
    *  n > 30k (``nndescent``): GPU NNDescent (vendored mlx-vis) — the scalable self-kNN path,
       replacing the old IVF (30k–250k) + CPU-pynndescent (>250k) ladder. Recall matches the
       former path at 2.2–4.7× the speed and stable wall time. See RESULTS_mlxvis_integration.md.

    ``backend`` selects the >30k engine: ``"nndescent"`` (default), or the retained oracles
    ``"ivf"`` / ``"pynndescent"`` (for validation/benchmarking). Brute is O(n²) and loses at scale,
    so anything doing kNN at scale (e.g. scrublet's ~3n doublet set) goes through this dispatch.
    """
    n = X_pca.shape[0]
    if approx is None:
        approx = n > 30_000
    if not approx:
        return _knn_gpu(X_pca, n_neighbors)
    if backend == "nndescent":
        return _knn_nndescent(X_pca, n_neighbors, random_state=random_state)
    if backend == "ivf":                                     # retained oracle
        return _knn_ivf(X_pca, n_neighbors, seed=random_state)
    if backend == "pynndescent":                             # retained CPU oracle
        from pynndescent import NNDescent
        return NNDescent(np.asarray(X_pca, dtype=np.float32), n_neighbors=n_neighbors,
                         random_state=random_state).neighbor_graph
    raise ValueError(f"unknown kNN backend {backend!r} (nndescent|ivf|pynndescent)")


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
