"""UMAP embedding with the force-layout optimization on the Metal GPU.

UMAP = (1) fuzzy graph [done in ``neighbors``], (2) optimize a low-dim layout by
SGD with attractive forces along graph edges and repulsive forces to negative
samples. Step (2) is the expensive part and parallelizes well, so we run it on
the GPU (MLX).

We drive the vendored mlx-vis UMAP optimizer (``_optimize``, pure MLX) with **our**
prebuilt connectivity graph — the same graph ``neighbors`` builds and ``leiden``
clusters on. This keeps the scverse cluster↔embedding contract (a visual blob is a
Leiden cluster) while getting mlx-vis's proper UMAP edge-sampling schedule
(``epochs_per_sample`` + negative sampling). On real atlas embeddings that lifts
trustworthiness from ~0.86 (our old all-edges-per-epoch layout) to ~0.95 and is ~4×
faster (0.66s vs 2.58s at 50k), and it removes the umap-learn dependency from this
module entirely. The embedding is stochastic, so we validate structure preservation,
not coordinates.

The **initialization** is ours (``_initial_embedding`` below) rather than mlx-vis's
``_spectral_init``, because a k-NN graph over a real atlas is routinely *disconnected*
and the vendored routine has no multi-component handling — see the module notes on
``_initial_embedding``.
"""

from __future__ import annotations

import warnings

import numpy as np

# A component smaller than this has no spectral structure worth recovering (we need
# dim+1 eigenvectors, and on a handful of nodes they are noise) — scatter it instead.
_MIN_SPECTRAL_COMPONENT = 16

# Below this, the 1st–99th percentile core of the init occupies so little of its own
# bounding box that essentially every cell sits on one of a few coincident points.
# Coincident points give the repulsive term its maximum (clipped) gradient every epoch,
# so the layout shears apart; measured 0.02–0.03 for a collapsed init vs ~0.79 healthy.
_DEGENERATE_CORE_FRACTION = 0.05

# Convergence of the orthogonal iteration is tested every this many steps (each test is a
# device sync, so testing every step costs more than the extra steps do).
_EIG_CHECK_EVERY = 10


def umap(connectivities, n_components: int = 2, n_epochs: int | None = None,
         min_dist: float = 0.5, spread: float = 1.0, random_state: int = 0,
         init: str | np.ndarray = "spectral") -> np.ndarray:
    """Optimize a UMAP embedding from a connectivity graph (GPU, mlx-vis optimizer).

    Lays out *our* shared fuzzy graph (``connectivities``) so the embedding matches the
    Leiden clustering; only the SGD optimizer is mlx-vis's.

    Args:
        init: ``"spectral"`` (default) for the component-aware spectral layout,
            ``"random"`` for a uniform random start, or an ``(n, n_components)``
            array of starting coordinates.
    """
    import mlx.core as mx
    from ._vendor.mlx_vis.umap import UMAP as _MlxUMAP

    n = connectivities.shape[0]
    if n_epochs is None:
        n_epochs = 500 if n <= 10_000 else 200

    a, b = _MlxUMAP._find_ab_params(spread, min_dist)  # exact drop-in for umap-learn's fit (~1e-7)

    # Drop edges too weak to be sampled even once over the run, as umap-learn does before
    # building its epoch schedule. We bring our own graph, so the vendored fuzzy-set builder
    # (which prunes at the same threshold) never sees it. Besides saving work, this keeps
    # `n_epochs / n_samples` in the optimizer's schedule from overflowing on denormal weights.
    graph = _prune_weak_edges(connectivities, n_epochs)

    # Seed before init + optimize so the (stochastic) layout is reproducible,
    # matching mlx-vis fit_transform's own `mx.random.seed(random_state)`.
    mx.random.seed(random_state)
    mv = _MlxUMAP(n_components=n_components, n_epochs=n_epochs, learning_rate=1.0,
                  random_state=random_state, pca_dim=None)

    coo = graph.tocoo()
    edge_from = mx.array(coo.row.astype(np.int32))
    edge_to = mx.array(coo.col.astype(np.int32))
    edge_weights = mx.array(coo.data.astype(np.float32))

    Y0 = _initial_embedding(graph, n_components, random_state, init)
    Y = mv._optimize(edge_from, edge_to, edge_weights, mx.array(Y0), a, b, n)
    mx.eval(Y)
    return np.asarray(Y)


def _prune_weak_edges(connectivities, n_epochs: int):
    """Zero out edges whose weight is below ``max_weight / n_epochs`` (umap-learn's rule).

    Such an edge is scheduled for fewer than one sample over the whole run, so it
    contributes nothing but arithmetic.
    """
    graph = connectivities.tocoo().copy()
    if graph.nnz == 0:
        return graph.tocsr()
    keep = graph.data >= (graph.data.max() / float(max(n_epochs, 1)))
    import scipy.sparse as sp
    return sp.coo_matrix((graph.data[keep], (graph.row[keep], graph.col[keep])),
                         shape=graph.shape).tocsr()


def _initial_embedding(graph, dim: int, random_state: int,
                       init: str | np.ndarray = "spectral") -> np.ndarray:
    """Starting coordinates for the SGD, component-aware.

    Why not a plain spectral layout: the normalized adjacency of a graph with ``c``
    connected components has eigenvalue 1 with multiplicity ``c``, and that eigenspace is
    spanned by the component indicator vectors. A power iteration therefore returns a
    *piecewise-constant* vector — every cell in a component lands on the same point.
    Real atlases disconnect readily (rare cell types, no HVG selection, large n), and the
    optimizer cannot recover: coincident points sit at the repulsive gradient's clip
    ceiling and drive the layout apart, which shows up as a pinhead of cells in the middle
    of an otherwise empty panel. So we detect components and lay each one out separately,
    then pack them — the same strategy as umap-learn's ``multi_component_layout``.
    """
    n = graph.shape[0]
    rng = np.random.RandomState(random_state)

    if isinstance(init, np.ndarray):
        Y = np.asarray(init, dtype=np.float32)
        if Y.shape != (n, dim):
            raise ValueError(f"init array has shape {Y.shape}, expected {(n, dim)}")
        return Y
    if init == "random":
        return _random_layout(n, dim, rng)
    if init != "spectral":
        raise ValueError(f"init must be 'spectral', 'random' or an array, got {init!r}")

    from scipy.sparse.csgraph import connected_components

    n_comp, labels = connected_components(graph, directed=False)
    if n_comp == 1:
        Y = _rescale(_spectral_layout(graph, dim, random_state), rng)
    else:
        warnings.warn(
            f"The neighbor graph has {n_comp} connected components; laying each out "
            "separately. Distances between components are not meaningful. Selecting "
            "highly variable genes before pp.pca usually yields a connected graph.",
            UserWarning, stacklevel=3,
        )
        Y = _multi_component_layout(graph, labels, n_comp, dim, random_state, rng)

    if _is_degenerate(Y):
        warnings.warn(
            "Spectral initialization collapsed (the graph has no usable low-frequency "
            "structure); falling back to a random initialization.",
            UserWarning, stacklevel=3,
        )
        Y = _random_layout(n, dim, rng)
    return Y


def _random_layout(n: int, dim: int, rng) -> np.ndarray:
    return rng.uniform(low=-10.0, high=10.0, size=(n, dim)).astype(np.float32)


def _rescale(Y: np.ndarray, rng, radius: float = 10.0) -> np.ndarray:
    """umap-learn's init scaling: one global expansion factor, plus a jitter to break ties.

    Note this is a *global* factor, not a per-axis min-max — rescaling each axis to a fixed
    range would stretch whatever handful of cells happen to be extreme on that axis out to
    the edge and crush everything else into the centre.
    """
    peak = np.max(np.abs(Y))
    if not np.isfinite(peak) or peak <= 0:
        return _random_layout(Y.shape[0], Y.shape[1], rng)
    Y = (Y * (radius / peak)).astype(np.float32)
    return Y + rng.normal(scale=1e-4, size=Y.shape).astype(np.float32)


def _spectral_layout(graph, dim: int, seed: int, n_iter: int = 200,
                     tol: float = 1e-7) -> np.ndarray:
    """The ``dim`` lowest non-trivial eigenvectors of the normalized Laplacian (MLX).

    Orthogonal iteration on the **shifted** operator ``(I + D^-1/2 A D^-1/2) / 2``, whose
    dominant eigenvectors are the Laplacian's low-frequency modes. The shift matters: on
    the unshifted operator an eigenvalue near ``-1`` (a near-bipartite subgraph) has the
    same magnitude as the mode we want, so the iteration can converge to an oscillatory
    mode instead.
    """
    import mlx.core as mx

    coo = graph.tocoo()
    n = graph.shape[0]
    k = dim + 1
    rows = mx.array(coo.row.astype(np.int32))
    cols = mx.array(coo.col.astype(np.int32))
    vals = mx.array(coo.data.astype(np.float32))

    degrees = mx.maximum(mx.zeros((n,)).at[rows].add(vals), 1e-10)
    d_inv_sqrt = 1.0 / mx.sqrt(degrees)
    w_norm = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]
    mx.eval(w_norm)

    mx.random.seed(seed)
    V = mx.random.normal((n, k))
    prev = None
    for it in range(n_iter):
        V = (mx.zeros_like(V).at[rows].add(w_norm[:, None] * V[cols]) + V) * 0.5
        for j in range(k):                                   # modified Gram-Schmidt
            for i in range(j):
                V = V.at[:, j].add(-mx.sum(V[:, j] * V[:, i]) * V[:, i])
            V = V.at[:, j].multiply(1.0 / mx.sqrt(mx.sum(V[:, j] * V[:, j]) + 1e-10))
        mx.eval(V)
        # Rayleigh quotients: stop once the eigenvalues stop moving. Each check costs a
        # device sync, so amortize it over a few iterations. (The sign of an eigenvector is
        # arbitrary, hence comparing the quotients rather than the vectors.)
        if (it + 1) % _EIG_CHECK_EVERY == 0:
            rq = np.asarray(mx.sum(V * ((mx.zeros_like(V).at[rows]
                                         .add(w_norm[:, None] * V[cols]) + V) * 0.5), axis=0))
            if prev is not None and np.max(np.abs(rq - prev)) < tol:
                break
            prev = rq

    return np.asarray(V[:, 1:k])


def _multi_component_layout(graph, labels: np.ndarray, n_comp: int, dim: int,
                            seed: int, rng) -> np.ndarray:
    """Lay out each connected component on its own, then pack the components.

    Components share no edges, so their relative placement is never optimized — it is set
    here and stays. We give each component a disc whose *area* is proportional to its cell
    count (radius ∝ √size) and pack those discs into concentric rings, so a rare 40-cell
    type reads as a small satellite rather than claiming as much canvas as the main body.
    """
    n = graph.shape[0]
    sizes = np.bincount(labels, minlength=n_comp)

    # Sort cells by component so every component is a contiguous block: two permutations of
    # the whole matrix instead of one fancy-index per component (which is O(nnz) each).
    order = np.argsort(labels, kind="stable")
    permuted = graph.tocsr()[order][:, order].tocsr()
    bounds = np.searchsorted(labels[order], np.arange(n_comp + 1))

    radii = np.sqrt(sizes / sizes.max()) * 10.0
    radii = np.maximum(radii, 0.4)                    # keep tiny components visible
    centers = _pack_discs(radii)

    Y = np.empty((n, dim), dtype=np.float32)
    for c in range(n_comp):
        start, stop = bounds[c], bounds[c + 1]
        idx = order[start:stop]
        size = stop - start
        if size < max(_MIN_SPECTRAL_COMPONENT, dim + 2):
            block = rng.normal(scale=radii[c] / 3.0, size=(size, dim)).astype(np.float32)
        else:
            block = _spectral_layout(permuted[start:stop, start:stop], dim, seed)
            peak = np.max(np.abs(block))
            if not np.isfinite(peak) or peak <= 0:    # collapsed subgraph — scatter it
                block = rng.normal(scale=radii[c] / 3.0, size=(size, dim)).astype(np.float32)
            else:
                block = (block * (radii[c] / peak)).astype(np.float32)
        Y[idx] = block + centers[c]

    return Y + rng.normal(scale=1e-4, size=Y.shape).astype(np.float32)


def _pack_discs(radii: np.ndarray, pad: float = 1.1) -> np.ndarray:
    """Centers for non-overlapping discs of the given radii: biggest first, then rings.

    Returned in the first two dimensions only (zero elsewhere for ``dim > 2``); the
    within-component layout still uses every dimension.
    """
    k = len(radii)
    centers = np.zeros((k, 2), dtype=np.float32)
    order = np.argsort(-radii)
    centers[order[0]] = 0.0
    inner = radii[order[0]] * pad
    i = 1
    while i < k:
        rest = order[i:]
        ring_r = radii[rest[0]]                       # size-sorted, so this is the largest left
        R = inner + ring_r * pad
        circumference = 2.0 * np.pi * R
        widths = 2.0 * radii[rest] * pad
        fits = max(1, int(np.searchsorted(np.cumsum(widths), circumference)))
        on_ring = rest[:fits]
        arc = np.concatenate([[0.0], np.cumsum(2.0 * radii[on_ring] * pad)])
        theta = (arc[:-1] + arc[1:]) / 2.0 / R
        centers[on_ring, 0] = R * np.cos(theta)
        centers[on_ring, 1] = R * np.sin(theta)
        inner = R + ring_r * pad
        i += fits
    return centers


def _is_degenerate(Y: np.ndarray) -> bool:
    """True when the init has collapsed onto a few coincident points (see the constant)."""
    if not np.all(np.isfinite(Y)):
        return True
    span = Y.max(axis=0) - Y.min(axis=0)
    if np.any(span <= 0):
        return True
    core = np.percentile(Y, 99, axis=0) - np.percentile(Y, 1, axis=0)
    return bool(np.all(core / span < _DEGENERATE_CORE_FRACTION))
