"""UMAP embedding with the force-layout optimization on the Metal GPU.

UMAP = (1) fuzzy graph [done in ``neighbors``], (2) optimize a low-dim layout by
SGD with attractive forces along graph edges and repulsive forces to negative
samples. Step (2) is the expensive part and parallelizes well, so we run it on
the GPU (MLX).

Both halves of step (2) — the initialization and the SGD — are ours, driven by the
connectivity graph ``neighbors`` builds and ``leiden`` clusters on. That keeps the scverse
cluster↔embedding contract (a visual blob is a Leiden cluster). The embedding is
stochastic, so we validate structure preservation, not coordinates.

``_optimize_layout`` is a vectorized port of umap-learn's ``optimize_layout_euclidean``:
same ``epochs_per_sample`` schedule, same attractive/repulsive gradients, same ±4 clip,
same ``alpha`` decay. It replaced the vendored mlx-vis optimizer, which mishandled the one
place a GPU port cannot follow the reference exactly. umap-learn walks edges **sequentially**
inside an epoch, so after a node is pushed its next gradient is computed from the new
position; we evaluate a whole epoch's edges in parallel from one snapshot of ``Y``. The
vendored version clipped each *edge's* gradient to ±4 and then scatter-added every edge onto
the same ``Y``, so a node accumulated one clipped step per incident edge with no feedback
between them. Measured on a 100k graph of max degree 492, that moved a node **477 units in
a single epoch** and tore the layout apart (github issue #1). We keep the parallel evaluation
— it is the whole speed advantage — and recover the missing feedback with a trust region:
the *accumulated* per-node step is capped at what one edge alone could contribute. See
``_MAX_NODE_STEP``.

The **initialization** is ours as well (``_initial_embedding``), because a k-NN graph over a
real atlas is routinely *disconnected* and the vendored spectral init has no multi-component
handling — see the notes on ``_initial_embedding``.
"""

from __future__ import annotations

import warnings

import numpy as np

# A component smaller than this has no spectral structure worth recovering (we need
# dim+1 eigenvectors, and on a handful of nodes they are noise) — scatter it instead.
_MIN_SPECTRAL_COMPONENT = 16

# Convergence of the orthogonal iteration is tested every this many steps (each test is a
# device sync, so testing every step costs more than the extra steps do).
_EIG_CHECK_EVERY = 10

# umap-learn clips each edge's gradient to this before applying it (`clip` in layouts.py).
_GRAD_CLIP = 4.0

# Trust region on the ACCUMULATED per-node step — see the module docstring for why a batched
# port needs one. Tuned on REAL data (1.3M-neuron atlas) and the issue-#1 reproduction, 3 seeds
# each; synthetic sets are indifferent to it and will mislead you here.
#
#   cap    atlas 100k trustworthiness    issue-#1 repro, core (collapse < 0.2)
#    16          0.9334 ± 0.0065                  0.907 ± 0.024
#    32          0.9410 ± 0.0017                  0.909 ± 0.013
#    64          0.9441 ± 0.0035                  0.927 ± 0.011
#   128               —                           0.401 ± 0.020   <- degrading
#   inf          0.9463 ± 0.0063                  0.113           <- the bug
#
# 64 is not a compromise: it is the best measured value on BOTH axes, and it matches the
# quality of the optimizer it replaces (0.9433 on the same data). A tighter cap under-optimizes
# — nodes cannot travel far enough — which costs quality *and* spread. Do not raise it past 64;
# 128 already degrades, and the whole failure mode returns without a finite bound.
_MAX_NODE_STEP = 64.0

# Graphs at or below this go to ARPACK rather than the GPU orthogonal iteration; see
# `_spectral_layout`. Set from the measured crossover.
_ARPACK_MAX_N = 20_000


def umap(connectivities, n_components: int = 2, n_epochs: int | None = None,
         min_dist: float = 0.5, spread: float = 1.0, random_state: int = 0,
         init: str | np.ndarray = "spectral", learning_rate: float = 1.0,
         negative_sample_rate: int = 5, gamma: float = 1.0,
         max_node_step: float = _MAX_NODE_STEP,
         a: float | None = None, b: float | None = None) -> np.ndarray:
    """Optimize a UMAP embedding from a connectivity graph (GPU).

    Lays out *our* shared fuzzy graph (``connectivities``) so the embedding matches the
    Leiden clustering.

    Args:
        init: ``"spectral"`` (default) for the component-aware spectral layout,
            ``"random"`` for a uniform random start, or an ``(n, n_components)``
            array of starting coordinates.
        gamma: weight applied to the repulsive term (umap-learn's ``repulsion_strength``).

    ``random_state`` pins the initialization and the negative samples, but the result is
    **not bit-for-bit reproducible**. An epoch's gradients are accumulated with an atomic
    scatter-add, and floating-point addition is not associative, so the order the GPU happens
    to apply them perturbs the sum at ~1e-7 and the SGD amplifies that over epochs. Repeat
    runs agree on structure, not coordinates: measured k-NN overlap 0.87–0.89 between runs at
    the same seed. (umap-learn on CPU is sequential and therefore exactly reproducible; making
    this so would need a sorted segmented reduction in place of the atomics.)
    """
    import mlx.core as mx
    from ._vendor.mlx_vis.umap import UMAP as _MlxUMAP

    n = connectivities.shape[0]
    if n_epochs is None:
        n_epochs = 500 if n <= 10_000 else 200

    if a is None or b is None:                    # else use the caller's curve, as scanpy allows
        a, b = _MlxUMAP._find_ab_params(spread, min_dist)  # drop-in for umap-learn's fit (~1e-7)

    # Drop edges too weak to be sampled even once over the run, as umap-learn does before
    # building its epoch schedule. We bring our own graph, so the vendored fuzzy-set builder
    # (which prunes at the same threshold) never sees it.
    graph = _prune_weak_edges(connectivities, n_epochs)

    # Seed before init + optimize so the (stochastic) layout is reproducible.
    mx.random.seed(random_state)

    Y0 = _initial_embedding(graph, n_components, random_state, init)
    coo = graph.tocoo()
    Y = _optimize_layout(coo.row, coo.col, coo.data, Y0, a, b, n, n_epochs,
                         negative_sample_rate=negative_sample_rate, gamma=gamma,
                         learning_rate=learning_rate, max_node_step=max_node_step,
                         seed=random_state)
    return Y


def _make_epochs_per_sample(weights: np.ndarray, n_epochs: int) -> np.ndarray:
    """umap-learn's ``make_epochs_per_sample``: how often each edge is sampled.

    An edge of the maximum weight is sampled every epoch; weaker edges proportionally less
    often. ``-1`` marks an edge that is never sampled.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    positive = n_samples > 0
    # A denormal weight overflows to +inf here, which is exactly the semantics we want
    # ("never sampled" — `inf <= epoch` is never true), so don't let it raise a warning.
    # `umap` prunes such edges beforehand; this keeps the helper safe when called directly.
    with np.errstate(over="ignore"):
        result[positive] = float(n_epochs) / n_samples[positive]
    return result


def _optimize_layout(rows: np.ndarray, cols: np.ndarray, weights: np.ndarray,
                     Y0: np.ndarray, a: float, b: float, n: int, n_epochs: int,
                     learning_rate: float = 1.0, negative_sample_rate: int = 5,
                     gamma: float = 1.0, max_node_step: float = _MAX_NODE_STEP,
                     seed: int = 0) -> np.ndarray:
    """Vectorized port of umap-learn's ``optimize_layout_euclidean`` (MLX / Metal).

    Faithful to the reference in the parts that define the algorithm: the
    ``epochs_per_sample`` schedule, both gradient forms, the ±4 per-edge clip, the linear
    ``alpha`` decay, skipping a negative sample that lands on the head itself, and the fixed
    push given to coincident points.

    It differs in the two places a GPU port must. (1) An epoch's edges are evaluated in
    parallel from one snapshot of ``Y`` rather than sequentially — that parallelism is the
    speedup, and ``max_node_step`` supplies the damping the reference gets for free from
    sequential updates. (2) Each active edge draws a fixed ``negative_sample_rate`` negatives
    rather than a number derived from its own schedule; the two agree in expectation.
    """
    import mlx.core as mx

    epochs_per_sample = _make_epochs_per_sample(np.asarray(weights, dtype=np.float64),
                                                n_epochs)
    # Pre-compute which edges are active in each epoch (all the bookkeeping upfront, so the
    # per-epoch work is pure GPU). `-1` edges never fire: `<= epoch` is never true for inf.
    next_sample = np.where(epochs_per_sample > 0, epochs_per_sample, np.inf)
    active_sets = []
    for epoch in range(n_epochs):
        active = np.flatnonzero(next_sample <= epoch)
        if active.size:
            next_sample[active] += epochs_per_sample[active]
            active_sets.append(mx.array(active.astype(np.int32)))
        else:
            active_sets.append(None)

    edge_from = mx.array(np.asarray(rows, dtype=np.int32))
    edge_to = mx.array(np.asarray(cols, dtype=np.int32))
    Y = mx.array(np.asarray(Y0, dtype=np.float32))
    a_mx, b_mx = mx.array(float(a)), mx.array(float(b))
    gamma_mx, cap_mx = mx.array(float(gamma)), mx.array(float(max_node_step))

    # Draw negatives from an EXPLICIT key chain, not MLX's global RNG. The global stream is
    # advanced by every other random call in the process, so which branch the initialization
    # took (ARPACK vs the GPU iteration) changed the layout for the same `random_state` —
    # `msc.tl.umap(adata, random_state=0)` twice gave different embeddings.
    key = mx.random.key(seed)

    for epoch in range(n_epochs):
        active = active_sets[epoch]
        if active is None:
            continue
        ef, et = edge_from[active], edge_to[active]
        n_active = active.shape[0]
        alpha = mx.array(learning_rate * (1.0 - epoch / n_epochs))

        n_neg = negative_sample_rate * n_active
        neg_from = ef[mx.arange(n_neg) % n_active]
        key, subkey = mx.random.split(key)
        neg_to = mx.random.randint(0, n, (n_neg,), key=subkey)

        Y = _sgd_epoch(Y, ef, et, neg_from, neg_to, alpha, a_mx, b_mx, gamma_mx, cap_mx)
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            mx.eval(Y)                       # bound the lazy graph without syncing every epoch

    mx.eval(Y)
    return np.asarray(Y)


def _sgd_epoch(Y, ef, et, neg_from, neg_to, alpha, a, b, gamma, cap):
    """One epoch of the layout SGD, all active edges at once. Compiled on first call."""
    import mlx.core as mx

    global _SGD_COMPILED
    if _SGD_COMPILED is None:
        _SGD_COMPILED = mx.compile(_sgd_epoch_impl)
    return _SGD_COMPILED(Y, ef, et, neg_from, neg_to, alpha, a, b, gamma, cap)


_SGD_COMPILED = None


def _sgd_epoch_impl(Y, ef, et, neg_from, neg_to, alpha, a, b, gamma, cap):
    import mlx.core as mx

    # Attraction along graph edges. umap-learn zeroes the gradient when the two points
    # coincide (`if dist_squared > 0.0 ... else grad_coeff = 0.0`), so guard rather than
    # clamping the distance up to some epsilon.
    diff = Y[ef] - Y[et]
    d2 = mx.sum(diff * diff, axis=1, keepdims=True)
    safe = mx.maximum(d2, 1e-12)
    coeff = -2.0 * a * b * mx.power(safe, b - 1.0) / (1.0 + a * mx.power(safe, b))
    pos_grad = mx.where(d2 > 0.0, mx.clip(coeff * diff, -_GRAD_CLIP, _GRAD_CLIP), 0.0)

    # Repulsion from negative samples. Coincident-but-distinct points get the reference's
    # fixed push of +4; a sample that lands on the head itself is skipped entirely.
    neg_diff = Y[neg_from] - Y[neg_to]
    nd2 = mx.sum(neg_diff * neg_diff, axis=1, keepdims=True)
    nsafe = mx.maximum(nd2, 1e-12)
    ncoeff = 2.0 * gamma * b / ((0.001 + nsafe) * (1.0 + a * mx.power(nsafe, b)))
    neg_grad = mx.where(nd2 > 0.0,
                        mx.clip(ncoeff * neg_diff, -_GRAD_CLIP, _GRAD_CLIP),
                        _GRAD_CLIP)
    neg_grad = mx.where((neg_from == neg_to)[:, None], 0.0, neg_grad)

    step = (mx.zeros_like(Y)
            .at[ef].add(pos_grad)
            .at[et].add(-pos_grad)
            .at[neg_from].add(neg_grad))

    # Trust region: cap the ACCUMULATED per-node step at one edge's worth. Without it a
    # high-degree node sums hundreds of clipped gradients computed from stale positions.
    norm = mx.sqrt(mx.sum(step * step, axis=1, keepdims=True))
    step = step * mx.minimum(1.0, cap / mx.maximum(norm, 1e-12))
    return Y + step * alpha


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

    if not _is_usable(Y):
        warnings.warn(
            "Spectral initialization produced no usable coordinates; falling back to a "
            "random initialization.", UserWarning, stacklevel=3,
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


def _spectral_layout_arpack(graph, dim: int, seed: int):
    """Shifted-operator eigenvectors via ARPACK. Returns ``None`` if it does not converge."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    n = graph.shape[0]
    k = dim + 1
    if n <= k + 1:                               # ARPACK needs k < n-1
        return None
    deg = np.maximum(np.asarray(graph.sum(axis=1)).ravel(), 1e-10)
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    shifted = (sp.identity(n, format="csr") + d_inv_sqrt @ graph @ d_inv_sqrt) * 0.5
    try:
        vals, vecs = eigsh(shifted, k=k, which="LM", tol=1e-4, maxiter=n * 5,
                           v0=np.random.RandomState(seed).normal(size=n))
    except Exception:                            # ARPACK non-convergence etc.
        return None
    keep = np.argsort(vals)[::-1][1:k]            # drop the trivial (largest) mode
    return np.asarray(vecs[:, keep], dtype=np.float32)


def _spectral_layout(graph, dim: int, seed: int, n_iter: int = 200,
                     tol: float = 1e-7) -> np.ndarray:
    """The ``dim`` lowest non-trivial eigenvectors of the normalized Laplacian.

    Both paths iterate the **shifted** operator ``(I + D^-1/2 A D^-1/2) / 2``, whose dominant
    eigenvectors are the Laplacian's low-frequency modes. The shift matters: on the unshifted
    operator an eigenvalue near ``-1`` (a near-bipartite subgraph) has the same magnitude as
    the mode we want, so the iteration can converge to an oscillatory mode instead.

    Small graphs go to ARPACK, large ones to the GPU. A disconnected atlas is laid out one
    component at a time, and components are typically a few thousand cells each — at that
    size the GPU orthogonal iteration is dominated by per-launch overhead (measured 110 ms
    for a 2,000-cell component, ~25× what ARPACK needs for the same subproblem), so the
    crossover is what keeps the multi-component path from costing more than the layout.
    """
    if graph.shape[0] <= _ARPACK_MAX_N:
        result = _spectral_layout_arpack(graph, dim, seed)
        if result is not None:
            return result                       # None => ARPACK failed; fall through to GPU

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

    V = mx.random.normal((n, k), key=mx.random.key(seed))   # explicit key, not global state
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


def _is_usable(Y: np.ndarray) -> bool:
    """False only when the init is unusable outright: non-finite, or zero extent on an axis.

    This deliberately does NOT try to judge whether a *finite* init is "too concentrated".
    An earlier version did, on the theory that a localized init was what tore the layout
    apart. It wasn't — the unbounded per-node step in the SGD was (see ``_optimize_layout``),
    and with a trust region in place the optimizer recovers from a concentrated start on its
    own. Measured after the fix: the threshold never fired on the issue-#1 reproduction, and
    where it did fire it replaced a good multi-component layout with a random one and made
    the result slightly worse (core 0.795 → 0.755). So the heuristic is gone and only the
    unambiguous failure is caught.
    """
    if not np.all(np.isfinite(Y)):
        return False
    return bool(np.all((Y.max(axis=0) - Y.min(axis=0)) > 0))
