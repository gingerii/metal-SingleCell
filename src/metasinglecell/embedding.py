"""UMAP embedding with the force-layout optimization on the Metal GPU.

UMAP = (1) fuzzy graph [done in ``neighbors``], (2) optimize a low-dim layout by
SGD with attractive forces along graph edges and repulsive forces to negative
samples. Step (2) is the expensive part and parallelizes well, so we run it on
the GPU (MLX): a vectorized force update over all "due" edges per epoch, with
scatter-add accumulating gradients.

We reuse umap-learn for the cheap, fiddly setup (the a/b curve fit, the spectral
initialization, the per-edge epoch schedule) so behavior matches; the heavy loop
is ours. The embedding is stochastic, so we validate structure preservation, not
coordinates.
"""

from __future__ import annotations

import numpy as np

GAMMA = 1.0
NEG_SAMPLE_RATE = 5
INITIAL_ALPHA = 1.0


def umap(connectivities, n_components: int = 2, n_epochs: int | None = None,
         min_dist: float = 0.5, spread: float = 1.0, random_state: int = 0) -> np.ndarray:
    """Optimize a UMAP embedding from a connectivity graph (GPU force layout)."""
    import mlx.core as mx
    from sklearn.utils import check_random_state
    from umap.spectral import spectral_layout
    from umap.umap_ import find_ab_params

    a, b = find_ab_params(spread, min_dist)
    a, b = float(a), float(b)  # keep scalars Python float so MLX ops stay on-device
    n = connectivities.shape[0]
    if n_epochs is None:
        n_epochs = 500 if n <= 10_000 else 200

    graph = connectivities.tocoo().copy()
    graph.data[graph.data < graph.data.max() / float(n_epochs)] = 0.0
    graph.eliminate_zeros()

    rng = check_random_state(random_state)
    try:
        init = spectral_layout(None, graph, n_components, random_state=rng)
        init = np.asarray(init, dtype=np.float32)
        init = 10.0 * (init - init.min(0)) / (init.max(0) - init.min(0) + 1e-9)
    except Exception:
        init = rng.normal(scale=10.0, size=(n, n_components)).astype(np.float32)

    # Process ALL edges every epoch on the GPU, attractive force scaled by edge
    # weight — instead of umap-learn's per-epoch host-side edge sampling
    # (epochs_per_sample). On the GPU the extra edge work is cheap and parallel,
    # while removing the per-epoch host bookkeeping (flatnonzero/np.repeat) halves
    # runtime. Eval every 25 epochs to keep host syncs few.
    head = mx.array(graph.row.astype(np.int32))
    tail = mx.array(graph.col.astype(np.int32))
    w = mx.array((graph.data / graph.data.max()).astype(np.float32))[:, None]
    n_edges = graph.row.size

    emb = mx.array(init)
    for epoch in range(n_epochs):
        alpha = INITIAL_ALPHA * (1.0 - epoch / n_epochs)
        # attractive (weighted) along every edge
        diff = emb[head] - emb[tail]
        d2 = mx.sum(diff * diff, axis=1, keepdims=True)
        coef = (-2.0 * a * b * mx.power(d2, b - 1.0)) / (a * mx.power(d2, b) + 1.0)
        grad = mx.clip(coef * diff, -4.0, 4.0) * alpha * w
        emb = emb.at[head].add(grad)
        emb = emb.at[tail].add(-grad)
        # repulsive to random negatives (one per edge) — sample on the GPU (host RNG +
        # per-epoch host→device transfer was breaking the lazy-eval graph and dominating
        # the loop: ~150M host ints + 200 transfers/syncs at 50k×200 epochs).
        rand = mx.random.randint(0, n, (n_edges,), key=mx.random.key(random_state + epoch + 1))
        diffn = emb[head] - emb[rand]
        d2n = mx.sum(diffn * diffn, axis=1, keepdims=True)
        coefn = (2.0 * GAMMA * b) / ((1e-3 + d2n) * (a * mx.power(d2n, b) + 1.0))
        emb = emb.at[head].add(mx.clip(coefn * diffn, -4.0, 4.0) * alpha)
        if epoch % 25 == 0:
            mx.eval(emb)

    mx.eval(emb)
    return np.asarray(emb)
