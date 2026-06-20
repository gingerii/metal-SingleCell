"""Graph clustering (scanpy ``sc.tl.leiden``).

Leiden is graph community detection — an irregular, largely sequential algorithm
that does not GPU-accelerate without a specialized graph engine (NVIDIA cuGraph
has no Metal equivalent). scanpy itself just calls igraph on CPU, and so do we.
The GPU value at this stage is upstream: building the neighbor graph fast. This
is the correct drop-in, run on our GPU-built connectivity graph.
"""

from __future__ import annotations

import numpy as np


def leiden(connectivities, resolution: float = 1.0, random_state: int = 0,
           n_iterations: int = 2) -> np.ndarray:
    """Leiden clustering on a connectivity graph; returns integer labels.

    Mirrors scanpy's default ``flavor="igraph"``: igraph ``community_leiden`` with
    the modularity objective, weighted, ``n_iterations=2``. ``connectivities`` is a
    symmetric scipy sparse graph.
    """
    import random as _random

    import igraph as ig

    coo = connectivities.tocoo()
    upper = coo.row < coo.col  # undirected: keep each edge once
    edges = np.column_stack([coo.row[upper], coo.col[upper]])

    _random.seed(random_state)
    ig.set_random_number_generator(_random)  # igraph wants a random-module-like RNG
    g = ig.Graph(n=connectivities.shape[0], edges=edges.tolist())
    g.es["weight"] = coo.data[upper].tolist()

    part = g.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=n_iterations,
    )
    return np.asarray(part.membership)
