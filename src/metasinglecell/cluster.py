"""Graph clustering (scanpy ``sc.tl.leiden``).

Two backends:
* ``"gpu"`` — our parallel Leiden on the Metal GPU (``graph.leiden``): faster than
  igraph at atlas scale with matching/higher modularity. The cuGraph-analog.
* ``"igraph"`` — igraph ``community_leiden`` (modularity), matching scanpy's
  ``flavor="igraph"``. CPU; the reference.
"""

from __future__ import annotations

import numpy as np


def leiden(connectivities, resolution: float = 1.0, random_state: int = 0,
           n_iterations: int = 2, backend: str = "igraph") -> np.ndarray:
    """Leiden clustering on a symmetric connectivity graph; returns integer labels.

    ``backend="gpu"`` uses the Metal parallel Leiden; ``"igraph"`` (default) uses
    igraph on CPU.
    """
    if backend == "gpu":
        from .graph import Graph
        from .graph.leiden import leiden as gpu_leiden

        g = Graph.from_scipy(connectivities)
        return gpu_leiden(g, resolution=resolution, random_state=random_state,
                          n_iterations=n_iterations)

    if backend != "igraph":
        raise ValueError(f"unknown backend {backend!r} (gpu|igraph)")

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
