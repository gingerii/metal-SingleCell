"""Graph clustering (scanpy ``sc.tl.leiden``).

Two backends:
* ``"gpu"`` — our parallel Leiden on the Metal GPU (``graph.leiden``): faster than
  igraph at atlas scale with matching/higher modularity. The cuGraph-analog.
* ``"igraph"`` — igraph ``community_leiden`` (modularity), matching scanpy's
  ``flavor="igraph"``. CPU; the reference.
"""

from __future__ import annotations

import contextlib
import numpy as np


@contextlib.contextmanager
def _seeded_igraph(random_state):
    """Seed igraph reproducibly without touching the process-global ``random`` stream.

    ``random.seed(); ig.set_random_number_generator(random)`` makes the run reproducible and
    also silently reseeds the stdlib generator for the rest of the session, so unrelated user
    code that draws from ``random`` changes behaviour after a clustering call. scanpy avoids
    exactly this with its own ``_RNGIgraph`` wrapper. Hand igraph a private instance and put
    the default back afterwards.
    """
    import random as _random

    import igraph as ig
    try:
        ig.set_random_number_generator(_random.Random(random_state))
        yield
    finally:
        ig.set_random_number_generator(_random)


def leiden(connectivities, resolution: float = 1.0, random_state: int = 0,
           n_iterations: int = 2, backend: str = "igraph",
           variant: str = "sync", commit_prob: float = 0.9) -> np.ndarray:
    """Leiden clustering on a symmetric connectivity graph; returns integer labels.

    ``backend="gpu"`` uses the Metal parallel Leiden; ``"igraph"`` (default) uses
    igraph on CPU. ``variant`` ("sync"|"colored") and ``commit_prob`` tune the GPU path.
    """
    if backend == "gpu":
        from .graph import Graph
        from .graph.leiden import leiden as gpu_leiden

        # Honor the caller's n_iterations (scanpy default 2). One multilevel pass reaches a
        # fixed point on clean/mid graphs (ARI 1.000 for n_iter 1 vs 2 there), but at ≥~1M
        # cells the fuzzy graph can be under-converged at a single pass (a 2nd iteration closes
        # the gap) — so silently clamping to 1 traded quality for speed at the target scale.
        g = Graph.from_scipy(connectivities)
        return gpu_leiden(g, resolution=resolution, random_state=random_state,
                          n_iterations=n_iterations, variant=variant, commit_prob=commit_prob)

    if backend != "igraph":
        raise ValueError(f"unknown backend {backend!r} (gpu|igraph)")

    import igraph as ig

    coo = connectivities.tocoo()
    upper = coo.row < coo.col  # undirected: keep each edge once
    edges = np.column_stack([coo.row[upper], coo.col[upper]])

    with _seeded_igraph(random_state):
        g = ig.Graph(n=connectivities.shape[0], edges=edges.tolist())
        g.es["weight"] = coo.data[upper].tolist()

        part = g.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=resolution,
            n_iterations=n_iterations,
        )
    return np.asarray(part.membership)
