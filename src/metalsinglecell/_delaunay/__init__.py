"""GPU Delaunay triangulation (gDel2D-style), Metal/MLX.

Nothing here is public API yet. ``gr.spatial_neighbors_delaunay`` still routes through
Qhull; this package is being built up underneath it, component by component, each one
pinned against an exact oracle before the next is written.
"""

from .predicates import (  # noqa: F401
    MAX_COORD,
    SAFE_ABS,
    condition_points,
    incircle,
    incircle_gpu,
    orient2d,
)
