"""GPU graph layer for Apple Metal — a cuGraph-analog, clustering-first.

Parallel graph clustering (Louvain → Leiden) on the Metal GPU via MLX, built on a
reusable sparse-graph substrate (CSR graph container + sort-based segment
reductions). Targets the atlas-scale clustering bottleneck that is sequential on
CPU. Heavy deps (mlx) are lazy-imported inside functions.
"""

from .csr_graph import Graph

__all__ = ["Graph"]
