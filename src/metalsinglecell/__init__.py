"""metal-SingleCell — GPU-accelerated single-cell analysis on Apple Silicon.

A Metal/MLX re-implementation of rapids-singlecell's scanpy drop-ins.
Heavy backends (mlx, scanpy) are lazy-imported inside functions so this
package imports cleanly in any environment.
"""

__version__ = "0.1.0"

from . import config, gr, pp, tl

__all__ = ["config", "pp", "tl", "gr", "__version__"]
