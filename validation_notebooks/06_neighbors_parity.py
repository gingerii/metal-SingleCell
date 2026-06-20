"""Stage 2 validation: Metal k-NN graph vs the fp64 CPU oracle.

Builds the neighbor graph from the oracle PCA embedding (``06_X_pca``) and
compares against the oracle graphs (``07_distances``/``07_connectivities``).
scanpy's k-NN can be approximate (pynndescent), so the meaningful check is the
per-cell neighbor-set overlap (ours is exact); connectivities are compared on
their shared support.

    conda activate metasinglecell
    python validation_notebooks/06_neighbors_parity.py
"""

import logging

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.neighbors import neighbors
from metasinglecell.reference import N_NEIGHBORS


def main() -> None:
    res_dir = config.results_dir("neighbors")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "neighbors.log", mode="w"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("neighbors")

    X_pca = validation.load_snapshot("06_X_pca")
    ref_dist = sp.load_npz(config.PROCESSED_DIR / "reference" / "07_distances.npz").tocsr()
    ref_conn = sp.load_npz(config.PROCESSED_DIR / "reference" / "07_connectivities.npz").tocsr()

    dist, conn = neighbors(X_pca, n_neighbors=N_NEIGHBORS)

    # Per-cell neighbor-set overlap (exclude self): how many of scanpy's neighbors
    # our exact k-NN recovers.
    n = X_pca.shape[0]
    overlaps = []
    for i in range(n):
        mine = set(dist.indices[dist.indptr[i]:dist.indptr[i + 1]])
        ref = set(ref_dist.indices[ref_dist.indptr[i]:ref_dist.indptr[i + 1]])
        overlaps.append(len(mine & ref) / max(len(ref), 1))
    overlaps = np.array(overlaps)
    log.info("k-NN neighbor overlap vs oracle: mean=%.4f  median=%.4f  min=%.4f",
             overlaps.mean(), np.median(overlaps), overlaps.min())

    # Connectivities: correlation on the union of supports (graph edge weights).
    both = (conn != 0).multiply(ref_conn != 0)
    edge_overlap = both.nnz / max((conn != 0).nnz, 1)
    mask = ref_conn.copy(); mask.data[:] = 1
    common = conn.multiply(mask)
    r = validation.compare("connectivities(shared support)",
                            common.data if common.nnz else np.zeros(1),
                            ref_conn.multiply((conn != 0).astype(float)).data
                            if ref_conn.nnz else np.zeros(1),
                            rtol=1e-2, atol=1e-2)
    log.info("connectivities: edge-support overlap=%.4f", edge_overlap)

    record = {
        "check": "knn_overlap", "passed": bool(overlaps.mean() > 0.95),
        "mean_overlap": float(overlaps.mean()), "median_overlap": float(np.median(overlaps)),
        "min_overlap": float(overlaps.min()), "edge_support_overlap": float(edge_overlap),
    }
    validation.write_report([record], "neighbors")
    print(f"\nneighbors parity: {'PASS' if record['passed'] else 'FAIL'} "
          f"(mean k-NN overlap {overlaps.mean():.3f})")


if __name__ == "__main__":
    main()
