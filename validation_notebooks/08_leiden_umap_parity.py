"""Stage 2 validation: leiden (CPU drop-in) + GPU UMAP vs the fp64 oracle.

leiden: cluster the oracle connectivity graph; agreement vs oracle labels by ARI.
umap: optimize a 2-D layout on the GPU; since the embedding is stochastic we
check local-structure preservation (fraction of each cell's graph neighbors that
land among its nearest neighbors in 2-D) and compare it to umap-learn's own
embedding. We also time the layout optimization GPU vs umap-learn.

    conda activate metasinglecell
    python validation_notebooks/08_leiden_umap_parity.py
"""

import logging
import time

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.cluster import leiden
from metasinglecell.embedding import umap
from metasinglecell.reference import LEIDEN_RES


def _graph_neighbors(dist_csr):
    return [set(dist_csr.indices[dist_csr.indptr[i]:dist_csr.indptr[i + 1]])
            for i in range(dist_csr.shape[0])]


def _neighbor_preservation(emb, ref_nbrs, k=15):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]
    return float(np.mean([len(set(idx[i]) & ref_nbrs[i]) / max(len(ref_nbrs[i]), 1)
                          for i in range(emb.shape[0])]))


def main() -> None:
    res_dir = config.results_dir("leiden_umap")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "leiden_umap.log", mode="w"), logging.StreamHandler()])
    log = logging.getLogger("leiden_umap")

    from sklearn.metrics import adjusted_rand_score

    refd = config.PROCESSED_DIR / "reference"
    conn = sp.load_npz(refd / "07_connectivities.npz").tocsr()
    dist = sp.load_npz(refd / "07_distances.npz").tocsr()
    X_pca = validation.load_snapshot("06_X_pca")
    leiden_ref = validation.load_snapshot("08_leiden")
    ref_nbrs = _graph_neighbors(dist)

    records = []

    # --- leiden: ARI vs oracle ---
    # Leiden is stochastic; scanpy's OWN seed-to-seed ARI vs this oracle spans
    # ~0.69-0.90 (measured). So the meaningful check is cluster-count match plus
    # ARI within that RNG floor (>=0.65), not exact agreement.
    RNG_FLOOR = 0.65
    labels = leiden(conn, resolution=LEIDEN_RES)
    ari = adjusted_rand_score(leiden_ref, labels)
    same_k = (labels.max() + 1) == int(leiden_ref.max() + 1)
    log.info("leiden: %d clusters (oracle %d), ARI=%.4f (RNG floor ~0.69-0.90 for scanpy itself)",
             labels.max() + 1, int(leiden_ref.max() + 1), ari)
    records.append({"check": "leiden_ARI", "passed": bool(ari >= RNG_FLOOR and same_k),
                    "ari": round(ari, 4), "n_clusters": int(labels.max() + 1),
                    "n_clusters_oracle": int(leiden_ref.max() + 1), "rng_floor": RNG_FLOOR})

    # --- umap: structure preservation + timing ---
    t = time.perf_counter(); emb = umap(conn); gpu_t = time.perf_counter() - t
    pres_gpu = _neighbor_preservation(emb, ref_nbrs)

    import umap as umap_learn
    t = time.perf_counter()
    ref_emb = umap_learn.UMAP(n_neighbors=15, min_dist=0.5, random_state=0).fit_transform(X_pca)
    cpu_t = time.perf_counter() - t
    pres_cpu = _neighbor_preservation(ref_emb, ref_nbrs)

    log.info("umap: neighbor-preservation ours=%.4f  umap-learn=%.4f", pres_gpu, pres_cpu)
    log.info("umap timing: GPU layout=%.3fs  umap-learn full(graph+layout)=%.3fs", gpu_t, cpu_t)
    # PASS if our preservation is within 80%% of umap-learn's (structurally comparable).
    records.append({"check": "umap_structure", "passed": bool(pres_gpu >= 0.8 * pres_cpu),
                    "preservation_gpu": round(pres_gpu, 4), "preservation_umaplearn": round(pres_cpu, 4),
                    "gpu_layout_s": round(gpu_t, 3), "umaplearn_full_s": round(cpu_t, 3)})

    validation.write_report(records, "leiden_umap")
    print(f"\nleiden ARI={ari:.3f} | umap preservation ours={pres_gpu:.3f} vs umap-learn={pres_cpu:.3f}")
    print(f"PASS: {all(r['passed'] for r in records)}")


if __name__ == "__main__":
    main()
