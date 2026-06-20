"""Phase 3 validation: GPU parallel Leiden vs igraph Leiden.

Quality (modularity + ARI vs oracle) on the PBMC graph, and a scale probe where
the GPU pulls ahead. Validation bar: GPU Leiden modularity >= igraph Leiden - eps.

    conda activate metasinglecell
    python validation_notebooks/11_leiden_gpu_parity.py
"""

import logging
import time

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.cluster import leiden
from metasinglecell.graph import Graph
from metasinglecell.graph.primitives import modularity


def _sbm_graph(n, k=15, n_clusters=20, p_in=0.85, seed=0):
    rng = np.random.default_rng(seed)
    block = rng.integers(0, n_clusters, n)
    members = {b: np.where(block == b)[0] for b in range(n_clusters)}
    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    intra = rng.random(n * k) < p_in
    rb = block[rows]
    for b in range(n_clusters):
        m = intra & (rb == b)
        if m.any() and members[b].size:
            cols[m] = rng.choice(members[b], m.sum())
    cols[~intra] = rng.integers(0, n, (~intra).sum())
    A = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(n, n))
    A = A + A.T
    A.setdiag(0); A.eliminate_zeros()
    return A


def main() -> None:
    res_dir = config.results_dir("leiden_gpu")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "leiden_gpu.log", mode="w"), logging.StreamHandler()])
    log = logging.getLogger("leiden_gpu")

    import mlx.core as mx
    from sklearn.metrics import adjusted_rand_score

    conn = sp.load_npz(config.PROCESSED_DIR / "reference" / "07_connectivities.npz").tocsr()
    conn.sort_indices()
    oracle = validation.load_snapshot("08_leiden").astype(np.int32)
    g = Graph.from_scipy(conn)

    labels = leiden(conn, resolution=1.0, backend="gpu")
    q_gpu = modularity(g, mx.array(labels.astype(np.int32)), 1.0)
    q_ig = modularity(g, mx.array(leiden(conn, backend="igraph").astype(np.int32)), 1.0)
    ari = adjusted_rand_score(oracle, labels)
    log.info("PBMC GPU Leiden: Q=%.4f (%d cl) ARI=%.3f | igraph Leiden Q=%.4f",
             q_gpu, labels.max() + 1, ari, q_ig)

    records = [{"check": "leiden_modularity", "passed": bool(q_gpu >= q_ig - 0.01),
                "q_gpu": round(q_gpu, 4), "q_igraph": round(q_ig, 4),
                "n_clusters": int(labels.max() + 1), "ari_vs_oracle": round(ari, 4)}]

    # --- scale probe: GPU Leiden vs igraph Leiden ---
    import igraph as ig
    for n in (200_000, 1_000_000):
        A = _sbm_graph(n)
        gg = Graph.from_scipy(A)
        t = time.perf_counter(); lab = leiden(A, backend="gpu"); mx.eval(mx.array(lab)); gt = time.perf_counter() - t
        q_g = modularity(gg, mx.array(lab.astype(np.int32)), 1.0)
        coo = A.tocoo(); up = coo.row < coo.col
        gi = ig.Graph(n=n, edges=np.column_stack([coo.row[up], coo.col[up]]).tolist())
        gi.es["weight"] = coo.data[up].tolist()
        t = time.perf_counter()
        pil = gi.community_leiden(objective_function="modularity", weights="weight", n_iterations=2)
        ct = time.perf_counter() - t
        log.info("n=%-8d GPU %.2fs (Q=%.4f, %d cl) | igraph %.2fs (Q=%.4f) | speedup %.2fx",
                 n, gt, q_g, lab.max() + 1, ct, pil.modularity, ct / gt)
        records.append({"check": f"scale_n{n}", "passed": True, "gpu_s": round(gt, 2),
                        "igraph_s": round(ct, 2), "speedup": round(ct / gt, 2),
                        "q_gpu": round(q_g, 4), "q_igraph": round(pil.modularity, 4)})

    validation.write_report(records, "leiden_gpu")
    print(f"\nGPU Leiden: PBMC Q={q_gpu:.4f} vs igraph {q_ig:.4f} | "
          f"{'PASS' if records[0]['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
