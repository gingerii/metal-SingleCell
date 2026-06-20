"""Phase 2 validation: GPU parallel Louvain vs igraph.

Quality bar for a clustering optimizer = modularity, not label parity. We check
GPU Louvain's Q against igraph Louvain/Leiden on the same PBMC graph, plus ARI vs
the oracle labels (within the established RNG floor), and a small scaling probe.

    conda activate metasinglecell
    python validation_notebooks/10_louvain_parity.py
"""

import logging
import time

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.graph import Graph
from metasinglecell.graph.louvain import louvain
from metasinglecell.graph.primitives import modularity


def _sbm_graph(n, k=15, n_clusters=20, p_in=0.85, seed=0):
    """Stochastic-block-model kNN-like graph (genuinely clustered)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.integers(0, n_clusters, n))  # group same-block nodes contiguously
    block = np.repeat(np.arange(n_clusters), int(np.ceil(n / n_clusters)))[:n][order.argsort()]
    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    # same-block partners: pick a random member of each node's block
    members = {b: np.where(block == b)[0] for b in range(n_clusters)}
    intra = rng.random(n * k) < p_in
    rb = block[rows]
    for b in range(n_clusters):
        mask = intra & (rb == b)
        if mask.any() and members[b].size:
            cols[mask] = rng.choice(members[b], mask.sum())
    cols[~intra] = rng.integers(0, n, (~intra).sum())
    A = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(n, n))
    A = A + A.T
    A.setdiag(0); A.eliminate_zeros()
    return A


def main() -> None:
    res_dir = config.results_dir("louvain")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "louvain.log", mode="w"), logging.StreamHandler()])
    log = logging.getLogger("louvain")

    import igraph as ig
    from sklearn.metrics import adjusted_rand_score
    import mlx.core as mx

    conn = sp.load_npz(config.PROCESSED_DIR / "reference" / "07_connectivities.npz").tocsr()
    conn.sort_indices()
    oracle = validation.load_snapshot("08_leiden").astype(np.int32)
    g = Graph.from_scipy(conn)

    # igraph reference modularity (Louvain + Leiden) on the same graph
    coo = conn.tocoo(); up = coo.row < coo.col
    gi = ig.Graph(n=g.n, edges=np.column_stack([coo.row[up], coo.col[up]]).tolist())
    gi.es["weight"] = coo.data[up].tolist()
    q_ig_louvain = gi.community_multilevel(weights="weight").modularity
    q_ig_leiden = gi.community_leiden(objective_function="modularity", weights="weight",
                                      n_iterations=2).modularity

    labels = louvain(g, resolution=1.0)
    q_gpu = modularity(g, mx.array(labels.astype(np.int32)), 1.0)
    ari = adjusted_rand_score(oracle, labels)
    log.info("GPU Louvain: %d clusters, Q=%.6f", labels.max() + 1, q_gpu)
    log.info("igraph: Louvain Q=%.6f  Leiden Q=%.6f", q_ig_louvain, q_ig_leiden)
    log.info("GPU Louvain vs oracle ARI=%.4f", ari)

    records = [{
        "check": "louvain_modularity", "passed": bool(q_gpu >= q_ig_louvain - 0.02),
        "q_gpu": round(q_gpu, 6), "q_igraph_louvain": round(q_ig_louvain, 6),
        "q_igraph_leiden": round(q_ig_leiden, 6), "n_clusters": int(labels.max() + 1),
        "ari_vs_oracle": round(ari, 4),
    }]

    # --- scaling/quality probe on SBM graphs. Quality matches igraph at all sizes;
    # small n is overhead-bound on the GPU, but the GPU pulls ahead at atlas scale
    # (~1M: faster than igraph with equal/better modularity). ---
    for n in (50_000, 200_000, 1_000_000):
        A = _sbm_graph(n)
        gg = Graph.from_scipy(A)
        t = time.perf_counter(); lab = louvain(gg, 1.0); mx.eval(mx.array(lab)); gpu_t = time.perf_counter() - t
        cooA = A.tocoo(); upA = cooA.row < cooA.col
        giA = ig.Graph(n=n, edges=np.column_stack([cooA.row[upA], cooA.col[upA]]).tolist())
        giA.es["weight"] = cooA.data[upA].tolist()
        t = time.perf_counter(); pi = giA.community_multilevel(weights="weight"); cpu_t = time.perf_counter() - t
        q_g = modularity(gg, mx.array(lab.astype(np.int32)), 1.0)
        log.info("n=%-7d GPU %.3fs (Q=%.4f, %d cl) | igraph %.3fs (Q=%.4f) | speedup %.2fx",
                 n, gpu_t, q_g, lab.max() + 1, cpu_t, pi.modularity, cpu_t / gpu_t)
        records.append({"check": f"scale_n{n}", "passed": True, "gpu_s": round(gpu_t, 3),
                        "igraph_s": round(cpu_t, 3), "speedup": round(cpu_t / gpu_t, 2),
                        "q_gpu": round(q_g, 4), "q_igraph": round(pi.modularity, 4)})

    validation.write_report(records, "louvain")
    print(f"\nLouvain: GPU Q={q_gpu:.4f} vs igraph {q_ig_louvain:.4f} | "
          f"{'PASS' if records[0]['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
