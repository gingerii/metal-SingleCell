"""Phase 1 validation: GPU graph primitives vs NumPy/igraph ground truth.

Deterministic reductions, so these must match exactly (fp tolerance). Run on the
PBMC neighbor graph (07_connectivities) with the oracle leiden labels (08_leiden).

    conda activate metasinglecell
    python validation_notebooks/09_graph_primitives_parity.py
"""

import logging

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.graph import Graph
from metasinglecell.graph.primitives import (contract, modularity,
                                             neighbor_community_weights)


def main() -> None:
    res_dir = config.results_dir("graph_primitives")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "graph_primitives.log", mode="w"), logging.StreamHandler()])
    log = logging.getLogger("graph_prim")

    import mlx.core as mx

    conn = sp.load_npz(config.PROCESSED_DIR / "reference" / "07_connectivities.npz").tocsr()
    conn.sort_indices()
    labels = validation.load_snapshot("08_leiden").astype(np.int32)
    g = Graph.from_scipy(conn)
    comm = mx.array(labels)
    records = []

    # --- (1) degrees == scipy row sums ---
    deg = np.asarray(g.degrees())
    deg_gt = np.asarray(conn.sum(axis=1)).ravel()
    r = validation.compare("degrees", deg, deg_gt, rtol=1e-5, atol=1e-4); records.append(r)
    log.info("degrees           max_abs_err=%.3g passed=%s", r["max_abs_err"], r["passed"])

    # --- (2) neighbor_community_weights == dense brute force ---
    src, com, w = neighbor_community_weights(g, comm)
    src, com, w = np.asarray(src), np.asarray(com), np.asarray(w)
    C = labels.max() + 1
    mine = np.zeros((g.n, C)); mine[src, com] = w
    gt = np.zeros((g.n, C))
    cx = conn.tocoo()
    np.add.at(gt, (cx.row, labels[cx.col]), cx.data)
    r = validation.compare("neighbor_comm_weights", mine, gt, rtol=1e-4, atol=1e-4); records.append(r)
    log.info("neighbor_comm_wts max_abs_err=%.3g passed=%s", r["max_abs_err"], r["passed"])

    # --- (3) contract == dense community-community aggregation ---
    cg = contract(g, comm)
    contracted = sp.csr_matrix((np.asarray(cg.weights),
                                (np.asarray(cg.edge_src), np.asarray(cg.indices))),
                               shape=(C, C)).toarray()
    gt_c = np.zeros((C, C))
    np.add.at(gt_c, (labels[cx.row], labels[cx.col]), cx.data)
    r = validation.compare("contract", contracted, gt_c, rtol=1e-4, atol=1e-3); records.append(r)
    log.info("contract          max_abs_err=%.3g passed=%s (C=%d super-nodes)",
             r["max_abs_err"], r["passed"], C)

    # --- (4) modularity == igraph ---
    import igraph as ig
    coo = conn.tocoo(); up = coo.row < coo.col
    gi = ig.Graph(n=g.n, edges=np.column_stack([coo.row[up], coo.col[up]]).tolist())
    gi.es["weight"] = coo.data[up].tolist()
    q_ig = gi.modularity(labels, weights="weight")
    q_mine = modularity(g, comm, resolution=1.0)
    r = {"check": "modularity", "passed": bool(abs(q_mine - q_ig) < 1e-4),
         "mine": round(q_mine, 6), "igraph": round(q_ig, 6), "abs_err": abs(q_mine - q_ig)}
    records.append(r)
    log.info("modularity        mine=%.6f igraph=%.6f abs_err=%.2g passed=%s",
             q_mine, q_ig, r["abs_err"], r["passed"])

    validation.write_report(records, "graph_primitives")
    print(f"\ngraph primitives parity: {'PASS' if all(x['passed'] for x in records) else 'FAIL'}")


if __name__ == "__main__":
    main()
