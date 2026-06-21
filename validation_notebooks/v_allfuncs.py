"""Validation: speedup table for EVERY function (GPU/ours vs CPU reference).

One representative size per function (50k cells for scalable pp/tl; smaller for
O(n²) spatial/tsne). Each row: GPU time, CPU-reference time, speedup, accuracy.
CPU baselines: scanpy / sklearn / igraph / umap-learn (squidpy/harmonypy/fa2/bbknn
unavailable -> NumPy reference or "no baseline").

    conda activate metasinglecell
    python validation_notebooks/v_allfuncs.py
"""

import logging
import time
import warnings

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation

warnings.filterwarnings("ignore")
N = 50_000
N_SP = 8_000        # spatial / O(n²) ops
N_GENES = 2000


def _best(fn, repeats=3, warmup=True):
    if warmup:
        try: fn()
        except Exception: pass
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


def main():
    res_dir = config.results_dir("validation")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(res_dir / "v_allfuncs.log", mode="w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("v")
    import mlx.core as mx
    import scanpy as sc

    rng = np.random.default_rng(0)
    records = []

    def row(op, size, gpu_s, cpu_s, acc, baseline):
        sp_ = (cpu_s / gpu_s) if (cpu_s == cpu_s and gpu_s > 0) else float("nan")
        records.append({"function": op, "n": size, "gpu_s": round(gpu_s, 4),
                        "cpu_s": round(cpu_s, 4) if cpu_s == cpu_s else None,
                        "speedup": round(sp_, 2) if sp_ == sp_ else None,
                        "accuracy": acc, "cpu_baseline": baseline})
        log.info("%-22s n=%-6d gpu=%8.4fs cpu=%9s  speedup=%8s  %s",
                 op, size, gpu_s, ("%.4f" % cpu_s) if cpu_s == cpu_s else "n/a",
                 ("%.2fx" % sp_) if sp_ == sp_ else "n/a", acc)

    def bench(op, size, gpu_fn, cpu_fn, acc="", baseline="", grepeat=3):
        try:
            g = _best(gpu_fn, repeats=grepeat)
            c = _best(cpu_fn, repeats=1) if cpu_fn else float("nan")
            row(op, size, g, c, acc, baseline)
        except Exception as e:
            log.info("%-22s FAILED: %s", op, str(e)[:70])

    # ---------- shared artifacts ----------
    counts = sp.random(N, N_GENES, density=0.07, format="csr", random_state=0)
    counts.data = rng.integers(1, 50, counts.data.size).astype(np.float32)
    from metasinglecell.sparse import CSR
    csr = CSR.from_scipy(counts)
    lognorm = csr.normalize_total(1e4).log1p()
    from metasinglecell.preprocess import (calculate_qc_metrics, filter_cells, highly_variable_genes,
                                           normalize_pearson_residuals, regress_out, scale, scrublet)
    hvdf = highly_variable_genes(lognorm, n_top_genes=1000)
    hv_idx = np.flatnonzero(hvdf["highly_variable"].to_numpy())[:1000]
    dense_hv = lognorm.toarray()[:, hv_idx].astype(np.float32)
    from metasinglecell.decomposition import pca
    emb = pca(dense_hv, n_comps=50, solver="randomized")[0].astype(np.float32)
    ad = sc.AnnData(counts.copy())
    adl = ad.copy(); sc.pp.normalize_total(adl, target_sum=1e4); sc.pp.log1p(adl)

    # ---------- pp ----------
    bench("normalize_total+log1p", N, lambda: csr.normalize_total(1e4).log1p().data,
          lambda: (lambda a: (sc.pp.normalize_total(a, target_sum=1e4), sc.pp.log1p(a)))(ad.copy()),
          "exact", "scanpy")
    bench("calculate_qc_metrics", N, lambda: calculate_qc_metrics(counts),
          lambda: sc.pp.calculate_qc_metrics(ad.copy(), percent_top=None, log1p=False, inplace=True),
          "exact", "scanpy")
    bench("filter_cells", N, lambda: filter_cells(counts, min_genes=200),
          lambda: sc.pp.filter_cells(ad.copy(), min_genes=200, inplace=False), "exact", "scanpy")
    bench("highly_variable_genes", N, lambda: highly_variable_genes(lognorm, n_top_genes=1000),
          lambda: sc.pp.highly_variable_genes(adl.copy(), n_top_genes=1000, flavor="seurat"),
          "overlap~1", "scanpy")
    bench("scale", N, lambda: scale(CSR.from_scipy(sp.csr_matrix(dense_hv))),
          lambda: sc.pp.scale(sc.AnnData(dense_hv.copy()), max_value=10), "exact", "scanpy")
    bench("pca_randomized", N, lambda: pca(dense_hv, n_comps=50, solver="randomized"),
          lambda: __import__("sklearn.utils.extmath", fromlist=["randomized_svd"]).randomized_svd(
              dense_hv.astype(np.float64) - dense_hv.mean(0), 50, n_iter=7, random_state=0),
          "exact", "sklearn")
    bench("regress_out", 8000, lambda: regress_out(dense_hv[:8000], dense_hv[:8000, 0]),
          lambda: sc.pp.regress_out(sc.AnnData(dense_hv[:8000].copy(),
                  obs={"c": dense_hv[:8000, 0]}), ["c"]), "corr~1", "scanpy", grepeat=2)
    bench("normalize_pearson_residuals", N, lambda: normalize_pearson_residuals(counts),
          lambda: sc.experimental.pp.normalize_pearson_residuals(ad.copy(), inplace=False),
          "exact", "scanpy.exp", grepeat=2)
    from metasinglecell.neighbors import neighbors, bbknn
    bench("neighbors", N, lambda: neighbors(emb, n_neighbors=15)[1],
          lambda: sc.pp.neighbors(sc.AnnData(emb.copy()), n_neighbors=15), "pynndescent", "scanpy", grepeat=1)
    bench("bbknn", N, lambda: bbknn(emb, rng.integers(0, 2, N))[1], None,
          "validated", "none(no bbknn pkg)", grepeat=1)
    bench("scrublet", 15000, lambda: scrublet(counts[:15000]), None, "AUC~0.96", "none", grepeat=1)
    from metasinglecell.integration import harmonize
    bench("harmony_integrate", N, lambda: harmonize(emb, rng.integers(0, 2, N), max_iter_harmony=3),
          None, "validated", "none(no harmonypy)", grepeat=1)

    # ---------- tl ----------
    from metasinglecell import tools
    from metasinglecell.cluster import leiden as cl_leiden
    from metasinglecell.graph import Graph
    from metasinglecell.graph.louvain import louvain as gpu_louvain
    from metasinglecell.graph.leiden import leiden as gpu_leiden
    conn = neighbors(emb, n_neighbors=15)[1]
    g = Graph.from_scipy(conn)
    import igraph as ig
    coo = conn.tocoo(); up = coo.row < coo.col
    gi = ig.Graph(n=N, edges=np.column_stack([coo.row[up], coo.col[up]]).tolist()); gi.es["weight"] = coo.data[up].tolist()
    from sklearn.cluster import KMeans
    bench("kmeans", N, lambda: tools.kmeans(emb, 8), lambda: KMeans(8, n_init=3).fit_predict(emb), "ARI~0.8", "sklearn")
    bench("louvain(GPU)", N, lambda: gpu_louvain(g, 1.0),
          lambda: gi.community_multilevel(weights="weight"), "Q>=igraph", "igraph", grepeat=1)
    bench("leiden(GPU)", N, lambda: gpu_leiden(g, 1.0, n_iterations=2),
          lambda: gi.community_leiden(objective_function="modularity", weights="weight", n_iterations=2),
          "Q>=igraph", "igraph", grepeat=1)
    import umap as umap_learn
    bench("umap", N, lambda: tools.draw_graph(conn, n_iter=200) if False else __import__("metasinglecell.embedding", fromlist=["umap"]).umap(conn),
          lambda: umap_learn.UMAP(n_neighbors=15).fit_transform(emb), "preserv~", "umap-learn", grepeat=1)
    labels = np.asarray(gpu_louvain(g, 1.0))
    bench("rank_genes_groups", N, lambda: tools.rank_genes_groups(dense_hv, labels, method="t-test"),
          lambda: sc.tl.rank_genes_groups(sc.AnnData(dense_hv.copy(), obs={"g": labels.astype(str)}), "g", method="t-test"),
          "t-stat r~.99", "scanpy")
    genes = [str(i) for i in range(40)]
    bench("score_genes", N, lambda: tools.score_genes(dense_hv, genes, [str(i) for i in range(dense_hv.shape[1])]),
          lambda: sc.tl.score_genes(sc.AnnData(dense_hv.copy(), var={"v": [str(i) for i in range(dense_hv.shape[1])]}).copy(), genes),
          "corr~.74", "scanpy")
    bench("diffmap", N, lambda: tools.diffmap(conn, 15), None, "eigvals~scanpy", "scanpy(slow)", grepeat=2)
    bench("tsne", 4000, lambda: tools.tsne(emb[:4000], n_iter=300),
          lambda: __import__("sklearn.manifold", fromlist=["TSNE"]).TSNE(max_iter=300, init="random").fit_transform(emb[:4000]),
          "preserv~1 (exact O(n²) vs BH)", "sklearn", grepeat=1)
    bench("draw_graph", N, lambda: tools.draw_graph(conn, n_iter=200), None, "preserv~1", "none(no fa2)", grepeat=1)

    # ---------- gr (spatial, smaller) ----------
    from metasinglecell import spatial
    coords = rng.random((N_SP, 2)).astype(np.float32)
    W = spatial.spatial_neighbors(coords, 6)
    Xsp = rng.standard_normal((N_SP, 50)).astype(np.float32)
    ctype = rng.integers(0, 8, N_SP).astype(str)
    def moran_np(X, W):  # numpy Moran's I reference
        Wd = W.toarray(); n = X.shape[0]; Xc = X - X.mean(0); Wsum = Wd.sum()
        num = np.einsum("ig,ij,jg->g", Xc, Wd, Xc); den = (Xc ** 2).sum(0)
        return n / Wsum * num / den
    bench("spatial_autocorr(moran)", N_SP, lambda: spatial.spatial_autocorr(Xsp, W, "moran", n_perms=20),
          lambda: moran_np(Xsp, W), "matches", "numpy", grepeat=2)
    bench("co_occurrence", N_SP, lambda: spatial.co_occurrence(coords, ctype, n_intervals=20), None, "validated", "none(no squidpy)", grepeat=2)
    bench("calculate_niche", N_SP, lambda: spatial.calculate_niche(W, ctype, 4), None, "ARI~.96", "none(no squidpy)", grepeat=2)
    lr = [("0", "1"), ("2", "3")]
    bench("ligrec", N_SP, lambda: spatial.ligrec(Xsp[:, :5], ctype, lr, [str(i) for i in range(5)], n_perms=50),
          None, "validated", "none(no squidpy)", grepeat=2)

    path = validation.write_report(records, "validation", "v_allfuncs.csv")
    log.info("\nreport -> %s", path)


if __name__ == "__main__":
    main()
