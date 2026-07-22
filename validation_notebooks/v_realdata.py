"""Real-data validation (CLAUDE.md pin: must confirm on real data, not only synthetic).

Uses PBMC3k (real counts, real structure). Confirms accuracy of the optimized/
flagged functions vs scanpy on real data, and tests the IVF-KNN scale path on a
real-structure embedding tiled up to 100k (real cluster structure, unlike the
random synthetic worst case).

    conda activate metalsinglecell
    python validation_notebooks/v_realdata.py
"""

import logging
import time
import warnings

import numpy as np

from metalsinglecell import config, validation

warnings.filterwarnings("ignore")


def main():
    res = config.results_dir("validation")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(res / "v_realdata.log", "w"), logging.StreamHandler()])
    log = logging.getLogger("real")
    import scanpy as sc
    import mlx.core as mx
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors

    from metalsinglecell import preprocess as pp, tools
    from metalsinglecell.cluster import leiden
    from metalsinglecell.decomposition import pca
    from metalsinglecell.embedding import umap
    from metalsinglecell.neighbors import _knn_ivf, neighbors
    from metalsinglecell.sparse import CSR

    # ---- real PBMC3k pipeline ----
    ad = sc.datasets.pbmc3k()
    sc.pp.filter_cells(ad, min_genes=200); sc.pp.filter_genes(ad, min_cells=3)
    counts = ad.X.copy()
    adl = ad.copy(); sc.pp.normalize_total(adl, target_sum=1e4); sc.pp.log1p(adl)
    csr = CSR.from_scipy(counts)
    lognorm = csr.normalize_total(1e4).log1p()
    log.info("PBMC3k real: %d cells x %d genes", *counts.shape)

    # HVG (real)
    mine_hv = pp.highly_variable_genes(lognorm, n_top_genes=2000)["highly_variable"].to_numpy()
    rh = adl.copy(); sc.pp.highly_variable_genes(rh, n_top_genes=2000, flavor="seurat")
    ov = (mine_hv & rh.var["highly_variable"].to_numpy()).sum() / rh.var["highly_variable"].sum()
    log.info("HVG (real) overlap vs scanpy: %.3f", ov)

    # regress_out (real)
    tot = np.asarray(adl.X.sum(1)).ravel()
    adr = adl.copy(); sc.pp.regress_out(adr, ["n_counts"] if "n_counts" in adr.obs else None) if False else None
    adr = adl.copy(); adr.obs["t"] = tot; sc.pp.regress_out(adr, ["t"])
    mine_ro = pp.regress_out(adl.X, tot)
    log.info("regress_out (real) corr vs scanpy: %.4f",
             np.corrcoef(np.asarray(adr.X).ravel(), mine_ro.ravel())[0, 1])

    # PCA (real) on HVG-subset scaled
    dense = pp.scale(CSR.from_scipy(adl[:, mine_hv].X))
    Xpca, comps, _ = pca(dense, n_comps=50, solver="randomized")
    from sklearn.utils.extmath import randomized_svd
    Xc = dense.astype(np.float64) - dense.astype(np.float64).mean(0)
    _, _, Vt = randomized_svd(Xc, 50, n_iter=7, random_state=0)
    log.info("PCA (real) subspace overlap vs sklearn: %.4f", validation.subspace_overlap(comps.T, Vt.T))
    emb = Xpca.astype(np.float32)

    # neighbors (real, exact path at this n) — recall vs exact
    knn_i, _ = neighbors(emb, n_neighbors=15)[0].nonzero(), None  # graph built
    exact = NearestNeighbors(n_neighbors=15).fit(emb).kneighbors(emb, return_distance=False)
    dist_g, conn = neighbors(emb, n_neighbors=15)
    rec = np.mean([len(set(dist_g.indices[dist_g.indptr[i]:dist_g.indptr[i+1]]) & set(exact[i])) / 14
                   for i in range(emb.shape[0])])
    log.info("neighbors (real) neighbor overlap vs exact: %.3f", rec)

    # leiden (real) ARI: GPU vs igraph on the real graph
    lab_gpu = leiden(conn, resolution=1.0, backend="gpu")
    lab_ig = leiden(conn, resolution=1.0, backend="igraph")
    log.info("leiden (real) GPU vs igraph ARI: %.3f (GPU %d cl, igraph %d cl)",
             adjusted_rand_score(lab_ig, lab_gpu), lab_gpu.max() + 1, lab_ig.max() + 1)

    # umap (real) structure preservation vs umap-learn
    import umap as ul
    E = umap(conn); Eul = ul.UMAP(n_neighbors=15, random_state=0).fit_transform(emb)
    refnb = [set(dist_g.indices[dist_g.indptr[i]:dist_g.indptr[i+1]]) for i in range(emb.shape[0])]
    def pres(Y):
        idx = NearestNeighbors(n_neighbors=16).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        return np.mean([len(set(idx[i]) & refnb[i]) / max(len(refnb[i]), 1) for i in range(len(Y))])
    log.info("umap (real) neighbor-preservation: ours %.3f vs umap-learn %.3f", pres(E), pres(Eul))

    # ---- IVF-KNN on REAL structure tiled to 100k (vs random synthetic worst case) ----
    reps = 100000 // emb.shape[0] + 1
    big = np.repeat(emb, reps, axis=0)[:100000].astype(np.float32)
    big += np.random.default_rng(0).normal(0, 0.01, big.shape).astype(np.float32)  # jitter dupes
    ex = NearestNeighbors(n_neighbors=15).fit(big).kneighbors(big[:3000], return_distance=False)
    _knn_ivf(big[:5000], 15)
    t = time.perf_counter(); idx, _ = _knn_ivf(big, 15); it = time.perf_counter() - t
    rec_ivf = np.mean([len(set(idx[i]) & set(ex[i])) / 15 for i in range(3000)])
    from pynndescent import NNDescent
    t = time.perf_counter(); NNDescent(big, n_neighbors=15).neighbor_graph; pt = time.perf_counter() - t
    log.info("IVF-KNN (real-structure 100k): %.2fs recall=%.3f | pynndescent %.2fs = %.2fx",
             it, rec_ivf, pt, pt / it)
    log.info("\nreal-data validation complete")


if __name__ == "__main__":
    main()
