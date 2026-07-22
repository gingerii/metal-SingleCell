"""Real Xenium validation on an external integrated cohort (path via XENIUM_H5).

Validates BOTH the core pipeline and the squidpy-GPU ``gr`` spatial functions on
REAL Xenium 5k data (human endometrium): ~2,035,266 cells x 5,101 genes across 72
sections, with real cell-type labels and real tissue coordinates.

Parity (vs scanpy / squidpy) is done on the largest single section (`p11_t3_s1`,
~104k cells) so the spatial coordinates share one coherent frame and real cell
types drive the label-based functions. A full-cohort 2M-cell scale test confirms
the pipeline runs at atlas scale on real Xenium without OOM.

READ-ONLY: the object belongs to another project. Opened backed='r' / h5py 'r';
nothing is ever written back.

    conda activate metalsinglecell
    python validation_notebooks/v_realxenium.py
"""

import logging
import os
import time
import warnings

import numpy as np

from metalsinglecell import config

warnings.filterwarnings("ignore")

# external, READ-ONLY Xenium .h5ad — never written. Set the path via the XENIUM_H5 env var.
XEN = os.environ.get("XENIUM_H5")
SECTION = "p11_t3_s1"   # largest single section -> coherent spatial frame


def main():
    res = config.results_dir("validation")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(res / "v_realxenium.log", "w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("xenium")

    import anndata as ad_mod
    import scanpy as sc
    import squidpy as sq
    from sklearn.metrics import adjusted_rand_score
    from sklearn.utils.extmath import randomized_svd

    from metalsinglecell import preprocess as pp, spatial as gr, validation
    from metalsinglecell.cluster import leiden
    from metalsinglecell.decomposition import pca
    from metalsinglecell.neighbors import neighbors
    from metalsinglecell.sparse import CSR

    # ---- load ONE real section (read-only backed -> to_memory on the subset) ----
    if not XEN:
        raise SystemExit("set XENIUM_H5 to a Xenium .h5ad to run this validation")
    t = time.perf_counter()
    big = sc.read_h5ad(XEN, backed="r")
    sub = big[big.obs["batch"] == SECTION].to_memory()
    big.file.close()
    counts = sub.layers["counts"].tocsr().astype(np.float32)
    coords = np.asarray(sub.obsm["spatial"], dtype=np.float32)
    labels = sub.obs["celltypes"].astype(str).to_numpy()
    log.info("REAL Xenium section %s: %d cells x %d genes, %d cell types (%.0fs)",
             SECTION, *counts.shape, len(np.unique(labels)), time.perf_counter() - t)

    # ===== core pipeline accuracy vs scanpy =====
    adl = ad_mod.AnnData(counts.copy())
    csr = CSR.from_scipy(counts)
    lognorm = csr.normalize_total(1e4).log1p()
    sc.pp.normalize_total(adl, target_sum=1e4); sc.pp.log1p(adl)
    d = np.abs(np.asarray(lognorm.data) - adl.X.tocsr().data).max()
    log.info("normalize+log1p: max|Δ|=%.2e vs scanpy", d)

    mine_hv = pp.highly_variable_genes(lognorm, n_top_genes=2000)["highly_variable"].to_numpy()
    sc.pp.highly_variable_genes(adl, n_top_genes=2000, flavor="seurat")
    ref_hv = adl.var["highly_variable"].to_numpy()
    log.info("HVG: overlap=%.3f vs scanpy", (mine_hv & ref_hv).sum() / max(ref_hv.sum(), 1))

    import scipy.sparse as sp
    import time as _t
    dense = pp.scale(CSR.from_scipy(counts[:, mine_hv].tocsc().tocsr()))
    t = _t.perf_counter(); Xpca, comps, _ = pca(dense, n_comps=50, solver="randomized"); td = _t.perf_counter() - t
    Xc = dense - dense.mean(0)
    _, _, Vt = randomized_svd(Xc.astype(np.float32), 50, n_iter=5, random_state=0)
    log.info("PCA (dense, scaled): subspace overlap=%.4f vs sklearn | %.2fs", validation.subspace_overlap(comps.T, Vt.T), td)
    emb = Xpca.astype(np.float32)

    # sparse-aware PCA (implicit mean-center, NO densify) — the scalable atlas path
    lognorm_hv = sp.csr_matrix(adl[:, mine_hv].X).astype(np.float32)
    t = _t.perf_counter(); Xs, comps_s, _ = pca(CSR.from_scipy(lognorm_hv), n_comps=50); ts = _t.perf_counter() - t
    Dref = np.asarray(lognorm_hv.todense(), np.float64); Dref -= Dref.mean(0)
    _, _, Vts = randomized_svd(Dref, 50, n_iter=7, random_state=0)
    log.info("PCA (sparse, implicit-center): subspace overlap=%.4f vs sklearn | %.2fs (vs dense %.2fs)",
             validation.subspace_overlap(comps_s.T, Vts.T), ts, td)

    dist_g, conn = neighbors(emb, n_neighbors=15)
    adn = ad_mod.AnnData(emb.copy()); adn.obsm["X_pca"] = emb
    sc.pp.neighbors(adn, n_neighbors=15, use_rep="X_pca")
    sc_d = adn.obsp["distances"].tocsr()
    samp = np.random.default_rng(0).choice(emb.shape[0], 3000, replace=False)
    agree = np.mean([len(set(dist_g.indices[dist_g.indptr[i]:dist_g.indptr[i+1]]) &
                         set(sc_d.indices[sc_d.indptr[i]:sc_d.indptr[i+1]])) /
                     max(dist_g.indptr[i+1] - dist_g.indptr[i], 1) for i in samp])
    log.info("neighbors: graph agreement=%.3f vs scanpy", agree)

    lab_gpu = leiden(conn, resolution=1.0, backend="gpu")
    lab_ig = leiden(conn, resolution=1.0, backend="igraph")
    log.info("leiden: GPU %d cl vs igraph %d cl, ARI=%.3f",
             lab_gpu.max()+1, lab_ig.max()+1, adjusted_rand_score(lab_ig, lab_gpu))

    # ===== spatial gr parity vs squidpy on REAL Xenium =====
    # co_occurrence / pairwise distance are O(n^2), so the spatial-parity block runs on
    # a 15k subsample of the section (still real coords + real cell types, coherent frame).
    rng = np.random.default_rng(0)
    nsp = min(15_000, counts.shape[0])
    sidx = np.sort(rng.choice(counts.shape[0], nsp, replace=False))
    coords_s = coords[sidx]; labels_s = labels[sidx]
    logn_s = np.asarray(CSR.from_scipy(counts[sidx]).normalize_total(1e4).log1p().toarray())
    log.info("spatial parity on %d-cell subsample of section", nsp)

    import pandas as pd
    adx = ad_mod.AnnData(logn_s)
    adx.obsm["spatial"] = coords_s
    adx.obs["celltypes"] = pd.Categorical(labels_s)   # squidpy requires categorical
    adx.var_names = [f"g{i}" for i in range(counts.shape[1])]

    sq.gr.spatial_neighbors(adx, n_neighs=6, coord_type="generic")
    A_sq = (adx.obsp["spatial_connectivities"] > 0).astype(np.float32)
    A_mine = (gr.spatial_neighbors(coords_s, n_neighs=6) > 0).astype(np.float32)
    jac = (A_sq.multiply(A_mine)).nnz / max((A_sq + A_mine > 0).nnz, 1)
    log.info("spatial_neighbors: edge Jaccard=%.3f vs squidpy", jac)

    conn_sp = adx.obsp["spatial_connectivities"]
    # pick 100 genes that are actually expressed (nonzero variance) in the subsample —
    # Xenium panel genes can be all-zero in a 15k slice, giving undefined autocorr.
    expressed = np.where(logn_s.var(0) > 1e-8)[0]
    gsel = rng.choice(expressed, min(100, len(expressed)), replace=False)
    gnames = [f"g{i}" for i in gsel]
    Xsel = logn_s[:, gsel]

    def fcorr(a, b):
        k = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[k], b[k])[0, 1], np.max(np.abs(a[k] - b[k]))

    sq.gr.spatial_autocorr(adx, mode="moran", genes=gnames, n_perms=None, seed=0, attr="X")
    I_sq = adx.uns["moranI"].loc[gnames, "I"].to_numpy()
    I_mine = gr.spatial_autocorr(Xsel, conn_sp, mode="moran", n_perms=10)["moran"]
    log.info("Moran's I: corr=%.4f max|Δ|=%.2e vs squidpy", *fcorr(I_sq, I_mine))

    sq.gr.spatial_autocorr(adx, mode="geary", genes=gnames, n_perms=None, seed=0, attr="X")
    C_sq = adx.uns["gearyC"].loc[gnames, "C"].to_numpy()
    C_mine = gr.spatial_autocorr(Xsel, conn_sp, mode="geary", n_perms=10)["geary"]
    log.info("Geary's C: corr=%.4f max|Δ|=%.2e vs squidpy", *fcorr(C_sq, C_mine))

    sq.gr.co_occurrence(adx, cluster_key="celltypes")
    occ_sq = adx.uns["celltypes_co_occurrence"]["occ"]
    interval = adx.uns["celltypes_co_occurrence"]["interval"]
    occ_mine = gr.co_occurrence(coords_s, labels_s, interval=interval)["occ"]
    m = np.isfinite(occ_sq) & np.isfinite(occ_mine)
    log.info("co_occurrence: corr=%.4f max|Δ|=%.2e vs squidpy", np.corrcoef(occ_sq[m], occ_mine[m])[0,1], np.max(np.abs(occ_sq[m]-occ_mine[m])))

    out_n = gr.calculate_niche(conn_sp, labels_s, n_niches=8)
    cats = np.unique(labels_s); code = np.searchsorted(cats, labels_s)
    onehot = (code[:,None] == np.arange(len(cats))[None,:]).astype(np.float32)
    comp_ref = conn_sp @ onehot; comp_ref = comp_ref / (comp_ref.sum(1, keepdims=True) + 1e-12)
    log.info("calculate_niche: composition max|Δ|=%.2e vs scipy", np.max(np.abs(np.asarray(comp_ref) - out_n["composition"])))

    # ===== full-cohort 2M scale confirmation (read counts via h5py, minimal memory) =====
    import gc
    import h5py
    import scipy.sparse as sp
    del adx, adl, dense, emb, conn, dist_g, adn, logn_s, counts; gc.collect()
    t = time.perf_counter()
    with h5py.File(XEN, "r") as f:
        L = f["layers"]["counts"]
        Xfull = sp.csr_matrix((L["data"][:], L["indices"][:], L["indptr"][:]),
                              shape=tuple(L.attrs["shape"]))
    nfull = Xfull.shape[0]
    log.info("\nfull-cohort read: %d x %d (%.0fs)", *Xfull.shape, time.perf_counter()-t)
    t = time.perf_counter()
    ln2 = CSR.from_scipy(Xfull.astype(np.float32)).normalize_total(1e4).log1p()
    hv2 = pp.highly_variable_genes(ln2, n_top_genes=2000)["highly_variable"].to_numpy()
    log.info("2M pipeline (normalize+log1p+HVG): %.1fs on real Xenium (no OOM)", time.perf_counter()-t)
    del Xfull; gc.collect()

    # sparse-aware PCA at the FULL 2M cells (HVG-subset lognorm) — this is exactly where
    # the dense scale+PCA path OOMs (would need ~2*2M*2000*4 bytes ≈ 32GB). Implicit
    # centering keeps it sparse so it runs on the 24GB M3.
    ln2_sp = sp.csr_matrix((np.asarray(ln2.data), np.asarray(ln2.indices),
                            np.asarray(ln2.indptr)), shape=ln2.shape)
    ln2_hv = CSR.from_scipy(ln2_sp[:, hv2].astype(np.float32))
    t = time.perf_counter(); X2, _, _ = pca(ln2_hv, n_comps=50); tp = time.perf_counter() - t
    log.info("2M sparse PCA (implicit-center): %.1fs -> X_pca %s (no OOM; dense path needs ~%dGB)",
             tp, X2.shape, int(2 * nfull * 2000 * 4 / 1e9))

    log.info("\nreal-Xenium validation complete")


if __name__ == "__main__":
    main()
