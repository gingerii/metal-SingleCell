"""Real SPATIAL-data validation (CLAUDE.md pin) — squidpy parity on real Visium.

The squidpy-GPU ``gr`` functions were previously only checked on synthetic spatial
patterns. This validates them on a REAL spatial dataset (10x Visium V1 Breast
Cancer, ~3.8k spots with real tissue coordinates) against squidpy itself.

    conda activate metasinglecell
    python validation_notebooks/v_realspatial.py
"""

import logging
import warnings

import numpy as np

from metasinglecell import config

warnings.filterwarnings("ignore")


def main():
    res = config.results_dir("validation")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(res / "v_realspatial.log", "w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("realspatial")

    import scanpy as sc
    import scipy.sparse as sp
    import squidpy as sq

    from metasinglecell import spatial as gr

    # ---- real Visium spatial data ----
    ad = sc.datasets.visium_sge("V1_Breast_Cancer_Block_A_Section_1")
    ad.var_names_make_unique()
    sc.pp.filter_genes(ad, min_cells=10)
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=200, flavor="seurat")
    coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
    log.info("Visium real: %d spots x %d genes (%d HVG)",
             ad.n_obs, ad.n_vars, int(ad.var["highly_variable"].sum()))

    # ---- spatial_neighbors parity ----
    sq.gr.spatial_neighbors(ad, n_neighs=6, coord_type="generic")
    A_sq = (ad.obsp["spatial_connectivities"] > 0).astype(np.float32)
    A_mine = (gr.spatial_neighbors(coords, n_neighs=6) > 0).astype(np.float32)
    jac = (A_sq.multiply(A_mine)).nnz / max((A_sq + A_mine > 0).nnz, 1)
    log.info("spatial_neighbors edge Jaccard vs squidpy: %.3f", jac)

    # use ONE shared graph (squidpy's) so autocorr is an apples-to-apples math check
    conn = ad.obsp["spatial_connectivities"]
    hv = ad.var["highly_variable"].to_numpy()
    Xhv = ad[:, hv].X
    genes_hv = ad.var_names[hv].to_numpy()

    # ---- spatial_autocorr: Moran's I parity (deterministic statistic) ----
    sq.gr.spatial_autocorr(ad, mode="moran", genes=list(genes_hv), n_perms=None,
                           seed=0, attr="X")
    I_sq = ad.uns["moranI"].loc[genes_hv, "I"].to_numpy()
    out_m = gr.spatial_autocorr(np.asarray(Xhv.todense()), conn, mode="moran", n_perms=20)
    I_mine = out_m["moran"]
    log.info("Moran's I vs squidpy: corr=%.4f  max|Δ|=%.2e  median|Δ|=%.2e",
             np.corrcoef(I_sq, I_mine)[0, 1], np.max(np.abs(I_sq - I_mine)),
             np.median(np.abs(I_sq - I_mine)))

    # ---- spatial_autocorr: Geary's C parity ----
    sq.gr.spatial_autocorr(ad, mode="geary", genes=list(genes_hv), n_perms=None,
                           seed=0, attr="X")
    C_sq = ad.uns["gearyC"].loc[genes_hv, "C"].to_numpy()
    out_g = gr.spatial_autocorr(np.asarray(Xhv.todense()), conn, mode="geary", n_perms=20)
    C_mine = out_g["geary"]
    log.info("Geary's C vs squidpy: corr=%.4f  max|Δ|=%.2e",
             np.corrcoef(C_sq, C_mine)[0, 1], np.max(np.abs(C_sq - C_mine)))

    # labels for label-based spatial functions (leiden on expression)
    sc.pp.pca(ad, n_comps=30); sc.pp.neighbors(ad); sc.tl.leiden(ad, resolution=0.5)
    labels = ad.obs["leiden"].to_numpy()
    log.info("leiden labels for spatial functions: %d clusters", len(np.unique(labels)))

    # ---- co_occurrence vs squidpy (same thresholds + same cumulative definition) ----
    sq.gr.co_occurrence(ad, cluster_key="leiden")
    occ_sq = ad.uns["leiden_co_occurrence"]["occ"]          # (K, K, n_int)
    interval = ad.uns["leiden_co_occurrence"]["interval"]   # squidpy's exact thresholds
    out_co = gr.co_occurrence(coords, labels, interval=interval)
    occ_mine = out_co["occ"]
    m = np.isfinite(occ_sq) & np.isfinite(occ_mine)
    log.info("co_occurrence vs squidpy: corr=%.4f  max|Δ|=%.2e (n=%d ratios)",
             np.corrcoef(occ_sq[m], occ_mine[m])[0, 1],
             np.max(np.abs(occ_sq[m] - occ_mine[m])), int(m.sum()))

    # ---- calculate_niche: neighborhood composition exactness (scipy reference) ----
    out_n = gr.calculate_niche(conn, labels, n_niches=8)
    comp_mine = out_n["composition"]
    cats = np.unique(labels)
    code = np.searchsorted(cats, labels)
    onehot = (code[:, None] == np.arange(len(cats))[None, :]).astype(np.float32)
    comp_ref = conn @ onehot
    comp_ref = comp_ref / (comp_ref.sum(1, keepdims=True) + 1e-12)
    log.info("calculate_niche composition vs scipy SpMM: max|Δ|=%.2e",
             np.max(np.abs(np.asarray(comp_ref) - comp_mine)))

    # ---- ligrec: cluster-mean scores vs numpy reference (means are deterministic) ----
    rng = np.random.default_rng(0)
    gi = rng.choice(ad.n_vars, 20, replace=False)
    pairs = [(ad.var_names[gi[2 * k]], ad.var_names[gi[2 * k + 1]]) for k in range(10)]
    out_lr = gr.ligrec(ad.X, labels, pairs, ad.var_names.to_numpy(), n_perms=50)
    # numpy reference cluster means
    Xd = np.asarray(ad.X.todense(), dtype=np.float32)
    means_ref = np.stack([Xd[code == c].mean(0) for c in range(len(cats))])  # K×G
    name_to_idx = {g: i for i, g in enumerate(ad.var_names.to_numpy())}
    lig = [name_to_idx[a] for a, b in pairs]; rec = [name_to_idx[b] for a, b in pairs]
    L = means_ref[:, lig].T; R = means_ref[:, rec].T
    score_ref = 0.5 * (L[:, :, None] + R[:, None, :])
    log.info("ligrec mean-scores vs numpy: max|Δ|=%.2e", np.max(np.abs(score_ref - out_lr["means"])))

    log.info("\nreal-spatial validation complete")


if __name__ == "__main__":
    main()
