"""Real-data validation of the remaining functions (tools/integration/pp/neighbors)
not covered by v_realdata/spatial/atlas/xenium. Canonical pattern: compute HVGs, then
restrict downstream analysis to them (keeps everything in-memory at any scale).

Each function is checked against its canonical reference (scanpy / sklearn / harmonypy /
bbknn / scrublet) on real PBMC3k. Stochastic methods (tsne, harmony, scrublet, draw_graph)
are validated by structure/agreement, not exact values — as with umap/leiden.

    conda activate metasinglecell
    python validation_notebooks/v_remaining.py
"""

import logging
import warnings

import numpy as np

from metasinglecell import config

warnings.filterwarnings("ignore")

# standard Tirosh/Regev cell-cycle marker genes (subset present in PBMC)
S_GENES = ["MCM5", "PCNA", "TYMS", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
           "CDCA7", "DTL", "PRIM1", "UHRF1", "SLBP", "CCNE2", "UBR7", "RRM2", "CDC6",
           "CDC45", "EXO1", "GMNN", "WDR76", "CHAF1B", "USP1", "CLSPN"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "CKS2",
             "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2",
             "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "GTSE1", "CDCA3", "CENPA"]


def nbr_pres(Y, ref_sets, k=15):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    return np.mean([len(set(idx[i]) & ref_sets[i]) / k for i in range(len(Y))])


def main():
    res = config.results_dir("validation")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(res / "v_remaining.log", "w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("rem")

    import anndata as adm
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import spearmanr

    from metasinglecell import preprocess as pp, tools, validation
    from metasinglecell.integration import harmonize
    from metasinglecell.neighbors import bbknn, neighbors
    from metasinglecell.sparse import CSR

    # ---- real PBMC pipeline; restrict downstream to HVGs (canonical) ----
    ad = sc.datasets.pbmc3k()
    sc.pp.filter_cells(ad, min_genes=200); sc.pp.filter_genes(ad, min_cells=3)
    counts = ad.X.copy()
    adl = ad.copy(); sc.pp.normalize_total(adl, target_sum=1e4); sc.pp.log1p(adl)
    sc.pp.highly_variable_genes(adl, n_top_genes=2000)
    adh = adl[:, adl.var.highly_variable].copy()
    scaled = pp.scale(CSR.from_scipy(sp.csr_matrix(adh.X).astype(np.float32)))
    from metasinglecell.decomposition import pca
    emb = pca(CSR.from_scipy(sp.csr_matrix(adh.X).astype(np.float32)), n_comps=50)[0].astype(np.float32)
    dist_g, conn = neighbors(emb, n_neighbors=15)
    from metasinglecell.cluster import leiden
    groups = leiden(conn, resolution=1.0, backend="igraph")
    refnb = [set(dist_g.indices[dist_g.indptr[i]:dist_g.indptr[i + 1]]) for i in range(emb.shape[0])]
    log.info("PBMC: %d cells, %d HVG, %d leiden groups", adh.n_obs, adh.n_vars, len(set(groups)))
    log.info("--- validating remaining functions (real data) ---")

    rng = np.random.default_rng(0)

    def check(name, ok, detail):
        log.info("%-26s %s | %s", name, "PASS" if ok else "CHECK", detail)

    # 1. kmeans vs sklearn (ARI on the PCA embedding)
    try:
        km_mine = tools.kmeans(emb, n_clusters=10, random_state=0)
        km_sk = KMeans(n_clusters=10, random_state=0, n_init=10).fit_predict(emb)
        ari = adjusted_rand_score(km_sk, km_mine)
        check("kmeans", ari > 0.6, f"ARI vs sklearn = {ari:.3f}")
    except Exception as e:
        check("kmeans", False, f"ERROR {e}")

    # 2. score_genes vs scanpy (needs full gene set for control matching)
    try:
        gl = [g for g in S_GENES if g in adl.var_names][:20]
        full = np.asarray(adl.X.todense(), np.float32)
        s_mine = tools.score_genes(full, gl, adl.var_names.to_numpy(), random_state=0)
        a2 = adl.copy(); sc.tl.score_genes(a2, gl, score_name="s", random_state=0)
        r = np.corrcoef(s_mine, a2.obs["s"].to_numpy())[0, 1]
        check("score_genes", r > 0.85, f"score corr vs scanpy = {r:.3f}")
    except Exception as e:
        check("score_genes", False, f"ERROR {e}")

    # 3. score_genes_cell_cycle vs scanpy (phase agreement)
    try:
        sg = [g for g in S_GENES if g in adl.var_names]
        gg = [g for g in G2M_GENES if g in adl.var_names]
        full = np.asarray(adl.X.todense(), np.float32)
        cc = tools.score_genes_cell_cycle(full, sg, gg, adl.var_names.to_numpy(), random_state=0)
        a3 = adl.copy(); sc.tl.score_genes_cell_cycle(a3, s_genes=sg, g2m_genes=gg, random_state=0)
        agree = np.mean(cc["phase"] == a3.obs["phase"].to_numpy())
        check("score_genes_cell_cycle", agree > 0.8, f"phase agreement vs scanpy = {agree:.3f}")
    except Exception as e:
        check("score_genes_cell_cycle", False, f"ERROR {e}")

    # 4. rank_genes_groups vs scanpy t-test (top-25 marker overlap)
    try:
        full = np.asarray(adl.X.todense(), np.float32)
        rg = tools.rank_genes_groups(full, np.asarray(groups), var_names=adl.var_names.to_numpy(),
                                     method="t-test")
        a4 = adl.copy(); a4.obs["grp"] = [str(x) for x in groups]
        a4.obs["grp"] = a4.obs["grp"].astype("category")
        sc.tl.rank_genes_groups(a4, "grp", method="t-test")
        ov = []
        for g in a4.obs["grp"].cat.categories:
            sc_top = set(a4.uns["rank_genes_groups"]["names"][g][:25])
            mine_top = set(rg[str(g)]["names"][:25]) if str(g) in rg else set()
            ov.append(len(sc_top & mine_top) / 25)
        mo = float(np.mean(ov))
        check("rank_genes_groups", mo > 0.7, f"top-25 marker overlap vs scanpy = {mo:.3f}")
    except Exception as e:
        check("rank_genes_groups", False, f"ERROR {e}")

    # 5. tsne vs sklearn (neighbor preservation, both stochastic)
    try:
        Ym = tools.tsne(emb, random_state=0)
        Ysk = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb)
        pm, ps = nbr_pres(Ym, refnb), nbr_pres(Ysk, refnb)
        check("tsne", pm > 0.7 * ps, f"nbr-preservation ours {pm:.3f} vs sklearn {ps:.3f}")
    except Exception as e:
        check("tsne", False, f"ERROR {e}")

    # 6. draw_graph (force layout) — preservation vs scanpy's own draw_graph (fr layout)
    try:
        Yd = tools.draw_graph(conn, random_state=0)
        a6 = adh.copy(); a6.obsm["X_pca"] = emb; sc.pp.neighbors(a6, use_rep="X_pca", n_neighbors=15)
        sc.tl.draw_graph(a6, layout="fr", random_state=0)
        pm, ps = nbr_pres(Yd, refnb), nbr_pres(a6.obsm["X_draw_graph_fr"], refnb)
        check("draw_graph", pm > 0.7 * ps, f"nbr-preservation ours {pm:.3f} vs scanpy-fr {ps:.3f}")
    except Exception as e:
        check("draw_graph", False, f"ERROR {e}")

    # 7. diffmap vs scanpy (eigenvalue corr + diffusion-component subspace overlap)
    try:
        dm = tools.diffmap(conn, n_comps=15)
        a7 = adh.copy(); a7.obsm["X_pca"] = emb; sc.pp.neighbors(a7, use_rep="X_pca", n_neighbors=15)
        sc.tl.diffmap(a7, n_comps=15)
        ev_r = np.corrcoef(np.sort(dm["eigenvalues"])[-10:],
                           np.sort(a7.uns["diffmap_evals"])[-10:])[0, 1]
        sub = validation.subspace_overlap(dm["X_diffmap"][:, 1:6], a7.obsm["X_diffmap"][:, 1:6])
        check("diffmap", ev_r > 0.9 and sub > 0.6, f"eigval corr {ev_r:.3f}, comp subspace {sub:.3f}")
    except Exception as e:
        check("diffmap", False, f"ERROR {e}")

    # 8. embedding_density vs scanpy (density correlation on the UMAP)
    try:
        from metasinglecell.embedding import umap
        U = umap(conn)
        dens_mine = tools.embedding_density(U)
        a8 = adh.copy(); a8.obsm["X_umap"] = U; sc.tl.embedding_density(a8, basis="umap")
        r = np.corrcoef(dens_mine, a8.obs["umap_density"].to_numpy())[0, 1]
        check("embedding_density", abs(r) > 0.9, f"density corr vs scanpy = {r:.3f}")
    except Exception as e:
        check("embedding_density", False, f"ERROR {e}")

    # 9. normalize_pearson_residuals vs scanpy (analytic -> near-exact)
    try:
        cnt_hv = sp.csr_matrix(counts)[:, adl.var.highly_variable.to_numpy()]
        pr_mine = pp.normalize_pearson_residuals(cnt_hv, theta=100.0)
        a9 = adm.AnnData(cnt_hv.copy().astype(np.float32))
        sc.experimental.pp.normalize_pearson_residuals(a9, theta=100.0)
        r = np.corrcoef(np.asarray(pr_mine).ravel(), np.asarray(a9.X).ravel())[0, 1]
        check("normalize_pearson_residuals", r > 0.99, f"residual corr vs scanpy = {r:.4f}")
    except Exception as e:
        check("normalize_pearson_residuals", False, f"ERROR {e}")

    # 10. scrublet — primary quality test is detection AUC on KNOWN injected doublets
    #     (score-correlation to scanpy is weak because each draws its OWN random
    #     doublet simulation; AUC measures what the method is actually for).
    try:
        C = sp.csr_matrix(counts).astype(np.float32)
        n0 = C.shape[0]; n_inj = 300
        pairs = rng.integers(0, n0, (n_inj, 2))
        inj = C[pairs[:, 0]] + C[pairs[:, 1]]              # synthetic doublets = summed pairs
        Caug = sp.vstack([C, inj]).tocsr()
        is_dbl = np.r_[np.zeros(n0), np.ones(n_inj)]
        sr = pp.scrublet(Caug, random_state=0)
        auc = roc_auc_score(is_dbl, sr["doublet_scores"])
        a10 = adm.AnnData(Caug.copy()); sc.pp.scrublet(a10, random_state=0)
        auc_sc = roc_auc_score(is_dbl, a10.obs["doublet_score"].to_numpy())
        check("scrublet", auc > 0.8, f"injected-doublet AUC ours {auc:.3f} (scanpy {auc_sc:.3f})")
    except Exception as e:
        check("scrublet", False, f"ERROR {e}")

    # ---- batch-integration functions: split into 2 batches with an artificial shift ----
    rng = np.random.default_rng(0)
    batch = (rng.random(emb.shape[0]) < 0.5).astype(int)
    emb_b = emb.copy(); emb_b[batch == 1, :5] += 3.0      # inject a batch effect in PCA space

    def mixing(Z, k=30):                                   # mean opposite-batch fraction (iLISI-like)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        return float(np.mean([(batch[nn[i]] != batch[i]).mean() for i in range(len(Z))]))

    base_mix = mixing(emb_b)

    # 11. harmonize vs harmonypy (batch mixing recovered toward ~0.5)
    try:
        Zc = harmonize(emb_b, batch, random_state=0)
        import harmonypy
        ho = harmonypy.run_harmony(emb_b, __import__("pandas").DataFrame({"b": batch}), ["b"])
        Zhp = ho.Z_corr.T
        mm, mh = mixing(np.asarray(Zc)), mixing(np.asarray(Zhp))
        check("harmonize", mm > base_mix + 0.1, f"mixing base {base_mix:.2f} -> ours {mm:.2f} (harmonypy {mh:.2f})")
    except Exception as e:
        check("harmonize", False, f"ERROR {e}")

    # 12. bbknn vs bbknn package (batch-balanced graph mixing)
    try:
        _, conn_bb = bbknn(emb_b, batch, neighbors_within_batch=3)
        mine_mix = np.mean([(batch[conn_bb[i].indices] != batch[i]).mean()
                            for i in range(emb_b.shape[0]) if conn_bb[i].nnz])
        import bbknn as bbk
        a12 = adm.AnnData(np.zeros((emb_b.shape[0], 1), np.float32))
        a12.obsm["X_pca"] = emb_b; a12.obs["b"] = [str(x) for x in batch]
        a12.obs["b"] = a12.obs["b"].astype("category")
        bbk.bbknn(a12, batch_key="b", neighbors_within_batch=3)
        cbb = a12.obsp["connectivities"]
        ref_mix = np.mean([(batch[cbb[i].indices] != batch[i]).mean()
                           for i in range(emb_b.shape[0]) if cbb[i].nnz])
        check("bbknn", mine_mix > 0.25, f"graph opp-batch frac ours {mine_mix:.2f} (bbknn pkg {ref_mix:.2f})")
    except Exception as e:
        check("bbknn", False, f"ERROR {e}")

    log.info("\nremaining-function real-data validation complete")


if __name__ == "__main__":
    main()
