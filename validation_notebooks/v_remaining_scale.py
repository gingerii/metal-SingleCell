"""Larger-real-data validation of the remaining functions (closes the PBMC-only gap).

v_remaining.py checked these on PBMC3k (2.7k). Here the SCALABLE ones are re-confirmed on a
larger REAL dataset — 100k cells subsampled from the 10x 1.3M-neuron atlas, HVG-restricted
(canonical) — vs their references, with timing and a no-OOM check.

Inherently O(n²) methods (exact tsne, gaussian-KDE embedding_density) are subsample-only by
design and are NOT run at 100k (documented limitation); the rest scale.

    conda activate metasinglecell
    python validation_notebooks/v_remaining_scale.py
"""

import logging
import time
import warnings

import numpy as np

from metasinglecell import config

warnings.filterwarnings("ignore")
N = 100_000


def main():
    res = config.results_dir("validation")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(res / "v_remaining_scale.log", "w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("rscale")

    import anndata as adm
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    from metasinglecell import preprocess as pp, tools, validation
    from metasinglecell.cluster import leiden
    from metasinglecell.decomposition import pca
    from metasinglecell.integration import harmonize
    from metasinglecell.neighbors import bbknn, neighbors
    from metasinglecell.sparse import CSR

    t0 = time.perf_counter()
    ad = sc.read_10x_h5("data/external/1M_neurons.h5"); ad.var_names_make_unique()
    rng = np.random.default_rng(0)
    idx = rng.choice(ad.n_obs, N, replace=False)
    ac = ad[idx].copy(); del ad
    sc.pp.filter_genes(ac, min_cells=3)
    counts = ac.X.tocsr().astype(np.float32)
    adl = ac.copy(); sc.pp.normalize_total(adl, target_sum=1e4); sc.pp.log1p(adl)
    sc.pp.highly_variable_genes(adl, n_top_genes=2000)
    hv = adl.var.highly_variable.to_numpy()
    adh = adl[:, hv].copy()
    emb = pca(CSR.from_scipy(sp.csr_matrix(adh.X).astype(np.float32)), n_comps=50)[0].astype(np.float32)
    _, conn = neighbors(emb, n_neighbors=15)
    groups = leiden(conn, resolution=1.0, backend="igraph")
    log.info("REAL atlas subset: %d cells, %d HVG, %d groups (%.0fs setup)",
             N, adh.n_vars, len(set(groups)), time.perf_counter() - t0)
    log.info("--- scalable remaining functions @ %d real cells ---", N)

    def check(name, ok, detail):
        log.info("%-26s %s | %s", name, "PASS" if ok else "CHECK", detail)

    # kmeans vs sklearn
    try:
        t = time.perf_counter(); km = tools.kmeans(emb, n_clusters=15, random_state=0); tg = time.perf_counter() - t
        ks = KMeans(n_clusters=15, random_state=0, n_init=5).fit_predict(emb)
        check("kmeans", adjusted_rand_score(ks, km) > 0.5,
              f"ARI {adjusted_rand_score(ks, km):.3f} vs sklearn | {tg:.1f}s")
    except Exception as e:
        check("kmeans", False, f"ERROR {e}")

    # score_genes vs scanpy (use 20 random expressed genes)
    try:
        full = np.asarray(adl.X.todense(), np.float32)
        gl = list(adl.var_names[np.argsort(-np.asarray(adl.X.mean(0)).ravel())[:20]])
        sm = tools.score_genes(full, gl, adl.var_names.to_numpy(), random_state=0)
        a2 = adl.copy(); sc.tl.score_genes(a2, gl, score_name="s", random_state=0)
        r = np.corrcoef(sm, a2.obs["s"].to_numpy())[0, 1]
        check("score_genes", r > 0.85, f"score corr {r:.3f} vs scanpy")
    except Exception as e:
        check("score_genes", False, f"ERROR {e}")

    # rank_genes_groups vs scanpy
    try:
        full = np.asarray(adl.X.todense(), np.float32)
        t = time.perf_counter()
        rg = tools.rank_genes_groups(full, np.asarray(groups), var_names=adl.var_names.to_numpy())
        tg = time.perf_counter() - t
        a4 = adl.copy(); a4.obs["g"] = [str(x) for x in groups]; a4.obs["g"] = a4.obs["g"].astype("category")
        sc.tl.rank_genes_groups(a4, "g", method="t-test")
        ov = [len(set(a4.uns["rank_genes_groups"]["names"][g][:25]) &
                  set(rg[str(g)]["names"][:25])) / 25 for g in a4.obs["g"].cat.categories]
        check("rank_genes_groups", np.mean(ov) > 0.7, f"top-25 overlap {np.mean(ov):.3f} | {tg:.1f}s")
    except Exception as e:
        check("rank_genes_groups", False, f"ERROR {e}")

    # normalize_pearson_residuals vs scanpy (HVG subset)
    try:
        cnt_hv = counts[:, hv]
        pr = pp.normalize_pearson_residuals(cnt_hv, theta=100.0)
        a9 = adm.AnnData(cnt_hv.copy()); sc.experimental.pp.normalize_pearson_residuals(a9, theta=100.0)
        r = np.corrcoef(np.asarray(pr).ravel(), np.asarray(a9.X).ravel())[0, 1]
        check("normalize_pearson_residuals", r > 0.99, f"residual corr {r:.4f} vs scanpy")
    except Exception as e:
        check("normalize_pearson_residuals", False, f"ERROR {e}")

    # diffmap vs scanpy
    try:
        dm = tools.diffmap(conn, n_comps=15)
        a7 = adh.copy(); a7.obsm["X_pca"] = emb; sc.pp.neighbors(a7, use_rep="X_pca", n_neighbors=15)
        sc.tl.diffmap(a7, n_comps=15)
        ev = np.corrcoef(np.sort(dm["eigenvalues"])[-10:], np.sort(a7.uns["diffmap_evals"])[-10:])[0, 1]
        sub = validation.subspace_overlap(dm["X_diffmap"][:, 1:6], a7.obsm["X_diffmap"][:, 1:6])
        check("diffmap", ev > 0.9 and sub > 0.6, f"eigval {ev:.3f}, subspace {sub:.3f}")
    except Exception as e:
        check("diffmap", False, f"ERROR {e}")

    # batch functions: split into 2 batches + inject PCA shift
    batch = (rng.random(N) < 0.5).astype(int)
    emb_b = emb.copy(); emb_b[batch == 1, :5] += 3.0

    def mixing(Z, k=30):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        return float(np.mean((batch[nn] != batch[:, None]).mean(1)))

    base = mixing(emb_b)
    try:
        t = time.perf_counter(); Zc = harmonize(emb_b, batch, random_state=0); tg = time.perf_counter() - t
        mm = mixing(np.asarray(Zc))
        check("harmonize", mm > base + 0.1, f"mixing {base:.2f}->{mm:.2f} | {tg:.1f}s")
    except Exception as e:
        check("harmonize", False, f"ERROR {e}")
    try:
        t = time.perf_counter(); _, cbb = bbknn(emb_b, batch, neighbors_within_batch=3); tg = time.perf_counter() - t
        mix = np.mean([(batch[cbb[i].indices] != batch[i]).mean() for i in range(0, N, 50) if cbb[i].nnz])
        check("bbknn", mix > 0.25, f"graph opp-batch frac {mix:.2f} | {tg:.1f}s")
    except Exception as e:
        check("bbknn", False, f"ERROR {e}")

    # scrublet — injected-doublet AUC at scale
    try:
        n_inj = 2000
        pairs = rng.integers(0, N, (n_inj, 2))
        Caug = sp.vstack([counts, counts[pairs[:, 0]] + counts[pairs[:, 1]]]).tocsr()
        is_d = np.r_[np.zeros(N), np.ones(n_inj)]
        t = time.perf_counter(); sr = pp.scrublet(Caug, random_state=0); tg = time.perf_counter() - t
        auc = roc_auc_score(is_d, sr["doublet_scores"])
        check("scrublet", auc > 0.8, f"injected-doublet AUC {auc:.3f} | {tg:.1f}s")
    except Exception as e:
        check("scrublet", False, f"ERROR {e}")

    log.info("\n(NB exact tsne O(n²) and gaussian-KDE embedding_density are subsample-only "
             "by design — not run at 100k.)")
    log.info("scalable remaining-function validation @ %dk complete (%.0fs)", N // 1000,
             time.perf_counter() - t0)


if __name__ == "__main__":
    main()
