"""Comprehensive real-data benchmark: per-function runtime + speedup + accuracy across
sizes (PBMC, 50k, 100k, 1M, 2M). Correct methodology: warm-up (MLX kernel compile),
best-of-N, mx.eval barriers; HVG-restricted downstream (canonical).

Run ONE SIZE PER PROCESS so an OOM at large n can't lose smaller results; each process
appends rows to results/validation/benchmark.csv.

    python v_benchmark.py <n|pbmc>          # e.g. pbmc, 50000, 100000, 1000000, 2000000

Speedup = CPU-reference / ours (both on this M3 Max). References (now installed): scanpy,
scikit-learn, harmonypy, bbknn, scrublet. A reference is skipped past a per-function size
cap (marked ref=NA) when it's impractically slow; ours is still timed. Functions that can't
run at a size (exact O(n²)) are skipped with a note.
"""

import csv
import gc
import os
import sys
import time
import warnings

import numpy as np

from metasinglecell import config

warnings.filterwarnings("ignore")


def best(fn, reps, warmup=1):
    """min-of-reps wall time after warm-up (defeats MLX first-call kernel compile)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    arg = sys.argv[1]
    res = config.results_dir("validation")
    csv_path = res / "benchmark.csv"
    import scanpy as sc
    import scipy.sparse as sp
    import mlx.core as mx
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.utils.extmath import randomized_svd

    from metasinglecell import preprocess as pp, tools, validation
    from metasinglecell.cluster import leiden as leiden_api
    from metasinglecell.decomposition import pca
    from metasinglecell.embedding import umap
    from metasinglecell.graph.csr_graph import Graph
    from metasinglecell.graph.louvain import louvain as louvain_gpu
    from metasinglecell.graph.leiden import leiden as leiden_gpu
    from metasinglecell.integration import harmonize
    from metasinglecell.neighbors import bbknn, neighbors
    from metasinglecell.sparse import CSR

    rng = np.random.default_rng(0)

    # ---- load real data ----
    if arg == "pbmc":
        label = "PBMC"
        ad = sc.datasets.pbmc3k(); sc.pp.filter_cells(ad, min_genes=200); sc.pp.filter_genes(ad, min_cells=3)
        counts = sp.csr_matrix(ad.X).astype(np.float32)
        var_names = ad.var_names.to_numpy()
    elif arg == "xenium2m":
        # real 2M dataset = an external Xenium cohort .h5ad (~2,035,266 x 5,101; lighter panel,
        # fits memory where 2M neurons x 20k would not). Set the path via the XENIUM_H5 env var.
        import h5py
        label = "2M"
        XEN = os.environ.get("XENIUM_H5")
        if not XEN:
            raise SystemExit("set XENIUM_H5 to a 2M-cell Xenium .h5ad to run the 2M benchmark")
        with h5py.File(XEN, "r") as f:
            L = f["layers"]["counts"]
            counts = sp.csr_matrix((L["data"][:], L["indices"][:], L["indptr"][:]),
                                   shape=tuple(L.attrs["shape"])).astype(np.float32)
            vi = f["var"].attrs.get("_index", "_index")
            var_names = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"][vi][:]])
        gmask = np.asarray((counts > 0).sum(0)).ravel() >= 3
        counts = counts[:, gmask]; var_names = var_names[gmask]
        gc.collect()
    else:
        N = int(arg); label = f"{N//1000}k" if N < 1_000_000 else f"{N//1_000_000}M"
        ad = sc.read_10x_h5("data/external/1M_neurons.h5"); ad.var_names_make_unique()
        base = ad.X.tocsr().astype(np.float32)
        idx = (rng.choice(base.shape[0], N, replace=(N > base.shape[0])))
        counts = base[idx]
        # drop all-zero genes to mirror a real filtered object
        gmask = np.asarray((counts > 0).sum(0)).ravel() >= 3
        counts = counts[:, gmask]
        var_names = ad.var_names.to_numpy()[gmask]
        del ad, base; gc.collect()
    n = counts.shape[0]
    reps = 3 if n <= 100_000 else 1
    print(f"=== {label}: {n} cells x {counts.shape[1]} genes (reps={reps}) ===", flush=True)

    # Incremental, resumable CSV: header once; append each row immediately (a hang never
    # loses prior functions); skip (size,function) pairs already recorded.
    new = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="")
    wcsv = csv.writer(fcsv)
    if new:
        wcsv.writerow(["size", "n", "function", "ours_s", "ref_s", "speedup",
                       "acc_metric", "acc_value", "note"]); fcsv.flush()
    done = set()
    if not new:
        import csv as _c
        with open(csv_path) as f:
            for row in _c.reader(f):
                if len(row) >= 3:
                    done.add((row[0], row[2]))

    def record(name, gpu_s, cpu_s, acc_name, acc_val, note=""):
        spd = (cpu_s / gpu_s) if (cpu_s and gpu_s) else float("nan")
        wcsv.writerow([label, n, name, f"{gpu_s:.4f}" if gpu_s else "",
                       f"{cpu_s:.4f}" if cpu_s else "NA", f"{spd:.2f}" if spd == spd else "NA",
                       acc_name, acc_val, note]); fcsv.flush()
        print(f"  {name:24s} ours={gpu_s:.3f}s ref={'NA' if not cpu_s else f'{cpu_s:.3f}s'} "
              f"spd={'NA' if spd!=spd else f'{spd:.1f}x'} {acc_name}={acc_val}", flush=True)

    def bench(name, gpu_fn, cpu_fn, acc_fn, ref_max=10**12, r=None):
        if (label, name) in done:
            print(f"  {name:24s} (skip; already recorded)", flush=True); return
        try:
            gs = best(gpu_fn, r or reps)
        except Exception as e:
            record(name, 0, None, "err", str(e)[:40], "ours-failed"); return None
        cs, acc_name, acc_val, note = None, "", "", ""
        if cpu_fn is not None and n <= ref_max:
            try:
                cs = best(cpu_fn, 1)
            except Exception as e:
                note = f"ref-err:{str(e)[:25]}"
        elif n > ref_max:
            note = "ref NA@scale"
        if acc_fn is not None:
            try:
                acc_name, acc_val = acc_fn()
            except Exception as e:
                acc_name, acc_val = "acc-err", str(e)[:25]
        record(name, gs, cs, acc_name, acc_val, note)
        gc.collect()

    # ===== preprocessing (full gene set) =====
    adl = sc.AnnData(counts.copy());
    bench("normalize+log1p",
          lambda: CSR.from_scipy(counts).normalize_total(1e4).log1p().data,
          lambda: _sc_norm(sc, counts),
          lambda: _acc_norm(sc, pp, counts))

    lognorm = CSR.from_scipy(counts).normalize_total(1e4).log1p()
    adl = sc.AnnData(counts.copy()); sc.pp.normalize_total(adl, target_sum=1e4); sc.pp.log1p(adl)
    bench("highly_variable_genes",
          lambda: pp.highly_variable_genes(lognorm, n_top_genes=2000)["highly_variable"].to_numpy(),
          lambda: _sc_hvg(sc, adl),
          lambda: _acc_hvg(sc, pp, lognorm, adl))

    hv = pp.highly_variable_genes(lognorm, n_top_genes=2000)["highly_variable"].to_numpy()
    counts_hv = counts[:, hv].tocsr()
    lognorm_hv_sp = sp.csr_matrix(adl[:, hv].X).astype(np.float32)
    # past 500k the full-gene objects (~12GB) + dense downstream OOM; the remaining
    # functions only need the HVG subset / embedding, so free them now.
    if n > 500_000:
        del counts, lognorm, adl; gc.collect()

    bench("pca(sparse)",
          lambda: pca(CSR.from_scipy(lognorm_hv_sp), n_comps=50, solver="randomized")[0],
          (lambda: randomized_svd(_centered(lognorm_hv_sp), 50, n_iter=5, random_state=0)) if n <= 1_000_000 else None,
          lambda: _acc_pca(pca, randomized_svd, validation, lognorm_hv_sp), ref_max=1_000_000)

    if n <= 500_000:  # dense n×2000 residual: ours 8GB + scanpy ref 8GB OOMs the 48GB M3 at 1M+
        bench("normalize_pearson_residuals",
              lambda: pp.normalize_pearson_residuals(counts_hv, theta=100.0),
              lambda: _sc_pr(sc, counts_hv),
              lambda: _acc_pr(sc, pp, counts_hv), ref_max=500_000)

    if n <= 200_000:
        tot = np.asarray(adl.X.sum(1)).ravel()
        bench("regress_out",
              lambda: pp.regress_out(adl[:, hv].X, tot),
              lambda: _sc_regress(sc, adl, hv, tot),
              lambda: _acc_regress(sc, pp, adl, hv, tot), ref_max=200_000)

    # PCA embedding + graph (prep for downstream); time neighbors
    emb = pca(CSR.from_scipy(lognorm_hv_sp), n_comps=50, solver="randomized")[0].astype(np.float32)
    bench("neighbors",
          lambda: neighbors(emb, n_neighbors=15),
          (lambda: _sc_neighbors(sc, emb)) if n <= 1_000_000 else None,
          None, ref_max=1_000_000)
    dist_g, conn = neighbors(emb, n_neighbors=15)
    g_graph = Graph.from_scipy(conn.astype(np.float32))     # our GPU graph — built OUTSIDE timing

    # Pre-build the igraph graph too (OUTSIDE timing) so the clustering benchmark measures the
    # ALGORITHM only on both sides — not graph construction (conn.nonzero + list(zip) over millions
    # of edges + ig.Graph build), which at 1M is seconds and would unfairly inflate the reference.
    g_ig = None
    if n <= 1_000_000:
        import igraph as ig
        _s, _d = conn.nonzero(); _m = _s < _d
        g_ig = ig.Graph(n=conn.shape[0], edges=list(zip(_s[_m].tolist(), _d[_m].tolist())))

    # ===== clustering (algorithm-only timing: graphs pre-built above) =====
    bench("louvain",
          lambda: louvain_gpu(g_graph, 1.0),
          (lambda: g_ig.community_multilevel()) if g_ig is not None else None,
          lambda: _acc_modularity(g_graph, louvain_gpu, _ig_louvain, conn, validation), ref_max=1_000_000, r=1)
    bench("leiden",
          lambda: leiden_gpu(g_graph, 1.0),
          (lambda: g_ig.community_leiden(objective_function="modularity")) if g_ig is not None else None,
          None, ref_max=1_000_000, r=1)
    groups = leiden_api(conn, resolution=1.0, backend="igraph")

    # ===== embeddings =====
    bench("umap", lambda: umap(conn),
          (lambda: _ul_umap(emb)) if n <= 200_000 else None, None, ref_max=200_000, r=1)
    if n <= 100_000:    # above 30k ours IS sklearn-BH; running it at 1M is slow & ~1x (skip)
        bench("tsne", lambda: tools.tsne(emb),
              lambda: _sk_tsne(emb), None, ref_max=100_000, r=1)
    # ARPACK eigsh on the sparse graph scales (ours at all sizes); scanpy reference only ≤1M
    bench("diffmap", lambda: tools.diffmap(conn, n_comps=15),
          (lambda: _sc_diffmap(sc, emb)) if n <= 1_000_000 else None, None, ref_max=1_000_000, r=1)
    if n <= 200_000:
        # igraph 'fr' layout reference is O(n²) (hangs at scale) -> reference only at PBMC.
        bench("draw_graph", lambda: tools.draw_graph(conn),
              (lambda: _sc_drawgraph(sc, emb)) if n <= 10_000 else None, None, ref_max=10_000, r=1)

    # ===== tools =====
    if n <= 500_000:
        full = np.asarray(adl.X.todense(), np.float32)
        bench("rank_genes_groups(t-test)",
              lambda: tools.rank_genes_groups(full, groups, var_names=var_names),
              lambda: _sc_rgg(sc, adl, groups, "t-test"), None, ref_max=500_000)
        if n <= 50_000:    # logreg = sklearn on all genes (both sides) -> minutes & ~1x past 50k
            bench("rank_genes_groups(logreg)",
                  lambda: tools.rank_genes_groups(full, groups, var_names=var_names, method="logreg"),
                  lambda: _sc_rgg(sc, adl, groups, "logreg"), None, ref_max=50_000, r=1)
        gl = list(var_names[np.flatnonzero(hv)[:20]])      # 20 HVGs (always valid in gene_pool)
        bench("score_genes",
              lambda: tools.score_genes(full, gl, var_names),
              lambda: _sc_score(sc, adl, gl), None, ref_max=500_000)
        del full; gc.collect()
    bench("kmeans", lambda: tools.kmeans(emb, 15),
          lambda: KMeans(15, n_init=3, random_state=0).fit_predict(emb) if n <= 500_000 else None,
          None, ref_max=500_000)

    # ===== integration + scrublet (batch split) =====
    if n <= 200_000:
        batch = (rng.random(n) < 0.5).astype(int)
        emb_b = emb.copy(); emb_b[batch == 1, :5] += 3.0
        bench("harmonize", lambda: harmonize(emb_b, batch),
              (lambda: _hp_harmony(emb_b, batch)) if n <= 200_000 else None, None, ref_max=200_000, r=1)
        bench("bbknn", lambda: bbknn(emb_b, batch, 3),
              (lambda: _bb_bbknn(emb_b, batch)) if n <= 200_000 else None, None, ref_max=200_000, r=1)
    if n <= 50_000:
        ni = max(1, n // 50)
        pr = rng.integers(0, n, (ni, 2))
        Caug = sp.vstack([counts, counts[pr[:, 0]] + counts[pr[:, 1]]]).tocsr()
        bench("scrublet", lambda: pp.scrublet(Caug),
              lambda: _sc_scrublet(sc, Caug), None, ref_max=50_000, r=1)

    fcsv.close()
    print("\ndone", flush=True)


# ---- reference / accuracy helpers (kept tiny; copies made where scanpy mutates) ----
def _sc_norm(sc, counts):
    a = sc.AnnData(counts.copy()); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); return a.X
def _acc_norm(sc, pp, counts):
    from metasinglecell.sparse import CSR
    import numpy as np
    a = sc.AnnData(counts.copy()); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    mine = np.asarray(CSR.from_scipy(counts).normalize_total(1e4).log1p().data)
    return "max|d|", f"{np.abs(mine - a.X.tocsr().data).max():.2e}"
def _sc_hvg(sc, adl):
    a = adl.copy(); sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat"); return a.var.highly_variable.to_numpy()
def _acc_hvg(sc, pp, lognorm, adl):
    import numpy as np
    m = pp.highly_variable_genes(lognorm, n_top_genes=2000)["highly_variable"].to_numpy()
    a = adl.copy(); sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    r = a.var.highly_variable.to_numpy()
    return "overlap", f"{(m & r).sum() / max(r.sum(),1):.3f}"
def _centered(X):
    import numpy as np
    D = np.asarray(X.todense(), np.float64); D -= D.mean(0); return D
def _acc_pca(pca, rsvd, validation, X):
    import numpy as np
    from metasinglecell.sparse import CSR
    _, comps, _ = pca(CSR.from_scipy(X), n_comps=50, solver="randomized")
    _, _, Vt = rsvd(_centered(X), 50, n_iter=5, random_state=0)
    return "subspace", f"{validation.subspace_overlap(comps.T, Vt.T):.3f}"
def _sc_pr(sc, counts_hv):
    a = sc.AnnData(counts_hv.copy()); sc.experimental.pp.normalize_pearson_residuals(a, theta=100.0); return a.X
def _acc_pr(sc, pp, counts_hv):
    import numpy as np
    m = pp.normalize_pearson_residuals(counts_hv, theta=100.0)
    a = sc.AnnData(counts_hv.copy()); sc.experimental.pp.normalize_pearson_residuals(a, theta=100.0)
    return "corr", f"{np.corrcoef(np.asarray(m).ravel(), np.asarray(a.X).ravel())[0,1]:.4f}"
def _sc_regress(sc, adl, hv, tot):
    a = adl[:, hv].copy(); a.obs["t"] = tot; sc.pp.regress_out(a, ["t"]); return a.X
def _acc_regress(sc, pp, adl, hv, tot):
    import numpy as np
    m = pp.regress_out(adl[:, hv].X, tot)
    a = adl[:, hv].copy(); a.obs["t"] = tot; sc.pp.regress_out(a, ["t"])
    return "corr", f"{np.corrcoef(np.asarray(a.X).ravel(), m.ravel())[0,1]:.4f}"
def _sc_neighbors(sc, emb):
    a = sc.AnnData(emb.copy()); a.obsm["X_pca"] = emb; sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca"); return a
def _ig_louvain(conn):
    import igraph as ig, numpy as np
    s, d = conn.nonzero(); m = s < d
    return ig.Graph(n=conn.shape[0], edges=list(zip(s[m].tolist(), d[m].tolist()))).community_multilevel()
def _ig_leiden(conn):
    import igraph as ig, numpy as np
    s, d = conn.nonzero(); m = s < d
    return ig.Graph(n=conn.shape[0], edges=list(zip(s[m].tolist(), d[m].tolist()))).community_leiden(objective_function="modularity")
def _acc_modularity(g, louvain_gpu, ig_fn, conn, validation):
    import numpy as np, mlx.core as mx
    from metasinglecell.graph.primitives import modularity
    lab = louvain_gpu(g, 1.0); _, d = np.unique(lab, return_inverse=True)
    return "Q", f"{float(modularity(g, mx.array(d.astype(np.int32)), 1.0)):.3f}"
def _ul_umap(emb):
    import umap as ul; return ul.UMAP(n_neighbors=15, random_state=0).fit_transform(emb)
def _sk_tsne(emb):
    from sklearn.manifold import TSNE; return TSNE(init="pca", random_state=0).fit_transform(emb)
def _sc_diffmap(sc, emb):
    a = sc.AnnData(emb.copy()); a.obsm["X_pca"] = emb; sc.pp.neighbors(a, use_rep="X_pca"); sc.tl.diffmap(a); return a
def _sc_drawgraph(sc, emb):
    a = sc.AnnData(emb.copy()); a.obsm["X_pca"] = emb; sc.pp.neighbors(a, use_rep="X_pca"); sc.tl.draw_graph(a, layout="fr"); return a
def _sc_rgg(sc, adl, groups, method):
    a = adl.copy(); a.obs["g"] = [str(x) for x in groups]; a.obs["g"] = a.obs["g"].astype("category")
    sc.tl.rank_genes_groups(a, "g", method=method); return a
def _sc_score(sc, adl, gl):
    a = adl.copy(); sc.tl.score_genes(a, gl, random_state=0); return a
def _hp_harmony(emb_b, batch):
    import harmonypy, pandas as pd
    return harmonypy.run_harmony(emb_b, pd.DataFrame({"b": batch}), ["b"]).Z_corr.T
def _bb_bbknn(emb_b, batch):
    import bbknn, scanpy as sc
    a = sc.AnnData(emb_b.copy()); a.obsm["X_pca"] = emb_b
    a.obs["b"] = [str(x) for x in batch]; a.obs["b"] = a.obs["b"].astype("category")
    bbknn.bbknn(a, batch_key="b", neighbors_within_batch=3); return a
def _sc_scrublet(sc, Caug):
    a = sc.AnnData(Caug.copy()); sc.pp.scrublet(a, random_state=0); return a


if __name__ == "__main__":
    main()
