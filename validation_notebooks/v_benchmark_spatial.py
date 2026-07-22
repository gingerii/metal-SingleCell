"""Spatial (`gr`) benchmark — our Metal/MLX squidpy-GPU functions vs squidpy CPU, real multi-platform data.

Extends the speed table (`results/validation/benchmark.csv`) with the five spatial functions, which had
correctness parity (`v_realspatial.py`) but were never timed. Same protocol as `v_benchmark.py`:
warm-up discarded, best-of-N, `mx.eval` barriers, matched work both sides (same graph / thresholds /
n_perms). One dataset per process (memory-safe at 2M).

    python v_benchmark_spatial.py <dataset_key>     # visium2k stereo19k xenium63k merfish81k xeniumbreast253k xenium2m

squidpy is timed only up to a per-function size cap (past it squidpy OOMs/hours → ours-only `(N s)`,
the same convention neighbors/umap use at 2M). Losses are reported, not suppressed.
"""
import csv
import gc
import sys
import time
import warnings

import numpy as np

from metalsinglecell import config

warnings.filterwarnings("ignore")

GD = "/Users/f006z2w/Library/CloudStorage/GoogleDrive-ian.gingerich.gr@dartmouth.edu/My Drive"
DATASETS = {
    "visium2k":        (f"{GD}/Quaternion_project/data/external/V1_Adult_Mouse_Brain.h5ad", "Visium"),
    "stereo19k":       (f"{GD}/Quaternion_project/data/external/stereoseq_olf.h5ad", "Stereo-seq"),
    "xenium63k":       (f"{GD}/Atlas_svd_project/data/processed/xenium_brain_5k.h5ad", "Xenium"),
    "merfish81k":      (f"{GD}/Atlas_svd_project/data/processed/merfish_sagital_brain.h5ad", "MERFISH"),
    "xeniumbreast253k": (f"{GD}/Quaternion_project/data/external/xenium_breast_matched/xenium_breast_s1bot_processed.h5ad", "Xenium"),
    "xenium2m":        ("/Users/f006z2w/Desktop/Xenium_Claude_test/data/processed/xenium/integrated_data.h5ad", "Xenium"),
}

# Per-function squidpy-reference size cap (cells). Past it, squidpy is impractical → ours-only.
REF_CAP = {
    "spatial_neighbors": 2_100_000,   # sklearn KD-tree scales on 2-D
    "spatial_autocorr_moran": 100_000,
    "spatial_autocorr_geary": 100_000,
    "co_occurrence": 100_000,         # squidpy's is the slow pairwise one
    "ligrec": 100_000,                # permutation-heavy
    "calculate_niche": 100_000,       # squidpy neighborhood flavor runs leiden
}
# spatial_neighbors now uses the uniform-grid (cell-list) index (_knn_grid) — exact and O(n), so it
# scales to 2M (the old brute O(n²) OOM'd past ~120k). No ours-cap needed anymore.
OURS_CAP = {}


def best(fn, reps, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    key = sys.argv[1]
    path, platform = DATASETS[key]
    res = config.results_dir("validation")
    csv_path = res / "benchmark.csv"

    import anndata as ad_io
    import scanpy as sc
    import scipy.sparse as sp
    import squidpy as sq
    import mlx.core as mx

    from metalsinglecell import spatial as gr
    from metalsinglecell import tools

    print(f"=== loading {key} ({platform}) ===", flush=True)
    ad = ad_io.read_h5ad(path)
    ad.var_names_make_unique()
    n = ad.n_obs
    coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
    # precise size label (avoid colliding with the non-spatial "2k"=2000 rows)
    if n < 10_000:
        label = f"{n/1000:.1f}k"
    elif n < 1_000_000:
        label = f"{n//1000}k"
    else:
        label = f"{n/1e6:.1f}M".replace(".0M", "M")

    # expression: lognorm (compute if the stored X looks like raw counts)
    X = ad.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.astype(np.float32)
    if X.max() > 50:                                        # looks like counts → normalize
        a2 = sc.AnnData(X.copy()); sc.pp.normalize_total(a2, target_sum=1e4); sc.pp.log1p(a2)
        X = sp.csr_matrix(a2.X).astype(np.float32)
    ad.X = X

    # embedding for labels: reuse stored X_pca if present, else a quick GPU PCA of HVG
    if "X_pca" in ad.obsm:
        emb = np.asarray(ad.obsm["X_pca"], dtype=np.float32)[:, :30]
    else:
        from metalsinglecell.sparse import CSR
        from metalsinglecell.decomposition import pca
        ng = min(2000, ad.n_vars)
        sc.pp.highly_variable_genes(ad, n_top_genes=ng, flavor="seurat")
        hv = ad.var["highly_variable"].to_numpy()
        emb = pca(CSR.from_scipy(sp.csr_matrix(ad[:, hv].X).astype(np.float32)),
                  n_comps=min(30, hv.sum() - 1), solver="randomized")[0].astype(np.float32)

    # labels via GPU kmeans (fast at any scale) → categorical obs for squidpy
    K = 15
    labels = tools.kmeans(emb, K).astype(str)
    ad.obs["clab"] = labels
    ad.obs["clab"] = ad.obs["clab"].astype("category")
    print(f"    {n} cells x {ad.n_vars} genes; {len(np.unique(labels))} kmeans labels", flush=True)

    reps = 3 if n <= 100_000 else 1
    new = not csv_path.exists()
    fcsv = open(csv_path, "a", newline=""); wcsv = csv.writer(fcsv)
    if new:
        wcsv.writerow(["size", "n", "function", "ours_s", "ref_s", "speedup",
                       "acc_metric", "acc_value", "note"])

    def record(name, gpu_s, cpu_s, acc_name, acc_val, note=""):
        spd = (cpu_s / gpu_s) if (cpu_s and gpu_s) else float("nan")
        wcsv.writerow([label, n, name, f"{gpu_s:.4f}" if gpu_s else "",
                       f"{cpu_s:.4f}" if cpu_s else "NA", f"{spd:.2f}" if spd == spd else "NA",
                       acc_name, acc_val, note]); fcsv.flush()
        print(f"  {name:26s} ours={gpu_s:.3f}s ref={'NA' if not cpu_s else f'{cpu_s:.3f}s'} "
              f"spd={'NA' if spd != spd else f'{spd:.2f}x'}  {note}", flush=True)

    def bench(name, gpu_fn, cpu_fn, acc_name="", acc_val="", note="", r=None):
        if name in OURS_CAP and n > OURS_CAP[name]:
            record(name, 0, None, acc_name, acc_val,
                   (note + f"; ours brute O(n²) impractical >{OURS_CAP[name]//1000}k — grid-hash follow-up").strip("; "))
            return
        try:
            gs = best(gpu_fn, r or reps)
        except Exception as e:
            record(name, 0, None, "err", str(e)[:40], "ours-failed"); return
        cs = None
        cap = REF_CAP.get(name, 10**12)
        if cpu_fn is not None and n <= cap:
            try:
                cs = best(cpu_fn, 1)
            except Exception as e:
                note = (note + f"; ref-err:{str(e)[:30]}").strip("; ")
        elif n > cap:
            note = (note + f"; squidpy impractical >{cap//1000}k").strip("; ")
        record(name, gs, cs, acc_name, acc_val, note)
        gc.collect()

    plat = f"{platform}"

    # ---- spatial_neighbors (build the shared graph; reuse squidpy's for autocorr) ----
    bench("spatial_neighbors",
          lambda: gr.spatial_neighbors(coords, n_neighs=6),
          lambda: sq.gr.spatial_neighbors(ad, n_neighs=6, coord_type="generic"),
          note=plat)
    sq.gr.spatial_neighbors(ad, n_neighs=6, coord_type="generic")   # ensure graph exists
    conn = ad.obsp["spatial_connectivities"]

    # ---- spatial_autocorr: Moran & Geary (same graph + same gene set + same n_perms) ----
    ng = min(200, ad.n_vars)
    genes = list(ad.var_names[:ng])
    Xg = np.asarray(ad[:, genes].X.todense(), dtype=np.float32)
    NP = 100
    for mode, key2 in [("moran", "spatial_autocorr_moran"), ("geary", "spatial_autocorr_geary")]:
        bench(key2,
              lambda m=mode: gr.spatial_autocorr(Xg, conn, mode=m, n_perms=NP),
              lambda m=mode: sq.gr.spatial_autocorr(ad, mode=m, genes=genes, n_perms=NP, seed=0,
                                                    connectivity_key="spatial_connectivities"),
              acc_name="corr", acc_val="1.00(v_realspatial)", note=f"{plat}; {ng}g np={NP}")

    # ---- co_occurrence (same interval count) ----
    NI = 50
    bench("co_occurrence",
          lambda: gr.co_occurrence(coords, labels, n_intervals=NI),
          lambda: sq.gr.co_occurrence(ad, cluster_key="clab", interval=NI),
          acc_name="corr", acc_val="1.0000", note=f"{plat}; {NI} intervals")

    # ---- ligrec (same LR pairs + same n_perms) ----
    rng = np.random.default_rng(0)
    gi = rng.choice(ad.n_vars, 20, replace=False)
    pairs = [(str(ad.var_names[gi[2 * k]]), str(ad.var_names[gi[2 * k + 1]])) for k in range(10)]
    LP = 100
    import pandas as pd
    inter = pd.DataFrame(pairs, columns=["source", "target"])
    bench("ligrec",
          lambda: gr.ligrec(ad.X, labels, pairs, ad.var_names.to_numpy(), n_perms=LP),
          lambda: sq.gr.ligrec(ad, cluster_key="clab", interactions=inter, n_perms=LP,
                               threshold=0.0, use_raw=False, seed=0),
          note=f"{plat}; 10 pairs np={LP}")

    # ---- calculate_niche (neighborhood composition) ----
    # note: squidpy's neighborhood flavor does composition→leiden; ours does composition→kmeans —
    # matched on the composition SpMM, different niche-clustering backend (flagged in the note).
    bench("calculate_niche",
          lambda: gr.calculate_niche(conn, labels, n_niches=K),
          lambda: sq.gr.calculate_niche(ad, flavor="neighborhood", groups="clab",
                                        n_neighbors=6, resolutions=1.0,
                                        spatial_connectivities_key="spatial_connectivities"),
          acc_name="composition", acc_val="exact(v_realspatial)",
          note=f"{plat}; sq=composition+leiden vs ours composition+kmeans")

    fcsv.close()
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
