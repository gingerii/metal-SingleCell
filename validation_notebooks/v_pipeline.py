"""Validation + benchmark: core front-end pipeline, GPU (ours) vs CPU (scanpy/sklearn).

Sweeps dataset sizes, measuring accuracy (vs CPU reference on the same data) and
speed (best-of-N walltime, GPU warmed up). Flags functions whose GPU speedup is
poor/negative so they can be optimized until hardware-bound. See CLAUDE.md
"Validation & benchmarking scheme".

    conda activate metasinglecell
    python validation_notebooks/v_pipeline.py            # default sizes
    python validation_notebooks/v_pipeline.py 10000 50000
"""

import logging
import sys
import time

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation

SIZES = [2_700, 10_000, 50_000, 100_000]
N_GENES = 2000


def _synthetic_counts(n, n_genes=N_GENES, density=0.07, seed=0):
    rng = np.random.default_rng(seed)
    m = sp.random(n, n_genes, density=density, format="csr", random_state=seed)
    m.data = rng.integers(1, 50, size=m.data.size).astype(np.float32)
    return m


def _best(fn, repeats=3, warmup=True):
    if warmup:
        fn()
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


def main():
    sizes = [int(s) for s in sys.argv[1:]] or SIZES
    res_dir = config.results_dir("validation")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(res_dir / "v_pipeline.log", mode="w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("v")

    import mlx.core as mx
    import scanpy as sc
    from sklearn.utils.extmath import randomized_svd

    from metasinglecell.decomposition import pca
    from metasinglecell.neighbors import _knn_gpu
    from metasinglecell.preprocess import highly_variable_genes
    from metasinglecell.sparse import CSR

    records = []

    def add(op, n, gpu_s, cpu_s, acc):
        sp_ = cpu_s / gpu_s
        flag = "" if sp_ >= 1.0 else "  <-- NEGATIVE SPEEDUP"
        records.append({"op": op, "n": n, "gpu_s": round(gpu_s, 4), "cpu_s": round(cpu_s, 4),
                        "speedup": round(sp_, 2), "accuracy": acc})
        log.info("%-16s n=%-8d gpu=%.4fs cpu=%.4fs  speedup=%6.2fx  acc=%s%s",
                 op, n, gpu_s, cpu_s, sp_, acc, flag)

    for n in sizes:
        log.info("\n=== n=%d cells ===", n)
        counts = _synthetic_counts(n)
        csr = CSR.from_scipy(counts)
        ad = sc.AnnData(counts.copy())

        # normalize_total + log1p
        gpu = _best(lambda: csr.normalize_total(1e4).log1p().data)
        cpu = _best(lambda: (lambda a: (sc.pp.normalize_total(a, target_sum=1e4), sc.pp.log1p(a)))(ad.copy()), repeats=1)
        mine_ln = csr.normalize_total(1e4).log1p()
        ref = ad.copy(); sc.pp.normalize_total(ref, target_sum=1e4); sc.pp.log1p(ref)
        acc = "r=%.5f" % np.corrcoef(np.asarray(mine_ln.toarray()).ravel(),
                                     np.asarray(ref.X.todense()).ravel())[0, 1]
        add("normalize+log1p", n, gpu, cpu, acc)

        # highly_variable_genes (seurat)
        gpu = _best(lambda: highly_variable_genes(mine_ln, n_top_genes=500))
        cpu = _best(lambda: sc.pp.highly_variable_genes(ref.copy(), n_top_genes=500, flavor="seurat"), repeats=1)
        mine_hv = highly_variable_genes(mine_ln, n_top_genes=500)["highly_variable"].to_numpy()
        rh = ref.copy(); sc.pp.highly_variable_genes(rh, n_top_genes=500, flavor="seurat")
        ov = (mine_hv & rh.var["highly_variable"].to_numpy()).sum() / max(rh.var["highly_variable"].sum(), 1)
        add("hvg_seurat", n, gpu, cpu, "overlap=%.3f" % ov)

        # PCA (randomized) on a dense block
        X = np.ascontiguousarray(mine_ln.toarray())
        gpu = _best(lambda: pca(X, n_comps=50, solver="randomized"))
        Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
        cpu = _best(lambda: randomized_svd(Xc, 50, n_iter=7, random_state=0), repeats=1)
        add("pca_randomized", n, gpu, cpu, "(seeded)")

        # KNN: our neighbors() uses exact GPU brute-force for small n, pynndescent
        # (scanpy's own default) for large n — the M3 GPU does not win this workload.
        from metasinglecell.neighbors import neighbors as _nbrs
        emb = X[:, :50].astype(np.float32)
        from sklearn.neighbors import NearestNeighbors
        gpu = _best(lambda: _nbrs(emb, n_neighbors=15)[0], repeats=1)
        cpu = _best(lambda: NearestNeighbors(n_neighbors=15).fit(emb).kneighbors(emb), repeats=1)
        add("knn(ours)", n, gpu, cpu, "brute<30k/pynndescent")

    path = validation.write_report(records, "validation", "v_pipeline.csv")
    neg = [r for r in records if r["speedup"] < 1.0]
    log.info("\n%d/%d ops with negative speedup (optimize): %s",
             len(neg), len(records), sorted({r["op"] for r in neg}))
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
