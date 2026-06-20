"""Benchmark: Metal/MLX ops vs CPU (scanpy/sklearn) across dataset sizes.

Honest timing: each GPU op is warmed up once (MLX compiles + caches the kernel on
first launch), then timed with the result forced via mx.eval and the intrinsic
host<->device transfer included. We report the best of a few repeats. Synthetic
sparse counts (~7% density) stand in for real data so we can scale n_cells.

The point is to find the crossover: tiny data favors the CPU (kernel-launch +
transfer overhead dominates); the GPU should win once the arithmetic dominates.

    conda activate metasinglecell
    python validation_notebooks/07_benchmark.py
"""

import logging
import time

import numpy as np
import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.decomposition import pca
from metasinglecell.sparse import CSR

N_GENES = 2000
SIZES = [2_700, 10_000, 30_000, 100_000]
KNN_SIZES = [2_700, 10_000, 30_000]  # brute-force KNN is O(n^2) memory


def _best(fn, repeats=3):
    """Best wall time (s) over `repeats`, after one warm-up call."""
    fn()  # warm up (compile/cache, allocations)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _synthetic_counts(n_cells, n_genes, density=0.07, seed=0):
    rng = np.random.default_rng(seed)
    m = sp.random(n_cells, n_genes, density=density, format="csr", random_state=seed)
    m.data = rng.integers(1, 50, size=m.data.size).astype(np.float32)
    return m


def main() -> None:
    res_dir = config.results_dir("benchmark")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "benchmark.log", mode="w"), logging.StreamHandler()])
    log = logging.getLogger("bench")

    import mlx.core as mx
    import scanpy as sc
    from sklearn.neighbors import NearestNeighbors
    from sklearn.utils.extmath import randomized_svd

    records = []

    def add(op, n, gpu_s, cpu_s):
        rec = {"op": op, "n_cells": n, "gpu_s": round(gpu_s, 5),
               "cpu_s": round(cpu_s, 5), "speedup": round(cpu_s / gpu_s, 2)}
        records.append(rec)
        log.info("%-18s n=%-7d gpu=%.4fs cpu=%.4fs  speedup=%.2fx",
                 op, n, gpu_s, cpu_s, rec["speedup"])

    for n in SIZES:
        counts = _synthetic_counts(n, N_GENES)
        # --- normalize_total + log1p ---
        csr = CSR.from_scipy(counts)
        gpu = _best(lambda: csr.normalize_total(1e4).log1p().data)

        ad = sc.AnnData(counts.copy())
        def cpu_norm():
            a = ad.copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        cpu = _best(cpu_norm, repeats=1)
        add("normalize+log1p", n, gpu, cpu)

        # --- per-gene moments (HVG reduction) ---
        lognorm = csr.normalize_total(1e4).log1p()
        gpu = _best(lambda: lognorm.gene_moments())
        from scanpy.preprocessing._utils import _get_mean_var
        X = lognorm.toarray()
        Xexp = np.expm1(X)
        cpu = _best(lambda: _get_mean_var(Xexp), repeats=1)
        add("gene_moments(HVG)", n, gpu, cpu)

        # --- PCA randomized (on a scaled-like dense block of HVG size) ---
        Xd = np.ascontiguousarray(X[:, :N_GENES].astype(np.float32))
        gpu = _best(lambda: pca(Xd, n_comps=50, solver="randomized"))
        Xc = Xd.astype(np.float64) - Xd.astype(np.float64).mean(0)
        cpu = _best(lambda: randomized_svd(Xc, 50, n_oversamples=10, n_iter=7, random_state=0), repeats=1)
        add("pca_randomized", n, gpu, cpu)

    # --- brute-force KNN (separate sizes; O(n^2)) ---
    for n in KNN_SIZES:
        emb = np.random.default_rng(0).standard_normal((n, 50)).astype(np.float32)
        from metasinglecell.neighbors import _knn_gpu
        gpu = _best(lambda: _knn_gpu(emb, 15))
        nn = NearestNeighbors(n_neighbors=15, algorithm="brute")
        cpu = _best(lambda: nn.fit(emb).kneighbors(emb), repeats=1)
        add("knn_bruteforce", n, gpu, cpu)

    validation.write_report(records, "benchmark", "benchmark.csv")
    print("\n=== GPU speedup (CPU walltime / GPU walltime), >1 means GPU faster ===")
    for r in records:
        print(f"  {r['op']:<18} n={r['n_cells']:<7} {r['speedup']:>6.2f}x")


if __name__ == "__main__":
    main()
