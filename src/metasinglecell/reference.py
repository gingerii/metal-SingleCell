"""CPU reference oracle (the "parity oracle").

Runs the standard scanpy single-cell workflow on PBMC3k in **float64** and
snapshots every intermediate to disk. Every GPU kernel / scanpy drop-in we build
in later stages is validated for numerical parity against these snapshots.

Why this exists: Apple GPUs are fp32-only, so "correct" means "reproduces the
fp64 CPU-scanpy result within a justified tolerance". You can't measure that
without a frozen ground truth — this module is that ground truth.

Heavy deps (scanpy) are imported inside the function so the package stays
importable in any environment. Run via ``validation_notebooks/00_cpu_reference_oracle.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import config

# Workflow knobs — kept explicit so the oracle is fully reproducible.
SEED = 0
MIN_GENES = 200
MIN_CELLS = 3
TARGET_SUM = 1e4
N_TOP_GENES = 2000
SCALE_MAX = 10.0
N_PCS = 50
N_NEIGHBORS = 15
LEIDEN_RES = 1.0


def _as_dense_f64(x) -> np.ndarray:
    return (x.toarray() if sp.issparse(x) else np.asarray(x)).astype(np.float64)


def _summary(name: str, arr: np.ndarray) -> dict:
    """Compact, comparable fingerprint of an array (shape + robust stats)."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    finite = a[np.isfinite(a)]
    return {
        "name": name,
        "shape": list(np.asarray(arr).shape),
        "dtype": str(np.asarray(arr).dtype),
        "n_nonfinite": int(a.size - finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "l2": float(np.linalg.norm(finite)) if finite.size else None,
    }


def run_cpu_reference(out_dir: Path | None = None, log: logging.Logger | None = None) -> Path:
    """Run the fp64 scanpy workflow on PBMC3k and snapshot every intermediate.

    Saves arrays to ``data/processed/reference/`` (regenerable inputs) and a
    manifest + summary log to ``results/reference/``. Returns the snapshot dir.
    """
    import scanpy as sc

    log = log or logging.getLogger("reference")
    snap_dir = out_dir or (config.PROCESSED_DIR / "reference")
    snap_dir.mkdir(parents=True, exist_ok=True)
    res_dir = config.results_dir("reference")

    summaries: list[dict] = []

    def snap(name: str, arr) -> None:
        """Persist a full-precision array and record its fingerprint."""
        a = np.asarray(arr)
        np.save(snap_dir / f"{name}.npy", a)
        s = _summary(name, a)
        summaries.append(s)
        log.info("snapshot %-22s shape=%s l2=%s", name, s["shape"], s["l2"])

    sc.settings.verbosity = 1
    np.random.seed(SEED)

    # --- load raw counts -------------------------------------------------
    adata = sc.datasets.pbmc3k()
    adata.X = adata.X.astype(np.float64)
    log.info("loaded pbmc3k: %d cells x %d genes", *adata.shape)

    # --- basic QC filter -------------------------------------------------
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    log.info("after filter: %d cells x %d genes", *adata.shape)
    snap("00_counts", _as_dense_f64(adata.X))

    # --- QC metrics (sparse reductions) ----------------------------------
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    snap("01_total_counts", adata.obs["total_counts"].to_numpy())
    snap("01_n_genes_by_counts", adata.obs["n_genes_by_counts"].to_numpy())

    # --- normalize_total + log1p (elementwise on sparse) -----------------
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    snap("02_normalized", _as_dense_f64(adata.X))
    sc.pp.log1p(adata)
    snap("03_lognorm", _as_dense_f64(adata.X))

    # --- highly variable genes (per-gene variance) ----------------------
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat")
    snap("04_hvg_means", adata.var["means"].to_numpy())
    snap("04_hvg_dispersions_norm", adata.var["dispersions_norm"].to_numpy())
    snap("04_hvg_flag", adata.var["highly_variable"].to_numpy().astype(np.int8))

    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()

    # --- scale (z-score, densifies) -------------------------------------
    sc.pp.scale(adata, max_value=SCALE_MAX)
    snap("05_scaled", _as_dense_f64(adata.X))

    # --- PCA (truncated SVD) --------------------------------------------
    sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack", random_state=SEED)
    snap("06_X_pca", adata.obsm["X_pca"])
    snap("06_pca_variance_ratio", adata.uns["pca"]["variance_ratio"])
    snap("06_pca_components", adata.varm["PCs"])

    # --- neighbors (KNN graph on the embedding) -------------------------
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=SEED)
    sp.save_npz(snap_dir / "07_connectivities.npz", adata.obsp["connectivities"].tocsr())
    sp.save_npz(snap_dir / "07_distances.npz", adata.obsp["distances"].tocsr())
    summaries.append(_summary("07_connectivities", adata.obsp["connectivities"].data))

    # --- leiden clustering ----------------------------------------------
    sc.tl.leiden(adata, resolution=LEIDEN_RES, random_state=SEED, flavor="igraph", n_iterations=2)
    snap("08_leiden", adata.obs["leiden"].cat.codes.to_numpy())

    # --- UMAP embedding --------------------------------------------------
    sc.tl.umap(adata, random_state=SEED)
    snap("09_X_umap", adata.obsm["X_umap"])

    # --- manifest --------------------------------------------------------
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dataset": "pbmc3k",
        "dtype": "float64",
        "params": {
            "seed": SEED, "min_genes": MIN_GENES, "min_cells": MIN_CELLS,
            "target_sum": TARGET_SUM, "n_top_genes": N_TOP_GENES,
            "scale_max": SCALE_MAX, "n_pcs": N_PCS,
            "n_neighbors": N_NEIGHBORS, "leiden_res": LEIDEN_RES,
        },
        "scanpy_version": sc.__version__,
        "numpy_version": np.__version__,
        "snapshots": summaries,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (res_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest with %d snapshots -> %s", len(summaries), snap_dir)
    return snap_dir
