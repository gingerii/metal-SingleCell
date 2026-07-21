"""Post-fix validation — re-verify the 7 review-fix commits on paths the parity suite can't see.

Targets the *default-arg*, *raise*, and *edge-case* paths the explicit-arg parity notebooks
never exercise (Steps 2/4/5-disputes/6 of BRIEF_postfix_validation.md). Scale checks (Step 3
fp64-moments at ≥1M, Step 5 Leiden 986k) live in results/code_review/postfix_scale.py.

Runnable two ways:
    pytest tests/test_postfix_fixes.py            # asserting CI seed (CPU + Metal)
    python tests/test_postfix_fixes.py            # also appends rows to postfix_validation.csv
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp

ROWS = []          # (step, check, metric, value, pass) collected when run as a script


def _row(step, check, metric, value, passed):
    ROWS.append(dict(step=step, check=check, metric=metric, value=value, passed=bool(passed)))
    return passed


def _pbmc():
    import scanpy as sc
    from metasinglecell import config
    a = sc.read_h5ad(config.REPO_ROOT / "data" / "pbmc3k_raw.h5ad")
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    return a


# ---------------- Step 2 — new drop-in defaults match scanpy ----------------
def test_hvg_default_cutoff_matches_scanpy():
    import scanpy as sc
    from metasinglecell import pp as msc_pp
    a = _pbmc(); msc_pp.normalize_total(a, target_sum=1e4); msc_pp.log1p(a)
    b = a.copy()
    msc_pp.highly_variable_genes(a)                     # default n_top_genes=None → cutoff mode
    sc.pp.highly_variable_genes(b)                      # scanpy default cutoff mode
    ov = (a.var["highly_variable"].to_numpy() & b.var["highly_variable"].to_numpy()).sum()
    u = (a.var["highly_variable"].to_numpy() | b.var["highly_variable"].to_numpy()).sum()
    jac = ov / u
    _row("2", "hvg_default_cutoff_vs_scanpy", "jaccard", round(float(jac), 4), jac == 1.0)
    assert jac == 1.0, f"HVG default-cutoff Jaccard {jac} != 1.0"


def test_scale_default_no_clip_matches_scanpy():
    import scanpy as sc
    from metasinglecell import pp as msc_pp
    a = _pbmc(); msc_pp.normalize_total(a, target_sum=1e4); msc_pp.log1p(a)
    b = a.copy()
    msc_pp.scale(a)                                     # default max_value=None (no clip)
    sc.pp.scale(b)                                      # scanpy default no clip
    Xa = np.asarray(a.X); Xb = np.asarray(b.X)
    maxabs = float(np.max(np.abs(Xa - Xb)))
    _row("2", "scale_default_noclip_vs_scanpy", "max_abs_err", maxabs, maxabs < 1e-4)
    assert maxabs < 1e-4, f"scale default vs scanpy max_abs {maxabs}"


def test_qc_slot_names_and_qc_vars_match_scanpy():
    import scanpy as sc
    from metasinglecell import pp as msc_pp
    a = _pbmc(); b = a.copy()
    msc_pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=[50], log1p=True)
    sc.pp.calculate_qc_metrics(b, qc_vars=["mt"], percent_top=[50], log1p=True, inplace=True)
    slots_obs = {"total_counts", "n_genes_by_counts", "pct_counts_mt", "log1p_total_counts",
                 "pct_counts_in_top_50_genes"}
    slots_var = {"total_counts", "n_cells_by_counts", "mean_counts", "pct_dropout_by_counts"}
    ok_names = slots_obs <= set(a.obs.columns) and slots_var <= set(a.var.columns)
    _row("2", "qc_slot_names_present", "all_slots", int(ok_names), ok_names)
    assert ok_names, f"missing QC slots: obs {slots_obs - set(a.obs.columns)}, var {slots_var - set(a.var.columns)}"
    # pct_counts_mt values match scanpy
    d = float(np.max(np.abs(a.obs["pct_counts_mt"].to_numpy() - b.obs["pct_counts_mt"].to_numpy())))
    _row("2", "pct_counts_mt_vs_scanpy", "max_abs_err", d, d < 1e-3)
    assert d < 1e-3, f"pct_counts_mt vs scanpy max_abs {d}"


def test_qc_var_rename_incore_vs_streaming_slotnames():
    """gene_total_counts→total_counts must apply on BOTH in-core and streaming QC paths."""
    import anndata as ad, zarr
    from anndata.io import sparse_dataset
    from metasinglecell import config, pp as msc_pp
    from metasinglecell.backed import write_backed_zarr
    a = _pbmc()
    incore = a.copy(); msc_pp.calculate_qc_metrics(incore)
    zp = config.PROCESSED_DIR / "backed" / "pbmc_postfix_qc.zarr"
    import shutil
    if zp.exists():
        shutil.rmtree(zp)
    write_backed_zarr(a.copy(), zp, block_rows=1000)
    backed = ad.AnnData(X=sparse_dataset(zarr.open(str(zp))["X"]))
    backed.var_names = a.var_names
    msc_pp.calculate_qc_metrics(backed)
    same = ("total_counts" in incore.var.columns and "total_counts" in backed.var.columns
            and "gene_total_counts" not in incore.var.columns
            and "gene_total_counts" not in backed.var.columns)
    _row("2", "qc_var_rename_both_paths", "total_counts_both", int(same), same)
    assert same, "gene_total_counts→total_counts not applied identically on in-core & streaming"


# ---------------- Step 4 — BLOCKER kNN top-k for n_neighbors>32 ----------------
def test_knn_topk_all_k_match_bruteforce():
    from metasinglecell.neighbors import _knn_gpu, _TOPK_KERNEL_MAX_K
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 20)).astype(np.float32)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    for k in (15, 33, 50):
        idx, _ = _knn_gpu(X, k)
        ref = np.argsort(D, axis=1)[:, :k]
        recall = np.mean([len(set(idx[i]) & set(ref[i])) / k for i in range(X.shape[0])])
        uses_fallback = k > _TOPK_KERNEL_MAX_K
        ok = idx.shape == (500, k) and recall >= 0.99
        _row("4", f"knn_k{k}_{'fallback' if uses_fallback else 'kernel'}", "recall",
             round(float(recall), 4), ok)
        assert ok, f"kNN k={k} recall {recall} shape {idx.shape}"


# ---------------- Step 5 — disputed severities ----------------
def test_pca_zero_center_false_is_uncentered():
    from metasinglecell.decomposition import pca
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 40)).astype(np.float32) + 5.0

    def subs(A, B, k=10):
        Qa, _ = np.linalg.qr(A[:, :k]); Qb, _ = np.linalg.qr(B[:, :k])
        return float(np.linalg.norm(Qa.T @ Qb, "fro") ** 2 / k)
    xp, _, _ = pca(X, n_comps=10, solver="full", zero_center=False)
    U, S, _ = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    unc = U[:, :10] * S[:10]
    Xc = X - X.mean(0); Uc, Sc, _ = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    cen = Uc[:, :10] * Sc[:10]
    s_unc, s_cen = subs(xp, unc), subs(xp, cen)
    ok = s_unc >= 0.99 and s_cen < 0.95
    _row("5", "pca_zero_center_false_uncentered", "subspace_vs_uncentered", round(s_unc, 4), ok)
    assert ok, f"zero_center=False subspace vs uncentered {s_unc}, vs centered {s_cen}"


def test_pca_sparse_zero_center_false_raises():
    from metasinglecell.decomposition import pca
    from metasinglecell.sparse import CSR
    csr = CSR.from_scipy(sp.random(100, 30, density=0.2, format="csr", dtype=np.float32))
    raised = False
    try:
        pca(csr, n_comps=5, solver="randomized", zero_center=False)
    except ValueError:
        raised = True
    _row("5", "pca_sparse_zero_center_false_raises", "raised", int(raised), raised)
    assert raised, "sparse CSR PCA with zero_center=False should raise"


def test_backed_wrappers_reject():
    import anndata as ad, zarr, shutil
    from anndata.io import sparse_dataset
    from metasinglecell import config, pp as msc_pp, tl as msc_tl
    from metasinglecell.backed import write_backed_zarr
    a = _pbmc()
    zp = config.PROCESSED_DIR / "backed" / "pbmc_postfix_reject.zarr"
    if zp.exists():
        shutil.rmtree(zp)
    write_backed_zarr(a.copy(), zp, block_rows=1000)
    b = ad.AnnData(X=sparse_dataset(zarr.open(str(zp))["X"]))
    b.obs["grp"] = np.array((["a", "b"] * (b.n_obs // 2 + 1))[:b.n_obs])
    cases = {"filter_cells": lambda: msc_pp.filter_cells(b, min_counts=1),
             "regress_out": lambda: msc_pp.regress_out(b, "grp"),
             "normalize_pearson_residuals": lambda: msc_pp.normalize_pearson_residuals(b),
             "scrublet": lambda: msc_pp.scrublet(b),
             "rank_genes_groups": lambda: msc_tl.rank_genes_groups(b, "grp"),
             "score_genes": lambda: msc_tl.score_genes(b, ["MALAT1"])}
    for name, fn in cases.items():
        raised = False
        try:
            fn()
        except NotImplementedError:
            raised = True
        _row("5", f"backed_reject_{name}", "raised", int(raised), raised)
        assert raised, f"{name} should reject backed .X"


# ---------------- Step 6 — MINOR edge cases + experimental exclusion ----------------
def test_bbknn_tiny_batch_no_crash():
    from metasinglecell.neighbors import bbknn
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 10)).astype(np.float32)
    batch = np.array(["a"] * 40 + ["b"] * 2 + ["c"] * 18)   # batch 'b' (2) < k=3
    _, conn = bbknn(X, batch, neighbors_within_batch=3)
    ok = conn.shape == (60, 60) and conn.nnz > 0
    _row("6", "bbknn_tiny_batch", "conn_nnz", int(conn.nnz), ok)
    assert ok


def test_umap_coincident_points_no_nan():
    from metasinglecell.neighbors import neighbors as nb
    from metasinglecell.embedding import umap as msc_umap
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 10)).astype(np.float32)
    Xd = np.vstack([X, X[:5]]).astype(np.float32)           # 5 exact duplicates
    _, conn = nb(Xd, n_neighbors=10)
    emb = msc_umap(conn, n_epochs=50, random_state=0)
    has_nan = bool(np.isnan(emb).any())
    _row("6", "umap_coincident_no_nan", "any_nan", int(has_nan), not has_nan)
    assert not has_nan


def test_experimental_modules_not_importable():
    import importlib
    excluded = []
    for m in ("metasinglecell.graph.louvain_fused_raw", "metasinglecell.graph.louvain_hybrid"):
        try:
            importlib.import_module(m); excluded.append(False)
        except ModuleNotFoundError:
            excluded.append(True)
    ok = all(excluded)
    _row("6", "experimental_excluded_from_package", "all_excluded", int(ok), ok)
    assert ok, "experimental modules still importable as metasinglecell.graph.*"


def main():
    import csv, traceback
    from metasinglecell import config
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            n_fail += 1
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    out = config.REPO_ROOT / "results" / "code_review" / "postfix_validation.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "check", "metric", "value", "passed"])
        w.writeheader(); w.writerows(ROWS)
    npass = sum(r["passed"] for r in ROWS)
    print(f"\n{npass}/{len(ROWS)} checks passed; {n_fail} test fns failed -> {out}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
