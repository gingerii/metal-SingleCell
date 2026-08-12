"""``msc.pp`` contracts against scanpy 1.11.5 (2026-08 API review).

Crashes, output slots and the scanpy call conventions — the things a value comparison on a
well-formed object never reaches.
"""

import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp

from metalsinglecell import pp as msc_pp

_HAS_SCANPY = importlib.util.find_spec("scanpy") is not None
needs_scanpy = pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy (oracle extra) not installed")


def counts(n=200, g=60, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(2, (n, g)).astype(np.float32)))
    a.var_names = [f"g{i}" for i in range(g)]
    return a


# --------------------------------------------------------------------------- crashes


@pytest.mark.metal
def test_regress_out_survives_a_singular_design():
    """A covariate that is identically zero makes DᵀD singular. mx.linalg.solve answers that
    with an uncatchable C++ abort (SIGABRT), so this has to run in a subprocess to be a test
    at all — an in-process assertion could not survive the failure it is checking for.

    Not exotic: pct_counts_mt is identically zero on any panel with no MT- genes.
    """
    script = textwrap.dedent("""
        import warnings; warnings.simplefilter("ignore")
        import numpy as np, anndata as ad, metalsinglecell as msc
        rng = np.random.default_rng(0)
        a = ad.AnnData(rng.poisson(3, (60, 30)).astype("float32"))
        a.obs["cov_ok"] = rng.normal(size=60)
        a.obs["cov_zero"] = np.zeros(60)
        msc.pp.regress_out(a, ["cov_ok", "cov_zero"])
        assert np.isfinite(a.X).all()
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, f"exit {r.returncode}: {r.stderr[-400:]}"
    assert "OK" in r.stdout


@pytest.mark.metal
def test_pearson_residuals_rejects_a_nonpositive_theta():
    a = counts()
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="theta"):
            msc_pp.normalize_pearson_residuals(a, theta=bad)


@pytest.mark.metal
def test_write_obsm_returns_the_path_that_exists(tmp_path):
    import os
    a = counts()
    a.obsm["X_emb"] = np.zeros((a.n_obs, 3), dtype=np.float32)
    got = msc_pp.write_obsm(a, "X_emb", tmp_path / "emb")     # np.save appends .npy
    assert os.path.exists(got)
    assert np.load(got).shape == (a.n_obs, 3)


# --------------------------------------------------------------------------- qc metrics


@pytest.mark.metal
def test_qc_metrics_pick_the_slot_by_axis_not_by_length():
    """Regression: the output slot was chosen by len(v), so on a SQUARE object every per-gene
    array also matched n_obs, landed in .obs, and the per-gene total overwrote the per-cell
    one. 2000 cells subset to 2000 HVGs is enough to hit it."""
    a = counts(n=80, g=80)
    msc_pp.calculate_qc_metrics(a, percent_top=None, inplace=True)
    assert {"total_counts", "n_genes_by_counts"} <= set(a.obs.columns)
    assert {"total_counts", "n_cells_by_counts", "mean_counts",
            "pct_dropout_by_counts"} <= set(a.var.columns)
    # the per-cell total must still be the per-cell total
    assert np.allclose(a.obs["total_counts"], np.asarray(a.X.sum(1)).ravel())
    assert np.allclose(a.var["total_counts"], np.asarray(a.X.sum(0)).ravel())


@pytest.mark.metal
def test_qc_metrics_defaults_to_returning_frames():
    """scanpy's default is inplace=False: return (obs_df, var_df), write nothing."""
    import pandas as pd
    a = counts()
    before_obs, before_var = set(a.obs.columns), set(a.var.columns)
    out = msc_pp.calculate_qc_metrics(a, percent_top=None)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(d, pd.DataFrame) for d in out)
    assert set(a.obs.columns) == before_obs and set(a.var.columns) == before_var
    assert list(out[0].index) == list(a.obs_names)
    assert list(out[1].index) == list(a.var_names)


@needs_scanpy
@pytest.mark.metal
def test_qc_metrics_column_set_matches_scanpy():
    import scanpy as sc
    a = counts(n=120, g=80)
    a.var["mt"] = np.r_[np.ones(5, bool), np.zeros(75, bool)]
    o, t = a.copy(), a.copy()
    msc_pp.calculate_qc_metrics(o, qc_vars=["mt"], percent_top=[10, 50], inplace=True)
    sc.pp.calculate_qc_metrics(t, qc_vars=["mt"], percent_top=[10, 50], inplace=True)
    assert set(o.obs.columns) == set(t.obs.columns)
    assert set(o.var.columns) == set(t.var.columns)
    for c in set(o.obs.columns) & set(t.obs.columns):
        if np.issubdtype(o.obs[c].dtype, np.number):
            assert np.allclose(o.obs[c], t.obs[c], rtol=1e-5, equal_nan=True), c


@needs_scanpy
@pytest.mark.metal
def test_qc_metrics_empty_cell_gives_nan_like_scanpy():
    import scanpy as sc
    a = counts(n=60, g=40)
    X = a.X.toarray(); X[0] = 0
    a.X = sp.csr_matrix(X)
    a.var["mt"] = np.r_[np.ones(4, bool), np.zeros(36, bool)]
    o, t = a.copy(), a.copy()
    msc_pp.calculate_qc_metrics(o, qc_vars=["mt"], percent_top=None, inplace=True)
    sc.pp.calculate_qc_metrics(t, qc_vars=["mt"], percent_top=None, inplace=True)
    assert np.isnan(o.obs["pct_counts_mt"].iloc[0]) == np.isnan(t.obs["pct_counts_mt"].iloc[0])


# --------------------------------------------------------------------------- regress_out


@needs_scanpy
@pytest.mark.metal
def test_regress_out_expands_a_categorical_covariate():
    """scanpy regresses a categorical on group indicators, so each group's residual mean is
    driven to zero. Coercing to float fits one ordinal slope instead — a different model that
    runs silently on numeric categories."""
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    rng = np.random.default_rng(0)
    grp = rng.integers(0, 3, 200)
    X = (rng.normal(size=(200, 40)) + grp[:, None] * 2.0).astype(np.float32)
    a = ad.AnnData(X)
    a.obs["batch"] = pd.Categorical(grp)
    o, t = a.copy(), a.copy()
    msc_pp.regress_out(o, ["batch"])
    sc.pp.regress_out(t, ["batch"])
    for g in range(3):
        m = (grp == g)
        assert abs(np.asarray(o.X)[m, 0].mean()) < 1e-4        # group means driven to zero
    assert np.allclose(np.asarray(o.X), np.asarray(t.X), atol=1e-3)


@pytest.mark.metal
def test_regress_out_accepts_string_categories():
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(1)
    a = ad.AnnData(rng.normal(size=(90, 20)).astype(np.float32))
    a.obs["batch"] = pd.Categorical(rng.choice(["a", "b", "c"], 90))
    msc_pp.regress_out(a, ["batch"])                            # used to raise on str -> float
    assert np.isfinite(a.X).all()


# --------------------------------------------------------------------------- output slots


@needs_scanpy
@pytest.mark.metal
@pytest.mark.parametrize("fn,kw,slot,cols", [
    ("filter_cells", {"min_genes": 3}, "obs", ["n_genes"]),
    ("filter_cells", {"min_counts": 5}, "obs", ["n_counts"]),
    ("filter_genes", {"min_cells": 3}, "var", ["n_cells"]),
    ("filter_genes", {"min_counts": 5}, "var", ["n_counts"]),
])
def test_filter_writes_the_count_column_scanpy_writes(fn, kw, slot, cols):
    a = counts()
    getattr(msc_pp, fn)(a, **kw)
    for c in cols:
        assert c in getattr(a, slot).columns, f"{fn} should write {slot}[{c!r}]"


@pytest.mark.metal
def test_scale_records_the_per_gene_mean_and_std():
    a = counts()
    msc_pp.scale(a)
    assert "mean" in a.var.columns and "std" in a.var.columns


@pytest.mark.metal
def test_pearson_residuals_records_its_parameters():
    a = counts()
    msc_pp.normalize_pearson_residuals(a, theta=50.0)
    rec = a.uns["pearson_residuals_normalization"]
    assert rec["theta"] == 50.0 and "clip" in rec and "computed_on" in rec
