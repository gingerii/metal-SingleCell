"""Full-column parity for ``gr.spatial_autocorr`` (github issue #6).

Through 0.1.2 we returned only the statistic and a ``pval_sim`` that was one-sided toward
clustering, where squidpy returns the normality-assumption statistics, the permutation
statistics, and an FDR column per p-value — and defines ``pval_sim`` as a folded two-tailed
count. The one-sided version reported p ~ 1 for a gene with significant *negative* spatial
autocorrelation, which is the case these tests are built around.
"""

import importlib.util

import numpy as np
import pytest

from metalsinglecell import gr as msc_gr

_HAS_SQUIDPY = importlib.util.find_spec("squidpy") is not None
needs_squidpy = pytest.mark.skipif(not _HAS_SQUIDPY, reason="squidpy (oracle extra) not installed")

ANALYTIC = ["pval_norm", "var_norm"]
PERM = ["pval_z_sim", "pval_sim", "var_sim"]


def lattice_with_signals(side=25, jitter=0.0, seed=0):
    """A lattice carrying a smooth gene, a checkerboard gene and a noise gene.

    The checkerboard is the point: its Moran's I is negative, and a one-sided permutation
    count toward clustering calls that non-significant.
    """
    import anndata as ad
    r, c = np.divmod(np.arange(side * side), side)
    xy = np.c_[c * 100.0 + (r % 2) * 50.0, r * 86.6]
    if jitter:
        xy = xy + np.random.default_rng(seed).normal(0, jitter, xy.shape)
    rng = np.random.default_rng(seed)
    X = np.c_[(r + c).astype(np.float32),            # smooth   -> I > 0
              ((r + c) % 2).astype(np.float32),      # checker  -> I < 0
              rng.normal(size=side * side)]          # noise    -> I ~ 0
    a = ad.AnnData(np.ascontiguousarray(X, dtype=np.float32))
    a.var_names = ["smooth", "checker", "noise"]
    a.obsm["spatial"] = xy
    return a


def irregular(side=22, seed=1):
    """Delaunay on a jittered lattice: degrees vary, so row-normalisation actually bites."""
    a = lattice_with_signals(side, jitter=8.0, seed=seed)
    msc_gr.spatial_neighbors_delaunay(a)
    return a


def regular(side=25):
    a = lattice_with_signals(side)
    msc_gr.spatial_neighbors_knn(a, n_neighs=6)
    return a


# --------------------------------------------------------------------------- the reported bug


@pytest.mark.metal
def test_negative_autocorrelation_is_significant():
    """The regression. `checker` has I < 0; a one-sided count toward clustering gives p ~ 1."""
    a = regular()
    msc_gr.spatial_autocorr(a, mode="moran", n_perms=999, seed=0)
    df = a.uns["moranI"]
    assert df.loc["checker", "I"] < 0
    assert df.loc["checker", "pval_sim"] == pytest.approx(1 / 1000, abs=1e-9)
    assert df.loc["smooth", "pval_sim"] == pytest.approx(1 / 1000, abs=1e-9)
    assert df.loc["noise", "pval_sim"] > 0.05


@pytest.mark.metal
@pytest.mark.parametrize("mode,stat", [("moran", "I"), ("geary", "C")])
def test_columns_and_sort_order(mode, stat):
    a = regular()
    msc_gr.spatial_autocorr(a, mode=mode)                     # no perms: analytic only
    df = a.uns["moranI" if mode == "moran" else "gearyC"]
    assert df.columns.tolist() == [stat, *ANALYTIC, "pval_norm_fdr_bh"]

    msc_gr.spatial_autocorr(a, mode=mode, n_perms=99)
    df = a.uns["moranI" if mode == "moran" else "gearyC"]
    assert df.columns.tolist() == [stat, *ANALYTIC, *PERM,
                                   "pval_norm_fdr_bh", "pval_z_sim_fdr_bh", "pval_sim_fdr_bh"]
    # Moran sorts descending (high = clustered), Geary ascending (low = clustered)
    order = df[stat].to_numpy()
    assert np.all(np.diff(order) <= 0) if mode == "moran" else np.all(np.diff(order) >= 0)


@pytest.mark.metal
def test_n_perms_none_omits_the_permutation_columns():
    a = regular()
    msc_gr.spatial_autocorr(a, mode="moran")
    assert not [c for c in a.uns["moranI"].columns if "sim" in c]


@pytest.mark.metal
def test_corr_method_none_omits_the_fdr_columns():
    a = regular()
    msc_gr.spatial_autocorr(a, mode="moran", n_perms=99, corr_method=None)
    assert not [c for c in a.uns["moranI"].columns if "fdr" in c]


@pytest.mark.metal
def test_copy_returns_the_frame_not_the_object():
    import pandas as pd
    a = regular()
    out = msc_gr.spatial_autocorr(a, mode="moran", copy=True)
    assert isinstance(out, pd.DataFrame)
    assert "moranI" not in a.uns


@pytest.mark.metal
def test_two_tailed_doubles_the_analytic_pvalue():
    a = regular()
    one = msc_gr.spatial_autocorr(a, mode="moran", copy=True)
    two = msc_gr.spatial_autocorr(a, mode="moran", two_tailed=True, copy=True)
    assert np.allclose(two["pval_norm"], 2.0 * one["pval_norm"])


@pytest.mark.metal
def test_transformation_changes_the_statistic_only_when_degrees_vary():
    """Row-normalisation is a global rescale on a regular graph, so Moran's I — which divides
    by S0 — cannot move. On an irregular graph it must."""
    a = regular()
    same = [msc_gr.spatial_autocorr(a, mode="moran", transformation=t, copy=True)["I"]
            for t in (True, False)]
    assert np.allclose(same[0], same[1], atol=1e-6)

    b = irregular()
    diff = [msc_gr.spatial_autocorr(b, mode="moran", transformation=t, copy=True)["I"]
            for t in (True, False)]
    assert not np.allclose(diff[0], diff[1], atol=1e-6)


@pytest.mark.metal
def test_attr_obs_scores_observation_columns():
    a = regular()
    a.obs["smooth_obs"] = np.asarray(a.X[:, 0]).ravel()
    df = msc_gr.spatial_autocorr(a, mode="moran", attr="obs", genes=["smooth_obs"], copy=True)
    assert df.index.tolist() == ["smooth_obs"]
    ref = msc_gr.spatial_autocorr(a, mode="moran", genes=["smooth"], copy=True)
    assert df.loc["smooth_obs", "I"] == pytest.approx(ref.loc["smooth", "I"], abs=1e-6)


# --------------------------------------------------------------------------- FDR correction


def test_benjamini_hochberg_matches_statsmodels():
    """We implement BH rather than depend on statsmodels, so pin it against the reference.
    Runs on the CPU lane — no MLX involved."""
    statsmodels = pytest.importorskip("statsmodels.stats.multitest")
    from metalsinglecell.spatial import multiple_testing
    rng = np.random.default_rng(0)
    for p in [rng.uniform(size=200),
              rng.uniform(size=50) ** 4,                    # many tiny p-values
              np.repeat(0.03, 20),                          # exact ties
              np.array([1.0, 1.0, 1e-12]),                  # saturation at 1
              np.array([0.5])]:
        for method in ("fdr_bh", "bonferroni"):
            ref = statsmodels.multipletests(p, alpha=0.05, method=method)[1]
            assert np.allclose(multiple_testing(p, method), ref, rtol=1e-12, atol=0), method


def test_unknown_corr_method_raises():
    from metalsinglecell.spatial import multiple_testing
    with pytest.raises(NotImplementedError, match="corr_method"):
        multiple_testing(np.array([0.1, 0.2]), "holm")


# --------------------------------------------------------------------------- squidpy parity


def _squidpy_fp64(a, **kw):
    """squidpy on an fp64 copy of the graph.

    squidpy row-normalises the connectivity in place, in whatever dtype it is stored in. Our
    builders store fp32, so squidpy's weights — and the moments derived from them — carry fp32
    rounding, while we normalise in fp64. Handing it an fp64 graph removes that difference and
    lets the analytic path be compared exactly; the size of the difference is pinned separately
    in ``test_fp32_normalisation_gap_is_negligible``.
    """
    import squidpy as sq
    b = a.copy()
    b.obsp["spatial_connectivities"] = a.obsp["spatial_connectivities"].astype(np.float64)
    return sq.gr.spatial_autocorr(b, copy=True, n_jobs=1, show_progress_bar=False, **kw)


@needs_squidpy
@pytest.mark.metal
@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("transformation", [True, False])
def test_analytic_columns_match_squidpy(mode, transformation):
    """The analytic path is deterministic, so it has to match element-wise. Run on an
    irregular graph, where row-normalisation and the moments both matter."""
    a = irregular()
    o = msc_gr.spatial_autocorr(a, mode=mode, transformation=transformation, copy=True)
    t = _squidpy_fp64(a, mode=mode, transformation=transformation)
    assert o.columns.tolist() == t.columns.tolist()
    t = t.loc[o.index]
    stat = "I" if mode == "moran" else "C"
    assert np.allclose(o[stat], t[stat], atol=1e-5)          # ours is fp32 on the GPU
    assert np.allclose(o["var_norm"], t["var_norm"], rtol=1e-12, atol=0)
    assert np.allclose(o["pval_norm"], t["pval_norm"], atol=1e-6)
    assert np.allclose(o["pval_norm_fdr_bh"], t["pval_norm_fdr_bh"], atol=1e-6)


@needs_squidpy
@pytest.mark.metal
def test_fp32_normalisation_gap_is_negligible():
    """Against squidpy on the fp32 graph as a user would actually hit it, the analytic
    variance differs only by squidpy's fp32 row-normalisation — ~1e-6 relative."""
    import squidpy as sq
    a = irregular()
    ours = msc_gr.spatial_autocorr(a, mode="geary", copy=True)["var_norm"].iloc[0]
    theirs = sq.gr.spatial_autocorr(a, mode="geary", copy=True, n_jobs=1,
                                    show_progress_bar=False)["var_norm"].iloc[0]
    assert ours == pytest.approx(theirs, rel=1e-5)
    assert ours == pytest.approx(_squidpy_fp64(a, mode="geary")["var_norm"].iloc[0], rel=1e-12)


@needs_squidpy
@pytest.mark.metal
def test_geary_uses_the_geary_variance_not_morans():
    """squidpy reused Moran's normality variance for Geary until 1.8.2 (scverse/squidpy#1183).
    Pin that we use the corrected one. The two formulas are close on this graph — 7.052e-4
    against 6.920e-4, about 1.9% — so the guard has to be tight to mean anything; matching
    squidpy 1.8.2 to 1e-12 is what actually establishes which formula we use."""
    a = irregular()
    ours = msc_gr.spatial_autocorr(a, mode="geary", copy=True)["var_norm"].iloc[0]
    theirs = _squidpy_fp64(a, mode="geary")["var_norm"].iloc[0]
    moran = msc_gr.spatial_autocorr(a, mode="moran", copy=True)["var_norm"].iloc[0]
    assert ours == pytest.approx(theirs, rel=1e-12)
    assert not np.isclose(ours, moran, rtol=1e-3)


@needs_squidpy
@pytest.mark.metal
def test_permutation_columns_agree_with_squidpy():
    """Different RNG streams, so these agree to Monte-Carlo error, not exactly. var_sim is the
    tight one — it is a property of the null, not of the draw."""
    import squidpy as sq
    a = irregular()
    o = msc_gr.spatial_autocorr(a, mode="moran", n_perms=500, seed=0, copy=True)
    t = sq.gr.spatial_autocorr(a, mode="moran", n_perms=500, seed=0, copy=True,
                               n_jobs=1, show_progress_bar=False).loc[o.index]
    assert np.allclose(o["var_sim"], t["var_sim"], rtol=0.25)
    assert np.allclose(o["pval_sim"], t["pval_sim"], atol=0.05)
    assert np.allclose(o["pval_z_sim"], t["pval_z_sim"], atol=0.05)
