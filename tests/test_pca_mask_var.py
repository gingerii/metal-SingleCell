"""`pp.pca(mask_var=...)` parity with scanpy >= 1.10 (github issue #3).

scanpy deprecated `use_highly_variable` in 1.10 in favour of the more general `mask_var`.
The subtlety worth pinning: **omitting** mask_var selects `highly_variable` when that column
exists, while passing `mask_var=None` explicitly means ALL variables. Conflating the two is
silent — you get a different PCA, not an error.
"""

import numpy as np
import pytest

from metalsinglecell import pp as msc_pp


@pytest.fixture
def adata():
    import anndata as ad
    rng = np.random.RandomState(0)
    a = ad.AnnData(rng.normal(size=(200, 40)).astype(np.float32))
    hv = np.zeros(40, dtype=bool)
    hv[:12] = True
    a.var["highly_variable"] = hv
    a.var["my_genes"] = ~hv
    return a


@pytest.mark.metal
def test_omitted_defaults_to_highly_variable(adata):
    msc_pp.pca(adata, n_comps=5)
    assert adata.uns["pca"]["params"]["mask_var"] == "highly_variable"
    assert adata.uns["pca"]["params"]["use_highly_variable"] is True
    # loadings exist only on the selected variables
    assert np.any(adata.varm["PCs"][:12] != 0)
    assert np.all(adata.varm["PCs"][12:] == 0)


@pytest.mark.metal
def test_explicit_none_means_all_variables(adata):
    """The distinction that is easy to get wrong: None is NOT the same as omitting."""
    msc_pp.pca(adata, n_comps=5, mask_var=None)
    assert adata.uns["pca"]["params"]["mask_var"] is None
    assert adata.uns["pca"]["params"]["use_highly_variable"] is False
    assert np.any(adata.varm["PCs"][12:] != 0)          # non-HVG variables were used


@pytest.mark.metal
def test_omitted_with_no_highly_variable_column_uses_everything(adata):
    del adata.var["highly_variable"]
    msc_pp.pca(adata, n_comps=5)
    assert adata.uns["pca"]["params"]["mask_var"] is None
    assert np.any(adata.varm["PCs"][12:] != 0)


@pytest.mark.metal
def test_mask_var_by_column_name(adata):
    msc_pp.pca(adata, n_comps=5, mask_var="my_genes")
    assert adata.uns["pca"]["params"]["mask_var"] == "my_genes"
    assert np.all(adata.varm["PCs"][:12] == 0)          # my_genes is the complement of hv
    assert np.any(adata.varm["PCs"][12:] != 0)


@pytest.mark.metal
def test_mask_var_by_boolean_array(adata):
    m = np.zeros(40, dtype=bool)
    m[20:25] = True
    msc_pp.pca(adata, n_comps=3, mask_var=m)
    assert np.array_equal(adata.uns["pca"]["params"]["mask_var"], m)
    assert np.all(adata.varm["PCs"][:20] == 0)
    assert np.any(adata.varm["PCs"][20:25] != 0)


@pytest.mark.metal
def test_deprecated_use_highly_variable_still_works_but_warns(adata):
    with pytest.warns(FutureWarning, match="use_highly_variable` is deprecated"):
        msc_pp.pca(adata, n_comps=5, use_highly_variable=True)
    assert adata.uns["pca"]["params"]["mask_var"] == "highly_variable"

    with pytest.warns(FutureWarning):
        msc_pp.pca(adata, n_comps=5, use_highly_variable=False)
    assert adata.uns["pca"]["params"]["mask_var"] is None


def test_passing_both_raises(adata):
    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="incompatible"):
            msc_pp.pca(adata, n_comps=5, use_highly_variable=True, mask_var="my_genes")


def test_bad_mask_rejected(adata):
    with pytest.raises(ValueError, match="Did not find"):
        msc_pp.pca(adata, n_comps=5, mask_var="nope")
    with pytest.raises(ValueError, match="boolean array"):
        msc_pp.pca(adata, n_comps=5, mask_var=np.zeros(7, dtype=bool))
    with pytest.raises(ValueError, match="boolean array"):
        msc_pp.pca(adata, n_comps=5, mask_var=np.arange(40))


@pytest.mark.metal
@pytest.mark.parametrize("n_selected", [4, 8, 12, 14, 20])
def test_small_variable_counts_do_not_abort(n_selected):
    """Few selected variables used to kill the interpreter, not raise (found via issue #3).

    The randomized solver — our default — sketches `n_comps + n_oversamples` columns. When
    that exceeded the feature count, MLX's eigh threw a C++ exception nothing caught and the
    process died with SIGABRT. Any PCA over a small panel or a small HVG set hit it.
    """
    import anndata as ad
    rng = np.random.RandomState(0)
    a = ad.AnnData(rng.normal(size=(200, 40)).astype(np.float32))
    m = np.zeros(40, dtype=bool)
    m[:n_selected] = True
    msc_pp.pca(a, n_comps=5, mask_var=m)
    assert a.obsm["X_pca"].shape == (200, min(5, n_selected))
    assert np.all(np.isfinite(a.obsm["X_pca"]))


@pytest.mark.metal
def test_n_comps_above_rank_is_clamped():
    import anndata as ad
    rng = np.random.RandomState(0)
    a = ad.AnnData(rng.normal(size=(30, 10)).astype(np.float32))
    msc_pp.pca(a, n_comps=50, mask_var=None)
    assert a.obsm["X_pca"].shape[1] == 10


@pytest.mark.metal
def test_matches_scanpy_semantics(adata):
    """Same selection as scanpy for each of the four ways of specifying it."""
    import scanpy as sc

    for kwargs in ({}, {"mask_var": None}, {"mask_var": "my_genes"}):
        ours = adata.copy()
        theirs = adata.copy()
        msc_pp.pca(ours, n_comps=5, **kwargs)
        sc.pp.pca(theirs, n_comps=5, **kwargs)
        assert (ours.uns["pca"]["params"]["mask_var"]
                == theirs.uns["pca"]["params"]["mask_var"]), kwargs
        # the same variables carry loadings, whatever the sign/rotation of the components
        assert np.array_equal(np.any(ours.varm["PCs"] != 0, axis=1),
                              np.any(theirs.varm["PCs"] != 0, axis=1)), kwargs
