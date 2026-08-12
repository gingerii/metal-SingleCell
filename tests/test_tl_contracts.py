"""``msc.tl`` contracts against scanpy 1.11.5 (2026-08 API review).

Formulas, ignored parameters and output slots. The values these guard are ones people put in
figures — fold changes, cluster codes, densities — so most of them compare against scanpy
rather than against a hand-derived expectation.
"""

import importlib.util
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from metalsinglecell import pp as msc_pp
from metalsinglecell import tl as msc_tl

_HAS_SCANPY = importlib.util.find_spec("scanpy") is not None
needs_scanpy = pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy (oracle extra) not installed")

pytestmark = pytest.mark.metal


@pytest.fixture(scope="module")
def clustered():
    """Log-normalised data with planted group structure, plus a neighbour graph."""
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(0)
    n, g, k = 300, 80, 4
    grp = rng.integers(0, k, n)
    X = rng.poisson(2.0, (n, g)).astype(np.float32)
    for j in range(k):                                   # plant markers per group
        X[grp == j, j * 5:(j + 1) * 5] += rng.poisson(8.0, (int((grp == j).sum()), 5))
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["grp"] = pd.Categorical([str(v) for v in grp])
    import scipy.sparse as sp
    a.X = sp.csr_matrix(a.X)
    msc_pp.normalize_total(a, target_sum=1e4)
    msc_pp.log1p(a)
    msc_pp.pca(a, n_comps=20)
    msc_pp.neighbors(a, n_neighbors=15)
    return a


# --------------------------------------------------------------------------- rank_genes_groups


@needs_scanpy
def test_logfoldchanges_is_log2_of_the_expression_ratio(clustered):
    """Regression: we reported mean_g - mean_r, a difference of log-means. Correlated 0.17
    with scanpy's log2 expm1 ratio, ~48x smaller, and it gutted every fold-change filter."""
    import scanpy as sc
    o, t = clustered.copy(), clustered.copy()
    msc_tl.rank_genes_groups(o, "grp", method="t-test", use_raw=False)
    sc.tl.rank_genes_groups(t, "grp", method="t-test", use_raw=False)
    grp = str(clustered.obs["grp"].cat.categories[0])
    ours = dict(zip(o.uns["rank_genes_groups"]["names"][grp],
                    o.uns["rank_genes_groups"]["logfoldchanges"][grp]))
    theirs = dict(zip(t.uns["rank_genes_groups"]["names"][grp],
                      t.uns["rank_genes_groups"]["logfoldchanges"][grp]))
    keys = [k for k in theirs if np.isfinite(theirs[k])]
    assert np.allclose([ours[k] for k in keys], [theirs[k] for k in keys], atol=1e-3)


@needs_scanpy
def test_fold_change_filtering_matches_scanpy(clustered):
    """The consequence that matters: sc.get's own accessor must select the same genes."""
    import scanpy as sc
    o, t = clustered.copy(), clustered.copy()
    msc_tl.rank_genes_groups(o, "grp", use_raw=False)
    sc.tl.rank_genes_groups(t, "grp", use_raw=False)
    grp = str(clustered.obs["grp"].cat.categories[0])
    for cut in (0.5, 1.0, 2.0):
        n_o = len(sc.get.rank_genes_groups_df(o, group=grp, log2fc_min=cut))
        n_t = len(sc.get.rank_genes_groups_df(t, group=grp, log2fc_min=cut))
        assert n_o == n_t, f"log2fc_min={cut}: ours {n_o}, scanpy {n_t}"


@needs_scanpy
def test_reference_group_is_actually_used(clustered):
    """`reference` was accepted, recorded in params, and never read."""
    import scanpy as sc
    cats = [str(c) for c in clustered.obs["grp"].cat.categories]
    rest, ref = clustered.copy(), clustered.copy()
    msc_tl.rank_genes_groups(rest, "grp", reference="rest", use_raw=False)
    msc_tl.rank_genes_groups(ref, "grp", reference=cats[0], use_raw=False)

    # the reference group is dropped from the output, as scanpy does
    assert set(ref.uns["rank_genes_groups"]["scores"].dtype.names) == set(cats[1:])
    # and the comparison actually changed
    g = cats[1]
    assert not np.array_equal(rest.uns["rank_genes_groups"]["scores"][g],
                              ref.uns["rank_genes_groups"]["scores"][g])

    t = clustered.copy()
    sc.tl.rank_genes_groups(t, "grp", reference=cats[0], use_raw=False)
    ours = dict(zip(ref.uns["rank_genes_groups"]["names"][g],
                    ref.uns["rank_genes_groups"]["scores"][g]))
    theirs = dict(zip(t.uns["rank_genes_groups"]["names"][g],
                      t.uns["rank_genes_groups"]["scores"][g]))
    assert np.allclose([ours[k] for k in theirs], list(theirs.values()), atol=1e-3)


def test_unknown_reference_raises(clustered):
    a = clustered.copy()
    with pytest.raises(ValueError, match="needs to be one of"):
        msc_tl.rank_genes_groups(a, "grp", reference="NOT_A_GROUP", use_raw=False)


def test_singleton_group_raises(clustered):
    """A one-cell group has no variance; clamping the denominator produced finite noise that
    ranked as a top marker. scanpy refuses."""
    import pandas as pd
    a = clustered.copy()
    lab = a.obs["grp"].astype(str).to_numpy().copy()
    lab[0] = "solo"
    a.obs["grp2"] = pd.Categorical(lab)
    with pytest.raises(ValueError, match="one sample"):
        msc_tl.rank_genes_groups(a, "grp2", use_raw=False)


@needs_scanpy
def test_use_raw_defaults_to_raw_when_present(clustered):
    a = clustered.copy()
    a.raw = a                                            # .raw now holds the same 80 genes
    msc_tl.rank_genes_groups(a, "grp")
    assert a.uns["rank_genes_groups"]["params"]["use_raw"] is True
    b = clustered.copy()
    msc_tl.rank_genes_groups(b, "grp")                   # no .raw -> use .X
    assert b.uns["rank_genes_groups"]["params"]["use_raw"] is False


def test_params_carry_layer_and_corr_method(clustered):
    a = clustered.copy()
    msc_tl.rank_genes_groups(a, "grp", use_raw=False)
    p = a.uns["rank_genes_groups"]["params"]
    assert p["corr_method"] == "benjamini-hochberg"
    assert "layer" in p and "use_raw" in p


# --------------------------------------------------------------------------- clustering


def test_cluster_categories_are_natsorted(clustered):
    """Below 10 clusters lexicographic and natural order agree; at 10+ they do not, and then
    .cat.codes stops matching the integer label."""
    a = clustered.copy()
    msc_tl.leiden(a, resolution=8.0, key_added="fine")
    cats = list(a.obs["fine"].cat.categories)
    if len(cats) < 11:
        pytest.skip(f"resolution gave only {len(cats)} clusters; need >= 11 to discriminate")
    assert cats == sorted(cats, key=lambda s: int(s))
    assert cats != sorted(cats)                          # i.e. lexicographic would differ


def test_louvain_is_reproducible_at_a_fixed_seed(clustered):
    """random_state was read only on the gpu backend, so the default igraph path was unseeded
    and three identical calls gave three answers."""
    runs = []
    for _ in range(3):
        a = clustered.copy()
        msc_tl.louvain(a, random_state=0)
        runs.append(a.obs["louvain"].astype(str).to_numpy())
    assert (runs[0] == runs[1]).all() and (runs[1] == runs[2]).all()


def test_clustering_does_not_disturb_the_global_random_stream(clustered):
    """scanpy is careful not to reseed the stdlib generator; we were calling random.seed()."""
    random.seed(1234)
    expected = [random.random() for _ in range(3)]
    random.seed(1234)
    a = clustered.copy()
    msc_tl.leiden(a, random_state=0)
    msc_tl.louvain(a, random_state=0)
    assert [random.random() for _ in range(3)] == expected


def test_leiden_and_louvain_record_params(clustered):
    a = clustered.copy()
    msc_tl.leiden(a, random_state=3)
    msc_tl.louvain(a, random_state=3)
    assert a.uns["leiden"]["params"]["random_state"] == 3
    assert a.uns["louvain"]["params"]["resolution"] == 1.0


# --------------------------------------------------------------------------- embeddings


def test_draw_graph_layout_is_honoured_and_validated(clustered):
    """`layout` only named the output key; every value ran the same SGD."""
    a = clustered.copy()
    with pytest.raises(ValueError, match="valid layout"):
        msc_tl.draw_graph(a, layout="banana")

    msc_tl.draw_graph(a, layout="fa", n_iter=50)
    msc_tl.draw_graph(a, layout="kk")
    assert "X_draw_graph_fa" in a.obsm and "X_draw_graph_kk" in a.obsm
    assert not np.allclose(a.obsm["X_draw_graph_fa"], a.obsm["X_draw_graph_kk"])
    assert a.uns["draw_graph"]["params"]["layout"] == "kk"


def test_tsne_rejects_a_missing_representation(clustered):
    """It silently fell back to .X, so a typo embedded the wrong matrix."""
    a = clustered.copy()
    with pytest.raises(ValueError, match="X_nope"):
        msc_tl.tsne(a, use_rep="X_nope")


def test_umap_and_tsne_record_params(clustered):
    a = clustered.copy()
    msc_tl.umap(a, maxiter=10)
    assert set(a.uns["umap"]["params"]) >= {"a", "b"}
    msc_tl.tsne(a, use_rep="X_pca")
    assert a.uns["tsne"]["params"]["perplexity"] == 30.0


@needs_scanpy
def test_diffmap_matches_scanpy(clustered):
    """We skipped Coifman density normalisation and rescaled the eigenvectors, which moved the
    spectrum (max eigenvalue difference 0.056) and every component with it."""
    import scanpy as sc
    o, t = clustered.copy(), clustered.copy()
    msc_tl.diffmap(o, n_comps=10)
    sc.tl.diffmap(t, n_comps=10)
    assert np.allclose(o.uns["diffmap_evals"], t.uns["diffmap_evals"], atol=1e-4)
    for i in range(1, 5):                                # signs are arbitrary; compare |corr|
        c = abs(np.corrcoef(o.obsm["X_diffmap"][:, i], t.obsm["X_diffmap"][:, i])[0, 1])
        assert c > 0.99, f"DC{i} |corr| = {c:.3f}"


def test_diffmap_rejects_too_few_components(clustered):
    a = clustered.copy()
    with pytest.raises(ValueError, match="greater than 2"):
        msc_tl.diffmap(a, n_comps=2)


# --------------------------------------------------------------------------- density


@needs_scanpy
def test_embedding_density_uses_two_components_and_scanpys_keys(clustered):
    """On a wide basis the KDE over every column collapsed to a numerically-zero range that
    still plots as a valid density. And the obs key ignored groupby, so scanpy's plotter
    rejected our output outright."""
    import scanpy as sc
    a = clustered.copy()
    msc_tl.umap(a, maxiter=30)
    msc_tl.embedding_density(a, basis="umap", groupby="grp")
    assert "umap_density_grp" in a.obs
    assert a.uns["umap_density_grp_params"]["covariate"] == "grp"
    sc.pl.embedding_density(a, basis="umap", key="umap_density_grp", show=False)

    # a 20-D basis must give a real density, not a 1e-11 sliver
    msc_tl.embedding_density(a, basis="pca")
    d = a.obs["pca_density"].to_numpy()
    assert d.max() > 0.5 and np.ptp(d) > 0.5


def test_embedding_density_two_calls_do_not_collide(clustered):
    import pandas as pd
    a = clustered.copy()
    msc_tl.umap(a, maxiter=30)
    a.obs["other"] = pd.Categorical(["x", "y"] * (a.n_obs // 2))
    msc_tl.embedding_density(a, basis="umap", groupby="grp")
    msc_tl.embedding_density(a, basis="umap", groupby="other")
    assert "umap_density_grp" in a.obs and "umap_density_other" in a.obs


# --------------------------------------------------------------------------- neighbors_key


def test_graph_consumers_honour_neighbors_key(clustered):
    """With two graphs on one object there was no way to point the consumers at the second."""
    a = clustered.copy()
    msc_pp.neighbors(a, n_neighbors=5, key_added="small")
    assert "small_connectivities" in a.obsp
    for fn in (msc_tl.leiden, msc_tl.louvain):
        fn(a, neighbors_key="small", key_added=f"{fn.__name__}_small")
        assert f"{fn.__name__}_small" in a.obs
    msc_tl.diffmap(a, n_comps=5, neighbors_key="small")
    msc_tl.draw_graph(a, layout="fa", n_iter=20, neighbors_key="small")


def test_rank_genes_groups_df_accepts_our_output(clustered):
    """scanpy's own accessor is the real contract test for the uns layout."""
    sc = pytest.importorskip("scanpy")
    a = clustered.copy()
    for method in ("t-test", "wilcoxon"):
        msc_tl.rank_genes_groups(a, "grp", method=method, use_raw=False)
        df = sc.get.rank_genes_groups_df(a, group=None)
        assert {"group", "names", "scores", "pvals", "pvals_adj",
                "logfoldchanges"} <= set(df.columns)
