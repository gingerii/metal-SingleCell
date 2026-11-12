"""Regression tests for the Greptile review findings on embeddings and tools (batch D).

Each test was run against the unfixed code first and failed there (the three-group logreg
test is the exception: it guards behaviour that must NOT change).
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


# ---------------- 1. UMAP, n_components != 2, disconnected graph ----------------

def _two_blocks(m=120, seed=0):
    """Two disconnected, internally connected blocks of ``m`` nodes each."""
    rng = np.random.RandomState(seed)
    blocks = []
    for _ in range(2):
        a = sp.random(m, m, density=0.12, random_state=rng, format="csr",
                      data_rvs=lambda k: rng.uniform(0.2, 1.0, k))
        blocks.append(a + a.T + sp.eye(m) * 0.5)
    return sp.block_diag(blocks, format="csr")


@pytest.mark.metal
@pytest.mark.parametrize("dim", [1, 3, 4])
def test_umap_init_disconnected_graph_any_dim(dim):
    from metalsinglecell import embedding
    graph = _two_blocks()
    with pytest.warns(UserWarning, match="connected components"):
        Y = embedding._initial_embedding(graph, dim, random_state=0)
    assert Y.shape == (240, dim)
    assert np.all(np.isfinite(Y))
    a, b = Y[:120], Y[120:]
    if dim == 1:     # laid end to end on the line, not on top of each other
        assert a.max() < b.min() or b.max() < a.min()
    else:            # discs packed in the first two dims; the other dims still carry the layout
        assert np.linalg.norm(a[:, :2].mean(0) - b[:, :2].mean(0)) > 10.0
        assert np.all(np.ptp(Y[:, 2:], axis=0) > 0)


@pytest.mark.metal
def test_tl_umap_three_components_on_disconnected_graph():
    import anndata as ad
    import metalsinglecell as msc
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (150, 10)),
                   rng.normal(100, 1, (150, 10))]).astype(np.float32)
    a = ad.AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(300)]))
    a.obsm["X_pca"] = X
    msc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
    with pytest.warns(UserWarning, match="connected components"):
        msc.tl.umap(a, n_components=3, maxiter=20)
    assert a.obsm["X_umap"].shape == (300, 3)
    assert np.all(np.isfinite(a.obsm["X_umap"]))


# ---------------- 2. t-SNE affinities: squared distances, not fourth powers ----------------

@pytest.mark.metal
def test_tsne_affinities_match_sklearn(monkeypatch):
    """P from the vendored t-SNE equals sklearn's on the same kNN graph."""
    from sklearn.manifold._t_sne import _joint_probabilities_nn
    from metalsinglecell._vendor.mlx_vis import tsne as vt
    from metalsinglecell._vendor.mlx_vis.knn import compute_knn

    n, perp = 300, 10.0
    k = int(3 * perp)
    X = np.random.default_rng(0).normal(size=(n, 10)).astype(np.float32)

    got = {}

    def capture(self, edge_from, edge_to, edge_weights, Y, n_, epoch_callback=None):
        got["P"] = sp.csr_matrix((np.array(edge_weights),
                                  (np.array(edge_from), np.array(edge_to))), shape=(n_, n_))
        return Y                                      # skip the optimisation itself

    monkeypatch.setattr(vt.TSNE, "_optimize", capture)
    vt.TSNE(perplexity=perp, pca_dim=None, knn_method="brute",
            random_state=0).fit_transform(X)

    # sklearn's affinities on the same neighbour sets, from fp64 squared Euclidean distances
    idx, _ = compute_knn(X, k, method="brute")
    X64 = X.astype(np.float64)
    d2 = np.sum((X64[:, None, :] - X64[idx]) ** 2, axis=-1)
    D = sp.csr_matrix((d2.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    P_ref = _joint_probabilities_nn(D, perp, 0).toarray()
    P = got["P"].toarray()
    assert P.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.abs(P - P_ref).sum() < 1e-3             # the d^4 kernel misplaced ~7-18% of the mass
    assert np.abs(P - P_ref).max() < 1e-2 * P_ref.max()


# ---------------- 3. tl.tsne forwards early_exaggeration ----------------

@pytest.mark.metal
def test_tl_tsne_forwards_early_exaggeration(monkeypatch):
    import anndata as ad
    import metalsinglecell as msc
    from metalsinglecell._vendor.mlx_vis import tsne as vt

    seen = []

    def fake_fit(self, X, epoch_callback=None):
        seen.append(self.early_exaggeration)
        return np.zeros((len(X), self.n_components), np.float32)

    monkeypatch.setattr(vt.TSNE, "fit_transform", fake_fit)
    X = np.random.default_rng(0).normal(size=(60, 8)).astype(np.float32)
    for ee in (1.0, 50.0):
        a = ad.AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]))
        a.obsm["X_pca"] = X
        msc.tl.tsne(a, use_rep="X_pca", early_exaggeration=ee)
        assert a.uns["tsne"]["params"]["early_exaggeration"] == ee
    assert seen == [1.0, 50.0]


# ---------------- 4. Two-group logreg: one correctly signed score row per group ----------------

def _planted_markers(groups):
    import anndata as ad
    rng = np.random.default_rng(0)
    n, g = len(groups), 6
    X = rng.normal(0, 1, (n, g)).astype(np.float32)
    for j, name in enumerate(sorted(set(groups))):
        X[groups == name, j] += 3.0                   # gene j marks group j
    a = ad.AnnData(X, obs=pd.DataFrame({"g": pd.Categorical(groups)},
                                       index=[f"c{i}" for i in range(n)]))
    a.var_names = [f"g{i}" for i in range(g)]
    return a


def test_logreg_two_groups_each_group_gets_its_own_signed_scores():
    import metalsinglecell as msc
    a = _planted_markers(np.array(["A"] * 100 + ["B"] * 100))
    msc.tl.rank_genes_groups(a, "g", method="logreg", use_raw=False)
    r = a.uns["rank_genes_groups"]
    assert r["names"].dtype.names == ("A", "B")
    assert r["names"]["A"][0] == "g0"                 # A's planted marker
    assert r["names"]["B"][0] == "g1"                 # B's planted marker
    score_a = dict(zip(r["names"]["A"], r["scores"]["A"]))
    score_b = dict(zip(r["names"]["B"], r["scores"]["B"]))
    for gene, s in score_a.items():
        assert score_b[gene] == pytest.approx(-s)


def test_logreg_three_groups_unchanged_vs_scanpy():
    sc = pytest.importorskip("scanpy")
    import metalsinglecell as msc
    a = _planted_markers(np.array(["A"] * 70 + ["B"] * 70 + ["C"] * 60))
    ref = a.copy()
    sc.tl.rank_genes_groups(ref, "g", method="logreg")
    msc.tl.rank_genes_groups(a, "g", method="logreg", use_raw=False)
    for c in "ABC":
        assert list(a.uns["rank_genes_groups"]["names"][c]) == \
            list(ref.uns["rank_genes_groups"]["names"][c])
        np.testing.assert_allclose(a.uns["rank_genes_groups"]["scores"][c],
                                   ref.uns["rank_genes_groups"]["scores"][c], rtol=1e-5)
    assert a.uns["rank_genes_groups"]["names"]["C"][0] == "g2"


# ---------------- 5. score_genes follows scanpy's use_raw / layer ----------------

S_GENES = ["G3", "G70", "G75", "G80", "G85", "G90"]
G2M_GENES = ["G5", "G72", "G77", "G82", "G88", "G95"]
SET = ["G5", "G70", "G90", "G100", "G110"]          # G70 and above live only in .raw


def _raw_lognorm_x_scaled(seed=0, center=True):
    """``.raw`` holds log-normalised values for 120 genes; ``.X`` is scaled over the first 60.

    With ``center=True`` (z-scaling, the tutorial object's state) every gene's mean in ``.X`` is
    ~0, so the expression bins that pick control genes are set by rounding noise; comparing
    scores computed *from* ``.X`` then needs ``center=False``.
    """
    import anndata as ad
    rng = np.random.default_rng(seed)
    n, g = 300, 120
    counts = rng.poisson(rng.gamma(2.0, 1.0, size=g) + 0.2, size=(n, g)).astype(np.float32)
    lognorm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    a = ad.AnnData(lognorm, obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    a.layers["lognorm"] = lognorm.copy()
    a.raw = a
    a = a[:, :60].copy()
    Xs = a.X - a.X.mean(0) if center else a.X
    a.X = (Xs / (a.X.std(0) + 1e-8)).astype(np.float32)
    return a


@pytest.mark.metal
def test_score_genes_default_uses_raw_like_scanpy():
    sc = pytest.importorskip("scanpy")
    import metalsinglecell as msc
    a = _raw_lognorm_x_scaled()
    ref = a.copy()
    sc.tl.score_genes(ref, SET)
    msc.tl.score_genes(a, SET)
    np.testing.assert_allclose(a.obs["score"], ref.obs["score"], rtol=1e-4, atol=1e-5)


@pytest.mark.metal
def test_score_genes_use_raw_false_and_layer_like_scanpy():
    sc = pytest.importorskip("scanpy")
    import metalsinglecell as msc
    a = _raw_lognorm_x_scaled(center=False)

    ref = a.copy()
    sc.tl.score_genes(ref, SET, use_raw=False)
    with pytest.warns(UserWarning, match="not in var_names"):
        msc.tl.score_genes(a, SET, use_raw=False)
    np.testing.assert_allclose(a.obs["score"], ref.obs["score"], rtol=1e-4, atol=1e-5)

    ref = a.copy()
    sc.tl.score_genes(ref, ["G5", "G10", "G20"], layer="lognorm")
    msc.tl.score_genes(a, ["G5", "G10", "G20"], layer="lognorm")
    np.testing.assert_allclose(a.obs["score"], ref.obs["score"], rtol=1e-4, atol=1e-5)

    with pytest.raises(ValueError, match="No valid genes"), \
            pytest.warns(UserWarning, match="not in var_names"):
        msc.tl.score_genes(a, ["NOPE1", "NOPE2"])


@pytest.mark.metal
def test_score_genes_cell_cycle_default_uses_raw_like_scanpy():
    sc = pytest.importorskip("scanpy")
    import metalsinglecell as msc
    a = _raw_lognorm_x_scaled()
    ref = a.copy()
    sc.tl.score_genes_cell_cycle(ref, s_genes=S_GENES, g2m_genes=G2M_GENES)
    msc.tl.score_genes_cell_cycle(a, S_GENES, G2M_GENES)
    for key in ("S_score", "G2M_score"):
        np.testing.assert_allclose(a.obs[key], ref.obs[key], rtol=1e-4, atol=1e-5)
    assert list(a.obs["phase"].astype(str)) == list(ref.obs["phase"].astype(str))


# ---------------- 6. k-means: duplicate rows, empty clusters ----------------

@pytest.mark.metal
def test_kmeans_identical_rows_do_not_crash():
    from metalsinglecell import tools
    with pytest.warns(UserWarning, match="distinct clusters"):
        lab = tools.kmeans(np.ones((50, 4), np.float32), n_clusters=2)
    assert len(np.unique(lab)) == 1


@pytest.mark.metal
def test_kmeans_more_clusters_than_distinct_rows():
    from metalsinglecell import tools
    X = np.repeat(np.eye(3, dtype=np.float32), 20, axis=0)
    with pytest.warns(UserWarning, match="distinct clusters"):
        lab = tools.kmeans(X, n_clusters=5)
    assert len(np.unique(lab)) == 3
    for r in range(3):                                # each distinct row is one cluster
        assert len(np.unique(lab[r * 20:(r + 1) * 20])) == 1
    assert len({lab[0], lab[20], lab[40]}) == 3


class _ForcedSeeds:
    """Stands in for the k-means++ generator so the seeds are chosen by the test."""

    def __init__(self, seeds):
        self.seeds = list(seeds)

    def integers(self, *args, **kwargs):
        return self.seeds.pop(0)

    def choice(self, *args, **kwargs):
        return self.seeds.pop(0)


@pytest.mark.metal
def test_kmeans_empty_cluster_keeps_its_centroid(monkeypatch):
    """An empty cluster must not jump to the origin.

    Seeds 0 and 4 are the same point, so cluster 2 is empty after the first assignment. A
    zero centroid made the result depend on where the origin is: k-means is translation
    invariant, so shifting the data must not change the labels.
    """
    from metalsinglecell import tools
    X = np.array([[1, -2], [-1, -7], [5, 3], [-1, 2], [1, -2], [3, -1], [-1, -2]], np.float32)
    labels = []
    for shift in (0.0, 50.0):
        monkeypatch.setattr(tools.np.random, "default_rng",
                            lambda *a, **k: _ForcedSeeds([0, 1, 4]))
        labels.append(tools.kmeans(X + shift, n_clusters=3))
    np.testing.assert_array_equal(labels[0], labels[1])
    assert len(np.unique(labels[1])) == 3


@pytest.mark.metal
def test_calculate_niche_few_distinct_compositions_default_n_niches():
    """Two cell types and a 6-NN graph allow only 7 compositions, fewer than the 10 niches."""
    import anndata as ad
    import metalsinglecell as msc
    rng = np.random.default_rng(0)
    xy = np.stack(np.meshgrid(np.arange(40), np.arange(40)), -1).reshape(-1, 2).astype(float)
    xy += rng.normal(0, 0.05, xy.shape)
    ct = np.where(xy[:, 0] < 20, "epi", "stroma")
    flip = rng.random(len(ct)) < 0.2
    ct[flip] = rng.choice(["epi", "stroma"], flip.sum())
    a = ad.AnnData(np.zeros((len(ct), 1), np.float32),
                   obs=pd.DataFrame({"ct": pd.Categorical(ct)},
                                    index=[f"c{i}" for i in range(len(ct))]))
    a.obsm["spatial"] = xy
    msc.gr.spatial_neighbors_knn(a, n_neighs=6)
    with pytest.warns(UserWarning, match="distinct clusters"):
        msc.gr.calculate_niche(a, "ct")
    assert 1 < a.obs["niche"].nunique() <= 7
