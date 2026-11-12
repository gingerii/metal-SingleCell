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
