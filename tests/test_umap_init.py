"""Regression tests for the UMAP initialization (github issue #1).

A k-NN graph over a real atlas is routinely disconnected. The old initialization ran a
plain power iteration over the whole graph, whose top eigenspace is then spanned by the
component indicator vectors — so every cell in a component landed on the same point, and
the optimizer sheared that degenerate start apart into a pinhead of cells surrounded by
empty panel. These tests pin the component-aware layout that replaced it.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from metalsinglecell import embedding


def _disconnected_graph(sizes, seed=0):
    """Block-diagonal symmetric connectivity graph: one dense-ish block per component."""
    rng = np.random.RandomState(seed)
    blocks = []
    for m in sizes:
        a = sp.random(m, m, density=min(1.0, 15.0 / max(m, 1)), random_state=rng,
                      format="csr", data_rvs=lambda k: rng.uniform(0.2, 1.0, k))
        a = a + a.T + sp.eye(m) * 0.5          # keep every block internally connected
        blocks.append(a)
    return sp.block_diag(blocks, format="csr")


def _spread(Y):
    """Fraction of the bounding box the 1st-99th percentile core occupies (per axis)."""
    core = np.percentile(Y, 99, axis=0) - np.percentile(Y, 1, axis=0)
    return core / (Y.max(axis=0) - Y.min(axis=0))


def test_disconnected_graph_init_is_not_collapsed():
    """The failure mode: one big component plus rare tight ones."""
    graph = _disconnected_graph([4000] + [40] * 20)
    with pytest.warns(UserWarning, match="connected components"):
        Y = embedding._initial_embedding(graph, 2, random_state=0)

    assert Y.shape == (graph.shape[0], 2)
    assert np.all(np.isfinite(Y))
    assert not embedding._is_degenerate(Y)
    # The old init put >97% of cells on a handful of points; the core spanned ~2% of the box.
    assert np.all(_spread(Y) > 0.3), _spread(Y)
    # Every component must occupy its own region, not one shared point.
    assert len(np.unique(np.round(Y, 2), axis=0)) > 0.9 * graph.shape[0]


def test_components_are_separated_and_sized_by_cell_count():
    graph = _disconnected_graph([2000, 2000, 30])
    with pytest.warns(UserWarning, match="connected components"):
        Y = embedding._initial_embedding(graph, 2, random_state=0)
    big_a, big_b, small = Y[:2000], Y[2000:4000], Y[4000:]

    # distinct regions
    for u, v in [(big_a, big_b), (big_a, small), (big_b, small)]:
        assert np.linalg.norm(u.mean(0) - v.mean(0)) > 1.0
    # a 30-cell component must not claim as much canvas as a 2000-cell one
    extent = lambda B: np.max(B.max(0) - B.min(0))
    assert extent(small) < extent(big_a)


def test_connected_graph_still_uses_a_plain_spectral_layout():
    graph = _disconnected_graph([3000])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # no component / degeneracy warning
        Y = embedding._initial_embedding(graph, 2, random_state=0)
    assert not embedding._is_degenerate(Y)
    assert np.all(_spread(Y) > 0.3)


def test_pack_discs_do_not_overlap():
    rng = np.random.RandomState(0)
    radii = np.concatenate([[10.0], rng.uniform(0.4, 6.0, 40)])
    centers = embedding._pack_discs(radii)
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    need = radii[:, None] + radii[None, :]
    np.fill_diagonal(d, np.inf)
    assert np.all(d >= need), (d - need).min()


def test_degeneracy_detector():
    n = 1000
    rng = np.random.RandomState(0)
    assert not embedding._is_degenerate(rng.uniform(-10, 10, (n, 2)))
    # piecewise-constant, as the old init produced on a disconnected graph
    collapsed = np.repeat([[0.0, 0.0], [10.0, 10.0]], [n - 2, 2], axis=0)
    assert embedding._is_degenerate(collapsed)
    assert embedding._is_degenerate(np.full((n, 2), np.nan))


def test_init_options():
    graph = _disconnected_graph([500])
    n = graph.shape[0]
    assert embedding._initial_embedding(graph, 2, 0, init="random").shape == (n, 2)

    given = np.zeros((n, 2), dtype=np.float32)
    assert np.array_equal(embedding._initial_embedding(graph, 2, 0, init=given), given)

    with pytest.raises(ValueError, match="expected"):
        embedding._initial_embedding(graph, 2, 0, init=np.zeros((n, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="init must be"):
        embedding._initial_embedding(graph, 2, 0, init="pca")


def test_weak_edges_are_pruned():
    """Edges scheduled for <1 sample are dropped, as umap-learn does."""
    g = sp.csr_matrix(np.array([[0.0, 1.0, 1e-9], [1.0, 0.0, 0.0], [1e-9, 0.0, 0.0]]))
    pruned = embedding._prune_weak_edges(g, n_epochs=200)
    assert pruned.nnz == 2
    assert pruned.data.min() == pytest.approx(1.0)


def test_umap_end_to_end_on_a_disconnected_graph():
    graph = _disconnected_graph([3000] + [40] * 10)
    with pytest.warns(UserWarning, match="connected components"):
        Y = embedding.umap(graph, n_epochs=50, random_state=0)
    assert Y.shape == (graph.shape[0], 2)
    assert np.all(np.isfinite(Y))
    # The bug's signature: a huge coordinate range driven by a few flung-out cells.
    assert np.all(_spread(Y) > 0.2), _spread(Y)
