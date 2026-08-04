"""Regression tests for the UMAP layout SGD (github issue #1, the actual cause).

umap-learn walks an epoch's edges sequentially, so once a node is pushed its next gradient
is computed from the new position. We evaluate a whole epoch in parallel from one snapshot
of Y — that parallelism is the speedup, but the vendored optimizer applied every incident
edge's clipped gradient to the same stale position with no feedback. A node of degree 492
moved 477 units in a single epoch, which tore the layout into a pinhead surrounded by empty
panel. `_optimize_layout` keeps the parallel evaluation and bounds the accumulated per-node
step instead. These tests pin that bound and the umap-learn semantics around it.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from metalsinglecell import embedding


def _hub_graph(n=4000, hub=400, seed=0):
    """Connected graph with a few very high-degree nodes — the shape that blew up."""
    rng = np.random.RandomState(seed)
    rows, cols = [], []
    for h in range(3):                                    # hubs wired to many neighbours
        targets = rng.choice(n, hub, replace=False)
        rows += [h] * hub
        cols += list(targets)
    rows += list(range(n - 1))                            # a path so the graph is connected
    cols += list(range(1, n))
    rows, cols = np.array(rows), np.array(cols)
    data = rng.uniform(0.5, 1.0, rows.size)
    g = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return (g + g.T).tocsr()


def _spread(Y):
    core = np.percentile(Y, 99, axis=0) - np.percentile(Y, 1, axis=0)
    return core / (Y.max(axis=0) - Y.min(axis=0))


def test_epochs_per_sample_matches_umap_learn():
    """The strongest edge fires every epoch; an unsampled edge is marked -1."""
    w = np.array([1.0, 0.5, 0.1, 0.0])
    eps = embedding._make_epochs_per_sample(w, 200)
    assert eps[0] == pytest.approx(1.0)                   # max weight -> every epoch
    assert eps[1] == pytest.approx(2.0)                   # half weight -> every other epoch
    assert eps[2] == pytest.approx(10.0)
    assert eps[3] == -1.0                                 # zero weight -> never sampled

    # A denormal weight overflows to +inf, i.e. "never sampled" — correct, and it must not
    # warn (the schedule loop tests `<= epoch`, which is never true for inf or -1).
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiny = embedding._make_epochs_per_sample(np.array([1.0, 1e-320]), 200)
    assert tiny[0] == pytest.approx(1.0)
    assert not (tiny[1] <= 199)


def _one_epoch(cap, n=600, degree=300, seed=0):
    """Apply a single SGD epoch to a hub node wired to `degree` neighbours. Returns (Y0, Y).

    Driving ``_sgd_epoch`` directly rather than ``_optimize_layout(n_epochs=1)``: the
    umap-learn schedule starts ``epoch_of_next_sample`` at ``epochs_per_sample >= 1``, so no
    edge fires in epoch 0 and a one-epoch call moves nothing at all.
    """
    import mlx.core as mx
    rng = np.random.RandomState(seed)
    Y0 = rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float32)
    head = np.zeros(degree, dtype=np.int32)                  # every edge shares one head
    tail = np.arange(1, degree + 1, dtype=np.int32)
    ef, et = mx.array(head), mx.array(tail)
    Y = embedding._sgd_epoch(mx.array(Y0), ef, et, ef, et, mx.array(1.0), mx.array(1.577),
                             mx.array(0.895), mx.array(1.0), mx.array(float(cap)))
    return Y0, np.asarray(Y)


@pytest.mark.metal
def test_per_node_step_is_bounded():
    """The bug: one epoch moved a node hundreds of units. Nothing may move by > cap*alpha."""
    cap = 4.0
    Y0, Y = _one_epoch(cap)
    moved = np.linalg.norm(Y - Y0, axis=1)
    assert moved.max() > 0.0                                 # the epoch actually did something
    assert moved.max() <= cap + 1e-4, moved.max()            # alpha == 1.0 here


@pytest.mark.metal
def test_hub_graph_does_not_shear_apart():
    """End-to-end on the pathological shape: bounded coordinates, cells fill the panel."""
    graph = _hub_graph()
    Y = embedding.umap(graph, n_epochs=100, random_state=0, init="random")
    assert np.all(np.isfinite(Y))
    assert np.all(_spread(Y) > 0.3), _spread(Y)
    # the shipped optimizer reached ranges in the hundreds on graphs like this
    assert np.max(Y.max(0) - Y.min(0)) < 200.0


@pytest.mark.metal
def test_trust_region_is_doing_real_work():
    """Guard the guard: without the cap, the hub really does take an outsized step.

    Asserted on the step itself rather than on a final layout — a few thousand nodes do not
    reliably shear apart end-to-end (the reproduction needs ~10^5 cells), so an end-to-end
    assertion at test scale would pass for the wrong reason and quietly stop testing anything.
    """
    Y0, capped = _one_epoch(4.0)
    _, uncapped = _one_epoch(np.inf)
    near = np.linalg.norm(capped - Y0, axis=1).max()
    far = np.linalg.norm(uncapped - Y0, axis=1).max()
    assert near <= 4.0 + 1e-4
    assert far > 10.0 * near, (far, near)       # the cap bounds it, not the data


@pytest.mark.metal
def test_coincident_points_do_not_explode():
    """Every cell starting on the same point is the degenerate case that triggered it."""
    graph = _hub_graph(n=1000, hub=200)
    Y0 = np.zeros((1000, 2), dtype=np.float32)            # all coincident
    Y = embedding.umap(graph, n_epochs=50, random_state=0, init=Y0)
    assert np.all(np.isfinite(Y))
    assert np.max(Y.max(0) - Y.min(0)) < 200.0


@pytest.mark.metal
def test_negative_sample_on_self_is_skipped():
    """umap-learn `continue`s when the negative sample lands on the head itself."""
    import mlx.core as mx
    Y = mx.array(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    idx = mx.array(np.array([0], dtype=np.int32))
    out = embedding._sgd_epoch(Y, idx, idx, idx, idx, mx.array(1.0), mx.array(1.577),
                               mx.array(0.895), mx.array(1.0), mx.array(4.0))
    # head == tail for both the attractive and the negative term -> no movement at all
    assert np.allclose(np.asarray(out), np.asarray(Y))


@pytest.mark.metal
def test_same_seed_reproduces_structure_not_coordinates():
    """An atomic scatter-add is order-dependent, so repeat runs agree on structure only.

    Bit-equality is unavailable by construction: float addition is not associative and the
    GPU applies an epoch's gradients in whatever order it likes. Pin the property we do have
    (and that users actually rely on) rather than one we cannot deliver.
    """
    graph = _hub_graph(n=1500, hub=150)
    a = embedding.umap(graph, n_epochs=30, random_state=7)
    b = embedding.umap(graph, n_epochs=30, random_state=7)

    def knn(Y, k=15):
        d = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        return np.argpartition(d, k, axis=1)[:, :k]

    overlap = np.mean([len(set(x) & set(y)) / 15 for x, y in zip(knn(a), knn(b))])
    assert overlap > 0.75, overlap


@pytest.mark.metal
def test_seed_actually_changes_the_layout():
    """...and the seed is not being ignored: different seeds must differ more than repeats."""
    graph = _hub_graph(n=1500, hub=150)
    a = embedding.umap(graph, n_epochs=30, random_state=7)
    b = embedding.umap(graph, n_epochs=30, random_state=7)
    c = embedding.umap(graph, n_epochs=30, random_state=99)
    same_seed = np.abs(a - b).mean()
    diff_seed = np.abs(a - c).mean()
    assert diff_seed > same_seed, (same_seed, diff_seed)
