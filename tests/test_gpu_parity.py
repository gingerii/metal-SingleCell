"""GPU parity — the accelerated functions vs their scanpy fp64 references.

Asserting versions of the `validation_notebooks/` parity logic (the notebooks compute the
same records but `print` and exit 0 regardless). scanpy is the fp64 reference; tolerances match
the documented parity bars (counts exact; fp32-derived rtol 1e-4; HVG overlap 1.000; PCA
subspace ≥0.98; clustering modularity ≥ igraph − ε). Requires the Metal GPU + PBMC3k data.
"""
import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = [pytest.mark.metal, pytest.mark.data]


def _lognorm(a):
    import scanpy as sc
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    return b


def test_qc_metrics_counts_exact(pbmc_counts):
    import scanpy as sc
    from metalsinglecell import pp as msc_pp, validation
    a = pbmc_counts.copy(); b = pbmc_counts.copy()
    msc_pp.calculate_qc_metrics(a)
    sc.pp.calculate_qc_metrics(b, inplace=True, percent_top=None, log1p=False)
    for col in ("total_counts", "n_genes_by_counts"):
        r = validation.compare(f"qc:{col}", a.obs[col].to_numpy(), b.obs[col].to_numpy(),
                               rtol=0, atol=0)
        assert r["passed"], (col, r)


def test_normalize_log1p_matches_scanpy(pbmc_counts):
    from metalsinglecell import pp as msc_pp, validation
    a = pbmc_counts.copy()
    msc_pp.normalize_total(a, target_sum=1e4); msc_pp.log1p(a)
    b = _lognorm(pbmc_counts)
    r = validation.compare("lognorm", np.asarray(a.X.todense()), np.asarray(b.X.todense()),
                           rtol=1e-4, atol=1e-4)
    assert r["passed"], r


def test_hvg_seurat_overlap_exact(pbmc_counts):
    import scanpy as sc
    from metalsinglecell import pp as msc_pp
    a = _lognorm(pbmc_counts); b = a.copy()
    msc_pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    sc.pp.highly_variable_genes(b, n_top_genes=2000, flavor="seurat")
    ov = (a.var["highly_variable"].to_numpy() & b.var["highly_variable"].to_numpy()).sum()
    assert ov == int(b.var["highly_variable"].sum()) == 2000


def test_scale_matches_scanpy(pbmc_counts):
    import scanpy as sc
    from metalsinglecell import pp as msc_pp, validation
    a = _lognorm(pbmc_counts); b = a.copy()
    msc_pp.scale(a, max_value=10.0)
    sc.pp.scale(b, max_value=10.0)
    r = validation.compare("scale", np.asarray(a.X), np.asarray(b.X), rtol=1e-4, atol=1e-3)
    assert r["passed"], r


@pytest.mark.parametrize("solver", ["full", "arpack", "covariance_eigh"])
def test_pca_dense_solvers_match_scanpy_subspace(pbmc_counts, solver):
    import scanpy as sc
    from metalsinglecell import pp as msc_pp, validation
    a = _lognorm(pbmc_counts)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(a, max_value=10.0)
    ref = a.copy()
    msc_pp.pca(a, n_comps=50, svd_solver=solver, use_highly_variable=False)
    sc.pp.pca(ref, n_comps=50, svd_solver="arpack")
    ov = validation.subspace_overlap(a.obsm["X_pca"], ref.obsm["X_pca"])
    assert ov >= 0.98, f"{solver} subspace {ov}"


def test_neighbors_knn_overlap(pbmc_counts):
    import scanpy as sc
    from metalsinglecell import pp as msc_pp
    a = _lognorm(pbmc_counts)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(a, max_value=10.0); sc.pp.pca(a, n_comps=50)
    msc_pp.neighbors(a, n_neighbors=15)
    # reference kNN via sklearn on the same PCA
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=15).fit(a.obsm["X_pca"])
    _, ref_idx = nn.kneighbors(a.obsm["X_pca"])
    dist = a.obsp["distances"]
    ours = [set(dist[i].indices) | {i} for i in range(a.n_obs)]
    recall = np.mean([len(ours[i] & set(ref_idx[i])) / 15 for i in range(a.n_obs)])
    assert recall >= 0.95, f"kNN recall {recall}"


def test_leiden_gpu_modularity_ge_igraph(pbmc_counts):
    import scanpy as sc
    from metalsinglecell import pp as msc_pp
    from metalsinglecell.cluster import leiden
    a = _lognorm(pbmc_counts)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(a, max_value=10.0); sc.pp.pca(a, n_comps=50)
    msc_pp.neighbors(a, n_neighbors=15)
    conn = a.obsp["connectivities"]
    lab_gpu = leiden(conn, resolution=1.0, backend="gpu", n_iterations=2)
    lab_ig = leiden(conn, resolution=1.0, backend="igraph", n_iterations=2)
    from metalsinglecell.graph import Graph
    from metalsinglecell.graph.primitives import modularity
    import mlx.core as mx
    g = Graph.from_scipy(conn)
    q_gpu = float(modularity(g, mx.array(lab_gpu.astype(np.int32)), 1.0))
    q_ig = float(modularity(g, mx.array(lab_ig.astype(np.int32)), 1.0))
    assert q_gpu >= q_ig - 0.02, f"GPU Q {q_gpu} vs igraph {q_ig}"
