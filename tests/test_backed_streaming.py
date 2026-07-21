"""Out-of-core streaming parity — the largest surface the parity notebooks never asserted.

On a small backed zarr store, the streaming front-end must be **bit-exact** vs in-core for the
linear ops (QC / normalize / log1p) and match to **subspace ≥ 0.999** for covariance-eigh PCA.
Promotes the M1/M2 validation logic (`results/zarr_outofcore/v_outofcore*.py`) into asserting
tests. Requires the Metal GPU + PBMC3k data.
"""
import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = [pytest.mark.metal, pytest.mark.data]


@pytest.fixture(scope="module")
def backed_pbmc(pbmc_counts, tmp_path_factory):
    """(in-core twin, make_backed factory, zarr path).

    A backed AnnData can't be ``.copy()``- d, so each test builds a FRESH one from the store
    via ``make_backed()`` (mirrors how the v_outofcore drivers re-open the reader per call)."""
    import anndata as ad
    import zarr
    from anndata.io import sparse_dataset
    from metasinglecell.backed import write_backed_zarr
    a = pbmc_counts.copy()
    keep = np.sort(np.argsort(-np.asarray(a.X.sum(0)).ravel())[:2000])   # top-2000-gene panel
    a = a[:, keep].copy()
    zp = tmp_path_factory.mktemp("backed") / "pbmc_panel.zarr"
    write_backed_zarr(a.copy(), zp, block_rows=500)

    def make_backed():
        b = ad.AnnData(X=sparse_dataset(zarr.open(str(zp))["X"]))
        b.var_names = a.var_names
        return b
    return a, make_backed, str(zp)


def _incore_lognorm(a):
    from metasinglecell.sparse import CSR
    ts = float(np.median(np.asarray(sp.csr_matrix(a.X).sum(1)).ravel()))
    return CSR.from_scipy(sp.csr_matrix(a.X)).normalize_total(ts).log1p()


def test_streaming_qc_bit_exact(backed_pbmc):
    from metasinglecell import pp as msc_pp, validation
    a, make_backed, _ = backed_pbmc
    incore = a.copy(); msc_pp.calculate_qc_metrics(incore)
    b = make_backed(); msc_pp.calculate_qc_metrics(b)
    for col in ("total_counts", "n_genes_by_counts"):
        r = validation.compare(f"stream_qc:{col}", b.obs[col].to_numpy(),
                               incore.obs[col].to_numpy(), rtol=0, atol=0)
        assert r["passed"] and r["exact_match"], (col, r)


def test_streaming_normalize_log1p_bit_exact(backed_pbmc):
    from metasinglecell import pp as msc_pp, validation
    from metasinglecell.backed import open_backed
    a, make_backed, zp = backed_pbmc
    b = make_backed()
    msc_pp.calculate_qc_metrics(b); msc_pp.normalize_total(b); msc_pp.log1p(b)
    tf = msc_pp._build_transform(b)
    rdr = open_backed(zp, default_block_rows=500)
    stream = sp.vstack([tf.apply(csr).to_scipy() for _, _, csr in rdr.iter_row_blocks()]).tocsr()
    incore = _incore_lognorm(a).to_scipy()
    r = validation.compare("stream_lognorm", stream.toarray(), incore.toarray(), rtol=0, atol=0)
    assert r["passed"] and r["exact_match"], r


def test_streaming_pca_subspace_matches_incore(backed_pbmc):
    from metasinglecell import pp as msc_pp, preprocess as _pp, validation
    from metasinglecell.decomposition import pca as _pca
    from metasinglecell.sparse import CSR
    a, make_backed, _ = backed_pbmc
    b = make_backed()
    msc_pp.calculate_qc_metrics(b); msc_pp.normalize_total(b); msc_pp.log1p(b)
    msc_pp.highly_variable_genes(b, n_top_genes=1000, flavor="seurat")
    msc_pp.scale(b, max_value=10.0)
    msc_pp.pca(b, n_comps=50, use_highly_variable=True)
    # in-core reference: lognorm → hvg → subset → scale → covariance_eigh
    ln = _incore_lognorm(a)
    mask = _pp.highly_variable_genes(ln, n_top_genes=1000, flavor="seurat")["highly_variable"].to_numpy()
    scaled = _pp.scale(CSR.from_scipy(ln.to_scipy()[:, mask]), max_value=10.0)
    xp_i, _, _ = _pca(scaled, n_comps=50, solver="covariance_eigh")
    ov = validation.subspace_overlap(b.obsm["X_pca"], xp_i)
    assert ov >= 0.999, f"streaming PCA subspace {ov}"


def test_materialize_roundtrip_and_guard(backed_pbmc, tmp_path):
    """materialize checkpoints post-log1p bit-exact; guards the log1p boundary."""
    from metasinglecell import pp as msc_pp, validation
    from metasinglecell.backed import open_backed
    a, make_backed, _ = backed_pbmc
    b = make_backed()
    msc_pp.calculate_qc_metrics(b); msc_pp.normalize_total(b); msc_pp.log1p(b)
    ckpt = tmp_path / "ckpt.zarr"
    msc_pp.materialize(b, ckpt)
    stream = sp.vstack([csr.to_scipy() for _, _, csr in open_backed(str(ckpt)).iter_row_blocks()]).tocsr()
    incore = _incore_lognorm(a).to_scipy()
    r = validation.compare("materialize", stream.toarray(), incore.toarray(), rtol=0, atol=0)
    assert r["passed"] and r["exact_match"], r
    # guard: materialize past the log1p boundary (scale recorded) must raise
    c = make_backed()
    msc_pp.calculate_qc_metrics(c); msc_pp.normalize_total(c); msc_pp.log1p(c); msc_pp.scale(c)
    with pytest.raises(ValueError):
        msc_pp.materialize(c, tmp_path / "nope.zarr")
