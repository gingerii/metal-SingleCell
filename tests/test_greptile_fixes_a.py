"""Regression tests for the backed/streaming review findings (Greptile set A).

Each test reproduces one verified bug on a tiny synthetic zarr store (seconds, no data files):
the deferred-transform registry leaking across objects, ``materialize`` wiping its own source,
backed PCA silently centring under ``zero_center=False``, the streaming solver crashing when
``n_comps`` exceeds the selected genes, ``normalize_total``'s median target vs scanpy 1.11.5,
backed ``normalize_total(inplace=False)`` mutating state, and the README streaming example
loading ``.X`` into memory.
"""
import gc
import importlib.util
import os
import re
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from metalsinglecell import pp as msc_pp

pytestmark = pytest.mark.metal

_HAS_SCANPY = importlib.util.find_spec("scanpy") is not None
needs_scanpy = pytest.mark.skipif(not _HAS_SCANPY, reason="scanpy (oracle extra) not installed")

_ZERO_MSG = "Some cells have zero counts"


def _store(path, X, block_rows=4, obs_names=None, var_names=None):
    """Write ``X`` as a chunked backed zarr at ``path`` and return the path string."""
    import anndata as ad
    from metalsinglecell.backed import write_backed_zarr
    a = ad.AnnData(sp.csr_matrix(np.asarray(X, dtype=np.float32)))
    if obs_names is not None:
        a.obs_names = obs_names
    if var_names is not None:
        a.var_names = var_names
    write_backed_zarr(a, str(path), block_rows=block_rows)
    return str(path)


def _backed(path):
    import anndata as ad
    import zarr
    from anndata.io import sparse_dataset
    return ad.AnnData(X=sparse_dataset(zarr.open(str(path), mode="r")["X"]))


def _poisson(n, g, lam=3.0, seed=0):
    return np.random.default_rng(seed).poisson(lam, (n, g)).astype(np.float32)


def _mostly_empty(seed=0):
    """12 cells x 6 genes with 7 empty rows: the all-cell median total is 0."""
    X = _poisson(12, 6, seed=seed)
    X[:7] = 0
    return X


def _snapshot(path):
    """Every file under a store with its bytes — to prove a store was left untouched."""
    out = {}
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, path)] = fh.read()
    return out


# ---------------------------------------------------------------- 1. registry id reuse


def test_stream_registry_entry_dies_with_its_object(tmp_path):
    p = _store(tmp_path / "s.zarr", _poisson(20, 8))
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    msc_pp.log1p(a)
    key = id(a)
    assert key in msc_pp._STREAM_TRANSFORMS
    del a
    gc.collect()
    assert key not in msc_pp._STREAM_TRANSFORMS, "deferred stages outlived their AnnData"


def test_fresh_backed_object_never_inherits_stages(tmp_path):
    """CPython reuses a dead object's id; a new object must not pick up its stages."""
    p = _store(tmp_path / "s.zarr", _poisson(20, 8))
    inherited = 0
    for _ in range(100):
        a = _backed(p)                          # reassigning frees the previous object
        if msc_pp._stream_stages(a):
            inherited += 1
        msc_pp.normalize_total(a, target_sum=1e4)
        msc_pp.log1p(a)
    del a
    gc.collect()
    assert inherited == 0, f"{inherited}/100 fresh objects started with another's stages"


# ------------------------------------------------------- 2. materialize onto its source


@pytest.mark.parametrize("where", ["same", "parent", "inside"])
def test_materialize_refuses_to_overwrite_its_source(tmp_path, where):
    src_dir = tmp_path / "stores"
    src_dir.mkdir()
    p = _store(src_dir / "src.zarr", _poisson(20, 8, lam=2.0))
    before = _snapshot(p)
    dest = {"same": p, "parent": str(src_dir), "inside": os.path.join(p, "ckpt.zarr")}[where]
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    msc_pp.log1p(a)
    with pytest.raises(ValueError, match="source"):
        msc_pp.materialize(a, dest)
    assert _snapshot(p) == before, "the source store was modified"
    assert not os.path.exists(os.path.join(p, "ckpt.zarr"))
    # the raw counts still read back
    assert _backed(p).X[:].nnz == sp.csr_matrix(_poisson(20, 8, lam=2.0)).nnz


def test_materialize_to_a_new_path_still_works(tmp_path):
    p = _store(tmp_path / "src.zarr", _poisson(20, 8))
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    msc_pp.log1p(a)
    msc_pp.materialize(a, tmp_path / "ckpt.zarr")
    assert a.X[:].nnz == sp.csr_matrix(_poisson(20, 8)).nnz
    assert msc_pp._stream_stages(a) == []


# ------------------------------------------------- 3. backed pca(zero_center=False)


def test_backed_pca_uncentred_raises_before_doing_anything(tmp_path):
    p = _store(tmp_path / "s.zarr", _poisson(60, 12), block_rows=16)
    a = _backed(p)
    msc_pp.normalize_total(a, target_sum=1e4)
    stages = list(msc_pp._stream_stages(a))
    with pytest.raises(NotImplementedError, match="zero_center"):
        msc_pp.pca(a, n_comps=4, zero_center=False, mask_var=None)
    assert "X_pca" not in a.obsm and "PCs" not in a.varm and "pca" not in a.uns
    assert msc_pp._stream_stages(a) == stages


# ------------------------------------------- 4. streaming n_comps > selected genes


@pytest.mark.parametrize("n_obs,n_vars,n_hvg,n_comps", [
    (60, 10, None, 50),       # explicit n_comps above the panel width
    (60, 30, 10, 20),         # explicit n_comps above an HVG mask
    (8, 30, None, 20),        # fewer cells than genes: clamp to n_obs
])
def test_backed_pca_clamps_n_comps_like_incore(tmp_path, n_obs, n_vars, n_hvg, n_comps):
    import anndata as ad
    X = _poisson(n_obs, n_vars)
    mask = None
    if n_hvg is not None:
        mask = np.zeros(n_vars, bool)
        mask[:n_hvg] = True
    p = _store(tmp_path / "s.zarr", X, block_rows=16)

    incore = ad.AnnData(sp.csr_matrix(X))
    b = _backed(p)
    kw = {"mask_var": mask}
    with warnings.catch_warnings(record=True) as w_in:
        warnings.simplefilter("always")
        msc_pp.pca(incore, n_comps=n_comps, **kw)
    with warnings.catch_warnings(record=True) as w_bk:
        warnings.simplefilter("always")
        msc_pp.pca(b, n_comps=n_comps, **kw)

    assert b.obsm["X_pca"].shape == incore.obsm["X_pca"].shape
    assert b.varm["PCs"].shape == incore.varm["PCs"].shape
    assert b.uns["pca"]["variance_ratio"].shape == incore.uns["pca"]["variance_ratio"].shape
    assert np.isfinite(b.obsm["X_pca"]).all()
    # same warnings policy as in-core (which clamps silently)
    def user_warnings(ws):
        return sorted({str(x.message) for x in ws if issubclass(x.category, UserWarning)})
    assert user_warnings(w_bk) == user_warnings(w_in)


# ---------------------------------------------- 5. normalize_total median vs scanpy


@needs_scanpy
@pytest.mark.parametrize("kind", ["dense", "csr", "csc"])
def test_normalize_total_median_target_matches_scanpy(kind):
    import anndata as ad
    import scanpy as sc
    X = _mostly_empty()
    make = {"dense": lambda: X.copy(), "csr": lambda: sp.csr_matrix(X),
            "csc": lambda: sp.csc_matrix(X)}[kind]
    ref = ad.AnnData(make())
    with pytest.warns(UserWarning, match=_ZERO_MSG):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # scanpy divides by a 0 target
            sc.pp.normalize_total(ref)
    got = ad.AnnData(make())
    with pytest.warns(UserWarning, match=_ZERO_MSG):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            msc_pp.normalize_total(got)

    def dense(M):
        return M.toarray() if sp.issparse(M) else np.asarray(M)
    np.testing.assert_allclose(dense(got.X), dense(ref.X), rtol=1e-6, atol=1e-6)


@needs_scanpy
def test_normalize_total_without_empty_cells_does_not_warn():
    import anndata as ad
    X = _poisson(12, 6) + 1
    for M in (X.copy(), sp.csr_matrix(X)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msc_pp.normalize_total(ad.AnnData(M))
        assert not [x for x in w if _ZERO_MSG in str(x.message)]


@needs_scanpy
@pytest.mark.parametrize("prior_qc", [False, True])
def test_streaming_normalize_total_matches_scanpy_csr(tmp_path, prior_qc):
    """The store is CSR, so the streaming target follows scanpy's CSR path (all-cell median)."""
    import anndata as ad
    import scanpy as sc
    X = _mostly_empty()
    ref = ad.AnnData(sp.csr_matrix(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf = sc.pp.normalize_total(ref, inplace=False)["norm_factor"]
    scanpy_target = float(np.median(X.sum(1)))              # what scanpy's CSR branch uses
    assert np.isinf(nf[-1]) and scanpy_target == 0.0         # confirms the oracle's behaviour

    p = _store(tmp_path / "s.zarr", X)
    b = _backed(p)
    if prior_qc:
        msc_pp.calculate_qc_metrics(b)
    with pytest.warns(UserWarning, match=_ZERO_MSG):
        msc_pp.normalize_total(b)
    assert msc_pp._stream_stages(b) == [("normalize_total", scanpy_target)]


# ---------------------------------- 6. backed normalize_total(inplace=False) / key_added


def test_backed_normalize_total_inplace_false_raises_without_recording(tmp_path):
    p = _store(tmp_path / "s.zarr", _poisson(12, 6))
    b = _backed(p)
    with pytest.raises(NotImplementedError, match="inplace"):
        msc_pp.normalize_total(b, target_sum=1e4, inplace=False)
    assert msc_pp._stream_stages(b) == []


@needs_scanpy
@pytest.mark.parametrize("target_sum", [None, 1e4])
def test_backed_normalize_total_writes_key_added_like_scanpy(tmp_path, target_sum):
    import anndata as ad
    import scanpy as sc
    X = _poisson(12, 6) + 1
    ref = ad.AnnData(sp.csr_matrix(X))
    sc.pp.normalize_total(ref, target_sum=target_sum, key_added="nf")
    b = _backed(_store(tmp_path / "s.zarr", X))
    msc_pp.normalize_total(b, target_sum=target_sum, key_added="nf")
    np.testing.assert_allclose(b.obs["nf"].to_numpy(), ref.obs["nf"].to_numpy(), rtol=1e-6)


# --------------------------------------------------------- 7. README streaming example


def _readme_streaming_snippet():
    from metalsinglecell import config
    text = (config.REPO_ROOT / "README.md").read_text()
    section = text.split("## Out-of-core", 1)[1]
    m = re.search(r"```python\n(.*?)```", section, re.S)
    assert m, "no python block in the README's out-of-core section"
    return m.group(1)


def test_readme_streaming_example_stays_backed(tmp_path):
    rng = np.random.default_rng(0)
    # varied gene means so the default seurat HVG cutoffs select a usable set
    lam = rng.gamma(0.6, 2.0, size=80)
    X = rng.poisson(lam[None, :] * rng.uniform(0.5, 1.5, (300, 1))).astype(np.float32)
    obs_names = [f"cell{i}" for i in range(300)]
    var_names = [f"gene{j}" for j in range(80)]
    p = _store(tmp_path / "atlas.zarr", X, block_rows=64, obs_names=obs_names,
               var_names=var_names)
    code = _readme_streaming_snippet().replace('"atlas.zarr"', repr(p))
    ns, err = {}, None
    try:
        exec(compile(code, "README.md:out-of-core", "exec"), ns)   # noqa: S102 - our README
    except Exception as e:                       # report the in-memory load first, if that is why
        err = e
    adata = ns.get("adata")
    assert adata is not None and msc_pp._backed_reader(adata) is not None, (
        f"the README example loaded .X into memory instead of streaming it (then: {err!r})")
    if err is not None:
        raise err
    assert list(adata.obs_names) == obs_names and list(adata.var_names) == var_names
    assert adata.obsm["X_pca"].shape[0] == 300
