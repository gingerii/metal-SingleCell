"""Out-of-core (zarr row-block streaming) front-end for datasets whose full ``.X``
does not fit in unified memory.

The store holds **raw counts only** (write-back of intermediates is out of scope this
milestone), so streaming ``normalize_total``/``log1p``/``scale`` cannot persist their
output. They are expressed as a :class:`BlockTransform` — an ordered prefix re-applied
to each raw-count block on each pass — which the terminal consumers (QC / HVG / PCA)
run on the fly. One :class:`ZarrRowReader` is opened once and reused by every function.

Design: single-GPU unified-memory (MLX/Metal), no Dask. The reader interface is kept
minimal so a Dask-backed accessor could later wrap it. See the ``out-of-core`` skill.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("metasinglecell.backed")

# Default budget for a densified per-block matrix (bytes). block_rows is auto-sized
# from the gene count H so an n_b×H fp32 block stays under this (PCA/scale densify).
_DENSE_BLOCK_BUDGET = 2 * 1024**3          # 2 GiB


def auto_block_rows(n_genes: int, requested: int, budget: int = _DENSE_BLOCK_BUDGET) -> int:
    """Largest block_rows ≤ ``requested`` whose dense n_b×n_genes fp32 block fits ``budget``."""
    cap = max(1, budget // (max(1, n_genes) * 4))
    return int(max(1, min(requested, cap)))


# --------------------------------------------------------------------------- reader
def _open_x_dataset(source):
    """Return (CSRDataset, (n_obs, n_vars)) for a zarr path / backed AnnData / CSRDataset."""
    import anndata.abc
    from anndata.io import sparse_dataset

    if isinstance(source, anndata.abc.CSRDataset):
        return source, tuple(source.shape)

    # A backed AnnData: use its already-open .X if it is a CSRDataset.
    if hasattr(source, "X") and isinstance(getattr(source, "X"), anndata.abc.CSRDataset):
        return source.X, tuple(source.X.shape)

    # Otherwise treat as a .zarr store path (or an open zarr group).
    import zarr

    grp = source if hasattr(source, "attrs") else zarr.open(str(source), mode="r")
    xgrp = grp["X"] if "X" in getattr(grp, "keys", lambda: [])() else grp
    dset = sparse_dataset(xgrp)          # CSR minor-axis row slicing, reads only block nnz
    return dset, tuple(dset.shape)


class ZarrRowReader:
    """Opened-once reader that yields cell-axis (row) blocks as GPU ``CSR`` objects."""

    def __init__(self, source, default_block_rows: int = 100_000):
        self._dset, self.shape = _open_x_dataset(source)
        self.n_obs, self.n_vars = self.shape
        self.default_block_rows = default_block_rows

    def iter_row_blocks(self, block_rows: int | None = None):
        """Yield ``(start, stop, CSR_block)``. ``block_rows`` defaults to the reader's."""
        from .sparse import CSR

        br = int(block_rows or self.default_block_rows)
        for start in range(0, self.n_obs, br):
            stop = min(start + br, self.n_obs)
            blk = self._dset[start:stop]                 # scipy CSR, reads only this block's nnz
            nnz_bytes = blk.data.nbytes + blk.indices.nbytes + blk.indptr.nbytes
            log.debug("block [%d:%d) rows=%d nnz=%d (%.2f GB sparse)",
                      start, stop, stop - start, blk.nnz, nnz_bytes / 1024**3)
            yield start, stop, CSR.from_scipy(blk)


def open_backed(source, default_block_rows: int = 100_000) -> ZarrRowReader:
    """Open a streaming reader over a ``.zarr`` path, backed AnnData, or CSRDataset."""
    return ZarrRowReader(source, default_block_rows=default_block_rows)


# ------------------------------------------------------------------ block transform
class BlockTransform:
    """An ordered transform prefix re-applied to each raw-count block on each pass.

    Stages (tuples): ``("normalize_total", ts)``, ``("log1p",)``,
    ``("hvg_subset", mask_bool)``, ``("scale", (mean, std, max_value, zero_center))``.
    Reuses the validated in-core ops: ``CSR.normalize_total``/``CSR.log1p`` and the
    exact z-score+clip of ``preprocess.scale``. ``apply`` returns a ``CSR`` while the
    chain is still sparse (through ``log1p``/``hvg_subset``) and a dense MLX array once
    ``scale`` is reached.
    """

    def __init__(self, stages=None):
        self.stages = list(stages or [])

    def then(self, *stage) -> "BlockTransform":
        return BlockTransform(self.stages + [tuple(stage)])

    def apply(self, csr_block):
        import mlx.core as mx

        from .sparse import CSR

        cur = csr_block                                   # CSR while sparse; dense MLX after scale
        for stage in self.stages:
            kind = stage[0]
            if kind == "normalize_total":
                cur = cur.normalize_total(float(stage[1]))
            elif kind == "log1p":
                cur = cur.log1p()
            elif kind == "hvg_subset":
                mask = stage[1]
                cur = CSR.from_scipy(cur.to_scipy()[:, mask])
            elif kind == "scale":
                mean, std, max_value, zero_center = stage[1]
                x = cur.toarray() if isinstance(cur, CSR) else np.asarray(cur)
                cur = _scale_apply_block(mx.array(x.astype(np.float32)),
                                         mean, std, max_value, zero_center)
            else:
                raise ValueError(f"unknown BlockTransform stage {kind!r}")
        return cur


def _scale_apply_block(x_mx, mean, std, max_value, zero_center):
    """Apply a PRE-COMPUTED per-gene z-score + clip to a dense block (MLX in/out).

    Mirrors ``preprocess.scale`` exactly but uses supplied ``mean``/``std`` (so it is a
    single per-block op, not a full-matrix pass). Lower clip only when ``zero_center``.
    """
    import mlx.core as mx

    mean_mx = mx.array(np.asarray(mean, dtype=np.float32))
    std_mx = mx.array(np.asarray(std, dtype=np.float32))
    x = x_mx - mean_mx if zero_center else x_mx
    x = x / std_mx
    if max_value is not None:
        upper = mx.minimum(x, mx.array(np.float32(max_value)))
        x = mx.maximum(upper, mx.array(np.float32(-max_value))) if zero_center else upper
    mx.eval(x)
    return x


# ---------------------------------------------------------------- streaming reducers
def stream_qc(reader: ZarrRowReader, block_rows: int | None = None) -> dict:
    """Streaming ``calculate_qc_metrics`` — matches ``preprocess.calculate_qc_metrics``.

    Per-cell metrics are exact (each cell's row lies wholly in one block); per-gene
    metrics accumulate additively across blocks (the column scatter-adds compose),
    finalized into the identical dict. Returns per-cell totals for reuse as the median
    ``target_sum`` of a subsequent streaming ``normalize_total``.
    """
    n, G = reader.n_obs, reader.n_vars
    cell_total = np.empty(n, dtype=np.float32)
    cell_ngenes = np.empty(n, dtype=np.int64)
    gene_total = np.zeros(G, dtype=np.float64)          # fp64 accumulate across blocks
    gene_ncells = np.zeros(G, dtype=np.int64)
    for s, e, csr in reader.iter_row_blocks(block_rows):
        rt, rn = csr.qc_metrics()
        cell_total[s:e] = np.asarray(rt)
        cell_ngenes[s:e] = np.asarray(rn).astype(np.int64)
        gt, gn = csr.gene_counts()
        gene_total += np.asarray(gt, dtype=np.float64)
        gene_ncells += np.asarray(gn, dtype=np.int64)
    return {
        "total_counts": cell_total,
        "n_genes_by_counts": cell_ngenes,
        "gene_total_counts": gene_total.astype(np.float32),
        "n_cells_by_counts": gene_ncells,
        "mean_counts": gene_total / n,
        "pct_dropout_by_counts": 100.0 * (1.0 - gene_ncells / n),
    }


def stream_gene_moments(reader: ZarrRowReader, transform: "BlockTransform",
                        flavor: str = "seurat", block_rows: int | None = None):
    """Per-gene ``(mean, var)`` for dispersion HVG, accumulated across blocks.

    ``seurat`` → moments of ``exp(x)−1`` of the log-normalized data (matches
    ``CSR.gene_moments``; MSL has no expm1 so the in-core kernel uses ``exp(x)−1``, which
    we replicate); ``cell_ranger`` → moments of the log-normalized values
    (``CSR.col_moments``). Column scatter-adds compose across blocks; finalize ddof=1 with
    implicit zeros folded via the full cell count.
    """
    import scipy.sparse as sp

    from .sparse import CSR

    G, n = reader.n_vars, reader.n_obs
    T = np.zeros(G, dtype=np.float64)
    Q = np.zeros(G, dtype=np.float64)
    for _, _, csr in reader.iter_row_blocks(block_rows):
        out = transform.apply(csr)
        b = out.to_scipy() if isinstance(out, CSR) else sp.csr_matrix(np.asarray(out))
        d = (np.exp(b.data) - 1.0) if flavor == "seurat" else b.data.astype(np.float64)
        cols = b.indices
        T += np.bincount(cols, weights=d, minlength=G)
        Q += np.bincount(cols, weights=d * d, minlength=G)
    mean = T / n
    var = (Q - n * mean**2) / (n - 1)
    return mean, var


def stream_scale_stats(reader: ZarrRowReader, transform: "BlockTransform",
                       block_rows: int | None = None):
    """Per-gene mean/std over the transformed (lognorm) blocks — pass 1 of streaming scale.

    Accumulates per-gene Σx, Σx² (fp64) via sparse column scatter-adds (no densification),
    then finalizes exactly like ``preprocess.scale``: ddof=1 variance including implicit
    zeros, ``std==0 → 1``. ``transform`` is the deferred prefix (normalize→log1p …), which
    keeps the block sparse so the moments are cheap.
    """
    import scipy.sparse as sp

    from .sparse import CSR

    G = reader.n_vars
    n = reader.n_obs
    colsum = np.zeros(G, dtype=np.float64)
    colsq = np.zeros(G, dtype=np.float64)
    for _, _, csr in reader.iter_row_blocks(block_rows):
        out = transform.apply(csr)
        b = out.to_scipy() if isinstance(out, CSR) else sp.csr_matrix(np.asarray(out))
        colsum += np.asarray(b.sum(0), dtype=np.float64).ravel()
        colsq += np.asarray(b.multiply(b).sum(0), dtype=np.float64).ravel()
    mean = colsum / n
    var = (colsq - n * mean**2) / (n - 1)              # ddof=1, implicit zeros folded in
    std = np.sqrt(np.maximum(var, 0.0))
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# ------------------------------------------------------------------------- prep util
def write_backed_zarr(adata, path, block_rows: int = 20_000):
    """Write an AnnData to a chunked-along-axis-0 zarr store (CSR ``.X``) for streaming.

    Round-trips a block to assert the store reads back cleanly before it is trusted.
    """
    import scipy.sparse as sp

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.X = sp.csr_matrix(adata.X).astype(np.float32)
    path = str(path)
    adata.write_zarr(path, chunks=(block_rows, adata.n_vars))

    rdr = open_backed(path)
    assert rdr.shape == adata.shape, (rdr.shape, adata.shape)
    s = min(128, adata.n_obs)
    got = rdr._dset[0:s]
    ref = sp.csr_matrix(adata.X)[0:s]
    assert (got != ref).nnz == 0, "zarr round-trip mismatch"
    log.info("wrote backed zarr %s  shape=%s  block_rows=%d (round-trip OK)",
             path, adata.shape, block_rows)
    return path


def convert_to_backed_zarr(in_path, out_path, block_rows: int = 20_000):
    """Convert an on-disk ``.h5ad`` / ``.h5`` (10x) counts matrix to a chunked backed zarr.

    The required one-time prep for any out-of-core run: reads the source with scanpy
    (``read_10x_h5`` for ``.h5``, ``read_h5ad`` otherwise), makes var names unique, and
    writes a cell-axis-chunked CSR zarr via :func:`write_backed_zarr` (which round-trip
    checks a block). Returns the output path. Reproducible from the shell:

        python -m metasinglecell.backed in.h5ad out.zarr --block-rows 100000
    """
    import scanpy as sc

    in_path, out_path = str(in_path), str(out_path)
    reader = sc.read_10x_h5 if in_path.endswith(".h5") else sc.read_h5ad
    adata = reader(in_path)
    adata.var_names_make_unique()
    log.info("read %s  shape=%s → writing %s", in_path, adata.shape, out_path)
    return write_backed_zarr(adata, out_path, block_rows=block_rows)


def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m metasinglecell.backed",
        description="Convert an .h5ad/.h5 counts matrix to a chunked backed zarr for streaming.")
    ap.add_argument("in_path", help="source .h5ad or 10x .h5")
    ap.add_argument("out_path", help="destination .zarr (chunked along the cell axis)")
    ap.add_argument("--block-rows", type=int, default=20_000,
                    help="cells per chunk (default 20000; larger = fewer, bigger chunks)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = convert_to_backed_zarr(args.in_path, args.out_path, block_rows=args.block_rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
