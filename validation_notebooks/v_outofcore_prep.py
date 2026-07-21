"""Prep + reader validation for the out-of-core streaming front-end.

Builds chunked-along-axis-0 zarr stores from source data and checks that a
:class:`ZarrRowReader` reconstructs ``.X`` exactly. Sources:
  * ``pbmc``   — data/pbmc3k_raw.h5ad (fast smoke test)
  * ``atlas``  — data/external/1M_neurons.h5, optionally subsampled via ``--cells``

    conda activate metasinglecell
    python validation_notebooks/v_outofcore_prep.py --source pbmc
    python validation_notebooks/v_outofcore_prep.py --source atlas --cells 200000
    python validation_notebooks/v_outofcore_prep.py --source atlas            # full 1.3M

The store path is data/processed/backed/<name>.zarr (a derived input, gitignored).
"""
import argparse
import logging
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp

from metasinglecell import config
from metasinglecell.backed import open_backed, write_backed_zarr


def _find(rel):
    d = os.getcwd()
    for _ in range(6):
        if os.path.exists(os.path.join(d, rel)):
            return os.path.join(d, rel)
        d = os.path.dirname(d)
    raise FileNotFoundError(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pbmc", "atlas"], default="pbmc")
    ap.add_argument("--cells", type=int, default=0, help="subsample N cells (0 = all)")
    ap.add_argument("--block-rows", type=int, default=20_000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import scanpy as sc

    if args.source == "pbmc":
        adata = sc.read_h5ad(_find("data/pbmc3k_raw.h5ad"))
        name = "pbmc"
    else:
        adata = sc.read_10x_h5(_find("data/external/1M_neurons.h5"))
        adata.var_names_make_unique()
        name = "atlas"
    if args.cells and args.cells < adata.n_obs:
        adata = adata[: args.cells].copy()
        name += f"_{args.cells}"
    adata.X = sp.csr_matrix(adata.X).astype(np.float32)
    print(f"source={args.source} shape={adata.shape} nnz={adata.X.nnz:,}")

    out_dir = config.PROCESSED_DIR / "backed"
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"{name}.zarr"
    write_backed_zarr(adata, zpath, block_rows=args.block_rows)

    # Reader reconstruction check (skip full densify for the big atlas — sample blocks).
    rdr = open_backed(zpath)
    if adata.n_obs <= 300_000:
        recon = sp.vstack([csr.to_scipy() for _, _, csr in rdr.iter_row_blocks(50_000)]).tocsr()
        ok = (recon != sp.csr_matrix(adata.X)).nnz == 0
        print(f"reader reconstructs .X exactly: {ok}")
        assert ok
    else:
        # sample-check a few blocks against the in-memory slice
        Xm = sp.csr_matrix(adata.X)
        for s, e, csr in rdr.iter_row_blocks(100_000):
            assert (csr.to_scipy() != Xm[s:e]).nnz == 0
            if s > 300_000:
                break
        print("reader block-sample matches in-memory slice: True")
    print(f"OK: {zpath}")


if __name__ == "__main__":
    main()
