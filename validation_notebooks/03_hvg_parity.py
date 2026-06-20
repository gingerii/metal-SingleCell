"""Stage 1 validation: Metal HVG (per-gene moments + seurat binning) vs oracle.

Builds a CSR from the log-normalized snapshot (``03_lognorm``), runs the GPU
gene-moments kernel + host seurat binning, and checks per-gene ``means``,
``dispersions_norm`` and the ``highly_variable`` flag against snapshots ``04_*``.

    conda activate metasinglecell
    python validation_notebooks/03_hvg_parity.py
"""

import logging

import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.preprocess import highly_variable_genes
from metasinglecell.reference import N_TOP_GENES
from metasinglecell.sparse import CSR


def main() -> None:
    res_dir = config.results_dir("hvg")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "hvg.log", mode="w"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("hvg")

    lognorm = validation.load_snapshot("03_lognorm")
    csr = CSR.from_scipy(sp.csr_matrix(lognorm))
    log.info("CSR on Metal: %d cells x %d genes, nnz=%d", csr.shape[0], csr.shape[1], csr.nnz)

    df = highly_variable_genes(csr, n_top_genes=N_TOP_GENES)

    records = [
        # log1p(mean) tracks the GPU mean kernel directly.
        validation.compare("means", df["means"].to_numpy(),
                           validation.load_snapshot("04_hvg_means"),
                           rtol=1e-4, atol=1e-5),
        # dispersions_norm carries NaNs (zero-variance genes); harness masks them.
        validation.compare("dispersions_norm", df["dispersions_norm"].to_numpy(),
                           validation.load_snapshot("04_hvg_dispersions_norm"),
                           rtol=1e-3, atol=1e-3),
        # the gene selection itself.
        validation.compare("highly_variable", df["highly_variable"].to_numpy().astype("int8"),
                           validation.load_snapshot("04_hvg_flag")),
    ]
    for r in records:
        log.info("%-18s passed=%s max_abs_err=%.3g max_rel_err=%.3g r=%.8f masks=%s",
                 r["check"], r["passed"], r.get("max_abs_err", float("nan")),
                 r.get("max_rel_err", float("nan")), r.get("pearson_r", float("nan")),
                 r.get("nonfinite_masks_match"))

    # How many of the 2000 selected genes agree?
    flag_got = df["highly_variable"].to_numpy().astype(bool)
    flag_exp = validation.load_snapshot("04_hvg_flag").astype(bool)
    agree = int((flag_got & flag_exp).sum())
    log.info("HVG selection overlap: %d / %d genes", agree, int(flag_exp.sum()))

    validation.write_report(records, "hvg")
    print(f"\nHVG parity: {'PASS' if all(r['passed'] for r in records) else 'FAIL'}")


if __name__ == "__main__":
    main()
