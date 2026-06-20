"""Stage 1 validation: Metal CSR QC-metrics kernel vs the fp64 CPU oracle.

Rebuilds the exact counts matrix the oracle used (snapshot ``00_counts``), runs
the GPU segmented-reduction kernel, and checks per-cell ``total_counts`` and
``n_genes_by_counts`` against snapshots ``01_*``. Writes a parity report + log to
results/qc_metrics/.

    conda activate metasinglecell
    python validation_notebooks/01_qc_metrics_parity.py
"""

import logging

import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.sparse import CSR


def main() -> None:
    res_dir = config.results_dir("qc_metrics")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "qc_metrics.log", mode="w"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("qc_metrics")

    # Same matrix the oracle computed QC on (dense snapshot -> CSR on GPU).
    counts = validation.load_snapshot("00_counts")
    csr = CSR.from_scipy(sp.csr_matrix(counts))
    log.info("CSR on Metal: %d cells x %d genes, nnz=%d",
             csr.shape[0], csr.shape[1], int(csr.data.size))

    total_counts, n_genes = csr.qc_metrics()

    records = [
        validation.compare("total_counts", total_counts,
                           validation.load_snapshot("01_total_counts")),
        validation.compare("n_genes_by_counts", n_genes,
                           validation.load_snapshot("01_n_genes_by_counts")),
    ]
    for r in records:
        log.info("%-20s passed=%s exact=%s max_abs_err=%.3g r=%.8f",
                 r["check"], r["passed"], r.get("exact_match"),
                 r.get("max_abs_err", float("nan")), r.get("pearson_r", float("nan")))

    path = validation.write_report(records, "qc_metrics")
    log.info("parity report -> %s", path)
    all_pass = all(r["passed"] for r in records)
    print(f"\nQC-metrics parity: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
