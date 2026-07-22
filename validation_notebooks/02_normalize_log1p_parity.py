"""Stage 1 validation: Metal normalize_total + log1p vs the fp64 CPU oracle.

Rebuilds the counts matrix (snapshot ``00_counts``), runs the GPU CSR
``normalize_total(target_sum=1e4)`` then ``log1p`` kernels, and checks the dense
results against snapshots ``02_normalized`` and ``03_lognorm``. fp32 GPU vs fp64
CPU, so we expect tiny (not exact) differences — tolerances are fp32-appropriate.

    conda activate metalsinglecell
    python validation_notebooks/02_normalize_log1p_parity.py
"""

import logging

import scipy.sparse as sp

from metalsinglecell import config, validation
from metalsinglecell.reference import TARGET_SUM
from metalsinglecell.sparse import CSR


def main() -> None:
    res_dir = config.results_dir("normalize_log1p")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "normalize_log1p.log", mode="w"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("normalize_log1p")

    counts = validation.load_snapshot("00_counts")
    csr = CSR.from_scipy(sp.csr_matrix(counts))
    log.info("CSR on Metal: %d x %d, nnz=%d", csr.shape[0], csr.shape[1], csr.nnz)

    normalized = csr.normalize_total(target_sum=TARGET_SUM)
    lognorm = normalized.log1p()

    # fp32 GPU vs fp64 CPU: allow fp32-level relative error.
    records = [
        validation.compare("normalized", normalized.toarray(),
                           validation.load_snapshot("02_normalized"),
                           rtol=1e-4, atol=1e-4),
        validation.compare("lognorm", lognorm.toarray(),
                           validation.load_snapshot("03_lognorm"),
                           rtol=1e-4, atol=1e-5),
    ]
    for r in records:
        log.info("%-12s passed=%s max_abs_err=%.3g max_rel_err=%.3g rmse=%.3g r=%.8f",
                 r["check"], r["passed"], r.get("max_abs_err", float("nan")),
                 r.get("max_rel_err", float("nan")), r.get("rmse", float("nan")),
                 r.get("pearson_r", float("nan")))

    path = validation.write_report(records, "normalize_log1p")
    log.info("parity report -> %s", path)
    print(f"\nnormalize+log1p parity: {'PASS' if all(r['passed'] for r in records) else 'FAIL'}")


if __name__ == "__main__":
    main()
