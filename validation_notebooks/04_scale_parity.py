"""Stage 1 validation: Metal scale (per-gene z-score + clip) vs the fp64 oracle.

Reconstructs the HVG-subset log-normalized matrix the oracle scaled (snapshot
``03_lognorm`` subset to the ``04_hvg_flag`` genes), runs the GPU ``scale``
(max_value=10), and checks the dense result against snapshot ``05_scaled``.

    conda activate metasinglecell
    python validation_notebooks/04_scale_parity.py
"""

import logging

import scipy.sparse as sp

from metasinglecell import config, validation
from metasinglecell.preprocess import scale
from metasinglecell.reference import SCALE_MAX
from metasinglecell.sparse import CSR


def main() -> None:
    res_dir = config.results_dir("scale")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "scale.log", mode="w"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("scale")

    lognorm = validation.load_snapshot("03_lognorm")
    hvg = validation.load_snapshot("04_hvg_flag").astype(bool)
    subset = lognorm[:, hvg]  # HVG-subset, original gene order (matches oracle)
    csr = CSR.from_scipy(sp.csr_matrix(subset))
    log.info("HVG-subset CSR on Metal: %d cells x %d genes", *csr.shape)

    scaled = scale(csr, max_value=SCALE_MAX)

    record = validation.compare("scaled", scaled,
                                validation.load_snapshot("05_scaled"),
                                rtol=1e-4, atol=1e-4)
    log.info("scaled passed=%s max_abs_err=%.3g max_rel_err=%.3g rmse=%.3g r=%.8f",
             record["passed"], record["max_abs_err"], record["max_rel_err"],
             record["rmse"], record["pearson_r"])

    validation.write_report([record], "scale")
    print(f"\nscale parity: {'PASS' if record['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
