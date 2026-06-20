"""Driver: build the fp64 CPU reference oracle (PBMC3k).

Lightweight — all logic lives in ``metasinglecell.reference``. Run inside the
dedicated env:

    conda activate metasinglecell
    python validation_notebooks/00_cpu_reference_oracle.py

Writes full-precision snapshots to data/processed/reference/ and a manifest +
log to results/reference/. These snapshots are the ground truth for validating
every Metal/MLX kernel built in later stages.
"""

import logging
from pathlib import Path

from metasinglecell import config
from metasinglecell.reference import run_cpu_reference


def main() -> None:
    config.ensure_data_dirs()
    log_path = config.results_dir("reference") / "cpu_reference.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    snap_dir = run_cpu_reference(log=logging.getLogger("reference"))
    print(f"\nReference snapshots written to: {snap_dir}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
