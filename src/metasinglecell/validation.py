"""Numerical-parity harness: compare a GPU result against the fp64 CPU oracle.

Every Stage-1/2 kernel is validated here against the snapshots written by
``metasinglecell.reference``. The harness reports the metrics that matter for an
fp32-GPU-vs-fp64-CPU comparison (max abs/rel error, correlation, allclose) and
writes a tidy CSV row per check so parity is tracked over time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config


def load_snapshot(name: str) -> np.ndarray:
    """Load an oracle snapshot array (``data/processed/reference/<name>.npy``)."""
    return np.load(config.PROCESSED_DIR / "reference" / f"{name}.npy")


def compare(
    name: str,
    got: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict:
    """Compare two arrays and return a parity record.

    ``rtol``/``atol`` default to fp32-appropriate tolerances. Correlation is
    computed on the flattened finite values; for integer-valued outputs the
    ``exact_match`` field is the relevant one.
    """
    got = np.asarray(got, dtype=np.float64).ravel()
    expected = np.asarray(expected, dtype=np.float64).ravel()
    if got.shape != expected.shape:
        return {
            "check": name, "passed": False, "reason": "shape_mismatch",
            "got_shape": got.size, "expected_shape": expected.size,
        }

    abs_err = np.abs(got - expected)
    denom = np.maximum(np.abs(expected), 1e-12)
    rel_err = abs_err / denom
    corr = float(np.corrcoef(got, expected)[0, 1]) if got.size > 1 else 1.0
    passed = bool(np.allclose(got, expected, rtol=rtol, atol=atol))

    return {
        "check": name,
        "passed": passed,
        "exact_match": bool(np.array_equal(got, expected)),
        "n": int(got.size),
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float(rel_err.max()),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "pearson_r": corr,
        "rtol": rtol,
        "atol": atol,
    }


def write_report(records: list[dict], analysis: str, filename: str = "parity_report.csv") -> Path:
    """Write parity records to ``results/<analysis>/<filename>`` and return the path."""
    out = config.results_dir(analysis) / filename
    fields = sorted({k for r in records for k in r})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    return out
