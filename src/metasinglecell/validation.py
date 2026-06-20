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

    ``rtol``/``atol`` default to fp32-appropriate tolerances. Non-finite entries
    (NaN/inf — e.g. HVG dispersions for zero-variance genes) are handled
    explicitly: error metrics are computed over positions finite in both arrays,
    and the comparison only passes if the non-finite *masks* match.
    """
    got = np.asarray(got, dtype=np.float64).ravel()
    expected = np.asarray(expected, dtype=np.float64).ravel()
    if got.shape != expected.shape:
        return {
            "check": name, "passed": False, "reason": "shape_mismatch",
            "got_shape": got.size, "expected_shape": expected.size,
        }

    finite = np.isfinite(got) & np.isfinite(expected)
    masks_match = bool(np.array_equal(np.isfinite(got), np.isfinite(expected)))
    g, e = got[finite], expected[finite]

    abs_err = np.abs(g - e)
    denom = np.maximum(np.abs(e), 1e-12)
    rel_err = abs_err / denom
    corr = float(np.corrcoef(g, e)[0, 1]) if g.size > 1 else 1.0
    passed = masks_match and bool(np.allclose(g, e, rtol=rtol, atol=atol))

    return {
        "check": name,
        "passed": passed,
        "exact_match": bool(np.array_equal(got, expected, equal_nan=True)),
        "nonfinite_masks_match": masks_match,
        "n": int(got.size),
        "n_compared": int(g.size),
        "max_abs_err": float(abs_err.max()) if g.size else 0.0,
        "max_rel_err": float(rel_err.max()) if g.size else 0.0,
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))) if g.size else 0.0,
        "pearson_r": corr,
        "rtol": rtol,
        "atol": atol,
    }


def compare_signed_columns(
    name: str,
    got: np.ndarray,
    expected: np.ndarray,
    min_abs_corr: float = 0.99,
) -> dict:
    """Compare matrices whose columns are defined only up to sign (PCA/embeddings).

    Computes the per-column absolute Pearson correlation between ``got`` and
    ``expected`` (both n x k) and passes if the worst column clears
    ``min_abs_corr``. Robust to SVD sign flips.
    """
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    k = min(got.shape[1], expected.shape[1])
    corrs = np.array([abs(np.corrcoef(got[:, i], expected[:, i])[0, 1]) for i in range(k)])
    return {
        "check": name,
        "passed": bool(corrs.min() >= min_abs_corr),
        "n_components": int(k),
        "min_abs_corr": float(corrs.min()),
        "mean_abs_corr": float(corrs.mean()),
        "min_abs_corr_threshold": min_abs_corr,
    }


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized overlap of two k-dim column subspaces, in [0, 1] (1 == identical).

    ``||Qa^T Qb||_F^2 / k`` where Qa, Qb are orthonormal bases of the columns.
    """
    qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64))
    k = min(a.shape[1], b.shape[1])
    return float(np.linalg.norm(qa[:, :k].T @ qb[:, :k], "fro") ** 2 / k)


def write_report(records: list[dict], analysis: str, filename: str = "parity_report.csv") -> Path:
    """Write parity records to ``results/<analysis>/<filename>`` and return the path."""
    out = config.results_dir(analysis) / filename
    fields = sorted({k for r in records for k in r})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    return out
