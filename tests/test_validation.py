"""Unit tests for the validation harness (validation.py) itself — it underpins every
parity assertion, so it must be trustworthy. Pure-CPU, GitHub-hosted lane."""
import numpy as np


def test_compare_identical_passes_exact():
    from metasinglecell import validation
    a = np.linspace(0, 1, 100)
    r = validation.compare("id", a, a.copy())
    assert r["passed"] and r["exact_match"]
    assert r["max_abs_err"] == 0.0 and r["pearson_r"] == 1.0


def test_compare_within_tolerance_passes_not_exact():
    from metasinglecell import validation
    a = np.linspace(1, 2, 100)
    r = validation.compare("close", a + 1e-8, a, rtol=1e-5, atol=1e-6)
    assert r["passed"] and not r["exact_match"]
    assert r["max_abs_err"] < 1e-6


def test_compare_beyond_tolerance_fails():
    from metasinglecell import validation
    a = np.ones(50)
    r = validation.compare("far", a + 0.1, a, rtol=1e-5, atol=1e-6)
    assert not r["passed"] and r["max_abs_err"] >= 0.1


def test_compare_shape_mismatch_fails_gracefully():
    from metasinglecell import validation
    r = validation.compare("shape", np.ones(10), np.ones(11))
    assert not r["passed"] and r["reason"] == "shape_mismatch"


def test_compare_nonfinite_masks_must_match():
    from metasinglecell import validation
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.0, np.nan, 4.0])
    assert validation.compare("nan_ok", a, b)["passed"]           # matching nan masks
    c = np.array([1.0, 2.0, 3.0, 4.0])                            # nan vs finite
    assert not validation.compare("nan_bad", a, c)["passed"]


def test_subspace_overlap_identical_and_orthogonal():
    from metasinglecell import validation
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 5))
    assert abs(validation.subspace_overlap(A, A.copy()) - 1.0) < 1e-9
    # a rotation within the same column space preserves the subspace
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    assert abs(validation.subspace_overlap(A, A @ Q) - 1.0) < 1e-9


def test_compare_signed_columns_sign_invariant():
    from metasinglecell import validation
    rng = np.random.default_rng(1)
    A = rng.standard_normal((80, 4))
    B = A * np.array([1, -1, 1, -1])                              # per-column sign flips
    r = validation.compare_signed_columns("signed", A, B)
    assert r["passed"] and r["min_abs_corr"] > 0.999


def test_write_report_roundtrips(tmp_path):
    import csv
    from metasinglecell import validation
    recs = [validation.compare("a", np.ones(3), np.ones(3)),
            validation.compare("b", np.ones(3), np.zeros(3))]
    out = validation.write_report(recs, "__unit_test_report__", filename="unit.csv")
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert {r["check"] for r in rows} == {"a", "b"}
    out.unlink()
