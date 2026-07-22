"""Harmony integration — quality-parity + on-GPU-correction asserting tests.

Guards the harmonize optimizations (E1–E4): (1) the GPU analytic block-inverse correction agrees with
the fp64-host oracle, (2) harmonize actually mixes batches (iLISI up vs raw) at least as well as the CPU
reference harmonypy, and (3) block_proportion is quality-neutral. Uses a small synthetic 2-batch
embedding with a real batch shift — no external data needed, so it runs in the CPU lane too (mlx-gated).
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.metal]


def _synthetic(n=1500, d=20, n_batches=2, effect=0.6, seed=0):
    """Two Gaussian cell-types × n_batches, with a per-batch offset (a removable batch effect)."""
    rng = np.random.default_rng(seed)
    ctype = rng.integers(0, 2, n)
    centers = rng.standard_normal((2, d)) * 3.0
    Z = centers[ctype] + rng.standard_normal((n, d)).astype(np.float32)
    batch = rng.integers(0, n_batches, n)
    offsets = rng.standard_normal((n_batches, d)).astype(np.float32) * effect
    Z = (Z + offsets[batch]).astype(np.float32)
    return Z, batch.astype(str), ctype


def _ilisi(Z, batch, k=30):
    from sklearn.neighbors import NearestNeighbors
    cats, code = np.unique(batch, return_inverse=True)
    idx = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z, return_distance=False)
    lab = code[idx]
    p = np.stack([(lab == b).mean(1) for b in range(len(cats))], 1)
    return float((1.0 / (p * p).sum(1)).mean())


def test_gpu_correction_matches_host_oracle():
    """The on-GPU analytic block-inverse correction ≈ the fp64-host solve."""
    from metalsinglecell.integration import harmonize
    from metalsinglecell.validation import subspace_overlap
    Z, batch, _ = _synthetic()
    Zg = harmonize(Z, batch, correction="gpu", random_state=0)
    Zh = harmonize(Z, batch, correction="host", random_state=0)
    assert np.isfinite(Zg).all()
    assert subspace_overlap(Zg, Zh) >= 0.99      # Jacobi-fp32 vs Gauss-Seidel-fp64


def test_harmonize_mixes_at_least_as_well_as_harmonypy():
    """Batch mixing (iLISI) increases vs raw and is no worse than CPU harmonypy."""
    import pandas as pd
    harmonypy = pytest.importorskip("harmonypy")
    from metalsinglecell.integration import harmonize
    Z, batch, _ = _synthetic()
    mix_raw = _ilisi(Z, batch)
    ho = harmonypy.run_harmony(Z.astype(np.float64), pd.DataFrame({"b": batch}), ["b"])
    Zc_hpy = np.asarray(ho.Z_corr)
    Zc_hpy = Zc_hpy if Zc_hpy.shape[0] == Z.shape[0] else Zc_hpy.T
    Zc_our = harmonize(Z, batch, random_state=0)
    mix_hpy, mix_our = _ilisi(Zc_hpy, batch), _ilisi(Zc_our, batch)
    assert mix_our > mix_raw + 0.05             # actually mixed the batches
    assert mix_our >= mix_hpy - 0.05            # no worse than the CPU reference


def test_block_proportion_quality_neutral():
    """Larger blocks (0.1) give essentially the same integration as the reference 0.05."""
    from metalsinglecell.integration import harmonize
    from metalsinglecell.validation import subspace_overlap
    Z, batch, _ = _synthetic()
    Z05 = harmonize(Z, batch, block_proportion=0.05, random_state=0)
    Z10 = harmonize(Z, batch, block_proportion=0.1, random_state=0)
    assert subspace_overlap(Z05, Z10) >= 0.99
