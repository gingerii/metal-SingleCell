"""Does the Delaunay triangulator hold up on real assay geometry?

Synthetic point sets are too kind. Real spatial data brings three things they do not:
Visium's hexagonal spot lattice, Stereo-seq's *square* binning (where a third of the edges
are cocircular and the triangulation is genuinely non-unique), and Xenium's irregular
float-micron centroids over fields with large empty gaps between sections — which is where
the in-circle operands blow past the 64-bit window and the 128-bit path earns its place.

Coordinates are read straight out of ``obsm/spatial`` with h5py, so nothing loads an
expression matrix and the 11 GB object costs the same as the small ones.

The four checks are the ones from ``delaunay_reference_validate.py``; see that file for why
"locally Delaunay" on its own is not enough.

Run: ``python tools/delaunay_real_data_validate.py [--max-n N]``
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

sys.path.insert(0, str(Path(__file__).parent))
from delaunay_oracle import check_delaunay, edge_set          # noqa: E402

from metalsinglecell._delaunay import reference as R          # noqa: E402
from metalsinglecell._delaunay.predicates import (            # noqa: E402
    SAFE_ABS, condition_points, orient2d,
)

warnings.simplefilter("ignore")

GD = ("/Users/f006z2w/Library/CloudStorage/"
      "GoogleDrive-ian.gingerich.gr@dartmouth.edu/My Drive")

DATASETS = [
    ("Visium 2.7k", "Visium", f"{GD}/Quaternion_project/data/external/V1_Adult_Mouse_Brain.h5ad"),
    ("Stereo-seq 19k", "Stereo-seq", f"{GD}/Quaternion_project/data/external/stereoseq_olf.h5ad"),
    ("Xenium brain 63k", "Xenium", f"{GD}/Atlas_svd_project/data/processed/xenium_brain_5k.h5ad"),
    ("MERFISH 81k", "MERFISH", f"{GD}/Atlas_svd_project/data/processed/merfish_sagital_brain.h5ad"),
    ("Xenium breast 253k", "Xenium",
     f"{GD}/Quaternion_project/data/external/xenium_breast_matched/"
     "xenium_breast_s1bot_processed.h5ad"),
    ("Xenium cohort 2M", "Xenium",
     "/Users/f006z2w/Desktop/Xenium_Claude_test/data/processed/xenium/integrated_data.h5ad"),
]


def area_sum(pts, tri):
    p = pts.astype(np.float64)
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum()


def boundary_size(pts):
    hull = ConvexHull(pts.astype(np.float64)).vertices
    on = {int(v) for v in hull}
    rest = np.setdiff1d(np.arange(len(pts)), hull)
    ring = list(hull) + [hull[0]]
    for u, v in zip(ring[:-1], ring[1:]):
        if len(rest) == 0:
            break
        o = orient2d(pts, np.full(len(rest), u), np.full(len(rest), v), rest)
        on.update(int(x) for x in rest[o == 0])
    return len(on)


def wide_quad_fraction(ipts, tri):
    """Share of in-circle tests whose operands need the 128-bit path."""
    em = {}
    for t, (i, j, k) in enumerate(tri):
        for e in ((i, j), (j, k), (k, i)):
            em.setdefault(tuple(sorted((int(e[0]), int(e[1])))), []).append(t)
    quads = np.array([[*tri[ts[0]], [v for v in tri[ts[1]] if v not in e][0]]
                      for e, ts in em.items() if len(ts) == 2], dtype=np.int64)
    if len(quads) == 0:
        return 0.0
    d = ipts[quads[:, 3]]
    widest = np.abs(np.stack([ipts[quads[:, k]] - d for k in (0, 1, 2)])).max(axis=(0, 2))
    return float((widest > SAFE_ABS).mean())


def crop(pts, target, rng):
    """A contiguous spatial window of about ``target`` points.

    Preferred over a uniform subsample: thinning a tissue section uniformly destroys the
    lattice regularity and the density contrast, which are the properties being tested.
    """
    if len(pts) <= target:
        return pts, "full"
    keep = np.arange(len(pts))
    order = np.argsort(pts[:, 0])
    lo = rng.integers(0, len(pts) - target)
    keep = order[lo:lo + target]
    sub = pts[keep]
    # trim any exact duplicates the crop happens to contain
    _, uniq = np.unique(sub, axis=0, return_index=True)
    return sub[np.sort(uniq)], f"crop of {len(pts):,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=120_000,
                    help="triangulate at most this many points (the reference is a "
                         "reference, not the fast path)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"{'dataset':<20} {'assay':<11} {'n':>8} {'source':>16} {'tris':>8} "
          f"{'delaunay':>9} {'count':>6} {'area err':>10} {'ties':>7} {'128-bit':>8} "
          f"{'jaccard':>8} {'sec':>7}")
    print("-" * 132)

    failures = missing = 0
    for name, assay, path in DATASETS:
        try:
            with h5py.File(path, "r") as f:
                sp = np.asarray(f["obsm/spatial"][:], dtype=np.float64)[:, :2]
        except Exception as exc:
            print(f"{name:<20} {assay:<11} unavailable ({type(exc).__name__})")
            missing += 1
            continue

        pts, source = crop(sp, args.max_n, rng)
        ipts, _ = condition_points(pts)

        t0 = time.perf_counter()
        tri = R.triangulate(pts)
        el = time.perf_counter() - t0

        res = check_delaunay(ipts, tri)
        expect = 2 * len(pts) - 2 - boundary_size(ipts)
        hull_area = ConvexHull(ipts.astype(np.float64)).volume
        area_err = abs(area_sum(ipts, tri) - hull_area) / hull_area
        wide = wide_quad_fraction(ipts, tri)

        qh = Delaunay(ipts.astype(np.float64)).simplices
        eo, eq = edge_set(tri), edge_set(qh)
        jac = len(eo & eq) / len(eo | eq)

        count_ok = len(tri) == expect
        ok = (res["is_delaunay"] and count_ok and area_err < 1e-12
              and (jac == 1.0 or res["cocircular_edges"] > 0))
        failures += not ok
        print(f"{name:<20} {assay:<11} {len(pts):>8} {source:>16} {len(tri):>8} "
              f"{str(res['is_delaunay']):>9} {str(count_ok):>6} {area_err:>10.2e} "
              f"{res['cocircular_edges']:>7} {100 * wide:>7.2f}% {jac:>8.4f} {el:>7.1f}"
              + ("" if ok else "   <-- FAIL"))

    print(f"\n{'all datasets passed' if not failures else f'{failures} FAILURES'}"
          + (f" ({missing} unavailable)" if missing else ""))
    print("""
Columns
  count      triangle count equals 2n - 2 - h (h = every point on the hull boundary)
  area err   triangles cover the convex hull exactly once; non-zero means a hole or a fold
  ties       exactly cocircular edges — where these exist Qhull parity is not achievable,
             which is the expected state for square-binned assays
  128-bit    in-circle tests whose operands exceed the 64-bit window
  jaccard    edge agreement with Qhull
""")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
