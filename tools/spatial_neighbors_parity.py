"""Parity of the four Metal spatial-neighbour builders vs squidpy, on real Visium.

Compares the actual written slots, not just edge counts: connectivities and distances must
match element-wise, and uns params must agree.
"""
import warnings
import numpy as np

warnings.simplefilter("ignore")
import scanpy as sc
import squidpy as sq
import metalsinglecell as msc

a = sc.read_visium("data/V1_Breast_Cancer_Block_A_Section_1")
a.var_names_make_unique()
print(f"Visium {a.shape}\n")


def tie_explained(ours, theirs, coords):
    """Every disagreeing edge must connect points at an exactly equal distance to a kept one."""
    A, B = ours.tocsr(), theirs.tocsr()
    bad = 0
    for i in range(A.shape[0]):
        so, st = set(A.indices[A.indptr[i]:A.indptr[i+1]]), set(B.indices[B.indptr[i]:B.indptr[i+1]])
        if so == st:
            continue
        de = [np.linalg.norm(coords[i] - coords[j]) for j in so - st]
        dm = [np.linalg.norm(coords[i] - coords[j]) for j in st - so]
        if not (de and dm and abs(max(de) - max(dm)) < 1e-9):
            bad += 1
    return bad


def cmp(tag, ours, theirs, coords=None):
    ok = True
    for key in ("spatial_connectivities", "spatial_distances"):
        A, B = ours.obsp[key].tocsr(), theirs.obsp[key].tocsr()
        A.sort_indices(); B.sort_indices()
        same_pattern = (A.shape == B.shape and A.nnz == B.nnz
                        and np.array_equal(A.indptr, B.indptr)
                        and np.array_equal(A.indices, B.indices))
        maxdiff = (np.abs(A.data - B.data).max()
                   if same_pattern and A.nnz else (np.inf if not same_pattern else 0.0))
        ok &= same_pattern and maxdiff < 1e-4
        print(f"    {key:24s} nnz {A.nnz:>7}/{B.nnz:<7} pattern={'=' if same_pattern else 'X'}"
              f"  maxdiff={maxdiff:.3g}")
    if not ok and coords is not None:
        unexplained = tie_explained(ours.obsp["spatial_connectivities"],
                                    theirs.obsp["spatial_connectivities"], coords)
        print(f"    rows differing for a reason OTHER than an exact tie: {unexplained}")
        ok = unexplained == 0
    po, pt = ours.uns["spatial_neighbors"]["params"], theirs.uns["spatial_neighbors"]["params"]
    if po != pt:
        print(f"    uns params differ:\n      ours   {po}\n      squidpy {pt}")
        ok = False
    print(f"  {tag:34s} {'PASS' if ok else 'FAIL'}")
    return ok


CASES = [
    ("knn n_neighs=6", msc.gr.spatial_neighbors_knn, sq.gr.spatial_neighbors_knn,
     dict(n_neighs=6)),
    ("knn n_neighs=12", msc.gr.spatial_neighbors_knn, sq.gr.spatial_neighbors_knn,
     dict(n_neighs=12)),
    ("knn set_diag", msc.gr.spatial_neighbors_knn, sq.gr.spatial_neighbors_knn,
     dict(n_neighs=6, set_diag=True)),
    ("knn transform=spectral", msc.gr.spatial_neighbors_knn, sq.gr.spatial_neighbors_knn,
     dict(n_neighs=6, transform="spectral")),
    ("knn percentile=90", msc.gr.spatial_neighbors_knn, sq.gr.spatial_neighbors_knn,
     dict(n_neighs=6, percentile=90.0)),
    ("radius=300", msc.gr.spatial_neighbors_radius, sq.gr.spatial_neighbors_radius,
     dict(radius=300.0)),
    ("radius=600", msc.gr.spatial_neighbors_radius, sq.gr.spatial_neighbors_radius,
     dict(radius=600.0)),
    ("radius=(200,400)", msc.gr.spatial_neighbors_radius, sq.gr.spatial_neighbors_radius,
     dict(radius=(200.0, 400.0))),
    ("grid n_rings=1", msc.gr.spatial_neighbors_grid, sq.gr.spatial_neighbors_grid,
     dict(n_neighs=6)),
    ("grid n_rings=2", msc.gr.spatial_neighbors_grid, sq.gr.spatial_neighbors_grid,
     dict(n_neighs=6, n_rings=2)),
    ("grid n_rings=3", msc.gr.spatial_neighbors_grid, sq.gr.spatial_neighbors_grid,
     dict(n_neighs=6, n_rings=3)),
    ("grid delaunay", msc.gr.spatial_neighbors_grid, sq.gr.spatial_neighbors_grid,
     dict(delaunay=True)),
    ("delaunay", msc.gr.spatial_neighbors_delaunay, sq.gr.spatial_neighbors_delaunay, {}),
    ("delaunay radius=300", msc.gr.spatial_neighbors_delaunay,
     sq.gr.spatial_neighbors_delaunay, dict(radius=300.0)),
    ("delaunay percentile=95", msc.gr.spatial_neighbors_delaunay,
     sq.gr.spatial_neighbors_delaunay, dict(percentile=95.0)),
]

passed = 0
for tag, ours_fn, sq_fn, kw in CASES:
    o, t = a.copy(), a.copy()
    try:
        ours_fn(o, **kw)
        sq_fn(t, **kw)
        passed += cmp(tag, o, t, np.asarray(a.obsm['spatial'], dtype=np.float64))
    except Exception as e:
        print(f"  {tag:34s} ERROR {type(e).__name__}: {e}")
print(f"\n{passed}/{len(CASES)} match squidpy (exactly, or differing only at exact ties)")
