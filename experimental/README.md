# experimental/

Research spikes kept for the record but **excluded from the installed package** (they live
outside `src/`, so they do not ship in the wheel/sdist). Not imported by any public API.

- `louvain_fused_raw.py` — fully-fused single-dispatch Louvain (spin-barrier + CAS float-atomic
  Σtot). Correct on tiny graphs but deadlocks nondeterministically with >1 threadgroup on M3
  (MLX gives no co-residency guarantee) and is core-starved at G=1. Non-viable on Metal.
- `louvain_hybrid.py` — GPU computes all move gains in one dispatch, CPU applies them in gain
  order. Fragments fuzzy graphs (per-pass snapshot can't give per-color-fresh Σtot) and the
  Python apply is far slower at scale.

See the `graph-clustering` skill / `RESULTS_clustering_workarounds.md` for the full evidence. The
shipped multi-core per-color path (in `src/metasinglecell/graph/louvain.py`) supersedes both.
