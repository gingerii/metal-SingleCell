# metal-SingleCell

[![CI](https://github.com/gingerii/metal-SingleCell/actions/workflows/ci.yml/badge.svg)](https://github.com/gingerii/metal-SingleCell/actions/workflows/ci.yml)

**GPU-accelerated single-cell analysis on Apple Silicon.**

A Metal/[MLX](https://github.com/ml-explore/mlx) re-implementation of
[rapids-singlecell](https://rapids-singlecell.readthedocs.io) — drop-in replacements for the core
[scanpy](https://scanpy.readthedocs.io) and [squidpy](https://squidpy.readthedocs.io) functions that
run on the **M-series GPU**. rapids-singlecell is CUDA/CuPy-only; there is no Apple-silicon path
because the M-series GPU has no native sparse-matrix support. This project builds that missing
sparse substrate and the scanpy/squidpy front-end on top of it.

It is a **drop-in API**: swap the import prefix and your existing pipeline runs on the GPU.

```python
import scanpy as sc
import metasinglecell as msc          # pp / tl / gr — mirror sc.pp, sc.tl, sq.gr

adata = sc.datasets.pbmc3k()
msc.pp.normalize_total(adata, target_sum=1e4)
msc.pp.log1p(adata)
msc.pp.highly_variable_genes(adata, n_top_genes=2000)
msc.pp.pca(adata)
msc.pp.neighbors(adata)
msc.tl.leiden(adata, backend="gpu")    # Metal parallel Leiden
msc.tl.umap(adata)
sc.pl.umap(adata, color="leiden")      # plot with scanpy as usual
```

Functions take an `AnnData`, compute on the GPU, and write results back to the **same slots scanpy
uses** (`.X`, `.obsm`, `.obsp`, `.var`, `.uns`), with scanpy's `copy=` semantics — so plotting and
downstream tooling are unchanged.

## Install

```bash
pip install -e .          # in a Python 3.11 env with mlx (Apple Silicon)
```

Heavy backends (`mlx`, `scanpy`, `squidpy`) are lazy-imported, so the package imports cleanly in any
environment. See [`envs/`](envs/) for the conda environment (`envs/metasinglecell.yml`), which is the
recommended setup.

**uv users:** run `uv python install 3.11` *before* creating your venv. The system python.org 3.11 on
macOS arm64 can hang indefinitely on first `import scanpy` (a numba/LLVM AOT-compile deadlock on
framework Python builds); uv's own managed 3.11 build (3.11.13) avoids it. Then:

```bash
uv python install 3.11
uv venv --python 3.11.13 .venv
uv pip install -e ".[oracle,metal,dev]"
```

## Tutorials

Four executable notebooks in [`notebooks/`](notebooks/), mirroring the rapids-singlecell tutorials:

| Notebook | Workflow |
|---|---|
| [`01_basic_workflow.ipynb`](notebooks/01_basic_workflow.ipynb) | QC → normalize → HVG → scale → PCA → neighbors → UMAP → Leiden → markers → Harmony → diffmap (PBMC 3k) |
| [`02_pearson_residuals.ipynb`](notebooks/02_pearson_residuals.ipynb) | Analytic Pearson-residual normalization → PCA → clustering (PBMC 3k) |
| [`04_squidpy.ipynb`](notebooks/04_squidpy.ipynb) | Spatial graph, Moran's I / Geary's C, co-occurrence (squidpy IMC) |
| [`brain_1M.ipynb`](notebooks/brain_1M.ipynb) | Full 1,000,000-cell workflow on a laptop GPU |

Notebooks 1–3 are self-contained (they auto-download their datasets). `brain_1M` needs the 10x
1.3M-neuron `.h5` placed at `data/external/1M_neurons.h5`; the 2M-cell Xenium benchmark
(`validation_notebooks/`) reads its path from the `XENIUM_H5` environment variable.

## Hardware tested

All benchmarks below were measured on a single laptop:

| | |
|---|---|
| **Chip** | Apple **M3 Max** |
| **GPU** | 40 cores, Metal 3, ~400 GB/s unified-memory bandwidth |
| **CPU** | 16 cores (12 performance + 4 efficiency) |
| **Memory** | 48 GB unified |
| **OS / runtime** | macOS 14.4 · MLX 0.31.2 · Python 3.11 |

## Speedups vs CPU

Speedup = CPU reference time ÷ our GPU time, **both on the same M3 Max** (vs scanpy / scikit-learn /
squidpy / igraph / harmonypy / bbknn / scrublet). Methodology: warm-up (defeats MLX first-call kernel
compile), best-of-N, on **real data** (PBMC 3k, a 1.3 M-neuron atlas at 50k/100k/1M, a 2 M-cell Xenium
cohort). Clustering is timed **algorithm-only** — the kNN graph is pre-built for *both* sides.
`–` = not run at that size; values in parentheses are our runtime in seconds where a CPU reference is
impractical at that scale.

Sizes are cells: **2k–2M are the 1.3 M-neuron atlas** (sub-/over-sampled) and the **2 M-cell Xenium
cohort** — one consistent data family across the table.

### Tier 1 — large GPU wins (parallel-arithmetic, bandwidth-bound)

| function | 2k | 50k | 100k | 1M | 2M |
|----------|---:|----:|-----:|---:|---:|
| highly_variable_genes | 5.8× | 25.9× | 32.9× | 15.8× | **49.2×** |
| umap | **32.2×** | 10.5× | 7.8× | (188 s) | (75 s) |
| scrublet | 15.1× | 6.4× | – | – | – |
| draw_graph | 16.1× | – | – | – | – |
| normalize_pearson_residuals | 7.4× | 9.8× | 9.1× | – | – |
| rank_genes_groups (t-test) | 3.7× | 9.6× | 7.6× | – | – |
| pca (sparse) | 2.1× | 4.3× | 4.4× | 4.5× | (8.0 s) |
| kmeans | 1.1× | 3.5× | 4.0× | (1.1 s) | (3.2 s) |
| diffmap | 1.7× | 2.9× | 2.7× | 3.6× | (36 s) |
| normalize + log1p | 5.4× | 3.5× | 3.5× | 1.7× | 2.8× |

### Tier 2 — GPU wins **at scale** (graph clustering)

| function | 2k | 50k | 100k | 1M | 2M | real 1M† |
|----------|---:|----:|-----:|---:|---:|---:|
| louvain | 0.14× | 21.9× | 21.0× | 9.7× | (16.1 s) | **58.6×** |
| leiden | 0.05× | 4.6× | 3.6× | 3.2× | 3.5× | **4.8×** |

†*real 1M* = the actual 986k-neuron kNN-15 graph (not synthetic); 50k–2M are SBM benchmark graphs where
igraph is unusually fast, so they understate the real-data advantage. Leiden timed at `n_iterations=1`
(the speed operating point — see note below).

`co_occurrence` (spatial): ~1.6× vs squidpy at 25k–100k, exact match, and scales past the n² memory
wall via a tiled device-atomic-histogram kernel.

Clustering crosses over with cell count: **Louvain wins from ~50k up (58.6× on the real 986k-neuron
graph)**, and **Leiden — after an O(degree) SIMD-group kernel rewrite + vertex pruning — went from a
catastrophic 0.05× to a 4.1× end-to-end speedup, winning at every scale ≥50k (4.8× vs igraph on the real
986k graph)** — see *What was solved*.

**Leiden quality/speed operating points.** The user-facing `msc.tl.leiden` defaults to
`n_iterations=2` (matching scanpy), which reaches igraph-parity modularity (Q 0.8586 vs igraph 0.8588 on
the 986k graph) at ~2.6× vs igraph. The `n_iterations=1` fast path is ~4.8× at Q 0.8504 (~1% under
igraph). Both are valid; pick speed or exact-parity per run.

### Accuracy

Every accelerated function is validated against its CPU reference: normalize/Pearson exact (Δ≈1e-6),
HVG gene-overlap **1.000**, PCA subspace 0.97–0.99, rank-genes top-k overlap 1.000, Leiden modularity at
igraph parity (Q 0.8586 vs 0.8588 at `n_iterations=2`; Q 0.8504 at the `n_iterations=1` speed point),
kNN recall ≈0.99, co-occurrence correlation **1.000** vs squidpy. An asserting `tests/` suite is in
progress (drop-in defaults, kNN and numerical-accuracy guards landed; streaming/out-of-core coverage is
being added).

## What was solved

- **A GPU sparse substrate for Apple Silicon** — CSR container, SpMM, segmented reductions, and
  custom `mx.fast.metal_kernel` kernels (QC, sparse PCA, a register-based top-k for kNN), since the
  M-series GPU has no native sparse support.
- **The full scanpy `pp`/`tl` + squidpy `gr` surface** — ~30 functions as a drop-in AnnData API.
- **Coloring-free parallel clustering** — the first parallel Louvain/Leiden on Metal. Replacing
  graph-coloring local-moving with cuGraph-style synchronous moves + a random-commit rule took
  **Louvain to 58.6×** on the real 986k-neuron graph. **Leiden** was then rebuilt with **O(degree)
  SIMD-group move/refine kernels** (`simd_sum`/`simd_any`, no atomics or grid barrier — retiring the old
  O(degree²) rescans) plus **vertex pruning** and **batched host-sync**, for a **4.1× end-to-end speedup**
  (11.9 s → 2.9 s on the 986k graph) at equal-or-better modularity — winning at every scale ≥50k.
- **A fused co-occurrence kernel** — a tiled device-atomic histogram that matches squidpy exactly,
  runs ~1.6× faster, and scales past the n² memory wall to 100k+ cells.
- **Out-of-core front-end (streaming from disk)** — the sparse, memory-bound front-end
  (QC → normalize → log1p → HVG → scale → PCA) streams cell-axis row-blocks from a chunked on-disk zarr
  store, so peak memory is bounded by one block, not the cell count. The **full 1,306,127 × 27,998 atlas
  (2.6 B non-zeros) — which OOMs an in-core run — completes end-to-end in ~300 s at 25.6 GB peak on a
  48 GB laptop.** PCA uses a single-pass covariance-eigh solver (rapids-singlecell's Dask-PCA choice).
  See [*Out-of-core*](#out-of-core-atlas-scale-on-a-laptop).
- **Validated at atlas scale** — a complete 1 M-cell workflow (and every function through 2 M cells)
  runs end-to-end on a 48 GB laptop; the full 28k-gene 1.3 M-neuron atlas runs out-of-core.

## Out-of-core: atlas-scale on a laptop

The sparse front-end can run on a dataset whose full expression matrix does **not** fit in unified
memory, by streaming cell-axis row-blocks from a chunked on-disk **zarr** store. Peak memory is bounded
by one block plus small accumulators — not the cell count — so datasets that OOM an in-core run go
through on the same 48 GB laptop.

```bash
# one-time: convert an .h5ad / 10x .h5 to a chunked-zarr store
python -m metasinglecell.backed  1M_neurons.h5  atlas.zarr  --block-rows 100000
```

```python
import anndata as ad, metasinglecell as msc
adata = ad.read_zarr("atlas.zarr")     # backed, not fully loaded
msc.pp.calculate_qc_metrics(adata)     # each pp step detects the backed .X and streams
msc.pp.normalize_total(adata); msc.pp.log1p(adata)
msc.pp.highly_variable_genes(adata); msc.pp.scale(adata)
msc.pp.pca(adata, svd_solver="covariance_eigh")   # single streaming covariance pass → dense eigh
# → adata.obsm["X_pca"]; downstream (neighbors/UMAP/clustering) fits in memory and runs as usual
```

**Full 1.3M-neuron atlas** (`1,306,127 × 27,998`, 2.6 B non-zeros) — the case that OOMs in-core:

| | in-core | out-of-core (streaming) |
|---|---|---|
| front-end (QC→normalize→log1p→HVG→scale→PCA) | **OOMs** (>48 GB) | **completes** |
| peak memory | — | **25.6 GB** (bounded by `block_rows`) |
| wall time (end-to-end) | — | ~300 s |

Design notes: streaming is **opt-in** — it activates only when `.X` is a backed zarr store, and every
in-core default is unchanged. Results are **bit-exact** vs in-core for the linear ops (QC / normalize /
log1p) and match to **subspace ≥ 0.999** for PCA. Out-of-core PCA uses a **covariance-eigh** solver — one
streaming pass accumulates the gene×gene covariance, then a dense fp64 eigendecomposition (the same choice
rapids-singlecell made for Dask PCA); randomized/Lanczos are in-core only. An optional post-log1p
checkpoint (`msc.pp.materialize`) trades disk for repeated compute. Downstream (neighbors/UMAP/clustering)
runs in memory as usual — the `n × 50` embedding is small even at atlas scale.

## When to use the CPU version instead

We benchmarked honestly; the GPU does not win everywhere on this hardware. **Use the CPU
implementation (scanpy / squidpy / `backend="igraph"`) when:**

- **k-nearest-neighbors above ~250k cells.** `neighbors` dispatches to CPU `pynndescent` past 250k
  because a graph-based ANN genuinely beats our GPU brute/IVF there (measured: GPU IVF is ~2× *slower*
  at 1M). `bbknn` (batch-balanced kNN) is CPU-favored at scale for the same reason.
- **Leiden / Louvain below ~50k cells.** igraph's lazy-sequential optimizer is extremely fast on small
  graphs; the GPU only wins once parallelism outweighs launch/coloring overhead (keep the default
  `backend="igraph"` for small data).
- **Harmony integration.** An iterative small-matrix algorithm the CPU wins (our mixing quality
  matches/beats harmonypy, but it's slower on M3).
- **t-SNE above ~30k cells.** We delegate to scikit-learn's Barnes-Hut (≈1×); exact t-SNE is GPU-only
  below 30k.

Rule of thumb: the M3 GPU wins on **parallel-arithmetic, bandwidth-bound** work (and on clustering
once graphs are large), and loses on **iterative / latency-bound / well-optimized-ANN** work. Against
an already-optimized CPU reference (numba / igraph / pynndescent), a GPU kernel on M3 typically wins
~1.5–2× (the bandwidth ratio), not 10× — the large wins come from removing genuine algorithmic waste.

## vs rapids-singlecell (NVIDIA)

rapids-singlecell's headline speedups (e.g. 470× Leiden) are GPU-vs-CPU on a datacenter GPU
(A100/3090) with ~2–5× the memory bandwidth and fully-fused CUDA. We do **not** match those absolute
numbers. What this project provides is the **only Apple-silicon path** for this workflow, with honest
laptop-scale speedups and an identical drop-in API — develop and run atlas-scale single-cell analysis
on a Mac, no CUDA required.

## Credits & acknowledgements

metal-SingleCell is an **independent, unaffiliated** project. It would not exist without the work it
builds on, and the API and workflows are modeled directly on these libraries — please cite them:

- **[rapids-singlecell](https://github.com/scverse/rapids-singlecell)** (MIT, part of
  [scverse®](https://scverse.org)) — the GPU single-cell API this project mirrors for Apple Silicon,
  and the source of the tutorials reproduced here. See their repository for the preferred citation.
- **[scanpy](https://scanpy.readthedocs.io)** — Wolf, F. A., Angerer, P. & Theis, F. J. *SCANPY:
  large-scale single-cell gene expression data analysis.* Genome Biology 19, 15 (2018).
- **[squidpy](https://squidpy.readthedocs.io)** — Palla, G. *et al. Squidpy: a scalable framework for
  spatial omics analysis.* Nature Methods 19, 171–178 (2022).
- **[MLX](https://github.com/ml-explore/mlx)** (Apple) — the array framework powering the Metal GPU
  kernels.

The `pp` / `tl` / `gr` function names and signatures intentionally follow scanpy / squidpy /
rapids-singlecell so existing pipelines port with a one-line import change. This project is not
endorsed by or affiliated with scverse, NVIDIA, or Apple.

## License

BSD-3-Clause (see [`pyproject.toml`](pyproject.toml)). rapids-singlecell, scanpy, and squidpy are
released under permissive licenses (MIT / BSD-3-Clause); MLX is MIT.

## Status

Functionally complete and validated (pp / tl / gr + tutorials + benchmark), with an **out-of-core
streaming front-end** for datasets larger than memory. An asserting `tests/` suite + CI is in progress
(pre-PyPI). Version `0.0.1`. The full benchmark with methodology is in
[`results/validation/RESULTS_v_benchmark.md`](results/validation/RESULTS_v_benchmark.md).
