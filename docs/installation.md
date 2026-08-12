# Installation

```bash
pip install metalsinglecell
```

Requires **Python ≥ 3.11**. On Apple Silicon this also pulls in `mlx` (the Metal GPU backend); on other
platforms it installs the pure-NumPy/SciPy core, so the package imports everywhere (`mlx` is a
Darwin/arm64-only dependency). Heavy backends (`mlx`, `scanpy`, `squidpy`) are lazy-imported, so it
imports cleanly in any environment.

## Extras

| extra | what it enables |
|---|---|
| `backed` | the out-of-core streaming path and `pp.materialize` (zarr) |
| `hvg` | `pp.highly_variable_genes(flavor="seurat_v3")`, which fits scanpy's lowess curve |
| `all` | both of the above |
| `oracle` | scanpy / squidpy and friends, for running the parity tests against the CPU reference |

```bash
pip install "metalsinglecell[all]"
```

`igraph` and `scikit-learn` are **core** dependencies rather than extras, since they sit on default
code paths: `tl.leiden` and `tl.louvain` use igraph by default, and scikit-learn backs
`rank_genes_groups(method="logreg")`, the `"cosine"` graph transform, and the t-SNE CPU fallback.
Through 0.1.2 they were listed only under `oracle`, so a clean install could not run a default
clustering call.

## Development install

```bash
git clone https://github.com/gingerii/metal-SingleCell.git
cd metal-SingleCell
pip install -e .
```

## Conda environment

A reproducible environment (used for the benchmarks) is provided:

```bash
conda env create -f envs/metalsinglecell.yml
conda activate metalsinglecell
pip install -e .
```

## uv users

Run `uv python install 3.11` *before* creating your venv. The system python.org 3.11 on macOS arm64 can
hang indefinitely on first `import scanpy` (a numba/LLVM AOT-compile deadlock on framework Python builds);
uv's own managed 3.11 build avoids it.

```bash
uv python install 3.11
uv venv --python 3.11.13 .venv
uv pip install -e ".[all,oracle,metal,dev]"
```
