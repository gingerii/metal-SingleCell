# Installation

```bash
pip install metalsinglecell
```

Requires **Python ≥ 3.11**. On Apple Silicon this also pulls in `mlx` (the Metal GPU backend); on other
platforms it installs the pure-NumPy/SciPy core, so the package imports everywhere (`mlx` is a
Darwin/arm64-only dependency). Heavy backends (`mlx`, `scanpy`, `squidpy`) are lazy-imported, so it
imports cleanly in any environment.

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
uv pip install -e ".[oracle,metal,dev]"
```
