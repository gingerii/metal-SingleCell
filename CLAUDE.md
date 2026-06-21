# CLAUDE.md — rapids/metal Single Cell project

## Project overview

This repository hosts a exploratory project that is attempting to re-implement the rapidsSingleCell python package. This package supports GPU accelerated omics analysis and has drop in relacements for most of the core scanpy functions. However this project works with CUPY and NVIDIA drivers only. There is no apple silicon version/support to run the the M series GPU. The main reason this is the case is that apple silicon does not currently handle sparse matricies. 

** Proposed Development stages ** 


*** Stage 1 *** 
The first step in this project will be implementing sparse matrix support in low level apple Metal. In addition, we will need to implement other sparse and non-sparse operations on the GPU that are not currently implemented. Note that the resources.md file contains brief descriptions and links to APIs and Githubs related to numerical computing on the apple GPU that will help speed up developement time on this project. Please reference both scanpy, rapids SingleCell, and scipy to discover what types of operations needs to be developed before re-implementing the scanpy specfic functions. 

*** Stage 2 *** 

After the basic linear algebra and graph operations are constructed, we need to develop the drop in replacements for the core scanpy functions. Use rapids SingleCell as a guide, as these are the funcitons we would like to implement. 


*** Stage 3 *** 

Testing numerical stability and runtime. We then need to check that our functions can re-capitulate the proper outputs when compared to the CPU, and benchmark the speed improvements achieved. 

## Resources 

* Resources are linked the in the resources.md document. If you require more detailed knowledge and cannot find it yourself, ask for it and I will give it to you. If you find new resources yourself, update the resources document so you can find it again. To use context tokens efficiently, not need to load in all the resouces into memory unless necessary. 

## Repository structure 

data/ * all test data lives here 
src/ * all dev code lives here 
resources.md * File for more knowledge about the problem and possible ways to solve it. 
validation_notebooks/ * We need a qualitative example of the scanpy workflow once everything is working that users can follow. It will live here. 
results/ * All .csv, .png, and .pdfs related to the project will live here. For each stage, and function, we should break down things by sub folder so things don't get over cluttered. 


## Code & data rules
* **Use dedicated project environments only** — never reuse another project's conda env (e.g. the Xenium
  `spatial`/`morphometrics` envs). Project envs are defined in `envs/` (e.g. `envs/metasinglecell.yml`);
  create with `conda env create -f envs/<name>.yml` then `pip install -e .`. See the `environments` skill.
* All code lives in `src/` (reusable library) or `validation_notebooks/` (drivers). Notebooks stay
  lightweight and call the library — no `sys.path` hacks; import the installed package
   New env: `pip install -e .`.
* Library submodules must **lazy-import heavy deps** (torch, etc.) inside functions, so the package
  installs and imports in every env.
* **Raw data is immutable.** Derived/shared objects (processed h5ads, etc) go
  in `data/processed/` — they are *inputs* to analyses, NOT results. `results/` holds only reportable
  artifacts, grouped by analysis type.
* Resolve paths through `src/config.py` (honoring `DATA_ROOT`); avoid hardcoding.
* **Consult the matching skill before writing code** (durable facts live in `.claude/skills/`; see `SKILLS.md`).
* **Git**: ask before every `git push` (commit locally freely). `results/` is gitignored except `.gitkeep` — figures/outputs stay local; force-add only a specific file if publishing it. Remote: private GitHub repo **gingerii/metal-SingleCell** (https://github.com/gingerii/metal-SingleCell). Project root is `~/Desktop/metal-SingleCell`; always run git from there. (Note: the Python import package is still `metasinglecell` and the conda env is `metasinglecell` — display name differs from these identifiers.)
* Secrets (e.g. the gated-UNI HF token): `huggingface_hub` login / env var only — never commit.



## Validation & benchmarking scheme (project-wide)
The end-of-build validation of our Metal/MLX rapids-singlecell reimplementation. We measure
**accuracy AND speed** of every function across a dataset-size sweep, and optimize until the
limit is the **hardware, not our implementation**.

* **Dataset sweep**: PBMC3k (real, ~2.7K), then **10K, 50K, 100K, 1M, 2M cells**. Larger sizes are
  synthesized (replicate/sample real cells or realistic sparse counts ~6–7% density) so they are
  reproducible; accuracy is anchored on sizes where a scanpy/sklearn CPU reference is computable.
* **Accuracy**: compare each GPU function to its scanpy/sklearn CPU reference on the *same* data,
  using the per-function metric + tolerance from the `rapids-api` skill (exact where deterministic;
  fp32 tolerance otherwise; structure/ARI/correlation for stochastic methods). Record max-abs/rel
  err, correlation, ARI, etc.
* **Speed**: GPU walltime vs CPU (scanpy/sklearn) walltime → speedup factor, per size. Honest method:
  **warm up** the GPU once (MLX compiles kernels on first launch), **best-of-N** for both GPU and CPU,
  `mx.eval` to defeat lazy eval, include intrinsic host↔device transfer.
* **The optimization loop**: any function with a **negative or poor speedup**, or that is
  implementation-bound (Python-orchestration / sync overhead / O(d²) / no-sparse-matmul workarounds),
  gets optimized — reduce host syncs, fuse kernels, batch launches, degree-bin, etc. — until the
  bottleneck is genuinely the M-series GPU (bandwidth/cores), not our code. The clustering effort is
  the template: profiling found coloring was 60% of runtime → recolor-every-3 + degree-binning.
* **Outputs**: `results/validation/` — per-size CSV (function, accuracy metric, gpu_s, cpu_s, speedup,
  bottleneck note) + a summary table/figure. Findings + chosen optimizations recorded in skills.
* **Known fp32/parity caveats** to confirm here are tracked per-function in the `rapids-api` skill.

## Logging
* Any code that does something writes a log to its `results/<analysis>/` folder.
* When transforming data, offer to make tables/figures capturing the transformation.


## Methods & Results (per analysis, in its results folder)
* `METHODS_<analysis>.md`: brief, scientifically-relevant methods of anything reportable. Active voice,
  first person plural, minimal jargon (Nature style). The coding process is invisible — only the science.
* `RESULTS_<analysis>.md`: very concise natural-language results (2–3 key points max). Active voice.

## Code availability
* Maintain the single `CODE_AVAILABILITY.md`: comma-separated packages+versions, **grouped by environment**
  (and language), e.g. "morphometrics env (Python 3.11): torch vX, timm vX, ...".

## Skills — durable knowledge (token-efficient)
* Detailed durable facts (file locations, versions, data quirks, pipeline internals, past findings) live as
  **skills** in `.claude/skills/<name>/SKILL.md`. Only each skill's one-line `description` stays in context; a
  skill's body loads **only when relevant** (progressive disclosure). Catalog + usage notes: `SKILLS.md`.
* Before writing code or answering a project question, let the matching skill load (automatic by topic; or name
  it / type `/xenium-<topic>`). The 7 skills: project, environments, data, cell-typing, morphology, integration, gotchas.
* **Record a new durable fact in the relevant skill's `SKILL.md`** (genuinely new topic → new skill dir + a row in
  `SKILLS.md`). Keep descriptions specific and bodies focused. This file holds only always-on rules — not facts.