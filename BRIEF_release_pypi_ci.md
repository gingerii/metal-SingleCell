# BRIEF — Release engineering: test suite + CI + PyPI (the scverse gate)

**Repo:** `~/Desktop/metal-SingleCell` · **Env:** conda `metasinglecell` · **HW:** M3 Max, MLX/Metal.
**Goal:** get the package to the state where it can be **published to PyPI and submitted to the scverse
ecosystem listing.** The code and numerics are done and validated; this is packaging + testing + CI only.
**Do not change library behavior** — if a test reveals a bug, report it, don't silently fix it.

> ⛔ **COMMIT ATTRIBUTION — standing rule.** Do **NOT** add any Claude / Anthropic / AI co-author or
> contributor attribution to commits: no `Co-Authored-By: Claude …`, no `Co-Authored-By:
> …@anthropic.com`, no "Generated with Claude Code" trailer, nothing naming an AI in the message,
> author, or committer fields. Commits are authored solely by the user. This applies to every commit in
> this and all future work on this repo. (It has snuck back in before — 28 existing commits carry the
> trailer; see the note the agent will hand back separately about cleaning those.) If a tool default
> re-adds it, strip it before committing.

**Read first:** `results/code_review/findings_lens4a.md` / `4b.md` (the release blueprint) and
`results/code_review/POSTFIX_VALIDATION.md`. Note that the release fix commit `e54f002` **already landed**
several lens-4 items — confirm current state before acting (see "Already done" below).

## Why these three, in this order
scverse ecosystem listing has a hard prerequisite chain: **installable from PyPI/conda-forge → has a real
test suite + CI → then submit the `meta.yaml` PR.** So: (1) tests, (2) CI, (3) PyPI. Each gates the next.

## Already done (verify, don't redo)
`pyproject.toml` already has: `mlx` as a **marker-gated core dep** (`platform_system=='Darwin' and
platform_machine=='arm64'`), OS/Python/Topic **classifiers**, `[project.urls]`, SPDX license + license-files.
The two experimental modules were **moved to `experimental/`** and excluded from the wheel. A **`tests/`
seed** exists (`tests/test_postfix_fixes.py`, asserting, CPU+Metal). Confirm each is still true, then build
on it — do not duplicate.

## Part 1 — Test suite (the largest piece; blueprint is in lens4a "Tests/CI needed for scverse")
Convert the numerical logic that already lives in `validation_notebooks/` into an **asserting** pytest
suite. The parity scripts compute `validation.compare(...)` records then `print` PASS/FAIL and **exit 0
regardless** — so they must become `assert record["passed"]`. Structure into two lanes:

**CPU lane (GitHub-hosted, no GPU — highest value, cheapest, do first):**
- **Import-contract smoke test** (the single highest-value test): `import metasinglecell` and its `pp`/`tl`/
  `gr` with **only** numpy/scipy/anndata installed, and **assert mlx and scanpy are NOT imported** — this
  guards the lazy-import promise that lets the package install anywhere. Catch it with
  `sys.modules` inspection after import.
- `config.py` and `validation.py` unit tests.
- The CPU reference-oracle build (`00_cpu_reference_oracle.py` → `reference.run_cpu_reference`) as a
  **pytest session fixture** producing the fp64 PBMC3k snapshots that blocks 01–09 compare against.

**GPU lane (self-hosted macOS-arm64 runner, `[metal]` extra):**
- Wrap each parity block as an asserting test: 01 QC, 02 normalize/log1p, 03 HVG, 04 scale, 05 pca
  (all three solvers incl. `covariance_eigh`), 06 neighbors, 08 umap+leiden, 09 graph primitives,
  10 louvain, 11 leiden(gpu).
- **Fold in the existing `tests/test_postfix_fixes.py`** (kNN k>32, fp64 moments, defaults, raise-paths).

**Net-new coverage the parity scripts never had (add assertions):**
- **`backed.py` out-of-core path** — the largest untested surface. Assert streaming QC/normalize/log1p are
  **bit-exact** vs in-core on a small backed zarr, and streaming covariance-eigh PCA is **subspace ≥0.999**
  vs in-core. (The M1/M2 validation scripts `results/zarr_outofcore/v_outofcore*.py` already contain this
  logic — promote it into `tests/`, asserting.)
- `pp` helpers with no parity coverage: `materialize`, `write_obsm`, `regress_out`, `filter_*`,
  `calculate_qc_metrics` (with `qc_vars`/`percent_top`).
- `integration.harmonize`, `neighbors.bbknn`, `gr.ligrec`, `gr.calculate_niche`.

Keep the private-data scripts (`v_realatlas.py`, `v_realxenium.py`, `v_realspatial.py`) **out of PR CI** —
they need local data. Mark them nightly/manual (`@pytest.mark.realdata`, skipped without the env var/file).

## Part 2 — CI (`.github/workflows/`)
- **`ci.yml`**: on push/PR, run the **CPU lane** on GitHub-hosted `macos-latest` (and optionally
  `ubuntu-latest` for the import-contract test — it proves the package installs & imports without mlx).
  Fast, free, and it's what a scverse reviewer will look for first.
- **GPU lane**: needs a **self-hosted Apple-Silicon runner with Metal** (GitHub offers no Metal GPU). Wire
  the workflow to run the GPU lane on a `self-hosted, macOS, arm64` runner label; document that the user
  must register one (this is the single infra dependency to note in the scverse submission). If no runner
  is available yet, the GPU lane should be a separate workflow that's manually-triggerable, so CPU CI is
  green regardless.
- Add a CI status badge to the README.

## Part 3 — PyPI publish (after tests+CI are green)
- Residual packaging cleanups (lens4 MINORs, verify each is still open first): remove the
  `package-data ["**/*.metal"]` glob (matches zero files — kernels are inline); fix the README link to
  `results/validation/RESULTS_v_benchmark.md` (it's **gitignored → 404s** on GitHub and in the PyPI
  long-description — either un-ignore it, move it somewhere tracked, or drop the link); fix "Notebooks 1–3"
  → only 01/02/04 exist; make `version` `dynamic` so it lives in one place; pin/lower-bound
  `envs/metasinglecell.yml` to match `CODE_AVAILABILITY.md`; remove the loose `scratchpad_*.py` at repo root.
- **Build & check:** `python -m build` → `twine check dist/*`. Confirm the sdist includes LICENSE+README
  and the wheel **excludes** `experimental/` and `data/`, `results/`.
- **Bump version off `0.0.1`** for the first real release (suggest `0.1.0` — first published, feature-complete
  with out-of-core). Confirm with the user before publishing.
- **Test-publish first:** upload to **TestPyPI**, install from it into a clean env on the M3, run the import
  smoke + one GPU test, then publish to real PyPI. Do **not** publish to real PyPI without the user's
  explicit go-ahead (it's irreversible — versions can't be reused).

## Deliverables
- `tests/` — full asserting suite (CPU + GPU lanes), incorporating the existing seed.
- `.github/workflows/ci.yml` (+ optional gpu workflow), README CI badge.
- Packaging cleanups in `pyproject.toml` / README / `envs/`.
- `results/code_review/RELEASE_READINESS.md`: what CI lane covers what, the self-hosted-runner
  requirement, TestPyPI dry-run result, and a checklist mapped to the scverse submission prerequisites.
- **Stop before the real-PyPI upload** and hand back to the user for the go/no-go.

## Guardrails
- **No library behavior changes.** Tests + packaging + CI only. A test that fails = a finding to report,
  not a fix to make.
- The import-contract test (no mlx/scanpy at import) is the one that most protects the "installs anywhere"
  promise — make it robust.
- Don't commit large data or `results/` artifacts into the wheel/sdist. Verify the built artifacts.
- Real-PyPI publish is user-gated and irreversible — TestPyPI first, then stop.
