"""The single highest-value test: the lazy-import contract.

`metasinglecell` promises it installs and imports anywhere — heavy backends (mlx, scanpy,
torch, squidpy) are lazy-imported *inside functions*, never at module load. This is what lets
`pip install metasinglecell` succeed on Linux/Intel (no Metal) and in a minimal CI runner.

We verify it in a **fresh subprocess** (so the test runner's own imports don't pollute
`sys.modules`): import the package + its public submodules, then assert the heavy modules are
absent from `sys.modules`. Pure-CPU, no data, no GPU — runs on GitHub-hosted CI.
"""
import subprocess
import sys
import textwrap


def _import_and_report(import_stmt):
    """Run `import_stmt` in a clean interpreter; return the set of loaded top-level modules."""
    code = textwrap.dedent(f"""
        import sys
        {import_stmt}
        heavy = ("mlx", "mlx.core", "scanpy", "torch", "squidpy", "umap", "igraph",
                 "leidenalg", "sklearn")
        loaded = sorted(m for m in heavy if m in sys.modules)
        print("LOADED:" + ",".join(loaded))
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"import failed:\n{r.stderr}"
    line = [l for l in r.stdout.splitlines() if l.startswith("LOADED:")][0]
    return set(filter(None, line[len("LOADED:"):].split(",")))


def test_package_import_pulls_no_heavy_backend():
    loaded = _import_and_report("import metasinglecell")
    assert loaded == set(), f"import metasinglecell eagerly loaded heavy backends: {loaded}"


def test_public_submodules_import_clean():
    loaded = _import_and_report(
        "import metasinglecell as m; _ = (m.pp, m.tl, m.gr, m.config)")
    assert loaded == set(), f"pp/tl/gr/config import eagerly loaded: {loaded}"


def test_version_matches_pyproject():
    import metasinglecell
    import tomllib
    from metasinglecell import config
    with open(config.REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject_version = tomllib.load(f)["project"]["version"]
    assert metasinglecell.__version__ == pyproject_version


def test_public_namespaces_present():
    import metasinglecell as m
    for ns in ("pp", "tl", "gr", "config"):
        assert hasattr(m, ns), f"missing public namespace {ns}"
