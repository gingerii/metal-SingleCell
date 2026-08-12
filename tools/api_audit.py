"""Audit every public msc function against the current scanpy / squidpy signature.

Reports, per function:
  RETIRED      the reference FUNCTION itself is deprecated upstream (we mirror a dead API)
  DEPRECATED   a parameter the reference has deprecated or removed (we are stale)
  MISSING      a reference parameter we do not expose at all
  DEFAULT      a parameter we both have, with a different default (silent behaviour drift)
  EXTRA        ours only (fine when intentional, listed so it can be judged)

RETIRED exists because comparing parameter lists cannot catch it: `sq.gr.spatial_neighbors`
kept its name and signature while being deprecated wholesale in squidpy 1.7 for removal in
1.9 (github issue #4). A user reported that before this audit did.
"""
import inspect
import re
import warnings

warnings.simplefilter("ignore")

import scanpy as sc
import metalsinglecell as msc

try:
    import squidpy as sq
except Exception:
    sq = None

# Parameters that are ours by design, or reference-only knobs we deliberately don't mirror.
OURS_BY_DESIGN = {"backend", "variant", "commit_prob", "exact_max_n", "approx", "solver",
                  "block_rows", "n_oversamples", "n_iter", "correction", "block_proportion"}
# Genuinely cosmetic: plotting knobs and parallelism switches with no meaning on the GPU path.
# `inplace`, `key_added` and `copy` are deliberately NOT here -- they are functional (they change
# what is written and what comes back), and `copy` in particular hides a live upstream
# deprecation on filter_cells/filter_genes. Suppressing them undercounted the gap by 17.
REF_NOISE = {"kwargs", "args", "show", "save", "ax", "return_fig", "palette", "title",
             "chunked", "chunk_size", "return_info",
             "dtype", "backend", "n_jobs", "show_progress_bar", "sort",
             "return_df", "figsize", "dpi"}

# Reference parameters scanpy/squidpy have deprecated — flagging OUR use of them. An entry may
# be a bare parameter name (deprecated wherever it appears) or "function.parameter" when the
# deprecation is specific: `copy` is deprecated on filter_cells/filter_genes only, and reporting
# it everywhere would bury the real ones.
KNOWN_DEPRECATED = {"use_highly_variable", "use_rep_neighbors", "n_dcs", "layers",
                    "layer_norm", "flavor_key", "use_raw_layer",
                    "filter_cells.copy", "filter_genes.copy"}

class _Namespace:
    """A reference namespace assembled from several scanpy modules.

    `normalize_pearson_residuals` lives in `sc.experimental.pp`, not `sc.pp`, so pairing
    `msc.pp` against `sc.pp` alone left it unaudited -- and it was missing `check_values`,
    `inplace` and `layer`, and validated nothing.
    """

    def __init__(self, *mods):
        self._mods = mods

    def __getattr__(self, name):
        for m in self._mods:
            if hasattr(m, name):
                return getattr(m, name)
        raise AttributeError(name)


_sc_pp = _Namespace(sc.pp, getattr(getattr(sc, "experimental", None), "pp", object()),
                    getattr(getattr(sc, "external", None), "pp", object()))

PAIRS = [
    ("pp", msc.pp, _sc_pp),
    ("tl", msc.tl, sc.tl),
]
if sq is not None:
    PAIRS.append(("gr", msc.gr, sq.gr))


def params(fn):
    try:
        s = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    return {n: p.default for n, p in s.parameters.items()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and n not in ("adata", "data")}


def show(v):
    r = repr(v)
    return r if len(r) <= 22 else r[:19] + "..."


_DEPRECATED_MARKER = re.compile(r"\.\.\s*deprecated::\s*([\d.]+)")


def retired(ref_fn):
    """Is the reference FUNCTION deprecated (not merely one of its parameters)?

    Both cases put a `.. deprecated::` directive in the same docstring, so position is what
    separates them: a function-level notice sits in the summary, above the Parameters block,
    while a parameter-level one (scanpy marking `method='rapids'`, say) sits inside it.
    """
    doc = inspect.getdoc(ref_fn) or ""
    m = _DEPRECATED_MARKER.search(doc)
    if not m:
        return None
    params_at = re.search(r"^\s*Parameters\s*$", doc, re.M)
    if params_at and m.start() > params_at.start():
        return None                                   # inside Parameters -> a parameter, not the fn
    body = doc[m.start():m.start() + 700]
    removal = re.search(r"removed in \S*\s*v?([\d.]+)", body)
    repl = sorted({a or b for a, b in re.findall(r":func:`[~.\w]*?(\w+)`|``(\w+)``", body)}
                  - {ref_fn.__name__})
    return m.group(1), (removal.group(1).rstrip(".") if removal else "?"), repl


findings = {"RETIRED": [], "DEPRECATED": [], "MISSING": [], "DEFAULT": [], "EXTRA": [],
            "NO-REF": []}

for ns, ours_mod, ref_mod in PAIRS:
    names = [n for n in dir(ours_mod) if not n.startswith("_")
             and callable(getattr(ours_mod, n))
             and getattr(getattr(ours_mod, n), "__module__", "").startswith("metalsinglecell")]
    for name in sorted(names):
        ours = params(getattr(ours_mod, name))
        ref_fn = getattr(ref_mod, name, None)
        if ref_fn is None:
            findings["NO-REF"].append(f"{ns}.{name}")
            continue
        gone = retired(ref_fn)
        if gone:
            since, removal, repl = gone
            findings["RETIRED"].append(
                f"{ns}.{name} — deprecated upstream in {since}, removal {removal}"
                + (f"; use {', '.join(repl)}" if repl else ""))
        ref = params(ref_fn)
        if ours is None or ref is None:
            continue
        for p in ours:
            if not (p in KNOWN_DEPRECATED or f"{name}.{p}" in KNOWN_DEPRECATED):
                continue
            why = "gone from the reference" if p not in ref else "deprecated upstream"
            findings["DEPRECATED"].append(f"{ns}.{name}({p}=) — {why}")
        for p, d in ref.items():
            if p in REF_NOISE or p.startswith("_"):
                continue
            if p not in ours:
                findings["MISSING"].append(f"{ns}.{name}({p}={show(d)})")
        for p, d in ours.items():
            if p in REF_NOISE or p in OURS_BY_DESIGN:
                continue
            if p not in ref:
                findings["EXTRA"].append(f"{ns}.{name}({p}=)")
            elif ref[p] is not inspect.Parameter.empty and d != ref[p] and not (
                    isinstance(d, float) and isinstance(ref[p], float) and d == ref[p]):
                findings["DEFAULT"].append(
                    f"{ns}.{name}({p}=): ours {show(d)} vs scanpy {show(ref[p])}")

print(f"scanpy {sc.__version__}" + (f", squidpy {sq.__version__}" if sq else ", squidpy MISSING"))
for k in ("RETIRED", "DEPRECATED", "DEFAULT", "MISSING", "NO-REF", "EXTRA"):
    v = findings[k]
    print(f"\n=== {k} ({len(v)}) ===")
    for line in v:
        print(f"  {line}")
