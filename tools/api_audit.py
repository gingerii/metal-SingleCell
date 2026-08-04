"""Audit every public msc function against the current scanpy / squidpy signature.

Reports, per function:
  DEPRECATED   a parameter the reference has deprecated or removed (we are stale)
  MISSING      a reference parameter we do not expose at all
  DEFAULT      a parameter we both have, with a different default (silent behaviour drift)
  EXTRA        ours only (fine when intentional, listed so it can be judged)
"""
import inspect
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
REF_NOISE = {"kwargs", "args", "show", "save", "ax", "return_fig", "palette", "title",
             "copy", "inplace", "chunked", "chunk_size", "key_added", "return_info",
             "dtype", "backend", "n_jobs", "show_progress_bar", "seed", "sort",
             "return_df", "figsize", "dpi"}

# Reference parameters scanpy/squidpy have deprecated — flagging OUR use of them.
KNOWN_DEPRECATED = {"use_highly_variable", "use_rep_neighbors", "n_dcs", "layers",
                    "layer_norm", "flavor_key", "use_raw_layer"}

PAIRS = [
    ("pp", msc.pp, sc.pp),
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


findings = {"DEPRECATED": [], "MISSING": [], "DEFAULT": [], "EXTRA": [], "NO-REF": []}

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
        ref = params(ref_fn)
        if ours is None or ref is None:
            continue
        for p in ours:
            if p in KNOWN_DEPRECATED and p not in ref:
                findings["DEPRECATED"].append(f"{ns}.{name}({p}=) — gone from the reference")
            elif p in KNOWN_DEPRECATED:
                findings["DEPRECATED"].append(f"{ns}.{name}({p}=) — deprecated upstream")
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
for k in ("DEPRECATED", "DEFAULT", "MISSING", "NO-REF", "EXTRA"):
    v = findings[k]
    print(f"\n=== {k} ({len(v)}) ===")
    for line in v:
        print(f"  {line}")
