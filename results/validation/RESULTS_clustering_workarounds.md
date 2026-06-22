# Clustering speed: working around the M3 hardware limits — two approaches tested

The fused single-kernel was earlier found non-viable because of two M3/Metal limits (no
grid-wide barrier; no float atomics). This round implements TWO workarounds and benchmarks
them head-to-head against the production multi-core colored implementation.

## Approach 1 — Hybrid GPU-gain / CPU-apply (`louvain_hybrid.py`)
One GPU dispatch computes every vertex's best target + gain (read-only, no coloring, no Σtot
mutation); the CPU applies moves in gain order with exact sequential Σtot. Sidesteps *both*
limits and uses unified memory for the handoff.

| graph | CURRENT | HYBRID | verdict |
|-------|---------|--------|---------|
| real PBMC (n=2700, fuzzy) | 1467 ms, Q 0.6606 / 10 cl | 797 ms (1.84×), **Q 0.6093 / 96 cl** | faster but **fragments** |
| SBM 10k | 0.16 s, Q 0.490 | 1.42 s (0.11×) | far slower |
| SBM 100k | 0.63 s, Q 0.931 | 11.1 s (0.06×) | far slower |

**Verdict: not viable.** Applying moves from a single per-pass snapshot can't reproduce
per-color-fresh Σtot, so fuzzy graphs over-fragment (96 vs 10 clusters); and the sequential
CPU apply (Python loop) is far too slow at scale.

## Approach 2 — Raw-Metal fused, synthesized primitives (`louvain_fused_raw.py`)
One dispatch, multi-threadgroup, with the missing primitives built in raw MSL:
- **grid-wide barrier** = sense-reversing barrier over a `device atomic_uint` counter
  (counters are zero-init device *outputs* — verified MLX zero-inits outputs).
- **float atomic-add** = CAS loop on `atomic_uint` reinterpreted as float → lets Σtot be
  maintained PER-COLOR inside the kernel (the freshness fuzzy graphs need).

What works:
- Spin-barrier validated in isolation to **≥24 threadgroups**, sub-ms.
- CAS float-add correct; tiny 2-block graph **Q 0.5000 = current exactly**.
- Single-level local moving **converges** on a small fuzzy graph (PBMC-400: by ~10 passes,
  Q 0.5635 vs current 0.5869 — within ~3%, 52 ms).

What fails (→ **not viable**):
1. **G>1 nondeterministic deadlock.** MLX gives no co-residency guarantee; the heavy real
   kernel doesn't reliably co-schedule G threadgroups, so the spin-barrier hangs the GPU
   (observed at G=8 and G=4 on PBMC). The isolated probe passed only because it was light.
2. **G=1 (safe) is single-core-slow** and the dense, high-degree contracted graphs of higher
   multilevel levels (no host fallback for degree>cap) don't converge in the pass budget →
   **>60 s on full PBMC** vs 0.5 s for the production path.

## Overall conclusion
The synthesized primitives genuinely work — a notable result on M3/Metal — but **composing
them into a correct + fast + robust multilevel clusterer does not beat the production
multi-core per-color implementation**, which remains optimal:
- Hybrid: can't keep per-color Σtot fresh → fragments fuzzy graphs; CPU apply slow at scale.
- Raw-fused: grid barrier needs guaranteed co-residency (MLX can't promise it) → deadlock;
  the single-threadgroup fallback is core-starved and non-convergent on dense levels.

This reaffirms the standing verdict with stronger evidence: the clustering crossover on M3 is
**hardware/runtime-bound**, not a missing optimization.

## Round 2 — pursuing the pure-Metal route (re-rooted the failure)
Tried to make the multi-threadgroup fused kernel correct + fast. Findings, in order:
1. **It is NOT a deadlock.** With capped passes, **G=8 completes in 254 ms** — but produces
   garbage (616 comms, **Q=0.05**). So the earlier "hangs" were the multilevel *non-convergence*
   on contracted graphs, not the spin-barrier. → occupancy control would not have helped.
2. **Metal atomics are `relaxed`-only.** Adding the textbook acquire/release barrier fails to
   compile: the system Metal compiler declares **only `memory_order_relaxed`** (no
   acquire/release/seq_cst). No lock-free cross-threadgroup happens-before is expressible.
3. **Coherent-memory workaround partially works.** Apple's device cache is coherent for relaxed
   *atomics* (plain stores stay per-core). Accessing all shared arrays (comm/color/sel/tgt) as
   relaxed atomics raised G=8 single-level from Q=0.05 → **Q=0.62 / 62 comms** — correctness is
   *approximately* recovered. But:
   - atomic accesses in the O(deg²) hot loop are **slower than production** (2.0 s vs 1.5 s);
   - a thread-local neighbor snapshot (O(deg) atomics) restores speed (**599 ms**) but its
     degree cap drops hub vertices (connectivities graph has degree>64 hubs) → **Q=0.40 / 337
     comms**;
   - **residual relaxed-ordering staleness** means G>1 never matches G=1/current EXACTLY
     (62 vs ~15 single-level comms) — and this is unfixable without acquire/release, which
     Metal lacks.

### Final verdict on the pure-Metal route
The blocker is **Metal's relaxed-only atomic memory model**, an MSL *language-level* limit that a
hand-rolled Metal extension shares. **Occupancy control does not address it** — co-residency
(deadlock) was never the problem; correctness + the atomic hot-loop cost are. A faithful
cuGraph-style fused kernel needs Metal to expose acquire/release (or seq_cst) atomics and native
float atomics; until then the multi-core per-color production path is optimal on M3.

Both experimental modules are kept (clearly marked non-viable); production `louvain`/`leiden`
are unchanged. Drivers: ad-hoc benchmarks over PBMC + SBM (see commits).
